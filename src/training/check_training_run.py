"""Check a completed baseline run and reproduce its best validation loss."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from src.data.tokenizer import CaptionTokenizer
from src.models.factory import build_model
from src.training.checkpoints import (
    json_value, read_checkpoint, resolve_checkpoint, write_json,
)
from src.training.optimizer import build_optimizer
from src.training.train_baseline import (
    prepare_data,
    require,
    state_fingerprint,
    tokenizer_fingerprint,
    trainable_state,
)
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


LOGGER = get_logger("image_captioning.training_run_check")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()

    config = load_config(run_dir / "config.json")

    context = json.loads(
        (run_dir / "context.json").read_text()
    )
    summary = json.loads(
        (run_dir / "run_summary.json").read_text()
    )
    identity = json.loads(
        (run_dir / "initial_state.json").read_text()
    )
    index = json.loads(
        (run_dir / "checkpoints" / "index.json").read_text()
    )

    seed_everything(config["project"]["seed"])

    loaders, tokenizer, current_context = prepare_data(
        config,
        context["val_images"],
    )

    require(
        context == current_context,
        "Run context changed: data/tokenizer/source mismatch.",
    )

    snapshot = CaptionTokenizer.load(run_dir / "tokenizer")

    require(
        tokenizer_fingerprint(snapshot) == context["tokenizer_sha256"],
        "Saved tokenizer snapshot differs.",
    )

    LOGGER.info(
        "Configuration, source, data and tokenizer fingerprints: PASS"
    )

    last = read_checkpoint(resolve_checkpoint(run_dir))

    require(
        last["contract"]["config"] == json_value(config),
        "Checkpoint config differs.",
    )
    require(
        last["contract"]["context"] == context,
        "Checkpoint context differs.",
    )

    history = last["trainer"]["history"]
    completed = last["trainer"]["epoch"]

    require(
        completed >= 1
        and [row["epoch"] for row in history]
        == list(range(1, completed + 1)),
        "Epoch history is incomplete.",
    )

    require(
        completed
        == summary["completed_epochs"]
        == index["last_epoch"],
        "Summary/index/checkpoint epoch mismatch; finish or resume the run first.",
    )

    require(
        summary["status"] == "training_finished",
        "Run has not finished.",
    )

    require(
        completed >= summary["target_epochs"]
        or (
            summary["early_stopped"]
            and last["trainer"]["early_stopping"]["bad_epochs"]
            >= config["training"]["patience"]
        ),
        "Requested training has not completed.",
    )

    for row in history:
        for split in ("train", "validation"):
            prefix = "train" if split == "train" else "val"
            counts = context["counts"][split]

            require(
                row[f"{prefix}_batches"] == counts["batches"],
                "A partial epoch was recorded.",
            )
            require(
                row[f"{prefix}_tokens"] == counts["target_tokens"],
                "Target token count differs.",
            )
            require(
                math.isfinite(row[f"{prefix}_loss"]),
                "Non-finite epoch loss.",
            )

        require(
            row["optimizer_steps"]
            == context["counts"]["train"]["batches"]
            and row["skipped_steps"] == 0,
            "Missing optimizer updates.",
        )

    updates = sum(
        row["optimizer_steps"]
        for row in history
    )

    require(
        updates
        == last["trainer"]["global_step"]
        == summary["global_step"],
        "Global update count differs.",
    )

    require(
        json.loads((run_dir / "metrics.json").read_text())
        == history,
        "Metric history differs from checkpoint; resume to regenerate it.",
    )

    best_row = min(
        history,
        key=lambda row: row["val_loss"],
    )

    require(
        best_row["epoch"]
        == index["best_epoch"]
        == summary["best_epoch"],
        "Best epoch selection differs.",
    )

    require(
        best_row["val_loss"] == summary["best_val_loss"],
        "Best score differs.",
    )

    LOGGER.info(
        "Complete epochs, token counts, optimizer updates and best selection: PASS"
    )

    del last
    gc.collect()

    best = read_checkpoint(
        resolve_checkpoint(run_dir, "best")
    )

    require(
        best["contract"]["context"] == context
        and best["contract"]["config"] == json_value(config),
        "Best checkpoint identity differs.",
    )

    require(
        best["trainer"]["epoch"] == best_row["epoch"],
        "Best checkpoint epoch differs.",
    )

    weights = best["model"]

    del best
    gc.collect()

    require(
        all(
            not tensor.is_floating_point()
            or bool(torch.isfinite(tensor).all())
            for tensor in weights.values()
        ),
        "Checkpoint contains non-finite model weights.",
    )

    model = build_model(
        config,
        snapshot,
        load_pretrained=False,
    )

    model.load_state_dict(
        weights,
        strict=True,
    )

    del weights
    gc.collect()

    require(
        state_fingerprint(model.encoder.state_dict())
        == identity["frozen_encoder_sha256"],
        "Frozen encoder weights/buffers changed.",
    )

    require(
        state_fingerprint(trainable_state(model))
        != identity["initial_trainable_sha256"],
        "Best checkpoint has unchanged trainable parameters.",
    )

    LOGGER.info(
        "Best checkpoint reload, finite weights and frozen encoder: PASS"
    )

    device = select_device(config["runtime"]["device"])
    model = model.to(device)

    optimizer = build_optimizer(model, config)
    trainer = Trainer(config, device, optimizer)

    LOGGER.info(
        "Rechecking best checkpoint on all %d validation images",
        context["counts"]["validation"]["images"],
    )

    metrics = trainer.validate(
        model,
        loaders["validation"],
    )

    require(
        math.isclose(
            metrics["loss"],
            best_row["val_loss"],
            rel_tol=1e-4,
            abs_tol=1e-4,
        ),
        (
            f"Reloaded validation loss differs: "
            f"saved={best_row['val_loss']}, "
            f"rerun={metrics['loss']}"
        ),
    )

    require(
        metrics["valid_tokens"]
        == context["counts"]["validation"]["target_tokens"],
        "Validation did not cover every target token.",
    )

    require(
        all(
            parameter.grad is None
            for parameter in model.parameters()
        ),
        "Validation created gradients.",
    )

    report = {
        "status": "passed",
        "device": str(device),
        "completed_epochs": completed,
        "global_step": updates,
        "best_epoch": best_row["epoch"],
        "saved_best_val_loss": best_row["val_loss"],
        "reloaded_val_loss": metrics["loss"],
        "counts": context["counts"],
        "caption_quality_evaluated": False,
    }

    write_json(
        run_dir / "training_validation.json",
        report,
    )

    LOGGER.info(
        "Saved val_loss=%.6f | reloaded val_loss=%.6f — PASS",
        best_row["val_loss"],
        metrics["loss"],
    )

    LOGGER.info(
        "Report: %s",
        run_dir / "training_validation.json",
    )

    LOGGER.info(
        "PHASE 9 BASELINE TRAINING CHECK: PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception(
            "PHASE 9 BASELINE TRAINING CHECK: FAIL"
        )
        raise SystemExit(1)