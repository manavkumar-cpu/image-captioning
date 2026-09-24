"""Disposable three-update training check on a tiny real-data subset."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from torch.utils.data import Subset

from src.data.dataloader import build_dataloaders, make_loader
from src.models.factory import build_model
from src.training.losses import CaptionCrossEntropy
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer, EarlyStopping
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


LOGGER = get_logger("image_captioning.trainer_check")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_loss():
    logits = torch.zeros(1, 3, 3, requires_grad=True)
    targets = torch.tensor([[0, 1, -100]], dtype=torch.long)

    result = CaptionCrossEntropy()(logits, targets)

    require(result["valid_tokens"] == 2, "Incorrect valid-token count")
    require(
        math.isclose(result["loss"].item(), math.log(3), abs_tol=1e-6),
        "Masked loss has the wrong denominator",
    )

    result["loss"].backward()

    require(
        torch.equal(logits.grad[0, 2], torch.zeros(3)),
        "Padding position received a loss gradient",
    )
    require(
        logits.grad[0, :2].abs().sum().item() > 0,
        "Valid positions received no gradient",
    )

    LOGGER.info("Masked loss value and padding gradients: PASS")


def check_control_logic(config):
    # These tiny CPU objects test scheduling without training another model.
    tiny_model = nn.Linear(1, 1)
    optimizer = build_optimizer(tiny_model, config)
    scheduler = build_scheduler(optimizer, config)

    initial_lr = optimizer.param_groups[0]["lr"]
    mode = config["training"]["monitor_mode"]

    scheduler.step(1.0)
    worse = 2.0 if mode == "min" else 0.0

    for _ in range(scheduler.patience + 1):
        scheduler.step(worse)

    require(
        math.isclose(
            optimizer.param_groups[0]["lr"],
            initial_lr * 0.5,
        ),
        "Scheduler did not reduce the learning rate",
    )

    stopping = EarlyStopping(patience=2, mode="min")

    require(stopping.update(3.0), "First score should improve")
    require(not stopping.update(4.0), "Worse score marked as improved")
    require(not stopping.should_stop, "Early stopping triggered too soon")
    require(not stopping.update(4.0), "Repeated worse score marked improved")
    require(stopping.should_stop, "Early stopping did not trigger")

    # An improvement must reset its counter.
    require(stopping.update(2.0), "Better score not recognized")
    require(stopping.bad_epochs == 0, "Improvement did not reset patience")

    LOGGER.info("Scheduler and early-stopping bookkeeping: PASS")


def distinct_image_pair_indices(dataset, count):
    """Choose one caption pair from each of a few different images."""

    selected = []
    seen = set()

    for index, sample in enumerate(dataset.samples):
        record_index = sample["record_index"]

        if record_index not in seen:
            seen.add(record_index)
            selected.append(index)

        if len(selected) == count:
            return selected

    raise ValueError("Not enough distinct images for the tiny check")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/baseline_debug.yaml",
    )
    parser.add_argument("--val-images", type=int, default=250)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        seed_everything(config["project"]["seed"])

        check_loss()
        check_control_logic(config)

        device = select_device(config["runtime"]["device"])
        loaders = build_dataloaders(config, args.val_images)
        tokenizer = loaders["tokenizer"]

        train_indices = distinct_image_pair_indices(
            loaders["train"].dataset, 6
        )
        val_indices = distinct_image_pair_indices(
            loaders["validation"].dataset, 4
        )

        train_loader = make_loader(
            Subset(loaders["train"].dataset, train_indices),
            batch_size=2,
            num_workers=0,
            shuffle=True,
            seed=config["project"]["seed"],
            collate_fn=loaders["train"].collate_fn,
        )

        val_loader = make_loader(
            Subset(loaders["validation"].dataset, val_indices),
            batch_size=2,
            num_workers=0,
            shuffle=False,
            seed=config["project"]["seed"] + 1,
            collate_fn=loaders["validation"].collate_fn,
        )

        LOGGER.info(
            "Disposable check: 6 training pairs, 4 validation pairs, "
            "batch size 2, one tiny-subset epoch"
        )

        model = build_model(config, tokenizer).to(device)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        trainer = Trainer(
            config=config,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        before_projection = (
            model.hidden_projection.weight.detach().cpu().clone()
        )
        before_encoder = model.encoder.conv1.weight.detach().cpu().clone()

        history = trainer.fit(
            model,
            train_loader,
            val_loader,
            epochs=1,
        )

        require(trainer.global_step == 3, "Expected three optimizer updates")
        require(len(history) == 1, "Expected one history record")
        require(history[0]["train_batches"] == 3, "Wrong training batch count")
        require(history[0]["val_batches"] == 2, "Wrong validation batch count")

        require(
            not torch.equal(
                before_projection,
                model.hidden_projection.weight.detach().cpu(),
            ),
            "Trainable projection weights did not change",
        )
        require(
            torch.equal(
                before_encoder,
                model.encoder.conv1.weight.detach().cpu(),
            ),
            "Frozen encoder weights changed",
        )

        LOGGER.info("Trainable weight update and frozen encoder check: PASS")

        before_validation = (
            model.hidden_projection.weight.detach().cpu().clone()
        )
        step_before = trainer.global_step

        validation = trainer.validate(model, val_loader)

        require(
            trainer.global_step == step_before,
            "Validation performed an optimizer update",
        )
        require(
            torch.equal(
                before_validation,
                model.hidden_projection.weight.detach().cpu(),
            ),
            "Validation changed projection weights",
        )
        require(
            all(parameter.grad is None for parameter in model.parameters()),
            "Validation unexpectedly created gradients",
        )
        require(
            math.isclose(
                validation["loss"],
                history[0]["val_loss"],
                rel_tol=1e-5,
                abs_tol=1e-5,
            ),
            "Repeated validation loss differs unexpectedly",
        )

        LOGGER.info("Validation without updates or gradients: PASS")

        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            + "_"
            + uuid4().hex[:8]
        )
        output_dir = (
            Path(config["paths"]["artifacts_root"])
            / "trainer_checks"
            / run_id
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        report = {
            "status": "passed",
            "device": str(device),
            "mixed_precision": trainer.use_amp,
            "training_pairs": 6,
            "validation_pairs": 4,
            "check_batch_size": 2,
            "optimizer_updates": trainer.global_step,
            "history": history,
            "repeat_validation": validation,
            "tokenizer_fingerprint": tokenizer.fingerprint(),
            "config": config,
            "scope": (
                "Disposable tiny-subset training check. "
                "No full development-set epoch or caption-quality evaluation. "
                "Updated model weights are not saved."
            ),
        }

        report_path = output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        LOGGER.info("Report: %s", report_path)
        LOGGER.info("PHASE 7 TRAINER CHECK: PASS")
        return 0

    except Exception as exc:
        LOGGER.error("TRAINER CHECK FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())