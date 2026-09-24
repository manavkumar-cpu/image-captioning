"""Train the existing baseline on the prepared, limited COCO subset."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from src.data.dataloader import build_dataloaders
from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from src.data.prepare import json_fingerprint, metadata_paths
from src.data.tokenizer import CaptionTokenizer, tokenizer_directory
from src.models.factory import build_model
from src.training.checkpoints import (
    json_value, load_checkpoint, resolve_checkpoint, write_json,
)
from src.training.optimizer import build_optimizer
from src.training.run_logging import RunLogger, capture_training_log, write_history
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


LOGGER = get_logger("image_captioning.trainer")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer):
    value = tokenizer.fingerprint
    return value() if callable(value) else value


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        header = [name, list(tensor.shape), str(tensor.dtype)]
        digest.update(json.dumps(header).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def trainable_state(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def validate_scope(config, val_images):
    require(
        1 <= config["data"]["debug_images"] <= 1000,
        "Phase 9 is limited to at most 1,000 training images.",
    )
    require(
        1 <= val_images <= 250,
        "Use at most 250 validation images.",
    )
    require(
        config["model"]["name"] == "cnn_lstm",
        "Expected the CNN-LSTM baseline.",
    )
    require(
        config["model"]["pretrained"]
        and config["model"]["freeze_encoder"],
        "This phase uses the pretrained, frozen encoder.",
    )
    require(
        config["runtime"]["num_workers"] == 0,
        "Keep num_workers=0 for verified resume.",
    )
    require(
        not config["runtime"]["mixed_precision"],
        "Keep mixed_precision=false for this run.",
    )
    require(
        not config["logging"]["tensorboard"],
        "This run uses JSON/CSV/text logging.",
    )
    require(
        config["training"]["monitor"] == "val_loss"
        and config["training"]["monitor_mode"] == "min",
        "Select the baseline by minimum validation loss.",
    )


def prepare_data(config, val_images):
    validate_scope(config, val_images)

    built = build_dataloaders(config, val_images=val_images)
    tokenizer = built["tokenizer"]
    loaders = {
        name: built[name]
        for name in ("train", "validation")
    }

    _, manifest_path = metadata_paths(config, val_images)
    manifest = json.loads(Path(manifest_path).read_text())

    raw_root = (
        Path(config["paths"]["data_root"]) / "raw" / "coco2017"
    ).resolve()

    image_hashes = []
    counts = {}

    LOGGER.info("Checking selected image fingerprints; no downloads")

    for split, loader in loaders.items():
        records = loader.dataset.records

        require(
            records == manifest[split],
            f"{split}: loader/manifest mismatch.",
        )

        for record in sorted(records, key=lambda item: item["image_id"]):
            path = (raw_root / record["relative_path"]).resolve()

            require(
                path.is_relative_to(raw_root),
                "Image path escapes the dataset root.",
            )

            image_hashes.append([
                split,
                record["image_id"],
                record["relative_path"],
                file_sha256(path),
            ])

        counts[split] = {
            "images": len(records),
            "pairs": len(loader.dataset),
            "batches": len(loader),
            "target_tokens": sum(
                len(sample["token_ids"]) - 1
                for sample in loader.dataset.samples
            ),
        }

    source_root = Path(__file__).resolve().parents[1]

    sources = [
        "data/preprocessing.py",
        "data/tokenizer.py",
        "data/dataset.py",
        "data/collate.py",
        "data/dataloader.py",
        "models/base_model.py",
        "models/cnn_lstm.py",
        "models/factory.py",
        "training/losses.py",
        "training/optimizer.py",
        "training/scheduler.py",
        "training/trainer.py",
        "training/checkpoints.py",
        "training/run_logging.py",
        "training/train_baseline.py",
    ]

    context = json_value({
        "kind": "coco2017_limited_baseline",
        "version": 1,
        "val_images": val_images,
        "manifest_sha256": json_fingerprint(manifest),
        "selected_image_files_sha256": json_fingerprint(image_hashes),
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "encoded_pairs_sha256": {
            split: json_fingerprint(loader.dataset.samples)
            for split, loader in loaders.items()
        },
        "preprocessing": {
            "color": "RGB",
            "resize": [config["data"]["image_size"]] * 2,
            "interpolation": "bilinear",
            "antialias": True,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "source_sha256": {
            name: file_sha256(source_root / name)
            for name in sources
        },
        "batch_limits": {
            "train": None,
            "validation": None,
        },
        "counts": counts,
    })

    return loaders, tokenizer, context


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/baseline_debug.yaml",
    )
    parser.add_argument(
        "--val-images",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Total target epochs, not additional epochs",
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--run-dir",
        type=Path,
        help="New output directory",
    )
    group.add_argument(
        "--resume",
        type=Path,
        help="Existing run directory",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    target_epochs = (
        config["training"]["epochs"]
        if args.epochs is None
        else args.epochs
    )

    require(target_epochs >= 1, "epochs must be positive.")

    run_dir = (
        args.resume if args.resume is not None else args.run_dir
    ).expanduser().resolve()

    resuming = args.resume is not None

    require(
        resuming or not run_dir.exists(),
        "Run directory exists: use --resume or a new name.",
    )

    if resuming:
        resolve_checkpoint(run_dir)

    seed_everything(config["project"]["seed"])
    device = select_device(config["runtime"]["device"])

    loaders, tokenizer, context = prepare_data(
        config,
        args.val_images,
    )

    if resuming:
        require(
            json.loads((run_dir / "config.json").read_text())
            == json_value(config),
            "Saved/current configuration differs.",
        )
        require(
            json.loads((run_dir / "context.json").read_text())
            == context,
            "Data, tokenizer, preprocessing or source code changed since this run.",
        )

        snapshot = CaptionTokenizer.load(run_dir / "tokenizer")

        require(
            tokenizer_fingerprint(snapshot)
            == tokenizer_fingerprint(tokenizer),
            "Saved tokenizer snapshot differs.",
        )

    LOGGER.info("Creating baseline on %s", device)

    model = build_model(
        config,
        tokenizer,
        load_pretrained=not resuming,
    ).to(device)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    trainer = Trainer(config, device, optimizer, scheduler)

    if resuming:
        load_checkpoint(
            resolve_checkpoint(run_dir),
            model,
            trainer,
            loaders,
            context,
        )

        require(
            target_epochs >= trainer.epoch,
            "Target epochs is below the saved epoch.",
        )

        identity = json.loads(
            (run_dir / "initial_state.json").read_text()
        )

        require(
            state_fingerprint(model.encoder.state_dict())
            == identity["frozen_encoder_sha256"],
            "Saved encoder differs from its initial frozen state.",
        )

        write_history(run_dir, trainer.history)

    else:
        identity = {
            "frozen_encoder_sha256": state_fingerprint(
                model.encoder.state_dict()
            ),
            "initial_trainable_sha256": state_fingerprint(
                trainable_state(model)
            ),
        }

    callback = RunLogger(
        run_dir,
        loaders,
        context,
        resume=resuming,
    )

    if not resuming:
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "context.json", context)
        write_json(run_dir / "initial_state.json", identity)

        shutil.copytree(
            tokenizer_directory(config),
            run_dir / "tokenizer",
        )

    with capture_training_log(run_dir):
        LOGGER.info("Run directory: %s", run_dir)

        LOGGER.info(
            "Device: %s | saved epoch: %d | target epochs: %d",
            device,
            trainer.epoch,
            target_epochs,
        )

        for split, count in context["counts"].items():
            LOGGER.info(
                "%s: images=%d, caption pairs=%d, batches=%d",
                split,
                count["images"],
                count["pairs"],
                count["batches"],
            )

        LOGGER.info(
            "Training with a frozen encoder; checkpoint after each completed epoch"
        )

        trainer.fit(
            model,
            loaders["train"],
            loaders["validation"],
            epochs=target_epochs,
            on_epoch_end=callback,
        )

        require(
            state_fingerprint(model.encoder.state_dict())
            == identity["frozen_encoder_sha256"],
            "Frozen encoder changed during training.",
        )

        require(
            state_fingerprint(trainable_state(model))
            != identity["initial_trainable_sha256"],
            "Trainable parameters did not change.",
        )

        index = json.loads(
            (run_dir / "checkpoints" / "index.json").read_text()
        )

        summary = {
            "status": "training_finished",
            "device": str(device),
            "target_epochs": target_epochs,
            "completed_epochs": trainer.epoch,
            "global_step": trainer.global_step,
            "early_stopped": trainer.early_stopping.should_stop,
            "best_epoch": index["best_epoch"],
            "best_val_loss": trainer.early_stopping.best,
            "last_checkpoint": str(resolve_checkpoint(run_dir)),
            "best_checkpoint": str(
                resolve_checkpoint(run_dir, "best")
            ),
            "counts": context["counts"],
            "caption_quality_evaluated": False,
        }

        write_json(run_dir / "run_summary.json", summary)

        LOGGER.info(
            "Best epoch: %d | best val_loss: %.6f",
            index["best_epoch"],
            trainer.early_stopping.best,
        )

        LOGGER.info(
            "PHASE 9 TRAINING FINISHED; run src.training.check_training_run next"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning(
            "Interrupted. Resume from the last completed epoch, if one was saved."
        )
        raise SystemExit(130)
    except Exception:
        LOGGER.exception("BASELINE TRAINING FAILED")
        raise SystemExit(1)