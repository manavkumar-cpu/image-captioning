"""Epoch callback: commit checkpoints and regenerate persistent metrics."""

from contextlib import contextmanager
import csv
import io
import json
import logging
from pathlib import Path
import uuid

from src.training.checkpoints import (
    SCHEMA_VERSION,
    atomic_write,
    resolve_checkpoint,
    save_checkpoint,
    write_json,
)
from src.utils.exceptions import CheckpointError
from src.utils.logger import get_logger


def write_history(run_dir, history):
    run_dir = Path(run_dir)

    write_json(
        run_dir / "metrics.json",
        history,
    )

    output = io.StringIO(newline="")

    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "optimizer_steps",
        "best_score",
        "improved",
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fields,
        extrasaction="ignore",
    )

    writer.writeheader()
    writer.writerows(history)

    atomic_write(
        run_dir / "metrics.csv",
        lambda stream: stream.write(
            output.getvalue().encode()
        ),
    )


@contextmanager
def capture_training_log(run_dir):
    logger = get_logger("image_captioning.trainer")

    handler = logging.FileHandler(
        Path(run_dir) / "training.log",
        encoding="utf-8",
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )

    logger.addHandler(handler)

    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


class RunLogger:
    def __init__(
        self,
        run_dir,
        loaders,
        context,
        resume=False,
    ):
        self.run_dir = Path(run_dir)
        self.loaders = loaders
        self.context = context

        self.checkpoint_dir = (
            self.run_dir / "checkpoints"
        )

        self.index_path = (
            self.checkpoint_dir / "index.json"
        )

        if resume:
            resolve_checkpoint(self.run_dir)
        else:
            self.run_dir.mkdir(
                parents=True,
                exist_ok=False,
            )
            self.checkpoint_dir.mkdir()

    def __call__(self, trainer, model, record):
        previous = None

        if self.index_path.exists():
            previous = json.loads(
                self.index_path.read_text()
            )

        expected_epoch = (
            1
            if previous is None
            else previous["last_epoch"] + 1
        )

        if (
            trainer.epoch != expected_epoch
            or record != trainer.history[-1]
        ):
            raise CheckpointError(
                "Callback must commit consecutive completed epochs."
            )

        if previous is None and not record["improved"]:
            raise CheckpointError(
                "The first completed epoch must establish a best score."
            )

        filename = (
            f"epoch_{trainer.epoch:04d}_"
            f"{uuid.uuid4().hex[:8]}.pt"
        )

        save_checkpoint(
            self.checkpoint_dir / filename,
            model,
            trainer,
            self.loaders,
            self.context,
        )

        best = (
            filename
            if record["improved"]
            else previous["best"]
        )

        best_epoch = (
            trainer.epoch
            if record["improved"]
            else previous["best_epoch"]
        )

        index = {
            "schema_version": SCHEMA_VERSION,
            "last": filename,
            "best": best,
            "last_epoch": trainer.epoch,
            "best_epoch": best_epoch,
        }

        # Commit last/best references together in one replacement.
        write_json(self.index_path, index)

        write_history(
            self.run_dir,
            trainer.history,
        )

        if previous is not None:
            obsolete = {
                previous["last"],
                previous["best"],
            } - {
                filename,
                best,
            }

            for old in obsolete:
                (self.checkpoint_dir / old).unlink(
                    missing_ok=True
                )