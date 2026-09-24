"""Small synthetic test of the real Trainer and checkpoint lifecycle."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate

from src.data.dataloader import make_loader
from src.training.checkpoints import (
    cpu_copy,
    load_checkpoint,
    read_checkpoint,
    resolve_checkpoint,
    write_json,
)
from src.training.optimizer import build_optimizer
from src.training.run_logging import (
    RunLogger,
    capture_training_log,
    write_history,
)
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.exceptions import CheckpointError
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


LOGGER = get_logger("image_captioning.checkpoint_check")

CONTEXT = {
    "kind": "synthetic_checkpoint_test",
    "fixture_version": 1,
    "vocab_size": 17,
}


class TinyCaptionFixture(nn.Module):
    """Test fixture only; never used as the project's captioning model."""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Linear(3, 8)
        self.encoder.requires_grad_(False)

        self.embedding = nn.Embedding(
            17,
            8,
            padding_idx=0,
        )

        self.projection = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(8, 17)

    def forward(self, images, input_ids, attention_mask):
        features = self.projection(
            self.encoder(
                images.mean(dim=(2, 3))
            )
        )

        hidden = (
            self.embedding(input_ids)
            + features[:, None, :]
        )

        hidden = hidden.masked_fill(
            ~attention_mask.unsqueeze(-1),
            0,
        )

        return {
            "logits": self.output(
                self.dropout(hidden)
            )
        }


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_same(left, right, tolerance=0.0):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(
            left.cpu(),
            right.cpu(),
            rtol=tolerance,
            atol=tolerance,
        )

    elif isinstance(left, dict):
        require(
            left.keys() == right.keys(),
            "Dictionary keys differ.",
        )

        for key in left:
            assert_same(
                left[key],
                right[key],
                tolerance,
            )

    elif isinstance(left, (list, tuple)):
        require(
            len(left) == len(right),
            "Sequence lengths differ.",
        )

        for a, b in zip(left, right):
            assert_same(a, b, tolerance)

    elif isinstance(left, float):
        require(
            math.isclose(
                left,
                right,
                rel_tol=tolerance,
                abs_tol=tolerance,
            ),
            "Floats differ.",
        )

    else:
        require(
            left == right,
            f"Values differ: {left!r}, {right!r}",
        )


def fixture_loaders(seed):
    generator = torch.Generator().manual_seed(17)

    samples = []

    for index in range(6):
        token = 3 + index

        samples.append({
            "images": torch.rand(
                3,
                8,
                8,
                generator=generator,
            ),
            "input_ids": torch.tensor([
                1,
                token,
                4,
                0,
            ]),
            "attention_mask": torch.tensor([
                True,
                True,
                True,
                False,
            ]),
            "target_ids": torch.tensor([
                token,
                4,
                2,
                -100,
            ]),
            "image_ids": index,
        })

    return {
        "train": make_loader(
            dataset=samples,
            batch_size=2,
            num_workers=0,
            shuffle=True,
            seed=seed,
            collate_fn=default_collate,
        ),
        "validation": make_loader(
            dataset=samples[:4],
            batch_size=2,
            num_workers=0,
            shuffle=False,
            seed=seed + 1,
            collate_fn=default_collate,
        ),
    }


def make_objects(config, device):
    model = TinyCaptionFixture().to(device)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    trainer = Trainer(
        config,
        device,
        optimizer,
        scheduler,
    )

    return model, trainer


def random_probe(device):
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "cpu": torch.rand(4),
        "device": torch.rand(
            4,
            device=device,
        ).cpu(),
    }


def run_check(config, device, directory):
    config = deepcopy(config)

    config["runtime"]["mixed_precision"] = False
    config["runtime"]["num_workers"] = 0

    config["training"]["patience"] = 10
    config["training"]["monitor"] = "check_score"
    config["training"]["monitor_mode"] = "min"

    seed = config["project"]["seed"]
    seed_everything(seed)

    loaders = fixture_loaders(seed)
    model, trainer = make_objects(config, device)

    run_dir = directory / "uninterrupted"
    callback = RunLogger(
        run_dir,
        loaders,
        CONTEXT,
    )

    # Controlled worsening scores retain epoch 1 as best.
    def metric_fn(model, loader):
        return {
            "check_score": float(trainer.epoch + 1)
        }

    with capture_training_log(run_dir):
        trainer.fit(
            model,
            loaders["train"],
            loaders["validation"],
            epochs=1,
            on_epoch_end=callback,
            metric_fn=metric_fn,
        )

    resumed_dir = directory / "resumed"
    shutil.copytree(run_dir, resumed_dir)

    checkpoint = resolve_checkpoint(resumed_dir)
    saved = read_checkpoint(checkpoint)

    assert_same(
        saved["model"],
        cpu_copy(model.state_dict()),
    )

    expected_random = random_probe(device)

    LOGGER.info(
        "%s: checkpoint read and saved weights: PASS",
        device,
    )

    with capture_training_log(run_dir):
        trainer.fit(
            model,
            loaders["train"],
            loaders["validation"],
            epochs=2,
            on_epoch_end=callback,
            metric_fn=metric_fn,
        )

    # Reconstruction consumes RNG; loading must restore it afterward.
    resumed_loaders = fixture_loaders(seed)

    resumed_model, resumed_trainer = make_objects(
        config,
        device,
    )

    before = cpu_copy(resumed_model.state_dict())
    rejected = False

    try:
        load_checkpoint(
            checkpoint,
            resumed_model,
            resumed_trainer,
            resumed_loaders,
            {"wrong": True},
        )
    except CheckpointError:
        rejected = True

    require(
        rejected,
        "A mismatched run context was accepted.",
    )

    assert_same(
        before,
        cpu_copy(resumed_model.state_dict()),
    )

    LOGGER.info(
        "%s: incompatible context rejected before weight changes: PASS",
        device,
    )

    epoch = load_checkpoint(
        checkpoint,
        resumed_model,
        resumed_trainer,
        resumed_loaders,
        CONTEXT,
    )

    require(
        epoch == 1 and resumed_trainer.global_step == 3,
        "Resume counters differ.",
    )

    assert_same(
        saved["optimizer"],
        cpu_copy(
            resumed_trainer.optimizer.state_dict()
        ),
    )

    assert_same(
        saved["scheduler"],
        resumed_trainer.scheduler.state_dict(),
    )

    assert_same(
        saved["scaler"],
        resumed_trainer.scaler.state_dict(),
    )

    assert_same(
        saved["trainer"]["history"],
        resumed_trainer.history,
    )

    assert_same(
        expected_random,
        random_probe(device),
    )

    for name, loader in resumed_loaders.items():
        assert_same(
            saved["loader_rng"][name],
            loader.generator.get_state(),
        )

    LOGGER.info(
        "%s: optimizer, counters, history and RNG restoration: PASS",
        device,
    )

    # Rebuild logs from authoritative checkpoint history.
    write_history(
        resumed_dir,
        resumed_trainer.history,
    )

    resumed_callback = RunLogger(
        resumed_dir,
        resumed_loaders,
        CONTEXT,
        resume=True,
    )

    with capture_training_log(resumed_dir):
        resumed_trainer.fit(
            resumed_model,
            resumed_loaders["train"],
            resumed_loaders["validation"],
            epochs=2,
            on_epoch_end=resumed_callback,
            metric_fn=lambda model, loader: {
                "check_score": float(
                    resumed_trainer.epoch + 1
                )
            },
        )

    tolerance = (
        0.0
        if device.type == "cpu"
        else 1e-5
    )

    assert_same(
        cpu_copy(model.state_dict()),
        cpu_copy(resumed_model.state_dict()),
        tolerance,
    )

    assert_same(
        cpu_copy(trainer.optimizer.state_dict()),
        cpu_copy(resumed_trainer.optimizer.state_dict()),
        tolerance,
    )

    assert_same(
        trainer.scheduler.state_dict(),
        resumed_trainer.scheduler.state_dict(),
        tolerance,
    )

    assert_same(
        trainer.early_stopping.best,
        resumed_trainer.early_stopping.best,
        tolerance,
    )

    require(
        trainer.early_stopping.bad_epochs
        == resumed_trainer.early_stopping.bad_epochs,
        "Early-stopping counters differ.",
    )

    require(
        trainer.global_step
        == resumed_trainer.global_step
        == 6,
        "Expected six total updates.",
    )

    for a, b in zip(
        trainer.history,
        resumed_trainer.history,
    ):
        for key in a:
            if not key.endswith("seconds"):
                assert_same(
                    a[key],
                    b[key],
                    tolerance,
                )

    LOGGER.info(
        "%s: uninterrupted versus resumed training: PASS",
        device,
    )

    history = json.loads(
        (resumed_dir / "metrics.json").read_text()
    )

    require(
        [row["epoch"] for row in history] == [1, 2],
        "Duplicated/missing metric epochs.",
    )

    csv_lines = (
        resumed_dir / "metrics.csv"
    ).read_text().splitlines()

    require(
        len(csv_lines) == 3,
        "CSV row count differs.",
    )

    log = (
        resumed_dir / "training.log"
    ).read_text()

    require(
        "Epoch 1" in log and "Epoch 2" in log,
        "Persistent training log is incomplete.",
    )

    best = read_checkpoint(
        resolve_checkpoint(
            resumed_dir,
            "best",
        )
    )

    expected_best = min(
        history,
        key=lambda row: row["check_score"],
    )["epoch"]

    require(
        best["trainer"]["epoch"] == expected_best,
        "Wrong best checkpoint.",
    )

    checkpoint_files = list(
        (resumed_dir / "checkpoints").glob("*.pt")
    )

    require(
        len(checkpoint_files) <= 2,
        "Retention failed.",
    )

    LOGGER.info(
        "%s: best/last selection, retention and persistent logs: PASS",
        device,
    )

    return {
        "device": str(device),
        "status": "passed",
        "total_updates": 6,
        "tolerance": tolerance,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/baseline_debug.yaml",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    stamp = (
        datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )
        + "_"
        + uuid.uuid4().hex[:8]
    )

    directory = (
        Path(config["paths"]["artifacts_root"])
        / "checkpoint_checks"
        / stamp
    )

    devices = [torch.device("cpu")]

    selected = select_device(
        config["runtime"]["device"]
    )

    if selected.type != "cpu":
        devices.append(selected)

    results = [
        run_check(
            config,
            device,
            directory / device.type,
        )
        for device in devices
    ]

    report = directory / "report.json"

    write_json(
        report,
        {
            "status": "passed",
            "checks": results,
            "scope": (
                "Synthetic fixture using the real Trainer; "
                "no COCO training or baseline weights."
            ),
        },
    )

    LOGGER.info("Report: %s", report)

    LOGGER.info(
        "PHASE 8 CHECKPOINT AND LOGGING CHECK: PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception(
            "PHASE 8 CHECKPOINT AND LOGGING CHECK: FAIL"
        )
        raise SystemExit(1)