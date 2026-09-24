"""Versioned, epoch-boundary training checkpoints."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from src.utils.exceptions import CheckpointError


SCHEMA_VERSION = 1


def json_value(value):
    """Normalize config Paths/tuples to portable JSON values."""
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Not JSON-compatible: {type(obj).__name__}")

    return json.loads(
        json.dumps(value, default=convert, allow_nan=False)
    )


def atomic_write(path, writer):
    """Replace one file only after its complete contents reach disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=".writing-",
    )

    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    data = json.dumps(
        json_value(value),
        indent=2,
        allow_nan=False,
    ) + "\n"

    atomic_write(
        path,
        lambda stream: stream.write(data.encode("utf-8")),
    )


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()

    if isinstance(value, dict):
        return {
            key: cpu_copy(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [cpu_copy(item) for item in value]

    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)

    return deepcopy(value)


def class_name(obj):
    if obj is None:
        return None

    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def contract(model, trainer, loaders, context):
    names = {
        id(parameter): name
        for name, parameter in model.named_parameters()
    }

    for loader in loaders.values():
        if loader.generator is None or loader.num_workers != 0:
            raise CheckpointError(
                "Resume currently requires a generator and num_workers=0."
            )

    return json_value({
        "config": trainer.config,
        "context": context,
        "device": str(trainer.device),
        "torch_version": str(torch.__version__),
        "model_class": class_name(model),
        "optimizer_class": class_name(trainer.optimizer),
        "scheduler_class": class_name(trainer.scheduler),
        "loss_class": class_name(trainer.loss_fn),
        "parameters": [
            [
                name,
                list(parameter.shape),
                str(parameter.dtype),
                parameter.requires_grad,
            ]
            for name, parameter in model.named_parameters()
        ],
        "optimizer_groups": [
            [names[id(parameter)] for parameter in group["params"]]
            for group in trainer.optimizer.param_groups
        ],
        "loaders": {
            name: {
                "samples": len(loader.dataset),
                "batch_size": loader.batch_size,
                "drop_last": loader.drop_last,
                "sampler": class_name(loader.sampler),
            }
            for name, loader in loaders.items()
        },
    })


def capture_rng(device):
    numpy_state = np.random.get_state()

    state = {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            *numpy_state[2:],
        ],
        "torch": torch.get_rng_state(),
    }

    if device.type == "mps":
        state["device"] = torch.mps.get_rng_state().cpu()

    elif device.type == "cuda":
        state["device"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng(state, device):
    random.setstate(state["python"])

    name, keys, position, has_gauss, cached = state["numpy"]

    np.random.set_state((
        name,
        np.asarray(keys, dtype=np.uint32),
        position,
        has_gauss,
        cached,
    ))

    torch.set_rng_state(state["torch"])

    if device.type == "mps":
        torch.mps.set_rng_state(state["device"])

    elif device.type == "cuda":
        torch.cuda.set_rng_state_all(state["device"])


def save_checkpoint(path, model, trainer, loaders, context):
    if trainer.epoch < 1 or len(trainer.history) != trainer.epoch:
        raise CheckpointError(
            "Save only after a completed epoch."
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": contract(
            model,
            trainer,
            loaders,
            context,
        ),
        "model": cpu_copy(model.state_dict()),
        "model_training": model.training,
        "optimizer": cpu_copy(trainer.optimizer.state_dict()),
        "scheduler": (
            None
            if trainer.scheduler is None
            else trainer.scheduler.state_dict()
        ),
        "scaler": trainer.scaler.state_dict(),
        "trainer": {
            "epoch": trainer.epoch,
            "global_step": trainer.global_step,
            "history": json_value(trainer.history),
            "early_stopping": {
                "best": trainer.early_stopping.best,
                "bad_epochs": trainer.early_stopping.bad_epochs,
            },
        },
        "rng": capture_rng(trainer.device),
        "loader_rng": {
            name: loader.generator.get_state()
            for name, loader in loaders.items()
        },
    }

    atomic_write(
        path,
        lambda stream: torch.save(payload, stream),
    )


def read_checkpoint(path):
    try:
        payload = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise CheckpointError(
            f"Cannot read checkpoint {path}: {exc}"
        ) from exc

    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise CheckpointError(
            "Unsupported checkpoint format."
        )

    return payload


def load_checkpoint(path, model, trainer, loaders, context):
    """Restore freshly constructed objects before creating loader iterators."""
    payload = read_checkpoint(path)

    expected = contract(
        model,
        trainer,
        loaders,
        context,
    )

    if payload["contract"] != expected:
        differences = [
            key
            for key in expected
            if payload["contract"].get(key) != expected[key]
        ]

        raise CheckpointError(
            f"Resume contract mismatch: {', '.join(differences)}"
        )

    state = payload["trainer"]

    if len(state["history"]) != state["epoch"]:
        raise CheckpointError(
            "Checkpoint history/epoch mismatch."
        )

    model.load_state_dict(
        payload["model"],
        strict=True,
    )

    trainer.optimizer.load_state_dict(payload["optimizer"])

    if trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(payload["scheduler"])

    trainer.scaler.load_state_dict(payload["scaler"])

    trainer.epoch = state["epoch"]
    trainer.global_step = state["global_step"]
    trainer.history = deepcopy(state["history"])

    trainer.early_stopping.best = (
        state["early_stopping"]["best"]
    )

    trainer.early_stopping.bad_epochs = (
        state["early_stopping"]["bad_epochs"]
    )

    trainer.optimizer.zero_grad(set_to_none=True)
    model.train(payload["model_training"])

    for name, loader in loaders.items():
        loader.generator.set_state(
            payload["loader_rng"][name]
        )

    restore_rng(payload["rng"], trainer.device)

    return state["epoch"]


def resolve_checkpoint(run_dir, which="last"):
    if which not in {"last", "best"}:
        raise ValueError(
            "which must be 'last' or 'best'."
        )

    directory = Path(run_dir) / "checkpoints"
    index = json.loads(
        (directory / "index.json").read_text()
    )

    if index.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError(
            "Unsupported checkpoint index."
        )

    filename = index[which]

    if (
        Path(filename).name != filename
        or not filename.endswith(".pt")
    ):
        raise CheckpointError(
            "Invalid checkpoint filename."
        )

    path = directory / filename

    if not path.is_file():
        raise CheckpointError(
            f"Missing {which} checkpoint: {path}"
        )

    return path