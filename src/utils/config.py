"""Load and validate baseline setup configuration; paths use the project root."""

from pathlib import Path
import math

import yaml

from .exceptions import ConfigError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"Cannot read configuration {path}: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a YAML mapping.")

    def value(section, key, expected):
        group = config.get(section)

        if not isinstance(group, dict) or key not in group:
            raise ConfigError(f"Missing {section}.{key}")

        item = group[key]

        if isinstance(item, bool) and expected is not bool:
            raise ConfigError(
                f"Invalid boolean for {section}.{key}"
            )

        if not isinstance(item, expected):
            raise ConfigError(f"Invalid type for {section}.{key}")

        return item

    string_fields = {
        "project": ["name"],
        "paths": ["data_root", "artifacts_root"],
        "runtime": ["device"],
        "data": ["dataset", "tokenizer_name"],
        "model": ["name", "encoder"],
        "training": ["monitor", "monitor_mode"],
    }

    for section, keys in string_fields.items():
        for key in keys:
            if not value(section, key, str).strip():
                raise ConfigError(f"Empty {section}.{key}")

    positive_integer_fields = {
        "data": [
            "debug_images",
            "image_size",
            "max_caption_length",
        ],
        "model": [
            "embedding_dim",
            "hidden_dim",
            "num_layers",
        ],
        "training": [
            "batch_size",
            "epochs",
            "patience",
        ],
    }

    for section, keys in positive_integer_fields.items():
        for key in keys:
            if value(section, key, int) <= 0:
                raise ConfigError(
                    f"{section}.{key} must be positive"
                )

    nonnegative_integer_fields = [
        ("project", "seed"),
        ("data", "split_seed"),
        ("runtime", "num_workers"),
    ]

    for section, key in nonnegative_integer_fields:
        if value(section, key, int) < 0:
            raise ConfigError(
                f"{section}.{key} must be nonnegative"
            )

    boolean_fields = [
        ("runtime", "mixed_precision"),
        ("model", "pretrained"),
        ("model", "freeze_encoder"),
        ("logging", "tensorboard"),
    ]

    for section, key in boolean_fields:
        value(section, key, bool)

    for key in [
        "learning_rate",
        "weight_decay",
        "gradient_clip",
    ]:
        number = value("training", key, (int, float))

        invalid = (
            not math.isfinite(number)
            or number < 0
            or (key != "weight_decay" and number == 0)
        )

        if invalid:
            raise ConfigError(f"Invalid training.{key}")

    dropout = value("model", "dropout", (int, float))

    if not 0 <= dropout < 1:
        raise ConfigError("model.dropout must be in [0, 1)")

    if config["data"]["max_caption_length"] < 3:
        raise ConfigError(
            "max_caption_length must allow BOS, a token, and EOS"
        )

    if config["runtime"]["device"] not in {
        "auto",
        "cpu",
        "cuda",
        "mps",
    }:
        raise ConfigError(
            "runtime.device must be auto, cpu, cuda, or mps"
        )

    if config["training"]["monitor_mode"] not in {"min", "max"}:
        raise ConfigError("monitor_mode must be min or max")

    if config["model"]["name"] != "cnn_lstm":
        raise ConfigError(
            "Phase 1 config supports cnn_lstm; "
            "advanced schema arrives with Person 2"
        )

    for key, raw in config["paths"].items():
        if key in {"data_root", "artifacts_root"}:
            resolved = Path(raw).expanduser()

            config["paths"][key] = str(
                (PROJECT_ROOT / resolved).resolve()
            )

    return config