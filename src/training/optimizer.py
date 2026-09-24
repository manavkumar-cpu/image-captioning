"""Architecture-independent AdamW construction."""

import math

from torch.optim import AdamW

from src.utils.exceptions import ConfigError


def build_optimizer(model, config, parameter_groups=None):
    """Optimize all trainable parameters exactly once.

    Optional groups may specify separate learning rates:
    {"name": "...", "params": [...], "lr": ...}
    """

    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        raise ConfigError("Model has no trainable parameters")

    if parameter_groups is None:
        groups = [
            {
                "name": "trainable",
                "params": trainable,
                "lr": learning_rate,
            }
        ]
    else:
        groups = []

        for group in parameter_groups:
            copied = dict(group)
            copied["params"] = list(copied["params"])
            groups.append(copied)

    selected = [
        parameter
        for group in groups
        for parameter in group["params"]
    ]

    selected_ids = [id(parameter) for parameter in selected]
    expected_ids = {id(parameter) for parameter in trainable}

    if len(selected_ids) != len(set(selected_ids)):
        raise ConfigError("A parameter appears in multiple optimizer groups")

    if set(selected_ids) != expected_ids:
        raise ConfigError(
            "Optimizer groups must contain every trainable parameter "
            "exactly once, and no frozen or unrelated parameters"
        )

    for group in groups:
        if not group["params"]:
            raise ConfigError("Optimizer groups cannot be empty")

        rate = group.get("lr", learning_rate)
        decay = group.get("weight_decay", weight_decay)

        if not math.isfinite(rate) or rate <= 0:
            raise ConfigError("Optimizer learning rates must be positive")

        if not math.isfinite(decay) or decay < 0:
            raise ConfigError("Optimizer weight decay must be nonnegative")

    return AdamW(
        groups,
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
    )