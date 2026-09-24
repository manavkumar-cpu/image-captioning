"""Validation-driven learning-rate scheduling."""

from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(optimizer, config):
    return ReduceLROnPlateau(
        optimizer,
        mode=config["training"]["monitor_mode"],
        factor=0.5,
        patience=max(1, config["training"]["patience"] // 2),
        threshold=0.0,
        threshold_mode="abs",
        min_lr=0.0,
    )