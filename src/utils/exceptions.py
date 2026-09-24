"""Errors with useful context at project boundaries."""


class CaptioningError(Exception):
    """Base project error."""


class ConfigError(CaptioningError):
    """Invalid configuration."""


class EnvironmentCheckError(CaptioningError):
    """A required package or device is unavailable."""


class DatasetError(CaptioningError):
    """Dataset is missing or invalid."""


class CheckpointError(CaptioningError):
    """Checkpoint cannot be saved or restored."""