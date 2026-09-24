"""Device selection fails explicitly for unavailable requested accelerators."""

from .exceptions import EnvironmentCheckError


def select_device(requested="auto"):
    import torch

    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise EnvironmentCheckError(
            f"Unknown device: {requested}"
        )

    mps = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

    cuda = torch.cuda.is_available()

    if requested == "auto":
        requested = (
            "cuda" if cuda
            else "mps" if mps
            else "cpu"
        )

    if (
        (requested == "cuda" and not cuda)
        or (requested == "mps" and not mps)
    ):
        raise EnvironmentCheckError(
            f"Requested {requested} is unavailable"
        )

    return torch.device(requested)