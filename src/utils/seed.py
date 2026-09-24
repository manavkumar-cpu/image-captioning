"""Seed libraries; exact cross-device reproducibility is not guaranteed."""

import random


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Used with a seeded DataLoader generator in Phase 5."""

    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)