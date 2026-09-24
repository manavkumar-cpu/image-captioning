"""Reusable image loading and preprocessing for image captioning."""

import math
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.utils.exceptions import ConfigError, DatasetError


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validate_normalization(mean, std):
    """Validate and return three-channel normalization parameters."""

    if len(mean) != 3 or len(std) != 3:
        raise ConfigError("Normalization requires three means and stds")

    mean = tuple(float(value) for value in mean)
    std = tuple(float(value) for value in std)

    if not all(math.isfinite(value) for value in mean + std):
        raise ConfigError("Normalization values must be finite")

    if any(value <= 0 for value in std):
        raise ConfigError("Normalization standard deviations must be positive")

    return mean, std


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Make grayscale and other PIL image modes compatible with the encoder."""

    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image")

    return image.convert("RGB")


def load_image(path: str | Path) -> Image.Image:
    """Read an image and return RGB pixels independent of the open file."""

    path = Path(path).expanduser()

    if not path.is_file():
        raise DatasetError(f"Image file does not exist: {path}")

    try:
        with Image.open(path) as image:
            image.load()
            return ensure_rgb(image)

    except Exception as exc:
        raise DatasetError(
            f"Cannot load image {path}: {exc}"
        ) from exc


def build_image_transform(
    image_size: int,
    split: str = "train",
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    """Build a CPU image transform with explicit normalization settings.

    Train, validation, test, and inference currently use identical,
    deterministic preprocessing. No augmentation is enabled.
    """

    if type(image_size) is not int or image_size <= 0:
        raise ConfigError("image_size must be a positive integer")

    if split not in {"train", "validation", "test", "inference"}:
        raise ConfigError(f"Unsupported preprocessing split: {split}")

    mean, std = validate_normalization(mean, std)

    return transforms.Compose(
        [
            ensure_rgb,
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def denormalize_image(
    tensor: torch.Tensor,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    """Undo channel normalization for visualization.

    Returns a detached CPU tensor. Does not clamp values, so checks
    can detect incorrect ranges.
    """

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Expected one image with shape [3, H, W]")

    if not tensor.is_floating_point():
        raise ValueError("Expected a floating-point image tensor")

    mean, std = validate_normalization(mean, std)

    image = tensor.detach().cpu()

    mean_tensor = image.new_tensor(mean).view(3, 1, 1)
    std_tensor = image.new_tensor(std).view(3, 1, 1)

    return image * std_tensor + mean_tensor