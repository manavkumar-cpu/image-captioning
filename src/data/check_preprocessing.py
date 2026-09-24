"""Check preprocessing numerically and create a small visual preview."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import textwrap
from uuid import uuid4

from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision.transforms import functional as TF

from src.data.prepare import metadata_paths
from src.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_image_transform,
    denormalize_image,
    load_image,
)
from src.utils.config import load_config
from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger


LOGGER = get_logger("image_captioning.preprocessing")


def check_tensor(tensor, image_size):
    """Validate shape, type, device, and finite values."""

    expected_shape = (3, image_size, image_size)

    if tuple(tensor.shape) != expected_shape:
        raise DatasetError(
            f"Expected shape {expected_shape}, got {tuple(tensor.shape)}"
        )

    if tensor.dtype != torch.float32:
        raise DatasetError(f"Expected float32, got {tensor.dtype}")

    if tensor.device.type != "cpu":
        raise DatasetError("Preprocessing should return a CPU tensor")

    if not torch.isfinite(tensor).all().item():
        raise DatasetError("Preprocessing produced NaN or infinite values")


def check_known_pixels(transform, image_size):
    """Check normalization against manually calculated pixel values."""

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for pixel_value in (0, 128, 255):
        image = Image.new(
            "RGB",
            (19, 13),
            color=(pixel_value, pixel_value, pixel_value),
        )

        result = transform(image)
        check_tensor(result, image_size)

        expected = (pixel_value / 255.0 - mean) / std
        expected = expected.expand_as(result)

        if not torch.allclose(result, expected, atol=1e-6, rtol=0):
            raise DatasetError(
                f"Normalization check failed for pixel value {pixel_value}"
            )

    # Non-RGB images must also produce three channels.
    for image in (
        Image.new("L", (17, 11), color=128),
        Image.new("RGBA", (17, 11), color=(100, 150, 200, 255)),
    ):
        check_tensor(transform(image), image_size)

    LOGGER.info("Known-pixel normalization and RGB conversion: PASS")


def make_preview(pairs, destination):
    """Show original images beside resized, denormalized model inputs."""

    tile_size = 224
    card_width = 480
    card_height = 330
    columns = 2
    rows = math.ceil(len(pairs) / columns)

    canvas = Image.new(
        "RGB",
        (columns * card_width, rows * card_height),
        color="white",
    )

    draw = ImageDraw.Draw(canvas)

    for index, pair in enumerate(pairs):
        column = index % columns
        row = index // columns

        x = column * card_width
        y = row * card_height

        # Padding here is only for displaying the original thumbnail.
        original_preview = ImageOps.pad(
            pair["original"],
            (tile_size, tile_size),
            method=Image.Resampling.BILINEAR,
            color=(235, 235, 235),
        )

        recovered = denormalize_image(pair["tensor"])
        processed_preview = TF.to_pil_image(recovered.clamp(0, 1))

        processed_preview = processed_preview.resize(
            (tile_size, tile_size),
            resample=Image.Resampling.BILINEAR,
        )

        title = f'{pair["split"]} | image {pair["image_id"]}'

        draw.text((x + 8, y + 8), title, fill="black")
        draw.text((x + 8, y + 28), "Original", fill="black")
        draw.text((x + 248, y + 28), "Model input: colors restored", fill="black")

        canvas.paste(original_preview, (x + 8, y + 48))
        canvas.paste(processed_preview, (x + 248, y + 48))

        caption = pair["caption"].encode(
            "ascii", errors="replace"
        ).decode("ascii")

        caption_lines = textwrap.wrap(caption, width=65)[:3]

        for line_number, line in enumerate(caption_lines):
            draw.text(
                (x + 8, y + 280 + line_number * 14),
                line,
                fill="black",
            )

    canvas.save(destination, quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default="configs/baseline_debug.yaml",
    )
    parser.add_argument("--val-images", type=int, default=250)
    parser.add_argument("--samples-per-split", type=int, default=4)

    args = parser.parse_args()

    try:
        if not 1 <= args.samples_per_split <= 16:
            raise DatasetError("samples-per-split must be between 1 and 16")

        config = load_config(args.config)
        image_size = config["data"]["image_size"]

        _, manifest_path = metadata_paths(config, args.val_images)

        if not manifest_path.is_file():
            raise DatasetError(
                f"Development manifest not found: {manifest_path}"
            )

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        train_transform = build_image_transform(
            image_size=image_size,
            split="train",
        )

        validation_transform = build_image_transform(
            image_size=image_size,
            split="validation",
        )

        check_known_pixels(train_transform, image_size)

        raw_root = (
            Path(config["paths"]["data_root"])
            / "raw"
            / "coco2017"
        )

        pairs = []
        records_for_report = []

        for split in ("train", "validation"):
            records = manifest[split][:args.samples_per_split]

            if len(records) != args.samples_per_split:
                raise DatasetError(
                    f"Not enough images in {split} for this sample count"
                )

            for record in records:
                path = raw_root / record["relative_path"]

                original = load_image(path)

                train_tensor = train_transform(original)
                validation_tensor = validation_transform(original)

                check_tensor(train_tensor, image_size)
                check_tensor(validation_tensor, image_size)

                if not torch.equal(train_tensor, validation_tensor):
                    raise DatasetError(
                        "Current train and validation transforms should match"
                    )

                repeated = train_transform(original)

                if not torch.equal(train_tensor, repeated):
                    raise DatasetError(
                        "Current preprocessing should be deterministic"
                    )

                restored = denormalize_image(train_tensor)

                if (
                    restored.min().item() < -1e-6
                    or restored.max().item() > 1.0 + 1e-6
                ):
                    raise DatasetError(
                        "Denormalized pixels are outside the expected range"
                    )

                info = {
                    "split": split,
                    "image_id": record["image_id"],
                    "original_size_wh": list(original.size),
                    "tensor_shape": list(train_tensor.shape),
                    "dtype": str(train_tensor.dtype),
                    "normalized_min": train_tensor.min().item(),
                    "normalized_max": train_tensor.max().item(),
                }

                records_for_report.append(info)

                pairs.append(
                    {
                        "split": split,
                        "image_id": record["image_id"],
                        "original": original,
                        "tensor": train_tensor,
                        "caption": record["captions"][0],
                    }
                )

                LOGGER.info(
                    "%s image %s: original=%s, tensor=%s",
                    split,
                    record["image_id"],
                    original.size,
                    tuple(train_tensor.shape),
                )

        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            + "_"
            + uuid4().hex[:8]
        )

        output_dir = (
            Path(config["paths"]["artifacts_root"])
            / "preprocessing"
            / run_id
        )

        output_dir.mkdir(parents=True, exist_ok=False)

        preview_path = output_dir / "preview.jpg"
        make_preview(pairs, preview_path)

        report = {
            "status": "automated_checks_passed",
            "visual_review": "pending_user_review",
            "manifest": str(manifest_path),
            "split_fingerprint": manifest["split_fingerprint"],
            "images_checked": len(pairs),
            "image_size": image_size,
            "resize_policy": "full_image_square_resize",
            "interpolation": "bilinear",
            "antialias": True,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "augmentation": "none",
            "records": records_for_report,
        }

        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        LOGGER.info("Shape, dtype, finite values and pixel range: PASS")
        LOGGER.info("Determinism and train/validation consistency: PASS")
        LOGGER.info("Images checked: %d", len(pairs))
        LOGGER.info("Preview: %s", preview_path)
        LOGGER.info("Report: %s", output_dir / "report.json")
        LOGGER.info("PHASE 3 AUTOMATED CHECKS: PASS")
        LOGGER.info("Open preview.jpg to complete the visual review")

        return 0

    except Exception as exc:
        LOGGER.error("PREPROCESSING CHECK FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())