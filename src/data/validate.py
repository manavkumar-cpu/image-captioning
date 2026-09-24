"""Validate COCO split metadata, caption mappings, and development images."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from src.data.prepare import (
    build_metadata,
    metadata_paths,
    sha256_file,
)
from src.utils.config import load_config
from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger


LOGGER = get_logger("image_captioning.data_validation")


def validate_image(path: Path, record: dict) -> str:
    """Check file structure, full decoding and annotated dimensions."""

    if not path.is_file():
        raise DatasetError(f"Missing image: {path}")

    if path.stat().st_size == 0:
        raise DatasetError(f"Empty image file: {path}")

    try:
        with Image.open(path) as image:
            image.verify()

        # Reopen because verify() does not leave the image ready for use.
        with Image.open(path) as image:
            image.load()

            actual_size = image.size
            expected_size = (
                record["width"],
                record["height"],
            )

            if actual_size != expected_size:
                raise DatasetError(
                    f"Image dimensions differ for {path.name}: "
                    f"expected {expected_size}, got {actual_size}"
                )

    except DatasetError:
        raise

    except Exception as exc:
        raise DatasetError(
            f"Cannot decode image {path}: {exc}"
        ) from exc

    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default="configs/baseline_debug.yaml",
    )
    parser.add_argument("--val-images", type=int, default=250)

    args = parser.parse_args()

    try:
        config = load_config(args.config)

        split_path, manifest_path = metadata_paths(
            config, args.val_images
        )

        if not split_path.is_file() or not manifest_path.is_file():
            raise DatasetError(
                "Metadata files are missing. Run src.data.prepare first."
            )

        splits = json.loads(
            split_path.read_text(encoding="utf-8")
        )
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        # Rebuild expected mappings from the original annotation files.
        expected_splits, expected_manifest = build_metadata(
            config, args.val_images
        )

        if splits != expected_splits:
            raise DatasetError(
                "Saved splits do not match the source annotations/config"
            )

        if manifest != expected_manifest:
            raise DatasetError(
                "Saved image-caption mappings do not match "
                "the source annotations/config"
            )

        LOGGER.info("Source annotation and manifest consistency: PASS")

        split_sets = {}

        for name in ("train", "validation", "test"):
            ids = splits[name]

            if len(ids) != len(set(ids)):
                raise DatasetError(f"Duplicate IDs in {name}")

            split_sets[name] = set(ids)

        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            if split_sets[first] & split_sets[second]:
                raise DatasetError(
                    f"Image-ID leakage between {first} and {second}"
                )

        LOGGER.info("Canonical image-ID split separation: PASS")

        raw_root = (
            Path(config["paths"]["data_root"])
            / "raw"
            / "coco2017"
        )

        seen_file_hashes = {}
        combined_digest = hashlib.sha256()

        for split_name in ("train", "validation"):
            records = manifest[split_name]
            selected_ids = [record["image_id"] for record in records]

            if len(selected_ids) != len(set(selected_ids)):
                raise DatasetError(
                    f"Repeated development image in {split_name}"
                )

            if not set(selected_ids) <= split_sets[split_name]:
                raise DatasetError(
                    f"Development images outside {split_name} partition"
                )

            caption_count = 0

            for index, record in enumerate(records, start=1):
                captions = record["captions"]
                caption_ids = record["caption_ids"]

                if not captions or len(captions) != len(caption_ids):
                    raise DatasetError(
                        f"Invalid captions for image {record['image_id']}"
                    )

                path = raw_root / record["relative_path"]
                digest = validate_image(path, record)

                previous_split = seen_file_hashes.get(digest)

                if (
                    previous_split is not None
                    and previous_split != split_name
                ):
                    raise DatasetError(
                        "Identical image-file bytes found across "
                        f"development splits: {path}"
                    )

                seen_file_hashes[digest] = split_name

                combined_digest.update(
                    (
                        f"{split_name}:{record['image_id']}:{digest}\n"
                    ).encode("utf-8")
                )

                caption_count += len(captions)

                if index % 100 == 0 or index == len(records):
                    LOGGER.info(
                        "%s image checks: %d/%d",
                        split_name,
                        index,
                        len(records),
                    )

            LOGGER.info(
                "%s: %d images, %d reference captions",
                split_name,
                len(records),
                caption_count,
            )

        LOGGER.info("Image files and dimensions: PASS")
        LOGGER.info("Development cross-split exact-file check: PASS")

        LOGGER.info(
            "Development image fingerprint: %s",
            combined_digest.hexdigest(),
        )

        LOGGER.info(
            "Scope: development images verified; "
            "test images and remaining training images not downloaded"
        )

        LOGGER.info("PHASE 2 DATASET CHECK: PASS")
        return 0

    except Exception as exc:
        LOGGER.error("DATA VALIDATION FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())