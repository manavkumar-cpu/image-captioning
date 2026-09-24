"""Download COCO annotations and a reproducible development image subset."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
from zipfile import ZipFile

from src.utils.config import load_config
from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger


LOGGER = get_logger("image_captioning.data")

# HTTPS access to the COCO image-storage bucket.
BASE_URL = "https://s3.amazonaws.com/images.cocodataset.org"

ANNOTATION_ARCHIVE = "annotations_trainval2017.zip"

ANNOTATION_FILES = (
    "captions_train2017.json",
    "captions_val2017.json",
)


def sha256_file(path: Path) -> str:
    """Hash a file without loading the entire file into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def json_fingerprint(value) -> str:
    """Create a stable identity for JSON-compatible metadata."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def save_immutable_json(path: Path, value) -> None:
    """Reuse identical metadata; refuse to overwrite different metadata."""

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))

        if existing != value:
            raise DatasetError(
                f"Existing metadata differs: {path}. "
                "No overwrite was performed. "
                "Check the configuration and dataset before proceeding."
            )

        LOGGER.info("Reusing metadata: %s", path.name)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")

    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def download_file(url: str, destination: Path) -> None:
    """Download atomically; retry interrupted downloads up to three times."""

    if destination.is_file() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")

    last_error = None

    for attempt in range(1, 4):
        try:
            request = Request(
                url,
                headers={"User-Agent": "image-captioning-project/1.0"},
            )

            with urlopen(request, timeout=60) as response:
                expected = response.headers.get("Content-Length")
                expected = int(expected) if expected is not None else None

                downloaded = 0
                next_progress = 32 * 1024 * 1024

                with temporary.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)

                        if not chunk:
                            break

                        output.write(chunk)
                        downloaded += len(chunk)

                        if downloaded >= next_progress:
                            LOGGER.info(
                                "%s: %.1f MiB downloaded",
                                destination.name,
                                downloaded / (1024 * 1024),
                            )
                            next_progress += 32 * 1024 * 1024

            if downloaded == 0:
                raise DatasetError(f"Empty download: {url}")

            if expected is not None and downloaded != expected:
                raise DatasetError(
                    f"Incomplete download: expected {expected} bytes, "
                    f"received {downloaded}"
                )

            temporary.replace(destination)
            return

        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)

            LOGGER.warning(
                "Download attempt %d/3 failed for %s: %s",
                attempt,
                destination.name,
                exc,
            )

    raise DatasetError(
        f"Download failed after three attempts: {url}"
    ) from last_error


def prepare_annotations(raw_root: Path) -> None:
    """Extract only the two caption annotation files we need."""

    annotation_dir = raw_root / "annotations"

    if all(
        (annotation_dir / name).is_file()
        for name in ANNOTATION_FILES
    ):
        LOGGER.info("Annotation JSON files already exist")
        return

    archive_path = raw_root / ANNOTATION_ARCHIVE

    LOGGER.info("Preparing official COCO annotation archive")

    download_file(
        f"{BASE_URL}/annotations/{ANNOTATION_ARCHIVE}",
        archive_path,
    )

    annotation_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive_path) as archive:
        for name in ANNOTATION_FILES:
            destination = annotation_dir / name

            if destination.is_file():
                continue

            temporary = destination.with_name(destination.name + ".part")

            # Extract a named member rather than extracting the whole archive.
            member = f"annotations/{name}"

            with archive.open(member) as source:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)

            temporary.replace(destination)
            LOGGER.info("Extracted %s", name)


def read_caption_records(path: Path, source_split: str) -> dict:
    """Return image_id -> image metadata plus all its captions."""

    document = json.loads(path.read_text(encoding="utf-8"))

    images = document.get("images")
    annotations = document.get("annotations")

    if not isinstance(images, list) or not isinstance(annotations, list):
        raise DatasetError(f"Invalid COCO annotation structure: {path}")

    records = {}

    for image in images:
        image_id = image["id"]
        filename = image["file_name"]

        if type(image_id) is not int or image_id < 0:
            raise DatasetError(f"Invalid image ID: {image_id}")

        if image_id in records:
            raise DatasetError(f"Duplicate image ID: {image_id}")

        expected_filename = f"{image_id:012d}.jpg"

        if filename != expected_filename:
            raise DatasetError(
                f"Unexpected filename for image {image_id}: {filename}"
            )

        width = image["width"]
        height = image["height"]

        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
        ):
            raise DatasetError(f"Invalid dimensions for image {image_id}")

        records[image_id] = {
            "image_id": image_id,
            "file_name": filename,
            "relative_path": f"{source_split}/{filename}",
            "width": width,
            "height": height,
            "captions": [],
            "caption_ids": [],
        }

    grouped = defaultdict(list)
    seen_caption_ids = set()

    for annotation in annotations:
        caption_id = annotation["id"]
        image_id = annotation["image_id"]
        caption = annotation["caption"]

        if type(caption_id) is not int:
            raise DatasetError(f"Invalid caption ID: {caption_id}")

        if caption_id in seen_caption_ids:
            raise DatasetError(f"Duplicate caption ID: {caption_id}")

        seen_caption_ids.add(caption_id)

        if image_id not in records:
            raise DatasetError(
                f"Caption {caption_id} references missing image {image_id}"
            )

        if not isinstance(caption, str) or not caption.strip():
            raise DatasetError(f"Empty or invalid caption: {caption_id}")

        grouped[image_id].append((caption_id, caption))

    for image_id, record in records.items():
        pairs = sorted(grouped[image_id])

        if not pairs:
            raise DatasetError(f"Image {image_id} has no captions")

        record["caption_ids"] = [pair[0] for pair in pairs]
        record["captions"] = [pair[1] for pair in pairs]

    return records


def ordered_ids(image_ids, seed: int, namespace: str) -> list:
    """Deterministic pseudo-random ordering, independent of input order."""

    def key(image_id):
        text = f"{seed}:{namespace}:{image_id}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest(), image_id

    return sorted(image_ids, key=key)


def build_metadata(config: dict, val_images: int):
    """Build canonical splits and a smaller development manifest."""

    if config["data"]["dataset"] != "coco2017":
        raise DatasetError("This phase supports coco2017 only")

    raw_root = Path(config["paths"]["data_root"]) / "raw" / "coco2017"
    annotation_dir = raw_root / "annotations"

    train_path = annotation_dir / "captions_train2017.json"
    val_path = annotation_dir / "captions_val2017.json"

    train_records = read_caption_records(train_path, "train2017")
    val_records = read_caption_records(val_path, "val2017")

    if set(train_records) & set(val_records):
        raise DatasetError("Official train and validation image IDs overlap")

    if len(val_records) != 5000:
        raise DatasetError(
            f"Expected 5000 COCO 2017 validation images; "
            f"found {len(val_records)}"
        )

    seed = config["data"]["split_seed"]
    train_count = config["data"]["debug_images"]

    validation_order = ordered_ids(val_records, seed, "validation-test")

    validation_ids = sorted(validation_order[:2500])
    test_ids = sorted(validation_order[2500:])

    split_document = {
        "schema_version": 1,
        "dataset": "coco2017",
        "protocol": "official_train__official_val_halved_v1",
        "split_seed": seed,
        "annotation_sha256": {
            "captions_train2017.json": sha256_file(train_path),
            "captions_val2017.json": sha256_file(val_path),
        },
        "train": sorted(train_records),
        "validation": validation_ids,
        "test": test_ids,
    }

    if not 1 <= train_count <= len(train_records):
        raise DatasetError("debug_images is outside the training-set size")

    if not 1 <= val_images <= len(validation_ids):
        raise DatasetError("val-images must be between 1 and 2500")

    selected_train = ordered_ids(
        train_records, seed, "development-train"
    )[:train_count]

    selected_validation = ordered_ids(
        validation_ids, seed, "development-validation"
    )[:val_images]

    manifest = {
        "schema_version": 1,
        "dataset": "coco2017",
        "split_fingerprint": json_fingerprint(split_document),
        "train": [train_records[i] for i in selected_train],
        "validation": [val_records[i] for i in selected_validation],
    }

    return split_document, manifest


def metadata_paths(config: dict, val_images: int):
    metadata_root = (
        Path(config["paths"]["data_root"])
        / "metadata"
        / "coco2017"
    )

    train_count = config["data"]["debug_images"]

    return (
        metadata_root / "splits.json",
        metadata_root
        / f"development_train{train_count}_val{val_images}.json",
    )


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

        if not 1 <= args.val_images <= 2500:
            raise DatasetError("val-images must be between 1 and 2500")

        raw_root = (
            Path(config["paths"]["data_root"])
            / "raw"
            / "coco2017"
        )

        prepare_annotations(raw_root)

        splits, manifest = build_metadata(config, args.val_images)
        split_path, manifest_path = metadata_paths(config, args.val_images)

        save_immutable_json(split_path, splits)
        save_immutable_json(manifest_path, manifest)

        LOGGER.info(
            "Canonical splits: train=%d, validation=%d, test=%d",
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["test"]),
        )

        for split_name in ("train", "validation"):
            records = manifest[split_name]

            LOGGER.info(
                "Preparing %d %s images",
                len(records),
                split_name,
            )

            for index, record in enumerate(records, start=1):
                relative_path = record["relative_path"]

                download_file(
                    f"{BASE_URL}/{relative_path}",
                    raw_root / relative_path,
                )

                if index % 50 == 0 or index == len(records):
                    LOGGER.info(
                        "%s images available: %d/%d",
                        split_name,
                        index,
                        len(records),
                    )

        LOGGER.info("Development manifest: %s", manifest_path)
        LOGGER.info(
            "Preparation finished. Run src.data.validate "
            "to verify metadata and image contents."
        )

        return 0

    except Exception as exc:
        LOGGER.error("DATA PREPARATION FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())