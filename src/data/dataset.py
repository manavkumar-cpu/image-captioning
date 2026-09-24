"""Caption-pair and image-level Dataset views."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.preprocessing import load_image
from src.utils.exceptions import DatasetError


def validate_records(records):
    """Check the record structure before constructing dataset views."""

    if not isinstance(records, list) or not records:
        raise DatasetError("Dataset records must be a nonempty list")

    image_ids = set()
    caption_ids = set()

    for record in records:
        image_id = record["image_id"]

        if image_id in image_ids:
            raise DatasetError(f"Repeated image record: {image_id}")

        image_ids.add(image_id)

        captions = record["captions"]
        ids = record["caption_ids"]

        if not captions or len(captions) != len(ids):
            raise DatasetError(
                f"Invalid caption mapping for image {image_id}"
            )

        for caption_id, caption in zip(ids, captions):
            if caption_id in caption_ids:
                raise DatasetError(f"Repeated caption ID: {caption_id}")

            caption_ids.add(caption_id)

            if not isinstance(caption, str) or not caption.strip():
                raise DatasetError(f"Invalid caption: {caption_id}")


def image_path(raw_root, record):
    """Resolve a manifest path within the COCO image directory."""

    root = Path(raw_root).resolve()
    path = (root / record["relative_path"]).resolve()

    if not path.is_relative_to(root):
        raise DatasetError("Image path escapes the dataset directory")

    return path


class CaptionPairDataset(Dataset):
    """One sample per image-caption pair.

    Token IDs include BOS and EOS. No shifting or padding happens here.
    """

    def __init__(self, records, raw_root, tokenizer, transform):
        validate_records(records)

        self.records = records
        self.raw_root = Path(raw_root)
        self.transform = transform
        self.tokenizer_fingerprint = tokenizer.fingerprint()

        # Small token lists are cached; image tensors are not.
        self.samples = []

        for record_index, record in enumerate(records):
            for caption_id, caption in zip(
                record["caption_ids"],
                record["captions"],
            ):
                self.samples.append(
                    {
                        "record_index": record_index,
                        "caption_id": caption_id,
                        "token_ids": tokenizer.encode(caption),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        record = self.records[sample["record_index"]]

        image = load_image(image_path(self.raw_root, record))
        image = self.transform(image)

        return {
            "image": image,
            "token_ids": torch.tensor(
                sample["token_ids"],
                dtype=torch.long,
            ),
            "image_id": record["image_id"],
            "caption_id": sample["caption_id"],
        }


class CaptionImageDataset(Dataset):
    """One sample per image, retaining all original reference captions."""

    def __init__(self, records, raw_root, transform):
        validate_records(records)

        self.records = records
        self.raw_root = Path(raw_root)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image = load_image(image_path(self.raw_root, record))
        image = self.transform(image)

        return {
            "image": image,
            "image_id": record["image_id"],
            "references": list(record["captions"]),
        }