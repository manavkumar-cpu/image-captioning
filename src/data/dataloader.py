"""Construct reusable loaders from the verified development manifest."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.collate import CaptionCollator, collate_images
from src.data.dataset import CaptionPairDataset, CaptionImageDataset
from src.data.prepare import json_fingerprint, metadata_paths
from src.data.preprocessing import build_image_transform
from src.data.tokenizer import load_caption_tokenizer
from src.utils.exceptions import DatasetError
from src.utils.seed import seed_worker


def make_loader(
    dataset,
    batch_size,
    num_workers,
    shuffle,
    seed,
    collate_fn,
):
    """Create an independently seeded loader that retains the final batch."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_dataloaders(config: dict, val_images: int = 250) -> dict:
    """Build pair loaders plus an image-level validation loader."""

    split_path, manifest_path = metadata_paths(config, val_images)

    if not split_path.is_file() or not manifest_path.is_file():
        raise DatasetError("Dataset metadata is missing")

    splits = json.loads(split_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest["split_fingerprint"] != json_fingerprint(splits):
        raise DatasetError("Manifest and split metadata identities differ")

    if len(manifest["train"]) != config["data"]["debug_images"]:
        raise DatasetError("Training image count differs from configuration")

    if len(manifest["validation"]) != val_images:
        raise DatasetError("Validation image count differs from request")

    train_ids = {record["image_id"] for record in manifest["train"]}
    val_ids = {record["image_id"] for record in manifest["validation"]}

    if train_ids & val_ids:
        raise DatasetError("Training and validation images overlap")

    if not train_ids <= set(splits["train"]):
        raise DatasetError("Training images are outside their assigned split")

    if not val_ids <= set(splits["validation"]):
        raise DatasetError("Validation images are outside their assigned split")

    tokenizer = load_caption_tokenizer(config, allow_download=False)

    raw_root = (
        Path(config["paths"]["data_root"])
        / "raw"
        / "coco2017"
    )

    image_size = config["data"]["image_size"]

    train_transform = build_image_transform(image_size, split="train")
    validation_transform = build_image_transform(
        image_size,
        split="validation",
    )

    train_dataset = CaptionPairDataset(
        manifest["train"],
        raw_root,
        tokenizer,
        train_transform,
    )

    validation_dataset = CaptionPairDataset(
        manifest["validation"],
        raw_root,
        tokenizer,
        validation_transform,
    )

    generation_dataset = CaptionImageDataset(
        manifest["validation"],
        raw_root,
        validation_transform,
    )

    collator = CaptionCollator(tokenizer)

    batch_size = config["training"]["batch_size"]
    num_workers = config["runtime"]["num_workers"]
    seed = config["project"]["seed"]

    return {
        "train": make_loader(
            train_dataset,
            batch_size,
            num_workers,
            shuffle=True,
            seed=seed,
            collate_fn=collator,
        ),
        "validation": make_loader(
            validation_dataset,
            batch_size,
            num_workers,
            shuffle=False,
            seed=seed + 1,
            collate_fn=collator,
        ),
        "generation_validation": make_loader(
            generation_dataset,
            batch_size,
            num_workers,
            shuffle=False,
            seed=seed + 2,
            collate_fn=collate_images,
        ),
        "tokenizer": tokenizer,
    }