"""Check batching, shifting, padding, references, and device transfer."""

import argparse
from datetime import datetime, timezone
from itertools import islice
import json
from pathlib import Path
from uuid import uuid4

import torch
from torch.utils.data import DataLoader, Subset

from src.data.collate import CaptionCollator, IGNORE_INDEX
from src.data.dataloader import build_dataloaders
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger


LOGGER = get_logger("image_captioning.dataloader_check")


def require(condition, message):
    if not condition:
        raise DatasetError(message)


def check_batch(batch, tokenizer, image_size):
    required = {
        "images",
        "input_ids",
        "attention_mask",
        "target_ids",
        "image_ids",
    }

    require(set(batch) == required, "Unexpected batch fields")

    images = batch["images"]
    inputs = batch["input_ids"]
    targets = batch["target_ids"]
    mask = batch["attention_mask"]

    require(inputs.ndim == 2, "Input IDs must have shape [B, T]")

    batch_size, sequence_length = inputs.shape

    require(
        tuple(images.shape) == (batch_size, 3, image_size, image_size),
        "Incorrect image batch shape",
    )
    require(images.dtype == torch.float32, "Images must be float32")
    require(torch.isfinite(images).all().item(), "Nonfinite image values")
    require(inputs.dtype == torch.long, "Input IDs must be int64")
    require(targets.dtype == torch.long, "Targets must be int64")
    require(mask.dtype == torch.bool, "Attention mask must be boolean")

    require(
        inputs.shape == targets.shape == mask.shape,
        "Caption tensor shapes do not match",
    )
    require(
        2 <= sequence_length <= tokenizer.max_length - 1,
        "Incorrect shifted sequence length",
    )
    require(
        len(batch["image_ids"]) == batch_size,
        "Image-ID count does not match batch size",
    )
    require(
        torch.equal(mask, inputs.ne(tokenizer.pad_token_id)),
        "Attention mask does not match input padding",
    )
    require(
        torch.equal(targets.eq(IGNORE_INDEX), ~mask),
        "Ignored targets do not match padded positions",
    )
    require(
        ((inputs >= 0) & (inputs < tokenizer.vocab_size)).all().item(),
        "Invalid input token ID",
    )

    lengths = mask.sum(dim=1)

    expected_mask = (
        torch.arange(sequence_length).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    require(torch.equal(mask, expected_mask), "Padding must be on the right")

    for row in range(batch_size):
        length = int(lengths[row].item())
        valid_inputs = inputs[row, :length]
        valid_targets = targets[row, :length]

        require(length >= 2, "Caption is too short")
        require(
            valid_inputs[0].item() == tokenizer.bos_token_id,
            "First input must be BOS",
        )
        require(
            valid_targets[-1].item() == tokenizer.eos_token_id,
            "Final valid target must be EOS",
        )
        require(
            torch.equal(valid_inputs[1:], valid_targets[:-1]),
            "Next-token alignment is incorrect",
        )
        require(
            (valid_inputs == tokenizer.eos_token_id).sum().item() == 0,
            "EOS should not appear in shifted inputs",
        )
        require(
            (valid_targets == tokenizer.bos_token_id).sum().item() == 0,
            "BOS should not appear in targets",
        )
        require(
            (valid_targets == tokenizer.eos_token_id).sum().item() == 1,
            "Expected exactly one target EOS",
        )
        require(
            (
                (valid_targets >= 0)
                & (valid_targets < tokenizer.vocab_size)
            ).all().item(),
            "Invalid non-padding target",
        )


def check_controlled_padding(tokenizer, image_size):
    """Test unequal lengths against the original complete sequences."""

    sequences = [
        tokenizer.encode("A cat."),
        tokenizer.encode("A small cat is sleeping on a comfortable bed."),
    ]

    require(
        len(sequences[0]) != len(sequences[1]),
        "Controlled examples must have different lengths",
    )

    samples = [
        {
            "image": torch.zeros(3, image_size, image_size),
            "token_ids": torch.tensor(ids, dtype=torch.long),
            "image_id": index,
        }
        for index, ids in enumerate(sequences)
    ]

    collator = CaptionCollator(tokenizer)
    batch = collator(samples)

    check_batch(batch, tokenizer, image_size)

    for row, sequence in enumerate(sequences):
        length = len(sequence) - 1

        require(
            batch["input_ids"][row, :length].tolist() == sequence[:-1],
            "Inputs differ from the original sequence prefix",
        )
        require(
            batch["target_ids"][row, :length].tolist() == sequence[1:],
            "Targets differ from the original sequence suffix",
        )

    require(
        (~batch["attention_mask"]).any().item(),
        "Controlled test did not exercise padding",
    )

    # Also exercise batch size one.
    check_batch(collator(samples[:1]), tokenizer, image_size)

    LOGGER.info("Controlled shift, unequal-length padding and batch size 1: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default="configs/baseline_debug.yaml",
    )
    parser.add_argument("--val-images", type=int, default=250)
    parser.add_argument("--batches", type=int, default=2)

    args = parser.parse_args()

    try:
        require(1 <= args.batches <= 10, "batches must be between 1 and 10")

        config = load_config(args.config)
        loaders = build_dataloaders(config, args.val_images)

        tokenizer = loaders["tokenizer"]
        image_size = config["data"]["image_size"]

        check_controlled_padding(tokenizer, image_size)

        counts = {}

        for name in ("train", "validation", "generation_validation"):
            loader = loaders[name]

            counts[name] = {
                "samples": len(loader.dataset),
                "batches_per_epoch": len(loader),
            }

            LOGGER.info(
                "%s: samples=%d, batches=%d",
                name,
                len(loader.dataset),
                len(loader),
            )

        # Check pair counts against the source reference lists.
        for name in ("train", "validation"):
            dataset = loaders[name].dataset
            expected = sum(len(r["captions"]) for r in dataset.records)

            require(
                len(dataset) == expected,
                f"{name} does not include every caption pair",
            )

        for name in ("train", "validation"):
            for index, batch in enumerate(
                islice(loaders[name], args.batches),
                start=1,
            ):
                check_batch(batch, tokenizer, image_size)

                LOGGER.info(
                    "%s batch %d: images=%s, tokens=%s",
                    name,
                    index,
                    tuple(batch["images"].shape),
                    tuple(batch["input_ids"].shape),
                )

        # Check deterministic dataset-item ordering against actual images.
        train_dataset = loaders["train"].dataset
        original_samples = [train_dataset[0], train_dataset[1]]
        direct_batch = loaders["train"].collate_fn(original_samples)

        for row, sample in enumerate(original_samples):
            require(
                torch.equal(direct_batch["images"][row], sample["image"]),
                "Image order changed during collation",
            )
            require(
                direct_batch["image_ids"][row] == sample["image_id"],
                "Image ID changed during collation",
            )

        # Exercise the last validation batch without reading the entire epoch.
        validation = loaders["validation"]
        count = len(validation.dataset)
        remainder = count % validation.batch_size or validation.batch_size

        tail = Subset(
            validation.dataset,
            range(count - remainder, count),
        )

        tail_loader = DataLoader(
            tail,
            batch_size=validation.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=validation.collate_fn,
        )

        tail_batch = next(iter(tail_loader))
        check_batch(tail_batch, tokenizer, image_size)

        require(
            len(tail_batch["image_ids"]) == remainder,
            "Final partial batch has the wrong size",
        )

        LOGGER.info("Final validation batch size: %d — PASS", remainder)

        # Image-level references must remain grouped by the correct image.
        generation = loaders["generation_validation"]
        generation_batch = next(iter(generation))

        source_references = {
            record["image_id"]: record["captions"]
            for record in generation.dataset.records
        }

        require(
            len(source_references) == len(generation.dataset),
            "Generation dataset contains duplicate images",
        )

        require(
            len(generation_batch["image_ids"])
            == len(generation_batch["references"]),
            "Generation references and image IDs differ in length",
        )

        for image_id, references in zip(
            generation_batch["image_ids"],
            generation_batch["references"],
        ):
            require(
                references == source_references[image_id],
                "Generation references are mismatched",
            )

        LOGGER.info("Image-level reference grouping: PASS")

        # Restore generator state and check that the first batch is reproduced.
        train_loader = loaders["train"]
        generator_state = train_loader.generator.get_state().clone()

        first = next(iter(train_loader))
        train_loader.generator.set_state(generator_state)
        repeated = next(iter(train_loader))

        require(
            first["image_ids"] == repeated["image_ids"],
            "Restoring generator state changed sampled image IDs",
        )

        for key in ("images", "input_ids", "attention_mask", "target_ids"):
            require(
                torch.equal(first[key], repeated[key]),
                f"Restoring generator state changed {key}",
            )

        LOGGER.info("Loader generator-state reproducibility: PASS")

        device = select_device(config["runtime"]["device"])

        moved = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in first.items()
        }

        for key in ("images", "input_ids", "attention_mask", "target_ids"):
            require(
                moved[key].device.type == device.type,
                f"{key} was not transferred to the selected device",
            )

        require(
            torch.isfinite(moved["images"]).all().item(),
            "Nonfinite image values after device transfer",
        )

        LOGGER.info("Batch transfer to %s: PASS", device)

        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            + "_"
            + uuid4().hex[:8]
        )

        output_dir = (
            Path(config["paths"]["artifacts_root"])
            / "dataloader"
            / run_id
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        report = {
            "status": "passed",
            "counts": counts,
            "batch_size": config["training"]["batch_size"],
            "num_workers": config["runtime"]["num_workers"],
            "device_transfer": str(device),
            "tokenizer_fingerprint": tokenizer.fingerprint(),
            "target_ignore_index": IGNORE_INDEX,
            "batch_fields": sorted(first),
            "scope": (
                "Controlled collation tests, sampled real batches, "
                "validation tail batch, reference grouping and device transfer. "
                "No full epoch or model training."
            ),
        }

        report_path = output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        LOGGER.info("Report: %s", report_path)
        LOGGER.info("PHASE 5 DATALOADER CHECK: PASS")
        return 0

    except Exception as exc:
        LOGGER.error("DATALOADER CHECK FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())