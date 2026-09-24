"""Validate caption tokenization on the existing limited dataset."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from src.data.prepare import metadata_paths
from src.data.tokenizer import (
    CaptionTokenizer,
    load_caption_tokenizer,
    tokenizer_directory,
)
from src.utils.config import load_config
from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger


LOGGER = get_logger("image_captioning.tokenizer_check")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


def check_sequence(tokenizer, ids):
    """Check an unpadded BOS/content/EOS sequence."""

    require(
        3 <= len(ids) <= tokenizer.max_length,
        "Encoded caption length is outside its allowed range",
    )

    require(
        ids[0] == tokenizer.bos_token_id,
        "Caption does not start with BOS",
    )

    require(
        ids[-1] == tokenizer.eos_token_id,
        "Caption does not end with EOS",
    )

    require(
        ids.count(tokenizer.bos_token_id) == 1,
        "Caption must contain exactly one BOS",
    )

    require(
        ids.count(tokenizer.eos_token_id) == 1,
        "Caption must contain exactly one EOS",
    )

    require(
        tokenizer.pad_token_id not in ids,
        "Individual captions should not be padded yet",
    )

    require(
        all(type(i) is int and 0 <= i < tokenizer.vocab_size for i in ids),
        "Caption contains an invalid vocabulary ID",
    )


def run_edge_checks(tokenizer):
    """Test boundary handling, truncation, and invalid input."""

    short_ids = tokenizer.encode("A dog runs.")
    check_sequence(tokenizer, short_ids)

    long_caption = "dog " * (tokenizer.max_length * 3)
    long_ids = tokenizer.encode(long_caption)

    check_sequence(tokenizer, long_ids)

    require(
        len(long_ids) == tokenizer.max_length,
        "Long caption was not truncated to the configured limit",
    )

    expected_text = tokenizer.decode(short_ids)

    # Tokens after EOS must not become part of the decoded caption.
    extra_ids = short_ids + tokenizer.content_ids("another sentence")

    require(
        tokenizer.decode(extra_ids) == expected_text,
        "Decoder did not stop at the first EOS",
    )

    for invalid_caption in ("", "   ", None):
        try:
            tokenizer.encode(invalid_caption)
        except DatasetError:
            pass
        else:
            raise DatasetError("Invalid caption was not rejected")

    LOGGER.info("Boundary tokens, truncation and invalid-input checks: PASS")


def check_split(tokenizer, reloaded, records, split_name):
    stats = {
        "images": len(records),
        "captions": 0,
        "truncated_captions": 0,
        "captions_with_unk": 0,
        "total_original_content_tokens": 0,
        "original_unk_tokens": 0,
        "max_original_length_including_boundaries": 0,
        "max_encoded_length": 0,
        "total_encoded_tokens": 0,
    }

    examples = []

    for record in records:
        for caption_id, caption in zip(
            record["caption_ids"],
            record["captions"],
        ):
            original_content = tokenizer.content_ids(caption)
            ids = tokenizer.encode(caption)

            check_sequence(tokenizer, ids)

            require(
                ids == reloaded.encode(caption),
                "Token IDs changed after saving and reloading",
            )

            require(
                ids[1:-1]
                == original_content[:tokenizer.max_length - 2],
                "Encoded content differs from the expected truncation",
            )

            original_length = len(original_content) + 2
            unknown_count = original_content.count(
                tokenizer.unk_token_id
            )

            stats["captions"] += 1
            stats["truncated_captions"] += int(
                original_length > tokenizer.max_length
            )
            stats["captions_with_unk"] += int(unknown_count > 0)
            stats["original_unk_tokens"] += unknown_count
            stats["total_original_content_tokens"] += len(original_content)
            stats["total_encoded_tokens"] += len(ids)

            stats["max_original_length_including_boundaries"] = max(
                stats["max_original_length_including_boundaries"],
                original_length,
            )

            stats["max_encoded_length"] = max(
                stats["max_encoded_length"],
                len(ids),
            )

            if len(examples) < 3:
                examples.append(
                    {
                        "image_id": record["image_id"],
                        "caption_id": caption_id,
                        "original_caption": caption,
                        "tokens": tokenizer.tokens(ids),
                        "token_ids": ids,
                        "decoded_caption": tokenizer.decode(ids),
                        "truncated": original_length > tokenizer.max_length,
                    }
                )

    require(stats["captions"] > 0, f"No captions in {split_name}")

    stats["truncation_percent"] = (
        100.0 * stats["truncated_captions"] / stats["captions"]
    )

    stats["unknown_content_token_percent"] = (
        100.0
        * stats["original_unk_tokens"]
        / stats["total_original_content_tokens"]
    )

    stats["mean_encoded_length"] = (
        stats["total_encoded_tokens"] / stats["captions"]
    )

    LOGGER.info(
        "%s: images=%d, captions=%d, truncated=%d (%.2f%%)",
        split_name,
        stats["images"],
        stats["captions"],
        stats["truncated_captions"],
        stats["truncation_percent"],
    )

    LOGGER.info(
        "%s: max encoded length=%d, mean encoded length=%.2f",
        split_name,
        stats["max_encoded_length"],
        stats["mean_encoded_length"],
    )

    LOGGER.info(
        "%s: captions containing UNK=%d",
        split_name,
        stats["captions_with_unk"],
    )

    return stats, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default="configs/baseline_debug.yaml",
    )
    parser.add_argument("--val-images", type=int, default=250)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the saved project tokenizer; do not download",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        _, manifest_path = metadata_paths(config, args.val_images)

        if not manifest_path.is_file():
            raise DatasetError(f"Manifest not found: {manifest_path}")

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        # Phase 2 validated these mappings. Check parallel lists before use.
        for split_name in ("train", "validation"):
            for record in manifest[split_name]:
                require(
                    len(record["captions"]) == len(record["caption_ids"]),
                    "Caption text and caption-ID counts differ",
                )

        tokenizer = load_caption_tokenizer(
            config,
            allow_download=not args.offline,
        )

        reloaded = CaptionTokenizer.load(
            tokenizer_directory(config)
        )

        LOGGER.info("Vocabulary size: %d", tokenizer.vocab_size)
        LOGGER.info(
            "BOS=%d, EOS=%d, PAD=%d, UNK=%s",
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        )
        LOGGER.info("Maximum full sequence length: %d", tokenizer.max_length)

        run_edge_checks(tokenizer)

        stats = {}
        examples = {}

        for split_name in ("train", "validation"):
            stats[split_name], examples[split_name] = check_split(
                tokenizer,
                reloaded,
                manifest[split_name],
                split_name,
            )

        sample = examples["train"][0]

        LOGGER.info("Sample caption: %s", sample["original_caption"])
        LOGGER.info("Sample tokens: %s", sample["tokens"])
        LOGGER.info("Sample IDs: %s", sample["token_ids"])
        LOGGER.info("Decoded sample: %s", sample["decoded_caption"])
        LOGGER.info("Offline reload and encoding consistency: PASS")

        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            + "_"
            + uuid4().hex[:8]
        )

        output_dir = (
            Path(config["paths"]["artifacts_root"])
            / "tokenizer"
            / run_id
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        report = {
            "status": "automated_checks_passed",
            "manifest": str(manifest_path),
            "split_fingerprint": manifest["split_fingerprint"],
            "tokenizer": tokenizer.metadata(),
            "statistics": stats,
            "examples": examples,
            "scope": (
                "Development caption tokenization only. "
                "No padding, next-token shifting, model or training."
            ),
        }

        report_path = output_dir / "report.json"

        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        LOGGER.info("Tokenizer report: %s", report_path)
        LOGGER.info("PHASE 4 TOKENIZER CHECK: PASS")

        return 0

    except Exception as exc:
        LOGGER.error("TOKENIZER CHECK FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())