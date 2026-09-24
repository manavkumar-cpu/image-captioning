"""Generate one caption per selected validation image and score BLEU/CIDEr."""

import argparse
import gc
import json
from pathlib import Path

import torch

from src.data.collate import collate_images
from src.data.dataloader import make_loader
from src.data.dataset import CaptionImageDataset
from src.data.tokenizer import CaptionTokenizer
from src.evaluation.metrics import (
    check_metric_environment,
    score_captions,
)
from src.models.factory import build_model
from src.training.checkpoints import (
    json_value,
    read_checkpoint,
    resolve_checkpoint,
    write_json,
)
from src.training.train_baseline import (
    file_sha256,
    prepare_data,
    require,
    tokenizer_fingerprint,
)
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


LOGGER = get_logger("image_captioning.baseline_evaluation")


def verify_generated_sequence(ids, tokenizer, max_new_tokens):
    if (
        not isinstance(ids, list)
        or not 1 <= len(ids) <= max_new_tokens
    ):
        raise RuntimeError(
            "Generated sequence has an invalid length."
        )

    if any(
        type(token) is not int
        or not 0 <= token < tokenizer.vocab_size
        for token in ids
    ):
        raise RuntimeError(
            "Generated token ID is outside the tokenizer vocabulary."
        )

    if tokenizer.bos_token_id in ids:
        raise RuntimeError(
            "Generation unexpectedly includes BOS."
        )

    if tokenizer.eos_token_id in ids:
        end = ids.index(tokenizer.eos_token_id)

        if any(
            token != tokenizer.pad_token_id
            for token in ids[end + 1:]
        ):
            raise RuntimeError(
                "EOS must be followed only by padding."
            )

        content = ids[:end + 1]
        ended_with_eos = True

    else:
        if (
            len(ids) != max_new_tokens
            or tokenizer.pad_token_id in ids
        ):
            raise RuntimeError(
                "Unfinished generation must reach the token limit without PAD."
            )

        content = ids
        ended_with_eos = False

    if tokenizer.pad_token_id in content:
        raise RuntimeError(
            "PAD appeared before EOS."
        )

    return content, ended_with_eos


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
    )

    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    require(
        not output_dir.exists(),
        "Output directory already exists; choose a new name.",
    )

    require(
        not output_dir.is_relative_to(run_dir / "checkpoints"),
        "Store evaluation outside the checkpoint directory.",
    )

    check_metric_environment()

    config = load_config(run_dir / "config.json")

    context = json.loads(
        (run_dir / "context.json").read_text()
    )

    trained = json.loads(
        (run_dir / "training_validation.json").read_text()
    )

    summary = json.loads(
        (run_dir / "run_summary.json").read_text()
    )

    require(
        trained["status"] == "passed"
        and summary["status"] == "training_finished",
        "First complete Phase 9 training and its saved-run check.",
    )

    require(
        context["kind"] == "coco2017_limited_baseline",
        "Not a baseline training run.",
    )

    require(
        context["val_images"] == 250
        and config["data"]["debug_images"] == 1000,
        "This baseline comparison requires the agreed 1000/250 subset.",
    )

    require(
        trained["best_epoch"] == summary["best_epoch"],
        "Best checkpoint identity differs.",
    )

    seed_everything(config["project"]["seed"])

    loaders, tokenizer, current_context = prepare_data(
        config,
        context["val_images"],
    )

    require(
        context == current_context,
        "Data, tokenizer, preprocessing or training code changed.",
    )

    snapshot = CaptionTokenizer.load(
        run_dir / "tokenizer"
    )

    require(
        tokenizer_fingerprint(snapshot)
        == tokenizer_fingerprint(tokenizer),
        "Saved tokenizer snapshot changed.",
    )

    max_tokens = (
        config["data"]["max_caption_length"] - 1
        if args.max_new_tokens is None
        else args.max_new_tokens
    )

    require(
        1 <= max_tokens
        <= config["data"]["max_caption_length"] - 1,
        "max_new_tokens must be between 1 and max_caption_length-1.",
    )

    device = select_device(
        config["runtime"]["device"]
    )

    checkpoint_path = resolve_checkpoint(
        run_dir,
        "best",
    )

    checksum = file_sha256(checkpoint_path)
    checkpoint = read_checkpoint(checkpoint_path)

    require(
        checkpoint["contract"]["context"] == context
        and checkpoint["contract"]["config"] == json_value(config)
        and checkpoint["contract"]["device"] == str(device),
        "Best checkpoint does not match the verified training run/device.",
    )

    require(
        checkpoint["trainer"]["epoch"]
        == trained["best_epoch"],
        "Best checkpoint epoch differs from Phase 9 report.",
    )

    model = build_model(
        config,
        snapshot,
        load_pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    del checkpoint
    gc.collect()

    model.to(device).eval()

    paired_validation = loaders["validation"].dataset

    image_dataset = CaptionImageDataset(
        paired_validation.records,
        paired_validation.raw_root,
        paired_validation.transform,
    )

    image_loader = make_loader(
        image_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=0,
        shuffle=False,
        seed=config["project"]["seed"] + 2,
        collate_fn=collate_images,
    )

    references = {
        record["image_id"]: record["captions"]
        for record in paired_validation.records
    }

    require(
        len(references)
        == len(image_dataset)
        == context["counts"]["validation"]["images"],
        "Image-level validation is not one row per image.",
    )

    predictions = {}
    generation = {}

    with torch.no_grad():
        for batch_index, batch in enumerate(
            image_loader,
            start=1,
        ):
            ids = batch["image_ids"]

            output = model.generate(
                batch["images"].to(device),
                max_new_tokens=max_tokens,
                strategy="greedy",
                beam_size=1,
            )

            sequences = output["sequences"]

            require(
                sequences.ndim == 2
                and sequences.size(0) == len(ids),
                "Generation returned one sequence per input image.",
            )

            for row, image_id in enumerate(ids):
                image_id = int(image_id)

                require(
                    image_id in references
                    and image_id not in predictions,
                    "Unexpected or duplicate validation image ID.",
                )

                require(
                    batch["references"][row]
                    == references[image_id],
                    "Generation loader references differ from original captions.",
                )

                tokens, ended = verify_generated_sequence(
                    sequences[row].cpu().tolist(),
                    snapshot,
                    max_tokens,
                )

                caption = snapshot.decode(
                    tokens
                ).strip()

                predictions[image_id] = caption

                generation[image_id] = {
                    "token_ids": tokens,
                    "ended_with_eos": ended,
                }

            if (
                batch_index % 8 == 0
                or batch_index == len(image_loader)
            ):
                LOGGER.info(
                    "Generated captions: %d/%d images",
                    len(predictions),
                    len(references),
                )

    require(
        set(predictions) == set(references),
        "Generation omitted validation images.",
    )

    scored = score_captions(
        references,
        predictions,
    )

    by_image = {
        row["image_id"]: row
        for row in scored["per_image"]
    }

    require(
        set(by_image) == set(references),
        "Per-image scores do not cover every image.",
    )

    ordered_ids = sorted(references)

    prediction_rows = [
        {
            "image_id": image_id,
            "caption": predictions[image_id],
        }
        for image_id in ordered_ids
    ]

    details = [
        {
            "image_id": image_id,
            "prediction": predictions[image_id],
            "references": references[image_id],
            **generation[image_id],
            "scores": {
                key: value
                for key, value
                in by_image[image_id].items()
                if key != "image_id"
            },
        }
        for image_id in ordered_ids
    ]

    diagnostics = {
        "empty_captions": sum(
            not text
            for text in predictions.values()
        ),
        "distinct_captions": len(
            set(predictions.values())
        ),
        "ended_with_eos": sum(
            row["ended_with_eos"]
            for row in generation.values()
        ),
        "hit_token_limit": sum(
            not row["ended_with_eos"]
            for row in generation.values()
        ),
    }

    metrics = {
        "status": "passed",
        "model": config["model"]["name"],
        "split": "custom_project_validation_250",
        "image_count": len(predictions),
        "references_count": sum(
            len(refs)
            for refs in references.values()
        ),
        "checkpoint_sha256": checksum,
        "best_epoch": trained["best_epoch"],
        "selected_manifest_sha256": (
            context["manifest_sha256"]
        ),
        "selected_images_sha256": (
            context["selected_image_files_sha256"]
        ),
        "tokenizer_sha256": context["tokenizer_sha256"],
        "decoding": {
            "strategy": "greedy",
            "max_new_tokens": max_tokens,
        },
        "scores": scored["corpus"],
        "diagnostics": diagnostics,
        "scoring_protocol": scored["protocol"],
        "official_coco_test_server_result": False,
        "note": (
            "Custom 250-image validation subset; "
            "use the same IDs/references for comparisons."
        ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_json(
        output_dir / "predictions.json",
        prediction_rows,
    )

    write_json(
        output_dir / "per_image.json",
        details,
    )

    write_json(
        output_dir / "tokenized.json",
        scored["tokenized"],
    )

    write_json(
        output_dir / "metrics.json",
        metrics,
    )

    for name, value in metrics["scores"].items():
        LOGGER.info(
            "%s: %.6f",
            name,
            value,
        )

    LOGGER.info(
        "Diagnostic counts: %s",
        diagnostics,
    )

    LOGGER.info(
        "Evaluation report: %s",
        output_dir / "metrics.json",
    )

    LOGGER.info(
        "PHASE 10 BASELINE BLEU/CIDEr EVALUATION: PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception(
            "PHASE 10 BASELINE BLEU/CIDEr EVALUATION: FAIL"
        )
        raise SystemExit(1)