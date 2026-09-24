"""Validate Person 1 artifacts and record a handoff snapshot."""

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys

from src.data.tokenizer import CaptionTokenizer
from src.evaluation.metrics import score_captions
from src.inference.predict import generated_content
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
from src.utils.logger import get_logger


LOGGER = get_logger("image_captioning.handoff")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def indexed(rows):
    require(isinstance(rows, list), "Expected a list of image records.")
    result = {}
    for row in rows:
        key = row["image_id"]
        require(
            type(key) is int and key not in result,
            "Invalid or duplicate image ID.",
        )
        result[key] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run = args.run_dir.expanduser().resolve()
    evaluation = args.evaluation_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()

    require(not output.exists(), "Choose a new handoff output directory.")
    require(
        not output.is_relative_to(run)
        and not output.is_relative_to(evaluation),
        "Keep the handoff snapshot outside the training and evaluation directories.",
    )

    config = load_config(run / "config.json")
    context = read_json(run / "context.json")
    training = read_json(run / "training_validation.json")
    summary = read_json(run / "run_summary.json")
    metrics = read_json(evaluation / "metrics.json")
    inference = read_json(run / "inference_validation.json")

    require(
        training["status"] == metrics["status"] == inference["status"] == "passed",
        "Phases 9, 10 and 11 must have passing reports.",
    )
    require(summary["status"] == "training_finished", "Training is incomplete.")
    require(
        config["data"]["debug_images"] == 1000 and context["val_images"] == 250,
        "Expected the agreed 1000/250 development subset.",
    )

    loaders, tokenizer, current = prepare_data(config, val_images=250)
    require(
        current == context,
        "Training source, data or tokenizer fingerprints changed.",
    )
    snapshot = CaptionTokenizer.load(run / "tokenizer")
    require(
        tokenizer_fingerprint(snapshot)
        == tokenizer_fingerprint(tokenizer)
        == context["tokenizer_sha256"],
        "Saved tokenizer snapshot differs.",
    )
    LOGGER.info(
        "Current source, selected images, data and tokenizer fingerprints: PASS"
    )

    best_path = resolve_checkpoint(run, "best")
    checksum = file_sha256(best_path)
    best = read_checkpoint(best_path)

    require(
        best["contract"]["context"] == context
        and best["contract"]["config"] == json_value(config),
        "Checkpoint contract differs.",
    )
    epoch = best["trainer"]["epoch"]
    result = inference["result"]

    require(
        epoch
        == training["best_epoch"]
        == summary["best_epoch"]
        == metrics["best_epoch"]
        == result["best_epoch"],
        "Best checkpoint epochs differ.",
    )
    require(
        checksum == metrics["checkpoint_sha256"] == result["checkpoint_sha256"],
        "Evaluation or inference used a different checkpoint.",
    )
    require(
        training["counts"] == context["counts"],
        "Training report counts differ.",
    )
    require(
        training["completed_epochs"] == summary["completed_epochs"]
        and training["global_step"] == summary["global_step"],
        "Training totals differ.",
    )
    del best
    gc.collect()
    LOGGER.info("Training, evaluation and inference checkpoint identity: PASS")

    predictions = indexed(read_json(evaluation / "predictions.json"))
    details = indexed(read_json(evaluation / "per_image.json"))
    references = {
        row["image_id"]: row["captions"]
        for row in loaders["validation"].dataset.records
    }

    require(
        set(predictions) == set(details) == set(references),
        "Evaluation IDs differ from the selected validation images.",
    )
    require(
        len(references) == metrics["image_count"] == 250,
        "Evaluation image count differs.",
    )
    require(
        sum(map(len, references.values())) == metrics["references_count"],
        "Evaluation reference count differs.",
    )
    require(
        metrics["selected_manifest_sha256"] == context["manifest_sha256"]
        and metrics["selected_images_sha256"]
        == context["selected_image_files_sha256"]
        and metrics["tokenizer_sha256"] == context["tokenizer_sha256"],
        "Evaluation data/tokenizer identity differs.",
    )

    limit = metrics["decoding"]["max_new_tokens"]
    require(
        metrics["decoding"]["strategy"] == "greedy"
        and type(limit) is int
        and 1 <= limit < config["data"]["max_caption_length"],
        "Unexpected baseline decoding settings.",
    )

    for image_id, row in details.items():
        tokens, ended = generated_content(row["token_ids"], snapshot, limit)
        require(
            row["references"] == references[image_id],
            "Reference captions changed.",
        )
        require(
            row["prediction"]
            == predictions[image_id]["caption"]
            == snapshot.decode(tokens).strip(),
            "Saved prediction/token mismatch.",
        )
        require(
            row["ended_with_eos"] == ended,
            "Saved EOS flag differs.",
        )

    LOGGER.info(
        "All 250 prediction IDs, original references and generated tokens: PASS"
    )

    rescored = score_captions(
        references,
        {key: row["caption"] for key, row in predictions.items()},
    )
    require(
        rescored["protocol"] == metrics["scoring_protocol"],
        "Scoring protocol changed.",
    )
    require(
        set(rescored["corpus"]) == set(metrics["scores"]),
        "Metric names differ.",
    )

    for name, value in rescored["corpus"].items():
        saved = metrics["scores"][name]
        require(
            math.isfinite(saved)
            and math.isclose(value, saved, rel_tol=1e-9, abs_tol=1e-9),
            f"Saved {name} differs from recomputed score.",
        )
        LOGGER.info("%s: %.6f", name, value)

    for row in rescored["per_image"]:
        saved_scores = details[row["image_id"]]["scores"]
        for name in rescored["corpus"]:
            require(
                math.isclose(
                    row[name],
                    saved_scores[name],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ),
                f"Per-image {name} differs for image {row['image_id']}.",
            )

    expected_diagnostics = {
        "empty_captions": sum(
            not row["caption"] for row in predictions.values()
        ),
        "distinct_captions": len(
            {row["caption"] for row in predictions.values()}
        ),
        "ended_with_eos": sum(
            row["ended_with_eos"] for row in details.values()
        ),
        "hit_token_limit": sum(
            not row["ended_with_eos"] for row in details.values()
        ),
    }
    require(
        metrics["diagnostics"] == expected_diagnostics,
        "Evaluation diagnostics differ.",
    )
    require(
        read_json(evaluation / "tokenized.json")
        == json_value(rescored["tokenized"]),
        "Saved PTB-tokenized captions differ.",
    )
    require(
        metrics["official_coco_test_server_result"] is False,
        "This is a custom validation subset, not an official test-server result.",
    )
    LOGGER.info("Recomputed BLEU/CIDEr and saved tokenized captions: PASS")

    require(
        type(result["max_new_tokens"]) is int
        and 1 <= result["max_new_tokens"] < config["data"]["max_caption_length"],
        "Invalid inference token budget.",
    )
    tokens, ended = generated_content(
        result["token_ids"],
        snapshot,
        result["max_new_tokens"],
    )
    require(
        snapshot.decode(tokens).strip() == result["caption"]
        and ended == result["ended_with_eos"],
        "Inference result fields disagree.",
    )
    require(
        {
            "repeatability",
            "eval_mode",
            "no_gradients",
            "missing_image",
            "token_limit",
        }
        <= set(inference["checks"]),
        "Inference report lacks required checks.",
    )

    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    paths = [
        run / name
        for name in (
            "config.json",
            "context.json",
            "run_summary.json",
            "training_validation.json",
            "inference_validation.json",
            "checkpoints/index.json",
        )
    ]
    paths += [
        evaluation / name
        for name in (
            "metrics.json",
            "predictions.json",
            "per_image.json",
            "tokenized.json",
        )
    ]
    paths += [
        Path(__file__).resolve().parents[1] / "docs" / "PERSON1_HANDOFF.md"
    ]
    hashes = {str(path): file_sha256(path) for path in paths}

    report = {
        "status": "passed",
        "scope": "person1_artifact_consistency_and_metric_recheck",
        "run_dir": str(run),
        "evaluation_dir": str(evaluation),
        "best_epoch": epoch,
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": checksum,
        "counts": context["counts"],
        "scores": rescored["corpus"],
        "inference_caption": result["caption"],
        "artifact_sha256": hashes,
        "python": sys.version,
        "test_set_evaluated": False,
        "handoff_acknowledged": False,
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "environment.freeze.txt").write_text(frozen, encoding="utf-8")
    write_json(output / "report.json", report)

    LOGGER.info("Report: %s", output / "report.json")
    LOGGER.info("PHASE 12 PERSON 1 HANDOFF CHECK: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("PHASE 12 PERSON 1 HANDOFF CHECK: FAIL")
        raise SystemExit(1)