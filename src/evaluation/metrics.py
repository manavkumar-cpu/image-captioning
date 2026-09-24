"""COCO-style PTB tokenization, corpus BLEU and CIDEr scoring."""

from functools import lru_cache
from importlib.metadata import version
import math
import shutil
import subprocess

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


METRIC_NAMES = (
    "BLEU_1",
    "BLEU_2",
    "BLEU_3",
    "BLEU_4",
    "CIDEr",
)


@lru_cache(maxsize=1)
def check_metric_environment():
    installed = version("pycocoevalcap")

    if installed != "1.2":
        raise RuntimeError(
            "Install requirements-eval.txt; expected pycocoevalcap==1.2."
        )

    java = shutil.which("java")

    if java is None:
        raise RuntimeError(
            "Java is required for the COCO PTB tokenizer. Run java -version."
        )

    result = subprocess.run(
        [java, "-version"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Java is not working: {result.stderr.strip()}"
        )

    details = (result.stderr + result.stdout).strip().splitlines()

    return {
        "pycocoevalcap": installed,
        "numpy": version("numpy"),
        "java": details[0] if details else "version unavailable",
    }


def ptb_tokenize(captions):
    """Input: image_id -> list[str]. Output preserves every caption and ID."""
    check_metric_environment()

    prepared = {
        image_id: [
            {
                "caption": text.replace("\r", " ").replace("\n", " ")
            }
            for text in captions[image_id]
        ]
        for image_id in sorted(captions)
    }

    tokenized = PTBTokenizer().tokenize(prepared)

    if set(tokenized) != set(prepared):
        raise RuntimeError(
            "PTB tokenizer lost image IDs; inspect Java output."
        )

    for image_id in prepared:
        if len(tokenized[image_id]) != len(prepared[image_id]):
            raise RuntimeError(
                "PTB tokenizer lost captions; inspect Java output."
            )

    return tokenized


def score_captions(references, predictions):
    """references: dict[int, list[str]]; predictions: dict[int, str]."""
    if not isinstance(references, dict) or not isinstance(predictions, dict):
        raise ValueError(
            "References and predictions must be dictionaries."
        )

    if len(references) < 2:
        raise ValueError(
            "Use at least two images; single-image CIDEr IDF is degenerate."
        )

    if set(references) != set(predictions):
        raise ValueError(
            "Prediction and reference image IDs must match exactly."
        )

    for image_id, captions in references.items():
        if type(image_id) is not int:
            raise ValueError("Image IDs must be integers.")

        if not isinstance(captions, list) or not captions:
            raise ValueError(
                "Every image needs a nonempty reference list."
            )

        if any(
            not isinstance(text, str) or not text.strip()
            for text in captions
        ):
            raise ValueError(
                "Reference captions must be nonempty strings."
            )

        if not isinstance(predictions[image_id], str):
            raise ValueError(
                "Each image needs one prediction string; empty strings are allowed."
            )

    gts = ptb_tokenize(references)

    res = ptb_tokenize({
        key: [predictions[key]]
        for key in sorted(predictions)
    })

    if any(
        not any(text.strip() for text in captions)
        for captions in gts.values()
    ):
        raise ValueError(
            "An image has no reference words after PTB tokenization."
        )

    bleu, image_bleu = Bleu(4).compute_score(
        gts,
        res,
        verbose=0,
    )

    cider, image_cider = Cider(
        n=4,
        sigma=6.0,
    ).compute_score(gts, res)

    scores = {
        f"BLEU_{i + 1}": float(value)
        for i, value in enumerate(bleu)
    }

    scores["CIDEr"] = float(cider)

    per_image = []

    for position, image_id in enumerate(gts):
        row = {"image_id": image_id}

        row.update({
            f"BLEU_{i + 1}": float(image_bleu[i][position])
            for i in range(4)
        })

        row["CIDEr"] = float(image_cider[position])
        per_image.append(row)

    for row in [scores, *per_image]:
        if any(
            not math.isfinite(row[name])
            for name in METRIC_NAMES
        ):
            raise RuntimeError(
                "A metric produced a non-finite score."
            )

        if any(
            not 0 <= row[name] <= 1.000001
            for name in METRIC_NAMES[:4]
        ):
            raise RuntimeError(
                "BLEU score outside its expected raw range."
            )

        if not 0 <= row["CIDEr"] <= 10.000001:
            raise RuntimeError(
                "CIDEr score outside this implementation's raw range."
            )

    return {
        "corpus": scores,
        "per_image": per_image,
        "tokenized": {
            "references": gts,
            "predictions": res,
        },
        "protocol": {
            "environment": dict(check_metric_environment()),
            "text_tokenization": (
                "COCO Stanford PTB, lowercase and punctuation filtering"
            ),
            "BLEU": (
                "Corpus BLEU orders 1-4; closest reference length; "
                "raw 0-1 scale"
            ),
            "CIDEr": (
                "pycocoevalcap Cider(n=4, sigma=6.0), clipping and "
                "length penalty; raw scale"
            ),
            "cider_document_frequency": (
                "All reference groups in this evaluation call"
            ),
            "additional_score_scaling": "none",
        },
    }