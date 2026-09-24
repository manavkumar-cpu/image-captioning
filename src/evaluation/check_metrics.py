"""Test the metric pipeline without loading a model or COCO images."""

import math

from src.evaluation.metrics import (
    METRIC_NAMES,
    check_metric_environment,
    ptb_tokenize,
    score_captions,
)
from src.utils.logger import get_logger


LOGGER = get_logger("image_captioning.metric_check")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    LOGGER.info(
        "Metric environment: %s",
        check_metric_environment(),
    )

    tokenized = ptb_tokenize({
        1: ["A CAT, on the MAT!"]
    })

    require(
        tokenized == {
            1: ["a cat on the mat"]
        },
        "PTB normalization differs.",
    )

    LOGGER.info(
        "Java and COCO PTB tokenization: PASS"
    )

    captions = {
        11: "a red bus drives along the street",
        22: "two dogs play beside a wooden fence",
        33: "a small boat floats across blue water",
    }

    references = {
        key: [text, text]
        for key, text in captions.items()
    }

    perfect = score_captions(
        references,
        captions,
    )

    for name in METRIC_NAMES[:4]:
        require(
            math.isclose(
                perfect["corpus"][name],
                1.0,
                abs_tol=1e-7,
            ),
            f"Perfect-match {name} is not near 1.",
        )

    require(
        math.isclose(
            perfect["corpus"]["CIDEr"],
            10.0,
            abs_tol=1e-7,
        ),
        "Perfect-match fixture CIDEr is not near 10.",
    )

    wrong = {
        11: captions[22],
        22: captions[33],
        33: captions[11],
    }

    mismatched = score_captions(
        references,
        wrong,
    )

    require(
        all(
            perfect["corpus"][name]
            > mismatched["corpus"][name]
            for name in METRIC_NAMES
        ),
        "Mismatched captions did not score lower.",
    )

    LOGGER.info(
        "Perfect matches versus mismatched captions: PASS"
    )

    reordered = score_captions(
        dict(reversed(list(references.items()))),
        dict(reversed(list(captions.items()))),
    )

    require(
        reordered["corpus"] == perfect["corpus"],
        "Scores depend on dictionary order.",
    )

    require(
        reordered["per_image"] == perfect["per_image"],
        "Per-image alignment changed.",
    )

    LOGGER.info(
        "Image-ID alignment and repeatability: PASS"
    )

    empty = score_captions(
        references,
        {
            key: ""
            for key in captions
        },
    )

    require(
        all(
            abs(value) < 1e-12
            for value in empty["corpus"].values()
        ),
        "Empty predictions must produce zero scores.",
    )

    LOGGER.info(
        "Empty predictions retained and scored: PASS"
    )

    for bad_references, bad_predictions in [
        (
            references,
            {11: captions[11]},
        ),
        (
            {11: references[11]},
            {11: captions[11]},
        ),
        (
            {**references, 11: []},
            captions,
        ),
    ]:
        rejected = False

        try:
            score_captions(
                bad_references,
                bad_predictions,
            )
        except ValueError:
            rejected = True

        require(
            rejected,
            "Invalid metric inputs were accepted.",
        )

    LOGGER.info(
        "Missing IDs, missing references and single-image CIDEr rejection: PASS"
    )

    LOGGER.info(
        "PHASE 10 METRIC CHECK: PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception(
            "PHASE 10 METRIC CHECK: FAIL"
        )
        raise SystemExit(1)