"""Check standalone inference using one previously downloaded COCO image."""

import argparse
from pathlib import Path

from src.inference.predict import BaselineCaptioner, require
from src.training.checkpoints import write_json
from src.utils.logger import get_logger


LOGGER = get_logger("image_captioning.inference_check")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True,
                        help="Any locally available image, including a COCO validation image")
    args = parser.parse_args()
    captioner = BaselineCaptioner(args.run_dir)
    image = args.image.expanduser().resolve()
    first = captioner.predict(image)
    second = captioner.predict(image)
    require(first == second, "Repeated inference on one image changed the output.")
    require(not captioner.model.training, "Inference unexpectedly switched the model to training.")
    require(all(parameter.grad is None for parameter in captioner.model.parameters()),
            "Inference produced parameter gradients.")
    require(bool(first["caption"]), "Generated caption is empty.")
    require(first["token_ids"] and first["best_epoch"] >= 1,
            "Result lacks generated tokens or a best checkpoint epoch.")
    require(first["checkpoint_sha256"] == captioner.checkpoint_sha256,
            "Result checkpoint identity changed.")
    try:
        captioner.predict(image.parent / "__phase11_missing_image__.jpg")
    except RuntimeError as error:
        require("does not exist" in str(error), "Missing-image error is unclear.")
    else:
        raise RuntimeError("Missing image was accepted.")
    try:
        captioner.predict(image, max_new_tokens=captioner.config["data"]["max_caption_length"])
    except RuntimeError as error:
        require("max_new_tokens" in str(error), "Token-limit error is unclear.")
    else:
        raise RuntimeError("Out-of-range token limit was accepted.")
    LOGGER.info("Verified best checkpoint, repeated output, eval mode and no gradients: PASS")
    LOGGER.info("Missing image and invalid token limit rejected: PASS")
    LOGGER.info("Device: %s | best epoch: %d | caption: %s",
                captioner.device, first["best_epoch"], first["caption"])
    report = {"status": "passed", "checks": ["repeatability", "eval_mode", "no_gradients",
                                            "nonempty_caption", "missing_image", "token_limit"],
              "result": first}
    report_path = args.run_dir.expanduser().resolve() / "inference_validation.json"
    write_json(report_path, report)
    LOGGER.info("Report: %s", report_path)
    LOGGER.info("PHASE 11 INFERENCE CHECK: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("PHASE 11 INFERENCE CHECK: FAIL")
        raise SystemExit(1)