"""Caption a local image with the verified Phase 9 baseline checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.data.preprocessing import build_image_transform, load_image
from src.data.tokenizer import CaptionTokenizer
from src.models.factory import build_model
from src.training.checkpoints import json_value, read_checkpoint, resolve_checkpoint, write_json
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.logger import get_logger


LOGGER = get_logger("image_captioning.inference")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generated_content(ids, tokenizer, limit):
    require(isinstance(ids, list) and 1 <= len(ids) <= limit,
            "Generated sequence has an invalid length.")
    require(all(type(token) is int and 0 <= token < tokenizer.vocab_size for token in ids),
            "Generated token is outside the tokenizer vocabulary.")
    require(tokenizer.bos_token_id not in ids, "Generated sequence contains BOS.")
    if tokenizer.eos_token_id in ids:
        end = ids.index(tokenizer.eos_token_id)
        require(all(token == tokenizer.pad_token_id for token in ids[end + 1:]),
                "Generated sequence has a non-padding token after EOS.")
        content, ended = ids[:end + 1], True
    else:
        require(len(ids) == limit, "Unfinished generation stopped before the token limit.")
        content, ended = ids, False
    require(tokenizer.pad_token_id not in content, "Generated sequence has padding before EOS.")
    return content, ended


class BaselineCaptioner:
    """Reusable single-image greedy predictor; COCO annotations are not required."""

    def __init__(self, run_dir, device="auto"):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.config = load_config(self.run_dir / "config.json")
        context = json.loads((self.run_dir / "context.json").read_text(encoding="utf-8"))
        summary = json.loads((self.run_dir / "run_summary.json").read_text(encoding="utf-8"))
        report = json.loads((self.run_dir / "training_validation.json").read_text(encoding="utf-8"))
        require(context["kind"] == "coco2017_limited_baseline", "Not a baseline run.")
        require(summary["status"] == "training_finished" and report["status"] == "passed",
                "First finish training and the Phase 9 saved-run check.")
        require(summary["best_epoch"] == report["best_epoch"],
                "Best epoch differs between run summary and validation report.")

        self.tokenizer = CaptionTokenizer.load(self.run_dir / "tokenizer")
        fingerprint = self.tokenizer.fingerprint
        fingerprint = fingerprint() if callable(fingerprint) else fingerprint
        require(fingerprint == context["tokenizer_sha256"], "Saved tokenizer differs from the run.")
        source_root = Path(__file__).resolve().parents[1]
        for relative in ("data/preprocessing.py", "data/tokenizer.py",
                         "models/base_model.py", "models/cnn_lstm.py", "models/factory.py"):
            require(sha256_file(source_root / relative) == context["source_sha256"][relative],
                    f"Model input or generation code changed since training: {relative}")

        self.device = select_device(device)
        checkpoint_path = resolve_checkpoint(self.run_dir, "best")
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        payload = read_checkpoint(checkpoint_path)
        require(payload["contract"]["context"] == context
                and payload["contract"]["config"] == json_value(self.config),
                "Best checkpoint does not belong to this verified run.")
        self.best_epoch = payload["trainer"]["epoch"]
        require(self.best_epoch == report["best_epoch"], "Best checkpoint epoch differs.")
        self.model = build_model(self.config, self.tokenizer, load_pretrained=False)
        self.model.load_state_dict(payload["model"], strict=True)
        del payload
        self.model.to(self.device).eval()
        self.transform = build_image_transform(self.config["data"]["image_size"],
                                               split="validation")
        LOGGER.info("Loaded best epoch %d on %s; no COCO scan or weights download",
                    self.best_epoch, self.device)

    def predict(self, image_path, max_new_tokens=None):
        image_path = Path(image_path).expanduser().resolve()
        require(image_path.is_file(), f"Image file does not exist: {image_path}")
        upper = self.config["data"]["max_caption_length"] - 1
        limit = upper if max_new_tokens is None else max_new_tokens
        require(type(limit) is int and 1 <= limit <= upper,
                f"max_new_tokens must be an integer between 1 and {upper}.")
        image = load_image(image_path)
        try:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        finally:
            image.close()
        with torch.inference_mode():
            output = self.model.generate(tensor, max_new_tokens=limit,
                                         strategy="greedy", beam_size=1)
        sequences = output["sequences"]
        require(sequences.ndim == 2 and sequences.size(0) == 1,
                "Model did not return one caption for one image.")
        tokens, ended = generated_content(sequences[0].cpu().tolist(), self.tokenizer, limit)
        return {
            "image": str(image_path), "caption": self.tokenizer.decode(tokens).strip(),
            "token_ids": tokens, "ended_with_eos": ended,
            "max_new_tokens": limit, "best_epoch": self.best_epoch,
            "checkpoint_sha256": self.checkpoint_sha256, "device": str(self.device),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output", type=Path, help="Optional new JSON result file")
    args = parser.parse_args()
    if args.output is not None:
        require(not args.output.expanduser().exists(), "Output file already exists.")
    captioner = BaselineCaptioner(args.run_dir, device=args.device)
    result = captioner.predict(args.image, max_new_tokens=args.max_new_tokens)
    LOGGER.info("Caption: %s", result["caption"])
    if args.output is not None:
        output = args.output.expanduser().resolve()
        write_json(output, result)
        LOGGER.info("Result: %s", output)
    LOGGER.info("PHASE 11 BASELINE INFERENCE: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("PHASE 11 BASELINE INFERENCE: FAIL")
        raise SystemExit(1)