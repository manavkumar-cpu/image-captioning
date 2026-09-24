"""Small baseline architecture check; no optimizer updates or training epoch."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import torch
from torch.nn import functional as F

from src.data.collate import IGNORE_INDEX
from src.data.dataloader import build_dataloaders
from src.models.factory import build_model
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


LOGGER = get_logger("image_captioning.baseline_check")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_generated_sequences(sequences, tokenizer, batch_size, limit):
    require(sequences.ndim == 2, "Generation must return [B, L]")
    require(sequences.shape[0] == batch_size, "Wrong generation batch size")
    require(1 <= sequences.shape[1] <= limit, "Wrong generation length")
    require(sequences.dtype == torch.long, "Generated IDs must be int64")

    for row in sequences.cpu().tolist():
        ended = False

        for token in row:
            require(
                0 <= token < tokenizer.vocab_size,
                "Invalid generated token ID",
            )

            if ended:
                require(
                    token == tokenizer.pad_token_id,
                    "Only PAD may appear after EOS",
                )
            else:
                require(
                    token not in {
                        tokenizer.bos_token_id,
                        tokenizer.pad_token_id,
                    },
                    "BOS/PAD appeared before EOS",
                )

                if token == tokenizer.eos_token_id:
                    ended = True


def check_controlled_eos(model, images, tokenizer):
    """Force different stopping times without modifying model weights."""

    ordinary_token = tokenizer.content_ids("dog")[0]
    calls = 0

    def controlled_scores(hidden):
        nonlocal calls

        scores = hidden.new_full(
            (hidden.shape[0], tokenizer.vocab_size),
            -1000.0,
        )

        if calls == 0:
            # Row 0 stops immediately. Remaining rows continue once.
            scores[:, ordinary_token] = 1000.0
            scores[0, ordinary_token] = -1000.0
            scores[0, tokenizer.eos_token_id] = 1000.0
        else:
            scores[:, tokenizer.eos_token_id] = 1000.0

        calls += 1
        return scores

    with patch.object(
        model.output_projection,
        "forward",
        side_effect=controlled_scores,
    ):
        result = model.generate(images, max_new_tokens=3)

    sequences = result["sequences"].cpu().tolist()

    expected = (
        [[tokenizer.eos_token_id]]
        if images.shape[0] == 1
        else [
            [tokenizer.eos_token_id, tokenizer.pad_token_id],
            *[
                [ordinary_token, tokenizer.eos_token_id]
                for _ in range(images.shape[0] - 1)
            ],
        ]
    )

    require(sequences == expected, "EOS stopping or post-EOS padding failed")

    LOGGER.info("Controlled EOS stopping and batch padding: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default="configs/baseline_debug.yaml",
    )
    parser.add_argument("--val-images", type=int, default=250)
    parser.add_argument("--samples", type=int, default=2)

    args = parser.parse_args()

    try:
        require(1 <= args.samples <= 4, "samples must be between 1 and 4")

        config = load_config(args.config)

        require(
            config["model"]["freeze_encoder"],
            "This phase's check expects the agreed frozen encoder",
        )

        seed_everything(config["project"]["seed"])
        device = select_device(config["runtime"]["device"])

        loaders = build_dataloaders(config, args.val_images)
        tokenizer = loaders["tokenizer"]

        original_batch = next(iter(loaders["train"]))
        sample_count = min(args.samples, len(original_batch["image_ids"]))

        batch = {
            key: value[:sample_count].to(device)
            if torch.is_tensor(value)
            else value[:sample_count]
            for key, value in original_batch.items()
        }

        # Remove padding only needed by samples excluded from this check.
        sequence_length = int(
            batch["attention_mask"].sum(dim=1).max().item()
        )

        for key in ("input_ids", "attention_mask", "target_ids"):
            batch[key] = batch[key][:, :sequence_length].contiguous()

        LOGGER.info(
            "Creating baseline on %s; pretrained weights may download once",
            device,
        )

        model = build_model(config, tokenizer).to(device)

        total_parameters = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        LOGGER.info("Total parameters: %s", f"{total_parameters:,}")
        LOGGER.info("Trainable parameters: %s", f"{trainable_parameters:,}")

        model.train()

        require(model.training, "Decoder did not enter training mode")
        require(
            all(not module.training for module in model.encoder.modules()),
            "Frozen encoder entered training mode",
        )
        require(
            all(not p.requires_grad for p in model.encoder.parameters()),
            "Encoder parameters are not frozen",
        )

        encoder_buffers = {
            name: value.detach().clone()
            for name, value in model.encoder.named_buffers()
        }

        model.zero_grad(set_to_none=True)

        result = model(
            images=batch["images"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = result["logits"]

        expected_shape = (
            sample_count,
            sequence_length,
            tokenizer.vocab_size,
        )

        require(
            tuple(logits.shape) == expected_shape,
            f"Expected logits {expected_shape}, got {tuple(logits.shape)}",
        )
        require(
            torch.isfinite(logits).all().item(),
            "Logits contain NaN or infinity",
        )

        # Diagnostic loss only. The reusable loss module comes in Phase 7.
        loss = F.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size),
            batch["target_ids"].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        require(torch.isfinite(loss).item(), "Diagnostic loss is not finite")

        loss.backward()

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                require(
                    parameter.grad is not None,
                    f"Missing gradient for {name}",
                )
                require(
                    torch.isfinite(parameter.grad).all().item(),
                    f"Nonfinite gradient for {name}",
                )
            else:
                require(
                    parameter.grad is None,
                    f"Frozen parameter received a gradient: {name}",
                )

        for layer_name in ("hidden_projection", "cell_projection"):
            gradient = getattr(model, layer_name).weight.grad

            require(
                gradient.abs().sum().item() > 0,
                f"No image-conditioning gradient through {layer_name}",
            )

        for name, value in model.encoder.named_buffers():
            require(
                torch.equal(value, encoder_buffers[name]),
                f"Frozen encoder buffer changed: {name}",
            )

        loss_value = loss.item()

        LOGGER.info("Forward logits shape: %s — PASS", tuple(logits.shape))
        LOGGER.info("Diagnostic loss: %.6f", loss_value)
        LOGGER.info("Decoder and image-projection gradient flow: PASS")
        LOGGER.info("Frozen encoder gradients and BatchNorm buffers: PASS")

        # Free backward buffers before checking generation.
        model.zero_grad(set_to_none=True)
        del logits, loss, result, encoder_buffers

        limit = min(8, config["data"]["max_caption_length"] - 1)

        generated = model.generate(
            batch["images"],
            max_new_tokens=limit,
        )

        check_generated_sequences(
            generated["sequences"],
            tokenizer,
            sample_count,
            limit,
        )

        require(
            generated["attention"] is None,
            "This baseline should not claim to return attention maps",
        )

        require(model.training, "Generation did not restore training mode")
        require(not model.encoder.training, "Frozen encoder mode changed")

        check_controlled_eos(model, batch["images"], tokenizer)

        require(model.training, "Controlled generation changed model mode")

        untrained_examples = [
            {
                "image_id": image_id,
                "token_ids": ids,
                "decoded_text": tokenizer.decode(ids),
            }
            for image_id, ids in zip(
                batch["image_ids"],
                generated["sequences"].cpu().tolist(),
            )
        ]

        LOGGER.info("Greedy generation structure and mode restoration: PASS")
        LOGGER.info(
            "Decoder is untrained; generated text is not a quality result"
        )

        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            + "_"
            + uuid4().hex[:8]
        )

        output_dir = (
            Path(config["paths"]["artifacts_root"])
            / "baseline_checks"
            / run_id
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        report = {
            "status": "passed",
            "device": str(device),
            "samples_checked": sample_count,
            "logits_shape": list(expected_shape),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "diagnostic_loss": loss_value,
            "encoder_weights": (
                "ResNet50_Weights.IMAGENET1K_V2"
                if config["model"]["pretrained"]
                else "random"
            ),
            "encoder_frozen": True,
            "tokenizer_fingerprint": tokenizer.fingerprint(),
            "untrained_generation_examples": untrained_examples,
            "config": config,
            "scope": (
                "Architecture, one backward pass, frozen encoder, "
                "and greedy generation checks. No optimizer updates, "
                "training epoch, caption metrics, or saved model checkpoint."
            ),
        }

        report_path = output_dir / "report.json"

        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        LOGGER.info("Report: %s", report_path)
        LOGGER.info("PHASE 6 BASELINE CHECK: PASS")

        return 0

    except Exception as exc:
        LOGGER.error("BASELINE CHECK FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())