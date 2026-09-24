"""Reusable caption training, validation and early stopping."""

from contextlib import nullcontext
from copy import deepcopy
from itertools import islice
import math
from time import perf_counter

import torch
from torch.nn.utils import clip_grad_norm_

from src.training.losses import CaptionCrossEntropy
from src.utils.exceptions import ConfigError
from src.utils.logger import get_logger


class EarlyStopping:
    def __init__(self, patience, mode="min"):
        if patience < 1 or mode not in {"min", "max"}:
            raise ValueError("Invalid early-stopping settings")

        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def update(self, score):
        if not math.isfinite(score):
            raise ValueError("Monitored score must be finite")

        improved = (
            self.best is None
            or (self.mode == "min" and score < self.best)
            or (self.mode == "max" and score > self.best)
        )

        if improved:
            self.best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return improved

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


class Trainer:
    def __init__(
        self,
        config,
        device,
        optimizer,
        scheduler=None,
        loss_fn=None,
    ):
        self.config = deepcopy(config)
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn if loss_fn is not None else CaptionCrossEntropy()
        self.logger = get_logger("image_captioning.trainer")

        self.gradient_clip = config["training"]["gradient_clip"]
        self.monitor = config["training"]["monitor"]
        self.use_amp = config["runtime"]["mixed_precision"]

        # Initial AMP implementation is CUDA-only.
        # Your current MPS configuration has mixed_precision=False.
        if self.use_amp and self.device.type != "cuda":
            raise ConfigError(
                "This Trainer's AMP path currently supports CUDA only. "
                "Keep mixed_precision=false on MPS/CPU."
            )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.early_stopping = EarlyStopping(
            patience=config["training"]["patience"],
            mode=config["training"]["monitor_mode"],
        )

        self.epoch = 0
        self.global_step = 0
        self.history = []

    def _precision_context(self):
        if self.use_amp:
            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )

        return nullcontext()

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _move_batch(self, batch):
        return {
            key: value.to(self.device)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }

    def _check_optimizer(self, model):
        expected = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }

        selected = [
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]

        if len(selected) != len(set(selected)) or set(selected) != expected:
            raise ValueError(
                "Optimizer parameters do not match model trainable parameters"
            )

        for parameter in model.parameters():
            if parameter.device.type != self.device.type:
                raise ValueError(
                    "Move the model to the selected device "
                    "before constructing its optimizer"
                )

            if (
                self.device.index is not None
                and parameter.device.index != self.device.index
            ):
                raise ValueError("Model is on a different device index")

    def _run_epoch(self, model, loader, training, max_batches=None):
        if max_batches is not None and max_batches < 1:
            raise ValueError("max_batches must be positive")

        previous_mode = model.training
        model.train(training)

        self._synchronize()
        started = perf_counter()

        loss_total = 0.0
        token_total = 0
        batches = 0
        updates = 0
        skipped_updates = 0

        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]

        iterator = (
            loader
            if max_batches is None
            else islice(loader, max_batches)
        )

        try:
            for raw_batch in iterator:
                batch = self._move_batch(raw_batch)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(training):
                    with self._precision_context():
                        output = model(
                            images=batch["images"],
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )

                        result = self.loss_fn(
                            output["logits"],
                            batch["target_ids"],
                        )

                        loss = result["loss"]

                    if not torch.isfinite(loss).item():
                        raise RuntimeError("Encountered a nonfinite loss")

                    if training:
                        self.scaler.scale(loss).backward()

                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)

                        clip_grad_norm_(
                            parameters,
                            max_norm=self.gradient_clip,
                            error_if_nonfinite=not self.use_amp,
                            foreach=False,
                        )

                        previous_scale = self.scaler.get_scale()

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                        # AMP lowers its scale when an update is skipped.
                        skipped = (
                            self.use_amp
                            and self.scaler.get_scale() < previous_scale
                        )

                        if skipped:
                            skipped_updates += 1
                        else:
                            updates += 1
                            self.global_step += 1

                        self.optimizer.zero_grad(set_to_none=True)

                loss_total += float(result["loss_sum"].item())
                token_total += result["valid_tokens"]
                batches += 1

                if training and batches % 50 == 0:
                    self.logger.info(
                        "Training batches=%d, token-mean loss=%.4f",
                        batches,
                        loss_total / token_total,
                    )

            if token_total == 0:
                raise RuntimeError("Loader produced no valid target tokens")

            self._synchronize()

            return {
                "loss": loss_total / token_total,
                "valid_tokens": token_total,
                "batches": batches,
                "optimizer_steps": updates,
                "skipped_steps": skipped_updates,
                "seconds": perf_counter() - started,
            }

        finally:
            model.train(previous_mode)

    def validate(self, model, loader, max_batches=None):
        return self._run_epoch(
            model,
            loader,
            training=False,
            max_batches=max_batches,
        )

    def fit(
        self,
        model,
        train_loader,
        validation_loader,
        epochs=None,
        max_train_batches=None,
        max_val_batches=None,
        on_epoch_end=None,
        metric_fn=None,
    ):
        """Train until a total target epoch count.

        metric_fn(model, validation_loader) may return extra scalar metrics.
        on_epoch_end(trainer, model, record) is reserved for persistence/logging.
        """

        self._check_optimizer(model)

        target_epochs = (
            self.config["training"]["epochs"]
            if epochs is None
            else epochs
        )

        if type(target_epochs) is not int or target_epochs < 1:
            raise ValueError("epochs must be a positive integer")

        if self.monitor != "val_loss" and metric_fn is None:
            raise ValueError(
                "A non-loss monitor requires a validation metric callback"
            )

        for epoch in range(self.epoch + 1, target_epochs + 1):
            if self.early_stopping.should_stop:
                break

            learning_rates = [
                group["lr"] for group in self.optimizer.param_groups
            ]

            train_result = self._run_epoch(
                model,
                train_loader,
                training=True,
                max_batches=max_train_batches,
            )

            val_result = self.validate(
                model,
                validation_loader,
                max_batches=max_val_batches,
            )

            record = {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "val_loss": val_result["loss"],
                "train_tokens": train_result["valid_tokens"],
                "val_tokens": val_result["valid_tokens"],
                "train_batches": train_result["batches"],
                "val_batches": val_result["batches"],
                "train_seconds": train_result["seconds"],
                "validation_seconds": val_result["seconds"],
                "optimizer_steps": train_result["optimizer_steps"],
                "skipped_steps": train_result["skipped_steps"],
                "learning_rates": learning_rates,
                "batch_limits": {
                    "train": max_train_batches,
                    "validation": max_val_batches,
                },
            }

            if metric_fn is not None:
                previous_mode = model.training
                model.eval()

                try:
                    with torch.no_grad():
                        extra = metric_fn(model, validation_loader)
                finally:
                    model.train(previous_mode)

                if set(extra) & set(record):
                    raise ValueError("Extra metrics overwrite reserved fields")

                record.update({
                    key: float(value) for key, value in extra.items()
                })

            if self.monitor not in record:
                raise ValueError(f"Missing monitored metric: {self.monitor}")

            score = float(record[self.monitor])
            improved = self.early_stopping.update(score)

            # The supplied scheduler is validation-metric driven.
            if self.scheduler is not None:
                self.scheduler.step(score)

            record["next_learning_rates"] = [
                group["lr"] for group in self.optimizer.param_groups
            ]
            record["improved"] = improved
            record["best_score"] = self.early_stopping.best
            record["bad_epochs"] = self.early_stopping.bad_epochs

            self.epoch = epoch
            self.history.append(record)

            if on_epoch_end is not None:
                on_epoch_end(self, model, record)

            self.logger.info(
                "Epoch %d | train_loss=%.4f | val_loss=%.4f | "
                "updates=%d | improved=%s",
                epoch,
                record["train_loss"],
                record["val_loss"],
                record["optimizer_steps"],
                improved,
            )

            if self.early_stopping.should_stop:
                self.logger.info("Early stopping triggered")
                break

        return list(self.history)