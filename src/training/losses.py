"""Token-masked caption loss."""

import torch
from torch import nn
from torch.nn import functional as F

from src.data.collate import IGNORE_INDEX


class CaptionCrossEntropy(nn.Module):
    def forward(self, logits, target_ids):
        if logits.ndim != 3:
            raise ValueError("Expected logits with shape [B, T, V]")

        if target_ids.shape != logits.shape[:2]:
            raise ValueError("Logits and target shapes do not match")

        if target_ids.dtype != torch.long:
            raise ValueError("Targets must be int64")

        valid_tokens = int(
            target_ids.ne(IGNORE_INDEX).sum().item()
        )

        if valid_tokens == 0:
            raise ValueError("Batch contains no valid target tokens")

        # Use float32 for stable loss calculation, including under AMP.
        loss_sum = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )

        return {
            "loss": loss_sum / valid_tokens,
            "loss_sum": loss_sum.detach(),
            "valid_tokens": valid_tokens,
        }