"""Shared interface for baseline and advanced captioning models."""

from abc import ABC, abstractmethod

from torch import nn


class BaseCaptionModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, images, input_ids, attention_mask):
        """Return {"logits": tensor[B, T, vocabulary_size]}.

        Inputs and targets are already shifted by the collator.
        The model must not shift them again.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        images,
        max_new_tokens,
        strategy="greedy",
        beam_size=1,
    ):
        """Return {"sequences": tensor[B, L], "attention": optional}.

        Sequences exclude BOS, include EOS when generated, and contain
        PAD after EOS when other batch members continue generating.
        """
        raise NotImplementedError