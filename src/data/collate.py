"""Batch assembly with exactly one next-token shift."""

import torch
from torch.nn.utils.rnn import pad_sequence

from src.utils.exceptions import DatasetError


IGNORE_INDEX = -100


class CaptionCollator:
    """Shift each complete sequence, then dynamically pad the batch."""

    def __init__(self, tokenizer):
        # Store only the small values required by the collator.
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.vocab_size = tokenizer.vocab_size
        self.max_length = tokenizer.max_length

    def __call__(self, samples):
        if not samples:
            raise DatasetError("Cannot collate an empty batch")

        inputs = []
        targets = []

        for sample in samples:
            ids = sample["token_ids"]

            if ids.ndim != 1 or ids.dtype != torch.long:
                raise DatasetError("Expected a one-dimensional int64 sequence")

            if not 3 <= len(ids) <= self.max_length:
                raise DatasetError("Caption length is outside the allowed range")

            if ids[0].item() != self.bos_id:
                raise DatasetError("Caption must begin with BOS")

            if ids[-1].item() != self.eos_id:
                raise DatasetError("Caption must end with EOS")

            if (ids == self.pad_id).any().item():
                raise DatasetError("Dataset sequences must be unpadded")

            if (
                (ids == self.bos_id).sum().item() != 1
                or (ids == self.eos_id).sum().item() != 1
            ):
                raise DatasetError("Expected exactly one BOS and one EOS")

            if ((ids < 0) | (ids >= self.vocab_size)).any().item():
                raise DatasetError("Caption contains an invalid token ID")

            # This is the only next-token shift in the data pipeline.
            inputs.append(ids[:-1])
            targets.append(ids[1:])

        input_ids = pad_sequence(
            inputs,
            batch_first=True,
            padding_value=self.pad_id,
        )

        target_ids = pad_sequence(
            targets,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        attention_mask = input_ids.ne(self.pad_id)

        return {
            "images": torch.stack(
                [sample["image"] for sample in samples]
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "image_ids": [sample["image_id"] for sample in samples],
        }


def collate_images(samples):
    """Collate image-level generation samples without caption inputs."""

    if not samples:
        raise DatasetError("Cannot collate an empty image batch")

    return {
        "images": torch.stack(
            [sample["image"] for sample in samples]
        ),
        "image_ids": [sample["image_id"] for sample in samples],
        "references": [
            list(sample["references"]) for sample in samples
        ],
    }