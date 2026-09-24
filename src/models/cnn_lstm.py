"""ResNet50 encoder with an image-initialized LSTM caption decoder."""

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from src.models.base_model import BaseCaptionModel


class CNNLSTMModel(BaseCaptionModel):
    def __init__(
        self,
        vocab_size,
        pad_token_id,
        bos_token_id,
        eos_token_id,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=1,
        dropout=0.3,
        pretrained=True,
        freeze_encoder=True,
        max_caption_length=32,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.freeze_encoder = freeze_encoder
        self.max_caption_length = max_caption_length

        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )

        self.encoder = resnet50(weights=weights)

        # Retain ResNet's global average pooling, remove classification.
        feature_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.feature_dim = feature_dim

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not freeze_encoder)

        if freeze_encoder:
            self.encoder.eval()

        self.hidden_projection = nn.Linear(
            feature_dim,
            num_layers * hidden_dim,
        )

        self.cell_projection = nn.Linear(
            feature_dim,
            num_layers * hidden_dim,
        )

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_dropout = nn.Dropout(dropout)

        self.output_projection = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    def train(self, mode=True):
        """Keep the frozen encoder, including BatchNorm, in eval mode."""

        super().train(mode)

        if self.freeze_encoder:
            self.encoder.eval()

        return self

    def encode_images(self, images):
        """Return a global feature vector for each image."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [B, 3, H, W]")

        if not images.is_floating_point():
            raise ValueError("Images must be floating-point tensors")

        if self.freeze_encoder:
            # no_grad avoids encoder backpropagation while leaving its
            # output usable by the trainable projection layers.
            with torch.no_grad():
                return self.encoder(images)

        return self.encoder(images)

    def initial_state(self, features):
        """Project visual features into initial LSTM hidden/cell states."""

        batch_size = features.shape[0]

        hidden = torch.tanh(self.hidden_projection(features))
        cell = torch.tanh(self.cell_projection(features))

        # LSTM states use [layers, batch, hidden], even with batch_first.
        hidden = hidden.reshape(
            batch_size,
            self.num_layers,
            self.hidden_dim,
        ).permute(1, 0, 2).contiguous()

        cell = cell.reshape(
            batch_size,
            self.num_layers,
            self.hidden_dim,
        ).permute(1, 0, 2).contiguous()

        return hidden, cell

    def forward(self, images, input_ids, attention_mask):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must be int64")

        if attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean")

        if attention_mask.shape != input_ids.shape:
            raise ValueError("Input IDs and attention mask shapes differ")

        if images.shape[0] != input_ids.shape[0]:
            raise ValueError("Image and caption batch sizes differ")

        if not 1 <= input_ids.shape[1] <= self.max_caption_length - 1:
            raise ValueError("Input sequence length is outside its limit")

        if not torch.equal(
            attention_mask,
            input_ids.ne(self.pad_token_id),
        ):
            raise ValueError("Attention mask does not match input padding")

        lengths = attention_mask.sum(dim=1)

        expected_mask = (
            torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
            ).unsqueeze(0)
            < lengths.unsqueeze(1)
        )

        if not torch.equal(attention_mask, expected_mask):
            raise ValueError("The baseline requires right-padded inputs")

        if (lengths == 0).any().item():
            raise ValueError("Empty caption inputs are not supported")

        features = self.encode_images(images)
        state = self.initial_state(features)

        embeddings = self.embedding(input_ids)
        embeddings = self.embedding_dropout(embeddings)

        embeddings = embeddings.masked_fill(
            ~attention_mask.unsqueeze(-1),
            0.0,
        )

        outputs, _ = self.lstm(embeddings, state)
        outputs = self.output_dropout(outputs)

        logits = self.output_projection(outputs)

        # Padding positions can have nonzero logits. The shared loss
        # will ignore their -100 targets.
        return {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        images,
        max_new_tokens,
        strategy="greedy",
        beam_size=1,
    ):
        if strategy != "greedy" or beam_size != 1:
            raise NotImplementedError(
                "The baseline currently supports greedy generation only"
            )

        if (
            type(max_new_tokens) is not int
            or not 1 <= max_new_tokens <= self.max_caption_length - 1
        ):
            raise ValueError(
                "max_new_tokens must be between 1 and "
                "max_caption_length - 1"
            )

        was_training = self.training
        self.eval()

        try:
            features = self.encode_images(images)
            state = self.initial_state(features)

            batch_size = images.shape[0]

            current_tokens = torch.full(
                (batch_size, 1),
                self.bos_token_id,
                dtype=torch.long,
                device=images.device,
            )

            finished = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=images.device,
            )

            generated = []

            for _ in range(max_new_tokens):
                embeddings = self.embedding(current_tokens)
                outputs, state = self.lstm(embeddings, state)

                scores = self.output_projection(outputs[:, -1, :])

                # BOS and PAD are not valid new caption content.
                scores[:, self.bos_token_id] = -torch.inf
                scores[:, self.pad_token_id] = -torch.inf

                next_tokens = scores.argmax(dim=-1)

                # Already-finished rows produce only PAD.
                next_tokens = torch.where(
                    finished,
                    torch.full_like(next_tokens, self.pad_token_id),
                    next_tokens,
                )

                generated.append(next_tokens)

                finished = finished | next_tokens.eq(self.eos_token_id)

                if finished.all().item():
                    break

                current_tokens = next_tokens.unsqueeze(1)

            return {
                "sequences": torch.stack(generated, dim=1),
                "attention": None,
            }

        finally:
            self.train(was_training)