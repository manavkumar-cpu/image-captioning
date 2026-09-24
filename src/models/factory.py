"""Construct a captioning model from the existing configuration."""

from src.models.cnn_lstm import CNNLSTMModel
from src.utils.exceptions import ConfigError


def build_model(config, tokenizer, load_pretrained=None):
    """Create the baseline.

    load_pretrained=None follows config.
    A later checkpoint loader can pass False before restoring all weights,
    avoiding an unnecessary ImageNet download.
    """

    settings = config["model"]

    if settings["name"] != "cnn_lstm":
        raise ConfigError(
            f"Unsupported model: {settings['name']}"
        )

    if settings["encoder"] != "resnet50":
        raise ConfigError(
            "The baseline currently requires resnet50"
        )

    if tokenizer.max_length != config["data"]["max_caption_length"]:
        raise ConfigError(
            "Tokenizer length differs from model configuration"
        )

    if load_pretrained is None:
        load_pretrained = settings["pretrained"]

    if type(load_pretrained) is not bool:
        raise ConfigError("load_pretrained must be boolean or None")

    return CNNLSTMModel(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        embedding_dim=settings["embedding_dim"],
        hidden_dim=settings["hidden_dim"],
        num_layers=settings["num_layers"],
        dropout=settings["dropout"],
        pretrained=load_pretrained,
        freeze_encoder=settings["freeze_encoder"],
        max_caption_length=config["data"]["max_caption_length"],
    )