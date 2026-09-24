"""Caption tokenization with explicit boundaries and offline persistence."""

import hashlib
import json
from pathlib import Path
import tempfile

from transformers import AutoTokenizer

from src.utils.exceptions import ConfigError, DatasetError
from src.utils.logger import get_logger


LOGGER = get_logger("image_captioning.tokenizer")


class CaptionTokenizer:
    """Wrap a saved Hugging Face tokenizer for caption generation."""

    def __init__(self, tokenizer, source_name: str, max_length: int):
        if type(max_length) is not int or max_length < 3:
            raise ConfigError(
                "max_length must allow BOS, at least one token, and EOS"
            )

        if not tokenizer.is_fast:
            raise ConfigError("This implementation requires a fast tokenizer")

        if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
            raise ConfigError("Tokenizer must provide CLS and SEP tokens")

        if tokenizer.pad_token_id is None:
            raise ConfigError("Tokenizer must provide a PAD token")

        self.tokenizer = tokenizer
        self.source_name = source_name
        self.max_length = max_length

        # Reuse existing vocabulary entries; do not add new tokens.
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token
        self.tokenizer.padding_side = "right"

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.vocab_size = len(self.tokenizer)

        boundary_ids = {
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
        }

        if len(boundary_ids) != 3:
            raise ConfigError("BOS, EOS, and PAD must have distinct IDs")

    def content_ids(self, caption: str) -> list[int]:
        """Encode caption content without adding boundaries or truncating."""

        if not isinstance(caption, str) or not caption.strip():
            raise DatasetError("Caption must be a nonempty string")

        encoded = self.tokenizer(
            caption,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        ids = encoded["input_ids"]

        if not ids:
            raise DatasetError("Caption produced no content tokens")

        reserved_ids = {
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
        }

        if any(token_id in reserved_ids for token_id in ids):
            raise DatasetError(
                "Caption contains a reserved BOS/EOS/PAD token"
            )

        return ids

    def encode(self, caption: str) -> list[int]:
        """Return an unpadded sequence with exactly one BOS and one EOS."""

        content = self.content_ids(caption)

        # Reserve two positions so truncation never removes EOS.
        content = content[:self.max_length - 2]

        return [
            self.bos_token_id,
            *content,
            self.eos_token_id,
        ]

    def decode(self, token_ids: list[int]) -> str:
        """Decode one sequence, stopping at its first EOS."""

        content = []

        for token_id in token_ids:
            if type(token_id) is not int:
                raise ValueError("decode expects a list of Python integers")

            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"Invalid vocabulary ID: {token_id}")

            if token_id == self.eos_token_id:
                break

            if token_id in {self.bos_token_id, self.pad_token_id}:
                continue

            content.append(token_id)

        return self.tokenizer.decode(
            content,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    def tokens(self, token_ids: list[int]) -> list[str]:
        """Return readable token strings for debugging."""

        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def fingerprint(self) -> str:
        """Identify tokenization rules, special IDs, and caption length."""

        backend = json.loads(
            self.tokenizer.backend_tokenizer.to_str()
        )

        # These runtime settings can change after encode calls.
        # Our wrapper explicitly disables both when encoding content.
        backend.pop("padding", None)
        backend.pop("truncation", None)

        identity = {
            "backend": backend,
            "source_name": self.source_name,
            "max_length": self.max_length,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "decode_clean_up_tokenization_spaces": True,
        }

        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    def metadata(self) -> dict:
        return {
            "schema_version": 1,
            "source_name": self.source_name,
            "max_length": self.max_length,
            "vocab_size": self.vocab_size,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "fingerprint": self.fingerprint(),
        }

    def save(self, directory: Path) -> None:
        """Save a complete tokenizer to a new directory atomically."""

        directory = Path(directory)

        if directory.exists():
            raise DatasetError(
                f"Refusing to overwrite tokenizer directory: {directory}"
            )

        directory.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=".tokenizer-",
            dir=directory.parent,
        ) as temporary:
            staged = Path(temporary) / "saved"
            staged.mkdir()

            self.tokenizer.save_pretrained(staged)

            (staged / "caption_tokenizer.json").write_text(
                json.dumps(self.metadata(), indent=2) + "\n",
                encoding="utf-8",
            )

            staged.rename(directory)

    @classmethod
    def load(cls, directory: Path):
        """Load only local files and verify the saved tokenizer identity."""

        directory = Path(directory)

        metadata_path = directory / "caption_tokenizer.json"

        if not metadata_path.is_file():
            raise DatasetError(
                f"Tokenizer metadata is missing: {metadata_path}"
            )

        saved = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if saved.get("schema_version") != 1:
            raise DatasetError("Unsupported caption tokenizer schema")

        tokenizer = AutoTokenizer.from_pretrained(
            str(directory),
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )

        result = cls(
            tokenizer=tokenizer,
            source_name=saved["source_name"],
            max_length=saved["max_length"],
        )

        if result.metadata() != saved:
            raise DatasetError(
                "Saved tokenizer identity does not match its files"
            )

        return result


def tokenizer_directory(config: dict) -> Path:
    """Resolve the saved tokenizer location from existing configuration."""

    source_name = config["data"]["tokenizer_name"]

    if source_name != "bert-base-uncased":
        raise ConfigError(
            "The agreed Phase 4 tokenizer is bert-base-uncased"
        )

    max_length = config["data"]["max_caption_length"]

    return (
        Path(config["paths"]["data_root"])
        / "processed"
        / "tokenizers"
        / f"bert-base-uncased_len{max_length}"
    )


def load_caption_tokenizer(
    config: dict,
    allow_download: bool = False,
) -> CaptionTokenizer:
    """Reuse a saved tokenizer; download only when explicitly allowed."""

    directory = tokenizer_directory(config)
    source_name = config["data"]["tokenizer_name"]
    max_length = config["data"]["max_caption_length"]

    if directory.exists():
        result = CaptionTokenizer.load(directory)

        if (
            result.source_name != source_name
            or result.max_length != max_length
        ):
            raise ConfigError(
                "Saved tokenizer does not match the configuration"
            )

        LOGGER.info("Loaded tokenizer from local files: %s", directory)
        return result

    if not allow_download:
        raise DatasetError(
            "Saved tokenizer not found. Run "
            "src.data.check_tokenizer once without --offline."
        )

    LOGGER.info("Loading tokenizer assets: %s", source_name)

    tokenizer = AutoTokenizer.from_pretrained(
        source_name,
        use_fast=True,
        trust_remote_code=False,
    )

    result = CaptionTokenizer(
        tokenizer=tokenizer,
        source_name=source_name,
        max_length=max_length,
    )

    result.save(directory)

    LOGGER.info("Saved tokenizer: %s", directory)

    # Verify the saved copy immediately using local files only.
    return CaptionTokenizer.load(directory)