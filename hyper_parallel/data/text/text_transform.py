# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Transform plaintext and conversation records into model samples."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.dataset_logging import get_dataset_logger

TextDataType = Literal["plaintext", "conversation"]
logger = get_dataset_logger(__name__)


def _get_record_value(sample: Mapping[str, Any], keys: str | Sequence[str]) -> Any:
    if isinstance(keys, str):
        try:
            return sample[keys]
        except KeyError as exc:
            raise ValueError(f"Sample does not contain field {keys!r}") from exc
    for key in keys:
        if key in sample:
            return sample[key]
    raise ValueError(f"Sample does not contain any configured text fields: {list(keys)!r}")


class IdentityDataTransform:
    """Return each input sample unchanged."""

    def __init__(self, tokenizer: Any = None, chat_template: Any = None) -> None:
        """Retain optional upstream assets for target compatibility.

        Args:
            tokenizer: Optional tokenizer built by the LLM Trainer.
            chat_template: Optional chat template built from model assets.
        """
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    @staticmethod
    def __call__(sample: Any) -> Any:
        """Return the input sample without modification."""
        return sample


@dataclass
class PlaintextTransform:
    """Tokenize plaintext records into one or more model samples."""

    tokenizer: Any
    max_seq_len: int
    text_keys: str | Sequence[str] = "text"
    drop_ragged_tail: bool = False

    def __post_init__(self) -> None:
        """Validate the tokenizer and sequence length configuration."""
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for plaintext data")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    def __call__(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Tokenize and chunk one plaintext record."""
        text = _get_record_value(sample, self.text_keys)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            token_ids = [*token_ids, eos_token_id]

        transformed = []
        for start in range(0, len(token_ids) - 1, self.max_seq_len):
            window = token_ids[start:start + self.max_seq_len + 1]
            if (self.drop_ragged_tail and start > 0
                    and len(window) < self.max_seq_len + 1):
                # Opt-in for models whose packed-sample boundaries must stay
                # aligned, such as the V4.1 CSA compression ratio and the
                # context-parallel shard size: a short tail window violates
                # both once it lands in a batch. Off by default, so every
                # other model keeps the trailing partial window.
                break
            text = torch.tensor(window, dtype=torch.long)
            model_sample = {
                "input_ids": text[:-1],
                "labels": text[1:],
            }
            transformed.append(model_sample)
        return transformed

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Return whether one source record contains non-empty plaintext."""
        text = _get_record_value(sample, self.text_keys)
        if not isinstance(text, str):
            raise ValueError("Plaintext sample text must be a string")
        is_valid = text.strip() != ""
        return is_valid


@dataclass
class TextConversationTransform:
    """Encode conversation records with a configured chat template."""

    chat_template: Any
    max_seq_len: int
    text_keys: str | Sequence[str] = "conversation"

    def __post_init__(self) -> None:
        """Validate the chat template and sequence length configuration."""
        if self.chat_template is None:
            raise ValueError("chat_template is required for conversation data")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    def __call__(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Encode one conversation record."""
        messages = _get_record_value(sample, self.text_keys)
        encoded = self.chat_template.encode_messages(messages, max_seq_len=self.max_seq_len)
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        labels = torch.as_tensor(encoded["labels"], dtype=torch.long)
        shifted_labels = labels[1:]
        if not bool(shifted_labels.ne(IGNORE_INDEX).any()):
            return []

        model_sample = {
            "input_ids": input_ids[:-1],
            "labels": shifted_labels,
        }
        return [model_sample]

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Return whether one source record contains conversation messages."""
        messages = _get_record_value(sample, self.text_keys)
        is_valid = bool(messages)
        return is_valid


@dataclass
class PreTokenizedTransform:
    """Validate prepared pretraining or SFT records without changing targets.

    Inputs are unpadded one-dimensional integer arrays. Labels already contain
    the next-token target at each input position. No tokenization, EOS insertion,
    shift, truncation or window sampling is performed here.

    Args:
        input_ids_key: Source token field.
        labels_key: Source next-token target field; -100 means ignored.
        boundaries_key: Source cumulative document boundaries, including zero.
            If absent, treat the complete record as one document.
        loss_mask_key: Optional binary mask, folded into an owned labels array.
        empty_target_policy: Reject empty supervision by default; ``drop`` keeps
            the historical JT filtering behavior before sampling.
        labels_alignment: Must be ``next_token``; unshifted targets require an
            upstream, per-document conversion, never a batch-wide shift.
        vocab_size: Optional exclusive vocabulary bound for token validation.
    """

    input_ids_key: str = "input_ids"
    labels_key: str = "labels"
    boundaries_key: str = "boundaries"
    loss_mask_key: str = "loss_mask"
    empty_target_policy: str = "error"
    labels_alignment: str = "next_token"
    vocab_size: int | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous target conventions and unsupported filtering modes."""
        if self.labels_alignment != "next_token":
            raise ValueError("PreTokenizedTransform requires labels_alignment='next_token'")
        if self.empty_target_policy not in ("error", "drop"):
            raise ValueError("empty_target_policy must be 'error' or 'drop'")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

    def _read_record(self, sample: Mapping[str, Any]) -> tuple[np.ndarray, ...] | None:
        tokens = np.asarray(sample[self.input_ids_key])
        labels = np.asarray(sample[self.labels_key])
        boundaries = np.asarray(sample.get(self.boundaries_key, [0, tokens.size]))
        self._validate_arrays(tokens, labels, boundaries)
        valid_targets = labels != IGNORE_INDEX
        if self.loss_mask_key in sample:
            mask = np.asarray(sample[self.loss_mask_key])
            if mask.shape != labels.shape or not np.isin(mask, (0, 1)).all():
                raise ValueError("loss_mask must be binary and match labels")
            valid_targets &= mask != 0
        if not valid_targets.any():
            if self.empty_target_policy == "error":
                raise ValueError("Prepared record has no effective targets")
            return None
        return tokens, labels, boundaries, valid_targets

    def _validate_arrays(self, tokens: np.ndarray, labels: np.ndarray, boundaries: np.ndarray) -> None:
        if any(value.ndim != 1 or value.dtype.kind not in "iu" for value in (tokens, labels, boundaries)):
            raise ValueError("Prepared fields must be one-dimensional integer arrays")
        if tokens.size == 0 or tokens.shape != labels.shape:
            raise ValueError("Tokens and next-token labels must have matching nonempty shapes")
        upper_bound = np.iinfo(np.int32).max + 1
        if self.vocab_size is not None:
            upper_bound = min(upper_bound, self.vocab_size)
        if (np.any(tokens < 0) or np.any(tokens >= upper_bound)
                or np.any((labels < 0) & (labels != IGNORE_INDEX)) or np.any(labels >= upper_bound)):
            raise ValueError("Token IDs must fit the vocabulary/int32 range; ignored labels must equal -100")
        if (tokens.size > np.iinfo(np.int32).max or len(boundaries) < 2
                or boundaries[0] != 0 or boundaries[-1] != tokens.size
                or np.any(boundaries[1:] <= boundaries[:-1])):
            raise ValueError("Document boundaries must strictly increase from zero to token count")

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Validate a source record before sampling.

        Args:
            sample: Prepared source fields before any tensor conversion.

        Returns:
            Whether the record has supervision; strict mode raises instead of dropping.
        """
        return self._read_record(sample) is not None

    def __call__(self, sample: Mapping[str, Any]) -> list[dict[str, torch.Tensor]]:
        """Return one complete record in the canonical training field format."""
        record = self._read_record(sample)
        if record is None:
            return []
        tokens, labels, boundaries, valid_targets = record
        # Arrow and mmap arrays may be read-only or shared with other records.
        labels = labels.astype(np.int64, copy=True)
        labels[~valid_targets] = IGNORE_INDEX
        return [{
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.from_numpy(labels),
            "cu_seq_lens": torch.tensor(boundaries, dtype=torch.int32),
        }]


def build_text_transform(
    data_type: TextDataType,
    *,
    tokenizer: Any = None,
    chat_template: Any = None,
    max_seq_len: int,
    text_keys: str | Sequence[str] = "text",
    drop_ragged_tail: bool = False,
) -> Callable[[Any], Any]:
    """Build the transform selected by the text data type.

    Args:
        data_type: Plaintext or conversation input format.
        tokenizer: Tokenizer used by plaintext transforms.
        chat_template: Chat template used by conversation transforms.
        max_seq_len: Maximum model sequence length.
        text_keys: Field or candidate fields containing the source text.
        drop_ragged_tail: Drop the trailing partial window of a multi-window
            plaintext record instead of emitting it as a shorter sample.

    Returns:
        The configured text sample transform.

    Raises:
        ValueError: If ``data_type`` is unsupported.
    """
    if data_type == "plaintext":
        data_transform = PlaintextTransform(
            tokenizer, max_seq_len, text_keys, drop_ragged_tail)
    elif data_type == "conversation":
        data_transform = TextConversationTransform(chat_template, max_seq_len, text_keys)
    else:
        raise ValueError(f"Unsupported text data type: {data_type!r}")

    logger.debug("Built text transform: data_type=%s, transform=%s", data_type, type(data_transform).__name__)
    return data_transform
