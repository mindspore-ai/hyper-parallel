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
