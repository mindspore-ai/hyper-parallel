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

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.online.format import InstructionMessages, normalize_conversation

TextDataType = Literal["plaintext", "sharegpt", "alpaca"]
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


def _normalize_tools(tools: Any, tools_key: str) -> list[dict[str, Any]] | None:
    """Normalize an explicitly configured tools field for native chat templates."""
    if tools is None or tools is False:
        return None
    if isinstance(tools, str):
        if not tools:
            return None
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tools field {tools_key!r} must contain valid JSON") from exc
        if tools is None or tools is False:
            return None
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        raise ValueError(f"Tools field {tools_key!r} must be a sequence of mappings")

    normalized_tools = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            raise ValueError(f"Every entry in tools field {tools_key!r} must be a mapping")
        normalized_tools.append(dict(tool))
    return normalized_tools or None


def _encode_messages(
        chat_template: Any,
        messages: Sequence[Mapping[str, Any]],
        max_seq_len: int,
        tools: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Encode normalized messages into a shifted causal-language-model sample."""
    encode_kwargs = {"max_seq_len": max_seq_len}
    if tools:
        encode_kwargs["tools"] = tools
    encoded = chat_template.encode_messages(messages, **encode_kwargs)
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.int64)
    labels = torch.tensor(encoded["labels"], dtype=torch.int64)
    return {
        "input_ids": input_ids[:-1],
        "labels": labels[1:],
    }


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
    tools_key: str | None = None
    message_formatter: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None = None

    def __post_init__(self) -> None:
        """Validate the chat template and sequence length configuration."""
        if self.chat_template is None:
            raise ValueError("chat_template is required for conversation data")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.tools_key is not None and (not isinstance(self.tools_key, str) or not self.tools_key):
            raise ValueError("tools_key must be a non-empty string or None")

    def __call__(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Encode one conversation record."""
        messages = (self.message_formatter(sample) if self.message_formatter is not None
                    else normalize_conversation(_get_record_value(sample, self.text_keys)))
        tools = None
        if self.tools_key is not None:
            tools = _normalize_tools(sample.get(self.tools_key), self.tools_key)
        encoded = _encode_messages(self.chat_template, messages, self.max_seq_len, tools)
        return [encoded] if bool(encoded["labels"].ne(IGNORE_INDEX).any()) else []

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Return whether one source record contains conversation messages."""
        messages = (self.message_formatter(sample) if self.message_formatter is not None
                    else _get_record_value(sample, self.text_keys))
        is_valid = bool(messages)
        return is_valid


def build_text_transform(
        data_type: TextDataType,
        *,
        tokenizer: Any = None,
        chat_template: Any = None,
        max_seq_len: int,
        text_keys: str | Sequence[str] | None = None,
        drop_ragged_tail: bool = False,
        tools_key: str | None = None,
        column_mapping: Mapping[str, str] | None = None,
        input_separator: str = "\n\n",
) -> Callable[[Any], Any]:
    """Build the transform selected by the text dataset format.

    Args:
        data_type: One of plaintext, sharegpt, or alpaca.
        tokenizer: Tokenizer used by plaintext transforms.
        chat_template: Chat template used by conversation transforms.
        max_seq_len: Maximum model sequence length.
        text_keys: Source field(s); defaults to text or conversations by format.
        drop_ragged_tail: Drop trailing partial plaintext windows.
        tools_key: Optional field containing tool definitions for conversation data.
        column_mapping: Logical instruction/input/output fields mapped to source columns.
        input_separator: Separator inserted between non-empty instruction and input fields.

    Returns:
        The configured text sample transform.

    Raises:
        ValueError: If ``data_type`` is unsupported.
    """
    if text_keys is None:
        text_keys = {"sharegpt": "conversations"}.get(data_type, "text")
    if data_type == "plaintext":
        if tools_key is not None:
            raise ValueError("tools_key is only supported for sharegpt data")
        data_transform = PlaintextTransform(tokenizer, max_seq_len, text_keys, drop_ragged_tail)
    elif data_type == "sharegpt":
        data_transform = TextConversationTransform(chat_template, max_seq_len, text_keys, tools_key)
    elif data_type == "alpaca":
        if tools_key is not None:
            raise ValueError("tools_key is only supported for sharegpt data")
        if column_mapping is None:
            raise ValueError("column_mapping is required for alpaca data")
        data_transform = TextConversationTransform(
            chat_template,
            max_seq_len,
            message_formatter=InstructionMessages(column_mapping, input_separator),
        )
    else:
        raise ValueError(f"Unsupported text data format: {data_type!r}")

    logger.debug("Built text transform: data_type=%s, transform=%s", data_type, type(data_transform).__name__)
    return data_transform
