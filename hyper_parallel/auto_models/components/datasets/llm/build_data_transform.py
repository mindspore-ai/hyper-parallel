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
"""Build LLM plaintext and conversation data transforms."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from hyper_parallel.auto_models.components.datasets.dataset_logging import get_dataset_logger

LLMDataType = Literal["plaintext", "conversation"]
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


_DEFAULT_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
}


def _format_record_text(sample: Mapping[str, Any], text_template: str) -> str:
    """Render a record with a top-level Python format-style template."""
    try:
        return text_template.format_map(sample)
    except KeyError as error:
        raise ValueError(
            f"Sample does not contain text_template field {error.args[0]!r}"
        ) from error
    except ValueError as error:
        raise ValueError(f"Invalid text_template {text_template!r}: {error}") from error


def _normalize_messages(
        messages: Any,
        *,
        role_key: str,
        content_key: str,
        role_map: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    """Convert configured conversation fields into the chat-template contract."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("Conversation field must be a sequence of message mappings")

    normalized_messages = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("Every conversation message must be a mapping")
        if role_key not in message:
            raise ValueError(f"Conversation message does not contain role field {role_key!r}")
        if content_key not in message:
            raise ValueError(f"Conversation message does not contain content field {content_key!r}")

        source_role = str(message[role_key]).strip()
        if not source_role:
            raise ValueError("Conversation message role must not be empty")
        normalized_role = _DEFAULT_ROLE_ALIASES.get(source_role.lower(), source_role)
        if role_map is not None:
            normalized_role = role_map.get(
                source_role,
                role_map.get(source_role.lower(), normalized_role),
            )
        normalized_message = dict(message)
        normalized_message["role"] = normalized_role
        normalized_message["content"] = message[content_key]
        normalized_messages.append(normalized_message)
    return normalized_messages


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

    def __call__(self, sample: Any) -> Any:
        """Return the input sample without modification."""
        return sample


@dataclass
class PlaintextTransform:
    """Tokenize plaintext records into one or more model samples."""

    tokenizer: Any
    max_seq_len: int
    text_keys: str | Sequence[str] = "text"
    text_template: str | None = None

    def __post_init__(self) -> None:
        """Validate the tokenizer and sequence length configuration."""
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for plaintext data")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    def __call__(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Tokenize and chunk one plaintext record."""
        if self.text_template is not None:
            text = _format_record_text(sample, self.text_template)
        else:
            text = _get_record_value(sample, self.text_keys)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            token_ids = [*token_ids, eos_token_id]

        transformed = []
        for start in range(0, len(token_ids), self.max_seq_len):
            input_ids = torch.tensor(token_ids[start:start + self.max_seq_len], dtype=torch.long)
            model_sample = {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
            }
            transformed.append(model_sample)
        return transformed


@dataclass
class TextConversationTransform:
    """Encode conversation records with a configured chat template."""

    chat_template: Any
    max_seq_len: int
    text_keys: str | Sequence[str] = "conversation"
    role_key: str = "role"
    content_key: str = "content"
    role_map: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate the chat template and sequence length configuration."""
        if self.chat_template is None:
            raise ValueError("chat_template is required for conversation data")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    def __call__(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Encode one conversation record."""
        messages = _get_record_value(sample, self.text_keys)
        messages = _normalize_messages(
            messages,
            role_key=self.role_key,
            content_key=self.content_key,
            role_map=self.role_map,
        )
        encoded = self.chat_template.encode_messages(messages, max_seq_len=self.max_seq_len)
        model_sample = {
            "input_ids": torch.as_tensor(encoded["input_ids"], dtype=torch.long),
            "labels": torch.as_tensor(encoded["labels"], dtype=torch.long),
        }
        return [model_sample]


def build_llm_data_transform(data_type: LLMDataType, *, tokenizer: Any = None, chat_template: Any = None,
                             max_seq_len: int, text_keys: str | Sequence[str] = "text",
                             text_template: str | None = None,
                             role_key: str = "role",
                             content_key: str = "content",
                             role_map: Mapping[str, str] | None = None) -> Callable[[Any], Any]:
    """Build the transform selected by the LLM data type.

    Args:
        data_type: Plaintext or conversation input format.
        tokenizer: Tokenizer used by plaintext transforms.
        chat_template: Chat template used by conversation transforms.
        max_seq_len: Maximum model sequence length.
        text_keys: Field or candidate fields containing the source text.
        text_template: Optional Python format template for plaintext records.
        role_key: Source message field containing a conversation role.
        content_key: Source message field containing conversation content.
        role_map: Optional aliases that override the built-in role normalization.

    Returns:
        The configured LLM sample transform.

    Raises:
        ValueError: If ``data_type`` is unsupported.
    """
    if data_type == "plaintext":
        data_transform = PlaintextTransform(tokenizer, max_seq_len, text_keys, text_template)
    elif data_type == "conversation":
        data_transform = TextConversationTransform(
            chat_template,
            max_seq_len,
            text_keys,
            role_key,
            content_key,
            role_map,
        )
    else:
        raise ValueError(f"Unsupported LLM data type: {data_type!r}")
    logger.debug("Built LLM data transform: data_type=%s, transform=%s", data_type, type(data_transform).__name__)
    return data_transform
