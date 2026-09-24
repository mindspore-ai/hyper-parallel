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
"""Build LLM plaintext and conversation data transforms.

Merged from ``components/datasets/llm/build_data_transform.py`` and
``components/data/identity_transform.py`` (05 §11.3): the two identity
transforms collapsed into the single ``IdentityDataTransform`` below, whose
``tokenizer``/``chat_template`` keyword superset covers both legacy call
styles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from hyper_parallel.data.dataset_logging import get_dataset_logger

LLMDataType = Literal["plaintext", "conversation"]
logger = get_dataset_logger(__name__)


def _validate_field_keys(keys: str | Sequence[str], name: str) -> None:
    """Validate a configured source-field name or ordered fallback list."""
    if isinstance(keys, str):
        if not keys.strip():
            raise ValueError(f"{name} must be a non-empty string or sequence of strings")
        return
    if isinstance(keys, (bytes, bytearray)) or not isinstance(keys, Sequence) or not keys:
        raise ValueError(f"{name} must be a non-empty string or sequence of strings")
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError(f"{name} must contain only non-empty strings")


def _get_record_value(sample: Mapping[str, Any], keys: str | Sequence[str]) -> Any:
    if not isinstance(sample, Mapping):
        raise ValueError(f"Online Dataset records must be mappings, got {type(sample).__name__}")
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
    "function": "tool",
    "observation": "tool",
    "tool": "tool",
}


def _format_record_text(sample: Mapping[str, Any], text_template: str) -> str:
    """Render one source record with a Python format-style template."""
    if not isinstance(sample, Mapping):
        raise ValueError(f"Online Dataset records must be mappings, got {type(sample).__name__}")
    try:
        return text_template.format_map(sample)
    except KeyError as error:
        raise ValueError(f"Sample does not contain text_template field {error.args[0]!r}") from error
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid text_template {text_template!r}: {error}") from error


def _normalize_messages(
        messages: Any,
        *,
        role_key: str,
        content_key: str,
        role_map: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    """Convert source-specific conversation fields to role/content messages."""
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

        source_role = message[role_key]
        if not isinstance(source_role, str):
            raise ValueError(f"Conversation message role must be a string, got {type(source_role).__name__}")
        source_role = source_role.strip()
        if not source_role:
            raise ValueError("Conversation message role must not be empty")
        content = message[content_key]
        if not isinstance(content, str):
            raise ValueError(
                f"Conversation message content must be a string, got {type(content).__name__}"
            )

        normalized_role = _DEFAULT_ROLE_ALIASES.get(source_role.lower(), source_role)
        if role_map is not None:
            normalized_role = role_map.get(
                source_role,
                role_map.get(source_role.lower(), normalized_role),
            )
        normalized_message = dict(message)
        normalized_message["role"] = normalized_role
        normalized_message["content"] = content
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
    _logged_first_sample: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the tokenizer and sequence length configuration."""
        if self.tokenizer is None or not callable(getattr(self.tokenizer, "encode", None)):
            raise ValueError("tokenizer is required for plaintext data")
        if isinstance(self.max_seq_len, bool) or not isinstance(self.max_seq_len, int) or self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        _validate_field_keys(self.text_keys, "text_keys")
        if self.text_template is not None and not isinstance(self.text_template, str):
            raise ValueError("text_template must be a string or None")

    def __call__(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Tokenize and chunk one plaintext record."""
        if self.text_template is not None:
            text = _format_record_text(sample, self.text_template)
        else:
            text = _get_record_value(sample, self.text_keys)
        if not isinstance(text, str):
            raise ValueError(f"Plaintext field must be a string, got {type(text).__name__}")
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
        if not self._logged_first_sample:
            logger.debug(
                "Converted online plaintext record: template=%s, token_count=%d, output_samples=%d",
                self.text_template is not None,
                len(token_ids),
                len(transformed),
            )
            self._logged_first_sample = True
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
    _logged_first_sample: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the chat template and sequence length configuration."""
        if self.chat_template is None or not callable(getattr(self.chat_template, "encode_messages", None)):
            raise ValueError("chat_template with encode_messages is required for conversation data")
        if isinstance(self.max_seq_len, bool) or not isinstance(self.max_seq_len, int) or self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        for name, value in (("role_key", self.role_key), ("content_key", self.content_key)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        _validate_field_keys(self.text_keys, "text_keys")
        if self.role_map is not None:
            if not isinstance(self.role_map, Mapping):
                raise ValueError("role_map must be a mapping or None")
            if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.role_map.items()):
                raise ValueError("role_map keys and values must be strings")

    def __call__(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Encode one conversation record."""
        messages = _normalize_messages(
            _get_record_value(sample, self.text_keys),
            role_key=self.role_key,
            content_key=self.content_key,
            role_map=self.role_map,
        )
        encoded = self.chat_template.encode_messages(messages, max_seq_len=self.max_seq_len)
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        labels = torch.as_tensor(encoded["labels"], dtype=torch.long)
        model_sample = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if not self._logged_first_sample:
            logger.debug(
                "Converted online conversation record: roles=%s, token_count=%d, max_seq_len=%d",
                [message["role"] for message in messages],
                len(model_sample["input_ids"]),
                self.max_seq_len,
            )
            self._logged_first_sample = True
        return [model_sample]


def build_llm_data_transform(data_type: LLMDataType, *, tokenizer: Any = None, chat_template: Any = None,
                             max_seq_len: int, text_keys: str | Sequence[str] = "text",
                             text_template: str | None = None, role_key: str = "role",
                             content_key: str = "content", role_map: Mapping[str, str] | None = None) -> Callable[[Any], Any]:
    """Build the transform selected by the LLM data type.

    Args:
        data_type: Plaintext or conversation input format.
        tokenizer: Tokenizer used by plaintext transforms.
        chat_template: Chat template used by conversation transforms.
        max_seq_len: Maximum model sequence length.
        text_keys: Field or candidate fields containing the source text.
        text_template: Optional format template for Instruction/Alpaca records.
        role_key: Source conversation field containing a role.
        content_key: Source conversation field containing message content.
        role_map: Optional aliases that override built-in role normalization.

    Returns:
        The configured LLM sample transform.

    Raises:
        ValueError: If ``data_type`` is unsupported.
    """
    if data_type == "plaintext":
        data_transform = PlaintextTransform(tokenizer, max_seq_len, text_keys, text_template)
    elif data_type == "conversation":
        data_transform = TextConversationTransform(
            chat_template, max_seq_len, text_keys, role_key, content_key, role_map,
        )
    else:
        raise ValueError(f"Unsupported LLM data type: {data_type!r}")
    logger.debug("Built LLM data transform: data_type=%s, transform=%s", data_type, type(data_transform).__name__)
    return data_transform
