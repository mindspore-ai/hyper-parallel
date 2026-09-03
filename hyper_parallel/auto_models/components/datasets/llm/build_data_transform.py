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
"""Build LLM plaintext, conversation, and instruction data transforms."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from hyper_parallel.auto_models.components.datasets.dataset_logging import get_dataset_logger
from hyper_parallel.platform import get_platform

LLMDataType = Literal["plaintext", "conversation", "instruction"]
logger = get_dataset_logger(__name__)
platform = get_platform()

_SHAREGPT_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "model": "assistant",
    "bot": "assistant",
}


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


def _normalize_conversation(messages: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI and ShareGPT messages to role/content records."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("Conversation messages must be a sequence of mappings")

    normalized_messages = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("Every conversation message must be a mapping")

        if "role" in message:
            if "content" not in message:
                raise ValueError("A role/content message must contain field 'content'")
            normalized_message = dict(message)
        elif "from" in message:
            if "value" not in message:
                raise ValueError("A ShareGPT from/value message must contain field 'value'")
            source_role = message["from"]
            try:
                role = _SHAREGPT_ROLE_MAP[source_role]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Unsupported ShareGPT role: {source_role!r}") from exc
            normalized_message = {
                "role": role,
                "content": message["value"],
            }
            if "loss_mask" in message:
                normalized_message["loss_mask"] = message["loss_mask"]
        else:
            raise ValueError("A conversation message must use role/content or from/value fields")

        normalized_messages.append(normalized_message)

    return normalized_messages


def _encode_messages(chat_template: Any, messages: Sequence[Mapping[str, Any]], max_seq_len: int) -> dict[str, Any]:
    """Encode normalized messages into a shifted causal-language-model sample."""
    encoded = chat_template.encode_messages(messages, max_seq_len=max_seq_len)
    input_ids = platform.tensor(encoded["input_ids"], dtype=platform.tensor_dtype.int64)
    labels = platform.tensor(encoded["labels"], dtype=platform.tensor_dtype.int64)
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

    def __call__(self, sample: Any) -> Any:
        """Return the input sample without modification."""
        return sample


@dataclass
class PlaintextTransform:
    """Tokenize plaintext records into one or more model samples."""

    tokenizer: Any
    max_seq_len: int
    text_keys: str | Sequence[str] = "text"

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
            text = platform.tensor(
                token_ids[start:start + self.max_seq_len + 1],
                dtype=platform.tensor_dtype.int64,
            )
            model_sample = {
                "input_ids": text[:-1],
                "labels": text[1:],
            }
            transformed.append(model_sample)
        return transformed


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
        messages = _normalize_conversation(_get_record_value(sample, self.text_keys))
        return [_encode_messages(self.chat_template, messages, self.max_seq_len)]


@dataclass
class TextInstructionTransform:
    """Convert column-mapped instruction records into conversation samples."""

    chat_template: Any
    max_seq_len: int
    column_mapping: Mapping[str, str]
    input_separator: str = "\n\n"

    def __post_init__(self) -> None:
        """Validate chat assets and logical-to-source column mappings."""
        if self.chat_template is None:
            raise ValueError("chat_template is required for instruction data")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not isinstance(self.column_mapping, Mapping):
            raise ValueError("column_mapping must be a mapping")
        for required_key in ("instruction", "output"):
            source_key = self.column_mapping.get(required_key)
            if not isinstance(source_key, str) or not source_key:
                raise ValueError(f"column_mapping must define a non-empty {required_key!r} source field")
        input_key = self.column_mapping.get("input")
        if input_key is not None and (not isinstance(input_key, str) or not input_key):
            raise ValueError("column_mapping 'input' source field must be a non-empty string")
        if not isinstance(self.input_separator, str):
            raise ValueError("input_separator must be a string")

    def _get_required_text(self, sample: Mapping[str, Any], logical_key: str) -> str:
        """Return one required string field selected by the column mapping."""
        source_key = self.column_mapping[logical_key]
        try:
            value = sample[source_key]
        except KeyError as exc:
            raise ValueError(f"Instruction sample does not contain field {source_key!r}") from exc
        if not isinstance(value, str):
            raise ValueError(f"Instruction field {source_key!r} must be a string")
        if not value:
            raise ValueError(f"Instruction field {source_key!r} must not be empty")
        return value

    def __call__(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Convert and encode one column-mapped instruction record."""
        instruction = self._get_required_text(sample, "instruction")
        output = self._get_required_text(sample, "output")

        input_text = ""
        input_key = self.column_mapping.get("input")
        if input_key is not None:
            input_text = sample.get(input_key, "")
            if not isinstance(input_text, str):
                raise ValueError(f"Instruction field {input_key!r} must be a string")

        user_content = instruction
        if input_text:
            user_content = f"{instruction}{self.input_separator}{input_text}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        return [_encode_messages(self.chat_template, messages, self.max_seq_len)]


def build_llm_data_transform(
        data_type: LLMDataType,
        *,
        tokenizer: Any = None,
        chat_template: Any = None,
        max_seq_len: int,
        text_keys: str | Sequence[str] = "text",
        column_mapping: Mapping[str, str] | None = None,
        input_separator: str = "\n\n",
) -> Callable[[Any], Any]:
    """Build the transform selected by the LLM data type.

    Args:
        data_type: Plaintext, conversation, or column-mapped instruction format.
        tokenizer: Tokenizer used by plaintext transforms.
        chat_template: Chat template used by conversation transforms.
        max_seq_len: Maximum model sequence length.
        text_keys: Field or candidate fields containing the source text.
        column_mapping: Logical instruction/input/output fields mapped to source columns.
        input_separator: Separator inserted between non-empty instruction and input fields.

    Returns:
        The configured LLM sample transform.

    Raises:
        ValueError: If ``data_type`` is unsupported.
    """
    if data_type == "plaintext":
        data_transform = PlaintextTransform(tokenizer, max_seq_len, text_keys)
    elif data_type == "conversation":
        data_transform = TextConversationTransform(chat_template, max_seq_len, text_keys)
    elif data_type == "instruction":
        if column_mapping is None:
            raise ValueError("column_mapping is required for instruction data")
        data_transform = TextInstructionTransform(
            chat_template,
            max_seq_len,
            column_mapping,
            input_separator,
        )
    else:
        raise ValueError(f"Unsupported LLM data type: {data_type!r}")
    logger.debug("Built LLM data transform: data_type=%s, transform=%s", data_type, type(data_transform).__name__)
    return data_transform
