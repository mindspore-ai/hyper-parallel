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
"""Normalize common supervised fine-tuning records for offline encoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any


_DEFAULT_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
}


def parse_role_map(value: str) -> dict[str, str]:
    """Parse a JSON role alias map supplied by the command line."""
    try:
        role_map = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("role_map must be a JSON object") from error
    if not isinstance(role_map, dict) or any(
        not isinstance(key, str) or not isinstance(mapped_role, str)
        for key, mapped_role in role_map.items()
    ):
        raise ValueError("role_map must map string role names to string role names")
    return role_map


@dataclass(frozen=True)
class OfflineRecordTransform:
    """Render source records into one plaintext field for indexed datasets."""

    output_key: str
    text_template: str | None = None
    conversation_key: str | None = None
    role_key: str = "role"
    content_key: str = "content"
    role_map: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output_key, str) or not self.output_key:
            raise ValueError("output_key must be a non-empty string")
        if self.text_template is not None and self.conversation_key is not None:
            raise ValueError("text_template and conversation_key cannot be configured together")
        if not isinstance(self.role_key, str) or not self.role_key:
            raise ValueError("role_key must be a non-empty string")
        if not isinstance(self.content_key, str) or not self.content_key:
            raise ValueError("content_key must be a non-empty string")

    @property
    def enabled(self) -> bool:
        """Return whether source schema conversion is configured."""
        return self.text_template is not None or self.conversation_key is not None

    def __call__(self, record: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
        """Convert one source record while preserving unrelated metadata fields."""
        if not isinstance(record, Mapping):
            raise ValueError("Offline source record must be a JSON object")
        normalized = dict(record)
        if self.text_template is not None:
            normalized[self.output_key] = self._format_text(record)
        elif self.conversation_key is not None:
            normalized[self.output_key] = self._render_conversation(record, tokenizer)
        return normalized

    def _format_text(self, record: Mapping[str, Any]) -> str:
        try:
            return self.text_template.format_map(record)
        except KeyError as error:
            raise ValueError(
                f"Record does not contain text_template field {error.args[0]!r}"
            ) from error
        except ValueError as error:
            raise ValueError(f"Invalid text_template {self.text_template!r}: {error}") from error

    def _render_conversation(self, record: Mapping[str, Any], tokenizer: Any) -> str:
        try:
            source_messages = record[self.conversation_key]
        except KeyError as error:
            raise ValueError(
                f"Record does not contain conversation field {self.conversation_key!r}"
            ) from error
        messages = self._normalize_messages(source_messages)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("conversation conversion requires a tokenizer with apply_chat_template")
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "conversation conversion requires a valid tokenizer chat_template"
            ) from error
        if not isinstance(text, str):
            raise ValueError("tokenizer.apply_chat_template(..., tokenize=False) must return text")
        return text

    def _normalize_messages(self, source_messages: Any) -> list[dict[str, str]]:
        if not isinstance(source_messages, Sequence) or isinstance(source_messages, (str, bytes)):
            raise ValueError("Conversation field must be a sequence of message mappings")
        messages = []
        for source_message in source_messages:
            if not isinstance(source_message, Mapping):
                raise ValueError("Every conversation message must be a mapping")
            if self.role_key not in source_message:
                raise ValueError(f"Conversation message does not contain role field {self.role_key!r}")
            if self.content_key not in source_message:
                raise ValueError(
                    f"Conversation message does not contain content field {self.content_key!r}"
                )
            role = str(source_message[self.role_key]).strip()
            if not role:
                raise ValueError("Conversation message role must not be empty")
            canonical_role = _DEFAULT_ROLE_ALIASES.get(role.lower(), role)
            if self.role_map is not None:
                canonical_role = self.role_map.get(
                    role,
                    self.role_map.get(role.lower(), canonical_role),
                )
            content = source_message[self.content_key]
            if not isinstance(content, str):
                raise ValueError("Offline conversation message content must be a string")
            messages.append({"role": canonical_role, "content": content})
        return messages


__all__ = ["OfflineRecordTransform", "parse_role_map"]
