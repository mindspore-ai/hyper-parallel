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
"""Tests for the Online text format conversion boundary."""

import pytest
import torch

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.text.build_data_transform import (
    PlaintextTransform,
    TextConversationTransform,
    build_llm_data_transform,
)


class _Tokenizer:
    """Minimal tokenizer implementing the Hugging Face encode contract."""

    eos_token_id = 9

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode whitespace-delimited integers."""
        assert add_special_tokens is False
        return [int(token) for token in text.split()]


class _RecordingChatTemplate:
    """Capture normalized messages passed to the chat-template boundary."""

    def __init__(self) -> None:
        self.messages = None

    def encode_messages(self, messages, max_seq_len=8192):
        """Return a deterministic encoded sample for transform tests."""
        del max_seq_len
        self.messages = messages
        return {"input_ids": [1, 2], "labels": [IGNORE_INDEX, 2]}


def test_instruction_template_renders_before_tokenization() -> None:
    """Render Alpaca fields before using the regular plaintext tokenizer path."""
    transform = build_llm_data_transform(
        "plaintext",
        tokenizer=_Tokenizer(),
        max_seq_len=16,
        text_template="{instruction} {input} {output}",
    )

    sample = transform({"instruction": "1", "input": "2", "output": "3"})[0]
    assert sample["input_ids"].tolist() == [1, 2, 3, 9]
    assert sample["labels"].tolist() == [1, 2, 3, 9]


def test_plaintext_transform_chunks_records() -> None:
    """Split long rendered records into bounded model samples."""
    transform = PlaintextTransform(_Tokenizer(), max_seq_len=3)
    samples = transform({"text": "1 2 3 4 5 6 7"})

    assert [sample["input_ids"].tolist() for sample in samples] == [
        [1, 2, 3], [4, 5, 6], [7, 9],
    ]


def test_plaintext_transform_validates_template_fields() -> None:
    """Report the missing source field at the transform boundary."""
    transform = PlaintextTransform(
        _Tokenizer(), max_seq_len=16, text_template="{instruction} {input} {output}"
    )

    with pytest.raises(ValueError, match="text_template field 'input'"):
        transform({"instruction": "1", "output": "3"})


def test_plaintext_transform_validates_source_type() -> None:
    """Reject non-string plaintext records before tokenizer invocation."""
    transform = PlaintextTransform(_Tokenizer(), max_seq_len=16)

    with pytest.raises(ValueError, match="Plaintext field must be a string"):
        transform({"text": 123})


@pytest.mark.parametrize("max_seq_len", [True, 0, -1])
def test_plaintext_transform_requires_positive_integer_sequence_length(max_seq_len) -> None:
    """Reject invalid sequence lengths during construction."""
    with pytest.raises(ValueError, match="positive integer"):
        PlaintextTransform(_Tokenizer(), max_seq_len=max_seq_len)


def test_sharegpt_roles_are_normalized_before_chat_template() -> None:
    """Map ShareGPT from/value records to the standard chat contract."""
    chat_template = _RecordingChatTemplate()
    transform = build_llm_data_transform(
        "conversation",
        chat_template=chat_template,
        max_seq_len=128,
        text_keys="conversations",
        role_key="from",
        content_key="value",
    )

    sample = transform({
        "conversations": [
            {"from": "system", "value": "rules"},
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "world"},
        ]
    })[0]

    assert [message["role"] for message in chat_template.messages] == [
        "system", "user", "assistant",
    ]
    assert sample["input_ids"].dtype == torch.long


def test_conversation_transform_supports_custom_role_aliases() -> None:
    """Allow a source-specific role map while preserving message metadata."""
    chat_template = _RecordingChatTemplate()
    transform = TextConversationTransform(
        chat_template,
        max_seq_len=128,
        text_keys="messages",
        role_map={"customer": "user"},
    )

    transform({
        "messages": [
            {"role": "customer", "content": "hello", "metadata": "keep"},
            {"role": "assistant", "content": "world"},
        ]
    })

    assert chat_template.messages[0] == {
        "role": "user", "content": "hello", "metadata": "keep",
    }


def test_conversation_transform_normalizes_tool_aliases() -> None:
    """Normalize function-call roles to the supported tool role."""
    chat_template = _RecordingChatTemplate()
    transform = TextConversationTransform(
        chat_template,
        max_seq_len=128,
        text_keys="conversations",
        role_key="from",
        content_key="value",
    )

    transform({
        "conversations": [
            {"from": "function", "value": "lookup"},
            {"from": "observation", "value": "done"},
        ]
    })

    assert [message["role"] for message in chat_template.messages] == ["tool", "tool"]


def test_conversation_transform_validates_message_content() -> None:
    """Reject malformed ShareGPT content before invoking the template."""
    transform = TextConversationTransform(_RecordingChatTemplate(), max_seq_len=16)

    with pytest.raises(ValueError, match="content must be a string"):
        transform({"conversation": [{"role": "user", "content": ["hello"]}]})
