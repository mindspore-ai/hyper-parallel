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
"""Unit tests for LLM data transforms."""

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from hyper_parallel.auto_models.components.datasets.llm.build_data_transform import (
    TextConversationTransform,
    TextInstructionTransform,
    build_llm_data_transform,
)


class _RecordingChatTemplate:
    """Encode message fields deterministically while recording the input."""

    def __init__(self) -> None:
        """Initialize the recorded message list."""
        self.messages = None

    def encode_messages(
            self,
            messages: Sequence[Mapping[str, Any]],
            max_seq_len: int,
    ) -> dict[str, list[int]]:
        """Return deterministic IDs and assistant-only labels."""
        self.messages = [dict(message) for message in messages]
        input_ids = []
        labels = []
        role_ids = {"system": 1, "user": 2, "assistant": 3}
        for message in messages:
            content = message["content"]
            token_ids = [role_ids[message["role"]], len(content), sum(ord(char) for char in content) % 997]
            input_ids.extend(token_ids)
            loss_mask = message.get("loss_mask", 1 if message["role"] == "assistant" else 0)
            labels.extend(token_ids if loss_mask else [-100] * len(token_ids))
        return {
            "input_ids": input_ids[-max_seq_len:],
            "labels": labels[-max_seq_len:],
        }


class TestTextConversationTransform(unittest.TestCase):
    """Test OpenAI and ShareGPT conversation normalization."""

    def test_openai_messages_remain_unchanged(self):
        """Preserve standard role/content messages and optional loss masks."""
        chat_template = _RecordingChatTemplate()
        transform = TextConversationTransform(chat_template, max_seq_len=32, text_keys="messages")
        messages = [
            {"role": "user", "content": "Question", "loss_mask": 0},
            {"role": "assistant", "content": "Answer", "loss_mask": 1},
        ]

        result = transform({"messages": messages})

        self.assertEqual(chat_template.messages, messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["input_ids"].dtype, torch.long)
        self.assertEqual(result[0]["labels"].dtype, torch.long)

    def test_sharegpt_messages_map_roles_and_content(self):
        """Convert common ShareGPT from/value turns to role/content messages."""
        chat_template = _RecordingChatTemplate()
        transform = TextConversationTransform(chat_template, max_seq_len=32, text_keys="conversations")

        transform(
            {
                "conversations": [
                    {"from": "human", "value": "Question"},
                    {"from": "gpt", "value": "Answer"},
                ]
            }
        )

        self.assertEqual(
            chat_template.messages,
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
        )

    def test_invalid_conversation_shapes_raise_value_error(self):
        """Reject unsupported message containers, fields, and ShareGPT roles."""
        transform = TextConversationTransform(_RecordingChatTemplate(), max_seq_len=32)
        invalid_messages = [
            "not-a-list",
            ["not-a-mapping"],
            [{"role": "user"}],
            [{"from": "human"}],
            [{"from": "function_call", "value": "{}"}],
            [{"text": "missing role"}],
        ]

        for messages in invalid_messages:
            with self.subTest(messages=messages):
                with self.assertRaises(ValueError):
                    transform({"conversation": messages})


class TestTextInstructionTransform(unittest.TestCase):
    """Test column-mapped instruction conversion and encoding."""

    def test_instruction_matches_equivalent_conversation(self):
        """Produce identical tensors for instruction and canonical conversation inputs."""
        instruction_template = _RecordingChatTemplate()
        instruction_transform = TextInstructionTransform(
            instruction_template,
            max_seq_len=32,
            column_mapping={
                "instruction": "prompt",
                "input": "context",
                "output": "answer",
            },
        )
        conversation_template = _RecordingChatTemplate()
        conversation_transform = TextConversationTransform(
            conversation_template,
            max_seq_len=32,
            text_keys="messages",
        )

        instruction_result = instruction_transform(
            {"prompt": "Explain TP", "context": "Use two ranks", "answer": "Shard matrices."}
        )
        conversation_result = conversation_transform(
            {
                "messages": [
                    {"role": "user", "content": "Explain TP\n\nUse two ranks"},
                    {"role": "assistant", "content": "Shard matrices."},
                ]
            }
        )

        self.assertTrue(torch.equal(instruction_result[0]["input_ids"], conversation_result[0]["input_ids"]))
        self.assertTrue(torch.equal(instruction_result[0]["labels"], conversation_result[0]["labels"]))

    def test_empty_or_missing_optional_input_uses_instruction_only(self):
        """Avoid appending a separator when the optional input is empty or absent."""
        for sample in (
            {"instruction": "Question", "input": "", "output": "Answer"},
            {"instruction": "Question", "output": "Answer"},
        ):
            with self.subTest(sample=sample):
                chat_template = _RecordingChatTemplate()
                transform = TextInstructionTransform(
                    chat_template,
                    max_seq_len=32,
                    column_mapping={
                        "instruction": "instruction",
                        "input": "input",
                        "output": "output",
                    },
                )

                transform(sample)

                self.assertEqual(chat_template.messages[0]["content"], "Question")

    def test_invalid_instruction_configuration_and_samples_raise_value_error(self):
        """Reject missing mappings, required values, and non-string input fields."""
        with self.assertRaises(ValueError):
            build_llm_data_transform(
                "instruction",
                chat_template=_RecordingChatTemplate(),
                max_seq_len=32,
            )

        transform = TextInstructionTransform(
            _RecordingChatTemplate(),
            max_seq_len=32,
            column_mapping={"instruction": "instruction", "input": "input", "output": "output"},
        )
        invalid_samples = [
            {"output": "Answer"},
            {"instruction": "", "output": "Answer"},
            {"instruction": "Question", "output": ""},
            {"instruction": "Question", "input": 1, "output": "Answer"},
        ]

        for sample in invalid_samples:
            with self.subTest(sample=sample):
                with self.assertRaises(ValueError):
                    transform(sample)


if __name__ == "__main__":
    unittest.main()
