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
"""Regression tests for shared conversation format normalization."""

import copy
import unittest

from hyper_parallel.data.constants import ONLINE_SOURCE_PATH_KEY
from hyper_parallel.data.online.format import InstructionMessages, convert_sharegpt_to_messages, normalize_conversation


class TestConversationFormat(unittest.TestCase):
    """Keep text and media conversion contracts independent of tokenization."""

    def test_instruction_messages_preserve_fields_and_separator(self):
        """Feature: Instruction formatting.

        Description: Convert mapped fields without tokenization or mutation.
        Expectation: Preserve input semantics and reject invalid configuration.
        """
        formatter = InstructionMessages({"instruction": "q", "input": "ctx", "output": "a"}, " | ")
        sample = {"q": "Question", "ctx": "Context", "a": "Answer"}
        original = copy.deepcopy(sample)
        self.assertEqual(formatter(sample), [
            {"role": "user", "content": "Question | Context"},
            {"role": "assistant", "content": "Answer"},
        ])
        self.assertEqual(sample, original)
        for mapping in (None, {}, {"instruction": "q", "output": ""},
                        {"instruction": "q", "output": "a", "input": 1}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                InstructionMessages(mapping)
        with self.assertRaises(ValueError):
            InstructionMessages({"instruction": "q", "output": "a"}, None)

    def test_text_aliases_and_loss_masks(self):
        """Feature: Shared role normalization.

        Description: Normalize all supported text ShareGPT aliases.
        Expectation: Roles and explicit loss masks survive without changing input.
        """
        roles = {"human": "user", "user": "user", "gpt": "assistant",
                 "assistant": "assistant", "chatgpt": "assistant", "model": "assistant",
                 "bot": "assistant", "system": "system"}
        for source, target in roles.items():
            with self.subTest(source=source):
                sample = [{"from": source, "value": "text", "loss_mask": 0}]
                before = copy.deepcopy(sample)
                result = normalize_conversation(sample)
                self.assertEqual(result, [{"role": target, "content": "text", "loss_mask": 0}])
                self.assertEqual(sample, before)

    def test_openai_tool_and_media_fields(self):
        """Feature: Native message preservation.

        Description: Normalize a tool-only assistant and typed media content.
        Expectation: Tool fields and media remain intact; only absent content is filled.
        """
        calls = [{"id": "call1", "type": "function", "function": {"name": "lookup"}}]
        sample = [{"role": "assistant", "tool_calls": calls, "loss_mask": 1},
                  {"role": "tool", "tool_call_id": "call1", "content": "result"},
                  {"role": "user", "content": [{"type": "image", "path": "a.png"}]}]
        before = copy.deepcopy(sample)
        result = normalize_conversation(sample)
        self.assertEqual(result[0], dict(sample[0], content=""))
        self.assertEqual(result[1:], sample[1:])
        self.assertEqual(sample, before)
        self.assertIsNot(result[0], sample[0])

    def test_media_sharegpt_preserves_mask_and_order(self):
        """Feature: Shared normalization inside media conversion.

        Description: Convert image/video placeholders and an explicitly masked reply.
        Expectation: Preserve source, mask, resolved media order and surrounding text.
        """
        sample = {"conversations": [
            {"from": "human", "value": "before<image>between<video>after"},
            {"from": "gpt", "value": "reply", "loss_mask": 0}],
            "images": ["a.png"], "videos": ["b.mp4"],
            ONLINE_SOURCE_PATH_KEY: "/data/train.jsonl"}
        before = copy.deepcopy(sample)
        result = convert_sharegpt_to_messages(sample)
        self.assertEqual(result[0], {"role": "user", "content": [
            {"type": "text", "text": "before"}, {"type": "image", "url": "/data/a.png"},
            {"type": "text", "text": "between"}, {"type": "video", "url": "/data/b.mp4"},
            {"type": "text", "text": "after"}]})
        self.assertEqual(result[1], {"role": "assistant", "content": "reply", "loss_mask": 0})
        self.assertEqual(sample, before)

    def test_invalid_messages_and_media_contract(self):
        """Feature: Format boundary validation.

        Description: Supply malformed text messages and invalid media records.
        Expectation: Reject them without weakening the fixed ShareGPT media contract.
        """
        for messages in ("bad", [1], [{"role": "user"}], [{"from": "human"}],
                         [{"from": "unknown", "value": "text"}], [{"value": "text"}]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                normalize_conversation(messages)
        invalid = [
            {"conversations": [{"from": "bot", "value": "reply"}]},
            {"conversations": [{"from": "human", "value": 1}]},
            {"conversations": [{"from": "human", "value": "<image>"}]},
            {"conversations": [{"from": "human", "value": "text"}], "images": ["unused.png"]},
        ]
        for sample in invalid:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                convert_sharegpt_to_messages(sample)
