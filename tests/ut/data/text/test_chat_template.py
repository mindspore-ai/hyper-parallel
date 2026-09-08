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
"""Golden cases for the merged chat template implementation.

``hyper_parallel.data.text.chat_template`` is the canonical merge (05
§11.3) of the former ``components/data/chat_template.py`` and
``components/datasets/llm/chat_template.py`` duplicates, which the
pre-merge golden cases proved behaviorally identical. These cases pin the
golden encodings the merge had to preserve; the final test pins the merged
module surface (single registry, shared IGNORE_INDEX, DatasetLogger).
"""
# pylint: disable=wrong-import-position

import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.data.dataset_logging import DatasetLogger
from hyper_parallel.data.text import chat_template

from tests.common.mark_utils import arg_mark
from tests.ut.data.conftest import (
    FailingSaveTokenizer,
    FakeTokenizer,
    GptOssTokenizer,
    PrefixRewritingTokenizer,
)

_MESSAGES = [
    {"role": "system", "content": "Be helpful.", "loss_mask": 0},
    {"role": "user", "content": "Hi there", "loss_mask": 0},
    {"role": "assistant", "content": "Hello!", "loss_mask": 1},
]


def _ids(text, add_special_tokens=False):
    """Golden token ids under the FakeTokenizer contract."""
    return FakeTokenizer().encode(text, add_special_tokens=add_special_tokens)


class TestChatTemplateGoldens(unittest.TestCase):
    """Golden encodings preserved by the merged chat template module."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_template_golden(self):
        """default: ``Role: content<eos>\n`` per message; loss_mask selects labels."""
        expected_ids = (
            _ids("System: Be helpful.</s>\n")
            + _ids("User: Hi there</s>\n")
            + _ids("Assistant: Hello!</s>\n")
        )
        result = chat_template.build_chat_template(
            "default", FakeTokenizer()
        ).encode_messages(_MESSAGES)
        self.assertEqual(result["input_ids"], expected_ids)
        self.assertEqual(result["attention_mask"], [1] * len(expected_ids))
        self.assertEqual(
            result["labels"],
            [-100] * (len(expected_ids) - len(_ids("Assistant: Hello!</s>\n")))
            + _ids("Assistant: Hello!</s>\n"),
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_template_requires_loss_mask(self):
        """default: a message without ``loss_mask`` raises KeyError."""
        messages = [{"role": "user", "content": "Hi"}]
        template = chat_template.build_chat_template("default", FakeTokenizer())
        with self.assertRaises(KeyError):
            template.encode_messages(messages)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_chatml_template_golden(self):
        """chatml: ``<|im_start|>role\ncontent<|im_end|>\n``; missing loss_mask defaults by role."""
        expected_segments = [
            _ids("<|im_start|>system\nBe helpful.<|im_end|>\n"),
            _ids("<|im_start|>user\nHi there<|im_end|>\n"),
            _ids("<|im_start|>assistant\nHello!<|im_end|>\n"),
        ]
        messages = [dict(message) for message in _MESSAGES]
        for message in messages:  # chatml defaults loss_mask by role
            del message["loss_mask"]
        result = chat_template.build_chat_template(
            "chatml", FakeTokenizer()
        ).encode_messages(messages)
        expected_ids = sum(expected_segments, [])
        self.assertEqual(result["input_ids"], expected_ids)
        self.assertEqual(
            result["labels"],
            [-100] * (len(expected_segments[0]) + len(expected_segments[1]))
            + expected_segments[2],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_llama2_template_golden(self):
        """llama2: ``<<SYS>>``/``[INST]`` format, tool role, unknown role raises."""
        messages = _MESSAGES + [{"role": "tool", "content": "result", "loss_mask": 0}]
        expected_ids = (
            _ids("<<SYS>>\nBe helpful.\n<</SYS>>\n\n")
            + _ids("<s>[INST] Hi there [/INST]")
            + _ids(" Hello! </s>")
            + _ids("<s>[TOOL] result [/TOOL]")
        )
        result = chat_template.build_chat_template(
            "llama2", FakeTokenizer()
        ).encode_messages(messages)
        self.assertEqual(result["input_ids"], expected_ids)
        self.assertEqual(
            result["labels"],
            [-100] * len(_ids("<<SYS>>\nBe helpful.\n<</SYS>>\n\n"))
            + [-100] * len(_ids("<s>[INST] Hi there [/INST]"))
            + _ids(" Hello! </s>")
            + [-100] * len(_ids("<s>[TOOL] result [/TOOL]")),
        )

        template = chat_template.build_chat_template("llama2", FakeTokenizer())
        with self.assertRaisesRegex(ValueError, "Unknown role"):
            template.encode_messages(
                [{"role": "narrator", "content": "x", "loss_mask": 0}]
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_janus_template_golden(self):
        """Janus: role formats, image seq/emb masks, generation task_type switch."""
        image_content = "<image_placeholder>" * 576
        messages = [
            {"role": "system", "content": "SYS", "loss_mask": 0},
            {"role": "user", "content": image_content, "loss_mask": 1},
            {"role": "assistant", "content": "A", "loss_mask": 1},
        ]
        result = chat_template.build_chat_template(
            "Janus", FakeTokenizer()
        ).encode_messages(messages)
        image_id = FakeTokenizer().vocab["<image_placeholder>"]
        # system ids include the BOS prefix; user/assistant do not.
        self.assertEqual(
            result["input_ids"],
            _ids("SYS\n\n", add_special_tokens=True)
            + _ids("User: " + image_content + "\n\n")
            + _ids("Assistant: A<｜end▁of▁sentence｜>"),
        )
        self.assertEqual(len(result["images_seq_mask"]), len(result["input_ids"]))
        self.assertEqual(result["images_emb_mask"], [[True] * 576])
        # non-generation task: only image placeholder positions keep loss in
        # the user message; the assistant message keeps full loss.
        image_positions = sum(
            1 for token in result["input_ids"] if token == image_id
        )
        self.assertEqual(image_positions, 576)
        kept = [token for token in result["labels"] if token != -100]
        self.assertEqual(
            kept,
            [image_id] * 576 + _ids("Assistant: A<｜end▁of▁sentence｜>"),
        )

        # generation task types keep full content loss instead of image-only loss
        generation = chat_template.build_chat_template(
            "Janus", FakeTokenizer()
        ).encode_messages(messages, task_type="wikihow_generation")
        user_labels = generation["labels"]
        self.assertIn(ord("U"), user_labels)  # "User: ..." prefix keeps loss

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_template_golden(self):
        """tokenizer: prefix-stable native template with assistant-only default loss."""
        messages = [dict(message) for message in _MESSAGES]
        for message in messages:
            del message["loss_mask"]
        result = chat_template.build_chat_template(
            "tokenizer", FakeTokenizer()
        ).encode_messages(messages)
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        assistant_segment = _ids("<assistant>Hello!</assistant>")
        expected_ids = _ids(rendered)
        self.assertEqual(result["input_ids"], expected_ids)
        self.assertEqual(result["attention_mask"], [1] * len(expected_ids))
        self.assertEqual(
            result["labels"],
            [-100] * (len(expected_ids) - len(assistant_segment)) + assistant_segment,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_template_rejects_prefix_rewrite(self):
        """tokenizer: structural prefix rewrites raise ValueError."""
        messages = [dict(message) for message in _MESSAGES]
        template = chat_template.build_chat_template(
            "tokenizer", PrefixRewritingTokenizer()
        )
        with self.assertRaisesRegex(ValueError, "prefix"):
            template.encode_messages(messages)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_gpt_oss_terminal_rewrite_golden(self):
        """gpt_oss: trailing ``<|return|>`` rewritten to ``<|end|>`` keeps its label."""
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        result = chat_template.build_chat_template(
            "gpt_oss", GptOssTokenizer()
        ).encode_messages(messages)
        end_id = GptOssTokenizer().vocab["<|end|>"]
        self.assertEqual(
            result["input_ids"],
            _ids("<user>q1</user>")
            + _ids("<assistant>a1<|end|>")
            + _ids("<user>q2</user>"),
        )
        # the rewritten terminal token keeps loss (not IGNORE_INDEX)
        rewritten_index = result["input_ids"].index(end_id)
        self.assertEqual(result["labels"][rewritten_index], end_id)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_max_seq_len_keeps_tail(self):
        """All templates truncate from the front, keeping the sequence tail."""
        long_messages = [
            {"role": "user", "content": "abcdefgh", "loss_mask": 0},
            {"role": "assistant", "content": "ijklmnop", "loss_mask": 1},
        ]
        for name in ("default", "chatml", "llama2"):
            template = chat_template.build_chat_template(name, FakeTokenizer())
            full = template.encode_messages(long_messages)
            truncated = template.encode_messages(long_messages, max_seq_len=8)
            self.assertEqual(truncated["input_ids"], full["input_ids"][-8:], name)
            self.assertEqual(truncated["labels"], full["labels"][-8:], name)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_save_pretrained_failure_is_swallowed(self):
        """save_pretrained logs and swallows tokenizer save failures."""
        tokenizer = FailingSaveTokenizer()
        template = chat_template.build_chat_template("default", tokenizer)
        with self.assertLogs(level="WARNING"):
            template.save_pretrained("/nonexistent-dir")
        self.assertEqual(
            tokenizer.chat_template, template.get_jinja_template()
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_merged_module_surface(self):
        """Pin how the merge resolved the recorded non-behavioral conflicts.

        * one ``CHAT_TEMPLATE_REGISTRY`` with the union of both registries;
        * ``IGNORE_INDEX`` is the shared constants import (==-100);
        * logging goes through ``DatasetLogger`` (the datasets/llm side).
        """
        self.assertEqual(chat_template.IGNORE_INDEX, -100)
        self.assertEqual(
            sorted(chat_template.CHAT_TEMPLATE_REGISTRY),
            ["Janus", "chatml", "default", "gpt_oss", "llama2", "tokenizer"],
        )
        self.assertIsInstance(chat_template.logger, DatasetLogger)


if __name__ == "__main__":
    unittest.main()
