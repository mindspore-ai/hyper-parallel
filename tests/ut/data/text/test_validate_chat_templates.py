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
"""Offline scanner tests without model downloads or accelerator initialization."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.qwen3_8.tools.validate_chat_templates import main, scan_dataset, validate_record
from hyper_parallel.data.text.build_data_transform import TextConversationTransform
from hyper_parallel.data.text.chat_template import TokenizerTemplate
from tests.ut.data.conftest import FakeTokenizer, PrefixRewritingTokenizer


class TestValidateChatTemplates(unittest.TestCase):
    """Pin full-render versus incremental-encoding failures and scan completion."""

    def setUp(self):
        self.record = {"messages": [{"role": "user", "content": "question"},
                                    {"role": "assistant", "content": "answer", "reasoning_content": ""}]}
        self.transform = TextConversationTransform(TokenizerTemplate(FakeTokenizer()), 4096, "messages", "tools")

    def test_tools_are_forwarded_and_shifted_lengths_match(self):
        """Forward normalized tools for both the full render and training encoder."""
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        self.record["tools"] = json.dumps(tools)
        tokenizer = self.transform.chat_template.tokenizer
        original = tokenizer.apply_chat_template

        def render(messages, **kwargs):
            self.assertEqual(kwargs.pop("tools"), tools)
            return original(messages, **kwargs)

        with patch.object(tokenizer, "apply_chat_template", side_effect=render) as mocked:
            result = validate_record(json.dumps(self.record), 3, self.transform)
        self.assertTrue(result["training_ok"])
        self.assertGreater(mocked.call_count, 1)
        self.assertEqual(result["line"], 3)
        self.assertEqual(result["tool_count"], 1)
        expected_tokens = len(tokenizer.encode(result["template_text"], add_special_tokens=False)) - 1
        self.assertEqual(result["train_tokens"], expected_tokens)

    def test_render_success_does_not_hide_training_prefix_failure(self):
        """A whole conversation can render while its incremental mask cannot be built."""
        transform = TextConversationTransform(TokenizerTemplate(PrefixRewritingTokenizer()), 4096, "messages")
        result = validate_record(json.dumps(self.record), 1, transform)
        self.assertTrue(result["render_ok"])
        self.assertFalse(result["training_ok"])
        self.assertEqual(result["stage"], "training_encode")
        self.assertIn("prefix", result["error"])

    def test_scan_continues_after_invalid_records_and_preserves_line_numbers(self):
        """Malformed lines do not prevent later records from being checked."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "input.jsonl"
            data.write_text("\ninvalid\n" + json.dumps(self.record) + "\n[]\n", encoding="utf-8")
            report = scan_dataset(data, root / "results", self.transform, {}, progress_every=0)
            errors = [json.loads(line) for line in (root / "results/errors.jsonl").read_text().splitlines()]
            self.assertEqual(report["checked"], 3)
            self.assertEqual(report["blank_lines"], 1)
            self.assertEqual(report["training_passed"], 1)
            self.assertEqual(report["failed"], 2)
            self.assertFalse(report["all_passed"])
            self.assertEqual([error["line"] for error in errors], [2, 4])
            with self.assertRaises(FileExistsError):
                scan_dataset(data, root / "results", self.transform, {})

    def test_empty_dataset_is_not_success(self):
        """An empty file must not produce a misleading all-passed report."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "empty.jsonl"
            data.write_text("\n", encoding="utf-8")
            report = scan_dataset(data, root / "results", self.transform, {})
            self.assertFalse(report["all_passed"])
            self.assertEqual(report["checked"], 0)

    def test_cli_returns_failure_for_bad_sample(self):
        """The executable returns a nonzero status even after finishing the entire scan."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "input.jsonl"
            data.write_text("invalid\n", encoding="utf-8")
            argv = ["validate", "--data-path", str(data), "--tokenizer-path", directory,
                    "--output-dir", str(root / "results"), "--template-kwargs", "{}"]
            with patch("sys.argv", argv), patch(
                    "examples.qwen3_8.tools.validate_chat_templates.AutoTokenizer.from_pretrained",
                    return_value=FakeTokenizer()):
                self.assertEqual(main(), 1)
