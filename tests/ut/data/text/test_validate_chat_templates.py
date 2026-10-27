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

import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from examples.qwen3_8.tools.validate_chat_templates import main, parse_args, scan_dataset, validate_record
from hyper_parallel.data.text.text_transform import TextConversationTransform
from hyper_parallel.data.text.chat_template import TokenizerTemplate
from tests.ut.data.conftest import FakeTokenizer, PrefixRewritingTokenizer


class TestValidateChatTemplates(unittest.TestCase):
    """Pin full-render versus incremental-encoding failures and scan completion."""

    def setUp(self) -> None:
        """Build a minimal conversation and its training transform."""
        self.record = {"messages": [{"role": "user", "content": "question"},
                                    {"role": "assistant", "content": "answer", "reasoning_content": ""}]}
        self.transform = TextConversationTransform(TokenizerTemplate(FakeTokenizer()), 4096, "messages", "tools")

    def test_tools_are_forwarded_and_shifted_lengths_match(self):
        """Forward normalized tools for both the full render and training encoder."""
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        self.record["tools"] = json.dumps(tools)
        tokenizer = self.transform.chat_template.tokenizer
        original = tokenizer.apply_chat_template

        def render(messages: Any, **kwargs: Any) -> Any:
            """Check tool forwarding before rendering with the fake tokenizer."""
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


class TestTemplateConfiguration(unittest.TestCase):
    """Verify YAML provenance, explicit overrides, and backward-compatible CLI defaults."""

    def setUp(self) -> None:
        """Create isolated paths accepted by the command-line parser."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.data = self.root / "input.jsonl"
        self.data.write_text("", encoding="utf-8")
        self.config = self.root / "train.yaml"
        self.argv = ["validate", "--data-path", str(self.data), "--tokenizer-path", str(self.root),
                     "--output-dir", str(self.root / "results")]

    def _parse(self, *extra: str):
        with patch("sys.argv", self.argv + list(extra)):
            return parse_args()

    def test_legacy_defaults_and_explicit_replacement(self) -> None:
        """Commands without a YAML retain their original replacement semantics."""
        self.assertEqual(self._parse().template_kwargs, {"enable_thinking": True, "reasoning_effort": "medium"})
        self.assertEqual(self._parse("--template-kwargs", "{}").template_kwargs, {})
        self.assertEqual(self._parse("--template-kwargs", '{"preserve_thinking": false}').template_kwargs,
                         {"preserve_thinking": False})

    def test_yaml_options_and_per_field_overrides(self) -> None:
        """Preserve unspecified YAML fields, including explicit false values."""
        self.config.write_text(
            "dataset:\n  model_assets:\n    chat_template_kwargs:\n"
            "      enable_thinking: true\n      preserve_thinking: true\n      reasoning_effort: medium\n",
            encoding="utf-8",
        )
        expected = {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "medium"}
        for extra in ((), ("--template-kwargs", "{}")):
            with self.subTest(extra=extra):
                self.assertEqual(self._parse("--config", str(self.config), *extra).template_kwargs, expected)
        expected["preserve_thinking"] = False
        self.assertEqual(self._parse("--config", str(self.config),
                                    "--template-kwargs", '{"preserve_thinking": false}').template_kwargs, expected)

    def test_yaml_missing_or_null_options_use_native_defaults(self) -> None:
        """A YAML without options must not inherit the legacy CLI thinking defaults."""
        for assets in ("{}", "{chat_template_kwargs: null}", "{chat_template_kwargs: {}}"):
            with self.subTest(assets=assets):
                self.config.write_text(f"dataset:\n  model_assets: {assets}\n", encoding="utf-8")
                self.assertEqual(self._parse("--config", str(self.config)).template_kwargs, {})

    def test_invalid_configuration_exits_before_scan(self) -> None:
        """Report missing files, malformed YAML, invalid mappings, and invalid JSON as CLI errors."""
        bad_configs = ("[", "[]", "dataset: []", "dataset: {model_assets: []}",
                       "dataset: {model_assets: {chat_template_kwargs: []}}",
                       "dataset: {model_assets: {chat_template_kwargs: {1: true}}}")
        for content in (None, *bad_configs):
            with self.subTest(config=content), patch("sys.stderr", new_callable=io.StringIO) as stderr:
                if content is not None:
                    self.config.write_text(content, encoding="utf-8")
                with self.assertRaises(SystemExit) as error:
                    self._parse("--config", str(self.config))
                self.assertEqual(error.exception.code, 2)
                self.assertIn("Invalid template configuration:", stderr.getvalue())
        for value in ("[", "[]", "null"):
            with self.subTest(overrides=value), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    self._parse("--template-kwargs", value)
                self.assertEqual(error.exception.code, 2)

    def test_cli_uses_yaml_options_and_records_provenance(self) -> None:
        """Pass resolved YAML fields through rendering and include them in the scan summary."""
        options = {"enable_thinking": True, "preserve_thinking": True}
        self.config.write_text(json.dumps({"dataset": {"model_assets": {"chat_template_kwargs": options}}}),
                               encoding="utf-8")
        self.data.write_text(json.dumps({"messages": [{"role": "user", "content": "question"},
                                                    {"role": "assistant", "content": "answer"}]}) + "\n",
                             encoding="utf-8")
        tokenizer = FakeTokenizer()
        original = tokenizer.apply_chat_template

        def render_with_options(messages: Any, **kwargs: Any) -> Any:
            """Consume native options not implemented by the simple fake tokenizer."""
            self.assertIs(kwargs.pop("preserve_thinking"), True)
            self.assertIs(kwargs.pop("enable_thinking"), True)
            return original(messages, **kwargs)

        with patch("sys.argv", self.argv + ["--config", str(self.config)]), patch(
                "examples.qwen3_8.tools.validate_chat_templates.AutoTokenizer.from_pretrained",
                return_value=tokenizer), patch.object(
                    tokenizer, "apply_chat_template", side_effect=render_with_options) as render:
            self.assertEqual(main(), 0)
        self.assertGreater(render.call_count, 1)
        for call in render.call_args_list:
            self.assertTrue(call.kwargs["preserve_thinking"])
            self.assertTrue(call.kwargs["enable_thinking"])
        summary = json.loads((self.root / "results/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["configuration"]["template_kwargs"], options)
        self.assertEqual(summary["configuration"]["config_path"], str(self.config.resolve()))
