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
"""Offline request validation, planning and atomic publication contracts."""

import ast
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_parallel.infer import GenerationOptions, GenerationRuntime
from hyper_parallel.predict import Predictor
from hyper_parallel.predict import predictor as predictor_module
from hyper_parallel.predict.batching import EncodedRequest, encode_requests, plan_batches
from hyper_parallel.predict.io import PredictionOutput, file_digest, read_requests
from hyper_parallel.predict.types import PredictionRequest
from tests.ut.infer.test_generation import IncrementModel


class Tokenizer:
    """Encode numeric prompts with a visible generation prefix for chat tests."""

    eos_token_id = pad_token_id = 6
    name_or_path = "test-tokenizer"

    def encode(self, text: str, **kwargs: object) -> list[int]:
        """Map whitespace-separated integers to token IDs."""
        del kwargs
        return [int(word) for word in text.split()]

    def apply_chat_template(self, messages: list[dict], **kwargs: object) -> list[int]:
        """Represent history length and require an assistant generation prefix."""
        if not kwargs["add_generation_prompt"]:
            raise AssertionError("generation prompt must be enabled")
        return [len(messages), 3]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Render token IDs while optionally removing the special token."""
        return " ".join(str(token) for token in tokens if not skip_special_tokens or token != 6)

    def get_vocab(self) -> dict[str, int]:
        """Expose the vocabulary used in prediction provenance."""
        return {str(i): i for i in range(10)}


class TestPredictor(unittest.TestCase):
    """End-to-end leaf-package tests use no Trainer or distributed setup."""

    def setUp(self) -> None:
        """Prepare isolated input and output paths for each test."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.source = self.root / "input.jsonl"
        self.output = self.root / "output.jsonl"
        self.config = SimpleNamespace(input=str(self.source), output_file=str(self.output), batch_size=2,
                                      max_batch_tokens=6, bucket_by_length=True, overwrite=False)

    def write_input(self, values: list[dict]) -> None:
        """Write test request records as JSONL."""
        self.source.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    def predictor(self, **kwargs: object) -> Predictor:
        """Construct a leaf predictor with deterministic model and tokenizer."""
        return Predictor(IncrementModel(), Tokenizer(), GenerationOptions(max_new_tokens=4),
                         GenerationRuntime(), self.config, **kwargs)

    def test_stale_lock_reports_recovery_guidance(self):
        """A leftover lock from a crashed run must explain how to recover."""
        first = PredictionOutput(str(self.output), "run-a", False)
        first.prepare()
        try:
            with self.assertRaisesRegex(FileExistsError, "stale lock"):
                PredictionOutput(str(self.output), "run-b", False).prepare()
        finally:
            first.release()

    def test_missing_distribution_metadata_does_not_fail_provenance(self):
        """A source checkout without installed metadata records 'unknown'."""
        self.assertEqual(predictor_module._package_version("hyper-parallel-absent-distribution"), "unknown")

    def test_bucketed_outputs_restore_order_and_correct_completion_boundaries(self):
        self.write_input([{"id": "long", "prompt": "1 2 3"}, {"prompt": "4"}, {"prompt": "1 2"}])
        summary = self.predictor().run_file()
        rows = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["id"] for row in rows], ["long", 1, 2])
        self.assertEqual([row["generated_text"] for row in rows], ["4 5", "5", "3 4 5"])
        self.assertEqual([row["generated_tokens"] for row in rows], [3, 2, 4])
        self.assertEqual(summary.num_samples, 3)
        manifest = json.loads(Path(str(self.output) + ".manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["output_sha256"], file_digest(self.output))
        self.assertEqual(manifest["input_sha256"], file_digest(self.source))
        self.assertEqual(manifest["generation"]["pad_token_id"], 6)
        self.assertIn("environment", manifest)

    def test_overwrite_is_explicit(self):
        self.write_input([{"prompt": "3"}])
        self.predictor().run_file()
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.predictor().run_file()
        self.assertEqual(self.output.read_bytes(), before)
        self.config.overwrite = True
        self.predictor().run_file()
        self.assertEqual(self.output.read_bytes(), before)

    def test_failed_generation_never_publishes_and_keeps_staging(self):
        self.write_input([{"prompt": "3"}])
        with patch("hyper_parallel.predict.predictor.generate", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                self.predictor().run_file()
        self.assertFalse(self.output.exists())
        self.assertTrue(list(self.root.glob(".output.jsonl.*/rank-0.jsonl")))
        self.assertFalse((self.root / ".output.jsonl.lock").exists())

    def test_invalid_input_fails_before_generation(self):
        values = [[], [{"id": "same", "prompt": "1"}, {"id": "same", "prompt": "2"}],
                  [{"prompt": "1", "messages": []}], [{"prompt": ""}], [{"id": True, "prompt": "1"}],
                  [{"messages": [{"role": "user", "content": [{"type": "image"}]}]}]]
        for records in values:
            self.write_input(records)
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.predictor().run_file()
        self.assertFalse(self.output.exists())

    def test_input_formats_and_explicit_chat_template(self):
        txt = self.root / "requests.txt"
        txt.write_text("1 2\n3\n", encoding="utf-8")
        self.assertEqual([request.input_id for request in read_requests(txt)], [0, 1])
        doc = self.root / "requests.json"
        doc.write_text(json.dumps([{"messages": [{"role": "system", "content": "s"},
                                                  {"role": "user", "content": "u"},
                                                  {"role": "assistant", "content": None,
                                                   "tool_calls": [{"id": "t"}]},
                                                  {"role": "tool", "content": "t", "tool_call_id": "t"}]}]),
                       encoding="utf-8")
        requests = read_requests(doc)
        with self.assertRaisesRegex(ValueError, "explicit"):
            encode_requests(requests, Tokenizer())
        encoded = encode_requests(requests, Tokenizer(), "template", {"enable_thinking": False})
        self.assertEqual(encoded[0].token_ids, [4, 3])

    def test_token_budget_counts_padded_width(self):
        requests = [EncodedRequest(PredictionRequest(i, i, prompt="x"), [1] * length)
                    for i, length in enumerate([1, 3, 2, 3])]
        batches = list(plan_batches(requests, self.config))
        self.assertEqual([len(batch) for batch in batches], [2, 2])
        self.assertTrue(all(max(len(item.token_ids) for item in batch) * len(batch) <= 6 for batch in batches))
        self.config.max_batch_tokens = 2
        with self.assertRaises(ValueError):
            list(plan_batches(requests, self.config))

    def test_missing_duplicate_or_misowned_shards_are_rejected(self):
        requests = [PredictionRequest(i, i, prompt="x") for i in range(3)]
        output = PredictionOutput(str(self.output), "run", False)
        output.prepare()
        self.addCleanup(output.release)
        output.write_shard(0, [{"id": 0, "ordinal": 0, "sample_idx": 0}])
        output.write_shard(1, [{"id": 1, "ordinal": 1, "sample_idx": 0}])
        with self.assertRaisesRegex(ValueError, "cover every input"):
            output.commit(requests, 2, {})
        self.assertFalse(self.output.exists())

    def test_predictor_does_not_import_task_host_or_evaluation(self):
        """The request-processing package depends only on its injected runtime."""
        root = Path(__file__).resolve().parents[3] / "hyper_parallel" / "predict"
        forbidden = ("hyper_parallel.trainer", "hyper_parallel.evaluation")
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith(forbidden), str(path))
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith(forbidden) for alias in node.names), str(path))
