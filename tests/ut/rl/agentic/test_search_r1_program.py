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
"""CPU regressions for search task selection, correction traces and failure classification."""

import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

import torch

from examples.search_r1.agent import SearchR1Program
from rl.agentic.codex.harness import CodexAgentProgram, CodexProgramFactory
from rl.agentic.core.types import RewardResult
from rl.dataset.contracts import Message, PromptRecord


def completion(ordinal: int) -> dict:
    """Create independently sampled, deliberately non-prefix-continuous calls."""
    return {"ordinal": ordinal, "request": {}, "response": {"choices": [{
        "prompt_token_ids": [10 + ordinal, 20], "token_ids": [30 + ordinal],
        "logprobs": {"content": [{"token_id": 30 + ordinal, "logprob": -0.5}]},
    }]}}


class SearchR1ProgramTests(unittest.TestCase):
    """Test actual row construction while mocking only execution and network boundaries."""

    def test_program_factory_is_opt_in(self) -> None:
        """Baseline keeps its program; only configured search tasks select the override."""
        runtime = SimpleNamespace(config={})
        self.assertIs(CodexProgramFactory(runtime, None, {}).program_type, CodexAgentProgram)
        runtime.config["program_callable"] = "examples.search_r1.agent:SearchR1Program"
        self.assertIs(CodexProgramFactory(runtime, None, {}).program_type, SearchR1Program)

    def test_repair_and_task_failures_preserve_original_actions(self) -> None:
        """Both correction calls train; genuine task failure stays, infrastructure failure raises."""
        prompt = PromptRecord(prompt_id="q", messages=(Message("user", "Question"),), ground_truth="X",
                              metadata={"input_ids": torch.tensor([1])})
        for reason in (None, "budget_exhausted", "context_exhausted", "upstream_error", "swallowed_http_error",
                       "container_exit"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary:
                artifact = Path(temporary)
                workspace = artifact / "workspace"
                (workspace / "articles").mkdir(parents=True)
                (workspace / "articles/000.txt").write_text("[REF: Title:0] Evidence\n")
                (artifact / "search-observations.json").write_text(json.dumps(["[REF: Title:0] Evidence"]))
                program = SearchR1Program(prompt, 0, 0, "http://127.0.0.1:8200", {
                    "max_turns": 3, "max_episode_tokens": 128,
                    "reward_callable": "examples.search_r1.reward:score_hotpotqa_answer",
                }, None)
                program._write_codex_config = Mock()
                program._validate_version = AsyncMock()
                program.reward_callable = Mock(return_value=RewardResult(1.0, {"answer": 1.0}))
                program._run_codex = AsyncMock(return_value=("ANSWER: X\nSOURCES: 000.txt:0", 0, []))
                program._capture = AsyncMock(side_effect=[
                    {"completions": [completion(0)]},
                    {"completions": [completion(0), completion(1)]},
                ])
                program._repair = Mock(return_value="ANSWER: X\nSOURCES: Title:0")
                if reason == "container_exit":
                    (artifact / "execution-result.json").write_text(json.dumps({"returncode": 137}))
                elif reason:
                    model_failure = reason == "budget_exhausted"
                    (artifact / "search-terminal.json").write_text(json.dumps({
                        "reason": reason, "failure_origin": "model" if model_failure else "infrastructure",
                        "trainable": model_failure,
                    }))
                    if reason != "swallowed_http_error":
                        program._run_codex.side_effect = RuntimeError("execution stopped")
                if reason in ("upstream_error", "context_exhausted", "swallowed_http_error", "container_exit"):
                    with self.assertRaises(RuntimeError):
                        asyncio.run(program._run_registered("s", artifact, workspace, artifact, 5))
                    program._repair.assert_not_called()
                    program.reward_callable.assert_not_called()
                    continue
                rows = asyncio.run(program._run_registered("s", artifact, workspace, artifact, 5))
                self.assertEqual(len(rows), 1 if reason else 2)
                self.assertEqual(rows[0].token_ids.tolist(), [10, 20, 30])
                self.assertTrue(all(row.reward == (0.0 if reason else 1.0) for row in rows))
                audit = json.loads((artifact / "training-audit.json").read_text())
                self.assertEqual(audit["training_context_mismatches"], 0)
                if not reason:
                    self.assertEqual(rows[1].token_ids.tolist(), [11, 20, 31])
                    self.assertEqual(audit["raw_prefix_mismatches"], 1)
                    program._repair.assert_called_once()
