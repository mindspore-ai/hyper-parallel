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
from unittest.mock import AsyncMock, Mock, patch

import torch

from rl.agentic.codex.harness import (
    CodexAgentProgram, CodexProgramFactory, _CompletionBudgetExit, _interpret_codex_output,
)
from rl.agentic.core.types import RewardResult
from rl.agentic.envs.docker_workspace import CommandResult
from rl.dataset.contracts import Message, PromptRecord

from examples.search_r1.agent import SearchR1Program
from examples.search_r1 import container_runtime


def completion(ordinal: int) -> dict:
    """Create independently sampled, deliberately non-prefix-continuous calls."""
    return {"ordinal": ordinal, "request": {}, "response": {"choices": [{
        "prompt_token_ids": [10 + ordinal, 20], "token_ids": [30 + ordinal],
        "logprobs": {"content": [{"token_id": 30 + ordinal, "logprob": -0.5}]},
    }]}}


class SearchR1ProgramTests(unittest.TestCase):
    """Test actual row construction while mocking only execution and network boundaries."""

    def test_host_execution_preserves_controller_output_files(self) -> None:
        """The host writes raw evidence separately from the controller's already-open stdout files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "execution-request.json"
            request.write_text(json.dumps({"bridge": "/results/bridge", "session_id": "s", "command": ["codex"]}))
            for name in ("codex-events.jsonl", "codex-stderr.log"):
                (root / name).write_text("controller-owned")
                (root / name).chmod(0o444)
            workspace = SimpleNamespace(
                name="candidate", start=AsyncMock(), close=AsyncMock(),
                exec=AsyncMock(side_effect=[CommandResult(0, "{}", ""), CommandResult(0, "events", "stderr")]),
                inspect=AsyncMock(return_value={"HostConfig": {}, "Mounts": [], "State": {}}),
            )
            relay = SimpleNamespace(start=AsyncMock(return_value="http://localhost:1234"),
                                    close=AsyncMock(), raise_if_failed=Mock())
            with patch.object(container_runtime, "episode_workspace", return_value=workspace), \
                    patch.object(container_runtime, "CandidateRelay", return_value=relay):
                container_runtime.execute_episode(request, root, "image", "prefix")
            self.assertEqual(json.loads((root / "execution-result.json").read_text()), {"returncode": 0})
            for name, expected in (("codex-events.jsonl", "events"), ("codex-stderr.log", "stderr")):
                self.assertEqual((root / name).read_text(), "controller-owned")
                self.assertEqual((root / "container" / name).read_text(), expected)

    def test_isolated_cli_resolves_model_instructions_inside_its_mount(self) -> None:
        """The instruction file is written on the host but referenced at the candidate mount."""
        prompt = PromptRecord(prompt_id="q", messages=(Message("user", "Question"),), ground_truth="X")
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            home = artifact / ".codex"
            home.mkdir()
            program = SearchR1Program(prompt, 0, 0, "http://127.0.0.1:18200", {
                "max_turns": 10, "max_episode_tokens": 8192,
                "reward_callable": "examples.search_r1.reward:score_hotpotqa_answer",
                "model_instructions": "Search the local files.",
            }, None)
            program._validate_version = AsyncMock()
            program._run_codex = AsyncMock(return_value=("ANSWER: X\nSOURCES: Title:0", 0, []))
            asyncio.run(program._execute_search_process("s", artifact, artifact, home))
            self.assertEqual((home / "model-instructions.md").read_text(), "Search the local files.")
            config = (home / "config.toml").read_text()
            self.assertIn('model_instructions_file = "/codex-home/model-instructions.md"', config)
            self.assertNotIn(str(home.resolve()), config)

    def test_search_checks_candidate_image_not_controller_executable(self) -> None:
        """Pin configuration to the actual candidate image and reject a stale or mismatched preflight."""
        prompt = PromptRecord("q", (Message("user", "Question"),))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "candidate-runtime.json"
            program = SearchR1Program(prompt, 0, 0, "http://gateway", {
                "session_root": str(root / "jobs"), "version": "0.152.1",
                "reward_callable": "examples.search_r1.reward:score_hotpotqa_answer",
            }, None)
            with self.assertRaises(FileNotFoundError):
                asyncio.run(program._validate_version())
            record.write_text(json.dumps({"version": "codex-cli 0.152.1", "image_id": "sha256:" + "a" * 64,
                                          "tools": {"codex": "/opt/codex/bin/codex"}}))
            asyncio.run(program._validate_version())
            program.config["version"] = "0.152.2"
            with self.assertRaisesRegex(RuntimeError, "candidate-image preflight"):
                asyncio.run(program._validate_version())

    def test_program_factory_is_opt_in(self) -> None:
        """Baseline keeps its program; only configured search tasks select the override."""
        runtime = SimpleNamespace(config={})
        self.assertIs(CodexProgramFactory(runtime, None, {}).program_type, CodexAgentProgram)
        runtime.config["program_callable"] = "examples.search_r1.agent:SearchR1Program"
        self.assertIs(CodexProgramFactory(runtime, None, {}).program_type, SearchR1Program)

    def test_shared_budgets_skip_citation_repair_and_keep_actions(self) -> None:
        """Search uses the same output, context and call-budget resolver as ordinary text tasks."""
        for kind in ("output", "context", "calls", "container_exit"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "articles").mkdir()
                prompt = PromptRecord("q", (Message("user", "Question"),),
                                      metadata={"input_ids": torch.tensor([10])})
                program = SearchR1Program(prompt, 0, 0, "http://gateway", {
                    "max_turns": 2, "reward_callable": "examples.search_r1.reward:score_hotpotqa_answer",
                }, None)
                record = completion(0)
                record.update(request={"max_tokens": 1}, metadata={"trainable": True, "failure_origin": None})
                record["response"]["choices"][0].update(
                    finish_reason="length", message={"reasoning": "Thought", "content": None})
                captured = {"policy_version": 0, "completions": [record]}
                execution_error = None
                try:
                    _interpret_codex_output(['{"type":"turn.completed"}'], [], 0, root)
                except RuntimeError as error:
                    execution_error = error
                if kind == "context":
                    execution_error = RuntimeError("Backend context limit")
                    captured["failure"] = {"failure_origin": "model", "trainable": True,
                                           "failure_reason": "context_budget_exhausted"}
                if kind == "calls":
                    captured["completions"].append(completion(1))
                    captured["termination"] = {"reason": "max_completions", "limit": 2,
                                               "completed": 2, "policy_version": 0}
                    execution_error = _CompletionBudgetExit(2, "http://gateway/responses")
                (root / "execution-result.json").write_text(json.dumps({
                    "returncode": 137 if kind == "container_exit" else 0,
                }))
                program._write_codex_config = Mock()
                program._validate_version = AsyncMock()
                program._run_codex = AsyncMock(side_effect=execution_error)
                program._capture = AsyncMock(return_value=captured)
                program._capture_raw = AsyncMock(return_value=captured)
                program._repair = Mock()
                program.reward_callable = Mock()
                if kind == "container_exit":
                    with self.assertRaisesRegex(RuntimeError, "container"):
                        asyncio.run(program._run_registered("s", root, root, root, 1))
                else:
                    rows = asyncio.run(program._run_registered("s", root, root, root, 1))
                    self.assertEqual(len(rows), len(captured["completions"]))
                    self.assertTrue(all(row.reward == 0 and row.truncated for row in rows))
                    self.assertEqual(rows[0].token_ids.tolist(), [10, 20, 30])
                    self.assertEqual(rows[0].rollout_log_probs.tolist(), [0.0, -0.5])
                program._repair.assert_not_called()
                program.reward_callable.assert_not_called()

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
                    {"policy_version": 0, "completions": [completion(0)]},
                    {"policy_version": 0, "completions": [completion(0), completion(1)]},
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
