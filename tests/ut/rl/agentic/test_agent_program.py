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
"""Exact-context trajectories and fully settled external program lifecycles."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import signal
import sys
import threading
import tempfile
from types import SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import AsyncMock, patch

import torch

from rl.agentic.codex import harness
from rl.agentic.codex.harness import build_codex_call_trajectories, build_codex_trajectory, _stop_process_group
from rl.agentic.core import program_runner as program_module
from rl.agentic.core.program_runner import ProgramAgentRunner
from rl.agentic.ds_harness.harness import build_deepseek_trajectory
from rl.dataset.contracts import Message, PromptRecord
from rl.roles.rollout.base import GenerationSettings


def _record(ordinal: int, prompt: list[int], action: list[int]) -> dict:
    """Construct actual model-call tokens, including sampled-token logprobs."""
    return {"ordinal": ordinal, "request": {"messages": []}, "original_request": {},
            "response": {"prompt_token_ids": prompt, "choices": [{"token_ids": action,
                         "finish_reason": "stop", "logprobs": {"content": [
                             {"token_id": token, "logprob": -0.25} for token in action]}}]}}


def _arguments() -> dict:
    """Return one stable prompt and policy identity."""
    return {"prompt": PromptRecord("task", (Message("user", "question"),),
                                   metadata={"input_ids": torch.tensor([1, 2])}),
            "policy_version": 3, "sample_index": 0, "reward": 1.0, "reward_components": {"outcome": 1.0}}


class TestAgentProgram(unittest.IsolatedAsyncioTestCase):
    """Check real-context training rows without starting a model service."""

    def test_segmented_rows_preserve_rewritten_prompt(self) -> None:
        """Every call trains only its actual action under its own true context."""
        records = [_record(0, [1, 2], [3, 9]), _record(1, [7, 8, 9], [4, 9])]
        rows = build_codex_call_trajectories(completion_records=records, **_arguments())
        self.assertEqual(rows[0].token_ids.tolist(), [1, 2, 3, 9])
        self.assertEqual(rows[1].token_ids.tolist(), [7, 8, 9, 4, 9])
        self.assertEqual(rows[1].action_mask.tolist(), [False, False, False, True, True])
        torch.testing.assert_close(rows[1].rollout_log_probs, torch.tensor([0.0, 0.0, -0.25, -0.25]))
        self.assertEqual([row.metadata["call_index"] for row in rows], [0, 1])
        self.assertEqual({row.metadata["call_count"] for row in rows}, {2})
        self.assertEqual({row.metadata["episode_id"] for row in rows}, {"task:3:0"})
        self.assertEqual([row.reward for row in rows], [1.0, 1.0])

    def test_continuous_harness_requires_exact_full_prefix(self) -> None:
        """Dense DeepSeek/Codex APIs remain one row but never rewrite sampled history."""
        first = _record(0, [1, 2], [3, 9])
        for builder in (build_codex_trajectory, build_deepseek_trajectory):
            with self.subTest(builder=builder.__name__):
                row = builder(completion_records=[first, _record(1, [1, 2, 3, 9, 6], [4, 9])], **_arguments())
                self.assertEqual(row.token_ids.tolist(), [1, 2, 3, 9, 6, 4, 9])
                self.assertNotIn("episode_id", row.metadata)
                for rewritten in ([1, 2, 5, 9, 6], [7, 2, 3, 9, 6], [1, 2]):
                    with self.assertRaisesRegex(ValueError, "exact sampled-action prefix"):
                        builder(completion_records=[first, _record(1, rewritten, [4, 9])], **_arguments())

    def test_missing_duplicate_calls_and_nonfinite_logprobs_fail(self) -> None:
        """Malformed evidence is rejected before trajectory batching."""
        for ordinal in (0, 2):
            with self.assertRaisesRegex(ValueError, "ordinals"):
                build_codex_call_trajectories(
                    completion_records=[_record(0, [1], [2]), _record(ordinal, [3], [4])], **_arguments())
        invalid = _record(0, [1], [2])
        invalid["response"]["choices"][0]["logprobs"]["content"][0]["logprob"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            build_codex_call_trajectories(completion_records=[invalid], **_arguments())

    async def test_batch_waits_for_siblings_before_reporting_failure(self) -> None:
        """One failed episode cannot leave siblings issuing requests after error synchronization."""
        completed = []

        async def run(index: int) -> tuple:
            """Fail one sample while the sibling records successful settlement."""
            if index == 0:
                raise RuntimeError("first episode failed")
            await asyncio.sleep(0.02)
            completed.append(index)
            return ()

        def factory(prompt: PromptRecord, version: int, index: int) -> SimpleNamespace:
            """Create the controlled sampled program."""
            del prompt, version
            return SimpleNamespace(run=lambda: run(index))

        settings = GenerationSettings(8, 1.0, 1.0, 0, True, 0, 9, True)
        runner = ProgramAgentRunner(factory, 2, settings)
        with self.assertRaisesRegex(RuntimeError, "draining all samples"):
            await runner._run([_arguments()["prompt"]], 3)
        self.assertEqual(completed, [1])

    async def test_program_tuple_requires_one_complete_episode(self) -> None:
        """A program cannot hide missing calls or return two sampled episodes as one."""
        rows = build_codex_call_trajectories(
            completion_records=[_record(0, [1], [2]), _record(1, [3], [4])], **_arguments())
        settings = GenerationSettings(8, 1.0, 1.0, 0, True, 0, 9, True)
        for result in (rows[:1], rows + (replace(rows[0], metadata={**rows[0].metadata,
                       "episode_id": "other", "call_count": 1}),)):
            runner = ProgramAgentRunner(
                lambda *args, captured=result: SimpleNamespace(run=AsyncMock(return_value=captured)), 1, settings)
            with self.assertRaises(ValueError):
                await runner._run([_arguments()["prompt"]], 3)

    async def test_program_result_must_match_its_submitted_prompt(self) -> None:
        """A different prompt in the same batch is still the wrong program identity."""
        args = _arguments()
        rows = build_codex_call_trajectories(completion_records=[_record(0, [1], [2])], **args)
        other = replace(args["prompt"], prompt_id="other")
        settings = GenerationSettings(8, 1.0, 1.0, 0, True, 0, 9, True)
        runner = ProgramAgentRunner(lambda *values: SimpleNamespace(run=AsyncMock(return_value=rows)), 1, settings)
        with self.assertRaisesRegex(ValueError, "different submitted prompt"):
            await runner._run([args["prompt"], other], 3)
        legacy = build_codex_trajectory(completion_records=[_record(0, [1], [2])], **args)
        runner = ProgramAgentRunner(lambda *values: SimpleNamespace(run=AsyncMock(return_value=legacy)), 2, settings)
        with self.assertRaisesRegex(ValueError, "duplicate trajectory IDs"):
            await runner._run([args["prompt"]], 3)

    def test_tp_failures_synchronize_before_and_after_payload(self) -> None:
        """Serialization and replica-only reconstruction/batch failures stop both TP ranks."""
        rows = build_codex_call_trajectories(completion_records=[_record(0, [1], [2])], **_arguments())
        settings = GenerationSettings(8, 1.0, 1.0, 0, True, 0, 9, True)

        def check_failure(failure: str) -> None:
            """Give each failure scenario independent collective state and closures."""
            barrier = threading.Barrier(2, timeout=5)
            state = {"errors": {}, "payload": None, "payload_calls": []}
            local = threading.local()
            original_batch = program_module.build_experience_batch

            def build_batch(**kwargs: Any) -> Any:
                """Inject one replica-only failure after reconstructing the collective payload."""
                if failure == "batch" and local.rank == 1:
                    raise ValueError("replica batch failed")
                return original_batch(**kwargs)

            def run_rank(rank: int) -> str:
                """Model the same two error collectives and one payload collective on each rank."""
                local.rank = rank
                phase = 0

                def synchronize(error: Optional[Exception], operation: str) -> None:
                    """Propagate either rank's error before either rank can return."""
                    nonlocal phase
                    state["errors"][(phase, rank)] = error
                    barrier.wait()
                    errors = [state["errors"][(phase, peer)] for peer in range(2)]
                    barrier.wait()
                    phase += 1
                    if any(item is not None for item in errors):
                        raise RuntimeError(operation)

                def payload(value: Any) -> Any:
                    """Replay the owner's serialized rows to both siblings."""
                    if rank == 0:
                        state["payload"] = value
                    barrier.wait()
                    state["payload_calls"].append(rank)
                    return state["payload"]

                engine = SimpleNamespace(is_request_owner=rank == 0, synchronize_error=synchronize,
                                         synchronize_agent_payload=payload)
                runner = ProgramAgentRunner(lambda *args: SimpleNamespace(run=AsyncMock(return_value=rows)),
                                            1, settings, engine)
                def fail(*_args: Any) -> Any:
                    """Inject a rank-local serialization or reconstruction failure."""
                    raise ValueError(failure)

                if failure == "serialize" and rank == 0:
                    runner._serialize_trajectories = fail
                if failure == "deserialize" and rank == 1:
                    runner._deserialize_trajectories = fail
                try:
                    runner.rollout([_arguments()["prompt"]], 3)
                except RuntimeError as error:
                    return str(error)
                raise AssertionError("rank incorrectly continued after a sibling failure")

            with self.subTest(failure=failure), patch.object(program_module, "build_experience_batch", build_batch):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(run_rank, rank) for rank in range(2)]
                    errors = [future.result(timeout=10) for future in futures]
                self.assertEqual(errors[0], errors[1])
                self.assertEqual(len(state["payload_calls"]), 0 if failure == "serialize" else 2)

        for failure in ("serialize", "deserialize", "batch"):
            check_failure(failure)

    async def test_only_explicit_terminal_model_failure_can_receive_zero(self) -> None:
        """Past model errors cannot mask a later service fault or unexplained process failure."""
        args = _arguments()
        captured = {"policy_version": 3, "completions": [_record(0, [1], [2])]}
        with patch.object(harness, "_load_reward_callable", return_value=lambda *values: 1.0):
            program = harness.CodexAgentProgram(args["prompt"], 3, 0, "http://gateway", {}, 9)
        with (
            patch.object(program, "_write_codex_config"),
            patch.object(program, "_validate_version", new=AsyncMock()),
            patch.object(program, "_run_codex", new=AsyncMock(side_effect=RuntimeError("program failed"))),
            patch.object(harness, "_http_json", return_value=captured),
        ):
            for origin in (None, "infrastructure", "unknown"):
                captured["failure"] = (None if origin is None else
                                       {"failure_origin": origin, "trainable": False})
                with self.assertRaises(RuntimeError):
                    await program._run_registered("session", Path("/tmp"), Path("/tmp"), Path("/tmp"), 1.0)
            captured["failure"] = {"failure_origin": "model", "trainable": True,
                                   "failure_reason": "tool_format_budget_exhausted"}
            rows = await program._run_registered("session", Path("/tmp"), Path("/tmp"), Path("/tmp"), 1.0)
            self.assertEqual(rows[0].reward, 0)
            self.assertEqual(rows[0].metadata["failure_origin"], "model")
            self.assertEqual(rows[0].token_ids.tolist(), [1, 2])

    async def test_successful_harness_cleans_descendants_holding_output_pipes(self) -> None:
        """A detached tool in the same group cannot hold successful episode capture open."""
        args = _arguments()
        command = [sys.executable, "-c", "import subprocess,sys,json; "
                   "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                   "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'done'}}))"]
        with patch.object(harness, "_load_reward_callable", return_value=lambda *values: 1.0):
            program = harness.CodexAgentProgram(args["prompt"], 3, 0, "http://gateway", {"timeout_seconds": 2}, 9)
        with tempfile.TemporaryDirectory() as directory, patch.object(program, "_codex_command", return_value=command):
            path = Path(directory)
            result = await asyncio.wait_for(program._run_codex("test", path, path, path), 5)
            self.assertEqual(result, ("done", 0, []))

    async def test_process_group_cleanup_reaps_running_child(self) -> None:
        """Cleanup sends signals to the complete isolated subprocess group."""
        process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(60)",
                                                       start_new_session=True)
        try:
            await _stop_process_group(process)
            self.assertIsNotNone(process.returncode)
            with self.assertRaises(ProcessLookupError):
                os.killpg(process.pid, 0)
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
