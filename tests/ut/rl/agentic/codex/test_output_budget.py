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
"""A missing final answer needs verified output-budget or reasoning-only EOS evidence to train."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

from rl.agentic.codex.harness import (
    CodexAgentProgram, _CodexExecutionFailure, _interpret_codex_output, build_codex_call_trajectories,
)
from rl.agentic.codex.gateway import ContextBudgetExceeded, ToolProtocolFailure, _Handler, _State, _context_budget_error
from rl.dataset.contracts import Message, PromptRecord


def captured_output() -> dict:
    """Build a completed call with finite sampled-action probabilities."""
    return {"policy_version": 0, "failure": None, "completions": [{
        "ordinal": 0, "request": {"max_tokens": 2},
        "metadata": {"trainable": True, "failure_origin": None},
        "response": {"prompt_token_ids": [10, 11], "choices": [{
            "token_ids": [12, 13], "finish_reason": "length",
            "message": {"content": None, "reasoning": "Unfinished thought"},
            "logprobs": {"content": [{"token_id": token, "logprob": -0.5} for token in (12, 13)]},
        }]},
    }]}


def missing_final(turn_completed: bool = True, return_code: int = 0) -> RuntimeError:
    """Use the real CLI interpreter, rather than manufacturing a budget exception."""
    events = [{"type": "item.completed", "item": {"type": "reasoning", "text": "Unfinished thought"}}]
    if turn_completed:
        events.append({"type": "turn.completed", "usage": {"output_tokens": 2}})
    try:
        _interpret_codex_output([json.dumps(event) for event in events], [], return_code, Path("/unused"))
    except RuntimeError as error:
        return error
    raise AssertionError("The CLI emitted no final message")


class TestOutputBudget(unittest.IsolatedAsyncioTestCase):
    """The shared runner keeps verified model endings while rejecting infrastructure faults."""

    def setUp(self) -> None:
        """Create a real program and mock only external execution and Gateway capture."""
        prompt = PromptRecord("q", (Message("user", "Question"),), metadata={"input_ids": torch.tensor([10])})
        self.program = CodexAgentProgram(prompt, 0, 0, "http://gateway", {
            "max_turns": 3, "reward_callable": "examples.gsm8k.agent:score_codex_gsm8k_answer",
        }, None)
        self.program.reward_callable = Mock(return_value=1.0)

    async def test_verified_model_endings_preserve_complete_calls(self) -> None:
        """Length exhaustion and reasoning-only EOS retain sampled actions with zero reward."""
        for finish, reason, truncated in (("length", "output_budget_exhausted", True),
                                           ("stop", "reasoning_only_eos", False)):
            with self.subTest(finish=finish), tempfile.TemporaryDirectory() as directory:
                capture = captured_output()
                record = capture["completions"][0]
                record["request"]["max_tokens"] = 2 if truncated else 1024
                record["response"]["choices"][0]["finish_reason"] = finish
                self.program.end_of_turn_token_id = 13
                original = deepcopy(capture)
                with patch.object(self.program, "_write_codex_config"), \
                        patch.object(self.program, "_validate_version", new=AsyncMock()), \
                        patch.object(self.program, "_run_codex", new=AsyncMock(side_effect=missing_final())) as run, \
                        patch.object(self.program, "_admin_request", new=AsyncMock(return_value=capture)):
                    root = Path(directory)
                    rows = await self.program._run_registered("s", root, root, root, 1)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].token_ids.tolist(), [10, 11, 12, 13])
                self.assertEqual(rows[0].action_mask.tolist(), [False, False, True, True])
                self.assertEqual(rows[0].rollout_log_probs.tolist(), [0.0, -0.5, -0.5])
                self.assertEqual(rows[0].truncated, truncated)
                self.assertEqual(rows[0].terminal_reason, "max_output_tokens" if truncated else reason)
                self.assertEqual(rows[0].reward, 0.0)
                self.assertEqual(rows[0].metadata["failure_reason"], reason)
                self.assertEqual(capture, original)
                self.program.reward_callable.assert_not_called()
                run.assert_awaited_once()

    def test_unknown_exit_and_incomplete_capture_remain_failures(self) -> None:
        """No-final text, process success, or length alone cannot establish a trainable episode."""
        for error in (RuntimeError("timeout"), missing_final(False), missing_final(return_code=137)):
            with self.subTest(error=str(error)), self.assertRaises(RuntimeError):
                self.program.resolve_execution_outcome(captured_output(), error, "")
        for limit, trainable in ((3, True), (2, False)):
            capture = captured_output()
            record = capture["completions"][0]
            record["request"]["max_tokens"] = limit
            record["metadata"]["trainable"] = trainable
            with self.subTest(limit=limit, trainable=trainable), self.assertRaises(RuntimeError):
                self.program.resolve_execution_outcome(capture, missing_final(), "")

    def test_real_final_answer_uses_task_scoring(self) -> None:
        """A token-limited response with an actual final answer is not forced to zero."""
        self.assertIsNone(self.program.resolve_execution_outcome(captured_output(), None, "#### 4"))

    def test_reasoning_stop_requires_sampled_eos_and_clean_cli(self) -> None:
        """A stop string, absent EOS, or abnormal execution cannot hide a missing answer."""
        capture = captured_output()
        capture["completions"][0]["request"]["max_tokens"] = 1024
        capture["completions"][0]["response"]["choices"][0]["finish_reason"] = "stop"
        for eos in (None, 14):
            self.program.end_of_turn_token_id = eos
            with self.subTest(eos=eos), self.assertRaises(RuntimeError):
                self.program.resolve_execution_outcome(capture, missing_final(), "")
        self.program.end_of_turn_token_id = 13
        for error in (missing_final(False), missing_final(return_code=137), RuntimeError("transport failed")):
            with self.subTest(error=str(error)), self.assertRaises(RuntimeError):
                self.program.resolve_execution_outcome(capture, error, "")
        capture["completions"][0]["response"]["choices"][0]["message"]["content"] = "Actual final answer"
        with self.assertRaises(RuntimeError):
            self.program.resolve_execution_outcome(capture, missing_final(), "")

    def test_process_failure_is_not_hidden_by_a_previous_model_failure(self) -> None:
        """OOM exits and timeouts remain fatal even if a model-failure record already exists."""
        captured = captured_output()
        captured["failure"] = {"failure_origin": "model", "trainable": True,
                               "failure_reason": "tool_format_budget_exhausted"}
        for error in (missing_final(return_code=137), _CodexExecutionFailure("CLI timed out")):
            with self.subTest(error=str(error)), self.assertRaises(RuntimeError):
                self.program.resolve_execution_outcome(captured, error, "")


def _error_body() -> dict:
    """Use the exact structured rejection captured in the failed P0 run."""
    return {"error": {"type": "BadRequestError", "param": "input_tokens", "code": 400,
                      "message": "This model's maximum context length is 8192 tokens. However, you requested "
                                 "1024 output tokens and your prompt contains at least 7169 input tokens, "
                                 "for a total of at least 8193 tokens. Please reduce the length of the input prompt "
                                 "or the number of requested output tokens. (parameter=input_tokens, value=7169)"}}


class TestContextBudget(unittest.TestCase):
    """Preserve every sampled action and refuse unrelated errors or empty episodes."""

    def test_exact_error_and_unknown_errors(self) -> None:
        """Unstructured and differently classified 400s remain infrastructure failures."""
        found = _context_budget_error(400, json.dumps(_error_body()))
        self.assertIsInstance(found, ContextBudgetExceeded)
        self.assertEqual(found.details["context_limit"], 8192)
        for code, payload in [(500, _error_body()), (400, {}), (400, {"error": "bad"}),
                              (400, {"error": {**_error_body()["error"], "param": "messages"}}),
                              (400, {"error": {**_error_body()["error"], "message": "input invalid"}})]:
            with self.subTest(code=code, payload=payload):
                self.assertIsNone(_context_budget_error(code, json.dumps(payload)))

    def test_budget_keeps_sampled_call_and_marks_truncation(self) -> None:
        """A rejected second call adds no fake action, loss row or model retry."""
        state = _State("http://unused", "qwen3", 2)
        state.register("s", {"policy_version": 0, "max_completions": 2})
        record = {"ordinal": 0, "request": {}, "response": {"prompt_token_ids": [10, 11], "choices": [
            {"token_ids": [12], "finish_reason": "tool_calls", "logprobs": {"content": [
                {"token_id": 12, "logprob": -0.5}]}}]}}
        state.save_completion("s", record)
        handler = object.__new__(_Handler)
        handler.server = SimpleNamespace(state=state)
        handler.headers = {"Authorization": "Bearer s"}
        handler._backend_request = Mock(side_effect=ContextBudgetExceeded(8192, 1024, 7169))
        handler._respond_to_session = lambda sid, _session, body: handler._model_completion(sid, body, {})
        with self.assertRaises(ToolProtocolFailure):
            handler._proxy_responses({})
        snapshot = state.snapshot("s")
        failure = snapshot["failure"]
        self.assertTrue(failure["trainable"])
        self.assertEqual(failure["failure_origin"], "model")
        self.assertEqual(len(snapshot["completions"]), 1)
        self.assertEqual(state.get("s").reserved_completions, 0)
        self.assertEqual(state.get("s").inflight_requests, 0)
        handler._backend_request.assert_called_once()
        rows = build_codex_call_trajectories(
            prompt=PromptRecord("p", (), metadata={"input_ids": torch.tensor([10, 11])}),
            policy_version=0, sample_index=0, completion_records=snapshot["completions"],
            reward=0., reward_components={"outcome": 0.}, metadata=failure, max_episode_tokens=8192)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].token_ids.tolist(), [10, 11, 12])
        self.assertEqual(rows[0].action_mask.tolist(), [0, 0, 1])
        self.assertEqual(rows[0].reward, 0.)
        self.assertTrue(rows[0].truncated)

    def test_empty_context_failure_is_not_trainable(self) -> None:
        """An oversized initial harness prompt has no sampled action to train."""
        state = _State("http://unused", "qwen3", 2)
        state.register("s", {"policy_version": 0, "max_completions": 2})
        handler = object.__new__(_Handler)
        handler.server = SimpleNamespace(state=state)
        handler.headers = {"Authorization": "Bearer s"}
        handler._backend_request = Mock(side_effect=ContextBudgetExceeded(8192, 1024, 7169))
        handler._respond_to_session = lambda sid, _session, body: handler._model_completion(sid, body, {})
        with self.assertRaises(ContextBudgetExceeded):
            handler._proxy_responses({})
        self.assertFalse(state.get("s").failure["trainable"])
        self.assertEqual(state.get("s").completions, [])
