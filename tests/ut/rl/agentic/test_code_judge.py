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
"""Code reward, environment and remote execution protocol contracts."""

import unittest
from unittest.mock import Mock

from aiohttp import web
import torch

from rl.agentic.core.types import Action, EpisodeContext, TurnContext
from rl.dataset.contracts import Message, PromptRecord
from examples.code.agent import CodeEnvironment
from examples.code.client import ExecutionResult, SandboxFusionExecutor, _parse_result
from examples.code.judge import extract_python, judge_stdio, validate_tests
from examples.code.prepare_data import adapt_row, prepare_row, DEFAULT_REVISION, _budget_rejection


def _response(status: str = "Finished", return_code: int = 0, stdout: str = "3\n") -> dict:
    return {"status": "Success" if status == "Finished" and return_code == 0 else "Failed",
            "compile_result": None, "run_result": {"status": status, "return_code": return_code,
                                                    "stdout": stdout, "execution_time": 0.1}}


class FakeExecutor:
    """Record submissions while returning controlled candidate outcomes."""

    def __init__(self, results: list) -> None:
        """Store controlled outcomes and track submitted requests."""
        self.results = iter(results)
        self.calls = []
        self.closed = False

    async def run(self, code: str, stdin: str, *, request_id: str) -> ExecutionResult:
        """Return one result or propagate a controlled service failure."""
        self.calls.append((code, stdin, request_id))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result

    async def close(self) -> None:
        """Release the fake transport."""
        self.closed = True


class TestCodeJudge(unittest.IsolatedAsyncioTestCase):
    """Check rewards without executing generated programs in the test process."""

    async def test_all_tests_run_and_only_full_success_scores(self) -> None:
        """Wrong answers do not suppress later tests or partial infrastructure failures."""
        truth = {"inputs": ["1 2", "2 3"], "outputs": ["3", "5"]}
        for first, expected in (("3", 1.0), ("4", 0.0)):
            executor = FakeExecutor([ExecutionResult("success", first, 0.1, 0),
                                     ExecutionResult("success", " 5\n", 0.2, 0)])
            reward = await judge_stdio("```python\nprint(3)\n```", truth, executor,
                                       candidate_id="task:0", runtime_version="test-image")
            self.assertEqual(reward.value, expected)
            self.assertEqual(len(executor.calls), 2)
            self.assertEqual(reward.components["total"], 2)
        executor = FakeExecutor([ExecutionResult("runtime_error", "", 0.1, 1), RuntimeError("service down")])
        with self.assertRaisesRegex(RuntimeError, "service down"):
            await judge_stdio("print(3)", truth, executor, candidate_id="task:0", runtime_version="test-image")

    async def test_candidate_failures_and_format_errors(self) -> None:
        """Execution failures score zero; empty or unsupported code never reaches the service."""
        truth = {"inputs": [""], "outputs": ["3"]}
        for status in ("runtime_error", "timeout", "output_limit"):
            executor = FakeExecutor([ExecutionResult(status, "", 0.1, 1)])
            reward = await judge_stdio("print(3)", truth, executor, candidate_id="t", runtime_version="image")
            self.assertEqual(reward.value, 0)
            self.assertEqual(reward.metadata["status"], status)
        executor = FakeExecutor([])
        reward = await judge_stdio("```javascript\ncode\n```", truth, executor,
                                   candidate_id="t", runtime_version="image")
        self.assertEqual(reward.metadata["status"], "format_error")
        self.assertEqual(executor.calls, [])

    async def test_environment_keeps_private_tests_and_original_tokens(self) -> None:
        """Reset exposes only prompt tokens; judging leaves sampled tokens and logprobs intact."""
        truth = {"inputs": ["PRIVATE_INPUT"], "outputs": ["PRIVATE_OUTPUT"]}
        prompt = PromptRecord("task", (Message("user", "Solve this problem"),), truth,
                              {"task_type": "code_stdio", "language": "python", "input_ids": torch.tensor([1, 2])})
        context = EpisodeContext(prompt, 1, 0, 1)
        executor = FakeExecutor([ExecutionResult("success", "PRIVATE_OUTPUT", 0.1, 0)])
        environment = CodeEnvironment(context, executor, "image")
        observation = await environment.reset(context)
        self.assertNotIn("PRIVATE", str(observation))
        tokens, logprobs = torch.tensor([3, 4]), torch.tensor([-0.1, -0.2])
        action = Action("```python\nprint('answer')\n```", tokens, logprobs, {"finish_reason": "length"})
        transition = await environment.step(action, TurnContext(context, 0, 0.0))
        self.assertEqual(transition.reward, 1)
        self.assertTrue(transition.done and transition.truncated)
        self.assertIs(action.token_ids, tokens)
        self.assertIs(action.rollout_log_probs, logprobs)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            await environment.step(action, TurnContext(context, 0, 0.0))
        await environment.close()
        self.assertTrue(executor.closed)

    async def test_http_and_protocol_failures_propagate_without_retries(self) -> None:
        """A local protocol fixture exercises real HTTP handling, including client closure."""
        responses = [web.json_response(_response()), web.Response(status=503),
                     web.Response(text="not json"), web.json_response({"status": "InternalError"})]
        calls = []

        async def handler(request: web.Request) -> web.Response:
            """Serve one predeclared response per request without running code."""
            calls.append(await request.json())
            return responses[len(calls) - 1]

        app = web.Application()
        app.router.add_post("/run_code", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        executor = SandboxFusionExecutor(f"http://127.0.0.1:{port}")
        try:
            self.assertEqual((await executor.run("print(3)", "", request_id="0")).stdout, "3\n")
            for index in range(1, 4):
                with self.assertRaises(RuntimeError):
                    await executor.run("print(3)", "", request_id=str(index))
            self.assertEqual(len(calls), 4)
        finally:
            await executor.close()
            await runner.cleanup()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await executor.run("print(3)", "", request_id="closed")


class TestCodeDataAndProtocol(unittest.TestCase):
    """Validate task preparation and response classification boundaries."""

    def test_execution_statuses_and_invalid_protocol(self) -> None:
        """Candidate timeout/output/runtime errors are distinct from malformed service data."""
        for status, code, expected in (("Finished", 0, "success"), ("Finished", 1, "runtime_error"),
                                       ("TimeLimitExceeded", 1, "timeout"), ("OutputLimitExceeded", 1, "output_limit")):
            self.assertEqual(_parse_result(_response(status, code)).status, expected)
        for payload in ({}, {"status": "InternalError"}, _response("InfrastructureError")):
            with self.assertRaises(RuntimeError):
                _parse_result(payload)
        payload = _response()
        payload["run_result"]["execution_time"] = float("nan")
        with self.assertRaises(RuntimeError):
            _parse_result(payload)

    def test_prepare_and_adapt_preserve_private_test_contract(self) -> None:
        """A prepared record carries private tests outside the original conversation."""
        row = {"ability": "code", "data_source": "apps", "prompt": [{"role": "user", "content": "Add two integers."}],
               "reward_model": {"ground_truth": {"inputs": ["SECRET"], "outputs": ["RESULT"]}}}
        prepared, reason = prepare_row(row, "train", 7, DEFAULT_REVISION)
        self.assertEqual(reason, "accepted")
        record = adapt_row(prepared, 99)
        self.assertEqual(record.prompt_id, prepared["extra_info"]["task_id"])
        self.assertEqual(record.ground_truth["inputs"], ["SECRET"])
        self.assertNotIn("SECRET", str(record.messages))
        self.assertEqual(len(record.messages), 2)
        row["reward_model"]["ground_truth"]["fn_name"] = "solve"
        self.assertEqual(prepare_row(row, "train", 7, DEFAULT_REVISION)[1], "unsupported_test_mode")

    def test_preparation_counts_encoded_tokens_not_mapping_fields(self) -> None:
        """A two-field tokenizer result may still exceed the prompt token budget."""
        prepared = {"prompt": [{"role": "user", "content": "A long problem"}],
                    "reward_model": {"ground_truth": {"inputs": ["1"], "outputs": ["1"]}}}
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = {"input_ids": list(range(9)), "attention_mask": [1] * 9}
        self.assertEqual(_budget_rejection(prepared, tokenizer, 8, 100, 2), "prompt_exceeds_token_budget")
        self.assertEqual(_budget_rejection(prepared, tokenizer, 9, 100, 2), "accepted")
        tokenizer.apply_chat_template.assert_called_with(
            prepared["prompt"], tokenize=True, add_generation_prompt=True, return_dict=True,
        )

    def test_code_extraction_and_invalid_tests(self) -> None:
        """Final fenced Python is extracted only for judging; malformed test data is rejected."""
        self.assertEqual(extract_python("Thinking\n```python\nprint(1)\n```\n```py\nprint(2)\n```"), "print(2)")
        self.assertEqual(extract_python("<think>unfinished"), "")
        for truth in ({"inputs": [], "outputs": []}, {"inputs": ["a"], "outputs": []},
                      {"inputs": [1], "outputs": ["a"]}, {"inputs": ["a"], "outputs": ["b"], "checker": "custom"}):
            with self.assertRaises(ValueError):
                validate_tests(truth)
