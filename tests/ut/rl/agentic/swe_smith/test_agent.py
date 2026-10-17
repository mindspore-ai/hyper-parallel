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
"""Repository program lifecycle, failure priority and controller credential contracts."""

import asyncio
import base64
from contextlib import asynccontextmanager
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import shutil
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import torch

from rl.agentic.codex import harness as shared_harness
from rl.agentic.core.types import RewardResult
from rl.agentic.envs.docker_workspace import CommandResult, InvalidSubmissionError, WorkspaceOutputLimitError
from rl.dataset.contracts import Message, PromptRecord
from examples.swe_smith import agent as harness


@asynccontextmanager
async def _slot(*_args):
    yield


class TestRepositoryProgram(unittest.IsolatedAsyncioTestCase):
    """Exercise orchestration without substituting model-token reconstruction."""

    def setUp(self) -> None:
        """Create isolated artifacts and replace only external lifecycle boundaries."""
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.config = {"reward_callable": "examples.swe_smith.reward:score_submission", "task": {}, "max_turns": 3,
                       "session_root": self.directory, "request_timeout": 2, "model_context_window": 40960,
                       "workspace": {},
                       "max_new_tokens": 10, "temperature": 1., "top_p": 1., "top_k": -1}
        self.task = SimpleNamespace(workspace_image="sha256:" + "a" * 64, artifact_limits={}, prepare=AsyncMock(),
                                    evaluate=AsyncMock(return_value=RewardResult(1., {"ok": 1.})))
        self.factory = MagicMock(return_value=self.task)
        self.workspace = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), close=AsyncMock(),
                                         copy_in=AsyncMock(), exec=AsyncMock(return_value=CommandResult(0, "", "")),
                                         export=AsyncMock(return_value=Path("frozen.tar")))
        self.relay = SimpleNamespace(start=AsyncMock(return_value="http://127.0.0.1:42/v1"), close=AsyncMock(),
                                     budget_exhausted=asyncio.Event(), budget_termination=None)
        self.capture = {"policy_version": 7, "completions": [{"ordinal": 0}]}
        self.http = MagicMock(side_effect=lambda method, *_args, **_kwargs: self.capture if method == "GET" else {})
        self.workspace_factory = MagicMock(return_value=self.workspace)
        for target, value in (("SWESmithTask", self.factory),
                              ("DockerWorkspace", self.workspace_factory),
                              ("CandidateRelay", MagicMock(return_value=self.relay)),
                              ("acquire_workspace_slot", _slot)):
            patcher = patch.object(harness, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(shared_harness, "_http_json", self.http)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.program = harness.SWESmithProgram(
            SimpleNamespace(prompt_id="repo", messages=[SimpleNamespace(content="fix")]), 7, 0,
            "http://model/v1", self.config, None, admin_url="http://controller", admin_token="controller-only",
        )
        self.program._validate_container_version = AsyncMock()
        self.program._run_container_codex = AsyncMock(return_value=("done", []))
        self.program._build_rows = MagicMock(return_value=("rows",))

    def test_pinned_cli_compaction_warning_is_diagnostic(self) -> None:
        """Successful auto compaction does not make a completed repository turn fatal."""
        warning = {"type": "item.completed", "item": {"type": "error", "message": shared_harness._COMPACTION_WARNING}}
        unknown = {"type": "item.completed", "item": {"type": "error", "message": "unexpected failure"}}
        answer = {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}
        final, diagnostics, failures = harness._classify_codex_events(map(json.dumps, (warning, answer)))
        self.assertEqual(final, "done")
        self.assertEqual(diagnostics, [warning])
        self.assertEqual(failures, [])
        self.assertEqual(harness._classify_codex_events([json.dumps(unknown)])[2], [unknown])

    async def test_task_workspace_image_is_selected_per_prompt(self) -> None:
        """Trusted task identity may select a distinct pinned image for each episode."""
        images = {"first": "sha256:" + "b" * 64, "second": "sha256:" + "c" * 64}

        def task_for_prompt(prompt: object, config: dict) -> object:
            """Resolve the task image independently of the shared recipe default."""
            self.assertNotIn("image", config)
            self.task.workspace_image = images[prompt.prompt_id]
            return self.task

        self.factory.side_effect = task_for_prompt
        for identity, expected in images.items():
            with self.subTest(identity=identity):
                self.program.prompt.prompt_id = identity
                self.assertEqual(await self.program.run(), ("rows",))
                self.assertEqual(self.workspace_factory.call_args.kwargs["image"], expected)
        self.assertNotIn("image", self.config["workspace"])

    async def test_optional_model_instructions_use_the_execution_home(self) -> None:
        """Keep CLI defaults unless explicitly replaced; resolve paths in the execution environment."""
        home = Path(self.directory) / "isolated-home"
        home.mkdir()
        self.program._write_codex_config(home, "session")
        self.assertNotIn("model_instructions_file", (home / "config.toml").read_text())
        self.assertFalse((home / "model-instructions.md").exists())
        instructions = "Read, edit and test the repository.\nKeep source changes focused.\n"
        self.program.config["model_instructions"] = instructions
        self.program._write_codex_config(home, "session")
        expected = home.resolve() / "model-instructions.md"
        self.assertIn(f'model_instructions_file = "{expected}"', (home / "config.toml").read_text())
        self.assertEqual(expected.read_text(), instructions)
        await self.program.run()
        copied_home = self.workspace.copy_in.call_args.args[0]
        copied_config = (copied_home / "config.toml").read_text()
        self.assertIn('model_instructions_file = "/opt/hyper-codex-home/model-instructions.md"', copied_config)
        self.assertEqual((copied_home / "model-instructions.md").read_text(), instructions)
        self.assertFalse((copied_home / "checked_patch.py").exists())

    def test_repository_cli_configuration_does_not_change_text_codex(self) -> None:
        """Repository tasks receive the verified CLI tool limits without changing text sessions."""
        home = Path(self.directory) / "tool-limit-home"
        home.mkdir()
        self.program._write_codex_config(home, "session")
        repository_config = (home / "config.toml").read_text(encoding="utf-8")
        self.assertIn("model_context_window = 40960\ntool_output_token_limit = 10000\n",
                      repository_config)
        self.assertIn("compact_prompt = ", repository_config)
        for invalid in (None, 0, -1, "40960", True):
            with self.subTest(invalid=invalid):
                self.program.config["model_context_window"] = invalid
                with self.assertRaisesRegex(ValueError, "positive model_context_window"):
                    self.program._write_codex_config(home, "session")
        self.program.config["model_context_window"] = 40960
        self.program.config["model_auto_compact_token_limit"] = 32768
        self.program._write_codex_config(home, "session")
        self.assertIn("model_auto_compact_token_limit = 32768", (home / "config.toml").read_text())
        for invalid in (0, 40961, True):
            self.program.config["model_auto_compact_token_limit"] = invalid
            with self.assertRaisesRegex(ValueError, "must fit"):
                self.program._write_codex_config(home, "session")
        self.program.config.pop("model_auto_compact_token_limit")
        self.assertIn("[features]\n", repository_config)
        for setting in ("view_image = false", "goals = false", "default_mode_request_user_input = false",
                        "[tools.experimental_request_user_input]\nenabled = false",
                        "[skills]\ninclude_instructions = false", "[skills.bundled]\nenabled = false"):
            with self.subTest(setting=setting):
                self.assertIn(setting, repository_config)

        text_program = shared_harness.CodexAgentProgram(
            self.program.prompt, 7, 0, "http://model/v1",
            {"reward_callable": "examples.swe_smith.reward:score_submission", "session_root": self.directory}, None,
        )
        text_program._write_codex_config(home, "session")
        text_config = (home / "config.toml").read_text(encoding="utf-8")
        self.assertNotIn("tool_output_token_limit", text_config)
        self.assertNotIn("compact_prompt", text_config)
        for setting in ("view_image = false", "goals = false", "default_mode_request_user_input = false",
                        "[tools.experimental_request_user_input]", "[skills]", "[skills.bundled]"):
            with self.subTest(setting=setting):
                self.assertNotIn(setting, text_config)


    async def test_frozen_submission_is_scored_and_credentials_stay_controller_side(self) -> None:
        """Candidate has only its own token and loopback model endpoint."""
        events = []
        self.relay.close.side_effect = lambda: events.append("relay-close")
        self.workspace.stop.side_effect = lambda: events.append("stop")
        self.workspace.export.side_effect = lambda *_args: events.append("export") or Path("frozen.tar")
        self.task.evaluate.side_effect = lambda *_args: events.append("grade") or RewardResult(1., {"ok": 1.})
        self.assertEqual(await self.program.run(), ("rows",))
        self.assertLess(events.index("relay-close"), events.index("stop"))
        self.assertLess(events.index("stop"), events.index("export"))
        self.assertLess(events.index("export"), events.index("grade"))
        self.task.evaluate.assert_awaited_once_with(Path("frozen.tar"))
        self.workspace.close.assert_awaited_once()
        self.assertEqual([call.args[0] for call in self.http.call_args_list], ["POST", "GET", "DELETE"])
        for call in self.http.call_args_list:
            self.assertTrue(call.args[1].startswith("http://controller/internal/"))
            self.assertEqual(call.args[-1], "controller-only")
            self.assertEqual(call.kwargs["max_response_bytes"], 32 * 1024 * 1024)
        home = next(Path(self.directory).glob("*/.codex"))
        self.assertIn("127.0.0.1:42/v1", (home / "config.toml").read_text())
        self.assertNotIn("controller-only", (home / "auth.json").read_text())
        self.assertIn("artifact_dir", self.factory.call_args.args[1])
        self.assertIs(self.program._build_rows.call_args.args[-2], self.capture)

    async def test_known_model_failure_zero_never_grades_artifact(self) -> None:
        self.capture["failure"] = {"failure_origin": "model", "trainable": True}
        self.program._run_container_codex.side_effect = harness._ExecutionError("turn failed")
        await self.program.run()
        self.assertEqual(self.program._build_rows.call_args.args[-1].value, 0.)
        self.task.evaluate.assert_not_awaited()
        self.workspace.export.assert_not_awaited()


    async def test_confirmed_budget_stops_then_grades_real_artifact(self) -> None:
        """Budget termination preserves graded rewards and per-call routes through real row construction."""
        del self.program._build_rows
        self.program.prompt = PromptRecord("repo", (Message("user", "fix"),),
                                           metadata={"input_ids": torch.tensor([1, 2])})
        self.program.config["max_turns"] = 2
        calls = (([1, 2], [3, 9], [[[0, 1]], [[1, 2]], [[2, 3]]]),
                 ([7, 8, 9], [4, 9], [[[3, 0]], [[2, 3]], [[1, 2]], [[0, 1]]]))
        records = []
        for ordinal, (prompt, action, routes) in enumerate(calls):
            stream = BytesIO()
            np.save(stream, np.array(routes, dtype=np.uint8), allow_pickle=False)
            records.append({
                "ordinal": ordinal, "request": {"messages": []}, "original_request": {},
                "response": {"prompt_token_ids": prompt, "choices": [{
                    "token_ids": action, "finish_reason": "stop",
                    "logprobs": {"content": [{"token_id": token, "logprob": -0.25} for token in action]},
                }]},
                "routed_experts": base64.b64encode(stream.getvalue()).decode("ascii"),
                "router_capture_spec": {"layer_ids": [2], "top_k": 2, "num_experts": 4},
            })
        stopped = asyncio.Event()
        termination = {"reason": "max_completions", "limit": 2, "completed": 2, "policy_version": 7}
        self.capture.update(termination=termination, completions=records)
        self.relay.budget_termination = termination
        events = []

        async def execute(*_args: object, expected_stop: asyncio.Event, expected_budget: dict) -> tuple[str, list]:
            """Wait for the controller's explicit stop before returning CLI evidence."""
            self.relay.budget_exhausted.set()
            await stopped.wait()
            self.assertTrue(expected_stop.is_set())
            self.assertEqual(expected_budget["termination"], termination)
            return "", []

        async def stop() -> None:
            """Record the actual stop after the relay has drained."""
            events.append("stop")
            stopped.set()

        self.program._run_container_codex.side_effect = execute
        self.relay.close.side_effect = lambda: events.append("relay-close")
        self.workspace.stop.side_effect = stop
        rows = await self.program.run()
        self.assertLess(events.index("relay-close"), events.index("stop"))
        self.task.evaluate.assert_awaited_once_with(Path("frozen.tar"))
        self.assertEqual(len(rows), 2)
        for index, (row, (prompt, action, routes)) in enumerate(zip(rows, calls)):
            with self.subTest(call=index):
                self.assertEqual(row.token_ids.tolist(), prompt + action)
                self.assertEqual(row.action_mask.tolist(), [False] * len(prompt) + [True] * len(action))
                self.assertEqual(row.rollout_log_probs.tolist(), [0.] * (len(prompt) - 1) + [-0.25] * len(action))
                self.assertIsNotNone(row.router_trace)
                self.assertEqual(row.router_trace.layer_ids, (2,))
                self.assertEqual(row.router_trace.expert_ids[:-1].tolist(), routes)
                self.assertEqual(row.router_trace.valid_mask.tolist(), [True] * len(routes) + [False])
                self.assertEqual(row.reward, 1.)
                self.assertEqual(row.reward_components, {"ok": 1.})
                self.assertEqual((row.policy_version, row.worker_policy_version), (7, 7))
                self.assertEqual(row.metadata["episode_id"], "repo:7:0")
                self.assertEqual((row.metadata["call_index"], row.metadata["call_count"]), (index, 2))
                self.assertEqual(row.metadata["budget_termination"], termination)
                self.assertTrue(row.metadata["repository_budget_truncated"])
                self.assertTrue(row.truncated)
                self.assertEqual(row.terminal_reason, "max_completions")
        self.assertTrue(self.http.call_args_list[0].args[2]["isolated_cli"])

    async def test_budget_signal_does_not_hide_cli_or_transport_failure(self) -> None:
        """A terminal response cannot turn independent execution failure into reward."""
        self.relay.budget_exhausted.set()
        self.program._run_container_codex.side_effect = harness._ExecutionError("unknown CLI error")
        with self.assertRaisesRegex(RuntimeError, "unknown CLI"):
            await self.program.run()
        self.task.evaluate.assert_not_awaited()

    async def test_controller_stopped_cli_preserves_output_without_inventing_answer(self) -> None:
        """Only the expected signal exit can omit a final answer; prior CLI failures remain errors."""
        event = asyncio.Event()
        event.set()
        directory = Path(self.directory)
        for stdout, code, accepted in (("", 137, True), ("", 143, True), ("", 1, False),
                                       ('{"type":"turn.failed","error":{"message":"unknown"}}\n', 137, False)):
            with self.subTest(stdout=stdout, code=code):
                self.workspace.exec = AsyncMock(return_value=CommandResult(code, stdout, "actual stderr"))
                invoke = harness.SWESmithProgram._run_container_codex(
                    self.program, self.workspace, "session", directory, expected_stop=event,
                )
                if accepted:
                    self.assertEqual(await invoke, ("", []))
                else:
                    with self.assertRaises(harness._ExecutionError):
                        await invoke
                self.assertEqual((directory / "codex-events.jsonl").read_text(), stdout)
                self.assertEqual((directory / "codex-stderr.log").read_text(), "actual stderr")

    async def test_budget_retry_diagnostic_requires_exact_controller_context(self) -> None:
        """A known retry stays observable; other HTTP errors and mixed failures still reject."""
        event = asyncio.Event()
        event.set()
        context = {"termination": {"limit": 3}, "url": "http://127.0.0.1:42/v1/responses"}
        message = ("Reconnecting... 1/5 (unexpected status 409 Conflict: "
                   "Repository session exhausted its 3 model calls, url: http://127.0.0.1:42/v1/responses)")
        exact = {"type": "error", "message": message}
        for events, evidence, accepted in (
                ([exact], context, True), ([exact], None, False),
                ([{"type": "error", "message": message.replace("3 model", "4 model")}], context, False),
                ([{"type": "error", "message": message.replace("409 Conflict", "500 Internal")}], context, False),
                ([{"type": "error", "message": message.replace("127.0.0.1", "elsewhere")}], context, False),
                ([{"type": "error", "message": message.replace("/responses", "/other")}], context, False),
                ([exact, {"type": "error", "message": "unrelated failure"}], context, False),
                ([{"type": "turn.failed", "message": message}], context, False)):
            with self.subTest(events=events, evidence=evidence):
                stdout = "\n".join(json.dumps(item) for item in events)
                self.workspace.exec = AsyncMock(return_value=CommandResult(137, stdout, ""))
                invoke = harness.SWESmithProgram._run_container_codex(
                    self.program, self.workspace, "session", Path(self.directory),
                    expected_stop=event, expected_budget=evidence,
                )
                if accepted:
                    self.assertEqual(await invoke, ("", [exact]))
                else:
                    with self.assertRaises(harness._ExecutionError):
                        await invoke
                self.assertEqual((Path(self.directory) / "codex-events.jsonl").read_text(), stdout)

    async def test_invalid_candidate_archive_scores_zero_without_grader(self) -> None:
        self.workspace.export.side_effect = InvalidSubmissionError("linked file")
        await self.program.run()
        reward = self.program._build_rows.call_args.args[-1]
        self.assertEqual(reward.value, 0.)
        self.assertEqual(reward.metadata["status"], "invalid_submission")
        self.task.evaluate.assert_not_awaited()

    async def test_infrastructure_failures_cannot_be_overridden_by_model_zero(self) -> None:
        """Infrastructure errors remain fatal even when a trainable model error is also recorded."""
        for origin in ("relay", "output", "gateway", "export", "grade"):
            with self.subTest(origin=origin):
                self.capture.pop("failure", None)
                self.relay.close.side_effect = None
                self.program._run_container_codex.side_effect = None
                self.workspace.export.side_effect = None
                self.task.evaluate.side_effect = None
                if origin == "relay":
                    self.capture["failure"] = {"failure_origin": "model", "trainable": True}
                    self.relay.close.side_effect = RuntimeError("transport")
                elif origin == "output":
                    self.capture["failure"] = {"failure_origin": "model", "trainable": True}
                    self.program._run_container_codex.side_effect = WorkspaceOutputLimitError(CommandResult(1, "", ""))
                elif origin == "gateway":
                    self.capture["failure"] = {"failure_origin": "infrastructure", "trainable": False}
                elif origin == "export":
                    self.workspace.export.side_effect = RuntimeError("export")
                else:
                    self.task.evaluate.side_effect = RuntimeError("grade")
                with self.assertRaises(RuntimeError):
                    await self.program.run()
                self.assertEqual(self.http.call_args.args[0], "DELETE")
        self.assertEqual(self.workspace.close.await_count, 5)
        self.program._build_rows.assert_not_called()

    async def test_cancel_registration_settles_before_delete(self) -> None:
        """Cancellation waits for session registration before attempting deletion."""
        entered, release = threading.Event(), threading.Event()
        methods = []

        def blocking(method: str, *_args: object, **_kwargs: object) -> dict:
            """Hold registration until the test releases its in-flight response."""
            methods.append(method)
            if method == "POST":
                entered.set()
                release.wait(2)
            return {}

        self.http.side_effect = blocking
        task = asyncio.create_task(self.program.run())
        await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(.01)
        self.assertEqual(methods, ["POST"])
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(methods, ["POST", "DELETE"])
        self.workspace.start.assert_not_awaited()

    async def test_cancel_prepare_or_grading_closes_candidate_and_session(self) -> None:
        """Cancellation during preparation or grading closes the candidate and session."""
        for boundary in (self.task.prepare, self.task.evaluate):
            with self.subTest(boundary=boundary):
                self.workspace.close.reset_mock()
                boundary.side_effect = asyncio.CancelledError()
                with self.assertRaises(asyncio.CancelledError):
                    await self.program.run()
                self.workspace.close.assert_awaited_once()
                self.assertEqual(self.http.call_args.args[0], "DELETE")
                if boundary is self.task.prepare:
                    self.task.evaluate.assert_not_awaited()
                boundary.side_effect = None

    async def test_registration_error_still_deletes_ambiguous_session(self) -> None:
        self.http.side_effect = [TimeoutError("registration timeout"), {}]
        with self.assertRaises(TimeoutError):
            await self.program.run()
        self.assertEqual([call.args[0] for call in self.http.call_args_list], ["POST", "DELETE"])
        self.workspace.start.assert_not_awaited()

    async def test_container_execution_passes_only_session_environment_and_saves_logs(self) -> None:
        """The candidate receives only session settings and its bounded output is retained."""
        result = CommandResult(0, '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n'
                               '{"type":"turn.completed"}\n',
                               "diagnostic")
        self.workspace.exec = AsyncMock(return_value=result)
        artifact = Path(self.directory)
        with patch.dict("os.environ", {"HOST_SECRET": "must-not-leak", "OPENAI_API_KEY": "ambient-key"}):
            answer, _ = await harness.SWESmithProgram._run_container_codex(
                self.program, self.workspace, "session-token", artifact,
            )
        self.assertEqual(answer, "done")
        env = self.workspace.exec.call_args.kwargs["env"]
        self.assertNotIn("HOST_SECRET", env)
        self.assertEqual(env["OPENAI_API_KEY"], "session-token")
        self.assertEqual(self.workspace.exec.call_args.kwargs["output_limit_bytes"], 4 * 1024 * 1024)
        self.assertEqual((artifact / "codex-stderr.log").read_text(), "diagnostic")

    async def test_output_budget_keeps_bounded_evidence_and_rejects(self) -> None:
        self.workspace.exec = AsyncMock(side_effect=WorkspaceOutputLimitError(CommandResult(-9, "partial", "tail")))
        artifact = Path(self.directory)
        with self.assertRaises(WorkspaceOutputLimitError):
            await harness.SWESmithProgram._run_container_codex(self.program, self.workspace, "token", artifact)
        self.assertEqual((artifact / "codex-events.jsonl").read_text(), "partial")
