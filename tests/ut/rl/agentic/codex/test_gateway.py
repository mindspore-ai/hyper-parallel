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
"""Gateway storage, policy admission and registration preserve episode boundaries."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from typing import Any, IO
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

from rl.agentic.codex.gateway import _Handler, _State
from rl.agentic.codex.harness import CodexAgentProgram
from rl.agentic.core.program_runner import GatewayTransportError
from rl.dataset.contracts import PromptRecord


class TestGatewayEventStorage(unittest.TestCase):
    """Keep complete session traces without holding the global state lock over I/O."""

    def test_blocked_trace_does_not_block_other_session(self) -> None:
        """Slow storage cannot prevent unrelated admission or mutate captured metadata."""
        with tempfile.TemporaryDirectory() as directory:
            state = _State("http://backend", "qwen3", 5)
            state.register("slow", {"policy_version": 0, "artifact_dir": directory, "max_completions": 2})
            entered, release = threading.Event(), threading.Event()
            original_open = Path.open
            payload = {"metadata": {"policy_version": 0}}
            state.reserve_completion("slow")

            def blocked_open(path: Path, *args: Any, **kwargs: Any) -> IO:
                """Block one trace file while other sessions continue."""
                if path == Path(directory) / "gateway-events.jsonl":
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("Test did not release trace writer")
                return original_open(path, *args, **kwargs)

            with ThreadPoolExecutor(max_workers=2) as pool, patch.object(Path, "open", blocked_open):
                writing = pool.submit(state.complete_reservation, "slow", payload)
                try:
                    self.assertTrue(entered.wait(2), "Trace writer did not reach storage")
                    registration = pool.submit(state.register, "other", {"policy_version": 1, "max_completions": 2})
                    registration.result(timeout=2)
                    session = state.reserve_completion("other")
                    state.release_completion(session)
                    next_call = state.reserve_completion("slow")
                    self.assertEqual(len(next_call.completions), 1)
                    state.release_completion(next_call)
                    with state.lock:
                        payload["metadata"]["policy_version"] = 9
                finally:
                    release.set()
                writing.result(timeout=2)
            rows = list(map(json.loads, (Path(directory) / "gateway-events.jsonl").read_text().splitlines()))
            self.assertEqual(rows[-1]["payload"]["metadata"]["policy_version"], 0)
            self.assertEqual(state.get("other").reserved_completions, 0)


class TestPolicyGate(unittest.TestCase):
    """Exercise future-version admission, cancellation, and stale-session rejection."""

    def test_remove_unblocks_parked_request_before_drain(self):
        """Cancelling a prepared session cannot deadlock DELETE against its request."""
        state = _State("http://backend", "model", 3, policy_gate=True)
        state.register("future", {"policy_version": 1, "max_completions": 2})
        entered = threading.Event()
        def wait() -> None:
            """Wait for admission in the request-owning background thread."""
            session = state.begin_request("future")
            entered.set()
            try:
                state.wait_for_policy(session)
            finally:
                state.finish_request(session)
        with ThreadPoolExecutor(1) as pool:
            task = pool.submit(wait)
            self.assertTrue(entered.wait(1))
            state.remove("future")
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                task.result(1)
        self.assertNotIn("future", state.sessions)

    def test_stale_policy_fails_and_disabled_gate_remains_compatible(self):
        """Wrong policies fail rather than silently sampling with new weights."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                state = _State("http://backend", "model", 1, policy_gate=enabled)
                state.register("s", {"policy_version": 0, "max_completions": 2})
                state.set_policy(1)
                if enabled:
                    with self.assertRaisesRegex(RuntimeError, "stale"):
                        state.wait_for_policy(state.get("s"))
                else:
                    state.wait_for_policy(state.get("s"))

    def test_wait_timeout_fails_without_a_completion(self):
        """Unpublished weights never produce a reward-bearing fake answer."""
        state = _State("http://backend", "model", 0.01, policy_gate=True)
        state.register("s", {"policy_version": 0, "max_completions": 2})
        with self.assertRaisesRegex(RuntimeError, "Timed out"):
            state.wait_for_policy(state.get("s"))
        self.assertEqual(state.get("s").completions, [])

    def test_model_handler_never_calls_backend_before_release(self):
        """The real completion path parks before HTTP generation and records the right policy."""
        state = _State("http://backend", "model", 3, policy_gate=True)
        state.register("s", {"policy_version": 1, "max_completions": 2})
        queued = threading.Event()
        original_event = state.event
        def event(session_id: str, event_type: str, payload: dict) -> None:
            """Signal request arrival while preserving the real event record."""
            if event_type == "completion.queued":
                queued.set()
            original_event(session_id, event_type, payload)
        state.event = event
        handler = object.__new__(_Handler)
        handler.server = SimpleNamespace(state=state)
        handler._backend_request = Mock(return_value={
            "id": "r", "choices": [{"message": {"role": "assistant", "content": "done"},
                                       "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        with ThreadPoolExecutor(1) as pool:
            task = pool.submit(handler._model_completion, "s", {}, {})
            try:
                self.assertTrue(queued.wait(1))
                state.set_policy(0)
                handler._backend_request.assert_not_called()
                self.assertFalse(task.done())
            finally:
                state.set_policy(1)
            task.result(2)
        handler._backend_request.assert_called_once()
        record = state.get("s").completions[0]
        self.assertEqual(record["metadata"]["policy_version"], 1)
        self.assertLessEqual(record["timing"]["policy_ready_at"], record["timing"]["admitted_at"])


class TestRegistrationRecovery(unittest.IsolatedAsyncioTestCase):
    """Accept replay of one management request while rejecting changed session identity."""
    async def test_lost_response_retries_same_registered_session(self) -> None:
        """A committed registration with a lost HTTP response remains exactly one session."""
        state = _State("http://backend", "qwen3", 2)
        program = object.__new__(CodexAgentProgram)
        program.admin_url = "http://gateway"
        program.admin_token = None
        calls = []
        body = {"session_id": "s", "policy_version": 3, "max_completions": 2}

        def request(method: str, url: str, payload: dict, timeout: float) -> dict:
            """Lose the first response only after the real registration has committed."""
            del method, url, timeout
            calls.append(payload["session_id"])
            state.register(payload["session_id"], {k: v for k, v in payload.items() if k != "session_id"})
            if len(calls) == 1:
                raise GatewayTransportError("response lost")
            return {"session_id": "s"}

        with patch("rl.agentic.codex.harness._http_json", side_effect=request):
            result = await program._admin_request("POST", "/internal/sessions", body, 2)
        self.assertEqual(result, {"session_id": "s"})
        self.assertEqual(calls, ["s", "s"])
        self.assertEqual(list(state.sessions), ["s"])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            state.register("s", {"policy_version": 4, "max_completions": 2})
        state.remove("s")
        state.remove("s")
        with self.assertRaisesRegex(ValueError, "released"):
            state.register("s", {"policy_version": 3, "max_completions": 2})

    async def test_original_registration_failure_is_persisted(self) -> None:
        """Even failing cleanup cannot replace the transport error in the evidence or caller."""
        with tempfile.TemporaryDirectory() as directory:
            program = CodexAgentProgram(PromptRecord("p", ()), 0, 0, "http://gateway", {
                "session_root": directory, "max_turns": 2,
                "reward_callable": "examples.gsm8k.agent:score_codex_gsm8k_answer",
            }, None)
            program._generation_settings = lambda: {}
            original = GatewayTransportError("connection reset during registration")
            program._admin_request = AsyncMock(side_effect=[original, RuntimeError("cleanup failed")])
            with self.assertRaises(GatewayTransportError) as caught:
                await program.run()
            self.assertIs(caught.exception, original)
            evidence = next(Path(directory).glob("*/performance.jsonl"))
            events = [json.loads(line) for line in evidence.read_text().splitlines()]
            failure = next(event for event in events if event["event"] == "program.failed")
            self.assertEqual(failure["error_type"], "GatewayTransportError")
            self.assertIn("connection reset", failure["error"])

    async def test_only_transport_failures_get_bounded_retries(self) -> None:
        """HTTP rejection is immediate; disconnect retries have short finite deadlines."""
        for error, attempts in ((RuntimeError("HTTP 400"), 1), (GatewayTransportError("reset"), 3)):
            with self.subTest(error=type(error).__name__):
                program = object.__new__(CodexAgentProgram)
                program._admin_request_once = AsyncMock(side_effect=error)
                with self.assertRaises(type(error)):
                    await program._admin_request("POST", "/internal/sessions", {}, 7200)
                self.assertEqual(program._admin_request_once.await_count, attempts)
                self.assertTrue(all(call.args[-1] == 30.0 for call in program._admin_request_once.call_args_list))
