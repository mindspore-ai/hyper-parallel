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
"""Regression tests for completed and uncertain session lifecycles."""
import asyncio
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rl.agentic.codex import gateway as codex
from rl.agentic.ds_harness import gateway as deepseek
from rl.agentic.ds_harness import harness


class TestReleasedSessions(unittest.TestCase):
    """Keep lifetime duplicate protection without a growing Python ID collection."""

    def test_old_tombstones_survive_with_a_fixed_page_cache(self) -> None:
        """A long release history still blocks the oldest delayed registration."""
        state = codex._State("http://unused", "model", 1)
        self.addCleanup(state.released_sessions.close)
        for index in range(10000):
            state.remove(str(index))
        self.assertEqual(state.sessions, {})
        self.assertIn("0", state.released_sessions)
        with self.assertRaisesRegex(ValueError, "already released"):
            state.register("0", {"policy_version": 1, "max_completions": 1})


    def test_deepseek_delete_blocks_a_late_registration(self) -> None:
        """A POST that commits after a timed-out DELETE cannot resurrect the session."""
        state = deepseek._State("http://unused", "model", 1)
        self.addCleanup(state.released_sessions.close)
        state.remove("late")
        state.remove("late")
        with self.assertRaisesRegex(ValueError, "already released"):
            state.register("late", {"policy_version": 1, "max_completions": 1})
        self.assertEqual(state.sessions, {})


class TestUncertainRegistration(unittest.IsolatedAsyncioTestCase):
    """Delete even when registration committed but no success response arrived."""

    async def test_failed_and_cancelled_registration_leave_no_session(self) -> None:
        """Cancellation waits for the worker POST before issuing idempotent DELETE."""
        for cancel in (False, True):
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as directory:
                state = deepseek._State("http://unused", "model", 1)
                started, release = threading.Event(), threading.Event()
                calls = []

                def request(method, _url, payload, _timeout):
                    calls.append(method)
                    if method == "POST":
                        started.set()
                        if cancel:
                            release.wait(3)
                        state.register(payload["session_id"], payload)
                        if not cancel:
                            raise RuntimeError("registration response lost")
                        return {}
                    state.remove(next(iter(state.sessions), "unknown"))
                    return {}

                program = SimpleNamespace(
                    config={"max_turns": 1}, policy_version=0, admin_url="http://unused",
                    _prepare_directories=lambda _sid: (Path(directory), Path(directory), Path(directory)),
                    _generation_settings=lambda: {},
                )
                with patch.object(harness, "_http_json", side_effect=request):
                    task = asyncio.create_task(harness.DeepSeekAgentProgram.run(program))
                    if cancel:
                        self.assertTrue(await asyncio.to_thread(started.wait, 2))
                        task.cancel()
                        await asyncio.sleep(0.01)
                        self.assertEqual(calls, ["POST"])
                        release.set()
                    with self.assertRaises(asyncio.CancelledError if cancel else RuntimeError):
                        await task
                self.assertEqual(calls, ["POST", "DELETE"])
                self.assertEqual(state.sessions, {})
                state.remove("unknown")
