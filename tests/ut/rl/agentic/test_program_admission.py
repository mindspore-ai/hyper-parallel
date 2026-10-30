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
"""Bounded external programs preserve ordering and stop admission after fatal errors."""

import asyncio
from types import SimpleNamespace
import unittest

from rl.agentic.core.program_runner import ProgramAgentRunner
from rl.dataset.contracts import PromptRecord


class TestProgramAdmission(unittest.IsolatedAsyncioTestCase):
    """Exercise concurrent scheduling without distributed initialization or model calls."""

    async def test_bound_and_submission_order(self) -> None:
        """Unequal program latency cannot reorder outputs or exceed the live limit."""
        state = {"active": 0, "peak": 0, "finished": []}

        async def run(index: int) -> int:
            """Force jobs to complete in a different order from submission."""
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            try:
                await asyncio.sleep(0.003 * (3 - index % 3))
                state["finished"].append(index)
                return index
            finally:
                state["active"] -= 1

        runner = ProgramAgentRunner(lambda _p, _v, i: SimpleNamespace(run=lambda: run(i)),
                                    12, None, max_concurrent_programs=3)
        result = await runner._run_bounded([(PromptRecord("p", ()), i) for i in range(12)], 7)
        self.assertEqual(result, list(range(12)))
        self.assertEqual(state["peak"], 3)
        self.assertEqual(state["active"], 0)
        self.assertNotEqual(state["finished"], result)

    async def test_failure_stops_queued_jobs_and_drains_started_jobs(self) -> None:
        """A failed episode cannot silently train or launch the remaining batch."""
        peer_started = asyncio.Event()
        first_failed = asyncio.Event()
        started, settled = [], []

        async def run(index: int) -> int:
            """Keep one already admitted peer alive while the first task fails."""
            started.append(index)
            try:
                if index == 0:
                    await peer_started.wait()
                    first_failed.set()
                    raise RuntimeError("registration disconnected")
                peer_started.set()
                await first_failed.wait()
                await asyncio.sleep(0)
                return index
            finally:
                settled.append(index)

        runner = ProgramAgentRunner(lambda _p, _v, i: SimpleNamespace(run=lambda: run(i)),
                                    20, None, max_concurrent_programs=2)
        with self.assertRaisesRegex(RuntimeError, "registration disconnected") as caught:
            await runner._run_bounded([(PromptRecord("p", ()), i) for i in range(20)], 0)
        self.assertEqual(started, [0, 1])
        self.assertEqual(sorted(settled), [0, 1])
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    async def test_cancellation_settles_active_programs(self) -> None:
        """Caller cancellation must await program cleanup before leaving the event loop."""
        both_started = asyncio.Event()
        never = asyncio.Event()
        started, settled = [], []

        async def run(index: int) -> None:
            """Retain live resources until the caller explicitly cancels the group."""
            started.append(index)
            if len(started) == 2:
                both_started.set()
            try:
                await never.wait()
            finally:
                await asyncio.sleep(0)
                settled.append(index)

        runner = ProgramAgentRunner(lambda _p, _v, i: SimpleNamespace(run=lambda: run(i)),
                                    5, None, max_concurrent_programs=2)
        task = asyncio.create_task(runner._run_bounded([(PromptRecord("p", ()), i) for i in range(5)], 0))
        await both_started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(sorted(settled), [0, 1])
        self.assertEqual(started, [0, 1])
