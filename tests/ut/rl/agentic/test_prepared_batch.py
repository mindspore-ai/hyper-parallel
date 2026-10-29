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
"""CPU preparation preserves ordering and owns cancellation of its background loop."""

import asyncio
import json
import os
import sys
import threading
import unittest

from rl.agentic.core.prepared_batch import PreparedProgramBatch
from rl.dataset.contracts import Message, PromptRecord


class TestPreparedProgramBatch(unittest.TestCase):
    """No model or distributed setup is required for preparation lifecycle checks."""

    @staticmethod
    def prompts() -> tuple[PromptRecord, ...]:
        """Return one immutable prompt for preparation lifecycle checks."""
        return (PromptRecord(prompt_id="p", messages=(Message("user", "question"),), ground_truth="42"),)

    def test_ordered_result_and_identity(self):
        """Prepared work is consumed only for the original policy and prompt content."""
        async def work() -> tuple[str, str]:
            """Return ordered results independently of scheduling."""
            return ("first", "second")
        batch = PreparedProgramBatch(self.prompts(), 4, work)
        try:
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                batch.result(self.prompts(), 3)
            changed = (PromptRecord(prompt_id="p", messages=(Message("user", "changed"),), ground_truth="42"),)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                batch.result(changed, 4)
            self.assertEqual(batch.result(self.prompts(), 4), ("first", "second"))
        finally:
            batch.close()

    def test_cancel_executes_async_cleanup(self):
        """Closing pending preparation joins the loop after all task cleanup."""
        entered, cleaned = threading.Event(), threading.Event()
        async def work() -> None:
            """Run cleanup when background work is cancelled."""
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()
        batch = PreparedProgramBatch(self.prompts(), 1, work)
        self.assertTrue(entered.wait(1))
        batch.close()
        self.assertTrue(cleaned.is_set())
        batch.close()

    def test_absolute_cpu_cap_and_fraction_use_stricter_limit(self) -> None:
        """Real child processes inherit both limits without changing the trainer mask."""
        original = os.sched_getaffinity(0)
        cases = ((0.5, None, max(1, len(original) // 2)), (0.5, 1, 1),
                 (0.5, len(original) + 1, max(1, len(original) // 2)),
                 (None, 2, min(2, len(original))), (1, len(original) + 1, len(original)))

        async def work() -> tuple[set[int], set[int]]:
            """Capture masks from the real asynchronous child process."""
            child = await asyncio.create_subprocess_exec(
                sys.executable, "-c", "import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))",
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(child.communicate(), timeout=10)
            self.assertEqual(child.returncode, 0)
            return os.sched_getaffinity(0), set(json.loads(stdout))

        for fraction, cap, expected in cases:
            with self.subTest(fraction=fraction, cap=cap):
                batch = PreparedProgramBatch(self.prompts(), 0, work, cpu_fraction=fraction, max_cpus=cap)
                try:
                    thread_mask, child_mask = batch.result(self.prompts(), 0)
                    self.assertEqual(thread_mask, child_mask)
                    self.assertEqual(len(thread_mask), expected)
                    self.assertTrue(thread_mask.issubset(original))
                    self.assertEqual(os.sched_getaffinity(0), original)
                finally:
                    batch.close()

    def test_invalid_cpu_limits_do_not_launch_work(self) -> None:
        """Invalid fraction or absolute cap must fail before preparation starts."""
        async def work() -> None:
            """Detect any task incorrectly launched after validation failure."""
            self.fail("Invalid CPU budget launched work")

        for key, values in (("cpu_fraction", (0, -0.1, 1.1, True, "half", float("nan"))),
                            ("max_cpus", (0, -1, True, 1.5, "64"))):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    PreparedProgramBatch(self.prompts(), 0, work, **{key: value})
