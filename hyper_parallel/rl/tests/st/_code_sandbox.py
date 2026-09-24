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
"""Worker-side contract checks against an explicitly configured real sandbox."""

import asyncio
import os

from examples.code.client import SandboxFusionExecutor
from examples.code.judge import judge_stdio


async def _validate() -> None:
    """Require real positive and negative outcomes, without changing test labels."""
    executor = SandboxFusionExecutor(os.environ["RL_ST_SANDBOX_ENDPOINT"], run_timeout=1.0)
    tests = {"inputs": ["1 2\n", "4 5\n"], "outputs": ["3\n", "9\n"]}
    cases = [
        ("passed", "print(sum(map(int, input().split())))", 1.0),
        ("wrong_answer", "print(0)", 0.0),
        ("runtime_error", "raise ValueError('candidate failure')", 0.0),
        ("timeout", "while True: pass", 0.0),
        ("output_limit", "print('x' * (2 * 1024 * 1024))", 0.0),
    ]
    try:
        for status, code, expected in cases:
            reward = await judge_stdio(code, tests, executor, candidate_id=f"st:{status}",
                                       runtime_version=os.environ["RL_ST_SANDBOX_RUNTIME_VERSION"])
            assert reward.value == expected, f"Unexpected reward for {status}: {reward}"
            assert reward.metadata["status"] == status, f"Unexpected status for {status}: {reward}"
            assert reward.components["total"] == 2, "The grader must retain the full declared test set"
    finally:
        await executor.close()


def test_contract() -> None:
    """Exercise the fixed Python-only SandboxFusion execution profile."""
    asyncio.run(_validate())
