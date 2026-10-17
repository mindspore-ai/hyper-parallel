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
from textwrap import dedent
from uuid import uuid4

from examples.code.client import SandboxFusionExecutor
from examples.code.judge import judge_stdio


async def _validate_isolation(executor: SandboxFusionExecutor) -> None:
    """Require candidate privilege, dependency, network and request isolation."""
    marker = f"/tmp/sandbox-isolation-{uuid4().hex}"
    # These probes run through the candidate API so service health cannot mask
    # missing lite isolation or accidental use of the server's Python environment.
    code = dedent(f"""\
        import importlib.util
        import os
        from pathlib import Path
        import socket

        assert os.geteuid() != 0, 'candidate must not run as root'
        capabilities = [line for line in Path('/proc/self/status').read_text().splitlines()
                        if line.startswith('CapEff:')]
        assert len(capabilities) == 1 and int(capabilities[0].split()[1], 16) == 0, 'candidate has capabilities'
        for package in ('fastapi', 'torch', 'vllm'):
            assert importlib.util.find_spec(package) is None, 'unexpected candidate dependency: ' + package
        try:
            with socket.socket() as connection:
                connection.settimeout(0.2)
                connection.connect(('127.0.0.1', 8080))
        except OSError:
            pass
        else:
            raise AssertionError('candidate can reach the sandbox service')
        Path({marker!r}).write_text('candidate-local')
        print('isolated')
        """)
    result = await executor.run(code, "", request_id="st:isolation:create")
    assert result.status == "success" and result.stdout.strip() == "isolated", f"Isolation probe failed: {result}"
    code = dedent(f"""\
        from pathlib import Path

        assert not Path({marker!r}).exists(), 'candidate file survived into another request'
        print('clean')
        """)
    result = await executor.run(code, "", request_id="st:isolation:check")
    assert result.status == "success" and result.stdout.strip() == "clean", f"Request isolation failed: {result}"


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
        await _validate_isolation(executor)
    finally:
        await executor.close()


def test_contract() -> None:
    """Exercise the fixed Python-only SandboxFusion execution profile."""
    asyncio.run(_validate())
