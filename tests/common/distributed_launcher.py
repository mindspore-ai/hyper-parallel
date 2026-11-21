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
"""Lightweight helpers to spawn ``torchrun`` workers.

These helpers must **not** import ``torch``, ``torch_npu``, or any training
framework. Pytest launchers and ``parallel_case`` run in a parent / wrapper
process that only needs to exec the distributed runner; pulling heavy
frameworks into that process pays startup cost twice (parent + workers) and
can inflate ST wall time.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from tests.common.port_utils import allocate_port


def torchrun_case(
    file_name: str,
    case_name: str | tuple[str, ...],
    master_port: Optional[int] = None,
    num_proc: int = 8,
) -> None:
    """Spawn *num_proc* workers via ``python -m torch.distributed.run``.

    Pass a tuple of pytest selectors to run cases from the same file in one
    worker session. Uses :data:`sys.executable` from the active environment.
    """
    abs_file = os.path.abspath(file_name)
    case_names = (case_name,) if isinstance(case_name, str) else case_name
    _run_node_ids([f"{abs_file}::{name}" for name in case_names], file_name, master_port, num_proc)


def torchrun_cases(node_ids: list[str], num_proc: int) -> None:
    """Run pytest node IDs from several worker files in one distributed session."""
    if not node_ids:
        raise ValueError("torchrun_cases requires at least one pytest node ID")
    file_name = node_ids[0].split("::", 1)[0]
    _run_node_ids(node_ids, file_name, None, num_proc)


def _run_node_ids(node_ids: list[str], file_name: str, master_port: Optional[int], num_proc: int) -> None:
    """Launch the worker session and collect per-rank diagnostics."""
    if not node_ids:
        raise ValueError("torchrun_case requires at least one pytest selector")
    env = os.environ.copy()
    log_name = node_ids[0].split("::", 1)[1]
    if len(node_ids) > 1:
        log_name = f"{log_name}_and_{len(node_ids) - 1}_more"
    max_attempts = 3
    for attempt in range(max_attempts):
        if master_port is None or attempt > 0:
            master_port = allocate_port()
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={num_proc}",
            f"--log-dir=./logs/{file_name}/{log_name}",
            # Keep rank logs and return worker tracebacks with failed CI output.
            "-t",
            "3",
            "--master_addr=127.0.0.1",
            f"--master_port={master_port}",
            "-m",
            "pytest",
            "-s",
            *node_ids,
        ]
        # Native runtime logs can contain bytes outside UTF-8; keep their diagnostics
        # without turning a successful worker exit into a decoding failure.
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
        )
        if result.returncode == 0:
            return
        combined = result.stdout + result.stderr
        if "address already in use" not in combined.lower():
            print(combined, file=sys.stderr)
            assert result.returncode == 0, f"torchrun failed with exit code {result.returncode}"
        if attempt == max_attempts - 1:
            print(combined, file=sys.stderr)
            assert False, f"Port {master_port} still in use after {max_attempts} attempts"
