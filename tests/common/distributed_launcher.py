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
"""Lightweight helpers to spawn ``torchrun`` / ``msrun`` workers.

These helpers must **not** import ``torch``, ``torch_npu``, or ``mindspore``.
Pytest launchers and ``parallel_case`` run in a parent / wrapper process that
only needs to exec the distributed runner; pulling heavy frameworks into that
process pays startup cost twice (parent + workers) and can inflate ST wall time.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional

from tests.common.port_utils import allocate_port


def torchrun_case(
    file_name: str,
    case_name: str,
    master_port: Optional[int] = None,
    num_proc: int = 8,
) -> None:
    """Spawn *num_proc* workers via ``python -m torch.distributed.run``.

    Uses :data:`sys.executable` so conda/env interpreters work when ``torchrun``
    is not on ``PATH``.
    """
    env = os.environ.copy()
    env.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
    abs_file = os.path.abspath(file_name)
    max_attempts = 3
    for attempt in range(max_attempts):
        if master_port is None or attempt > 0:
            master_port = allocate_port()
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={num_proc}",
            f"--log-dir=./logs/{file_name}/{case_name}",
            "-r",
            "3",
            "--master_addr=127.0.0.1",
            f"--master_port={master_port}",
            "-m",
            "pytest",
            "-s",
            f"{abs_file}::{case_name}",
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return
        combined = result.stdout + result.stderr
        if "address already in use" not in combined.lower():
            print(combined, file=sys.stderr)
            assert result.returncode == 0, f"torchrun failed with exit code {result.returncode}"
        if attempt == max_attempts - 1:
            print(combined, file=sys.stderr)
            assert False, f"Port {master_port} still in use after {max_attempts} attempts"


def msrun_case(
    glog_v,
    file_name,
    case_name,
    master_port,
    worker_num=8,
    local_worker_num=8,
) -> None:
    """Spawn MindSpore distributed workers via ``msrun`` + pytest."""
    filename = file_name.split(".py")[0]
    log_path = f"./logs/{filename}/{case_name}"
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    cmd = (
        f"export GLOG_v={glog_v} && msrun --worker_num={worker_num} "
        f"--local_worker_num={local_worker_num} "
        f"--master_addr=127.0.0.1 --master_port={master_port} "
        f"--join=True --log_dir={log_path} pytest -s -v "
        f"{file_name}::{case_name}"
    )
    ret = os.system(cmd)
    assert ret == 0
