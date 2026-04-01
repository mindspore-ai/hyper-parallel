# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test utils"""
import os
import subprocess
import sys

import torch
import torch.distributed as dist


def init_dist():
    """init dist"""
    if not dist.is_initialized():
        dist.init_process_group()
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    device_id = rank % 8
    torch.npu.set_device(device_id)
    return rank, device_id


def torchrun_case(file_name, case_name, master_port, num_proc=8):
    """Spawn *num_proc* workers via ``python -m torch.distributed.run`` (same as torchrun).

    Uses :data:`sys.executable` so conda/env interpreters work when ``torchrun`` is not on ``PATH``.
    """
    env = os.environ.copy()
    env.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
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
        f"{file_name}::{case_name}",
    ]
    ret = subprocess.call(cmd, env=env)
    assert ret == 0
