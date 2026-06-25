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
"""Shared helpers for Llama3 Torch demo scripts."""
# pylint: disable=C0413
from __future__ import annotations

import os

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
import torch.distributed as dist

TRAIN_STEPS = 10


def read_positive_int(name: str, default: str) -> int:
    """Parse a positive integer from an environment variable."""
    raw = os.environ.get(name, default).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1 (got {value}).")
    return value


def train_steps() -> int:
    """Return the training step count (``LLAMA3_NUM_STEPS``, default ``10``)."""
    return read_positive_int("LLAMA3_NUM_STEPS", str(TRAIN_STEPS))


def init_dist() -> tuple[int, int, str]:
    """Initialize the process group and bind one device per rank."""
    if not dist.is_initialized():
        dist.init_process_group()
    rank = dist.get_rank()
    world = dist.get_world_size()
    device_type = os.environ.get("LLAMA3_DEVICE_TYPE", "npu").strip().lower()
    if device_type == "npu":
        torch.npu.set_device(rank)
    elif device_type == "cuda":
        torch.cuda.set_device(rank)
    else:
        raise ValueError(f"Unsupported LLAMA3_DEVICE_TYPE={device_type!r} (use npu or cuda).")
    return rank, world, device_type
