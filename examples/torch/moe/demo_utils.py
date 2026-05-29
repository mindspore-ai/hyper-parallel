# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Shared helpers for MoE EP demo scripts."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

TRAIN_STEPS = 10


def read_positive_int(name: str, default: str) -> int:
    """Parse a positive integer from an environment variable.

    Args:
        name: Environment variable name.
        default: Default value when the variable is unset.

    Returns:
        Parsed positive integer.

    Raises:
        ValueError: If the value is not an integer or is less than 1.
    """
    raw = os.environ.get(name, default).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1 (got {value}).")
    return value


def train_steps() -> int:
    """Return the training step count (``MOE_NUM_STEPS``, default ``10``).

    Raises:
        ValueError: If ``MOE_NUM_STEPS`` is not a positive integer.
    """
    return read_positive_int("MOE_NUM_STEPS", str(TRAIN_STEPS))


def init_dist() -> tuple[int, int, str]:
    """Initialize the process group and bind one device per rank.

    Returns:
        Tuple of ``(rank, world_size, device_type)``.

    Raises:
        ValueError: If ``MOE_DEVICE_TYPE`` is not ``npu`` or ``cuda``.
    """
    if not dist.is_initialized():
        dist.init_process_group()
    rank = dist.get_rank()
    world = dist.get_world_size()
    device_type = os.environ.get("MOE_DEVICE_TYPE", "npu").strip().lower()
    if device_type == "npu":
        torch.npu.set_device(rank)
    elif device_type == "cuda":
        torch.cuda.set_device(rank)
    else:
        raise ValueError(
            f"Unsupported MOE_DEVICE_TYPE={device_type!r} (use npu or cuda)."
        )
    return rank, world, device_type
