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
"""Torch worker-side test utils (imports torch — do not use from pytest launchers).

For spawning ``torchrun`` from a launcher / ``parallel_case`` wrapper, import
:func:`tests.common.distributed_launcher.torchrun_case` instead. That module
intentionally avoids importing ``torch`` / ``torch_npu``.
"""
import os
from typing import Union

import torch
import torch.distributed as dist


def init_dist() -> tuple[int, int]:
    """init dist"""
    if not dist.is_initialized():
        dist.init_process_group()
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    device_id = rank % 8
    torch.npu.set_device(device_id)
    return rank, device_id


def init_dist_gloo() -> int:
    """Init dist with gloo CPU backend (no NPU required)."""
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    return rank


def init_backend(device_type: str) -> Union[tuple[int, int], int]:
    """Initialize distributed backend for the given device type.

    Args:
        device_type: "npu" for Ascend NPU (hccl backend), "cpu" for Gloo CPU backend.

    Returns:
        Same as init_dist() for NPU or init_dist_gloo() for CPU.
    """
    if device_type == "npu":
        return init_dist()
    return init_dist_gloo()


def to_device(tensor: torch.Tensor, device_type: str) -> torch.Tensor:
    """Move tensor to the appropriate device based on device_type.

    Args:
        tensor: Input tensor to move.
        device_type: "npu" moves to NPU, "cpu" keeps on CPU.

    Returns:
        Tensor on the target device.
    """
    if device_type == "npu":
        return tensor.npu()
    return tensor


def get_device_type() -> str:
    """Return the device type for torch distributed tests.

    Checks HYPER_PARALLEL_TEST_DEVICE_TYPE first. If unset, detects
    torch_npu availability and returns "npu" or "cpu".

    Returns:
        "npu" or "cpu"

    Raises:
        ValueError: If HYPER_PARALLEL_TEST_DEVICE_TYPE is set to an unsupported value.
    """
    env_val = os.environ.get("HYPER_PARALLEL_TEST_DEVICE_TYPE", "").strip().lower()
    if env_val:
        if env_val not in ("npu", "cpu"):
            raise ValueError(
                f"HYPER_PARALLEL_TEST_DEVICE_TYPE must be 'npu' or 'cpu', got {env_val!r}"
            )
        return env_val
    try:
        import torch_npu  # pylint: disable=C0415,W0611
        return "npu"
    except ImportError:
        return "cpu"


_DEVICE_TYPE = get_device_type()
