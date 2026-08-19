# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Following codes are inspired from https://github.com/volcengine/verl/blob/main/verl/utils/device.py

import logging
from typing import Any

import torch

# Importing torch_npu registers the ``torch.npu`` namespace and HCCL backend.
# The dependency is optional on CUDA-only and CPU-only installations.
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


logger = logging.getLogger(__name__)


IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = torch_npu is not None and torch.npu.is_available()

if IS_NPU_AVAILABLE:
    torch.npu.config.allow_internal_format = False


def get_device_type() -> str:
    """Get device type based on current machine, currently only support CPU, CUDA, NPU."""
    if IS_CUDA_AVAILABLE:
        device = "cuda"
    elif IS_NPU_AVAILABLE:
        device = "npu"
    else:
        device = "cpu"

    return device


def get_device_name() -> str:
    """Get real device name, e.g. A100, H100"""
    return get_torch_device().get_device_name()


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda or torch.npu"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load 'torch.cuda'.")
        return torch.cuda


def get_device_id() -> int:
    """Get current device id based on device type."""
    return get_torch_device().current_device()


def get_dist_comm_backend() -> str:
    """Return distributed communication backend type based on device type."""
    if IS_CUDA_AVAILABLE:
        return "nccl"
    elif IS_NPU_AVAILABLE:
        return "hccl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {get_device_type()}.")


def synchronize() -> None:
    """Execute torch synchronize operation."""
    get_torch_device().synchronize()


def stream_synchronize() -> None:
    """Execute device stream synchronize operation."""
    if IS_CUDA_AVAILABLE:
        torch.cuda.current_stream().synchronize()
    elif IS_NPU_AVAILABLE:
        torch.npu.current_stream().synchronize()
    else:
        synchronize()


def empty_cache() -> None:
    """Execute torch empty cache operation."""
    get_torch_device().empty_cache()


def set_device(device: torch.types.Device) -> None:
    """Execute set device operation."""
    get_torch_device().set_device(device)


def get_device_rng_state() -> Any:
    """Return the current accelerator RNG state, or ``None`` when unavailable.

    Checkpointing needs the device RNG alongside the CPU one: dropout,
    ``torch.randn`` initializers and any device-side sampling draw from it, so a
    resume that only restores ``torch.get_rng_state()`` diverges from the
    original run.
    """
    if get_device_type() == "cpu":
        return None
    get_rng_state = getattr(get_torch_device(), "get_rng_state", None)
    if get_rng_state is None:
        return None
    return get_rng_state()


def set_device_rng_state(state: Any) -> None:
    """Restore an accelerator RNG state captured by :func:`get_device_rng_state`."""
    if state is None or get_device_type() == "cpu":
        return
    set_rng_state = getattr(get_torch_device(), "set_rng_state", None)
    if set_rng_state is None:
        logger.warning("Device namespace has no set_rng_state; device RNG not restored.")
        return
    set_rng_state(state)


def is_nccl_backend(backend: str | None = None) -> bool:
    """Check if the distributed communication backend is NCCL."""
    return (backend or get_dist_comm_backend()) == "nccl"


def is_hccl_backend() -> bool:
    """Check if the distributed communication backend is HCCL."""
    return get_dist_comm_backend() == "hccl"


def get_gpu_compute_capability(device: torch.types.Device | int | None = None) -> int:
    """Return the compute capability as an integer (e.g. 70, 80, 90), or 0 if no GPU."""
    if not IS_CUDA_AVAILABLE:
        return 0
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def is_sm90_or_above() -> bool:
    """Check if the current CUDA device has SM90+ capability."""
    return get_gpu_compute_capability() >= 90


def get_compute_units():
    """
    Returns the number of streaming multiprocessors (SMs) or equivalent compute units
    for the available accelerator. Assigns the value to NUM_SMS.
    """
    NUM_SMS = None
    device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")

    # Use match/case for device-specific logic (Python 3.10+)
    match device_type:
        case "cuda":
            device_properties = torch.cuda.get_device_properties(0)
            NUM_SMS = device_properties.multi_processor_count
        case "xpu":
            device_properties = torch.xpu.get_device_properties(0)
            NUM_SMS = device_properties.max_compute_units
        case _:
            print("No CUDA or XPU device available. Using CPU.")
            # For CPU, you might want to use the number of CPU cores
            NUM_SMS = torch.get_num_threads()

    return NUM_SMS
