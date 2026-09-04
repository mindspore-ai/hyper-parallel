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

"""Trainer-side host/device memory helpers.

Split out of the former ``auto_models/components/utils/helper.py`` in stage 7
(05 §10.4); function names and signatures are unchanged.
"""

import gc

import psutil

from hyper_parallel.models.build_options import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    get_device_type,
    get_torch_device,
)
from hyper_parallel.trainer.runtime.logging import _info_rank0


def print_device_mem_info(prompt: str = "VRAM usage") -> None:
    """
    Logs VRAM info.
    """
    if get_device_type() == "cpu":
        print_cpu_memory_info()
    else:
        memory_allocated = get_torch_device().memory_allocated() / (1024**3)
        max_memory_allocated = get_torch_device().max_memory_allocated() / (1024**3)
        _info_rank0(f"{prompt}: cur {memory_allocated:.2f}GB, max {max_memory_allocated:.2f}GB.")


def print_cpu_memory_info() -> None:
    """Log CPU usage and system memory information on global rank zero."""
    cpu_usage = psutil.cpu_percent(interval=1)  # sampling for 1 sec
    _info_rank0(f"CPU Usage: {cpu_usage}%")

    memory_info = psutil.virtual_memory()
    _info_rank0(f"Total Memory: {memory_info.total / (1024**3):.2f} GB")
    _info_rank0(f"Available Memory: {memory_info.available / (1024**3):.2f} GB")
    _info_rank0(f"Used Memory: {memory_info.used / (1024**3):.2f} GB")
    _info_rank0(f"Memory Usage: {memory_info.percent}%")


def empty_cache() -> None:
    """
    Collects system memory.
    """
    gc.collect()

    if IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE:
        get_torch_device().empty_cache()


__all__ = [
    "empty_cache",
    "print_cpu_memory_info",
    "print_device_mem_info",
]
