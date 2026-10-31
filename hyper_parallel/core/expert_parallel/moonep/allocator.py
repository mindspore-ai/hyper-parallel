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
"""Load the MoonEP placement kernels independently of expert execution."""
import ctypes
import importlib
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch


# Public enum from CANN's acl/acl_platform.h; the capacity itself is queried.
_ACL_PLATFORM_AICORE_UB_SIZE = 1


@lru_cache(maxsize=1)
def _platform_query() -> Any:
    """Resolve the active CANN runtime without starting the Python compiler stack."""
    library = ctypes.CDLL("libascendcl.so")
    query = getattr(library, "aclplatformGetDeviceInfo", None)
    if query is None:
        raise RuntimeError("MoonEP requires the CANN platform-query API matching its native build")
    query.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    query.restype = ctypes.c_int
    return query


@lru_cache(maxsize=None)
def _device_ub_bytes(device_index: int) -> int:
    """Read this device's UB size without mutating the caller's compiler SOC."""
    value = ctypes.create_string_buffer(32)
    with torch.npu.device(device_index):
        status = _platform_query()(_ACL_PLATFORM_AICORE_UB_SIZE, value, ctypes.sizeof(value))
    if status:
        raise RuntimeError(f"CANN UB query failed for device {device_index}: status={status}")
    try:
        size = int(value.value)
    except ValueError as error:
        raise RuntimeError(f"CANN returned an invalid UB size: {value.value!r}") from error
    if size <= 0:
        raise RuntimeError(f"CANN returned a nonpositive UB size for device {device_index}: {size}")
    return size


def native_library_path(filename: str) -> Path:
    """Locate an installed library or the source checkout's prepared native payload."""
    package = Path(__file__).resolve().parent
    installed = package / filename
    if installed.is_file():
        return installed
    default = package.parents[3] / "build/native/payload/hyper_parallel"
    payload = Path(os.getenv("HYPER_PARALLEL_NATIVE_OUTPUT_ROOT", str(default)))
    built = payload / "core/expert_parallel/moonep" / filename
    if not built.is_file():
        raise FileNotFoundError(
            f"MoonEP native library is missing: {built}. Build it with "
            "python -m hyper_parallel.core.expert_parallel.moonep.build "
            "(add --shared for the MegaMoE transport)."
        )
    return built


def load_native_extension(name: str) -> ModuleType:
    """Load a Python extension from an installed wheel or the source-build payload."""
    module_name = __package__ + "." + name
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
    spec = importlib.util.spec_from_file_location(module_name, native_library_path(name + ".so"))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MoonEP extension: {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


@lru_cache(maxsize=1)
def load_allocator() -> ModuleType:
    """Load the device allocator without initializing another SHMEM runtime."""
    return load_native_extension("_a3_allocator")


def allocate_experts(counts: torch.Tensor, capacity: int) -> list[torch.Tensor]:
    """Run expert placement with the current device's CANN-reported UB budget."""
    if counts.device.type != "npu":
        raise ValueError("MoonEP placement requires counts on an Ascend NPU")
    return load_allocator().allocate(counts, capacity, _device_ub_bytes(counts.device.index))
