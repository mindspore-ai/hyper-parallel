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
"""Torch symmetric-memory operations backed by the shared lifecycle."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch

_is_shmem_available = False

_manager = None
_ops = None
_NATIVE_LOAD_LOCK = threading.Lock()


def _require_library() -> Path:
    """Locate the installed or source-build Torch symmetric-memory adapter."""
    module_path = Path(__file__).resolve()
    package_root = module_path.parents[3]
    relative_path = Path("core/multicore/shmem/lib/framework/torch/libaclshmem_torch.so")
    candidates = [package_root / relative_path]
    repository_root = module_path.parents[4]
    if (repository_root / "setup.py").is_file():
        candidates.insert(0, repository_root / "build/native/payload/hyper_parallel" / relative_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise ImportError(
        "[HP-NATIVE-PAYLOAD-MISSING] component=multicore/shmem framework=torch "
        f"searched={searched}. The required private SHMEM component is absent, or the local build failed; "
        "inspect the build log and run ./build.sh --multicore on for source/PYTHONPATH development."
    )


def _load_native() -> None:
    """Load the optional native adapter exactly once on first SHMEM use."""
    # pylint: disable=global-statement
    global _is_shmem_available, _manager, _ops
    if _manager is not None:
        return
    with _NATIVE_LOAD_LOCK:
        if _manager is not None:
            return
        file_path = str(_require_library())
        try:
            torch.ops.load_library(file_path)
            manager = torch.classes.SymmetricMemory.Manager()
            ops = torch.classes.SymmetricMemory.Ops()
        except (OSError, RuntimeError) as error:
            raise ImportError(
                "[HP-NATIVE-LOAD-FAILED] component=multicore/shmem framework=torch "
                f"library={file_path} error={error}. Check the Python/Torch/torch_npu/CANN "
                "version combination and build log."
            ) from error
        _manager = manager
        _ops = ops
        _is_shmem_available = True


def _get_manager() -> Any:
    """Return the native manager loaded by this binding module."""
    _load_native()
    return _manager



def get_ops() -> Any:
    """Return the native one-sided communication operations."""
    _load_native()
    return _ops
