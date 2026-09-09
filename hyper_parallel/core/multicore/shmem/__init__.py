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
"""Process-wide one-sided SHMEM communication for the MultiCore component."""

import sys as _sys
from pathlib import Path as _Path

from ._api import (
    all_gather,
    barrier,
    empty,
    free,
    get,
    put,
    signal,
    wait_signal,
)
from ._debug import debug_state
from ._lifecycle import acquire, release

__all__ = [
    "acquire",
    "all_gather",
    "barrier",
    "debug_state",
    "empty",
    "free",
    "get",
    "put",
    "release",
    "signal",
    "wait_signal",
]


def _register_native_module_path() -> None:
    """Register install directories of the native binding module on ``sys.path``.

    ``_runtime`` imports the pybind11 module ``hyper_parallel_shmem_torch`` by
    name through Python's standard module cache, while its shared library is
    installed outside the Python package tree. The candidate directories are
    registered here, before any boundary function triggers the lazy import;
    missing directories are skipped so importing this package never fails.
    """
    package_dir = _Path(__file__).resolve()
    candidates = [package_dir.parent / "lib" / "framework" / "torch"]
    repository_root = package_dir.parents[4]
    if (repository_root / "setup.py").is_file():
        candidates.insert(
            0,
            repository_root
            / "build/native/payload/hyper_parallel/core/multicore/shmem/lib/framework/torch",
        )
    for candidate in candidates:
        if candidate.is_dir():
            entry = str(candidate)
            if entry not in _sys.path:
                _sys.path.insert(0, entry)


_register_native_module_path()
