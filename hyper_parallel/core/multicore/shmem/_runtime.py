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
"""Lazy access to the private Torch SHMEM Runtime binding."""

from __future__ import annotations

import importlib
from typing import Any, Final


_NATIVE_MODULE: Final = "hyper_parallel_shmem_torch"

_native: Any | None = None


def _load_native() -> Any:
    """Import and cache the private binding after the first successful load."""
    global _native  # pylint: disable=global-statement

    if _native is None:
        native = importlib.import_module(_NATIVE_MODULE)
        _native = native
    return _native


def _torch_modules() -> tuple[Any, Any]:
    """Load Torch only when the internal SHMEM boundary is used."""
    import torch  # pylint: disable=C0415
    import torch.distributed as dist  # pylint: disable=C0415

    return torch, dist


__all__ = [
    "_load_native",
    "_torch_modules",
]
