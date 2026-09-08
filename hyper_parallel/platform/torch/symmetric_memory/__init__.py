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
"""Torch Symmetric memory module for hyper-parallel."""

from importlib import import_module as _import_module  # pylint: disable=invalid-name
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lifecycle import acquire_symmetric_memory
    from .symmetric_memory import TorchSymmetricMemoryHandler

__all__ = [
    "TorchSymmetricMemoryHandler",
    "acquire_symmetric_memory",
]

_LAZY_EXPORTS = {
    "TorchSymmetricMemoryHandler": ".symmetric_memory",
    "acquire_symmetric_memory": ".lifecycle",
}


def __getattr__(name):  # pylint: disable=invalid-name
    """Lazily import the requested Torch symmetric-memory symbol."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():  # pylint: disable=invalid-name
    """Include lazy Torch symmetric-memory exports in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
