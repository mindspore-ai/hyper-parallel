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
# pylint: disable=undefined-all-variable
"""Tensor parallel declarative APIs (parallel styles and module parallelization)."""
from importlib import import_module as _import_module  # pylint: disable=invalid-name

from hyper_parallel.core.tensor_parallel.api import parallelize_module
from hyper_parallel.core.tensor_parallel.style import (
    ColwiseParallel,
    NoParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
from hyper_parallel.core.tensor_parallel.loss_parallel import (
    loss_parallel,
    is_loss_parallel_active,
)

# MC2 fused kernels import torch at module load; keep them off the eager path so
# MindSpore-only environments can import this package without torch installed.
_LAZY_EXPORTS = {
    "MC2Linear": ".mc2",
    "MC2ColwiseParallel": ".mc2_style",
    "MC2RowwiseParallel": ".mc2_style",
}


def __getattr__(name):  # pylint: disable=invalid-name
    """Lazily import MC2 symbols that require torch."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():  # pylint: disable=invalid-name
    """Include lazy MC2 exports in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    # Parallel styles
    "ColwiseParallel",
    "MC2ColwiseParallel",
    "MC2RowwiseParallel",
    "NoParallel",
    "ParallelStyle",
    "PrepareModuleInput",
    "PrepareModuleInputOutput",
    "PrepareModuleOutput",
    "RowwiseParallel",
    "SequenceParallel",
    # MC2 fused linear
    "MC2Linear",
    # Module parallelization
    "parallelize_module",
    # Loss parallel
    "loss_parallel",
    "is_loss_parallel_active",
]
