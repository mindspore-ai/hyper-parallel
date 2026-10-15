# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Public interfaces for SAPP-ND memory estimation."""

import importlib
from typing import TYPE_CHECKING, Any

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hook_base import MemEvalHook, hook_runner
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory, Unit
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2, estimate_memory

# The estimator loads on first use. It reaches the cost model's config, whose
# variables import this package's size module, so loading it here closed an
# import cycle for whichever module of that ring a process imported first.
_ESTIMATOR = "hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2"
_ESTIMATOR_NAMES = ("EvaluatorV2", "estimate_memory")


# PEP 562 names this hook: a module-level dunder, which the naming rule for functions does not allow.
def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
    """Load the estimator's public names when they are first asked for (PEP 562).

    Args:
        name: The attribute looked up on this package.

    Returns:
        The estimator's class or function of that name.

    Raises:
        AttributeError: For a name this package does not provide.
    """
    if name in _ESTIMATOR_NAMES:
        return getattr(importlib.import_module(_ESTIMATOR), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EvaluatorV2",
    "LayerType",
    "MemEvalHook",
    "Memory",
    "Unit",
    "estimate_memory",
    "hook_runner",
]
