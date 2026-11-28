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
"""Quantized module implementations (format-generic, model-agnostic)."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hyper_parallel.components.modules.linear.linear import LowPrecisionLinear
    from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
        LowPrecisionGroupedExperts as GroupedExperts,
    )

_EXPORT_TO_ATTR = {
    "GroupedExperts": (
        "hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts",
        "LowPrecisionGroupedExperts",
    ),
    "LowPrecisionLinear": (
        "hyper_parallel.components.modules.linear.linear",
        "LowPrecisionLinear",
    ),
}

__all__ = [
    "GroupedExperts",
    "LowPrecisionLinear",
]


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
    """Lazily resolve canonical low-precision module implementations."""
    target = _EXPORT_TO_ATTR.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = target
    value = getattr(importlib.import_module(module_path), attr_name)
    globals()[name] = value
    return value
