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
"""Public routed-expert configuration, family factory, and low-precision members."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
        GroupedExpertsConfig,
        GroupedExpertsModule,
    )
    from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts_family import GroupedExperts
    from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
        EPLowPrecisionGroupedExperts,
        LowPrecisionGroupedExperts,
        MoonEPLowPrecisionGroupedExperts,
        replace_fake_w4a8_grouped_experts,
        replace_grouped_experts_with_policy,
        replace_mxfp8_grouped_experts,
        replace_w4a8_grouped_experts,
    )

_LOW_PRECISION_EXPORTS = frozenset({
    "EPLowPrecisionGroupedExperts",
    "LowPrecisionGroupedExperts",
    "MoonEPLowPrecisionGroupedExperts",
    "replace_fake_w4a8_grouped_experts",
    "replace_grouped_experts_with_policy",
    "replace_mxfp8_grouped_experts",
    "replace_w4a8_grouped_experts",
})

_EXPORT_TO_MODULE = {
    "GroupedExperts": "grouped_experts_family",
    "GroupedExpertsConfig": "grouped_experts",
    "GroupedExpertsModule": "grouped_experts",
    **{name: "low_precision_grouped_experts" for name in _LOW_PRECISION_EXPORTS},
}

__all__ = [
    "EPLowPrecisionGroupedExperts",
    "GroupedExperts",
    "GroupedExpertsConfig",
    "GroupedExpertsModule",
    "LowPrecisionGroupedExperts",
    "MoonEPLowPrecisionGroupedExperts",
    "replace_fake_w4a8_grouped_experts",
    "replace_grouped_experts_with_policy",
    "replace_mxfp8_grouped_experts",
    "replace_w4a8_grouped_experts",
]


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
    """Lazily resolve routed-expert and low-precision exports without eager backend imports."""
    submodule = _EXPORT_TO_MODULE.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(
        f"hyper_parallel.components.modules.moe.grouped_experts.{submodule}"
    )
    value = getattr(module, name)
    globals()[name] = value
    return value
