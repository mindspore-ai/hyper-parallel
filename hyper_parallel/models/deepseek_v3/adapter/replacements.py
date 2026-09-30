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
"""DeepSeek-V3 packed-expert replacement factories (low-precision adapters).

Model recognition and parameter mapping for the DeepSeek-V3 packed
gate/up/down expert containers; the generic grouped-linear modules live in
``hyper_parallel.components.quantization.modules``. These factories require
TP=CP=EP=PP=1 and validate the NPU runtime before converting.
"""

from collections.abc import Mapping
from typing import Any

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.quantization.functional import build_low_precision_strategy
from hyper_parallel.components.quantization.modules.hifloat8_grouped_linear import (
    HiFloat8GroupedExperts,
)
from hyper_parallel.components.quantization.modules.grouped_experts import (
    GroupedExperts,
)
from hyper_parallel.components.quantization.ops import validate_hifloat8_gmm_runtime


def _check_ep1_only(context: Mapping[str, Any], factory_name: str) -> None:
    """Reject any active model-parallel axis (EP=1 packed experts only)."""
    active_model_parallel_axes = [
        axis.upper()
        for axis in ("tp", "cp", "ep", "pp")
        if context.get(axis)
    ]
    if active_model_parallel_axes:
        raise NotImplementedError(
            f"{factory_name} currently require TP=CP=EP=PP=1; "
            f"active axes: {active_model_parallel_axes}."
        )


@module_replacement
def replace_hifloat8_grouped_experts(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> HiFloat8GroupedExperts:
    """Replace one EP=1 packed expert container with HiFloat8 GMMs."""

    _check_ep1_only(context, "HiFloat8 grouped experts")
    validate_hifloat8_gmm_runtime()
    return HiFloat8GroupedExperts.from_module(module, fqn=module_fqn)


@module_replacement
def replace_grouped_experts(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> GroupedExperts:
    """Replace packed experts with the policy-selected grouped-linear strategy."""

    _check_ep1_only(context, "Low-precision grouped experts")
    parameters = tuple(module.parameters(recurse=False))
    strategy = build_low_precision_strategy(
        context.get("low_precision"),
        tile_shapes=tuple(tuple(parameter.shape) for parameter in parameters),
    )
    return GroupedExperts.from_module(
        module,
        fqn=module_fqn,
        grouped_linear=strategy,
    )


__all__ = ["replace_grouped_experts", "replace_hifloat8_grouped_experts"]
