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
"""DeepSeek-V3 replacement factories for low-precision Linear and MoE.

This adapter owns model replacement policy and topology checks. The reusable
compute shells and strategies remain in the quantization components.
"""

from collections.abc import Mapping
from typing import Any

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional import build_low_precision_strategy
from hyper_parallel.components.quantization.functional.linear_strategy_factory import (
    build_linear_strategy,
)
from hyper_parallel.components.quantization.modules.grouped_experts import (
    GroupedExperts,
)
from hyper_parallel.components.quantization.modules.linear import LowPrecisionLinear
from hyper_parallel.models.replacement import module_replacement


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
def replace_linear(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> LowPrecisionLinear:
    """Replace one exact Dense Linear using its resolved low-precision policy."""

    if type(module) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
        raise TypeError(
            f"{module_fqn!r} must be exact nn.Linear, got {type(module).__name__}"
        )
    if context.get("pp"):
        raise NotImplementedError(
            "Low-precision Linear training is not yet supported with pipeline parallelism."
        )
    try:
        strategy = build_linear_strategy(
            context.get("low_precision"),
            in_features=module.in_features,
            out_features=module.out_features,
        )
    except ValueError as error:
        raise ValueError(
            f"Low-precision Linear target {module_fqn!r} is invalid: {error}"
        ) from error
    return LowPrecisionLinear.from_linear(module, strategy=strategy)


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


__all__ = ["replace_linear", "replace_grouped_experts"]
