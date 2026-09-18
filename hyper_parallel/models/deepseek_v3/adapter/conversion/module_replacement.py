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
from hyper_parallel.components.quantization.functional import (
    validate_hifloat8_gmm_runtime,
    validate_npu_gmm_runtime,
)
from hyper_parallel.components.quantization.modules.hifloat8_grouped_linear import (
    HiFloat8GroupedExperts,
)
from hyper_parallel.components.quantization.modules.mxfp8_grouped_linear import (
    MXFP8GroupedExperts,
)


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
def replace_mxfp8_grouped_experts(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> MXFP8GroupedExperts:
    """Replace one EP=1 packed gate/up/down expert container with MXFP8 GMMs."""

    _check_ep1_only(context, "MXFP8 grouped experts")
    parameters = tuple(module.parameters(recurse=False))
    if any(dimension % 32 for parameter in parameters for dimension in parameter.shape[-2:]):
        shapes = [tuple(parameter.shape) for parameter in parameters]
        raise ValueError(
            f"{module_fqn!r} is not MXFP8 tile aligned: {shapes} "
            "requires matrix dimensions that are multiples of 32."
        )
    validate_npu_gmm_runtime()
    return MXFP8GroupedExperts.from_module(module, fqn=module_fqn)


__all__ = ["replace_hifloat8_grouped_experts", "replace_mxfp8_grouped_experts"]
