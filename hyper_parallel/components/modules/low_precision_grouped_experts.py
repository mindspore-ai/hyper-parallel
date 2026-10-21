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
"""Scheme-selected grouped experts using the existing QAT computation strategies."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from torch import Tensor, nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.quantization.functional.base_gmm_func import GroupedLinear, _GroupedLinearFunction
from hyper_parallel.components.quantization.functional.strategy_factory import _validate_tile_alignment
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.components.quantization.ops.npu_mxfp8 import validate_npu_gmm_runtime
from hyper_parallel.components.quantization.ops.npu_w4a8 import validate_w4a8_gmm_runtime
from hyper_parallel.components.quantization.ops.npu_fake_w4a8 import validate_fake_w4a8_gmm_runtime
from hyper_parallel.models.replacement import module_replacement


def _swiglu(projected: Tensor, *, limit: float | None = None) -> Tensor:
    """Preserve projection dtype and optionally apply the native DeepSeek V4/V4.1 clamp."""
    gate, up = projected.chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class LowPrecisionGroupedExperts(GroupedExperts):
    """Preserve high-performance expert layout/routing and bind one existing QAT strategy."""

    def __init__(self, *, module: nn.Module, strategy: GroupedLinear, module_fqn: str = "",
                 context: Mapping[str, Any]) -> None:
        """Build local expert computation from the already resolved dtype scheme.

        Args:
            module: Source packed experts.
            strategy: Computation strategy selected by the Family rule.
            module_fqn: Module path used in diagnostics.
            context: Existing replacement context containing the resolved low_precision policy.
        """
        prefix = f"{module_fqn or '<root>'}: LowPrecisionGroupedExperts"
        if any(context.get(axis) for axis in ("tp", "cp", "ep", "pp")) or context.get("ep_size", 1) != 1:
            raise NotImplementedError(f"{prefix} requires TP=CP=EP=PP=1")
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        activation = getattr(self.config, "hidden_act", "silu")
        if (not self.has_gate or self.add_bias or self.use_2d_experts
                or not self.router_gating_in_fp32 or activation not in ("silu", "swiglu")):
            raise ValueError(f"{prefix} requires 3D bias-free SwiGLU with routing scores after GMM2")
        self.grouped_linear = strategy
        _validate_tile_alignment(
            type(strategy).__name__, context["low_precision"].block_size,
            (tuple(self.gate_up_proj.shape), tuple(self.down_proj.shape)),
        )

    @staticmethod
    def _activation(module: nn.Module, config: Any, gated_linear_unit: bool) -> Any:
        """Adapt V4.1's native expert activation without retaining its source parameters."""
        if isinstance(module, DeepseekV4Experts):
            return partial(_swiglu, limit=module.limit)
        # Keep the parent's source validation and the existing non-fused SiLU/multiply numerics.
        GroupedExperts._activation(module, config, gated_linear_unit)
        return _swiglu

    def _grouped_gemm_expert_forward(self, gate_up_proj, down_proj, permuted,
                                     tokens_per_expert, permuted_probs):
        """Adapt layout at the shared strategy boundary; outer routing applies scores."""
        del permuted_probs
        projected = _GroupedLinearFunction.apply(
            permuted, gate_up_proj.transpose(-2, -1), tokens_per_expert, self.grouped_linear, 1)
        return _GroupedLinearFunction.apply(
            self.activation_func(projected), down_proj.transpose(-2, -1),
            tokens_per_expert, self.grouped_linear, 1)


@module_replacement
def replace_mxfp8_grouped_experts(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionGroupedExperts:
    """Bind the MXFP8GroupedLinear strategy selected by the Family rule.

    Args:
        module: Source packed experts.
        module_fqn: Module path for diagnostics.
        context: Replacement context containing the resolved dtype scheme.
    """
    validate_npu_gmm_runtime()
    return LowPrecisionGroupedExperts(
        module=module, module_fqn=module_fqn, context=context,
        strategy=MXFP8GroupedLinear(),
    )


@module_replacement
def replace_w4a8_grouped_experts(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionGroupedExperts:
    """Bind the W4A8GroupedLinear strategy selected by the Family rule.

    Args:
        module: Source packed experts.
        module_fqn: Module path for diagnostics.
        context: Replacement context containing the resolved dtype scheme.
    """
    validate_w4a8_gmm_runtime()
    return LowPrecisionGroupedExperts(
        module=module, module_fqn=module_fqn, context=context,
        strategy=W4A8GroupedLinear(context["low_precision"].block_size),
    )


@module_replacement
def replace_fake_w4a8_grouped_experts(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionGroupedExperts:
    """Bind the FakeW4A8GroupedLinear strategy selected by the Family rule.

    Args:
        module: Source packed experts.
        module_fqn: Module path for diagnostics.
        context: Replacement context containing the resolved dtype scheme.
    """
    validate_fake_w4a8_gmm_runtime()
    return LowPrecisionGroupedExperts(
        module=module, module_fqn=module_fqn, context=context,
        strategy=FakeW4A8GroupedLinear(context["low_precision"].block_size),
    )
