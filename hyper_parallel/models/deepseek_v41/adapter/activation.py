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
"""DeepSeek-V4.1 activation configuration for unclamped precision validation."""

from types import MethodType

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.deepseek_v41.configuration import validate_swiglu_limit


def _unclamped_expert_gate(module: nn.Module, gate_up: torch.Tensor) -> torch.Tensor:
    """Use the source activation without clipping either projection."""
    gate, up = gate_up.chunk(2, dim=-1)
    return module.act_fn(gate) * up


def _unclamped_shared_forward(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply ordinary SwiGLU to the shared expert projections."""
    return module.down_proj(module.act_fn(module.gate_proj(hidden_states)) * module.up_proj(hidden_states))


def configure_deepseek_v41_swiglu(module: nn.Module) -> None:
    """Give a newly constructed HF MoE block the V4.1 zero-limit convention.

    Args:
        module: Unsharded MoE block with routed and shared experts.

    Raises:
        ValueError: If the two branches disagree or have an invalid limit.

    Note:
        Call before EP forward binding. HF 5.13 clamps unconditionally, so a
        zero limit needs explicit unclamped methods on both expert branches.
        Positive limits retain the original HF implementation and parameters.
    """
    routed_limit = validate_swiglu_limit(module.experts.limit)
    shared_limit = validate_swiglu_limit(module.shared_experts.limit)
    if routed_limit != shared_limit:
        raise ValueError("Routed and shared experts must use the same swiglu_limit")
    if routed_limit == 0:
        module.experts._apply_gate = MethodType(  # pylint: disable=protected-access
            _unclamped_expert_gate, module.experts
        )
        module.shared_experts.forward = MethodType(_unclamped_shared_forward, module.shared_experts)


__all__ = ["configure_deepseek_v41_swiglu"]
