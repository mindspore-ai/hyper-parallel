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
"""Independent CPU FP32 math oracle for unclamped DSV4.1 MoE validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.nn import functional


def _linear(hidden: torch.Tensor, parameters: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    output = hidden @ parameters[f"{prefix}.weight"].T
    bias = parameters.get(f"{prefix}.bias")
    return output if bias is None else output + bias


def evaluate_fp32_moe(
    hidden_states: torch.Tensor,
    output_gradient: torch.Tensor,
    parameters: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    learned_routing: bool,
    scaling_factor: float,
    scoring_func: str = "sqrtsoftplus",
) -> dict[str, Any]:
    """Differentiate a CPU FP32 MoE without calling HF or native expert code.

    Args:
        hidden_states: The BF16 path's actual inputs, copied to CPU FP32.
        output_gradient: Identical loss derivative supplied to both executions.
        parameters: Full HF-layout weights at the start of the current step.
        indices: The actual selected experts; discrete selection is held fixed.
        weights: Fixed route weights for the hotspot case.
        learned_routing: Recompute differentiable scores for the selected IDs.
        scaling_factor: Scaling applied once after TopK normalization.
        scoring_func: V4.1 score function for learned routing.

    Returns:
        CPU output, input/route gradients, parameter gradients, route values,
        and parameter values before the optimizer step. Unused parameters have
        no gradient. No BF16 intermediate rounding is simulated.
    """
    hidden = hidden_states.detach().to(device="cpu", dtype=torch.float32).requires_grad_()
    params = {name: value.detach().to(device="cpu", dtype=torch.float32).clone().requires_grad_()
              for name, value in parameters.items()}
    selected = indices.detach().to(device="cpu", dtype=torch.int64)
    flat = hidden.reshape(-1, hidden.shape[-1])
    if learned_routing:
        logits = flat @ params["gate.weight"].T
        if scoring_func == "sqrtsoftplus":
            scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
        elif scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        elif scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise ValueError(f"Unsupported oracle scoring function: {scoring_func!r}")
        route_weights = scores.gather(1, selected)
        if selected.shape[-1] > 1:
            route_weights = route_weights / (route_weights.sum(-1, keepdim=True) + 1e-20)
        route_weights = route_weights * scaling_factor
    else:
        route_weights = weights.detach().to(device="cpu", dtype=torch.float32).clone().requires_grad_()
    route_weights.retain_grad()
    routed = torch.zeros_like(flat)
    for expert_id in range(params["experts.gate_up_proj"].shape[0]):
        rows, slots = torch.where(selected == expert_id)
        gate, up = (flat[rows] @ params["experts.gate_up_proj"][expert_id].T).chunk(2, dim=-1)
        expert_output = (functional.silu(gate) * up) @ params["experts.down_proj"][expert_id].T
        routed = routed.index_add(0, rows, expert_output * route_weights[rows, slots, None])
    shared_gate = _linear(flat, params, "shared_experts.gate_proj")
    shared_up = _linear(flat, params, "shared_experts.up_proj")
    shared = _linear(functional.silu(shared_gate) * shared_up, params, "shared_experts.down_proj")
    output = (routed + shared).reshape_as(hidden)
    (output * output_gradient.detach().to(device="cpu", dtype=torch.float32)).sum().backward()
    return {
        "output": output.detach(), "input_grad": hidden.grad,
        "route_weights": route_weights.detach(), "route_weight_grad": route_weights.grad,
        "parameters": {name: value.detach() for name, value in params.items()},
        "gradients": {name: value.grad for name, value in params.items()},
    }
