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
"""Reusable expert-slot computation with optional device primitives."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch  # pylint: disable=forbidden-backend-import
from torch.utils.checkpoint import checkpoint  # pylint: disable=forbidden-backend-import


def _grouped_matmul(states: torch.Tensor, weight: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    if states.device.type != "npu":
        raise ValueError("Packed grouped experts require local Ascend NPU tensors")
    # Keep the optional torch_npu dependency outside portable model imports.
    operation = importlib.import_module("hyper_parallel.components.functional.grouped_matmul").grouped_matmul
    return operation(states, weight, group_list=ends, gemm_fusion=False)


def _validate_grouped_inputs(hidden_states: torch.Tensor, local_expert_indices: torch.Tensor,
                             weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> int:
    """Validate expert tensor ownership and return the complete local expert count."""
    gate_weight, down_weight, up_weight = weights
    if hidden_states.ndim != 2 or local_expert_indices.shape != hidden_states.shape[:1]:
        raise ValueError("Grouped expert slots require [rows, width] states and one expert index per row")
    if local_expert_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("Grouped expert indices must have int32 or int64 dtype")
    if any(weight.ndim != 3 for weight in weights):
        raise ValueError("Grouped expert weights must have three dimensions")
    expert_count, intermediate, width = gate_weight.shape
    if up_weight.shape != gate_weight.shape or down_weight.shape != (expert_count, width, intermediate):
        raise ValueError("Grouped w1/w3 and w2 matrices have incompatible expert dimensions")
    if not expert_count or hidden_states.shape[-1] != width:
        raise ValueError("Grouped experts need at least one local expert and the declared latent width")
    if any(weight.dtype != gate_weight.dtype or weight.device != hidden_states.device for weight in weights):
        raise ValueError("Grouped weights must share a compute dtype and the input device")
    return expert_count


def grouped_expert_mlp(states: torch.Tensor, weights: tuple[torch.Tensor, ...],
                       ends: torch.Tensor, *, activation: Callable[[torch.Tensor], torch.Tensor],
                       recompute: str = "none", matmul: Callable | None = None) -> torch.Tensor:
    """Apply grouped FC1, a full GLU, and FC2 to expert-major rows.

    Args:
        states: Expert-major tokens in the expert compute dtype.
        weights: Output/input matrices (w1, w2, w3) or (w13, w2).
        ends: Cumulative row counts for all local experts.
        activation: Callable consuming packed gate/up rows.
        recompute: none, activation, or fc1_activation; no routing is replayed.
        matmul: Optional existing GMM binding with the standard keyword contract.

    Returns:
        Expert-major outputs without routing probabilities or transport casts.
    """
    if recompute not in ("none", "activation", "fc1_activation"):
        raise ValueError("Expert recompute must be none, activation or fc1_activation")
    fused_fc1 = len(weights) == 2
    down_weight = weights[1]

    def _project(inputs: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Project expert-major rows with the caller-selected GMM binding."""
        if matmul is None:
            return _grouped_matmul(inputs, matrix.transpose(1, 2), ends)
        return matmul(inputs, matrix.transpose(1, 2), group_list=ends, group_type=0, group_list_type=0)

    def _project_and_activate(inputs: torch.Tensor, *matrices: torch.Tensor) -> torch.Tensor:
        """Apply FC1 and the complete GLU within the requested replay scope."""
        if fused_fc1:
            projected = _project(inputs, matrices[0])
        else:
            gate = _project(inputs, matrices[0])
            up = _project(inputs, matrices[1])
            projected = torch.cat((gate, up), dim=-1)
        if recompute == "activation" and torch.is_grad_enabled():
            return checkpoint(activation, projected, use_reentrant=False)
        return activation(projected)

    fc1_weights = (weights[0],) if fused_fc1 else (weights[0], weights[2])
    if recompute == "fc1_activation" and torch.is_grad_enabled():
        activated = checkpoint(_project_and_activate, states, *fc1_weights, use_reentrant=False)
    else:
        activated = _project_and_activate(states, *fc1_weights)
    return _project(activated, down_weight)


def grouped_expert_slots(
    hidden_states: torch.Tensor,
    local_expert_indices: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    *,
    activation: Callable[[torch.Tensor], torch.Tensor],
    permute_backend: str = "torch",
    recompute: str = "none",
) -> torch.Tensor:
    """Run complete packed expert matrices on arbitrary incoming expert slots.

    Args:
        hidden_states: Received slots ``[rows, latent_width]`` in transport dtype.
        local_expert_indices: Local expert ID for each row; all IDs must be valid.
        weights: Complete local (w1, w2, w3) or prepacked (w13, w2) tensors.
        activation: Complete packed-GLU function consuming concatenated gate/up rows.
        permute_backend: Native sort/index-copy or native NPU permute/unpermute.
        recompute: none, activation, or fc1_activation; no collectives or router are replayed.

    Returns:
        Slot outputs in their original order and transport dtype. No routing
        probability is applied here. Empty slots retain every autograd edge.

    Note:
        The caller enters the expert module through ``__call__`` before reading
        these tensors, preserving nested FSDP unshard/reshard hooks. GMM returns
        ordinary gradients; the standard optimizer owns FP32 main gradients.
    """
    fused_fc1 = len(weights) == 2
    if fused_fc1:
        gate_weight, up_weight = weights[0].chunk(2, dim=1)
        down_weight = weights[1]
    else:
        gate_weight, down_weight, up_weight = weights
    expert_count = _validate_grouped_inputs(hidden_states, local_expert_indices,
                                            (gate_weight, down_weight, up_weight))
    if hidden_states.shape[0] == 0:
        # Avoid a zero-grid kernel while retaining a zero gradient for every shard.
        zero = sum(weight[:0].sum() for weight in weights)
        return hidden_states + zero.to(hidden_states.dtype)
    if permute_backend not in ("torch", "npu"):
        raise ValueError("Packed expert permutation must be torch or npu")
    if permute_backend == "npu":
        # Optional NPU bindings remain outside portable model imports.
        permute = importlib.import_module("hyper_parallel.components.functional.moe_token_permute").moe_token_permute
        states, order = permute(hidden_states.to(gate_weight.dtype), local_expert_indices[:, None].to(torch.int32))
    else:
        keys = local_expert_indices.float() if expert_count <= 2 ** 24 else local_expert_indices
        order = keys.argsort(stable=True)
        states = hidden_states[order].to(gate_weight.dtype)
    indices = local_expert_indices.long()
    counts = torch.zeros(expert_count, device=indices.device, dtype=torch.int64)
    ends = counts.scatter_add_(0, indices, torch.ones_like(indices)).cumsum(0)
    values = grouped_expert_mlp(states, weights, ends, activation=activation,
                                recompute=recompute).to(hidden_states.dtype)
    if permute_backend == "npu":
        module = importlib.import_module("hyper_parallel.components.functional.moe_token_unpermute")
        unpermute = module.moe_token_unpermute
        return unpermute(values, order)
    return torch.zeros_like(hidden_states).index_copy(0, order, values)


def grouped_routed_experts(states: torch.Tensor, indices: torch.Tensor, probabilities: torch.Tensor,
                           weights: tuple[torch.Tensor, ...], *,
                           activation: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Permute local tokens, evaluate experts and combine in FP32.

    Args:
        states: Local expert-space token rows.
        indices: Selected expert IDs per token/slot.
        probabilities: Mixture weights, applied after expert evaluation.
        weights: Complete (w1, w2, w3) or (w13, w2) matrices.
        activation: Full packed-GLU callable.

    Returns:
        Weighted expert values in the input dtype and token order.
    """
    # These device bindings are optional for portable reference consumers.
    permute = importlib.import_module("hyper_parallel.components.functional.moe_token_permute").moe_token_permute
    unpermute = importlib.import_module("hyper_parallel.components.functional.moe_token_unpermute").moe_token_unpermute
    packed, order = permute(states, indices.to(torch.int32))
    flat_indices = indices.reshape(-1).long()
    counts = torch.zeros(weights[0].shape[0], device=states.device, dtype=torch.int64)
    counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices))
    output = grouped_expert_mlp(packed, weights, counts.cumsum(0), activation=activation)
    return unpermute(output.float(), order, probabilities.float()).to(states.dtype)
