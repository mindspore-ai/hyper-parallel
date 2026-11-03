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
"""Packed K3 SiTU expert compute on the shared Ascend grouped-matmul primitive."""

import importlib

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.activation_situ import situ_packed


def _grouped_matmul(states: torch.Tensor, weight: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    if states.device.type != "npu":
        raise ValueError("K3 packed grouped experts require local Ascend NPU tensors")
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


def grouped_expert_slots(
    hidden_states: torch.Tensor,
    local_expert_indices: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    beta: float,
    linear_beta: float,
    situ_backend: str,
    permute_backend: str = "torch",
) -> torch.Tensor:
    """Run complete packed SiTU matrices on arbitrary incoming expert slots.

    Args:
        hidden_states: Received slots ``[rows, latent_width]`` in transport dtype.
        local_expert_indices: Local expert ID for each row; all IDs must be valid.
        weights: Complete local ``(w1, w2, w3)`` tensors, with expert axis first.
        beta: K3 gate saturation scale.
        linear_beta: K3 up-branch saturation scale.
        situ_backend: Existing explicit SiTU backend selection.
        permute_backend: Native sort/index-copy or native NPU permute/unpermute.

    Returns:
        Slot outputs in their original order and transport dtype. No routing
        probability is applied here. Empty slots retain every autograd edge.

    Note:
        The caller enters the expert module through ``__call__`` before reading
        these tensors, preserving nested FSDP unshard/reshard hooks. GMM returns
        ordinary gradients; the standard optimizer owns FP32 main gradients.
    """
    gate_weight, down_weight, up_weight = weights
    expert_count = _validate_grouped_inputs(hidden_states, local_expert_indices, weights)
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
    gate = _grouped_matmul(states, gate_weight.transpose(1, 2), ends)
    up = _grouped_matmul(states, up_weight.transpose(1, 2), ends)
    activated = situ_packed(torch.cat((gate, up), dim=-1), beta=beta,
                            linear_beta=linear_beta, backend=situ_backend)
    values = _grouped_matmul(activated, down_weight.transpose(1, 2), ends).to(hidden_states.dtype)
    if permute_backend == "npu":
        module = importlib.import_module("hyper_parallel.components.functional.moe_token_unpermute")
        unpermute = module.moe_token_unpermute
        return unpermute(values, order)
    return torch.zeros_like(hidden_states).index_copy(0, order, values)
