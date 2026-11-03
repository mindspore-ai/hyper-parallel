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
"""K3 SiTU grouped expert compute using shared NPU primitives."""

import importlib

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.components.functional.activation_situ import situ


def grouped_experts(latent: torch.Tensor, indices: torch.Tensor, probabilities: torch.Tensor,
                    experts: nn.ModuleList, *, situ_backend: str, beta: float, linear_beta: float) -> torch.Tensor:
    """Compute routed local experts without replacing their checkpoint parameters.

    The differentiable stacks preserve w1/w2/w3 FQNs and optimizer ownership,
    but add transient weight storage. This EP1 path is not a packed-parameter
    deployment for the full expert population; EP packing needs its own adapter.
    Token combination uses FP32 values and probabilities like the eager model.

    Args:
        latent: Local NPU token representations with shape ``[tokens, hidden]``.
        indices: Selected local expert indices with shape ``[tokens, top_k]``.
        probabilities: Combination weights matching ``indices``.
        experts: Local expert modules exposing unsharded w1, w2, and w3 weights.
        situ_backend: SiTU implementation, ``eager`` or ``ascendc``.
        beta: Positive saturation scale for the expert gate projection.
        linear_beta: Saturation scale for the up projection; zero disables it.

    Returns:
        Weighted expert outputs in the latent input's shape and dtype.
    """
    if latent.device.type != "npu":
        raise ValueError("K3 grouped experts require Ascend NPU tensors")
    if isinstance(latent, DTensor) or any(isinstance(expert.w1.weight, DTensor) for expert in experts):
        raise TypeError("K3 grouped experts require local, unsharded expert parameters during compute")
    # Optional device primitives stay out of the portable model import path.
    gmm = importlib.import_module("hyper_parallel.components.functional.grouped_matmul").grouped_matmul
    permute = importlib.import_module("hyper_parallel.components.functional.moe_token_permute").moe_token_permute
    unpermute = importlib.import_module("hyper_parallel.components.functional.moe_token_unpermute").moe_token_unpermute
    packed, order = permute(latent, indices.to(torch.int32))
    flat_indices = indices.reshape(-1).long()
    counts = torch.zeros(len(experts), device=latent.device, dtype=torch.int64)
    counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices))
    ends = counts.cumsum(0)
    gate_weight = torch.stack([expert.w1.weight for expert in experts]).transpose(1, 2)
    up_weight = torch.stack([expert.w3.weight for expert in experts]).transpose(1, 2)
    down_weight = torch.stack([expert.w2.weight for expert in experts]).transpose(1, 2)
    gate = gmm(packed, gate_weight, group_list=ends)
    up = gmm(packed, up_weight, group_list=ends)
    activated = situ(gate, up, beta=beta, linear_beta=linear_beta, backend=situ_backend)
    output = gmm(activated, down_weight, group_list=ends)
    return unpermute(output.float(), order, probabilities.float()).to(latent.dtype)
