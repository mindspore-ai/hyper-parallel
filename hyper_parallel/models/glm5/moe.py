# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""GLM5 mixture-of-experts modules."""
import importlib

import torch
from torch import nn
from torch.nn import functional as F


def _run_experts_for_loop(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    hidden_states: torch.Tensor,
    num_tokens_per_expert: list[int],
) -> torch.Tensor:
    """Run packed experts with one batched matmul per non-empty expert."""
    outputs = []
    offset = 0
    for expert_idx, count in enumerate(num_tokens_per_expert):
        if count == 0:
            continue
        current_state = hidden_states[offset:offset + count]
        gate, up = F.linear(
            current_state, gate_up_proj[expert_idx],
        ).chunk(2, dim=-1)
        current_state = F.silu(gate) * up
        outputs.append(F.linear(current_state, down_proj[expert_idx]))
        offset += count
    if not outputs:
        return hidden_states * 0.0
    return torch.cat(outputs, dim=0)


def _run_experts_grouped_mm_npu(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    hidden_states: torch.Tensor,
    num_tokens_per_expert: list[int],
) -> torch.Tensor:
    """Run packed experts with Ascend grouped matmul kernels."""
    torch_npu = importlib.import_module("torch_npu")

    expert_inputs = list(torch.split(hidden_states, num_tokens_per_expert, dim=0))
    gate_proj, up_proj = gate_up_proj.chunk(2, dim=1)
    gate_weights = [
        gate_proj[idx].T.contiguous() for idx in range(gate_proj.shape[0])
    ]
    up_weights = [
        up_proj[idx].T.contiguous() for idx in range(up_proj.shape[0])
    ]
    down_weights = [
        down_proj[idx].T.contiguous() for idx in range(down_proj.shape[0])
    ]

    gate_outputs = torch_npu.npu_grouped_matmul(
        expert_inputs, gate_weights, group_type=-1,
    )
    up_outputs = torch_npu.npu_grouped_matmul(
        expert_inputs, up_weights, group_type=-1,
    )
    hidden_outputs = [
        F.silu(gate) * up for gate, up in zip(gate_outputs, up_outputs)
    ]
    output_list = torch_npu.npu_grouped_matmul(
        hidden_outputs, down_weights, group_type=-1,
    )
    return torch.cat(output_list, dim=0)


class GLM5MoEExperts(nn.Module):
    """Packed GLM5 experts with an expert-major forward interface."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        """Initialize packed expert parameters."""
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Run local experts on expert-major routed tokens."""
        gate_up_proj = (
            self.gate_up_proj.to_local()
            if hasattr(self.gate_up_proj, "to_local")
            else self.gate_up_proj
        )
        down_proj = (
            self.down_proj.to_local()
            if hasattr(self.down_proj, "to_local")
            else self.down_proj
        )
        if num_tokens_per_expert.ndim != 1:
            raise ValueError("num_tokens_per_expert must be a 1-D tensor")
        if num_tokens_per_expert.shape[0] != gate_up_proj.shape[0]:
            raise ValueError(
                "num_tokens_per_expert length must match local experts"
            )
        if hidden_states.shape[0] == 0:
            return hidden_states * 0.0
        tokens_per_expert = num_tokens_per_expert.tolist()
        if sum(tokens_per_expert) != hidden_states.shape[0]:
            raise ValueError(
                "sum(num_tokens_per_expert) must match routed token count"
            )

        if hidden_states.device.type == "npu":
            return _run_experts_grouped_mm_npu(
                gate_up_proj, down_proj, hidden_states, tokens_per_expert,
            )
        return _run_experts_for_loop(
            gate_up_proj, down_proj, hidden_states, tokens_per_expert,
        )


class GLM5MoE(nn.Module):
    """GLM5 top-k router with expert-major token dispatch boundaries."""

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        router_type: str = "softmax",
        n_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
    ) -> None:
        """Initialize the router and packed experts."""
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_type = router_type
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.gate = (
            GLM5MoERouter(hidden_size, num_experts)
            if router_type == "glm_moe_dsa"
            else nn.Linear(hidden_size, num_experts, bias=False)
        )
        self.experts = GLM5MoEExperts(
            num_experts, hidden_size, moe_intermediate_size,
        )
        self.shared_experts = (
            GLM5SharedExperts(
                hidden_size,
                moe_intermediate_size * n_shared_experts,
            )
            if n_shared_experts > 0
            else None
        )

    def _forward_softmax(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the original softmax top-k expert router."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        router_logits = self.gate(flat_states)
        topk_logits, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1,
        )
        topk_weights = F.softmax(
            topk_logits, dim=-1, dtype=torch.float32,
        ).to(hidden_states.dtype)

        flat_experts = selected_experts.flatten()
        permutation = flat_experts.argsort(stable=True)
        token_indices = permutation // self.top_k
        routed_states = flat_states[token_indices]
        tokens_per_expert = torch.bincount(
            flat_experts, minlength=self.num_experts,
        )

        expert_output = self.experts(routed_states, tokens_per_expert)
        sorted_weights = topk_weights.flatten()[permutation]
        expert_output = expert_output * sorted_weights.unsqueeze(-1)
        combined = torch.zeros(
            flat_states.shape,
            dtype=expert_output.dtype,
            device=expert_output.device,
        ).scatter_add(
            0,
            token_indices.unsqueeze(-1).expand(-1, hidden_size),
            expert_output,
        )
        return combined.view(batch_size, seq_len, hidden_size)

    def _select_glm_moe_dsa_experts(
        self,
        router_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select experts using the Transformers GLM-MoE-DSA router rule."""
        scores_for_choice = router_scores
        correction = getattr(self.gate, "e_score_correction_bias", None)
        if correction is not None:
            scores_for_choice = scores_for_choice + correction
        if self.n_group > 1:
            group_scores = scores_for_choice.view(
                -1, self.n_group, self.num_experts // self.n_group,
            )
            group_scores = group_scores.topk(2, dim=-1)[0].sum(dim=-1)
            group_indices = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False,
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_indices, 1)
            score_mask = group_mask.unsqueeze(-1).expand(
                -1, self.n_group, self.num_experts // self.n_group,
            ).reshape(-1, self.num_experts)
            scores_for_choice = scores_for_choice.masked_fill(
                ~score_mask.bool(), float("-inf"),
            )
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False,
        )[1]
        topk_weights = router_scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def _forward_glm_moe_dsa(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the official GLM-MoE-DSA sigmoid router."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        router_logits = F.linear(flat_states.float(), self.gate.weight.float())
        router_scores = router_logits.sigmoid()
        topk_indices, topk_weights = self._select_glm_moe_dsa_experts(router_scores)
        final_hidden_states = torch.zeros_like(flat_states)

        gate_up_proj = (
            self.experts.gate_up_proj.to_local()
            if hasattr(self.experts.gate_up_proj, "to_local")
            else self.experts.gate_up_proj
        )
        down_proj = (
            self.experts.down_proj.to_local()
            if hasattr(self.experts.down_proj, "to_local")
            else self.experts.down_proj
        )
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.num_experts,
        ).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_slots, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = flat_states[token_idx]
            gate, up = F.linear(
                current_state, gate_up_proj[expert_idx],
            ).chunk(2, dim=-1)
            current_state = F.silu(gate) * up
            current_state = F.linear(current_state, down_proj[expert_idx])
            current_state = current_state * topk_weights[token_idx, expert_slots, None]
            final_hidden_states.index_add_(0, token_idx, current_state)
        output = final_hidden_states.view(batch_size, seq_len, hidden_size)
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route tokens, run experts, and combine weighted outputs."""
        if self.router_type == "glm_moe_dsa":
            return self._forward_glm_moe_dsa(hidden_states)
        return self._forward_softmax(hidden_states)


class GLM5SharedExperts(nn.Module):
    """Shared GLM-MoE-DSA experts."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """Initialize shared expert projections."""
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the shared expert MLP."""
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(
            hidden_states,
        )
        return self.down_proj(hidden_states)


class GLM5MoERouter(nn.Module):
    """Official GLM-MoE-DSA router parameters."""

    def __init__(self, hidden_size: int, num_experts: int) -> None:
        """Initialize router score parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)


__all__ = ["GLM5MoE", "GLM5MoEExperts"]
