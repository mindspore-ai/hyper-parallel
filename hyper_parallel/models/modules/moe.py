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
"""Mixture-of-Experts building blocks.

Submodule named ``experts`` so a model-level ``_ep_modules = ["*.experts"]``
finds it for EP application.
"""
import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.modules.feed_forward import SwiGLUMLP


class MoEExperts(nn.Module):
    """Packed MoE experts: ``num_experts`` experts as 3-D parameters.

    Stores all expert weights in two 3-D parameters keyed on dim-0 by
    expert index. Forward iterates over the experts that received tokens
    (``expert_hit``) and applies SwiGLU per expert with per-token routing
    weights, then ``index_add_`` into the output buffer.

    Parameter naming follows the upstream Qwen3.5-MoE checkpoint::

        experts.gate_up_proj : (E, 2I, H)
        experts.down_proj    : (E, H, I)
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
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
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Buffer + sum(dim=1) uses fp32-internal accumulation, more stable
        # than ``index_add_`` (atomic add in output dtype).
        """Forward pass."""
        num_tokens, hidden_dim = hidden_states.shape
        top_k = top_k_index.size(-1)
        expert_outputs = torch.zeros(
            num_tokens, top_k, hidden_dim,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            expert_outputs[token_idx, top_k_pos] = current_hidden_states.to(expert_outputs.dtype)
        return expert_outputs.sum(dim=1)


class MoE(nn.Module):
    """MoE block: router + packed experts with top-k dispatch."""

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = MoEExperts(num_experts, hidden_size, moe_intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route input through top-k experts and return weighted output."""
        bsz, seq_len, h = x.shape
        x_flat = x.view(-1, h)

        scores = self.gate(x_flat)
        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1, dtype=torch.float32).to(x.dtype)

        out = self.experts(x_flat, topk_idx, topk_weights)
        return out.view(bsz, seq_len, h)


class TopKRouter(nn.Module):
    """Top-k MoE router with softmax-then-topk-then-renormalize logic.

    softmax is applied to ALL
    routing logits in fp32 first, then top-k is taken on the softmax
    probabilities, and the selected weights are renormalized so they
    sum to 1 within each token. This differs from the older
    "topk-then-softmax" routing in :class:`MoE` (which softmaxes
    only over the top-k logits — Qwen3-VL-MoE convention).

    Stored as a free ``nn.Parameter`` named ``weight`` of shape
    ``(num_experts, hidden_size)`` to match HF state-dict layout
    directly (``mlp.gate.weight``).
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        """Route tokens via softmax-then-topk with renormalization."""
        flat = hidden_states.reshape(-1, self.hidden_size)
        router_logits = F.linear(flat, self.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_value, top_index = torch.topk(router_probs, self.top_k, dim=-1)
        top_value /= top_value.sum(dim=-1, keepdim=True)
        top_value = top_value.to(router_logits.dtype)

        return router_logits, top_value, top_index

class SharedExpertMoE(nn.Module):
    """MoE block with a shared expert summed in (Qwen3.5 / DeepSeek-V3 style).

    Forward (matches the standard packed-expert routing op sequence):

    1. Run the shared expert on every token.
    2. Top-k route via :class:`TopKRouter`.
    3. Run the selected experts via packed :class:`MoEExperts` (per-token
       routing-weighted sum across experts that received tokens).
    4. ``out = experts_out + sigmoid(shared_expert_gate) * shared_out``.

    Parameter naming::

        mlp.gate.weight                   (num_experts, hidden_size)
        mlp.experts.gate_up_proj          (E, 2I, H)
        mlp.experts.down_proj             (E, H, I)
        mlp.shared_expert.gate_proj.weight (Is, H)
        mlp.shared_expert.up_proj.weight  (Is, H)
        mlp.shared_expert.down_proj.weight (H, Is)
        mlp.shared_expert_gate.weight     (1, H)
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        shared_expert_intermediate_size: int,
        shared_expert_cls: type = SwiGLUMLP,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = MoEExperts(num_experts, hidden_size, moe_intermediate_size)
        self.shared_expert = shared_expert_cls(
            hidden_size, shared_expert_intermediate_size, bias=False,
        )
        # ``nn.Linear(H, 1)`` weight shape ``(1, H)``. FSDP cannot shard
        # dim-0=1 on ``world_size > 1`` and routes the param through
        # ``replicate_params`` via the ``size(0) % world_size != 0`` rule.
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.shared_expert_gate.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        bsz, seq_len, h = x.shape
        x_flat = x.view(-1, h)

        shared_out = self.shared_expert(x_flat)
        _, top_value, top_index = self.gate(x_flat)
        routed = self.experts(x_flat, top_index, top_value)
        shared_out = torch.sigmoid(self.shared_expert_gate(x_flat)) * shared_out

        out = routed + shared_out
        return out.view(bsz, seq_len, h)
