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
"""CPU model fixtures for testing existing CP/EP adapters with replacements."""

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.modules.gqa_attention import GQAAttention
from hyper_parallel.components.modules.grouped_experts import GroupedExperts


class PackedExperts(nn.Module):
    """Small native expert layout with an unsharded routing oracle."""

    def __init__(self) -> None:
        """Construct packed native SwiGLU parameters."""
        super().__init__()
        self.num_experts = 4
        self.hidden_size = 4
        self.intermediate_size = 3
        self.config = SimpleNamespace(hidden_act="silu", use_fused_swiglu=False)
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(torch.randn(4, 6, 4, dtype=torch.float64) * .2)
        self.down_proj = nn.Parameter(torch.randn(4, 4, 3, dtype=torch.float64) * .2)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """Compute every routed slot, applying router weights once."""
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        output = flat * 0
        for expert in range(self.num_experts):
            rows, slots = torch.where(top_k_index == expert)
            gate, up = F.linear(flat[rows], self.gate_up_proj[expert]).chunk(2, -1)  # pylint: disable=not-callable
            values = F.linear(F.silu(gate) * up, self.down_proj[expert])  # pylint: disable=not-callable
            output = output.index_add(0, rows, values * top_k_weights[rows, slots, None])
        return output.reshape(hidden_states.shape)


def cpu_grouped_matmul(tokens: torch.Tensor, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Replace only the device GEMM; retain GroupedExperts' actual forward."""
    outputs = []
    begin = 0
    for expert, end in enumerate(kwargs["group_list"].tolist()):
        outputs.append(tokens[begin:end] @ weight[expert])
        begin = end
    return torch.cat(outputs)


def make_grouped_experts(source: PackedExperts) -> GroupedExperts:
    """Build the actual replacement, including its transposed weight layout."""
    return GroupedExperts(module=source)


def cpu_attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  attention_mask: Any = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
    """CPU implementation of the selected module's BNSD-to-BSND interface."""
    output = F.scaled_dot_product_attention(  # pylint: disable=not-callable
        query, key, value, attn_mask=attention_mask,
        is_causal=kwargs.get("is_causal", module.is_causal and attention_mask is None), enable_gqa=True,
        scale=kwargs.get("scaling"), dropout_p=kwargs.get("dropout", 0.0))
    return output.transpose(1, 2), None


class NativeAttention(nn.Module):
    """Projection owner with the forward contract used by GQA replacements."""

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Run the unsharded CPU attention oracle."""
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, length, 4, 2).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, length, 2, 2).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, length, 2, 2).transpose(1, 2)
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, 8)), None


def make_attention_source() -> nn.Module:
    """Build native projection parameters for an actual GQA replacement."""
    source = NativeAttention()
    source.config = SimpleNamespace(hidden_size=8, num_attention_heads=4, num_key_value_heads=2,
                                    head_dim=2, sliding_window=None)
    source.q_proj = nn.Linear(8, 8, bias=False, dtype=torch.float64)
    source.k_proj = nn.Linear(8, 4, bias=False, dtype=torch.float64)
    source.v_proj = nn.Linear(8, 4, bias=False, dtype=torch.float64)
    source.o_proj = nn.Linear(8, 8, bias=False, dtype=torch.float64)
    return source


def make_attention() -> GQAAttention:
    """Build an actual GQA replacement with a selected CPU computation."""
    return GQAAttention(module=make_attention_source(), attention_interface=cpu_attention)
