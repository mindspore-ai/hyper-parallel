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
"""Small deterministic packed MoE used by CPU and real NPU family tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config import TrainerConfig, entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import resolve_config


def cpu_token_permute(tokens: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Emulate the NPU token permutation and its source-to-permuted map."""
    flat_indices = indices.reshape(-1).to(torch.int64)
    order = flat_indices.argsort(stable=True)
    inverse = torch.empty_like(order, dtype=torch.int32)
    inverse.scatter_(0, order, torch.arange(order.numel(), dtype=torch.int32))
    return tokens.index_select(0, order // indices.shape[1]), inverse


def cpu_token_unpermute(
    values: torch.Tensor, inverse: torch.Tensor, probabilities: torch.Tensor | None = None,
) -> torch.Tensor:
    """Emulate the fused inverse permutation, optional weighting and top-k sum."""
    source_slots = values.index_select(0, inverse.to(torch.int64))
    if probabilities is None:
        return source_slots
    source_slots = source_slots.reshape(*probabilities.shape, values.shape[-1])
    return (source_slots * probabilities.unsqueeze(-1)).sum(dim=1)


def cpu_swiglu(inputs: torch.Tensor) -> torch.Tensor:
    """Emulate the existing fused SwiGLU boundary for hardware-independent routing tests."""
    gate, up = inputs.chunk(2, dim=-1)
    return F.silu(gate) * up


def cpu_grouped_matmul(
    inputs: torch.Tensor, weights: torch.Tensor, *, group_list: torch.Tensor,
) -> torch.Tensor:
    """Apply the same expert row selection as grouped GEMM using CPU batched matmul."""
    counts = torch.diff(group_list, prepend=group_list.new_zeros(1))
    expert_ids = torch.arange(counts.numel(), device=counts.device).repeat_interleave(counts)
    return torch.bmm(inputs.unsqueeze(1), weights.index_select(0, expert_ids)).squeeze(1)


class PackedExperts(nn.Module):
    """Transformers-layout packed experts with canonical [E, 2I, H] and [E, H, I] weights."""

    def __init__(self, *, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        """Create reproducible expert projections in canonical packed layout."""
        super().__init__()
        self.config = SimpleNamespace(hidden_act="silu", initializer_range=0.02, num_hidden_layers=1)
        self.num_experts, self.hidden_size, self.intermediate_size = 4, 128, 256
        self.gate_up_proj = nn.Parameter(torch.randn(4, 512, 128, device=device, dtype=dtype) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(4, 128, 256, device=device, dtype=dtype) * 0.02)
        self.act_fn = nn.SiLU()

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Apply weights once after the second expert projection."""
        flat = hidden_states.reshape(-1, 128)
        output = torch.zeros_like(flat)
        for expert in range(4):
            token, slot = torch.where(top_k_index == expert)
            w1, w2 = self.gate_up_proj[expert], self.down_proj[expert]
            gate, up = F.linear(flat[token], w1).chunk(2, -1)  # pylint: disable=not-callable
            values = F.linear(F.silu(gate) * up, w2)  # pylint: disable=not-callable
            output.index_add_(0, token, values * top_k_weights[token, slot, None].to(values.dtype))
        return output.reshape_as(hidden_states)


class FixedRouter(nn.Module):
    """Fixed top-k choices and differentiable fixed scores for recipe tests."""

    def __init__(self) -> None:
        """Start with no routing assignments."""
        super().__init__()
        self.indices = None
        self.scores = None

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """Return the existing Qwen3 router tuple contract."""
        del hidden_states
        return None, self.scores, self.indices


class TinyMoE(nn.Module):
    """Minimal model with the same router/experts boundary as Qwen3-MoE."""

    def __init__(self, **kwargs: Any) -> None:
        """Build the packed expert model with explicit Qwen3 metadata."""
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["Qwen3MoeForCausalLM"],
            model_type="qwen3_moe",
            num_experts=4,
            num_experts_per_tok=2,
            hidden_size=128,
            moe_intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            tie_word_embeddings=False,
            hidden_act="silu",
            initializer_range=0.02,
        )
        self.mlp = nn.Module()
        self.mlp.config = self.config
        self.mlp.gate = FixedRouter()
        self.mlp.experts = PackedExperts(**kwargs)
        self.mlp = MoEBlock(self.mlp)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Execute the existing MoE boundary."""
        return self.mlp(hidden_states)


class MoEBlock(nn.Module):
    """Router then experts, with no additional output composition."""

    def __init__(self, source: nn.Module) -> None:
        """Retain the source router and expert container."""
        super().__init__()
        self.config, self.gate, self.experts = source.config, source.gate, source.experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Keep model-specific routing outside the expert family."""
        _, weights, indices = self.gate(hidden_states)
        return self.experts(hidden_states, indices, weights)


def family_config(**selectors: Any) -> TrainerConfig:
    """Resolve a complete trainer mapping through the production parser."""
    return resolve_config({
        "model": {"_target_": "tests.common.grouped_experts_family.TinyMoE"},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
        "plan_overrides": [{
            "match": "mlp.experts",
            "replace": {
                "kind": "module",
                "source_type": "tests.common.grouped_experts_family.PackedExperts",
                "implementation": {
                    "_target_": "hyper_parallel.components.modules.GroupedExperts", **selectors,
                },
            },
        }],
    })


def replace_experts(model: TinyMoE, config: TrainerConfig, *, context: dict | None = None) -> list:
    """Execute real replacement and its checkpoint transforms on loaded weights."""
    mapping = []
    rules = entries_to_module_replacements(
        config.plan_overrides,
        low_precision_config=config.training.low_precision,
    )
    plan = compile_module_replacements(model, rules)
    apply_module_replacements(model, plan, weights_mapping=mapping, context=context)
    return mapping
