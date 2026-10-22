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

from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.trainer.config import TrainerConfig, entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import resolve_config


class PackedExperts(nn.Module):
    """Transformers-layout packed experts with an explicit reference forward."""

    def __init__(self, *, canonical: bool = False, device: str = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        """Create reproducible expert projections in either accepted layout."""
        super().__init__()
        self.config = SimpleNamespace(hidden_act="silu", initializer_range=0.02)
        self.num_experts, self.hidden_size, self.intermediate_size = 4, 128, 256
        self.is_transposed = canonical
        fc1 = torch.randn(4, 512, 128, device=device, dtype=dtype) * 0.02
        fc2 = torch.randn(4, 128, 256, device=device, dtype=dtype) * 0.02
        if canonical:
            self.down_proj = nn.Parameter(fc2.transpose(-2, -1).contiguous())
            self.gate_up_proj = nn.Parameter(fc1.transpose(-2, -1).contiguous())
        else:
            self.gate_up_proj = nn.Parameter(fc1)
            self.down_proj = nn.Parameter(fc2)


    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """Apply weights once after the second expert projection.

        Args:
            hidden_states: Input with hidden dimension last.
            top_k_index: Fixed token-to-expert assignments.
            top_k_weights: Fixed routing scores, applied once after GMM2.
        """
        flat = hidden_states.reshape(-1, 128)
        output = torch.zeros_like(flat)
        for expert in range(4):
            token, slot = torch.where(top_k_index == expert)
            w1, w2 = self.gate_up_proj[expert], self.down_proj[expert]
            if self.is_transposed:
                w1, w2 = w1.T, w2.T
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
        """Return the existing Qwen3 router tuple contract.

        Args:
            hidden_states: Unused input; routing is fixed by the test.
        """
        del hidden_states
        return None, self.scores, self.indices


class TinyMoE(nn.Module):
    """Minimal model with the same router/experts boundary as Qwen3-MoE."""

    def __init__(self, **kwargs: Any) -> None:
        """Build the packed expert model with explicit Qwen3 metadata."""
        super().__init__()
        self.config = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"], model_type="qwen3_moe",
                                      num_experts=4, num_experts_per_tok=2, hidden_size=128,
                                      num_attention_heads=4, num_key_value_heads=4,
                                      tie_word_embeddings=False, hidden_act="silu")
        self.mlp = nn.Module()
        self.mlp.config = self.config
        self.mlp.gate = FixedRouter()
        self.mlp.experts = PackedExperts(**kwargs)
        # A named module class gives the planner an ordinary hidden_states signature.
        self.mlp = MoEBlock(self.mlp)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Execute the existing MoE boundary.

        Args:
            hidden_states: Local token states passed to the MoE block.
        """
        return self.mlp(hidden_states)


class MoEBlock(nn.Module):
    """Router then experts, with no additional output composition."""

    def __init__(self, source: nn.Module) -> None:
        """Retain the source router and expert container."""
        super().__init__()
        self.config, self.gate, self.experts = source.config, source.gate, source.experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Keep model-specific routing outside the expert family.

        Args:
            hidden_states: Token states passed to the existing router and experts.
        """
        _, weights, indices = self.gate(hidden_states)
        return self.experts(hidden_states, indices, weights)


def family_config(**selectors: Any) -> TrainerConfig:
    """Resolve a complete trainer mapping through the production parser."""
    return resolve_config({
        "model": {"_target_": "tests.common.grouped_experts_family.TinyMoE"},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
        "plan_overrides": [{
            "match": "mlp.experts", "module_type": "tests.common.grouped_experts_family.PackedExperts",
            "replace_module": {"_target_": "hyper_parallel.components.modules.GroupedExperts", **selectors},
        }],
    })


def replace_experts(model: TinyMoE, config: TrainerConfig, *, context: dict | None = None) -> list:
    """Execute real replacement and its checkpoint transforms on loaded weights.

    Args:
        model: Toy model with source-layout expert weights.
        config: Configuration produced by the real trainer parser.
        context: Optional replacement topology context.

    Returns:
        Checkpoint converters emitted by the replacement executor.
    """
    mapping = []
    _apply_module_replacement_actions(model, entries_to_module_replacements(config.plan_overrides),
                                      weights_mapping=mapping, context=context)
    return mapping
