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
"""DeepSeek-V3 routing and checkpoint mapping for the common chunk executor."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hyper_parallel.components.checkpoint.conversion_ops import Split
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter, WeightRenaming
from hyper_parallel.components.modules.moe import FeedForward, MoE, TokenChoiceTopKRouter
from hyper_parallel.models.replacement import module_replacement


class DeepseekV3ChunkRouter(TokenChoiceTopKRouter):
    """Preserve the source model's FP32 gate, grouped choice and normalized scores."""

    def __init__(self, source: nn.Module) -> None:
        """Reuse the source gate parameter and its routing configuration."""
        config = source.config
        super().__init__(config.hidden_size, config.n_routed_experts, config.num_experts_per_tok)
        self.gate.weight = source.gate.weight
        self.groups = source.n_group
        self.selected_groups = source.topk_group
        self.normalize = source.norm_topk_prob
        self.route_scale = source.routed_scaling_factor

    def _scores(self, inputs):
        # PyTorch exposes linear through a C binding that pylint cannot infer.
        return F.linear(inputs.float(), self.gate.weight.float()).sigmoid()  # pylint: disable=not-callable

    def _selected_scores(self, scores, indices):
        selected = scores.gather(1, indices)
        if self.normalize:
            selected = selected / (selected.sum(dim=-1, keepdim=True) + 1e-20)
        return selected * self.route_scale

    def forward(self, x: torch.Tensor, expert_bias: torch.Tensor | None = None) -> tuple:
        """Return unbiased selected scores, original expert IDs and counts.

        Args:
            x: Flattened token inputs with hidden width last.
            expert_bias: Optional selection-only correction bias per expert.

        Returns:
            Selected probabilities, expert IDs and integer token counts."""
        scores = self._scores(x)
        choices = scores if expert_bias is None else scores + expert_bias
        grouped = choices.view(-1, self.groups, self.num_experts // self.groups)
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        groups = group_scores.topk(self.selected_groups, dim=-1, sorted=False).indices
        mask = torch.zeros_like(group_scores).scatter_(1, groups, 1)
        mask = mask.unsqueeze(-1).expand_as(grouped).reshape_as(choices).bool()
        indices = choices.masked_fill(~mask, 0.0).topk(self.top_k, dim=-1, sorted=False).indices
        counts = torch.bincount(indices.reshape(-1), minlength=self.num_experts)
        return self._selected_scores(scores, indices), indices, counts

    def scores_for_selected(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Rebuild differentiable probabilities without repeating expert selection.

        Args:
            x: Flattened token inputs with hidden width last.
            indices: Original forward expert IDs for each token.

        Returns:
            Differentiable probabilities for the original selected experts."""
        return self._selected_scores(self._scores(x), indices)


@module_replacement
class DeepseekV3ChunkMoE(MoE):
    """Adapt an existing DeepSeek MoE to common experts before optimizer creation.

    Args:
        module: Source DeepseekV3MoE with batched gate_up_proj/down_proj weights.
        module_fqn: Replacement boundary in the source model.
        context: Build context supplied by the generic replacement executor.
        chunk_size: Fixed source-token chunk, or None for the common whole-input baseline.
    """

    def __init__(self, *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
                 chunk_size: int | None = 2048) -> None:
        """Bind source weights and preserve model-owned routing before optimizer creation."""
        del context
        config = module.config
        if config.hidden_act != "silu" or not hasattr(module.experts, "gate_up_proj"):
            raise ValueError(f"{module_fqn}: expected batched DeepSeek-V3 SwiGLU experts.")
        if not config.n_shared_experts or not hasattr(module, "shared_experts"):
            raise ValueError(f"{module_fqn}: expected DeepSeek-V3 shared experts.")
        width, intermediate = config.hidden_size, config.moe_intermediate_size
        with torch.device("meta"):
            super().__init__(width, intermediate, config.n_routed_experts,
                             top_k=config.num_experts_per_tok, score_before_experts=False,
                             use_grouped_mm=True, chunk_size=chunk_size)
            self.router = DeepseekV3ChunkRouter(module)
            self.shared_expert = FeedForward(width, intermediate * config.n_shared_experts)
        self.config = config
        gate_up = module.experts.gate_up_proj
        self.experts.w1 = nn.Parameter(gate_up[:, :intermediate].detach().clone(), gate_up.requires_grad)
        self.experts.w3 = nn.Parameter(gate_up[:, intermediate:].detach().clone(), gate_up.requires_grad)
        self.experts.w2 = module.experts.down_proj
        self.shared_expert.w1.weight = module.shared_experts.gate_proj.weight
        self.shared_expert.w2.weight = module.shared_experts.down_proj.weight
        self.shared_expert.w3.weight = module.shared_experts.up_proj.weight
        self.expert_bias = module.gate.e_score_correction_bias
        # Runtime statistics are not part of the source checkpoint contract.
        self.register_buffer("tokens_per_expert", torch.zeros_like(self.expert_bias), persistent=False)
        self.train(module.training)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Preserve the source keyword argument while using the common executor.

        Args:
            hidden_states: Local batch, sequence and hidden dimensions.

        Returns:
            MoE output matching the input shape."""
        return super().forward(x=hidden_states)

    def make_transforms(self) -> list[WeightConverter | WeightRenaming]:
        """Declare reversible source-to-common parameter and buffer conversions."""
        intermediate = self.config.moe_intermediate_size
        transforms = [WeightConverter("experts.gate_up_proj", ["experts.w1", "experts.w3"],
                                      [Split((intermediate, intermediate), dim=1)])]
        names = [("experts.down_proj", "experts.w2"), ("gate.weight", "router.gate.weight"),
                 ("gate.e_score_correction_bias", "expert_bias"),
                 ("shared_experts.gate_proj.weight", "shared_expert.w1.weight"),
                 ("shared_experts.down_proj.weight", "shared_expert.w2.weight"),
                 ("shared_experts.up_proj.weight", "shared_expert.w3.weight")]
        transforms.extend(WeightRenaming(source, target) for source, target in names)
        return transforms
