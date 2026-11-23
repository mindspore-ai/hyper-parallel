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
"""Tracked construction for the existing public token-choice router contract."""
from __future__ import annotations

from typing import Any
from math import isfinite

import torch
from torch import nn

from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.modules.gate_family import routing_scores, select_topk_scores


class TokenChoiceTopKRouterModule(nn.Module):
    """Keep gate state and the weights/indices/counts output contract unchanged."""

    def __init__(self, *, config: Any = None, module: nn.Module | None = None) -> None:
        """Create a public router or reuse its gate Parameters and routing settings."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        source = config if config is not None else module
        dim = config.dim if config is not None else module.gate.in_features
        for name, value in (("dim", dim), ("num_experts", source.num_experts), ("top_k", source.top_k)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if source.top_k > source.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if (isinstance(source.route_scale, bool) or not isinstance(source.route_scale, (float, int))
                or not isfinite(source.route_scale)):
            raise ValueError("route_scale must be finite")
        self._validate_groups(source)
        if source.score_func not in ("sigmoid", "softmax"):
            raise ValueError("score_func must be 'sigmoid' or 'softmax'")
        for name in ("num_experts", "top_k", "score_func", "num_expert_groups",
                     "num_limited_groups", "route_scale"):
            setattr(self, name, getattr(source, name))
        if module is not None:
            if module.__class__ is not TokenChoiceTopKRouterModule:
                raise TypeError("model-specific routers require their own adapter")
            self.gate = module.gate
            self.train(module.training)
        else:
            self.gate = nn.Linear(config.dim, config.num_experts, bias=False)

    @staticmethod
    def _validate_groups(source: Any) -> None:
        """Validate expert-group coverage and the number of selectable experts."""
        groups, limited = source.num_expert_groups, source.num_limited_groups
        if groups is not None:
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (groups, limited)):
                raise ValueError("expert groups and limited groups must be positive integers")
            if source.num_experts % groups or limited > groups:
                raise ValueError("expert groups must divide experts and cover the selected limited groups")
            if source.top_k > limited * (source.num_experts // groups):
                raise ValueError("top_k exceeds the number of experts in selected groups")
        elif limited is not None:
            raise ValueError("num_limited_groups requires num_expert_groups")

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return unbiased weights, expert IDs and counts for the selected experts.

        Scale scores before group selection. Correction bias affects expert
        selection only, and the returned weights retain their original scores.
        """
        scores = routing_scores(self.gate(x).float(), self.score_func)
        if self.route_scale != 1.0:
            scores = scores * self.route_scale
        selection_scores = scores
        if self.num_expert_groups is not None:
            selection_scores = self._get_node_limited_routing_scores(scores)
        if expert_bias is not None:
            selection_scores = selection_scores + expert_bias.float()
        selected_experts = selection_scores.topk(self.top_k, dim=-1).indices
        top_scores = scores.gather(1, selected_experts)
        counts = torch.bincount(selected_experts.flatten(), minlength=self.num_experts)
        return top_scores, selected_experts, counts

    def _get_node_limited_routing_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Mask experts outside the groups selected by their highest scores."""
        num_tokens, num_experts = scores.shape
        experts_per_group = num_experts // self.num_expert_groups
        grouped_scores = scores.view(num_tokens, self.num_expert_groups, experts_per_group)
        selected_groups = grouped_scores.max(dim=-1).values.topk(self.num_limited_groups, dim=-1).indices
        mask = scores.new_zeros(num_tokens, self.num_expert_groups)
        mask.scatter_(1, selected_groups, 1.0)
        mask = mask.unsqueeze(-1).expand_as(grouped_scores).reshape_as(scores)
        return scores.masked_fill(mask == 0, float("-inf"))


@module_replacement
def TokenChoiceTopKRouter(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None,
) -> nn.Module:
    """Build only routing math; auxiliary losses and statistics remain model-owned."""
    return build_family(TokenChoiceTopKRouter, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh)


register_module_family(ModuleFamily(
    name="token_choice_topk_router", baseline=TokenChoiceTopKRouter,
    variants={"local": "hyper_parallel.components.modules.router_family.TokenChoiceTopKRouterModule"},
    default_variant="local",
))


__all__ = ["TokenChoiceTopKRouter",
           "TokenChoiceTopKRouterModule", "routing_scores", "select_topk_scores"]
