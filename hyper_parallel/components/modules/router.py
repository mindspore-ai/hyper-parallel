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
from torch import nn

from hyper_parallel.components.modules.moe import TokenChoiceTopKRouter as SourceTokenChoiceTopKRouter
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.modules.gate import routing_scores, select_topk_scores


class TokenChoiceTopKRouterModule(SourceTokenChoiceTopKRouter):
    """Keep gate state and the weights/indices/counts output contract unchanged."""

    def __init__(self, *, config: Any = None, module: nn.Module | None = None) -> None:
        """Create a public router or reuse its gate Parameters and routing settings."""
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
        if module is not None:
            if type(module) not in (SourceTokenChoiceTopKRouter, TokenChoiceTopKRouterModule):
                raise TypeError("model-specific routers require their own adapter")
            nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
            self.gate = module.gate
            for name in ("num_experts", "top_k", "score_func", "num_expert_groups",
                         "num_limited_groups", "route_scale"):
                setattr(self, name, getattr(module, name))
            self.train(module.training)
        else:
            super().__init__(dim=config.dim, num_experts=config.num_experts, top_k=config.top_k,
                             score_func=config.score_func, num_expert_groups=config.num_expert_groups,
                             num_limited_groups=config.num_limited_groups, route_scale=config.route_scale)

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

    def reset_parameters(self) -> None:
        """Initialize the gate after meta materialization."""
        self.gate.reset_parameters()


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
    variants={"local": "hyper_parallel.components.modules.router.TokenChoiceTopKRouterModule"},
    default_variant="local",
))


__all__ = ["TokenChoiceTopKRouter",
           "TokenChoiceTopKRouterModule", "routing_scores", "select_topk_scores"]
