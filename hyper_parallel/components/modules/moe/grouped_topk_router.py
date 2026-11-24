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
"""FP32 token routing with correction-only biases and top-two expert-group selection."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.components.modules.moe.router import routing_scores
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


class ScoreCorrectedTopKRouterModule(nn.Module):
    """Own selection biases, source matrix identities and the FP32 unbiased routing weights."""

    def __init__(
        self,
        *,
        config: Any = None,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Any = None,
        source_contract: str = "ids_weights",
    ) -> None:
        """Construct explicit routing settings or adopt a separate matrix/bias source."""
        super().__init__()
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if source_contract not in ("ids_weights", "logits_weights_ids"):
            raise ValueError("Unsupported corrected-router source contract")
        source = config if module is None else module
        self.source_contract = source_contract
        self.hidden_size = config.hidden_size if module is None else module.weight.shape[1]
        self.num_experts = config.num_experts if module is None else module.weight.shape[0]
        self.top_k = source.top_k
        self.scoring_func = getattr(source, "moe_router_activation_func", getattr(source, "scoring_func", None))
        self.num_groups = getattr(source, "num_expert_group", getattr(source, "num_groups", 1))
        self.groups_selected = getattr(source, "topk_group", getattr(source, "groups_selected", 1))
        self.normalize_topk = getattr(source, "moe_renormalize", getattr(source, "normalize_topk", True))
        self.routed_scaling_factor = source.routed_scaling_factor
        self._validate_settings()
        if module is None:
            self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)
            self.reset_parameters()
        else:
            if set(dict(module.named_parameters(recurse=False))) != {"weight", "e_score_correction_bias"}:
                raise ValueError("Corrected router source must own its matrix and correction Parameter separately")
            if list(module.named_children()) or module.e_score_correction_bias.shape != (self.num_experts,):
                raise ValueError("Unsupported corrected router source structure")
            self.weight, self.e_score_correction_bias = module.weight, module.e_score_correction_bias
            self.training = module.training

    def _validate_settings(self) -> None:
        """Validate grouped top-k coverage, scoring and normalization before weight allocation."""
        for name in ("hidden_size", "num_experts", "top_k", "num_groups", "groups_selected"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.num_experts % self.num_groups
            or self.groups_selected > self.num_groups
            or self.top_k > self.groups_selected * (self.num_experts // self.num_groups)
        ):
            raise ValueError("Router groups do not cover the requested top-k experts")
        if self.num_groups > self.groups_selected and self.num_experts // self.num_groups < 2:
            raise ValueError("Top-two group scoring needs at least two experts in each group")
        if self.scoring_func not in ("softmax", "sigmoid") or not isinstance(self.normalize_topk, bool):
            raise ValueError("Corrected routing requires sigmoid/softmax and a boolean normalization policy")
        if not isinstance(self.routed_scaling_factor, (int, float)) or not math.isfinite(self.routed_scaling_factor):
            raise ValueError("routed_scaling_factor must be finite")

    def reset_parameters(self) -> None:
        """Restore the source constructor matrix law and a defined neutral selection bias."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.e_score_correction_bias)

    def make_transforms(self) -> list:
        """Retain both source keys and their independent parameter identities."""
        return []

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Select groups with biased scores and return normalized unbiased expert weights."""
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("Router inputs must end in hidden_size and include token dimensions")
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            # Torch exposes this C-extension callable without a Python signature.
            logits = F.linear(  # pylint: disable=not-callable
                hidden_states.reshape(-1, self.hidden_size).float(), self.weight.float()
            )
            scores = routing_scores(logits, self.scoring_func)
        selection = scores + self.e_score_correction_bias.float().unsqueeze(0)
        if self.num_groups > self.groups_selected:
            group_scores = selection.view(-1, self.num_groups, self.num_experts // self.num_groups)
            group_scores = group_scores.topk(2, dim=-1).values.sum(dim=-1)
            selected = group_scores.topk(self.groups_selected, dim=-1, sorted=False).indices
            mask = torch.zeros_like(group_scores).scatter_(1, selected, 1)
            mask = mask.unsqueeze(-1).expand(-1, -1, self.num_experts // self.num_groups).reshape_as(selection)
            selection = selection.masked_fill(~mask.bool(), float("-inf"))
        indices = selection.topk(self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        if self.top_k > 1 and self.normalize_topk:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return (indices, weights) if self.source_contract == "ids_weights" else (logits, weights, indices)


def corrected_router_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Replicate router matrices and preserve the source's local token ownership."""
    del module_fqn, member
    axes = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    tokens = dict(axes)
    rows = dict(axes)
    for axis in ("cp", "tp"):
        if context.axis_sizes[axis] > 1 and (axis == "cp" or context.sequence_parallel):
            tokens[axis], rows[axis] = Shard(1), Shard(0)
    outputs = ("indices", "weights") if module.source_contract == "ids_weights" else ("logits", "weights", "indices")
    return ModuleShardingSpec(
        params={name: dict(axes) for name, _ in module.named_parameters()},
        in_src={"hidden_states": dict(tokens)},
        in_dst={"hidden_states": dict(tokens)},
        out_src={name: dict(rows) for name in outputs},
        out_dst={name: dict(rows) for name in outputs},
        out_names=outputs,
        is_boundary=True,
    )


@module_replacement
def ScoreCorrectedTopKRouter(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
    source_contract: str = "ids_weights",
) -> nn.Module:
    """Build complete group-aware corrected routing through the shared family lifecycle."""
    return build_family(
        ScoreCorrectedTopKRouter,
        config=config,
        module=module,
        module_fqn=module_fqn,
        context=context,
        mesh=mesh,
        source_contract=source_contract,
    )


register_module_family(
    ModuleFamily(
        name="score_corrected_topk_router",
        baseline=ScoreCorrectedTopKRouter,
        variants={"local": "hyper_parallel.components.modules.moe.grouped_topk_router.ScoreCorrectedTopKRouterModule"},
        default_variant="local",
        sharding=corrected_router_sharding,
    )
)
