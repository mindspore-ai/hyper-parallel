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
"""Trainable latent-width routed SiTU experts with source-indexed independent matrices."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.common.rms_norm import NativeRMSNormModule
from hyper_parallel.components.modules.common.situ_mlp import SiTUMLPModule
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.components.modules.moe.grouped_topk_router import ScoreCorrectedTopKRouterModule
from hyper_parallel.components.modules.moe.independent_experts import IndependentExpertsModule
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


def _mlp_config(config: Any, hidden_size: int, intermediate_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="situ",
        mlp_bias=False,
        activation_situ_beta=config.activation_situ_beta,
        activation_situ_linear_beta=config.activation_situ_linear_beta,
        num_experts=config.num_experts,
    )


class KimiLatentMoEModule(nn.Module):
    """Own differentiable routing/dispatch/reduction while retaining every expert's source matrix boundary."""

    def __init__(
        self, *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None
    ) -> None:
        """Build from the native architecture or adopt loaded source state without packing experts."""
        super().__init__()
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        self.config = config if module is None else module.config
        config = self.config
        if config.hidden_act != "situ":
            raise ValueError("K3 latent MoE requires the released SiTU activation")
        self.hidden_dim, self.num_experts, self.top_k = (
            config.hidden_size,
            config.num_experts,
            config.num_experts_per_token,
        )
        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)
        self.ep_size, self.ep_rank, self.experts_per_rank = 1, 0, self.num_experts
        if module is not None:
            expected = {"experts", "gate"}
            if config.num_shared_experts is not None:
                expected.add("shared_experts")
            if self.use_latent_moe:
                expected.update(("routed_expert_down_proj", "routed_expert_up_proj"))
                if self.latent_moe_use_norm:
                    expected.add("routed_expert_norm")
            if set(dict(module.named_children())) != expected or list(module.named_parameters(recurse=False)):
                raise ValueError("Unsupported latent independent-expert source structure")
            for name, child in module._modules.items():
                if name == "experts":
                    child = IndependentExpertsModule(module=child)
                elif name == "gate":
                    child = ScoreCorrectedTopKRouterModule(module=child)
                elif name == "shared_experts":
                    child = SiTUMLPModule(module=child)
                elif name == "routed_expert_norm":
                    child = NativeRMSNormModule(module=child, cast_order="before_weight")
                self.add_module(name, child)
            self.training = module.training
            return
        expert_config = _mlp_config(config, self.moe_hidden_size, config.moe_intermediate_size)
        self.experts = IndependentExpertsModule(config=expert_config)
        router_config = SimpleNamespace(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            scoring_func=config.moe_router_activation_func,
            num_groups=getattr(config, "num_expert_group", 1),
            groups_selected=getattr(config, "topk_group", 1),
            normalize_topk=config.moe_renormalize,
            routed_scaling_factor=config.routed_scaling_factor,
        )
        self.gate = ScoreCorrectedTopKRouterModule(config=router_config)
        if config.num_shared_experts is not None:
            self.shared_experts = SiTUMLPModule(
                config=_mlp_config(
                    config, config.hidden_size, config.moe_intermediate_size * config.num_shared_experts
                ),
            )
        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(config.hidden_size, self.moe_hidden_size, bias=False)
            self.routed_expert_up_proj = nn.Linear(self.moe_hidden_size, config.hidden_size, bias=False)
            if self.latent_moe_use_norm:
                norm_config = SimpleNamespace(hidden_size=self.moe_hidden_size, eps=config.rms_norm_eps)
                self.routed_expert_norm = NativeRMSNormModule(config=norm_config, cast_order="before_weight")

    def reset_parameters(self) -> None:
        """Initialize bare routing state and optional normalization after meta materialization."""
        self.gate.reset_parameters()
        if self.use_latent_moe and self.latent_moe_use_norm:
            self.routed_expert_norm.reset_parameters()

    def make_transforms(self) -> list:
        """Retain source expert indices, gate state and latent projection keys."""
        return []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply source-ordered independent experts with differentiable FP32 top-k reduction.

        The ordinary member reads a compact expert-count vector on CPU, as the
        released reference does. A fused independent-matrix transport member
        can replace this dispatch without changing Muon's parameter matrices.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_dim or hidden_states.numel() == 0:
            raise ValueError("Latent MoE requires nonempty BSH inputs with the configured hidden width")
        shape = hidden_states.shape
        indices, weights = self.gate(hidden_states)
        flat = hidden_states.reshape(-1, self.hidden_dim)
        if self.use_latent_moe:
            flat = self.routed_expert_down_proj(flat)
        order = indices.reshape(-1).argsort()
        sorted_tokens = flat[order // self.top_k]
        counts = torch.bincount(indices.reshape(-1), minlength=self.num_experts).cpu().tolist()
        outputs = []
        offset = 0
        for expert, count in zip(self.experts, counts):
            if count:
                outputs.append(expert(sorted_tokens[offset : offset + count]))
                offset += count
        packed = torch.cat(outputs)
        restored = torch.empty_like(packed)
        restored[order] = packed
        routed = (restored.view(*indices.shape, -1).to(weights.dtype) * weights.unsqueeze(-1)).sum(1).to(packed.dtype)
        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                routed = self.routed_expert_norm(routed)
            routed = self.routed_expert_up_proj(routed)
        routed = routed.view(shape)
        if self.config.num_shared_experts is not None:
            routed = routed + self.shared_experts(hidden_states)
        return routed


def _sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own independent expert matrices and latent projections as one complete local family."""
    del module_fqn, member
    axes = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    return ModuleShardingSpec(
        params={name: dict(axes) for name, _ in module.named_parameters()},
        in_src={"hidden_states": dict(axes)},
        in_dst={"hidden_states": dict(axes)},
        out_src={"output": dict(axes)},
        out_dst={"output": dict(axes)},
        is_boundary=True,
    )


def _validate(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    del config, module, context
    if any(member.axis_sizes[axis] > 1 for axis in ("tp", "cp", "ep")):
        raise NotImplementedError("K3 independent-matrix latent MoE currently supports FSDP DP with TP1/CP1/EP1")


@module_replacement
def KimiLatentMoE(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
) -> nn.Module:
    """Build a complete trainable K3 MoE family with unchanged source keys and Muon matrices."""
    return build_family(KimiLatentMoE, config=config, module=module, module_fqn=module_fqn, context=context, mesh=mesh)


register_module_family(
    ModuleFamily(
        name="kimi_k3_latent_moe",
        baseline=KimiLatentMoE,
        variants={"local": "hyper_parallel.models.kimi_k3.components.latent_moe.KimiLatentMoEModule"},
        default_variant="local",
        sharding=_sharding,
        validate=_validate,
    )
)
