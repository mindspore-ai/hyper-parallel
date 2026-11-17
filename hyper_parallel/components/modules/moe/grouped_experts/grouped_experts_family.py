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
"""Opt-in expert family using the shared module-family mechanism from PR #1523.

Register local/EP and the planned Mega/MoonEP members in one family. Unfinished
kernels fail construction explicitly instead of falling back to local math.
The MoE package exports the factory and config; the legacy production entry is unchanged.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    FamilyShardingContext, ModuleFamily, ResolvedFamily, SelectorSpec, VariantRule,
    build_family, register_module_family,
)
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig, GroupedExpertsModule,
)
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


def validate_grouped_experts(
    *, config: GroupedExpertsConfig | None, module: nn.Module | None, context: Any, member: ResolvedFamily,
) -> None:
    """Validate expert count and reject unfinished execution before adopting a source."""
    del context
    if module is not None and not isinstance(module, GroupedExpertsModule):
        gate_up = getattr(module, "gate_up_proj", None)
        if not isinstance(gate_up, nn.Parameter) or gate_up.ndim != 3:
            raise ValueError("staged experts require a canonical member or a packed expert source")
        num_experts = gate_up.shape[0]
    else:
        facts = config if module is None else module.config
        if not isinstance(facts, GroupedExpertsConfig):
            raise ValueError("staged experts require canonical GroupedExpertsConfig")
        num_experts = facts.num_experts
    if num_experts % member.axis_sizes["ep"]:
        raise ValueError("EP size must divide the global expert count")
    if member.variant not in ("local", "ep"):
        raise NotImplementedError(f"expert member {member.variant!r} is registered but execution is not implemented")


def grouped_experts_sharding(
    *, module: nn.Module, module_fqn: str, member: ResolvedFamily, context: FamilyShardingContext,
) -> ModuleShardingSpec:
    """Use the same ownership rule as the PR #1523 expert example.

    Every canonical parameter starts with E, so EP uses Shard(0) for both projections.
    Local parameters and boundary I/O have explicit empty layout declarations;
    explicit stages own activation transport, and FSDP owns storage sharding.
    """
    del module_fqn, context
    placements = {"ep": Shard(0)} if member.selectors["parallel"] else {}
    return ModuleShardingSpec(
        params={name: dict(placements) for name, _ in module.named_parameters()},
        in_src={}, in_dst={}, is_boundary=False,
    )


@module_replacement
def GroupedExperts(  # pylint: disable=invalid-name
    *, config: GroupedExpertsConfig | None = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Any = None, mesh: Any = None,
    megakernel: bool = False, moonep: bool = False,
) -> nn.Module:
    """Construct the local/EP member through the existing shared builder.

    Args:
        config: Global algorithm configuration, mutually exclusive with module.
        module: Existing canonical member or an unsharded packed SwiGLU expert module.
        module_fqn: Model-owned expert boundary.
        context: Framework construction facts.
        mesh: Existing job mesh, outside serialized family arguments.
        megakernel: Select the reserved fused Mega member when True.
        moonep: Select the reserved MoonEP execution member when True.

    Note:
        Local NPU grouped-GEMM and ordinary EP slot transport are implemented.
        Dense TP uses TP-extend-EP with complete expert matrices (ETP=1).
        Selecting a registered Mega/MoonEP placeholder raises before parameter allocation.
    """
    return build_family(GroupedExperts, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, megakernel=megakernel, moonep=moonep)


register_module_family(ModuleFamily(
    name="routed_grouped_experts", baseline=GroupedExperts,
    variants={
        "local": "hyper_parallel.components.modules.moe.grouped_experts.grouped_experts.GroupedExpertsModule",
        "ep": "hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts.EPGroupedExperts",
        "mega": "hyper_parallel.components.modules.moe.grouped_experts.mega_grouped_experts.MegaGroupedExperts",
        "ep_mega": ("hyper_parallel.components.modules.moe.grouped_experts.ep."
                    "ep_mega_grouped_experts.EPMegaGroupedExperts"),
        "ep_moonep": ("hyper_parallel.components.modules.moe.grouped_experts.ep."
                      "moonep_grouped_experts.MoonEPGroupedExperts"),
    },
    default_variant="local", parallel_axis="ep",
    selector_specs={"megakernel": SelectorSpec(bool, False), "moonep": SelectorSpec(bool, False)},
    variant_rules=(
        VariantRule({"megakernel": False, "moonep": False, "parallel": True}, "ep"),
        VariantRule({"megakernel": True, "moonep": False, "parallel": False}, "mega"),
        VariantRule({"megakernel": True, "moonep": False, "parallel": True}, "ep_mega"),
        VariantRule({"megakernel": False, "moonep": True, "parallel": True}, "ep_moonep"),
    ),
    sharding=grouped_experts_sharding, validate=validate_grouped_experts,
))
