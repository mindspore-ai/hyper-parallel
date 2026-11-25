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

Register local/EP/Mega/MoonEP members in one family with shared algorithm semantics.
The MoE package exports the factory and config; the legacy production entry is unchanged.
"""

from __future__ import annotations

import warnings
from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    FamilyShardingContext,
    ModuleFamily,
    ResolvedFamily,
    SelectorSpec,
    VariantRule,
    build_family,
    register_module_family,
)
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig,
    GroupedExpertsModule,
)
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


def _validate_portable_selection(member: ResolvedFamily) -> None:
    """Keep portable compute selection independent of accelerator transport choices."""
    if not member.selectors["use_grouped_gemm"] and (member.selectors["megakernel"] or member.selectors["moonep"]):
        raise ValueError("portable Torch experts cannot select Mega or MoonEP kernels")


def validate_grouped_experts(
    *, config: GroupedExpertsConfig | None, module: nn.Module | None, context: Any, member: ResolvedFamily,
) -> None:
    """Validate expert ownership and backend-specific transport options."""
    del context
    _validate_portable_selection(member)
    if not isinstance(member.arguments.get("overlap_shared_expert", True), bool):
        raise ValueError("overlap_shared_expert must be bool")
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
    if member.axis_sizes["ep"] == 1 and any((member.selectors["megakernel"], member.selectors["moonep"])):
        warnings.warn(
            "megakernel=True or moonep=True requires EP>1; EP=1 falls back to ordinary GroupedExperts. "
            "Mega transport options are ignored.",
            UserWarning, stacklevel=3,
        )
    elif not member.selectors["megakernel"] and (
            member.arguments.get("dispatch_mode", "push") != "push"
            or member.arguments.get("initial_capacity_factor") is not None
            or member.arguments.get("capacity_growth_factor") is not None):
        raise ValueError("Mega transport options require megakernel=True")


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
    megakernel: bool = False, moonep: bool = False, overlap_shared_expert: bool = True,
    use_grouped_gemm: bool = True, activation_dtype: str = "input", swiglu_limit: float | None = None,
    dispatch_mode: str = "push", initial_capacity_factor: float | None = None,
    capacity_growth_factor: float | None = None, moonep_replica_capacity: int | None = None,
) -> nn.Module:
    """Construct the local/EP member through the existing shared builder.

    Args:
        config: Global algorithm configuration, mutually exclusive with module.
        module: Existing canonical member or an unsharded packed SwiGLU expert module.
        module_fqn: Model-owned expert boundary.
        context: Framework construction facts.
        mesh: Existing job mesh, outside serialized family arguments.
        megakernel: Select fused Mega transport/compute for EP>1; EP=1 warns and uses local GMM.
        moonep: Select collective MoonEP with dynamic execution replicas.
        overlap_shared_expert: Let ordinary/Moon EP schedule a bound shared family member.
        use_grouped_gemm: False selects source-compatible portable Torch execution.
        activation_dtype: Precision of the portable/source SwiGLU calculation.
        swiglu_limit: Optional explicit source clamp; omitted uses its own configuration.
        dispatch_mode: Mega transport policy, push or pull.
        initial_capacity_factor: Mega push receive-capacity factor; None uses the backend default.
        capacity_growth_factor: Mega push growth factor; None uses the backend default.
        moonep_replica_capacity: Maximum remote expert rows per rank; None means E/EP.

    Note:
        Local NPU grouped-GEMM and ordinary EP slot transport are implemented.
        Dense TP uses TP-extend-EP with complete expert matrices (ETP=1).
        Mega uses the optional HyperMulticore executor. MoonEP uses ordinary differentiable collective transport.
    """
    return build_family(GroupedExperts, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, megakernel=megakernel, moonep=moonep,
                        overlap_shared_expert=overlap_shared_expert,
                        use_grouped_gemm=use_grouped_gemm, activation_dtype=activation_dtype, swiglu_limit=swiglu_limit,
                        dispatch_mode=dispatch_mode, initial_capacity_factor=initial_capacity_factor,
                        capacity_growth_factor=capacity_growth_factor, moonep_replica_capacity=moonep_replica_capacity)


register_module_family(ModuleFamily(
    name="routed_grouped_experts", baseline=GroupedExperts,
    variants={
        "torch": "hyper_parallel.components.modules.moe.grouped_experts.torch_grouped_experts.TorchGroupedExperts",
        "ep_torch": "hyper_parallel.components.modules.moe.grouped_experts.torch_grouped_experts.EPTorchGroupedExperts",
        "local": "hyper_parallel.components.modules.moe.grouped_experts.grouped_experts.GroupedExpertsModule",
        "ep": "hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts.EPGroupedExperts",
        "ep_mega": ("hyper_parallel.components.modules.moe.grouped_experts.ep."
                    "ep_mega_grouped_experts.EPMegaGroupedExperts"),
        "ep_moonep": ("hyper_parallel.components.modules.moe.grouped_experts.ep."
                      "moonep_grouped_experts.MoonEPGroupedExperts"),
    },
    default_variant="local", parallel_axis="ep",
    selector_specs={"megakernel": SelectorSpec(bool, False), "moonep": SelectorSpec(bool, False),
                    "use_grouped_gemm": SelectorSpec(bool, True)},
    variant_rules=(
        VariantRule({"megakernel": False, "moonep": False, "use_grouped_gemm": False, "parallel": False}, "torch"),
        VariantRule({"megakernel": False, "moonep": False, "use_grouped_gemm": False, "parallel": True}, "ep_torch"),
        VariantRule({"use_grouped_gemm": True, "megakernel": False, "moonep": False, "parallel": True}, "ep"),
        VariantRule({"use_grouped_gemm": True, "megakernel": True, "moonep": False, "parallel": False}, "local"),
        VariantRule({"use_grouped_gemm": True, "megakernel": True, "moonep": False, "parallel": True}, "ep_mega"),
        VariantRule({"use_grouped_gemm": True, "megakernel": False, "moonep": True, "parallel": False}, "local"),
        VariantRule({"use_grouped_gemm": True, "megakernel": False, "moonep": True, "parallel": True}, "ep_moonep"),
    ),
    sharding=grouped_experts_sharding, validate=validate_grouped_experts,
))
