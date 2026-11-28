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
"""Unified routed-expert family registering local, EP, Mega, MoonEP, and low-precision members."""

from __future__ import annotations

from typing import Any
import warnings

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

_LP_PACKAGE = "hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts"
_LP_SELECTOR_MODES: tuple[tuple[dict[str, Any], str], ...] = (
    (
        {"use_mxfp8": True, "fused": False, "weight_format": "", "act_format": "", "is_fake_quantize": False},
        "mxfp8",
    ),
    (
        {"use_mxfp8": True, "fused": True, "weight_format": "", "act_format": "", "is_fake_quantize": False},
        "fused_mxfp8",
    ),
    (
        {
            "use_mxfp8": False,
            "fused": False,
            "weight_format": "mxfp8",
            "act_format": "mxfp8",
            "is_fake_quantize": False,
        },
        "w8a8",
    ),
    (
        {
            "use_mxfp8": False,
            "fused": True,
            "weight_format": "mxfp8",
            "act_format": "mxfp8",
            "is_fake_quantize": False,
        },
        "fused_mxfp8",
    ),
    (
        {
            "use_mxfp8": False,
            "fused": False,
            "weight_format": "mxfp4",
            "act_format": "mxfp8",
            "is_fake_quantize": False,
        },
        "w4a8",
    ),
    (
        {
            "use_mxfp8": False,
            "fused": False,
            "weight_format": "mxfp4",
            "act_format": "mxfp8",
            "is_fake_quantize": True,
        },
        "fake_w4a8",
    ),
    (
        {
            "use_mxfp8": False,
            "fused": False,
            "weight_format": "hif8",
            "act_format": "hif8",
            "is_fake_quantize": False,
        },
        "hif8",
    ),
)
_DEFAULT_LP_SELECTORS = {
    "use_mxfp8": False,
    "fused": False,
    "weight_format": "",
    "act_format": "",
    "is_fake_quantize": False,
}


def validate_grouped_experts(
    *, config: GroupedExpertsConfig | None, module: nn.Module | None, context: Any, member: ResolvedFamily,
) -> None:
    """Validate expert ownership and backend-specific transport options."""
    del context
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
    dispatch_mode: str = "push", initial_capacity_factor: float | None = None,
    capacity_growth_factor: float | None = None, moonep_replica_capacity: int | None = None,
    use_mxfp8: bool = False, fused: bool = False,
    weight_format: str = "", act_format: str = "", is_fake_quantize: bool = False,
) -> nn.Module:
    """Construct the local/EP/MoonEP/Mega/low-precision member through the shared builder.

    Args:
        config: Global algorithm configuration, mutually exclusive with module.
        module: Existing canonical member or an unsharded packed SwiGLU expert module.
        module_fqn: Model-owned expert boundary.
        context: Framework construction facts.
        mesh: Existing job mesh, outside serialized family arguments.
        megakernel: Select fused Mega transport/compute for EP>1; EP=1 warns and uses local GMM.
        moonep: Select collective MoonEP with dynamic execution replicas.
        overlap_shared_expert: Let ordinary/Moon EP schedule a bound shared family member.
        dispatch_mode: Mega transport policy, push or pull.
        initial_capacity_factor: Mega push receive-capacity factor; None uses the backend default.
        capacity_growth_factor: Mega push growth factor; None uses the backend default.
        moonep_replica_capacity: Maximum remote expert rows per rank; None means E/EP.
        use_mxfp8: Explicit MXFP8 selector when not using a named dtype scheme.
        fused: Select fused MXFP8 SwiGLU dual-axis quantization when paired with MXFP8.
        weight_format: Quantized weight format selector injected from a dtype scheme.
        act_format: Quantized activation format selector injected from a dtype scheme.
        is_fake_quantize: Fake-quantization selector injected from a dtype scheme.
    """
    return build_family(
        GroupedExperts, config=config, module=module, module_fqn=module_fqn,
        context=context, mesh=mesh, megakernel=megakernel, moonep=moonep,
        overlap_shared_expert=overlap_shared_expert,
        dispatch_mode=dispatch_mode, initial_capacity_factor=initial_capacity_factor,
        capacity_growth_factor=capacity_growth_factor, moonep_replica_capacity=moonep_replica_capacity,
        use_mxfp8=use_mxfp8, fused=fused, weight_format=weight_format,
        act_format=act_format, is_fake_quantize=is_fake_quantize,
    )


def _build_variants_and_rules() -> tuple[dict[str, str], tuple[VariantRule, ...]]:
    """Assemble BF16 and low-precision variants across local, EP, MoonEP, and Mega topologies."""
    variants = {
        "local": "hyper_parallel.components.modules.moe.grouped_experts.grouped_experts.GroupedExpertsModule",
        "ep": "hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts.EPGroupedExperts",
        "ep_mega": (
            "hyper_parallel.components.modules.moe.grouped_experts.ep."
            "ep_mega_grouped_experts.EPMegaGroupedExperts"
        ),
        "ep_moonep": (
            "hyper_parallel.components.modules.moe.grouped_experts.ep."
            "moonep_grouped_experts.MoonEPGroupedExperts"
        ),
    }
    for mode in ("mxfp8", "fused_mxfp8", "w8a8", "w4a8", "fake_w4a8", "hif8"):
        variants[f"local_{mode}"] = f"{_LP_PACKAGE}.build_local_{mode}_experts"
        variants[f"ep_{mode}"] = f"{_LP_PACKAGE}.build_ep_{mode}_experts"
        variants[f"ep_moonep_{mode}"] = f"{_LP_PACKAGE}.build_moonep_{mode}_experts"

    rules: list[VariantRule] = [
        VariantRule({"megakernel": False, "moonep": False, **_DEFAULT_LP_SELECTORS, "parallel": True}, "ep"),
        VariantRule({"megakernel": True, "moonep": False, **_DEFAULT_LP_SELECTORS, "parallel": False}, "local"),
        VariantRule({"megakernel": True, "moonep": False, **_DEFAULT_LP_SELECTORS, "parallel": True}, "ep_mega"),
        VariantRule({"megakernel": False, "moonep": True, **_DEFAULT_LP_SELECTORS, "parallel": False}, "local"),
        VariantRule({"megakernel": False, "moonep": True, **_DEFAULT_LP_SELECTORS, "parallel": True}, "ep_moonep"),
    ]
    for lp_cond, mode in _LP_SELECTOR_MODES:
        rules.extend((
            VariantRule({"megakernel": False, "moonep": False, **lp_cond, "parallel": False}, f"local_{mode}"),
            VariantRule({"megakernel": False, "moonep": True, **lp_cond, "parallel": False}, f"local_{mode}"),
            VariantRule({"megakernel": False, "moonep": False, **lp_cond, "parallel": True}, f"ep_{mode}"),
            VariantRule({"megakernel": False, "moonep": True, **lp_cond, "parallel": True}, f"ep_moonep_{mode}"),
        ))
    return variants, tuple(rules)


_VARIANTS, _VARIANT_RULES = _build_variants_and_rules()

register_module_family(ModuleFamily(
    name="routed_grouped_experts",
    baseline=GroupedExperts,
    variants=_VARIANTS,
    default_variant="local",
    parallel_axis="ep",
    selector_specs={
        "megakernel": SelectorSpec(bool, False),
        "moonep": SelectorSpec(bool, False),
        "use_mxfp8": SelectorSpec(bool, False),
        "fused": SelectorSpec(bool, False, aliases=("use_fused_mxfp8",)),
        "weight_format": SelectorSpec(str, ""),
        "act_format": SelectorSpec(str, ""),
        "is_fake_quantize": SelectorSpec(bool, False),
    },
    variant_rules=_VARIANT_RULES,
    sharding=grouped_experts_sharding,
    validate=validate_grouped_experts,
))
