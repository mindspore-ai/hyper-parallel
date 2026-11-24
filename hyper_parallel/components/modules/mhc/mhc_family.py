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
"""mHC replacement entry points and family declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.components.modules.family import SelectorSpec, VariantRule
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec


@module_replacement
def MHCPre(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
) -> nn.Module:
    """Construct a tracked mhc_pre member without changing its execution contract."""
    return build_family(MHCPre, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh)


register_module_family(ModuleFamily(
    name="mhc_pre", baseline=MHCPre,
    variants={"local": "hyper_parallel.components.modules.mhc.mhc_pre.MhcPreModule"}, default_variant="local",
))


@module_replacement
def MHCPost(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
) -> nn.Module:
    """Construct a tracked mhc_post member without changing its execution contract."""
    return build_family(MHCPost, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh)


register_module_family(ModuleFamily(
    name="mhc_post", baseline=MHCPost,
    variants={"local": "hyper_parallel.components.modules.mhc.mhc_post.MhcPostModule"}, default_variant="local",
))


def pipelined_mhc_sharding(*, module: nn.Module, module_fqn: str, member: Any,
                           context: Any) -> ModuleShardingSpec:
    """Own every coefficient matrix and keep token-local coefficient outputs aligned."""
    del module_fqn, member
    replicated = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    tokens = dict(replicated)
    if context.axis_sizes["cp"] > 1:
        tokens["cp"] = Shard(1)
    if context.axis_sizes["tp"] > 1 and context.sequence_parallel:
        tokens["tp"] = Shard(1)
    outputs = {name: dict(tokens) for name in ("pre", "post", "residual")}
    return ModuleShardingSpec(
        params={name: dict(replicated) for name, _ in module.named_parameters()},
        in_src={"hidden_streams": dict(tokens)}, in_dst={"hidden_streams": dict(tokens)},
        out_src=outputs, out_dst={name: dict(value) for name, value in outputs.items()},
        out_names=("pre", "post", "residual"), is_boundary=True,
    )


@module_replacement
def PipelinedMhc(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
    shifted: bool = False, use_fused_ops: bool = False, token_tile: int = 32,
) -> nn.Module:
    """Construct a tracked PipelinedMhc member without changing its execution contract."""
    return build_family(PipelinedMhc, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, shifted=shifted, use_fused_ops=use_fused_ops, token_tile=token_tile)


def _validate_pipelined(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Reject fused execution without the matching shifted boundary protocol."""
    del config, module, context
    if member.selectors["use_fused_ops"] and not member.selectors["shifted"]:
        raise ValueError("Native pipelined mHC requires shifted=True")
    if member.selectors["use_fused_ops"] and any(member.axis_sizes[axis] > 1 for axis in ("tp", "cp")):
        raise NotImplementedError("Native shifted mHC requires TP=CP=1 until its advance boundary is qualified")


register_module_family(ModuleFamily(
    name="pipelined_mhc", baseline=PipelinedMhc,
    variants={"local": "hyper_parallel.components.modules.mhc.pipelined_mhc.PipelinedMhcModule",
              "shifted": "hyper_parallel.components.modules.mhc.pipelined_mhc.ShiftedPipelinedMhcModule",
              "shifted_npu": "hyper_parallel.components.modules.mhc.pipelined_mhc.NativeShiftedPipelinedMhcModule"},
    default_variant="local", selector_specs={"shifted": SelectorSpec(bool, False),
                                             "use_fused_ops": SelectorSpec(bool, False)},
    variant_rules=(VariantRule({"shifted": True, "use_fused_ops": False, "parallel": False}, "shifted"),
                   VariantRule({"shifted": True, "use_fused_ops": True, "parallel": False}, "shifted_npu")),
    validate=_validate_pipelined, sharding=pipelined_mhc_sharding,
))


@module_replacement
def MegaMhc(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Mapping[str, Any] | None = None, mesh: Any = None, backend: str = "torch",
) -> nn.Module:
    """Build a complete shifted boundary with an explicit execution backend."""
    return build_family(MegaMhc, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, backend=backend)


register_module_family(ModuleFamily(
    name="mega_mhc", baseline=MegaMhc,
    variants={"local": "hyper_parallel.components.modules.mhc.mega_mhc.MegaMhcModule",
              "fused": "hyper_parallel.components.modules.mhc.mega_mhc.FusedMegaMhcModule"},
    selector_specs={"backend": SelectorSpec(str, "torch")},
    default_variant="local",
    variant_rules=(VariantRule({"backend": "fused", "parallel": False}, "fused"),),
))
