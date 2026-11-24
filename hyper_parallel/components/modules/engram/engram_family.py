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
"""Engram family with device-table parallelism and a separately managed host table."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    FamilyShardingContext, ModuleFamily, ResolvedFamily, SelectorSpec, VariantRule,
    build_family, register_module_family,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


def _engram_sharding(
    *, module: nn.Module, module_fqn: str, member: ResolvedFamily, context: FamilyShardingContext,
) -> ModuleShardingSpec:
    """Keep projection and scale parameters replicated and shard device table rows across EP."""
    del module_fqn, member
    params = {name: {"tp": Replicate()} for name, _ in module.named_parameters()}
    if context.axis_sizes["ep"] > 1:
        params["embed.weight"] = {"ep": Shard(0)}
    return ModuleShardingSpec(params=params, in_src={}, in_dst={}, is_boundary=False)


def _validate_engram_layout(
    *, module: nn.Module, sharding: ModuleShardingSpec, context: FamilyShardingContext,
) -> None:
    """Validate final row ownership and preserve the decoder's token layout."""
    if sharding.is_boundary or any((sharding.in_src, sharding.in_dst, sharding.out_src, sharding.out_dst)):
        raise ValueError("Engram consumes and returns the decoder's existing token layout")
    expected = {"embed.weight", "wkv.weight", "q_weight", "k_weight"}
    if set(sharding.params) != expected:
        raise ValueError("Device Engram requires complete table, projection and scale ownership")
    ep_size = context.axis_sizes["ep"]
    for name, axes in sharding.params.items():
        shards = {axis: placement for axis, placement in axes.items() if placement != Replicate()}
        required = {"ep": Shard(0)} if name == "embed.weight" and ep_size > 1 else {}
        if shards != required:
            raise ValueError(f"Unsupported Engram layout for {name}: expected {required}")
    rows = module.padded_num_embeddings
    if module.embed.weight.shape[0] != rows or rows % ep_size:
        raise ValueError(f"Engram padded table rows must match the embedding and be divisible by EP size {ep_size}")


def _engram_parallel_spec(*, module: nn.Module, member: ResolvedFamily) -> ParallelModuleSpec:
    """Borrow the job meshes instead of an expert slot-dispatch context."""
    del module
    if any(member.axis_sizes[axis] > 1 for axis in ("ep", "cp", "tp")):
        return ParallelModuleSpec(
            protocol="engram.table.mesh", binding_kind="mesh",
            stages={"forward": ("hidden_states", "input_ids", "segment_starts", "token_mask")},
            validate=_validate_engram_layout,
        )
    return ParallelModuleSpec(protocol="engram.table.local", validate=_validate_engram_layout)


def _device_engram(
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
) -> nn.Module:
    """Build the device family that owns its table layout and execution."""
    return build_family(_device_engram, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh)


def _host_engram(
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
) -> nn.Module:
    """Plan Host row ownership before the model materializes its external table."""
    return build_family(_host_engram, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh)


@module_replacement
def build_engram(
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
    engram_storage_backend: str = "device",
) -> nn.Module:
    """Select a planner-owned Device or Host Engram family."""
    if engram_storage_backend == "device":
        return _device_engram(config=config, module=module, module_fqn=module_fqn, context=context, mesh=mesh)
    if engram_storage_backend == "host":
        return _host_engram(config=config, module=module, module_fqn=module_fqn, context=context, mesh=mesh)
    raise ValueError("engram_storage_backend must be 'device' or 'host'")


@module_replacement
def Engram(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
    engram_storage_backend: str = "device",
) -> nn.Module:
    """Build a member for recipes that already declare table placement and forward binding."""
    return build_family(Engram, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, engram_storage_backend=engram_storage_backend)


register_module_family(ModuleFamily(
    name="engram", baseline=Engram,
    variants={
        "local": "hyper_parallel.components.modules.engram.engram.EngramModule",
        "host": "hyper_parallel.components.modules.engram.host_engram.HostEngramModule",
    },
    selector_specs={"engram_storage_backend": SelectorSpec(str, "device")},
    default_variant="local",
    variant_rules=(VariantRule({"engram_storage_backend": "host", "parallel": False}, "host"),),
))

register_module_family(ModuleFamily(
    name="device_engram", baseline=_device_engram,
    variants={
        "local": "hyper_parallel.components.modules.engram.engram.EngramModule",
        "ep": "hyper_parallel.components.modules.engram.engram.EPEngramModule",
    },
    default_variant="local", parallel_axis="ep",
    variant_rules=(VariantRule({"parallel": True}, "ep"),),
    sharding=_engram_sharding, parallel_spec=_engram_parallel_spec,
))

register_module_family(ModuleFamily(
    name="host_engram", baseline=_host_engram,
    variants={"host": "hyper_parallel.components.modules.engram.host_engram.HostEngramModule"},
    default_variant="host",
    sharding=_engram_sharding, parallel_spec=_engram_parallel_spec,
))
