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
"""Explicit stage protocols and resource-only binding for family and native modules."""

from __future__ import annotations

import inspect
from copy import deepcopy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import _same_mesh, get_family_build_info
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed.context_parallel.cp_context import CPContext
from hyper_parallel.distributed.expert_parallel.ep_context import EPContext


@dataclass(frozen=True)
class ParallelModuleSpec:
    """A class-owned execution protocol; validation proves its supported layouts."""

    protocol: str
    binding_kind: str = "none"
    stages: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    validate: Callable | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1"
    validate_materialized: Callable | None = None

    def __post_init__(self) -> None:
        if not self.protocol or not self.version:
            raise ValueError("parallel protocol and version must be non-empty")
        if self.binding_kind not in ("none", "cp", "ep", "mesh"):
            raise ValueError("binding_kind must be none, cp, ep or mesh")
        if self.binding_kind != "none" and (not self.stages or self.validate is None):
            raise ValueError("parallel members must declare stage signatures and a layout validator")
        object.__setattr__(self, "stages", MappingProxyType({name: tuple(args) for name, args in self.stages.items()}))
        object.__setattr__(self, "options", MappingProxyType(deepcopy(dict(self.options))))


@dataclass(frozen=True)
class NativeParallelBindingRequest:
    """Bind an existing component whose class already delegates explicit stages."""

    match: tuple[str, ...]
    parallel: ParallelModuleSpec
    sharding: Callable

    def __post_init__(self) -> None:
        if isinstance(self.match, str) or not self.match or not all(isinstance(name, str) for name in self.match):
            raise ValueError("native matches must be a non-empty tuple of module FQN patterns")
        if not callable(self.sharding) or not isinstance(self.parallel, ParallelModuleSpec):
            raise ValueError("native binding requires a complete layout provider and ParallelModuleSpec")
        object.__setattr__(self, "match", tuple(self.match))


@dataclass(frozen=True)
class MeshBinding:
    """Borrowed topology; all communication groups are owned by MeshContext."""

    mesh_context: Any

    def axis_mesh(self, axis: str) -> Any:
        """Return a prebuilt one-dimensional mesh without constructing groups.

        Args:
            axis: cp or ep, in the runtime's established rank order.
        """
        if axis == "ep":
            region = self.mesh_context.fsdp_moe_mesh
            if region is not None:
                return region["ep"]
        region = self.mesh_context.device_mesh
        if region is None or axis not in tuple(region.mesh_dim_names or ()):
            raise ValueError(f"no prebuilt {axis!r} mesh is available for parallel binding")
        return region[axis]


@dataclass(frozen=True)
class ParallelBinding:
    """One resource object injected through a class-defined bind_parallel method."""

    mesh_binding: MeshBinding
    cp_context: CPContext | None = None
    ep_context: EPContext | None = None


def validate_parallel_module(module: nn.Module, parallel: ParallelModuleSpec, sharding: Any, context: Any) -> None:
    """Validate class stages and capabilities after owned-layout port checks.

    Args:
        module: Actual final member or native component.
        parallel: Versioned stage declaration from the class/model spec.
        sharding: Final ModuleShardingSpec after user overrides.
        context: Immutable logical layout context.
    """
    for name, arguments in parallel.stages.items():
        implementation = inspect.getattr_static(type(module), name, None)
        if not callable(implementation):
            raise ValueError(f"parallel stage {name!r} must be implemented in the module class")
        inspect.signature(getattr(module, name)).bind(**dict.fromkeys(arguments))
    if parallel.binding_kind != "none":
        implementation = inspect.getattr_static(type(module), "bind_parallel", None)
        if not callable(implementation):
            raise ValueError("parallel modules must define bind_parallel(binding) in their class")
        inspect.signature(module.bind_parallel).bind(None)
    if getattr(sharding, "_ep_stack", None):
        raise ValueError("explicit expert stages cannot inherit legacy whole-MoE stacking")
    if sharding.in_src != sharding.in_dst:
        raise ValueError("explicit stage modules must receive their declared computation layout directly")
    if sharding.out_dst is not None and sharding.out_src != sharding.out_dst:
        raise ValueError("explicit stage modules must restore their declared output layout themselves")
    if parallel.binding_kind == "ep":
        _validate_expert_ownership(module, parallel, sharding, context)
    if parallel.validate is not None:
        parallel.validate(module=module, sharding=sharding, context=context)


def _validate_expert_ownership(module, parallel, sharding, context):
    """Check the bound EP mesh against local and global expert ownership."""
    experts = parallel.options.get("num_experts")
    expert_type = type(experts)
    if expert_type is not int or experts < 1 or experts % context.axis_sizes["ep"]:
        raise ValueError("EP protocols must declare a divisible global num_experts")
    sharded = {name: named["ep"] for name, named in sharding.params.items()
               if isinstance(named.get("ep"), Shard)}
    if not sharded:
        raise ValueError("EP transport requires explicit DTensor expert ownership")
    for name, placement in sharded.items():
        parameter = module.get_parameter(name)
        if placement.dim != 0 or parameter.shape[0] != experts:
            raise ValueError("EPContext supports balanced contiguous expert Shard(0) ownership")


def bind_parallel_modules(model: nn.Module, plan: Any, mesh_context: Any) -> None:
    """Inject shared resources after layout and before activation wrappers/FSDP2.

    Args:
        model: Final model tree with computation-sharded parameters.
        plan: Plan carrying validated family/native protocol declarations.
        mesh_context: Existing HyperParallel MeshContext; never a new topology.
    """
    mesh_binding = MeshBinding(mesh_context)
    prepared = []
    for fqn, parallel in plan.parallel_modules.items():
        if parallel.binding_kind == "none":
            continue
        module = model.get_submodule(fqn)
        existing = getattr(module, "_hp_parallel_binding", None)
        if existing is not None:
            info = get_family_build_info(module)
            cached = info.layout if info is not None else None
            if (cached is None or not cached.applied or cached.parallel != parallel
                    or not _same_mesh(existing.mesh_binding.mesh_context, mesh_context)):
                raise ValueError(f"module {fqn!r} already has a communication owner")
            continue
        cp = CPContext(mesh_binding.axis_mesh("cp")) if parallel.binding_kind == "cp" else None
        ep = (EPContext(mesh_binding.axis_mesh("ep"), num_experts=parallel.options["num_experts"])
              if parallel.binding_kind == "ep" else None)
        prepared.append((module, ParallelBinding(mesh_binding, cp, ep)))
    for module, binding in prepared:
        module.bind_parallel(binding)
        module._hp_parallel_binding = binding  # pylint: disable=protected-access
    model._hp_parallel_module_specs = dict(plan.parallel_modules)  # pylint: disable=protected-access


def validate_materialized_modules(model: nn.Module) -> None:
    """Check final device/kernel resources after materialization and before compile.

    Args:
        model: Final loaded model whose class protocols may validate physical resources.
    """
    for fqn, parallel in getattr(model, "_hp_parallel_module_specs", {}).items():
        module = model.get_submodule(fqn)
        if parallel.validate_materialized is not None:
            parallel.validate_materialized(module=module, binding=getattr(module, "_hp_parallel_binding", None))
