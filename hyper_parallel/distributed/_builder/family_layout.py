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
"""Discover complete module-owned layouts before generic parameter templates."""

from __future__ import annotations

import copy
import fnmatch
import inspect
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    collect_family_modules, _family_parameter_shapes, get_family_build_info,
    _record_family_layouts, _validate_applied_layout,
)
from hyper_parallel.core.dtensor.placement_types import Partial
from hyper_parallel.distributed.module_binding import (
    NativeParallelBindingRequest, ParallelModuleSpec, validate_materialized_modules, validate_parallel_module,
)
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, _normalize_out_fields


@dataclass
class OwnedLayouts:
    """Construction-local ownership, indexed by final Module/Parameter identity."""

    specs: dict[str, ModuleShardingSpec] = field(default_factory=dict)
    parallel: dict[str, ParallelModuleSpec] = field(default_factory=dict)
    owned_names: set[str] = field(default_factory=set)
    expected_ids: set[int] = field(default_factory=set)
    required_by_fqn: dict[str, set[int]] = field(default_factory=dict)
    aliases: list[tuple[str, str]] = field(default_factory=list)


def _full(fqn, name):
    return f"{fqn}.{name}" if fqn else name


def _is_child(fqn, parent):
    return fqn.startswith(parent + ".") if parent else bool(fqn)


def _module_parallel(module, member):
    provider = getattr(module, "parallel_module_spec", None)
    declaration = provider() if provider is not None else None
    if declaration is not None and not isinstance(declaration, ParallelModuleSpec):
        raise TypeError("parallel_module_spec() must return ParallelModuleSpec")
    if member is not None and member.selectors["parallel"] and declaration is None:
        raise ValueError("a parallel family member must declare its explicit execution protocol")
    return declaration


def _add_provider(entries, model, fqn, sharding, parallel):
    module = model.get_submodule(fqn)
    identity = id(module)
    if identity in entries:
        raise ValueError(f"multiple layout/communication owners for module {fqn!r}")
    if not isinstance(sharding, ModuleShardingSpec):
        raise TypeError("layout providers must return ModuleShardingSpec")
    sharding = copy.deepcopy(sharding)
    _normalize_out_fields(sharding)
    entries[identity] = (fqn, module, sharding, parallel)


def _native_providers(entries, model, spec, context, boundaries):
    """Locate model-owned sharding providers for native parallel components."""
    if spec is None or spec.native_parallel is None:
        return
    requests = spec.native_parallel(model=model, context=context)
    all_modules = tuple(model.named_modules(remove_duplicate=False))
    family_subtrees = {id(child) for boundary in boundaries for child in boundary.module.modules()}
    for request in requests:
        if not isinstance(request, NativeParallelBindingRequest):
            raise TypeError("native_parallel must return NativeParallelBindingRequest objects")
        matched = set()
        for pattern in request.match:
            hits = [(fqn, module) for fqn, module in all_modules if fnmatch.fnmatchcase(fqn, pattern)]
            if not hits:
                raise ValueError(f"native binding pattern {pattern!r} matched no module")
            for fqn, module in hits:
                if id(module) in matched:
                    continue
                if _overlaps_family(module, family_subtrees):
                    raise ValueError(f"native binding at {fqn!r} overlaps a family-owned subtree")
                matched.add(id(module))
                layout = request.sharding(module=module, module_fqn=fqn, member=None, context=context)
                _add_provider(entries, model, fqn, layout, request.parallel)


def _overlaps_family(module, family_subtrees):
    return any(id(child) in family_subtrees for child in module.modules())


def _validate_ports(module, spec):
    """Validate named input and output layout declarations against the module interface."""
    injection_fields = ("local_compute_fn", "inner_wrapper", "inner_target", "region_dispatch")
    if any(getattr(spec, name) is not None for name in injection_fields):
        raise ValueError("module-owned layouts cannot install legacy compute injection")
    if spec.params is None or spec.in_src is None or spec.in_dst is None:
        raise ValueError("module-owned providers must declare params, in_src and in_dst explicitly")
    if set(spec.in_src) != set(spec.in_dst):
        raise ValueError("module-owned in_src and in_dst must declare the same input names")
    signature = inspect.signature(module.forward)
    accepts_kwargs = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values())
    unknown = set(spec.in_src) - set(signature.parameters)
    if unknown and not accepts_kwargs:
        raise ValueError(f"layout declares unknown forward inputs {sorted(unknown)}")
    if spec.out_names is not None and len(set(spec.out_names)) != len(spec.out_names):
        raise ValueError("output names must be unique")


def _validate_placements(spec, context):
    """Reject unknown mesh axes and incompatible parameter placement declarations."""
    mappings = [*spec.params.values(), *spec.in_src.values(), *spec.in_dst.values()]
    mappings.extend((spec.out_src or {}).values())
    mappings.extend((spec.out_dst or {}).values())
    for named in mappings:
        for axis, placement in named.items():
            if axis not in context.axis_sizes:
                raise ValueError(f"module layouts cannot declare storage/unknown axis {axis!r}")
            if not hasattr(placement, "is_shard"):
                raise TypeError("module providers must return concrete Placement objects")
    if any(isinstance(placement, Partial) for named in spec.params.values() for placement in named.values()):
        raise ValueError("parameter source layouts cannot be Partial")


def _claim_layout_parameters(fqn, parameters, spec, children, owner_by_id, canonical_by_id):
    """Assign each Parameter identity once and verify every alias has the same layout."""
    declared = set()
    for name, placement in tuple(spec.params.items()):
        parameter = parameters.get(name)
        if parameter is None:
            raise ValueError(f"layout parameter {_full(fqn, name)!r} does not exist")
        full_name = _full(fqn, name)
        if any(full_name.startswith(child + ".") for child in children):
            raise ValueError(f"parent layout duplicates nested module ownership at {full_name!r}")
        identity = id(parameter)
        declared.add(identity)
        if identity in owner_by_id:
            if owner_by_id[identity] != placement:
                raise ValueError(f"shared parameter {full_name!r} has conflicting layouts")
            del spec.params[name]
        else:
            owner_by_id[identity] = placement
            canonical_by_id[identity] = full_name
    return declared


def _assign_parameters(result, entries, model):
    """Assign one owned layout per logical parameter while retaining shared aliases."""
    all_parameters = dict(model.named_parameters(remove_duplicate=False))
    owner_by_id = {}
    canonical_by_id = {}
    for fqn, module, spec, _ in entries.values():
        parameters = dict(module.named_parameters(remove_duplicate=False))
        children = [name for name, *_ in entries.values() if _is_child(name, fqn)]
        expected = {id(parameter) for name, parameter in parameters.items()
                    if not any(_full(fqn, name).startswith(child + ".") for child in children)}
        declared = _claim_layout_parameters(fqn, parameters, spec, children, owner_by_id, canonical_by_id)
        if expected - declared:
            missing = [name for name, parameter in parameters.items() if id(parameter) in expected - declared]
            raise ValueError(f"module-owned layout {fqn!r} omits parameters {missing}")
        result.expected_ids.update(expected)
        result.specs[fqn] = spec
        result.required_by_fqn[fqn] = {id(parameters[name]) for name in spec.params}
    for name, parameter in all_parameters.items():
        identity = id(parameter)
        if identity not in owner_by_id:
            continue
        result.owned_names.add(name)
        canonical = canonical_by_id[identity]
        if canonical != name:
            result.aliases.append((canonical, name))


def collect_owned_layouts(model: nn.Module, context: Any, spec: Any = None) -> OwnedLayouts:
    """Discover model/native/family providers independently of generic TP templates.

    Args:
        model: Final logical model, before any parameter sharding.
        context: FamilyShardingContext with logical axes and global parameter shapes.
        spec: Effective per-build model specification, if any.
    """
    entries = {}
    if spec is not None and spec.sharding is not None:
        for fqn, layout in spec.sharding(model=model, context=context).items():
            _add_provider(entries, model, fqn, layout, _module_parallel(model.get_submodule(fqn), None))
    boundaries = collect_family_modules(model)
    for boundary in boundaries:
        member = boundary.info.resolved
        if member.sharding is None:
            continue
        fqn = boundary.fqns[0]
        cached = boundary.info.layout
        layout = (cached.sharding if cached is not None else
                  member.sharding(module=boundary.module, module_fqn=fqn, member=member, context=context))
        entries.pop(id(boundary.module), None)
        parallel = cached.parallel if cached is not None else _module_parallel(boundary.module, member)
        _add_provider(entries, model, fqn, layout, parallel)
    _native_providers(entries, model, spec, context, boundaries)
    _validate_parallel_nesting(entries)
    result = OwnedLayouts()
    for fqn, module, layout, parallel in entries.values():
        _validate_ports(module, layout)
        _validate_placements(layout, context)
        info = get_family_build_info(module)
        cached = info.layout if info is not None else None
        if parallel is not None and cached is None:
            validate_parallel_module(module, parallel, layout, context)
        if parallel is not None:
            result.parallel[fqn] = parallel
    _assign_parameters(result, entries, model)
    return result


def _validate_parallel_nesting(entries):
    active = [(fqn, parallel.binding_kind) for fqn, _, _, parallel in entries.values()
              if parallel is not None and parallel.binding_kind != "none"]
    for fqn, kind in active:
        for parent, parent_kind in active:
            if _is_child(fqn, parent) and (kind == parent_kind or "mesh" in (kind, parent_kind)):
                raise ValueError(f"overlapping communication owners at {parent!r} and {fqn!r}")


def validate_owned_layouts(model: nn.Module, plan: Any, owned: OwnedLayouts, context: Any) -> None:
    """Check ownership and member capabilities after the final explicit overrides.

    Args:
        model: Final model tree, still carrying global logical shapes.
        plan: Plan after user layout overrides.
        owned: Ownership captured before templates.
        context: Shared immutable logical layout context.
    """
    if not owned.specs:
        return
    covered = set()
    declarations = {}
    for fqn, spec in plan.modules.items():
        module = model.get_submodule(fqn)
        parameters = dict(module.named_parameters(remove_duplicate=False))
        for name, placement in spec.params.items():
            parameter = parameters.get(name)
            if parameter is None:
                continue
            identity = id(parameter)
            if identity in declarations and declarations[identity] != placement:
                raise ValueError(f"explicit overrides give shared parameter {_full(fqn, name)!r} conflicting layouts")
            declarations[identity] = placement
            covered.add(identity)
        if fqn in owned.specs:
            local_coverage = {id(parameters[name]) for name in spec.params if name in parameters}
            if owned.required_by_fqn[fqn] - local_coverage:
                raise ValueError(f"final override removed required parameter ownership from {fqn!r}")
            _validate_ports(module, spec)
            _validate_placements(spec, context)
            parallel = owned.parallel.get(fqn)
            info = get_family_build_info(module)
            cached = info.layout if info is not None else None
            if cached is not None:
                _validate_applied_layout(cached, spec, fqn)
            elif parallel is not None:
                validate_parallel_module(module, parallel, spec, context)
    if owned.expected_ids - covered:
        raise ValueError("final layout overrides removed required module-owned parameter coverage")


def _prepare_family_layout(module: nn.Module, mesh: Any) -> None:
    """Apply a self-described component through the existing planner and executor."""
    # Planner and apply import layout discovery; defer these cyclic runtime dependencies.
    from hyper_parallel.distributed._builder.planner import ShardingPlanner  # pylint: disable=C0415,R0401
    from hyper_parallel.distributed.apply import apply_sharding_plan  # pylint: disable=C0415,R0401

    info = get_family_build_info(module)
    if info.resolved.sharding is None:
        if info.resolved.selectors["parallel"]:
            raise ValueError("a mesh-constructed parallel family must declare sharding")
        return
    # A component's config identifies its containing model, whose providers do not apply here.
    module._hp_model_adapter_spec = None  # pylint: disable=protected-access
    planner = ShardingPlanner()
    plan = planner.plan(module, mesh.device_mesh, tp_size=mesh.tp_size, cp_size=mesh.cp_size,
                        ep_size=mesh.ep_size, sequence_parallel=mesh.sequence_parallel,
                        loss_parallel=mesh.loss_parallel)
    for fqn, spec in plan.modules.items():
        if fqn not in plan.parallel_modules:
            validate_parallel_module(module.get_submodule(fqn), ParallelModuleSpec("family.local"), spec, None)
    _record_family_layouts(module, plan, mesh, _family_parameter_shapes(module), None, applied=False)
    if callable(getattr(mesh.device_mesh, "get_group", None)):
        apply_sharding_plan(module, plan, mesh)
        if not any(parameter.is_meta for parameter in module.parameters()):
            validate_materialized_modules(module)
