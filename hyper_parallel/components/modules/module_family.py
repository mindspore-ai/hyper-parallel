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
"""Shared construction of concrete family members for direct and source models."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable

from torch import nn

from hyper_parallel.components.modules.family import (
    ModuleFamily,
    SelectorSpec,
    VariantRule,
    _FAMILIES,
    _validate_deferred_arguments,
    _select_variant,
    get_family_selector_specs,
    get_module_family,
    register_module_family,
    resolve_module_family_target,
)


__all__ = [
    "ModuleFamily", "SelectorSpec", "VariantRule", "FamilyRequest", "ResolvedFamily",
    "FamilyBuildInfo", "FamilyBoundary", "register_module_family", "get_module_family",
    "get_family_selector_specs", "resolve_module_family_target", "resolve_family", "build_family",
    "get_family_build_info", "collect_family_modules", "validate_family_context",
    "FamilyShardingContext",
]


@dataclass(frozen=True)
class FamilyShardingContext:
    """Logical computation axes and global shapes; no communication resources."""

    mesh_dim_names: tuple[str, ...]
    axis_sizes: Mapping[str, int]
    parameter_shapes: Mapping[str, tuple[int, ...]]
    sequence_parallel: bool = False
    loss_parallel: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_sizes", MappingProxyType(dict(self.axis_sizes)))
        object.__setattr__(self, "parameter_shapes", MappingProxyType(dict(self.parameter_shapes)))


@dataclass(frozen=True)
class FamilyRequest:
    """A public family and its explicit construction arguments, before selection."""

    target: Callable
    kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        family = get_module_family(self.target)
        if family is None or family.default_variant is None:
            raise ValueError("FamilyRequest requires a family with a concrete default_variant")
        _validate_deferred_arguments(family, self.kwargs)
        object.__setattr__(self, "kwargs", MappingProxyType(deepcopy(dict(self.kwargs))))


@dataclass(frozen=True)
class ResolvedFamily:
    """One validated implementation choice without parameters or communication resources."""

    family: ModuleFamily
    variant: str
    constructor: Callable
    arguments: Mapping[str, Any]
    selectors: Mapping[str, Any]
    axis_sizes: Mapping[str, int]

    @property
    def sharding(self) -> Callable | None:
        """Return the selected family's layout provider."""
        return self.family.sharding


@dataclass(frozen=True)
class _FamilyLayout:
    """Borrowed mesh and completed component layout, outside serialized requests."""

    mesh: Any
    sharding: Any
    parallel: Any
    parameter_shapes: Mapping[str, tuple[int, ...]]
    source_shard_info: Mapping[str, Any]
    applied: bool = True


@dataclass(frozen=True)
class FamilyBuildInfo:
    """Construction provenance and layout records; execution resources stay on the module."""

    request: FamilyRequest
    resolved: ResolvedFamily
    config_snapshot: Any
    origin: str
    layout: _FamilyLayout | None = None


@dataclass(frozen=True)
class FamilyBoundary:
    """One concrete family instance with all its paths in the final model tree."""

    module: nn.Module
    fqns: tuple[str, ...]
    info: FamilyBuildInfo


def _axis_sizes(context: Mapping[str, Any]) -> Mapping[str, int]:
    """Read and validate logical parallel degrees from the construction context."""
    result = {}
    for axis in ("tp", "cp", "ep", "pp"):
        size = context.get(f"{axis}_size")
        if size is None:
            if context.get(axis):
                raise ValueError(f"{axis}_size is required when {axis} is active")
            size = 1
        actual_type = type(size)
        if actual_type is not int or size < 1:
            raise ValueError(f"{axis}_size must be a positive integer")
        result[axis] = size
    return MappingProxyType(result)


def _mesh_axis_size(mesh: Any, axis: str) -> int:
    """Validate declared sizes against the actual borrowed axis views."""
    device_mesh = mesh.device_mesh
    region = getattr(mesh, "fsdp_moe_mesh", None) if axis == "ep" else None
    if region is None:
        region = device_mesh
    names = tuple(getattr(region, "mesh_dim_names", ()) or ())
    actual = region[axis].size() if axis in names else 1
    size = getattr(mesh, f"{axis}_size", actual)
    actual_type = type(size)
    if actual_type is not int or size < 1:
        raise ValueError(f"{axis}_size must be a positive integer")
    if axis in names and size != actual:
        raise ValueError(f"{axis}_size={size} disagrees with the mesh size {actual}")
    if axis not in names and size > 1 and axis != "pp":
        raise ValueError(f"{axis}_size={size} requires an axis named {axis!r}; available mesh axes: {names!r}")
    return size


def _context_from_mesh(mesh: Any, context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Read actual job axes without copying mesh resources into a request."""
    result = dict(context or {})
    if mesh is None:
        mesh = result.get("mesh")
        if getattr(mesh, "device_mesh", None) is None:
            return result
    device_mesh = getattr(mesh, "device_mesh", None)
    if device_mesh is None:
        raise ValueError("mesh requires a prebuilt device_mesh")
    previous = result.get("mesh")
    if previous is not None and not _same_mesh(previous, mesh):
        raise ValueError("context and mesh belong to different jobs")
    for axis in ("tp", "cp", "ep", "pp"):
        size = _mesh_axis_size(mesh, axis)
        if f"{axis}_size" in result and result[f"{axis}_size"] != size:
            raise ValueError(f"context and mesh disagree on {axis}_size")
        result[f"{axis}_size"] = size
        result[axis] = size > 1
    result["mesh"] = mesh
    result["sequence_parallel"] = bool(getattr(mesh, "sequence_parallel", False))
    result["loss_parallel"] = bool(getattr(mesh, "loss_parallel", False))
    return result


def resolve_family(request: FamilyRequest, *, context: Mapping[str, Any] | None = None) -> ResolvedFamily:
    """Resolve exactly one allowlisted member using the current job context.

    Args:
        request: Original family and explicit arguments.
        context: Read-only job topology used for automatic CP/EP selection.

    Returns:
        Concrete construction information without allocating model resources.
    """
    family = get_module_family(request.target)
    context = context or {}
    axes = _axis_sizes(context)
    args = deepcopy(dict(request.kwargs))
    parallel = family.parallel_axis is not None and axes[family.parallel_axis] > 1
    constructor, variant, selectors = _select_variant(family, args, parallel=parallel)
    if constructor is family.baseline:
        raise ValueError("a concrete family member cannot be its public factory")
    return ResolvedFamily(
        family, variant, constructor, MappingProxyType(args), MappingProxyType(selectors), axes,
    )


def _constructor_arguments(
    member: ResolvedFamily, config: Any, module: nn.Module | None, fqn: str, context: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind source or config construction arguments accepted by the selected member."""
    signature = inspect.signature(member.constructor)
    required = "config" if module is None else "module"
    if required not in signature.parameters:
        raise TypeError(f"member {member.variant!r} must explicitly accept {required}")
    kwargs = dict(member.arguments)
    unknown = set(kwargs) - set(signature.parameters)
    if unknown:
        raise TypeError(f"member {member.variant!r} does not accept {sorted(unknown)}")
    runtime = {"config": config, "module": module, "module_fqn": fqn, "context": context}
    kwargs.update({name: value for name, value in runtime.items() if name in signature.parameters})
    signature.bind(**kwargs)
    return kwargs


def _normalize_family_request(target, kwargs):
    """Resolve a public name without selecting a topology-dependent member."""
    if isinstance(target, str):
        matches = [family.baseline for family in _FAMILIES.values() if family.name == target]
        if len(matches) != 1:
            raise ValueError(f"family name {target!r} must identify exactly one registration")
        target = matches[0]
    if isinstance(target, FamilyRequest):
        if kwargs:
            raise ValueError("request arguments cannot be combined with additional keywords")
        return target
    return FamilyRequest(target, kwargs)


def _build_family_member(
    request: FamilyRequest,
    member: ResolvedFamily,
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Mapping[str, Any],
) -> nn.Module:
    """Construct an already resolved choice without parsing the same request again."""
    context = MappingProxyType(dict(context))
    if member.family.validate is not None:
        member.family.validate(config=config, module=module, context=context, member=member)
    config = deepcopy(config)
    result = member.constructor(**_constructor_arguments(member, config, module, module_fqn, context))
    if not isinstance(result, nn.Module):
        raise TypeError("a family constructor must return nn.Module")
    info = FamilyBuildInfo(request, member, deepcopy(config), "direct" if module is None else "replacement")
    result._hp_family_build_info = info  # pylint: disable=protected-access
    if config is not None and callable(getattr(result, "reset_parameters", None)):
        result._hp_reset_after_materialization = True  # pylint: disable=protected-access
    return result


def build_family(
    target: Callable | str | FamilyRequest,
    *,
    config: Any = None,
    mesh: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Construct a selected member and retain its request for model preparation.

    Args:
        target: Public family callable, registered name, or preserved request.
        config: Complete logical model configuration for direct construction.
        mesh: Prebuilt job mesh context; select, shard and bind this component when supplied.
            Active CP requires a 'cp' axis in device_mesh. Active EP requires an 'ep' axis
            in fsdp_moe_mesh, or device_mesh when no expert view is supplied.
        module: Existing source module; mutually exclusive with config.
        module_fqn: Source path when converting an existing module.
        context: Optional job constraints; without mesh, may supply logical topology for selection.
        **kwargs: Explicit selectors and ordinary public factory arguments.

    Returns:
        The concrete member with construction provenance. Logical/meta builds finish physical work at runtime.
    """
    request = _normalize_family_request(target, kwargs)
    arguments = dict(request.kwargs)
    if "config" in arguments:
        if config is not None:
            raise ValueError("config was supplied twice")
        config = arguments.pop("config")
        request = FamilyRequest(request.target, arguments)
    if (config is None) == (module is None):
        raise ValueError("provide exactly one of config or module")
    context = _context_from_mesh(mesh, context)
    member = resolve_family(request, context=context)
    result = _build_family_member(request, member, config=config, module=module,
                                 module_fqn=module_fqn, context=context)
    if mesh is not None:
        # Layout discovery imports this module; defer its runtime dependency to avoid a cycle.
        from hyper_parallel.distributed._builder.family_layout import _prepare_family_layout  # pylint: disable=C0415,R0401
        _prepare_family_layout(result, mesh)
    return result


def get_family_build_info(module: nn.Module) -> FamilyBuildInfo | None:
    """Return only framework-created family provenance, or None for ordinary modules.

    Args:
        module: Module whose construction provenance is queried.
    """
    value = getattr(module, "_hp_family_build_info", None)
    return value if isinstance(value, FamilyBuildInfo) else None


def _family_parameter_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    """Return logical parameter shapes even after component construction localized weights."""
    by_identity = {}
    for boundary in collect_family_modules(model):
        layout = boundary.info.layout
        if layout is not None:
            for name, shape in layout.parameter_shapes.items():
                identity = id(boundary.module.get_parameter(name))
                if by_identity.setdefault(identity, shape) != shape:
                    raise ValueError(f"shared family parameter {name!r} has conflicting logical shapes")
    return {name: by_identity.get(id(value), tuple(value.shape))
            for name, value in model.named_parameters(remove_duplicate=False)}


def _module_aliases(model: nn.Module) -> dict[int, tuple[nn.Module, tuple[str, ...]]]:
    """Inventory registered module identities and all paths, including the root."""
    found: dict[int, tuple[nn.Module, list[str]]] = {}
    for fqn, module in model.named_modules(remove_duplicate=False):
        found.setdefault(id(module), (module, []))[1].append(fqn)
    return {identity: (module, tuple(paths)) for identity, (module, paths) in found.items()}


def collect_family_modules(model: nn.Module) -> tuple[FamilyBoundary, ...]:
    """Discover final family boundaries by object identity, including root and aliases.

    Args:
        model: Final or source model tree for this build.
    """
    boundaries = []
    for module, paths in _module_aliases(model).values():
        info = get_family_build_info(module)
        if info is not None:
            boundaries.append(FamilyBoundary(module, paths, info))
    return tuple(boundaries)


def validate_family_context(model: nn.Module, context: Mapping[str, Any]) -> None:
    """Validate that model preparation has finalized members for the job context.

    Args:
        model: Final or source model tree for this build.
        context: Read-only job context and effective model facts.
    """
    axes = _axis_sizes(_context_from_mesh(None, context))
    for boundary in collect_family_modules(model):
        if boundary.info.layout is not None and boundary.info.layout.applied:
            _context_from_mesh(boundary.info.layout.mesh, context)
        member = boundary.info.resolved
        for axis in ("cp", "ep"):
            if member.axis_sizes[axis] != axes[axis]:
                raise ValueError(
                    f"family at {boundary.fqns} was constructed with {axis}={member.axis_sizes[axis]}, "
                    f"job requires {axes[axis]}; call prepare_model_components with the job context before layout"
                )


def _validate_applied_layout(cached: _FamilyLayout, spec: Any, fqn: str) -> None:
    """Reject post-construction changes to a component's declared computation layout."""
    original = cached.sharding
    if any(name not in original.params or named != original.params[name] for name, named in spec.params.items()):
        raise ValueError(f"layout at {fqn!r} conflicts with its constructed family layout")
    for field_name in ("in_src", "in_dst", "out_src", "out_dst", "out_names", "is_boundary"):
        if getattr(spec, field_name) != getattr(original, field_name):
            raise ValueError(f"{field_name} at {fqn!r} conflicts with its constructed family layout")


def _record_family_layouts(model, plan, mesh, shapes, source_shard_info, *, applied=True):
    """Retain relative declarations once for constructor and model-level application."""
    for boundary in collect_family_modules(model):
        fqn = boundary.fqns[0]
        if fqn not in plan.modules or boundary.info.resolved.sharding is None:
            continue
        existing = boundary.info.layout
        if existing is not None and existing.applied:
            continue
        prefix = fqn + "." if fqn else ""
        names = plan.modules[fqn].params
        relative_shapes = {name: shapes[prefix + name] for name in names}
        source = {name: source_shard_info[prefix + name] for name in names
                  if source_shard_info is not None and prefix + name in source_shard_info}
        layout = _FamilyLayout(mesh, deepcopy(plan.modules[fqn]), plan.parallel_modules.get(fqn),
                              MappingProxyType(relative_shapes), MappingProxyType(source), applied)
        boundary.module._hp_family_build_info = replace(boundary.info, layout=layout)  # pylint: disable=protected-access


def _family_source_shard_info(model: nn.Module) -> dict[str, Any]:
    """Gather constructed component source layouts under their final model paths."""
    result = {}
    for boundary in collect_family_modules(model):
        layout = boundary.info.layout
        if layout is not None and layout.applied:
            for fqn in boundary.fqns:
                result.update({f"{fqn}.{name}" if fqn else name: value
                               for name, value in layout.source_shard_info.items()})
    return result


def _same_mesh(first: Any, second: Any) -> bool:
    """Compare borrowed job views without copying or reconstructing their groups."""
    return (getattr(first, "device_mesh", first) is getattr(second, "device_mesh", second)
            and getattr(first, "fsdp_moe_mesh", None) is getattr(second, "fsdp_moe_mesh", None))


def _family_mesh(model: nn.Module, mesh: Any) -> Any:
    """Reuse the complete context when a caller passes its dense DeviceMesh view."""
    layouts = [boundary.info.layout for boundary in collect_family_modules(model)
               if boundary.info.layout is not None and boundary.info.layout.applied]
    if not hasattr(mesh, "device_mesh"):
        mesh = next((layout.mesh for layout in layouts if layout.mesh.device_mesh is mesh), mesh)
    if any(not _same_mesh(layout.mesh, mesh) for layout in layouts):
        raise ValueError("constructed families cannot be applied with a different job mesh")
    return mesh
