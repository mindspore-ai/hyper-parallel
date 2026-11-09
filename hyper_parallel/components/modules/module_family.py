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

import hashlib
import inspect
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Any, Callable

import torch
from torch import nn

from hyper_parallel.components.modules.family import (
    BackendSpec,
    ModuleFamily,
    SelectorSpec,
    VariantRule,
    _FAMILIES,
    _validate_deferred_arguments,
    _validate_selector_value,
    get_family_selector_specs,
    get_module_family,
    register_module_family,
    resolve_module_family_target,
)


__all__ = [
    "BackendSpec", "ModuleFamily", "SelectorSpec", "VariantRule", "FamilyRequest", "ResolvedFamily",
    "FamilyBuildInfo", "FamilyBoundary", "register_module_family", "get_module_family",
    "get_family_selector_specs", "resolve_module_family_target", "resolve_family", "build_family",
    "get_family_build_info", "collect_family_modules", "validate_family_context",
    "family_plan_snapshot",
]

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
    local_variant: str
    constructor: Callable
    arguments: Mapping[str, Any]
    selectors: Mapping[str, Any]
    backend: str | None
    axis_sizes: Mapping[str, int]

    @property
    def sharding(self) -> Callable | None:
        """Return the selected family's layout provider."""
        return self.family.sharding


@dataclass(frozen=True)
class FamilyBuildInfo:
    """Immutable construction provenance; runtime bindings live outside this record."""

    request: FamilyRequest
    resolved: ResolvedFamily
    config_snapshot: Any
    origin: str


@dataclass(frozen=True)
class FamilyBoundary:
    """One concrete family instance with all its paths in the final model tree."""

    module: nn.Module
    fqns: tuple[str, ...]
    info: FamilyBuildInfo


def _symbol(path: str) -> Any:
    module_path, _, name = path.rpartition(".")
    if not module_path or not name:
        raise ValueError(f"expected a stable import path, got {path!r}")
    return getattr(import_module(module_path), name)


def _axis_sizes(context: Mapping[str, Any]) -> Mapping[str, int]:
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

def _same_mesh(first: Any, second: Any) -> bool:
    """Compare borrowed job views without copying or reconstructing their groups."""
    return (getattr(first, "device_mesh", first) is getattr(second, "device_mesh", second)
            and getattr(first, "fsdp_moe_mesh", None) is getattr(second, "fsdp_moe_mesh", None))

def _context_from_mesh(mesh: Any, context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Read actual job axes without copying mesh resources into a request."""
    result = dict(context or {})
    if mesh is None:
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


def _resolve_backend(
    family: ModuleFamily,
    local_variant: str,
    requested: str | None,
    context: Mapping[str, Any],
) -> str | None:
    backend = requested if requested is not None else family.default_backends.get(local_variant)
    if not family.backend_specs:
        if backend is not None:
            raise ValueError(f"family {family.name!r} does not declare backend {backend!r}")
        return None
    declaration = family.backend_specs.get(backend)
    if declaration is None or local_variant not in declaration.local_variants:
        raise ValueError(f"backend {backend!r} does not support local variant {local_variant!r}")
    for key, allowed in (("device_type", declaration.device_types), ("dtype", declaration.dtypes)):
        actual = context.get(key)
        actual = str(actual).removeprefix("torch.") if actual is not None else None
        if actual is not None and allowed and actual not in allowed:
            raise ValueError(f"backend {backend!r} does not support {key}={actual!r}")
    missing = set(context.get("features", ())) - declaration.features
    if missing:
        raise ValueError(f"backend {backend!r} does not support features {sorted(missing)}")
    if declaration.implementation is not None:
        _symbol(declaration.implementation)
    return backend


def resolve_family(request: FamilyRequest, *, context: Mapping[str, Any] | None = None) -> ResolvedFamily:
    """Resolve exactly one allowlisted member using the current job context.

    Args:
        request: Original family and explicit arguments.
        context: Read-only logical topology and target numerical constraints.

    Returns:
        Concrete construction information without allocating model resources.
    """
    family = get_module_family(request.target)
    context = dict(context or {})
    axes = _axis_sizes(context)
    args = deepcopy(dict(request.kwargs))
    selectors = {name: args.pop(name, spec.default) for name, spec in family.selector_specs.items()}
    for name, value in selectors.items():
        _validate_selector_value(family, name, value)
    selectors["parallel"] = family.parallel_axis is not None and axes[family.parallel_axis] > 1
    defaults = {name: spec.default for name, spec in family.selector_specs.items()}
    defaults["parallel"] = False
    if selectors == defaults:
        variant = family.default_variant
    else:
        matches = [rule.variant for rule in family.variant_rules if dict(rule.when) == selectors]
        if len(matches) != 1:
            raise ValueError(f"family {family.name!r} has no legal combination for {selectors}")
        variant = matches[0]
    local_variant = family.local_variant_of[variant]
    backend = _resolve_backend(family, local_variant, args.pop("backend", None), context)
    constructor = _symbol(family.variants[variant])
    if constructor is family.baseline:
        raise ValueError("a concrete family member cannot be its public factory")
    return ResolvedFamily(
        family, variant, local_variant, constructor, MappingProxyType(args),
        MappingProxyType(selectors), backend, axes,
    )


def _constructor_arguments(
    member: ResolvedFamily, config: Any, module: nn.Module | None, fqn: str, context: Mapping[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(member.constructor)
    required = "config" if module is None else "module"
    if required not in signature.parameters:
        raise TypeError(f"member {member.variant!r} must explicitly accept {required}")
    kwargs = dict(member.arguments)
    unknown = set(kwargs) - set(signature.parameters)
    if unknown:
        raise TypeError(f"member {member.variant!r} does not accept {sorted(unknown)}")
    runtime = {"config": config, "module": module, "module_fqn": fqn, "context": context}
    if member.backend is not None:
        runtime["backend"] = member.backend
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
        mesh: Existing job mesh context used for concrete member selection.
            Active CP requires a 'cp' axis in device_mesh. Active EP requires an 'ep' axis
            in fsdp_moe_mesh, or device_mesh when no expert view is supplied.
        module: Existing source module; mutually exclusive with config.
        module_fqn: Source path when converting an existing module.
        context: Job topology and numerical constraints; omission initially selects local execution.
        **kwargs: Explicit selectors and ordinary public factory arguments.

    Returns:
        The runnable current member with its discoverable family request and construction provenance.
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
    context = MappingProxyType(dict(_context_from_mesh(mesh, context)))
    member = resolve_family(request, context=context)
    return _build_family_member(request, member, config=config, module=module,
                                module_fqn=module_fqn, context=context)


def get_family_build_info(module: nn.Module) -> FamilyBuildInfo | None:
    """Return only framework-created family provenance, or None for ordinary modules.

    Args:
        module: Module whose construction provenance is queried.
    """
    value = getattr(module, "_hp_family_build_info", None)
    return value if isinstance(value, FamilyBuildInfo) else None


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
    axes = _axis_sizes(context)
    for boundary in collect_family_modules(model):
        member = boundary.info.resolved
        for axis in ("cp", "ep"):
            if member.axis_sizes[axis] != axes[axis]:
                raise ValueError(
                    f"family at {boundary.fqns} was constructed with {axis}={member.axis_sizes[axis]}, "
                    f"job requires {axes[axis]}; call prepare_model_components with the job context before layout"
                )
        _resolve_backend(member.family, member.local_variant, member.backend, context)


def _plain_callable(value: Callable) -> dict[str, str]:
    """Serialize only an importable callable with stable implementation identity."""
    name = getattr(value, "__qualname__", None)
    if not isinstance(name, str):
        raise ValueError("codegen requires stable importable callables")
    path = f"{value.__module__}.{name}"
    if "<" in path or _symbol(path) is not value:
        raise ValueError("codegen requires stable importable callables")
    return {"callable": path, "source_hash": _code_hash(value)}


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (frozenset, set)):
        return [_plain(item) for item in sorted(value, key=str)]
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if callable(value):
        return _plain_callable(value)
    raise ValueError(f"cannot serialize family value of type {type(value).__name__}")


def family_plan_snapshot(model: nn.Module) -> list[dict[str, Any]]:
    """Describe effective members for cache validation and checkpoint provenance.

    Args:
        model: Final or source model tree for this build.
    """
    records = []
    for boundary in collect_family_modules(model):
        info = boundary.info
        member = info.resolved
        records.append({
            "fqns": list(boundary.fqns), "family": member.family.name, "version": member.family.version,
            "origin": info.origin, "variant": member.variant,
            "local_variant": member.local_variant, "backend": member.backend,
            "selectors": dict(member.selectors), "axis_sizes": dict(member.axis_sizes),
            "config": _plain(info.config_snapshot), "kwargs": _plain(info.request.kwargs),
            "constructor": _plain(member.constructor), "sharding": _plain(member.sharding),
            "backend_specs": _plain(member.family.backend_specs),
            "default_backends": dict(member.family.default_backends),
        })
    return records


def _code_hash(value: Callable) -> str:
    """Include inherited algorithms while requiring external kernels to declare versions."""
    definitions = value.__mro__ if isinstance(value, type) else (value,)
    sources = []
    for definition in definitions:
        if definition is object or getattr(definition, "__module__", "").startswith("torch."):
            continue
        try:
            sources.append(inspect.getsource(definition))
        except (OSError, TypeError):
            sources.append(str(getattr(definition, "__qualname__", "")))
    return hashlib.sha256("\n".join(sources).encode()).hexdigest()
