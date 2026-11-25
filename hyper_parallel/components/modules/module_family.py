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
"""Construction-time module-family registry, allowlisted selection, and member builder."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from importlib import import_module
from inspect import Parameter, signature
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from torch import nn


__all__ = [
    "ModuleFamily",
    "SelectorSpec",
    "VariantRule",
    "FamilyRequest",
    "ResolvedFamily",
    "FamilyBuildInfo",
    "FamilyBoundary",
    "register_module_family",
    "get_module_family",
    "get_family_selector_specs",
    "normalize_family_selectors",
    "resolve_module_family_target",
    "resolve_family",
    "build_family",
    "get_family_build_info",
    "collect_family_modules",
    "validate_family_context",
    "FamilyShardingContext",
]


@dataclass(frozen=True)
class SelectorSpec:
    """Exact scalar type and baseline default for one selection parameter."""

    value_type: type
    default: Any
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class VariantRule:
    """A complete named selector combination and its declared variant name."""

    when: Mapping[str, Any]
    variant: str


@dataclass(frozen=True)
class ModuleFamily:
    """One baseline and explicitly allowed construction-time alternatives."""

    name: str
    baseline: Callable
    variants: Mapping[str, str]
    selector_specs: Mapping[str, SelectorSpec] = field(default_factory=dict)
    variant_rules: tuple[VariantRule, ...] = ()
    default_variant: str | None = None
    parallel_axis: str | None = None
    sharding: Callable | None = None
    validate: Callable | None = None
    # Optional declaration for members whose math has no distributed protocol methods.
    parallel_spec: Callable | None = None


# Keep the baseline in the value so its identity cannot be reused while registered.
_FAMILIES: dict[int, ModuleFamily] = {}
_SELECTOR_TYPES = (bool, int, float, str)
_RESERVED_SELECTORS = frozenset({"_target_", "target_path", "module", "module_fqn", "context", "mesh"})


def _validate_selector_value(family: ModuleFamily, name: str, value: Any) -> None:
    """Validate exact types; non-finite floats cannot define stable rules."""
    expected = family.selector_specs[name].value_type
    actual_type = type(value)
    if actual_type is not expected:  # bool must not also match int.
        raise TypeError(
            f"ModuleFamily {family.name}: selector {name!r} must be {expected.__name__}, "
            f"got {type(value).__name__}: {value!r}"
        )
    if expected is float and not isfinite(value):
        raise ValueError(f"ModuleFamily {family.name}: selector {name!r} must be finite")


def _validate_selector_specs(family: ModuleFamily) -> dict[str, Any]:
    """Validate scalar declarations and return the baseline selector values."""
    defaults = {}
    names = set(family.selector_specs)
    for name, spec in family.selector_specs.items():
        if not isinstance(name, str) or not name.isidentifier() or name in _RESERVED_SELECTORS:
            raise ValueError(f"ModuleFamily {family.name}: invalid selector name {name!r}")
        if not isinstance(spec, SelectorSpec) or spec.value_type not in _SELECTOR_TYPES:
            raise TypeError(f"ModuleFamily {family.name}: {name!r} requires a scalar SelectorSpec")
        _validate_selector_value(family, name, spec.default)
        for alias in spec.aliases:
            if (not isinstance(alias, str) or not alias.isidentifier()
                    or alias in names or alias in _RESERVED_SELECTORS):
                raise ValueError(f"ModuleFamily {family.name}: invalid selector alias {alias!r}")
            names.add(alias)
        defaults[name] = spec.default
    return defaults


def _validate_variant_paths(family: ModuleFamily) -> None:
    """Check lazy import declarations without loading device implementations."""
    for name, path in family.variants.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"ModuleFamily {family.name}: variant names must be non-empty strings")
        if not isinstance(path, str) or "." not in path:
            raise ValueError(f"ModuleFamily {family.name}: invalid variant import path {path!r}")
        if not all(part.isidentifier() for part in path.split(".")):
            raise ValueError(f"ModuleFamily {family.name}: invalid variant import path {path!r}")


def _validate_family_rules(family: ModuleFamily) -> None:
    """Check complete, unique whitelist entries before registration."""
    defaults = _validate_selector_specs(family)
    _validate_variant_paths(family)
    if family.parallel_axis not in (None, "cp", "ep"):
        raise ValueError("parallel_axis must be cp, ep or None")
    if family.default_variant is not None:
        if family.default_variant not in family.variants:
            raise ValueError("default_variant must name a declared concrete variant")
        if "parallel" in family.selector_specs:
            raise ValueError("parallel is derived from context, not a public selector")
        defaults = {**defaults, "parallel": False}
    conditions = [defaults]  # The default combination is an explicit baseline allowance.
    for rule in family.variant_rules:
        if not isinstance(rule, VariantRule) or not isinstance(rule.when, Mapping):
            raise TypeError(f"ModuleFamily {family.name}: expected VariantRule with a mapping condition")
        if set(rule.when) != set(defaults):
            raise ValueError(f"ModuleFamily {family.name}: each rule must contain exactly {sorted(defaults)}")
        for name, value in rule.when.items():
            if name == "parallel" and family.default_variant is not None:
                actual_type = type(value)
                if actual_type is not bool:
                    raise TypeError("derived parallel conditions must be bool")
            else:
                _validate_selector_value(family, name, value)
        if rule.variant not in family.variants:
            raise ValueError(f"ModuleFamily {family.name}: undeclared variant {rule.variant!r}")
        condition = dict(rule.when)
        if condition in conditions:
            raise ValueError(f"ModuleFamily {family.name}: duplicate or baseline-default rule {condition!r}")
        conditions.append(condition)


def register_module_family(family: ModuleFamily) -> None:
    """Register a checked snapshot for an exact baseline callable.

    Args:
        family: Scalar selector schema, complete rules and lazy variant paths.
    """
    if not isinstance(family.name, str) or not family.name.strip():
        raise ValueError("ModuleFamily name must be a non-empty string")
    if not callable(family.baseline):
        raise TypeError(f"ModuleFamily {family.name}: baseline must be callable")
    if id(family.baseline) in _FAMILIES:
        raise ValueError(f"Module family already registered: {family.name}")
    _validate_family_rules(family)
    if family.parallel_spec is not None and not callable(family.parallel_spec):
        raise TypeError("ModuleFamily parallel_spec must be callable")
    snapshot = replace(
        family,
        variants=MappingProxyType(dict(family.variants)),
        selector_specs=MappingProxyType(dict(family.selector_specs)),
        variant_rules=tuple(
            VariantRule(MappingProxyType(dict(rule.when)), rule.variant) for rule in family.variant_rules
        ),
    )
    _FAMILIES[id(family.baseline)] = snapshot


def get_module_family(target: Callable) -> Optional[ModuleFamily]:
    """Return the declaration for an exact baseline object.

    Args:
        target: Requested callable; class subclasses require their own registration.

    Returns:
        Registered declaration, or None for an ordinary callable.
    """
    return _FAMILIES.get(id(target))


def get_family_selector_specs(target: Callable) -> Mapping[str, SelectorSpec]:
    """Return the selector schema for the requested baseline.

    Args:
        target: Requested replacement callable.

    Returns:
        Read-only declarations including aliases, or an empty mapping.
    """
    family = get_module_family(target)
    if family is None:
        return {}
    return MappingProxyType({
        spelling: spec for name, spec in family.selector_specs.items()
        for spelling in (name, *spec.aliases)
    })


def normalize_family_selectors(target: Callable, arguments: Mapping) -> dict:
    """Replace declared aliases with canonical selector names.

    Args:
        target: Requested replacement callable.
        arguments: Explicit selectors and constructor arguments.

    Returns:
        New arguments with aliases consumed; conflicting spellings are rejected.
    """
    args = dict(arguments)
    family = get_module_family(target)
    if family is None:
        return args
    for name, spec in family.selector_specs.items():
        for alias in spec.aliases:
            if alias not in args:
                continue
            value = args.pop(alias)
            _validate_selector_value(family, name, value)
            if name in args and (type(args[name]) is not type(value) or args[name] != value):
                raise ValueError(f"ModuleFamily {family.name}: conflicting selectors {name!r} and {alias!r}")
            args[name] = value
    return args


def _select_variant(
    family: ModuleFamily, args: dict, *, parallel: bool = False,
) -> tuple[Callable, str | None, dict[str, Any]]:
    """Consume target selectors and resolve one concrete constructor for both entry paths."""
    selectors = {name: args.pop(name, spec.default) for name, spec in family.selector_specs.items()}
    for name, value in selectors.items():
        _validate_selector_value(family, name, value)
    defaults = {name: spec.default for name, spec in family.selector_specs.items()}
    if family.default_variant is not None:
        selectors["parallel"] = parallel
        defaults["parallel"] = False
    if selectors == defaults:
        variant = family.default_variant
    else:
        matches = [rule.variant for rule in family.variant_rules if selectors == dict(rule.when)]
        if len(matches) != 1:
            legal = [defaults, *(dict(rule.when) for rule in family.variant_rules)]
            raise ValueError(
                f"ModuleFamily {family.name}: unsupported selector combination {selectors}; legal combinations: {legal}"
            )
        variant = matches[0]
    constructor = family.baseline
    if variant is not None:
        module_path, _, name = family.variants[variant].rpartition(".")
        constructor = getattr(import_module(module_path), name)
    return constructor, variant, selectors


def resolve_module_family_target(target: Callable, arguments: Mapping) -> tuple[Callable, dict]:
    """Select a registered implementation and validate its ordinary keywords.

    Args:
        target: Registered baseline or an ordinary replacement callable.
        arguments: Explicit selector values and constructor arguments.

    Returns:
        Selected callable and a new argument dictionary without selectors.
        Ordinary targets retain their existing argument-validation policy.
    """
    args = normalize_family_selectors(target, arguments)
    family = get_module_family(target)
    if family is None:
        return target, args
    if family.default_variant is not None:
        _validate_deferred_arguments(family, args)
        return target, args
    target, _, _ = _select_variant(family, args)
    target_signature = signature(target)
    allowed = {
        name
        for name, parameter in target_signature.parameters.items()
        if parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    unknown = set(args) - allowed
    if unknown:
        raise TypeError(f"ModuleFamily {family.name}: unknown replacement arguments: {sorted(unknown)}")
    target_signature.bind_partial(**args)
    return target, args


def _validate_deferred_arguments(family: ModuleFamily, args: Mapping[str, Any]) -> None:
    """Validate a new-family request without choosing a topology-dependent member."""
    allowed = set(signature(family.baseline).parameters) - {"module", "module_fqn", "context", "mesh"}
    unknown = set(args) - allowed
    if unknown:
        raise TypeError(f"ModuleFamily {family.name}: unknown replacement arguments: {sorted(unknown)}")
    for name in family.selector_specs:
        if name in args:
            _validate_selector_value(family, name, args[name])


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
    """Project validated public options onto the selected member's explicit signature."""
    constructor_signature = inspect.signature(member.constructor)
    required = "config" if module is None else "module"
    if required not in constructor_signature.parameters:
        raise TypeError(f"member {member.variant!r} must explicitly accept {required}")
    # FamilyRequest checks public names; family validation sees every option before this point.
    # Members need not declare options used only by another implementation in the family.
    kwargs = {name: value for name, value in member.arguments.items() if name in constructor_signature.parameters}
    runtime = {"config": config, "module": module, "module_fqn": fqn, "context": context}
    kwargs.update({name: value for name, value in runtime.items() if name in constructor_signature.parameters})
    constructor_signature.bind(**kwargs)
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


def _validate_initialization_contract(module: nn.Module) -> bool:
    """Require replacement-owned reset hooks to declare checkpoint conversion too.

    Torch's inherited default reset is not a replacement initialization policy.
    An empty conversion list still declares that policy and permits a later reset.
    Conversion-only modules may rely on their ordinary children for initialization.
    """
    make_transforms = getattr(module, "make_transforms", None)
    if make_transforms is not None and not callable(make_transforms):
        raise TypeError("replacement make_transforms must be callable")
    reset_parameters = getattr(module, "reset_parameters", None)
    reset_owner = next((base for base in type(module).__mro__ if "reset_parameters" in vars(base)), None)
    custom_reset = (callable(reset_parameters) and reset_owner is not None
                    and not reset_owner.__module__.startswith("torch.nn."))
    if custom_reset and not callable(make_transforms):
        raise TypeError(
            f"{type(module).__name__}.reset_parameters requires a callable make_transforms(); "
            "return [] when checkpoint conversion is unnecessary"
        )
    return callable(make_transforms) and callable(reset_parameters)


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
    # Registry and selector validation remain usable without loading the tensor runtime.
    from torch import nn as torch_nn  # pylint: disable=import-outside-toplevel

    context = MappingProxyType(dict(context))
    if member.family.validate is not None:
        member.family.validate(config=config, module=module, context=context, member=member)
    config = deepcopy(config)
    result = member.constructor(**_constructor_arguments(member, config, module, module_fqn, context))
    if not isinstance(result, torch_nn.Module):
        raise TypeError("a family constructor must return nn.Module")
    requires_reset = _validate_initialization_contract(result)
    info = FamilyBuildInfo(request, member, deepcopy(config), "direct" if module is None else "replacement")
    result._hp_family_build_info = info  # pylint: disable=protected-access
    if config is not None and requires_reset:
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

    Note:
        The family validator receives all public options. Each concrete constructor receives
        only the options it explicitly declares; unknown public argument names still raise.
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
