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
"""Construction-time, allowlisted module selection without device dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib import import_module
from inspect import Parameter, signature
from math import isfinite
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional


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
class BackendSpec:
    """Capabilities of one explicitly selected numerical implementation.

    Attributes:
        local_variants: Numerical variants supported by this backend.
        device_types: Accepted device types; empty means unrestricted.
        dtypes: Accepted dtype names; empty means unrestricted.
        features: Supported source/config features when explicitly requested.
        implementation: Optional import path checked only for the selected backend.
    """

    local_variants: tuple[str, ...]
    device_types: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    features: frozenset[str] = frozenset()
    implementation: str | None = None
    version: str = "1"


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
    backend_specs: Mapping[str, "BackendSpec"] = field(default_factory=dict)
    default_backends: Mapping[str, str] = field(default_factory=dict)
    local_variant_of: Mapping[str, str] = field(default_factory=dict)
    validate: Callable | None = None
    version: str = "1"


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


def _validate_backend_entries(family: ModuleFamily, local_variants: set[str]) -> None:
    """Validate static backend capabilities against concrete numerical members."""
    for name, spec in family.backend_specs.items():
        if not isinstance(name, str) or not name or not isinstance(spec, BackendSpec):
            raise TypeError("backend_specs must map non-empty names to BackendSpec")
        if not set(spec.local_variants) <= local_variants:
            raise ValueError(f"backend {name!r} declares unknown local variants")
    for variant, backend in family.default_backends.items():
        if backend not in family.backend_specs or variant not in family.backend_specs[backend].local_variants:
            raise ValueError(f"default backend {backend!r} does not support {variant!r}")


def _validate_backends(family: ModuleFamily) -> None:
    """Validate complete local-variant mappings without importing implementations."""
    if family.default_variant is None:
        if family.backend_specs or family.default_backends or family.local_variant_of:
            raise ValueError("backend declarations require default_variant")
        return
    if set(family.local_variant_of) != set(family.variants):
        raise ValueError("local_variant_of must cover every concrete variant")
    local_variants = set(family.local_variant_of.values())
    if not local_variants <= set(family.variants):
        raise ValueError("local_variant_of must reference concrete numerical variants")
    if not isinstance(family.version, str) or not family.version:
        raise ValueError("family version must be a non-empty string")
    if family.backend_specs and set(family.default_backends) != local_variants:
        raise ValueError("default_backends must cover every local variant")
    _validate_backend_entries(family, local_variants)


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
    _validate_backends(family)
    snapshot = replace(
        family,
        variants=MappingProxyType(dict(family.variants)),
        selector_specs=MappingProxyType(dict(family.selector_specs)),
        backend_specs=MappingProxyType(dict(family.backend_specs)),
        default_backends=MappingProxyType(dict(family.default_backends)),
        local_variant_of=MappingProxyType(dict(family.local_variant_of)),
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


def _select_variant(family: ModuleFamily, args: dict) -> Callable:
    """Consume selectors and accept only the baseline defaults or one full rule."""
    selectors = {name: args.pop(name, spec.default) for name, spec in family.selector_specs.items()}
    for name, value in selectors.items():
        _validate_selector_value(family, name, value)
    defaults = {name: spec.default for name, spec in family.selector_specs.items()}
    if selectors == defaults:
        return family.baseline
    matches = [rule for rule in family.variant_rules if selectors == dict(rule.when)]
    if len(matches) != 1:
        legal = [defaults, *(dict(rule.when) for rule in family.variant_rules)]
        raise ValueError(
            f"ModuleFamily {family.name}: unsupported selector combination {selectors}; " f"legal combinations: {legal}"
        )
    module_path, _, name = family.variants[matches[0].variant].rpartition(".")
    return getattr(import_module(module_path), name)


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
    target = _select_variant(family, args)
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
    for name, spec in family.selector_specs.items():
        if name in args:
            _validate_selector_value(family, name, args[name])
    if "backend" in args and args["backend"] is not None:
        if not isinstance(args["backend"], str) or args["backend"] not in family.backend_specs:
            raise ValueError(f"ModuleFamily {family.name}: unknown backend {args['backend']!r}")
