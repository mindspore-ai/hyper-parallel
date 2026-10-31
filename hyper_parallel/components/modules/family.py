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


@dataclass(frozen=True)
class VariantRule:
    """A complete named selector combination and its declared variant name."""

    when: Mapping[str, Any]
    variant: str


@dataclass(frozen=True)
class ModuleFamily:
    """Construction-time variants and declared parallel adapter bindings.

    Parallel declarations name existing wrapper/factory contracts. They do not
    implement execution or choose a backend.
    """

    name: str
    baseline: Callable
    variants: Mapping[str, str]
    selector_specs: Mapping[str, SelectorSpec] = field(default_factory=dict)
    variant_rules: tuple[VariantRule, ...] = ()
    parallel_adapters: Mapping[str, str] = field(default_factory=dict)


# Keep the baseline in the value so its identity cannot be reused while registered.
_FAMILIES: dict[int, ModuleFamily] = {}
_SELECTOR_TYPES = (bool, int, float, str)
_RESERVED_SELECTORS = frozenset({"_target_", "target_path", "module", "module_fqn", "context"})


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
    for name, spec in family.selector_specs.items():
        if not isinstance(name, str) or not name.isidentifier() or name in _RESERVED_SELECTORS:
            raise ValueError(f"ModuleFamily {family.name}: invalid selector name {name!r}")
        if not isinstance(spec, SelectorSpec) or spec.value_type not in _SELECTOR_TYPES:
            raise TypeError(f"ModuleFamily {family.name}: {name!r} requires a scalar SelectorSpec")
        _validate_selector_value(family, name, spec.default)
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
    conditions = [defaults]  # The default combination is an explicit baseline allowance.
    for rule in family.variant_rules:
        if not isinstance(rule, VariantRule) or not isinstance(rule.when, Mapping):
            raise TypeError(f"ModuleFamily {family.name}: expected VariantRule with a mapping condition")
        if set(rule.when) != set(defaults):
            raise ValueError(f"ModuleFamily {family.name}: each rule must contain exactly {sorted(defaults)}")
        for name, value in rule.when.items():
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
        family: Selector schema, variant rules and lazy parallel adapter declarations.
    """
    if not isinstance(family.name, str) or not family.name.strip():
        raise ValueError("ModuleFamily name must be a non-empty string")
    if not callable(family.baseline):
        raise TypeError(f"ModuleFamily {family.name}: baseline must be callable")
    if id(family.baseline) in _FAMILIES:
        raise ValueError(f"Module family already registered: {family.name}")
    _validate_family_rules(family)
    if not isinstance(family.parallel_adapters, Mapping):
        raise TypeError("ModuleFamily parallel_adapters must be a mapping")
    for axis, path in family.parallel_adapters.items():
        if axis not in ("cp", "ep") or not isinstance(path, str) or "." not in path:
            raise ValueError(f"ModuleFamily {family.name}: invalid parallel adapter {axis!r}: {path!r}")
        if not all(part.isidentifier() for part in path.split(".")):
            raise ValueError(f"ModuleFamily {family.name}: invalid parallel adapter path {path!r}")
    snapshot = replace(
        family,
        variants=MappingProxyType(dict(family.variants)),
        parallel_adapters=MappingProxyType(dict(family.parallel_adapters)),
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


def get_family_parallel_adapter(target: Callable, axis: str) -> Optional[Callable]:
    """Resolve a registered concrete type's adapter only when it is requested.

    Args:
        target: Final module type; variants do not inherit baseline adapters.
        axis: Parallel execution contract, ``cp`` or ``ep``.

    Returns:
        Existing wrapper/factory callable, or None if the type has no binding.
        Import failures propagate; no alternative implementation is selected.
    """
    if axis not in ("cp", "ep"):
        raise ValueError(f"Unknown parallel axis {axis!r}; expected cp or ep")
    family = get_module_family(target)
    path = family.parallel_adapters.get(axis) if family is not None else None
    if path is None:
        return None
    module_path, _, name = path.rpartition(".")
    adapter = getattr(import_module(module_path), name)
    if not callable(adapter):
        raise TypeError(f"ModuleFamily {family.name}: parallel adapter {path!r} is not callable")
    return adapter


def get_family_selector_specs(target: Callable) -> Mapping[str, SelectorSpec]:
    """Return the selector schema for the requested baseline.

    Args:
        target: Requested replacement callable.

    Returns:
        Read-only registered declarations, or an empty mapping.
    """
    family = get_module_family(target)
    return family.selector_specs if family is not None else {}


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
    args = dict(arguments)
    family = get_module_family(target)
    if family is None:
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
