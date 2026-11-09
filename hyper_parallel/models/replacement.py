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
"""Compile and incrementally apply structure-preserving module replacements."""

from __future__ import annotations

import fnmatch
import inspect
import logging
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Iterable, Optional

from torch import nn

from hyper_parallel.components.modules.module_family import (
    FamilyRequest, _module_aliases, build_family, get_family_build_info, get_module_family,
)

from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
    get_model_conversion_mapping,
)


ModuleReplacementFactory = Callable[..., nn.Module]
MethodReplacementFactory = Callable[..., Callable[..., Any]]
logger = logging.getLogger(__name__)


class ModuleReplacementMatchError(ValueError):
    """A required replacement pattern did not match the final model tree."""

    def __init__(self, unmatched: tuple[dict[str, str], ...]) -> None:
        """Initialize the error with unmatched replacement declarations."""
        self.unmatched = unmatched
        patterns = [item["pattern"] for item in unmatched]
        message = ("matched no compatible module" if any(
            item["reason"] == "matched_no_compatible_module" for item in unmatched
        ) else "matched no final module")
        super().__init__(
            "module replacement pattern(s) matched no final module: "
            f"{patterns} ({message}). Inspect the finalized module inventory and update the "
            "adapter pattern; do not silently skip a required replacement."
        )


def module_replacement(factory: ModuleReplacementFactory) -> ModuleReplacementFactory:
    """Declare a factory compatible with the generic replacement executor.

    Args:
        factory: Construction callable or protocol factory.
    """

    signature = inspect.signature(factory)
    required = {"module", "module_fqn", "context"}
    missing = required - set(signature.parameters)
    if missing:
        raise TypeError(
            f"module replacement factory {factory!r} must declare {sorted(missing)}"
        )
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        raise TypeError("module replacement factories must not declare **kwargs")
    factory._hp_module_replacement = True  # type: ignore[attr-defined]  # pylint: disable=protected-access
    return factory


def method_replacement(factory: MethodReplacementFactory) -> MethodReplacementFactory:
    """Declare a factory that replaces one bound module method.

    The factory receives the owning module, the full method FQN, the original
    bound method, and a read-only build context. It returns a callable whose
    arguments are the same as calls to the original bound method.

    Args:
        factory: Construction callable or protocol factory.

    """
    signature = inspect.signature(factory)
    required = {"module", "method_fqn", "original_method", "context"}
    missing = required - set(signature.parameters)
    if missing:
        raise TypeError(f"method replacement factory {factory!r} must declare {sorted(missing)}")
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        raise TypeError("method replacement factories must not declare **kwargs")
    factory._hp_method_replacement = True  # type: ignore[attr-defined]  # pylint: disable=protected-access
    return factory


@dataclass(frozen=True)
class MethodReplacementSpec:
    """One rule selecting a named method on matching module instances."""

    match: str
    factory: MethodReplacementFactory

    def __post_init__(self) -> None:
        owner_pattern, separator, method_name = self.match.rpartition(".")
        if (not separator or not owner_pattern or not method_name.isidentifier()
                or (method_name.startswith("__") and method_name.endswith("__"))):
            raise ValueError("method replacement match must end in a literal method name")
        if method_name == "forward":
            raise ValueError("forward replacements belong to the sharding planner")
        if not callable(self.factory) or not getattr(self.factory, "_hp_method_replacement", False):
            raise TypeError("method replacement factory must use @method_replacement")


def _compile_method_replacements(model: nn.Module, specs: Iterable[MethodReplacementSpec]) -> tuple:
    """Resolve method patterns against the finalized module tree before mutation."""
    selected = {}
    aliases = _all_module_aliases(model)
    for spec in specs:
        owner_pattern, _, method_name = spec.match.rpartition(".")
        matched = False
        for module_id, (module, fqns) in aliases.items():
            matched_fqns = tuple(fqn for fqn in fqns if fnmatch.fnmatchcase(fqn, owner_pattern))
            if not matched_fqns:
                continue
            matched = True
            if (method_name in module._modules or method_name in module._parameters  # pylint: disable=protected-access
                    or method_name in module._buffers):  # pylint: disable=protected-access
                raise TypeError(f"method replacement {spec.match!r} selected a registered module or tensor")
            original = getattr(module, method_name, None)
            if not callable(original):
                raise ValueError(f"method replacement {spec.match!r} selected {matched_fqns[0]!r} without a callable "
                                 f"{method_name!r} method")
            key = (module_id, method_name)
            previous = selected.get(key)
            if previous is not None:
                if previous[3] is not spec:
                    raise ValueError(f"method replacement conflict for {matched_fqns[0]}.{method_name}")
                continue
            method_fqns = tuple(f"{fqn}.{method_name}" for fqn in matched_fqns)
            selected[key] = (module, method_fqns, original, spec)
        if not matched:
            raise ValueError(f"method replacement pattern {spec.match!r} matched no module")
    return tuple(selected.values())


def apply_method_replacements(
    model: nn.Module,
    specs: Iterable[MethodReplacementSpec],
    *,
    context: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Install method replacements before the sharding planner runs.

    Args:
        model: Finalized model after structure-changing module replacements.
        specs: Rules with a glob module path and a literal method suffix.
        context: Build-time topology and precision values for factories.

    Returns:
        The same model with replacement callables bound to selected instances.
    """
    targets = _compile_method_replacements(model, specs)
    if targets:
        logger.info("method replacements resolved: %s", [
            fqn for _, method_fqns, _, _ in targets for fqn in method_fqns
        ])
    frozen_context = MappingProxyType(dict(context or {}))
    for module, method_fqns, original, spec in targets:
        method_fqn = method_fqns[0]
        replacement = spec.factory(
            module=module, method_fqn=method_fqn, original_method=original, context=frozen_context,
        )
        if not callable(replacement) or isinstance(replacement, nn.Module):
            raise TypeError(f"method replacement factory for {method_fqn!r} must return a callable")
        _validate_callable_compatibility(original, replacement, method_fqn)
        setattr(module, method_fqn.rpartition(".")[2], replacement)
    return model


def _validate_module_match(
    match: tuple[str, ...], module_type: type[nn.Module] | None, exact_type: bool,
) -> None:
    """Validate source selection fields shared by requests and executable specs."""
    if not match or any(not isinstance(pattern, str) or not pattern for pattern in match):
        raise ValueError("module replacement match must contain non-empty FQN patterns")
    if module_type is not None and (not isinstance(module_type, type) or not issubclass(module_type, nn.Module)):
        raise TypeError("module_type must be an nn.Module type")
    if exact_type and module_type is None:
        raise ValueError("exact_type requires module_type")


@dataclass(frozen=True)
class ModuleReplacementRequest:
    """Framework-neutral requested family and explicit arguments, before selection.

    Attributes:
        match: Real source module paths or glob patterns.
        target: Public family entry or an ordinary replacement factory.
        kwargs: Explicit construction and numerical policy arguments.
        origin: Diagnostic origin, assigned by the normalization layer.
        module_type: Optional source type constraint owned by model code.
        exact_type: Match only the exact declared source type.
    """

    match: tuple[str, ...]
    target: Callable
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    origin: str = "adapter"
    module_type: type[nn.Module] | None = None
    exact_type: bool = False
    declaration_id: str = ""

    def __post_init__(self) -> None:
        _validate_module_match(self.match, self.module_type, self.exact_type)
        if not callable(self.target) or not getattr(self.target, "_hp_module_replacement", False):
            raise TypeError("replacement request target must use @module_replacement")
        reserved = {"module", "module_fqn", "context", "origin"} & set(self.kwargs)
        if reserved:
            raise ValueError(f"replacement kwargs contain reserved fields: {sorted(reserved)}")
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))


@dataclass(frozen=True)
class ModuleReplacementSpec:
    """One declarative, structure-preserving module replacement rule."""

    match: tuple[str, ...]
    factory: ModuleReplacementFactory
    module_type: type[nn.Module] | None = None
    exact_type: bool = False
    declaration_id: str = ""
    replacement_target: Optional[str] = None
    static_arguments: Mapping[str, Any] | None = None

    request: ModuleReplacementRequest | None = None

    def __post_init__(self) -> None:
        self._validate_match_and_type()
        self._validate_factory_and_codegen_metadata()

    def _validate_match_and_type(self) -> None:
        """Validate matching and source-module constraints."""
        _validate_module_match(self.match, self.module_type, self.exact_type)

    def _validate_factory_and_codegen_metadata(self) -> None:
        """Validate the factory and optional serializable declaration metadata."""
        if not callable(self.factory) or not getattr(self.factory, "_hp_module_replacement", False):
            raise TypeError("module replacement factory must be decorated with @module_replacement")
        if not isinstance(self.declaration_id, str):
            raise TypeError("module replacement declaration_id must be a string")
        if self.replacement_target is not None and not isinstance(self.replacement_target, str):
            raise TypeError("module replacement replacement_target must be a string or None")
        if self.static_arguments is not None and not isinstance(self.static_arguments, Mapping):
            raise TypeError("module replacement static_arguments must be a mapping or None")


@dataclass(frozen=True)
class ModuleReplacementTarget:
    """One source module's aliases without retaining its parameter storage."""

    module_fqns: tuple[str, ...]
    source_ref: weakref.ReferenceType[nn.Module]
    spec: ModuleReplacementSpec

    @property
    def source(self) -> nn.Module:
        """Resolve the original module or reject a plan whose source expired."""
        source = self.source_ref()
        if source is None:
            raise ValueError(f"replacement target {self.module_fqns[0]!r} no longer exists")
        return source


@dataclass(frozen=True)
class ModuleReplacementPlan:
    """Immutable result of matching replacement rules against one model tree."""

    targets: tuple[ModuleReplacementTarget, ...]
    specs: tuple[ModuleReplacementSpec, ...] = ()
    atomic: bool = False


def _all_module_aliases(model: nn.Module) -> dict[int, tuple[nn.Module, tuple[str, ...]]]:
    return {
        module_id: (module, tuple(sorted(fqns)))
        for module_id, (module, fqns) in _module_aliases(model).items() if fqns[0]
    }


def _matches_module_type(module: nn.Module, rule: ModuleReplacementSpec | ModuleReplacementRequest) -> bool:
    """Share source type checks while leaving error policy to each rule consumer."""
    if rule.module_type is None:
        return True
    return (type(module) is rule.module_type  # pylint: disable=unidiomatic-typecheck
            if rule.exact_type else isinstance(module, rule.module_type))


def _match_module_patterns(aliases: dict, patterns: tuple[str, ...], cache: dict) -> dict[int, tuple[str, ...]]:
    """Expand each pattern tuple once per compilation, preserving inventory order."""
    patterns = tuple(patterns)
    if patterns not in cache:
        matches = {}
        for identity, (_, fqns) in aliases.items():
            matched = tuple(pattern for pattern in patterns if any(fnmatch.fnmatchcase(fqn, pattern) for fqn in fqns))
            if matched:
                matches[identity] = matched
        cache[patterns] = matches
    return cache[patterns]


def _validate_module_type(
    module: nn.Module,
    fqns: tuple[str, ...],
    spec: ModuleReplacementSpec,
) -> None:
    """Validate one matched module against a replacement specification."""
    if not _matches_module_type(module, spec):
        raise TypeError(
            f"module replacement {spec.match} selected {fqns[0]!r} "
            f"of type {type(module).__name__}, expected "
            f"{'exact ' if spec.exact_type else ''}{spec.module_type.__name__}"
        )


def _select_replacement_spec(
    spec: ModuleReplacementSpec,
    aliases: dict[int, tuple[nn.Module, tuple[str, ...]]],
    selected: dict[int, ModuleReplacementTarget],
    cache: dict,
    *,
    spec_index: int = 0,
) -> None:
    """Match one replacement specification and merge its targets."""
    matches = _match_module_patterns(aliases, spec.match, cache)
    for module_id in matches:
        module, fqns = aliases[module_id]
        _validate_module_type(module, fqns, spec)
        previous = selected.get(module_id)
        if previous is not None and previous.spec is not spec:
            raise ValueError(
                f"module replacement conflict for aliases {fqns}: "
                "one source module may match only one factory"
            )
        selected[module_id] = ModuleReplacementTarget(fqns, weakref.ref(module), spec)
    matched_patterns = {pattern for patterns in matches.values() for pattern in patterns}
    unmatched_patterns = [pattern for pattern in spec.match if pattern not in matched_patterns]
    if unmatched_patterns:
        raise ModuleReplacementMatchError(tuple(
            {"rule_id": _replacement_rule_id(spec, spec_index), "pattern": pattern,
             "reason": "matched_no_final_module"}
            for pattern in unmatched_patterns
        ))


def _validate_non_nested_targets(targets: tuple[ModuleReplacementTarget, ...]) -> None:
    """Reject plans that select both a module and one of its descendants."""
    selected_sources = {id(target.source) for target in targets}
    for target in targets:
        for child in target.source.modules():
            if child is not target.source and id(child) in selected_sources:
                raise ValueError(
                    "module replacement does not support selecting a module and its descendant: "
                    f"{target.module_fqns}"
                )


def compile_module_replacements(
    model: nn.Module,
    specs: Iterable[ModuleReplacementSpec] = (),
    *,
    defaults: Iterable[ModuleReplacementRequest] = (),
    overrides: Iterable[ModuleReplacementRequest] = (),
) -> ModuleReplacementPlan:
    """Compose source rules into one plan using a build-local pattern cache.

    Args:
        model: Final or source model tree for this build.
        specs: Declarative executable replacement rules.
        defaults: Model-owned requests, skipped on directly constructed families.
        overrides: User requests overriding defaults by identity.

    Returns:
        A final plan consumed directly by application, without matching again.
    """

    specs, defaults, overrides = tuple(specs), tuple(defaults), tuple(overrides)
    if not (specs or defaults or overrides):
        return ModuleReplacementPlan(())

    inventory = _module_aliases(model)
    direct_ids = {identity for identity, (module, _) in inventory.items()
                  if (info := get_family_build_info(module)) is not None and info.origin == "direct"}
    aliases = {identity: (module, tuple(sorted(paths))) for identity, (module, paths) in inventory.items() if paths[0]}
    cache, selected, opaque = {}, {}, {}
    _compose_request_layer(aliases, defaults, selected, direct_ids, cache, explicit=False)
    explicit_ids = _compose_request_layer(aliases, overrides, selected, direct_ids, cache, explicit=True)
    for index, spec in enumerate(specs):
        _select_replacement_spec(spec, aliases, opaque, cache, spec_index=index)
    for identity, target in opaque.items():
        if identity in direct_ids or identity in explicit_ids:
            raise ValueError(f"explicit replacement conflict at {target.module_fqns}")
        selected.pop(identity, None)
    targets = []
    for module, request in selected.values():
        spec = ModuleReplacementSpec(request.match, _factory_for_request(request),
                                     request.module_type, request.exact_type, request=request,
                                     replacement_target=_symbol_path(request.target), static_arguments=request.kwargs,
                                     declaration_id=request.declaration_id)
        targets.append(ModuleReplacementTarget(request.match, weakref.ref(module), spec))
    targets.extend(opaque.values())

    _validate_non_nested_targets(tuple(targets))
    return ModuleReplacementPlan(tuple(targets), specs=tuple(target.spec for target in targets) if selected else specs,
                                 atomic=any(target.spec.request is not None for target in targets))


def _symbol_path(symbol: Any) -> str | None:
    """Return a stable import-like identity without retaining the symbol."""
    if symbol is None:
        return None
    return f"{symbol.__module__}:{symbol.__qualname__}"


def _replacement_rule_id(spec: ModuleReplacementSpec, index: int) -> str:
    """Return the declared rule id or a stable fallback for Python callers."""
    if spec.declaration_id:
        return spec.declaration_id
    return f"python-replacement-{index:04d}"


def describe_replacement_declarations(
    specs: Iterable[ModuleReplacementSpec],
) -> tuple[dict[str, Any], ...]:
    """Project normalized replacement rules to JSON-safe declarations."""
    return tuple(
        {
            "rule_id": _replacement_rule_id(spec, index),
            "match": list(spec.match),
            "source_type": _symbol_path(spec.module_type),
            "exact_type": spec.exact_type,
            "replacement_target": spec.replacement_target or _symbol_path(spec.factory),
            "static_arguments": dict(spec.static_arguments or {}),
            "order": index,
        }
        for index, spec in enumerate(specs)
    )


def _validate_candidate_projection(
    model: nn.Module,
    replacements: Mapping[str, nn.Module],
    assignments: Mapping[tuple[int, str], tuple[nn.Module, str, nn.Module]],
) -> None:
    """Check every target remains reachable after all scheduled parent-slot changes."""
    for fqn, candidate in replacements.items():
        current = model
        for name in filter(None, fqn.split(".")):
            assignment = assignments.get((id(current), name))
            current = (assignment[2] if assignment is not None
                       else current._modules.get(name))  # pylint: disable=protected-access
            if not isinstance(current, nn.Module):
                break
        if current is not candidate:
            raise ValueError(f"conflicting replacement paths on the final module tree at {fqn!r}")


def _install_module_candidates(
    model: nn.Module,
    candidates: Iterable[tuple[tuple[str, ...], nn.Module, nn.Module]],
) -> nn.Module:
    """Preflight and install (aliases, source, candidate) records, returning the final root."""
    replacements = {}
    for fqns, source, candidate in candidates:
        for fqn in fqns:
            try:
                current = model.get_submodule(fqn)
            except AttributeError as exc:
                raise ValueError(f"replacement target {fqn!r} is no longer registered") from exc
            if current is not source:
                raise ValueError(f"replacement target {fqn!r} changed while plan was being applied")
            if replacements.setdefault(fqn, candidate) is not candidate:
                raise ValueError(f"conflicting replacement candidates at {fqn!r}")

    assignments = {}
    for fqn, candidate in replacements.items():
        if not fqn:
            continue
        parent, _, name = fqn.rpartition(".")
        ancestor = parent
        while ancestor and ancestor not in replacements:
            ancestor = ancestor.rpartition(".")[0]
        root = replacements.get(ancestor, model)
        relative = parent[len(ancestor):].lstrip(".") if ancestor in replacements else parent
        try:
            owner = root.get_submodule(relative)
        except AttributeError as exc:
            raise ValueError(f"replacement parent for {fqn!r} is no longer registered") from exc
        if not isinstance(owner._modules.get(name), nn.Module):  # pylint: disable=protected-access
            raise ValueError(f"replacement target {fqn!r} is no longer registered on its final parent")
        slot = (id(owner), name)
        previous = assignments.setdefault(slot, (owner, name, candidate))
        if previous[2] is not candidate:
            raise ValueError(f"conflicting replacement candidates on the final parent slot for {fqn!r}")
    final_model = replacements.get("", model)
    _validate_candidate_projection(final_model, replacements, assignments)
    for owner, name, candidate in assignments.values():
        owner._modules[name] = candidate  # pylint: disable=protected-access
    return final_model


def _registered_names(module: nn.Module) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return direct module, parameter, and buffer registry names."""

    return (  # pylint: disable=protected-access
        tuple(module._modules),
        tuple(module._parameters),
        tuple(module._buffers),
    )


def _named_identities(module: nn.Module, *, kind: str) -> dict[str, object]:
    """Return every registered parameter or buffer without removing aliases."""

    if kind == "parameter":
        return dict(module.named_parameters(remove_duplicate=False))
    if kind == "buffer":
        return dict(module.named_buffers(remove_duplicate=False))
    raise ValueError(f"unsupported module identity kind {kind!r}")


def _named_tensor_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    """Capture parameter and persistent-buffer shapes before replacement."""
    shapes = {
        name: tuple(value.shape)
        for name, value in model.named_parameters(remove_duplicate=False)
    }
    for module_name, module in model.named_modules(remove_duplicate=False):
        for name, value in module._buffers.items():  # pylint: disable=protected-access
            if value is None or name in module._non_persistent_buffers_set:  # pylint: disable=protected-access
                continue
            fqn = f"{module_name}.{name}" if module_name else name
            shapes[fqn] = tuple(value.shape)
    return shapes


def _validate_callable_compatibility(
    source: Callable, replacement: Callable, fqn: str, *, kind: str = "call",
) -> None:
    """Ensure a replacement callable accepts the source's supported call shapes."""

    try:
        source_signature = inspect.signature(source)
        replacement_signature = inspect.signature(replacement)
        source_parameters = tuple(source_signature.parameters.values())
        source_kinds = {parameter.kind for parameter in source_parameters}
        replacement_kinds = {parameter.kind for parameter in replacement_signature.parameters.values()}
        required_variadics = source_kinds & {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        if not required_variadics.issubset(replacement_kinds):
            raise TypeError("replacement drops source variadic arguments")

        all_positional_args = []
        required_positional_args = []
        keyword_probe_positional_args = []
        keyword_args = {}
        for parameter in source_parameters:
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                all_positional_args.append(None)
                if parameter.default is inspect.Parameter.empty:
                    required_positional_args.append(None)
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                keyword_probe_positional_args.append(None)
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                keyword_args[parameter.name] = None

        required_keyword_args = {
            parameter.name: None for parameter in source_parameters
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is inspect.Parameter.empty
        }
        probes = [(all_positional_args, required_keyword_args), (keyword_probe_positional_args, keyword_args)]
        if required_positional_args != all_positional_args:
            probes.append((required_positional_args, required_keyword_args))
        for args, kwargs in probes:
            source_signature.bind(*args, **kwargs)
            replacement_signature.bind(*args, **kwargs)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"replacement for {fqn!r} has an incompatible "
            f"{kind} signature; preserve the source call contract or add an explicit "
            "model-owned wrapper"
        ) from error


def _validate_replacement(
    source: nn.Module,
    replacement: nn.Module,
    fqn: str,
    *,
    has_weight_transforms: bool = False,
) -> None:
    """Validate that a replacement preserves the source module contract."""

    if not isinstance(replacement, nn.Module):
        raise TypeError(f"replacement factory for {fqn!r} must return nn.Module")
    if source.training != replacement.training:
        raise ValueError(
            f"replacement for {fqn!r} must preserve training state"
        )
    if not has_weight_transforms:
        if _registered_names(source) != _registered_names(replacement):
            raise ValueError(
                f"replacement for {fqn!r} changed registered "
                "module/parameter/buffer names without checkpoint transforms"
            )
        for kind in ("parameter", "buffer"):
            source_identities = _named_identities(source, kind=kind)
            replacement_identities = _named_identities(replacement, kind=kind)
            if tuple(source_identities) != tuple(replacement_identities):
                raise ValueError(
                    f"replacement for {fqn!r} changed {kind} names"
                )
            for name, source_value in source_identities.items():
                if replacement_identities[name] is not source_value:
                    raise ValueError(
                        f"replacement for {fqn!r} must preserve "
                        f"{kind} {name!r} identity"
                    )
        if tuple(source.state_dict()) != tuple(replacement.state_dict()):
            raise ValueError(
                f"replacement for {fqn!r} changed state_dict keys"
            )
    _validate_callable_compatibility(source.forward, replacement.forward, fqn, kind="forward")
    hook_registries = {
        name: value for name, value in vars(source).items()
        if "hook" in name and isinstance(value, Mapping) and value
    }
    if hook_registries:
        raise ValueError(
            f"replacement for {fqn!r} cannot migrate existing module hooks"
        )


def _apply_module_replacement_actions(
    model: nn.Module,
    replacement_rules: Iterable[ModuleReplacementSpec] | ModuleReplacementPlan | None = None,
    *,
    weights_mapping: list[WeightRenaming | WeightConverter] | None = None,
    context: Mapping[str, Any] | None = None,
    capture_checkpoint_metadata: bool = True,
) -> tuple[nn.Module, list[WeightRenaming | WeightConverter] | None]:
    """Apply normalized replacement rules before sharding sees the model.

    Moved from ``_transformers/infrastructure.py`` (05 §15.2.1). The rules
    arrive already normalized: Trainer YAML ``plan_overrides`` entries are
    desugared into ``ModuleReplacementSpec`` rules on the Trainer side
    (05 §15.2.6), so this module never imports trainer config. ``context``
    carries the low-precision policy object (or None) plus the active
    parallel axes consumed by replacement factories.
    """

    if weights_mapping is None:
        weights_mapping = get_model_conversion_mapping(model)
    context = context or {}
    active_model_parallel_axes = [
        axis.upper()
        for axis in ("tp", "cp", "ep", "pp")
        if context.get(axis)
    ]
    if context.get("low_precision") is not None and active_model_parallel_axes:
        raise NotImplementedError(
            "Low-precision online training currently requires TP=CP=EP=PP=1; "
            f"active axes: {active_model_parallel_axes}."
        )
    plan = (replacement_rules if isinstance(replacement_rules, ModuleReplacementPlan)
            else compile_module_replacements(model, tuple(replacement_rules or ())))
    return apply_module_replacements(
        model,
        plan,
        weights_mapping=weights_mapping,
        context=context,
        capture_checkpoint_metadata=capture_checkpoint_metadata,
    )


def _prepare_replacement_target(
    target: ModuleReplacementTarget,
    context: Mapping[str, Any],
    weights_mapping: list[WeightRenaming | WeightConverter] | None,
) -> tuple[nn.Module, list[WeightRenaming | WeightConverter]]:
    """Build and validate a source-read-only candidate without mutating the model tree."""
    source = target.source
    replacement = target.spec.factory(
        module=source,
        module_fqn=target.module_fqns[0],
        context=context,
    )
    transforms = []
    if getattr(replacement, "make_transforms", None) is not None:
        transforms = replacement.make_transforms()
    if not isinstance(transforms, list) or any(
        not isinstance(transform, (WeightRenaming, WeightConverter))
        for transform in transforms
    ):
        raise TypeError(
            "replacement make_transforms() must return "
            "list[WeightRenaming | WeightConverter]"
        )
    for transform in transforms:
        transform.scope_prefix = target.module_fqns[0]
    _validate_replacement(
        source,
        replacement,
        target.module_fqns[0],
        has_weight_transforms=bool(transforms),
    )
    if transforms and callable(getattr(replacement, "reset_parameters", None)):
        replacement._hp_reset_after_materialization = True  # pylint: disable=protected-access
    if transforms and weights_mapping is None:
        raise ValueError(
            "weights_mapping is required when a replacement defines make_transforms()"
        )
    return replacement, transforms


def _prepare_replacement_batches(plan, context, weights_mapping):
    """Prepare all atomic candidates together or one legacy candidate at a time."""
    candidates = ((target, *_prepare_replacement_target(target, context, weights_mapping))
                  for target in plan.targets)
    if plan.atomic:
        yield list(candidates)
    else:
        for candidate in candidates:
            yield [candidate]


def apply_module_replacements(
    model: nn.Module,
    plan: ModuleReplacementPlan,
    *,
    weights_mapping: list[WeightRenaming | WeightConverter] | None = None,
    context: Mapping[str, Any] | None = None,
    capture_checkpoint_metadata: bool = True,
) -> tuple[nn.Module, list[WeightRenaming | WeightConverter] | None]:
    """Install a validated request batch, or a legacy incremental plan.

    Neutral requests construct and validate every candidate before any tree change.
    Opaque legacy factories retain bounded incremental conversion; on their failure
    callers discard the partially converted model.

    Args:
        model: Model tree whose source boundaries are finalized.
        plan: Precompiled aliases, source constraints and installation policy.
        weights_mapping: Mutable mapping receiving successful checkpoint transforms.
        context: Read-only logical job context.
        capture_checkpoint_metadata: Whether source shapes are retained for loading.
    """

    context = MappingProxyType(dict(context or {}))
    source_shapes = None
    extra_transforms: list[WeightRenaming | WeightConverter] = []
    for prepared in _prepare_replacement_batches(plan, context, weights_mapping):
        if capture_checkpoint_metadata and source_shapes is None and any(transforms for _, _, transforms in prepared):
            source_shapes = _named_tensor_shapes(model)
        model = _install_module_candidates(
            model, ((target.module_fqns, target.source, candidate) for target, candidate, _ in prepared),
        )
        for _, _, transforms in prepared:
            extra_transforms.extend(transforms)
    if extra_transforms:
        if capture_checkpoint_metadata:
            model._hp_checkpoint_source_shapes = source_shapes  # pylint: disable=protected-access
            model._hp_replacement_weight_conversions = extra_transforms  # pylint: disable=protected-access
        weights_mapping[:0] = extra_transforms
    previous_replacements = tuple(getattr(model, "_hp_replaced_module_fqns", ()) or ())
    current_replacements = tuple(
        fqn
        for target in plan.targets
        for fqn in target.module_fqns
    )
    model._hp_replaced_module_fqns = tuple(  # pylint: disable=protected-access
        dict.fromkeys((*previous_replacements, *current_replacements))
    )
    return model, weights_mapping


def _factory_for_request(request: ModuleReplacementRequest) -> ModuleReplacementFactory:
    @module_replacement
    def factory(*, module: nn.Module, module_fqn: str, context: Mapping[str, Any]) -> nn.Module:
        """Construct one already-composed source boundary using its preserved request."""
        family = get_module_family(request.target)
        if family is not None and family.default_variant is not None:
            return build_family(
                FamilyRequest(request.target, request.kwargs),
                module=module, module_fqn=module_fqn, context=context,
            )
        return request.target(module=module, module_fqn=module_fqn, context=context, **request.kwargs)
    return factory


def _request_matches(aliases: dict, request: ModuleReplacementRequest, cache: dict) -> dict[int, tuple]:
    """Filter shared path matches using the request's source type contract."""
    matches = {identity: aliases[identity] for identity in _match_module_patterns(aliases, request.match, cache)
               if _matches_module_type(aliases[identity][0], request)}
    if not matches:
        raise ModuleReplacementMatchError(tuple(
            {"rule_id": request.declaration_id or "python-replacement-0000", "pattern": pattern,
             "reason": "matched_no_compatible_module"} for pattern in request.match
        ))
    return matches


def _compose_request_layer(aliases, requests, selected, direct_ids, cache, *, explicit):
    """Compose one precedence layer without allocating candidate modules."""
    current_ids = set()
    for request in requests:
        if not isinstance(request, ModuleReplacementRequest):
            raise TypeError("replacement providers must return ModuleReplacementRequest objects")
        for identity, (module, paths) in _request_matches(aliases, request, cache).items():
            if identity in direct_ids:
                if explicit:
                    raise ValueError(f"explicit replacement conflicts with directly constructed family at {paths}")
                continue
            if any(id(child) in direct_ids for child in module.modules()):
                raise ValueError(f"parent replacement at {paths} overlaps a directly constructed family")
            if identity in current_ids:
                raise ValueError(f"duplicate {'user' if explicit else 'default'} replacement for aliases {paths}")
            current_ids.add(identity)
            old = selected.get(identity)
            same_family = (explicit and old is not None and old[1].target is request.target
                           and get_module_family(request.target) is not None)
            arguments = {**old[1].kwargs, **request.kwargs} if same_family else request.kwargs
            selected[identity] = (module, replace(request, kwargs=arguments, match=paths))
    return current_ids
