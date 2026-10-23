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
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable

from torch import nn

from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
    get_model_conversion_mapping,
)


ModuleReplacementFactory = Callable[..., nn.Module]
MethodReplacementFactory = Callable[..., Callable[..., Any]]
logger = logging.getLogger(__name__)


def module_replacement(factory: ModuleReplacementFactory) -> ModuleReplacementFactory:
    """Declare a factory compatible with the generic replacement executor."""

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


@dataclass(frozen=True)
class ModuleReplacementSpec:
    """One declarative, structure-preserving module replacement rule."""

    match: tuple[str, ...]
    factory: ModuleReplacementFactory
    module_type: type[nn.Module] | None = None
    exact_type: bool = False

    def __post_init__(self) -> None:
        if not self.match or any(not isinstance(pattern, str) or not pattern for pattern in self.match):
            raise ValueError("module replacement match must contain non-empty FQN patterns")
        if self.module_type is not None and (
                not isinstance(self.module_type, type) or not issubclass(self.module_type, nn.Module)):
            raise TypeError("module_type must be an nn.Module type")
        if self.exact_type and self.module_type is None:
            raise ValueError("exact_type requires module_type")
        if not callable(self.factory) or not getattr(self.factory, "_hp_module_replacement", False):
            raise TypeError("module replacement factory must be decorated with @module_replacement")


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


def _all_module_aliases(model: nn.Module) -> dict[int, tuple[nn.Module, tuple[str, ...]]]:
    aliases: dict[int, tuple[nn.Module, list[str]]] = {}
    for fqn, module in model.named_modules(remove_duplicate=False):
        if not fqn:
            continue
        module_id = id(module)
        if module_id not in aliases:
            aliases[module_id] = (module, [])
        aliases[module_id][1].append(fqn)
    return {
        module_id: (module, tuple(sorted(fqns)))
        for module_id, (module, fqns) in aliases.items()
    }


def _validate_module_type(
    module: nn.Module,
    fqns: tuple[str, ...],
    spec: ModuleReplacementSpec,
) -> None:
    """Validate one matched module against a replacement specification."""
    if spec.module_type is None:
        return
    # exact_type deliberately requires an exact type match, so the type()
    # comparison is intentional and must not become isinstance().
    type_matches = (
        type(module) is spec.module_type  # pylint: disable=unidiomatic-typecheck
        if spec.exact_type
        else isinstance(module, spec.module_type)
    )
    if not type_matches:
        raise TypeError(
            f"module replacement {spec.match} selected {fqns[0]!r} "
            f"of type {type(module).__name__}, expected "
            f"{'exact ' if spec.exact_type else ''}{spec.module_type.__name__}"
        )


def _select_replacement_spec(
    spec: ModuleReplacementSpec,
    aliases: dict[int, tuple[nn.Module, tuple[str, ...]]],
    selected: dict[int, ModuleReplacementTarget],
) -> None:
    """Match one replacement specification and merge its targets."""
    matched_ids_by_pattern = {pattern: set() for pattern in spec.match}
    for module_id, (module, fqns) in aliases.items():
        matched_patterns = tuple(
            pattern for pattern in spec.match if any(fnmatch.fnmatchcase(fqn, pattern) for fqn in fqns)
        )
        if not matched_patterns:
            continue
        _validate_module_type(module, fqns, spec)
        for pattern in matched_patterns:
            matched_ids_by_pattern[pattern].add(module_id)
        previous = selected.get(module_id)
        if previous is not None and previous.spec is not spec:
            raise ValueError(
                f"module replacement conflict for aliases {fqns}: "
                "one source module may match only one factory"
            )
        selected[module_id] = ModuleReplacementTarget(fqns, weakref.ref(module), spec)
    unmatched_patterns = [
        pattern for pattern, matched_ids in matched_ids_by_pattern.items() if not matched_ids
    ]
    if unmatched_patterns:
        raise ValueError(
            "module replacement pattern(s) matched no final module: "
            f"{unmatched_patterns}. Inspect the finalized module inventory and update the "
            "adapter pattern; do not silently skip a required replacement."
        )


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
    specs: Iterable[ModuleReplacementSpec],
) -> ModuleReplacementPlan:
    """Match rules once, validate aliases, and return an immutable plan."""

    specs = tuple(specs)
    if not specs:
        return ModuleReplacementPlan(())

    aliases = _all_module_aliases(model)
    selected: dict[int, ModuleReplacementTarget] = {}
    for spec in specs:
        _select_replacement_spec(spec, aliases, selected)

    targets = tuple(selected.values())
    _validate_non_nested_targets(targets)
    return ModuleReplacementPlan(targets)


def _parent_and_name(model: nn.Module, fqn: str) -> tuple[nn.Module, str]:
    """Resolve a replacement target's parent module and local name."""

    parent_fqn, _, name = fqn.rpartition(".")
    parent = model.get_submodule(parent_fqn) if parent_fqn else model
    if not name or parent._modules.get(name) is None:  # pylint: disable=protected-access
        raise ValueError(f"replacement target {fqn!r} is no longer registered on its parent")
    return parent, name


def _validate_target_binding(model: nn.Module, target: ModuleReplacementTarget) -> None:
    """Check all aliases before installing any part of one replacement."""
    source = target.source
    for fqn in target.module_fqns:
        parent, name = _parent_and_name(model, fqn)
        if parent._modules[name] is not source:  # pylint: disable=protected-access
            raise ValueError(f"replacement target {fqn!r} changed while plan was being applied")


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


def _validate_forward_compatibility(
    source: nn.Module,
    replacement: nn.Module,
    fqn: str,
) -> None:
    """Ensure replacement accepts every call shape supported by source.forward."""

    _validate_callable_compatibility(source.forward, replacement.forward, fqn, kind="forward")


def _validate_callable_compatibility(
    source: Callable, replacement: Callable, fqn: str, *, kind: str = "call",
) -> None:
    """Ensure a replacement callable accepts the source's supported call shapes."""

    try:
        source_signature = inspect.signature(source)
        replacement_signature = inspect.signature(replacement)
        source_parameters = tuple(source_signature.parameters.values())
        source_has_varargs = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in source_parameters
        )
        source_has_varkwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in source_parameters
        )
        replacement_has_varargs = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in replacement_signature.parameters.values()
        )
        replacement_has_varkwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in replacement_signature.parameters.values()
        )
        if source_has_varargs and not replacement_has_varargs:
            raise TypeError("replacement drops source *args support")
        if source_has_varkwargs and not replacement_has_varkwargs:
            raise TypeError("replacement drops source **kwargs support")

        all_positional_args = []
        keyword_probe_positional_args = []
        keyword_args = {}
        for parameter in source_parameters:
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                all_positional_args.append(None)
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                keyword_probe_positional_args.append(None)
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                keyword_args[parameter.name] = None

        source_signature.bind(*all_positional_args)
        source_signature.bind(*keyword_probe_positional_args, **keyword_args)
        replacement_signature.bind(*all_positional_args)
        replacement_signature.bind(*keyword_probe_positional_args, **keyword_args)
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
    _validate_forward_compatibility(source, replacement, fqn)
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
    replacement_rules: Iterable[ModuleReplacementSpec] | None = None,
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
    context = dict(context or {})
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
    plan = compile_module_replacements(model, tuple(replacement_rules or ()))
    return apply_module_replacements(
        model,
        plan,
        weights_mapping=weights_mapping,
        context=context,
        capture_checkpoint_metadata=capture_checkpoint_metadata,
    )


def _install_replacement_target(
    model: nn.Module,
    target: ModuleReplacementTarget,
    context: Mapping[str, Any],
    extra_transforms: list[WeightRenaming | WeightConverter],
    weights_mapping: list[WeightRenaming | WeightConverter] | None,
) -> None:
    """Build, validate, and install one replacement target in place."""
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
        extra_transforms.append(transform)
    _validate_replacement(
        source,
        replacement,
        target.module_fqns[0],
        has_weight_transforms=bool(transforms),
    )
    if transforms and callable(getattr(replacement, "reset_parameters", None)):
        replacement._hp_reset_after_materialization = True  # pylint: disable=protected-access
    _validate_target_binding(model, target)
    if transforms and weights_mapping is None:
        raise ValueError(
            "weights_mapping is required when a replacement defines make_transforms()"
        )
    for fqn in target.module_fqns:
        parent, name = _parent_and_name(model, fqn)
        parent._modules[name] = replacement  # pylint: disable=protected-access


def apply_module_replacements(
    model: nn.Module,
    plan: ModuleReplacementPlan,
    *,
    weights_mapping: list[WeightRenaming | WeightConverter] | None = None,
    context: Mapping[str, Any] | None = None,
    capture_checkpoint_metadata: bool = True,
) -> tuple[nn.Module, list[WeightRenaming | WeightConverter] | None]:
    """Build, validate, and install one target at a time to bound live storage.

    Plans do not retain replaced source modules. A factory or validation
    failure may leave earlier targets installed; callers must discard the
    partially converted model and rebuild rather than retrying this plan.
    """

    context = MappingProxyType(dict(context or {}))
    source_shapes = _named_tensor_shapes(model) if plan.targets and capture_checkpoint_metadata else None
    extra_transforms: list[WeightRenaming | WeightConverter] = []
    for target in plan.targets:
        _install_replacement_target(
            model,
            target,
            context,
            extra_transforms,
            weights_mapping,
        )
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
