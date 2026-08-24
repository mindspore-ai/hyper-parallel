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
"""Compile and atomically apply structure-preserving module replacements."""

from __future__ import annotations

import fnmatch
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable

from torch import nn
from transformers.core_model_loading import WeightConverter, WeightRenaming


ModuleReplacementFactory = Callable[..., nn.Module]


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
    factory._hp_module_replacement = True  # type: ignore[attr-defined]
    return factory


@dataclass(frozen=True)
class ModuleReplacementSpec:
    """One declarative, structure-preserving module replacement rule."""

    match: tuple[str, ...]
    factory: ModuleReplacementFactory
    module_type: type[nn.Module]
    exact_type: bool = False

    def __post_init__(self) -> None:
        if not self.match or any(not isinstance(pattern, str) or not pattern for pattern in self.match):
            raise ValueError("module replacement match must contain non-empty FQN patterns")
        if not isinstance(self.module_type, type) or not issubclass(self.module_type, nn.Module):
            raise TypeError("module_type must be an nn.Module type")
        if not callable(self.factory) or not getattr(self.factory, "_hp_module_replacement", False):
            raise TypeError("module replacement factory must be decorated with @module_replacement")


@dataclass(frozen=True)
class ModuleReplacementTarget:
    """One source module and every registered FQN that aliases it."""

    module_fqns: tuple[str, ...]
    source: nn.Module
    spec: ModuleReplacementSpec


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
        matched_ids_by_pattern = {pattern: set() for pattern in spec.match}
        for module_id, (module, fqns) in aliases.items():
            matched_patterns = tuple(
                pattern for pattern in spec.match
                if any(fnmatch.fnmatchcase(fqn, pattern) for fqn in fqns)
            )
            if not matched_patterns:
                continue
            # exact_type deliberately requires an exact type match, so the
            # type() comparison is intentional and must not become isinstance().
            type_matches = (
                type(module) is spec.module_type  # pylint: disable=unidiomatic-typecheck
                if spec.exact_type else isinstance(module, spec.module_type)
            )
            if not type_matches:
                raise TypeError(
                    f"module replacement {spec.match} selected {fqns[0]!r} "
                    f"of type {type(module).__name__}, expected "
                    f"{'exact ' if spec.exact_type else ''}{spec.module_type.__name__}"
                )
            for pattern in matched_patterns:
                matched_ids_by_pattern[pattern].add(module_id)
            previous = selected.get(module_id)
            if previous is not None and previous.spec is not spec:
                raise ValueError(
                    f"module replacement conflict for aliases {fqns}: "
                    "one source module may match only one factory"
                )
            selected[module_id] = ModuleReplacementTarget(fqns, module, spec)
        unmatched_patterns = [
            pattern for pattern, matched_ids in matched_ids_by_pattern.items()
            if not matched_ids
        ]
        if unmatched_patterns:
            raise ValueError(
                "module replacement pattern(s) matched no module: "
                f"{unmatched_patterns}"
            )

    targets = tuple(selected.values())
    selected_sources = {id(target.source) for target in targets}
    for target in targets:
        for child in target.source.modules():
            if child is not target.source and id(child) in selected_sources:
                raise ValueError(
                    "module replacement does not support selecting a module and its descendant: "
                    f"{target.module_fqns}"
                )
    return ModuleReplacementPlan(targets)


def _parent_and_name(model: nn.Module, fqn: str) -> tuple[nn.Module, str]:
    parent_fqn, _, name = fqn.rpartition(".")
    parent = model.get_submodule(parent_fqn) if parent_fqn else model
    if not name or parent._modules.get(name) is None:
        raise ValueError(f"replacement target {fqn!r} is no longer registered on its parent")
    return parent, name


def _registered_names(module: nn.Module) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return tuple(module._modules), tuple(module._parameters), tuple(module._buffers)


def _named_identities(module: nn.Module, *, kind: str) -> dict[str, object]:
    """Return every registered parameter or buffer without removing aliases."""

    if kind == "parameter":
        return dict(module.named_parameters(remove_duplicate=False))
    if kind == "buffer":
        return dict(module.named_buffers(remove_duplicate=False))
    raise ValueError(f"unsupported module identity kind {kind!r}")


def _validate_forward_compatibility(
    source: nn.Module,
    replacement: nn.Module,
    fqn: str,
) -> None:
    """Ensure replacement accepts every call shape supported by source.forward."""

    try:
        source_signature = inspect.signature(source.forward)
        replacement_signature = inspect.signature(replacement.forward)
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
            f"replacement for {fqn!r} has an incompatible forward signature"
        ) from error


def _validate_replacement(
    source: nn.Module,
    replacement: nn.Module,
    fqn: str,
    *,
    has_weight_transforms: bool = False,
) -> None:
    if not isinstance(replacement, nn.Module):
        raise TypeError(f"replacement factory for {fqn!r} must return nn.Module")
    if source.training != replacement.training:
        raise ValueError(f"replacement for {fqn!r} must preserve training state")
    if not has_weight_transforms:
        if _registered_names(source) != _registered_names(replacement):
            raise ValueError(f"replacement for {fqn!r} changed registered module/parameter/buffer names")
        for kind in ("parameter", "buffer"):
            source_identities = _named_identities(source, kind=kind)
            replacement_identities = _named_identities(replacement, kind=kind)
            if tuple(source_identities) != tuple(replacement_identities):
                raise ValueError(f"replacement for {fqn!r} changed {kind} names")
            for name, source_value in source_identities.items():
                if replacement_identities[name] is not source_value:
                    raise ValueError(
                        f"replacement for {fqn!r} must preserve {kind} {name!r} identity"
                    )
        if tuple(source.state_dict()) != tuple(replacement.state_dict()):
            raise ValueError(f"replacement for {fqn!r} changed state_dict keys")
    _validate_forward_compatibility(source, replacement, fqn)
    hook_registries = {
        name: value for name, value in vars(source).items()
        if "hook" in name and isinstance(value, Mapping) and value
    }
    if hook_registries:
        raise ValueError(f"replacement for {fqn!r} cannot migrate existing module hooks")


def apply_module_replacements(
    model: nn.Module,
    plan: ModuleReplacementPlan,
    *,
    weights_mapping: list[WeightRenaming | WeightConverter] | None = None,
    context: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, list[WeightRenaming | WeightConverter] | None]:
    """Build all replacements, validate them, then install them atomically."""

    factory_context = MappingProxyType(dict(context or {}))
    prepared: list[tuple[ModuleReplacementTarget, nn.Module]] = []
    extra_transforms: list[WeightRenaming | WeightConverter] = []
    for target in plan.targets:
        replacement = target.spec.factory(
            module=target.source,
            module_fqn=target.module_fqns[0],
            context=factory_context,
        )
        make_transforms = getattr(replacement, "make_transforms", None)
        transforms = [] if make_transforms is None else make_transforms()
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
            target.source,
            replacement,
            target.module_fqns[0],
            has_weight_transforms=bool(transforms),
        )
        prepared.append((target, replacement))

    for target, _ in prepared:
        for fqn in target.module_fqns:
            parent, name = _parent_and_name(model, fqn)
            if parent._modules[name] is not target.source:
                raise ValueError(f"replacement target {fqn!r} changed while plan was being applied")
    if extra_transforms:
        if weights_mapping is None:
            raise ValueError(
                "weights_mapping is required when a replacement defines make_transforms()"
            )
        weights_mapping[:0] = extra_transforms
    for target, replacement in prepared:
        for fqn in target.module_fqns:
            parent, name = _parent_and_name(model, fqn)
            parent._modules[name] = replacement
    return model, weights_mapping
