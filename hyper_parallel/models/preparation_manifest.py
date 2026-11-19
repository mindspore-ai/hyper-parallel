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
"""Stable, resource-free records produced by shared model preparation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import partial
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from torch import Tensor, nn

from hyper_parallel.components.modules.module_family import (
    collect_family_modules,
    get_family_build_info,
    resolve_module_family_target,
)
from hyper_parallel.models.prepared_family import PREPARED_FAMILY_RECORD_VERSION

if TYPE_CHECKING:
    from hyper_parallel.models.replacement import ModuleReplacementPlan, ModuleReplacementSpec, ModuleReplacementTarget


_JSON_SCALARS = (type(None), bool, int, float, str)


@dataclass(frozen=True, order=True)
class SymbolRef:
    """Importable identity of a class or callable without retaining the object."""

    module: str
    qualname: str

    @classmethod
    def from_object(cls, value: Any, *, path: str = "symbol") -> "SymbolRef":
        """Validate and describe one globally importable object."""
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
            raise TypeError(f"{path} must be an importable class or callable")
        if module == "__main__" or "<locals>" in qualname:
            raise ValueError(f"{path} is not stably importable: {module}.{qualname}")
        resolved = import_module(module)
        for name in qualname.split("."):
            resolved = getattr(resolved, name, None)
            if resolved is None:
                break
        if resolved is not value:
            raise ValueError(f"{path} does not resolve to the same object: {module}.{qualname}")
        return cls(module, qualname)

    def to_dict(self) -> dict[str, str]:
        """Return the canonical JSON representation."""
        return {"module": self.module, "qualname": self.qualname}

    def legacy_path(self, separator: str = ":") -> str:
        """Return the temporary path syntax consumed by the existing emitter."""
        return f"{self.module}{separator}{self.qualname}"


@dataclass(frozen=True)
class PartialRef:
    """Stable description of ``functools.partial`` construction arguments."""

    target: SymbolRef
    args: tuple[Any, ...] = ()
    keywords: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(_freeze_stable_value(value, path="partial.args") for value in self.args))
        object.__setattr__(
            self,
            "keywords",
            MappingProxyType({
                name: _freeze_stable_value(value, path=f"partial.keywords.{name}")
                for name, value in self.keywords.items()
            }),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        return {
            "target": self.target.to_dict(),
            "args": [_stable_value_to_data(value) for value in self.args],
            "keywords": {
                name: _stable_value_to_data(value) for name, value in self.keywords.items()
            },
        }


def _freeze_stable_value(value: Any, *, path: str) -> Any:
    """Copy one manifest value while rejecting runtime-only or local objects."""
    if type(value) in _JSON_SCALARS:
        return value
    if isinstance(value, (Tensor, nn.Module)):
        raise TypeError(
            f"{path} contains unsupported runtime value "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )
    if isinstance(value, SymbolRef):
        return value
    if isinstance(value, PartialRef):
        return value
    if isinstance(value, partial):
        return PartialRef(
            SymbolRef.from_object(value.func, path=f"{path}.func"),
            tuple(value.args),
            value.keywords or {},
        )
    if isinstance(value, Mapping):
        if not all(isinstance(name, str) for name in value):
            raise TypeError(f"{path} mapping keys must be strings")
        return MappingProxyType({
            name: _freeze_stable_value(item, path=f"{path}.{name}")
            for name, item in value.items()
        })
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_stable_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (type,)) or callable(value):
        return SymbolRef.from_object(value, path=path)
    raise TypeError(f"{path} contains unsupported runtime value {type(value).__module__}.{type(value).__qualname__}")


def _stable_value_to_data(value: Any) -> Any:
    """Convert a frozen manifest value to ordinary JSON-compatible data."""
    if isinstance(value, SymbolRef):
        return value.to_dict()
    if isinstance(value, PartialRef):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {name: _stable_value_to_data(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_stable_value_to_data(item) for item in value]
    return value


def _encode_construction_value(value: Any) -> Any:
    """Encode replayable arguments without confusing symbols and user mappings."""
    if isinstance(value, Mapping):
        return {"mapping": {key: _encode_construction_value(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        kind = "list" if isinstance(value, list) else "tuple"
        return {kind: [_encode_construction_value(item) for item in value]}
    if isinstance(value, SymbolRef):
        return {"symbol": value.legacy_path()}
    if isinstance(value, PartialRef):
        return {"partial": [
            value.target.legacy_path(),
            _encode_construction_value(value.args),
            _encode_construction_value(value.keywords),
        ]}
    if isinstance(value, partial):
        return {"partial": [
            SymbolRef.from_object(value.func).legacy_path(),
            _encode_construction_value(value.args),
            _encode_construction_value(value.keywords or {}),
        ]}
    if callable(value):
        return {"symbol": SymbolRef.from_object(value).legacy_path()}
    return value


@dataclass(frozen=True)
class WeightConversionRecord:
    """Resource-free checkpoint conversion owned by one prepared boundary."""

    transform_type: SymbolRef
    source_patterns: tuple[str, ...] = ()
    target_patterns: tuple[str, ...] = ()
    scope_prefix: str | None = None
    operations: tuple[SymbolRef, ...] = ()

    @classmethod
    def from_transform(cls, transform: Any) -> "WeightConversionRecord":
        """Copy stable fields from a Transformers-compatible weight transform."""
        return cls(
            transform_type=SymbolRef.from_object(type(transform), path="weight transform type"),
            source_patterns=tuple(getattr(transform, "source_patterns", ()) or ()),
            target_patterns=tuple(getattr(transform, "target_patterns", ()) or ()),
            scope_prefix=getattr(transform, "scope_prefix", None),
            operations=tuple(
                SymbolRef.from_object(type(operation), path="weight conversion operation")
                for operation in (getattr(transform, "operations", ()) or ())
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        return {
            "transform_type": self.transform_type.to_dict(),
            "source_patterns": list(self.source_patterns),
            "target_patterns": list(self.target_patterns),
            "scope_prefix": self.scope_prefix,
            "operations": [operation.to_dict() for operation in self.operations],
        }


@dataclass(frozen=True)
class ReplacementApplication:
    """One replacement committed by the shared atomic installer."""

    boundary_id: str
    declaration_id: str
    aliases: tuple[str, ...]
    source_type: SymbolRef
    replacement_type: SymbolRef
    request_target: SymbolRef | None = None
    resolved_member: SymbolRef | None = None
    selectors: Mapping[str, Any] = field(default_factory=dict)
    request_arguments: Mapping[str, Any] = field(default_factory=dict)
    axis_sizes: Mapping[str, Any] = field(default_factory=dict)
    weight_conversions: tuple[WeightConversionRecord, ...] = ()
    resolved_variant: str | None = None
    resolved_arguments: Mapping[str, Any] = field(default_factory=dict)
    construction_context: Mapping[str, Any] = field(default_factory=dict)
    replay: Mapping[str, Any] = field(default_factory=dict)
    _prepared_arguments: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        prepared = {
            name: _encode_construction_value(getattr(self, name))
            for name in ("request_arguments", "resolved_arguments", "construction_context")
        } if self.resolved_variant is not None else {}
        object.__setattr__(self, "_prepared_arguments", _freeze_stable_value(prepared, path="prepared_arguments"))
        object.__setattr__(self, "replay", _freeze_stable_value(self.replay, path="replacement.replay"))
        for name in ("selectors", "request_arguments", "axis_sizes", "resolved_arguments", "construction_context"):
            value = _freeze_stable_value(getattr(self, name), path=f"replacement.{name}")
            object.__setattr__(self, name, value)

    @property
    def primary_fqn(self) -> str:
        """Return the canonical path of this identity boundary."""
        return self.aliases[0]

    def to_legacy_dict(self) -> dict[str, Any]:
        """Project into the temporary snapshot format consumed by the emitter."""
        record = {
            "rule_id": self.declaration_id,
            "module_fqns": list(self.aliases),
            "primary_fqn": self.primary_fqn,
            "source_type": self.source_type.legacy_path(),
            "replacement_type": self.replacement_type.legacy_path(),
            "weight_transforms": [item.transform_type.legacy_path() for item in self.weight_conversions],
            "opaque_legacy": self.request_target is None,
        }
        if self.request_target is not None:
            record["family_request"] = {
                "factory": self.request_target.legacy_path(),
                "kwargs": _stable_value_to_data(self.request_arguments),
                "context": _stable_value_to_data(self.axis_sizes),
            }
            if self.resolved_variant is not None:
                record["family_request"]["selection"] = {
                    "target": self.request_target.legacy_path(),
                    "version": PREPARED_FAMILY_RECORD_VERSION,
                    "variant": self.resolved_variant,
                    "member": self.resolved_member.legacy_path(),
                    "selectors": _stable_value_to_data(self.selectors),
                    **_stable_value_to_data(self._prepared_arguments),
                }
        if self.replay:
            record["replay"] = _stable_value_to_data(self.replay)
        return record

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        return {
            "boundary_id": self.boundary_id,
            "declaration_id": self.declaration_id,
            "aliases": list(self.aliases),
            "source_type": self.source_type.to_dict(),
            "replacement_type": self.replacement_type.to_dict(),
            "request_target": None if self.request_target is None else self.request_target.to_dict(),
            "resolved_member": None if self.resolved_member is None else self.resolved_member.to_dict(),
            "resolved_variant": self.resolved_variant,
            "resolved_arguments": _stable_value_to_data(self.resolved_arguments),
            "construction_context": _stable_value_to_data(self.construction_context),
            "selectors": _stable_value_to_data(self.selectors),
            "request_arguments": _stable_value_to_data(self.request_arguments),
            "axis_sizes": _stable_value_to_data(self.axis_sizes),
            "weight_conversions": [item.to_dict() for item in self.weight_conversions],
        }


@dataclass(frozen=True)
class PreparedComponent:
    """Final component type, family choice, aliases, and emission ownership."""

    boundary_id: str
    parent_boundary_id: str | None
    emission_owner_boundary_id: str
    origin: Literal["direct", "replacement", "native"]
    aliases: tuple[str, ...]
    request_target: SymbolRef | None = None
    resolved_member: SymbolRef | None = None
    selectors: Mapping[str, Any] = field(default_factory=dict)
    declaration_id: str | None = None
    source_type: SymbolRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "selectors", _freeze_stable_value(self.selectors, path="component.selectors"))

    @property
    def primary_fqn(self) -> str:
        """Return the canonical path of this identity boundary."""
        return self.aliases[0]

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        return {
            "boundary_id": self.boundary_id,
            "parent_boundary_id": self.parent_boundary_id,
            "emission_owner_boundary_id": self.emission_owner_boundary_id,
            "origin": self.origin,
            "aliases": list(self.aliases),
            "request_target": None if self.request_target is None else self.request_target.to_dict(),
            "resolved_member": None if self.resolved_member is None else self.resolved_member.to_dict(),
            "selectors": _stable_value_to_data(self.selectors),
            "declaration_id": self.declaration_id,
            "source_type": None if self.source_type is None else self.source_type.to_dict(),
        }


@dataclass(frozen=True)
class NativeBindingRecord:
    """Versioned resource-binding declaration from the final shared plan."""

    boundary_id: str
    aliases: tuple[str, ...]
    module_type: SymbolRef
    protocol: str
    version: str
    binding_kind: str
    stages: Mapping[str, tuple[str, ...]]
    options: Mapping[str, Any]
    layout_owner_boundary_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", _freeze_stable_value(self.stages, path="binding.stages"))
        object.__setattr__(self, "options", _freeze_stable_value(self.options, path="binding.options"))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        return {
            "boundary_id": self.boundary_id,
            "aliases": list(self.aliases),
            "module_type": self.module_type.to_dict(),
            "protocol": self.protocol,
            "version": self.version,
            "binding_kind": self.binding_kind,
            "stages": _stable_value_to_data(self.stages),
            "options": _stable_value_to_data(self.options),
            "layout_owner_boundary_id": self.layout_owner_boundary_id,
        }


@dataclass(frozen=True)
class PreparationManifest:
    """Immutable output of shared preparation, excluding models and runtime resources."""

    model_spec_id: str
    declarations: tuple[Mapping[str, Any], ...] = ()
    components: tuple[PreparedComponent, ...] = ()
    replacements: tuple[ReplacementApplication, ...] = ()
    native_bindings: tuple[NativeBindingRecord, ...] = ()
    weight_conversions: tuple[WeightConversionRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "declarations",
            tuple(_freeze_stable_value(item, path="declaration") for item in self.declarations),
        )

    def to_legacy_records(self) -> dict[str, list[dict[str, Any]]]:
        """Return the existing emitter's declaration/application representation."""
        return {
            "declarations": [_stable_value_to_data(item) for item in self.declarations],
            "applications": [item.to_legacy_dict() for item in self.replacements],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical resource-free representation."""
        return {
            "model_spec_id": self.model_spec_id,
            "declarations": [_stable_value_to_data(item) for item in self.declarations],
            "components": [item.to_dict() for item in self.components],
            "replacements": [item.to_dict() for item in self.replacements],
            "native_bindings": [item.to_dict() for item in self.native_bindings],
            "weight_conversions": [item.to_dict() for item in self.weight_conversions],
        }


_ACTIVE_APPLICATIONS: ContextVar[list[ReplacementApplication] | None] = ContextVar(
    "hyper_parallel_preparation_applications",
    default=None,
)
_CAPTURE_MANIFEST: ContextVar[bool] = ContextVar("hyper_parallel_capture_preparation_manifest", default=False)


@contextmanager
def capture_preparation_manifest() -> Iterator[None]:
    """Opt in to collecting stable preparation metadata for builds in this context."""
    token = _CAPTURE_MANIFEST.set(True)
    try:
        yield
    finally:
        _CAPTURE_MANIFEST.reset(token)


def preparation_application_capture() -> AbstractContextManager[list[ReplacementApplication] | None]:
    """Use an isolated application collector when a manifest was requested."""
    return collect_replacement_applications() if _CAPTURE_MANIFEST.get() else nullcontext()


@contextmanager
def collect_replacement_applications() -> Iterator[list[ReplacementApplication]]:
    """Collect applications committed by one shared preparation invocation."""
    applications: list[ReplacementApplication] = []
    token = _ACTIVE_APPLICATIONS.set(applications)
    try:
        yield applications
    finally:
        _ACTIVE_APPLICATIONS.reset(token)


def record_replacement_applications(applications: Sequence[ReplacementApplication]) -> None:
    """Commit records only when the shared installer has completed successfully."""
    target = _ACTIVE_APPLICATIONS.get()
    if target is not None:
        target.extend(applications)


def replacement_application_capture(
    plan: ModuleReplacementPlan,
    context: Mapping[str, Any],
    resolve_rule_id: Callable[[ModuleReplacementSpec, int], str],
) -> Callable[[Iterable[tuple[ModuleReplacementTarget, nn.Module, list]]], tuple[ReplacementApplication, ...]] | None:
    """Set up batch observation only within the opt-in preparation capture."""
    if _ACTIVE_APPLICATIONS.get() is None:
        return None
    rule_ids = {id(spec): resolve_rule_id(spec, index) for index, spec in enumerate(plan.specs)}
    for target in plan.targets:
        identity = id(target.spec)
        if identity not in rule_ids:
            rule_ids[identity] = resolve_rule_id(target.spec, len(rule_ids))

    def describe_batch(
        prepared: Iterable[tuple[ModuleReplacementTarget, nn.Module, list]],
    ) -> tuple[ReplacementApplication, ...]:
        return _describe_prepared_applications(prepared, rule_ids, context)

    return describe_batch


def _describe_prepared_applications(
    prepared: Iterable[tuple[ModuleReplacementTarget, nn.Module, list]],
    rule_ids: Mapping[int, str],
    context: Mapping[str, Any],
) -> tuple[ReplacementApplication, ...]:
    """Describe one candidate batch before successful installation releases its sources."""
    records = []
    for target, candidate, transforms in prepared:
        declaration_id = rule_ids[id(target.spec)]
        family = get_family_build_info(candidate)
        request_target = resolved_member = resolved_variant = None
        selectors = request_arguments = resolved_arguments = axis_sizes = construction_context = {}
        if family is not None and family.origin == "replacement":
            request_target = SymbolRef.from_object(family.request.target, path=f"replacement {declaration_id} request")
            resolved_member = SymbolRef.from_object(
                family.resolved.constructor,
                path=f"replacement {declaration_id} member",
            )
            selectors = family.resolved.selectors
            request_arguments = family.request.kwargs
            resolved_variant = family.resolved.variant
            resolved_arguments = family.resolved.arguments
            axis_sizes = {
                f"{axis}_size": size for axis, size in family.resolved.axis_sizes.items()
            }
            construction_context = {
                **axis_sizes,
                **{axis: size > 1 for axis, size in family.resolved.axis_sizes.items()},
                **{name: context[name] for name in (
                    "sequence_parallel", "loss_parallel", "mesh_dim_names", "device_type", "dtype",
                ) if name in context},
            }
        records.append(ReplacementApplication(
            boundary_id=f"replacement:{declaration_id}:{target.module_fqns[0] or '<root>'}",
            declaration_id=declaration_id,
            aliases=target.module_fqns,
            source_type=SymbolRef.from_object(type(target.source), path=f"replacement {declaration_id} source"),
            replacement_type=SymbolRef.from_object(type(candidate), path=f"replacement {declaration_id} candidate"),
            request_target=request_target,
            resolved_member=resolved_member,
            selectors=selectors,
            request_arguments=request_arguments,
            axis_sizes=axis_sizes,
            weight_conversions=tuple(WeightConversionRecord.from_transform(item) for item in transforms),
            resolved_variant=resolved_variant,
            resolved_arguments=resolved_arguments,
            construction_context=construction_context,
            replay=_replacement_replay_call(target.spec, context, resolved_member),
        ))
    return tuple(records)


def _replacement_replay_call(spec, context, member: SymbolRef | None) -> dict[str, Any]:
    """Add the executable constructor and arguments to the captured recipe."""
    replay = _replacement_replay(spec, context)
    if not replay:
        return replay
    arguments = {}
    if member is not None:
        constructor = member.legacy_path()
    elif spec.request is not None:
        factory, arguments = resolve_module_family_target(spec.request.target, spec.request.kwargs)
        constructor = SymbolRef.from_object(factory).legacy_path()
    else:
        constructor = replay["target"]
    return {**replay, "constructor": constructor, "constructor_arguments": _encode_construction_value(arguments)}


def _replacement_replay(spec: ModuleReplacementSpec, context: Mapping[str, Any]) -> dict[str, Any]:
    """Capture the actual configured call for a preinstalled constructor."""
    request = spec.request
    if request is not None:
        target = SymbolRef.from_object(request.target).legacy_path()
        arguments = request.kwargs
    else:
        try:
            target = SymbolRef.from_object(spec.factory).legacy_path()
        except (TypeError, ValueError):
            return {}
        arguments = {}
    logical_context = {
        name: context[name]
        for name in ("tp", "cp", "ep", "pp", "tp_size", "cp_size", "ep_size", "pp_size",
                     "sequence_parallel", "loss_parallel", "mesh_dim_names", "device_type", "dtype")
        if name in context
    }
    return {
        "version": PREPARED_FAMILY_RECORD_VERSION,
        "target": target,
        "arguments": _encode_construction_value(arguments),
        "context": _encode_construction_value(logical_context),
        "request": request is not None,
    }


def _boundary_id(kind: str, primary_fqn: str, identity: str) -> str:
    return f"{kind}:{identity}:{primary_fqn or '<root>'}"


def _contains_fqn(alias: str, fqn: str) -> bool:
    """Return whether an alias denotes the same module or an ancestor boundary."""
    return alias == "" or fqn == alias or fqn.startswith(f"{alias}.")


def _nearest_parent(component: PreparedComponent, components: Sequence[PreparedComponent]) -> PreparedComponent | None:
    candidates = []
    for candidate in components:
        if candidate is component:
            continue
        for alias in candidate.aliases:
            if alias != component.primary_fqn and _contains_fqn(alias, component.primary_fqn):
                candidates.append((len(alias), candidate))
    return max(candidates, key=lambda item: item[0], default=(0, None))[1]


def _with_component_ownership(components: Sequence[PreparedComponent]) -> tuple[PreparedComponent, ...]:
    result = []
    for component in components:
        parent = _nearest_parent(component, components)
        owner = component
        ancestor = parent
        while component.declaration_id is None and ancestor is not None:
            if ancestor.origin == "replacement":
                owner = ancestor
                break
            ancestor = _nearest_parent(ancestor, components)
        result.append(replace(
            component,
            parent_boundary_id=None if parent is None else parent.boundary_id,
            emission_owner_boundary_id=owner.boundary_id,
        ))
    return tuple(result)


def build_preparation_manifest(
    model: nn.Module,
    *,
    model_spec: Any,
    declarations: Sequence[Mapping[str, Any]],
    replacements: Sequence[ReplacementApplication],
    weight_conversions: Sequence[Any],
) -> PreparationManifest:
    """Build the stable manifest from decisions already made by shared preparation."""
    application_by_alias = {
        alias: application for application in replacements for alias in application.aliases
    }
    components: list[PreparedComponent] = []
    family_aliases = set()
    for boundary in collect_family_modules(model):
        info = boundary.info
        request_target = SymbolRef.from_object(info.request.target, path="prepared family request")
        aliases = tuple(sorted(boundary.fqns))
        family_aliases.update(aliases)
        application = next((application_by_alias[alias] for alias in aliases if alias in application_by_alias), None)
        declaration_id = None if application is None else application.declaration_id
        source_type = None if application is None else application.source_type
        boundary_id = (
            _boundary_id("family", aliases[0], request_target.legacy_path())
            if application is None
            else application.boundary_id
        )
        components.append(PreparedComponent(
            boundary_id=boundary_id,
            parent_boundary_id=None,
            emission_owner_boundary_id=boundary_id,
            origin=info.origin if application is None else "replacement",
            aliases=aliases,
            request_target=request_target,
            resolved_member=SymbolRef.from_object(info.resolved.constructor, path="prepared family member"),
            selectors=info.resolved.selectors,
            declaration_id=declaration_id,
            source_type=source_type,
        ))
    for application in replacements:
        if any(alias in family_aliases for alias in application.aliases):
            continue
        boundary_id = application.boundary_id
        components.append(PreparedComponent(
            boundary_id=boundary_id,
            parent_boundary_id=None,
            emission_owner_boundary_id=boundary_id,
            origin="replacement",
            aliases=application.aliases,
            declaration_id=application.declaration_id,
            source_type=application.source_type,
        ))
    components.sort(key=lambda item: (item.primary_fqn.count("."), item.primary_fqn, item.boundary_id))
    model_spec_id = "" if model_spec is None else f"{model_spec.model_type}:{model_spec.architecture}"
    manifest = PreparationManifest(
        model_spec_id=model_spec_id,
        declarations=tuple(declarations),
        components=_with_component_ownership(components),
        replacements=tuple(replacements),
        weight_conversions=tuple(WeightConversionRecord.from_transform(item) for item in weight_conversions),
    )
    model._hp_preparation_manifest = manifest  # pylint: disable=protected-access
    return manifest


def record_native_bindings(model: nn.Module, plan: Any) -> PreparationManifest:
    """Enrich an existing preparation manifest with binding decisions from the shared plan."""
    manifest = get_preparation_manifest(model)
    bindings = []
    components = list(manifest.components)
    aliases = dict(model.named_modules(remove_duplicate=False))
    for fqn, parallel in sorted(getattr(plan, "parallel_modules", {}).items()):
        module = aliases[fqn]
        owner = max(
            (
                component for component in components
                if any(_contains_fqn(alias, fqn) for alias in component.aliases)
            ),
            key=lambda item: len(item.primary_fqn),
            default=None,
        )
        boundary_id = _boundary_id("native", fqn, f"{parallel.protocol}@{parallel.version}")
        if owner is None:
            owner = PreparedComponent(
                boundary_id=boundary_id,
                parent_boundary_id=None,
                emission_owner_boundary_id=boundary_id,
                origin="native",
                aliases=(fqn,),
            )
            components.append(owner)
        bindings.append(NativeBindingRecord(
            boundary_id=boundary_id,
            aliases=(fqn,),
            module_type=SymbolRef.from_object(type(module), path=f"native binding {fqn}"),
            protocol=parallel.protocol,
            version=parallel.version,
            binding_kind=parallel.binding_kind,
            stages=parallel.stages,
            options=parallel.options,
            layout_owner_boundary_id=owner.boundary_id,
        ))
    enriched = replace(
        manifest,
        components=_with_component_ownership(components),
        native_bindings=tuple(bindings),
    )
    model._hp_preparation_manifest = enriched  # pylint: disable=protected-access
    return enriched


def get_preparation_manifest(model: nn.Module) -> PreparationManifest:
    """Return the immutable metadata attached by an opted-in preparation build."""
    manifest = getattr(model, "_hp_preparation_manifest", None)
    if not isinstance(manifest, PreparationManifest):
        raise RuntimeError("model components have not produced a PreparationManifest")
    return manifest


__all__ = [
    "NativeBindingRecord",
    "PartialRef",
    "PreparationManifest",
    "PreparedComponent",
    "ReplacementApplication",
    "SymbolRef",
    "WeightConversionRecord",
    "capture_preparation_manifest",
    "get_preparation_manifest",
]
