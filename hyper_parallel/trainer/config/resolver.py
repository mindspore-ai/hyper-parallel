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
"""Resolve target-selected YAML groups into a typed trainer config tree."""

from __future__ import annotations

__all__ = [
    "ConfigResolutionError",
    "replace_override_path",
    "resolve_config",
]

import dataclasses
import difflib
import importlib
import inspect
import types
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, fields, is_dataclass, replace
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from hyper_parallel.trainer.config.data import (
    DataLoaderConfig,
    DatasetConfig,
    ModelAssetsConfig,
)
from hyper_parallel.trainer.config.optimization import OptimizerConfig
from hyper_parallel.trainer.config.target import Target
from hyper_parallel.trainer.config.trainer import TrainerConfig


class ConfigResolutionError(ValueError):
    """A target or typed configuration value is invalid."""

    def __init__(self, location: str, message: str) -> None:
        """Render ``location: message`` so every error names where it came from."""
        super().__init__(f"{location}: {message}")


def import_target(target_path: str, *, location: str) -> object:
    """Import a dotted callable, including nested callable attributes."""

    if not isinstance(target_path, str) or not target_path.strip():
        raise ConfigResolutionError(location, "_target_ must be a non-empty dotted path")

    parts = target_path.split(".")
    if any(not part for part in parts):
        raise ConfigResolutionError(location, f"invalid target path {target_path!r}")

    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            target = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                continue
            raise ConfigResolutionError(
                location,
                f"target {target_path!r} failed while importing dependency {exc.name!r}",
            ) from exc
        except ImportError as exc:
            raise ConfigResolutionError(location, f"target {target_path!r} could not be imported: {exc}") from exc

        for attribute in parts[split_at:]:
            if not hasattr(target, attribute):
                raise ConfigResolutionError(
                    location,
                    f"target {target_path!r} has no attribute {attribute!r}",
                )
            target = getattr(target, attribute)

        if not callable(target):
            raise ConfigResolutionError(location, f"target {target_path!r} is not callable")
        return target

    raise ConfigResolutionError(location, f"target {target_path!r} could not be imported")


def _is_union(annotation: object) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)


def _annotation_name(annotation: object) -> str:
    if annotation is Any:
        return "Any"
    if isinstance(annotation, type):
        return annotation.__qualname__
    return str(annotation).replace("typing.", "")


def _target_hints(target: object, *, path: str) -> dict[str, object]:
    """Resolve annotations for a target function or class constructor."""
    hint_source = target.__init__ if inspect.isclass(target) else target
    try:
        return get_type_hints(hint_source)
    except (NameError, TypeError) as exc:
        raise ConfigResolutionError(path, f"could not resolve target type annotations: {exc}") from exc


# Annotations whose payload is a container of other values.
_COLLECTION_TYPES: tuple[type, ...] = (list, tuple, dict, Mapping)


# ── Value normalization ─────────────────────────────────────────────────


def _normalize_sequence(
    value: object,
    item_types: tuple,
    *,
    uniform: bool,
    kind: str,
    path: str,
) -> list | tuple:
    """Normalize the items of a list or tuple against their declared item types.

    ``item_types`` holds one entry per element when ``uniform`` is false; when it
    is true a single entry applies to every element (``list[int]``,
    ``tuple[int, ...]``).  An empty ``item_types`` means a bare ``tuple``, which
    is passed through without normalizing its items.
    """
    if not isinstance(value, (list, tuple)):
        raise ConfigResolutionError(path, f"expected {kind}, got {type(value).__name__}")
    if not item_types:
        return tuple(value)
    if uniform:
        item_types = item_types * len(value)
    if len(item_types) != len(value):
        raise ConfigResolutionError(
            path,
            f"expected {kind} of length {len(item_types)}, got {len(value)}",
        )
    normalized = [
        normalize_value(item, item_type, path=f"{path}[{index}]")
        for index, (item, item_type) in enumerate(zip(value, item_types))
    ]
    return tuple(normalized) if kind == "tuple" else normalized


def _require_none_allowed(annotation: object, *, path: str) -> None:
    """Accept ``None`` only when the annotation permits it."""

    if annotation is types.NoneType or (
        _is_union(annotation) and types.NoneType in get_args(annotation)
    ):
        return None
    raise ConfigResolutionError(path, f"expected {_annotation_name(annotation)}, got None")


def _normalize_union(value: object, annotation: object, *, path: str) -> object:
    """Normalize a value against an ``Optional`` or general union."""

    members = get_args(annotation)
    non_none_members = tuple(member for member in members if member is not types.NoneType)
    if len(non_none_members) == 1 and len(non_none_members) != len(members):
        return normalize_value(value, non_none_members[0], path=path)

    for member in members:
        try:
            return normalize_value(value, member, path=path)
        except ConfigResolutionError:
            continue
    raise ConfigResolutionError(
        path,
        f"expected {_annotation_name(annotation)}, got {type(value).__name__}",
    )


def _normalize_scalar(value: object, annotation: object, *, path: str) -> object:
    """Normalize one strict bool, integer, float, or string value."""

    if annotation is bool and isinstance(value, bool):
        return value
    if annotation is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if annotation is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if annotation is str and isinstance(value, str):
        return value
    raise ConfigResolutionError(
        path,
        f"expected {_annotation_name(annotation)}, got {type(value).__name__}",
    )


def _normalize_literal(value: object, choices: tuple, *, path: str) -> object:
    """Validate a value against the exact choices of a ``Literal``."""

    # PyYAML 1.1 parses unquoted on/off/yes/no/true/false scalars as bool.
    # Map the bool back to the matching word when it is one of the choices.
    if isinstance(value, bool):
        words = ("on", "yes", "true") if value else ("off", "no", "false")
        for word in words:
            if word in choices:
                return word
    if any(type(value) is type(choice) and value == choice for choice in choices):
        return value
    expected = ", ".join(repr(choice) for choice in choices)
    raise ConfigResolutionError(path, f"expected one of ({expected}), got {value!r}")


def _normalize_collection(
    value: object,
    annotation: object,
    origin: object,
    args: tuple,
    *,
    path: str,
) -> object:
    """Normalize a list, tuple, or mapping annotation."""

    if origin is list or annotation is list:
        return _normalize_sequence(
            value,
            (args[0] if args else Any,),
            uniform=True,
            kind="list",
            path=path,
        )
    if origin is tuple or annotation is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return _normalize_sequence(value, args[:1], uniform=True, kind="tuple", path=path)
        return _normalize_sequence(value, args, uniform=False, kind="tuple", path=path)
    if not isinstance(value, Mapping):
        raise ConfigResolutionError(path, f"expected mapping, got {type(value).__name__}")
    return dict(value)


def _normalize_class(value: object, annotation: object, *, path: str) -> object:
    """Normalize a value against a concrete class annotation.

    Also the dispatcher's fallback: an annotation that is not a type at all is
    reported here as unsupported.
    """

    if not isinstance(annotation, type):
        raise ConfigResolutionError(path, f"unsupported type annotation {_annotation_name(annotation)}")
    if isinstance(value, annotation):
        return value
    raise ConfigResolutionError(
        path,
        f"expected {_annotation_name(annotation)}, got {type(value).__name__}",
    )


def _normalize_override_scalar(value: object, annotation: object) -> object:
    """Best-effort string-to-numeric conversion for CLI override scalars.

    PyYAML 1.1 only recognizes floats with a decimal point, so
    ``yaml.safe_load("1e-4")`` yields the string ``"1e-4"``. When the target
    annotation is ``int``/``float`` (optionally wrapped in ``Optional``), try a
    direct conversion before strict coercion; anything that does not parse is
    returned unchanged so ``normalize_value`` reports the original type error.
    """

    if not isinstance(value, str):
        return value
    target = annotation
    if get_origin(target) in (Union, types.UnionType):
        members = [member for member in get_args(target) if member is not types.NoneType]
        if len(members) != 1:
            return value
        target = members[0]
    if target is int:
        try:
            return int(value)
        except ValueError:
            return value
    if target is float:
        try:
            return float(value)
        except ValueError:
            return value
    return value


def normalize_value(value: object, annotation: object, *, path: str) -> object:
    """Validate and normalize one target argument or typed CLI override."""

    if annotation in (Any, object):
        return value
    if isinstance(annotation, dataclasses.InitVar):
        return normalize_value(value, annotation.type, path=path)
    if value is None:
        return _require_none_allowed(annotation, path=path)
    if _is_union(annotation):
        return _normalize_union(value, annotation, path=path)
    if annotation in (bool, int, float, str):
        return _normalize_scalar(value, annotation, path=path)

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return _normalize_literal(value, args, path=path)
    if origin in _COLLECTION_TYPES or annotation in _COLLECTION_TYPES:
        return _normalize_collection(value, annotation, origin, args, path=path)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        # Nested dataclass items resolve from mappings in the same way as
        # top-level dataclass components.
        return _resolve_dataclass(value, annotation, path=path)
    return _normalize_class(value, annotation, path=path)


# ── Dotted overrides ────────────────────────────────────────────────────


def _suggestion(name: str, candidates: Iterable[str]) -> str:
    """Build a ``did you mean`` hint for an unknown configuration name."""
    matches = difflib.get_close_matches(name, candidates, n=1)
    return f"; did you mean {matches[0]!r}?" if matches else ""


def _replace_target_path(
    config: Target[Any],
    parts: list[str],
    value: object,
    *,
    path: str,
) -> Target[Any]:
    """Return a target copy with one configured argument replaced."""
    name = parts[0]
    full_path = f"{path}.{name}" if path else name
    if name == "_target_":
        raise ConfigResolutionError(
            f"CLI.{full_path}", "changing _target_ through an override is not supported"
        )

    signature = inspect.signature(config.callable)
    parameter = signature.parameters.get(name)
    has_var_kwargs = any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )
    if parameter is None and not has_var_kwargs:
        candidates = [
            item.name
            for item in signature.parameters.values()
            if item.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        raise ConfigResolutionError(
            f"CLI.{full_path}",
            f"unknown target argument {name!r}{_suggestion(name, candidates)}",
        )

    if len(parts) > 1:
        try:
            child = getattr(config, name)
        except AttributeError as exc:
            raise ConfigResolutionError(
                f"CLI.{full_path}", "target argument is not configured"
            ) from exc
        return config.replace(**{name: replace_override_path(child, parts[1:], value, path=full_path)})

    normalized = value
    if parameter is not None:
        annotation = _target_hints(config.callable, path=path).get(
            name,
            parameter.annotation,
        )
        if annotation not in (Any, object, inspect.Signature.empty):
            normalized = normalize_value(
                _normalize_override_scalar(value, annotation),
                annotation,
                path=f"CLI.{full_path}",
            )
    return config.replace(**{name: normalized})


def replace_override_path(config: object, parts: list[str], value: object, *, path: str) -> object:
    """Return a config copy with one typed dotted-path value replaced."""

    if isinstance(config, Target):
        return _replace_target_path(config, parts, value, path=path)

    if isinstance(config, Mapping):
        name = parts[0]
        full_path = f"{path}.{name}" if path else name
        if name not in config:
            raise ConfigResolutionError(f"CLI.{path}", f"unknown mapping key {name!r}")
        if len(parts) == 1:
            return {**config, name: value}
        return {**config, name: replace_override_path(config[name], parts[1:], value, path=full_path)}

    location = f"CLI.{path}" if path else "CLI"
    if not is_dataclass(config):
        raise ConfigResolutionError(
            location, f"value of type {type(config).__name__} has no configurable fields"
        )

    config_fields = {field.name: field for field in fields(config)}
    name = parts[0]
    if name not in config_fields:
        target = getattr(config, "target", None)
        if isinstance(target, Target):
            return replace(config, target=_replace_target_path(target, parts, value, path=path))
        raise ConfigResolutionError(
            location,
            f"unknown field {name!r} on {type(config).__name__}"
            f"{_suggestion(name, config_fields)}",
        )

    full_path = f"{path}.{name}" if path else name
    if len(parts) == 1:
        annotation = get_type_hints(type(config))[name]
        normalized = normalize_value(
            _normalize_override_scalar(value, annotation), annotation, path=f"CLI.{full_path}"
        )
        return replace(config, **{name: normalized})

    child = getattr(config, name)
    if child is None:
        raise ConfigResolutionError(
            f"CLI.{full_path}", "component was not selected by the YAML"
        )
    return replace(config, **{name: replace_override_path(child, parts[1:], value, path=full_path)})


# ── Node resolution ─────────────────────────────────────────────────────


def _resolve_union(node: object, annotation: object, *, path: str) -> object:
    """Resolve one value against an Optional or general union."""
    non_none_members = [
        member for member in get_args(annotation) if member is not types.NoneType
    ]
    if len(non_none_members) == 1:
        # Single-member union (e.g. Optional[Target]): resolve directly so the
        # member's specific error (e.g. an unexpected target argument) reaches
        # the user instead of being swallowed by the generic union failure.
        return resolve_component(node, annotation=non_none_members[0], path=path)

    last_error: ConfigResolutionError | None = None
    for member in non_none_members:
        try:
            return resolve_component(node, annotation=member, path=path)
        except ConfigResolutionError as exc:
            last_error = exc
    raise ConfigResolutionError(
        path,
        f"expected {_annotation_name(annotation)}, got {type(node).__name__}",
    ) from last_error


def _resolve_dataclass(node: object, config_type: type, *, path: str) -> object:
    """Construct one pure-parameter dataclass from a YAML mapping."""
    if not isinstance(node, Mapping):
        raise ConfigResolutionError(path, "configuration section must be a YAML mapping")

    config_fields = {field.name: field for field in fields(config_type)}
    unknown = sorted(set(node) - set(config_fields))
    if unknown:
        raise ConfigResolutionError(path, f"unknown configuration fields: {unknown}")

    required = [
        field.name
        for field in config_fields.values()
        if field.default is MISSING and field.default_factory is MISSING
    ]
    missing = [name for name in required if name not in node]
    if missing:
        raise ConfigResolutionError(path, f"missing required configuration fields: {missing}")

    try:
        hints = get_type_hints(config_type)
    except (NameError, TypeError) as exc:
        raise ConfigResolutionError(path, f"could not resolve configuration type annotations: {exc}") from exc

    resolved = {
        name: resolve_component(
            value,
            annotation=hints[name],
            path=f"{path}.{name}",
        )
        for name, value in node.items()
    }
    try:
        return config_type(**resolved)
    except TypeError as exc:
        raise ConfigResolutionError(path, f"could not construct {config_type.__name__}: {exc}") from exc


def _resolve_target_args(
    raw_args: Mapping[str, object],
    signature: inspect.Signature,
    hints: Mapping[str, object],
    *,
    path: str,
) -> dict[str, object]:
    """Validate YAML target arguments and apply callable defaults.

    Positional-only parameters are skipped here; a target that *requires* one is
    rejected earlier by ``_resolve_target``, which fails fast with a clearer message.
    """
    try:
        signature.bind_partial(**raw_args)
    except TypeError as exc:
        raise ConfigResolutionError(path, f"target arguments are invalid: {exc}") from exc

    normalized = {}
    for name, value in raw_args.items():
        parameter = signature.parameters.get(name)
        if parameter is None:
            normalized[name] = value
            continue

        annotation = hints.get(name, parameter.annotation)
        if annotation in (Any, object, inspect.Signature.empty):
            normalized[name] = value
        else:
            normalized[name] = normalize_value(
                value,
                annotation,
                path=f"{path}.{name}",
            )

    for name, parameter in signature.parameters.items():
        if (
            name in normalized
            or parameter.default is inspect.Signature.empty
            or parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        ):
            continue
        normalized[name] = parameter.default

    return normalized


def _resolve_target(node: object, *, path: str) -> Target[Any]:
    """Resolve one YAML target without invoking its callable."""
    if not isinstance(node, Mapping):
        raise ConfigResolutionError(path, "target section must be a YAML mapping")
    if "_target_" not in node:
        raise ConfigResolutionError(path, "target section is missing required _target_")

    target_path = node["_target_"]
    target = import_target(target_path, location=f"{path}._target_")
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise ConfigResolutionError(path, f"target signature is unavailable: {exc}") from exc

    for parameter in signature.parameters.values():
        if (
            parameter.kind is inspect.Parameter.POSITIONAL_ONLY
            and parameter.default is inspect.Signature.empty
        ):
            raise ConfigResolutionError(
                path,
                f"target parameter {parameter.name!r} must be callable by keyword",
            )

    raw_args = {key: value for key, value in node.items() if key != "_target_"}
    normalized_args = _resolve_target_args(
        raw_args,
        signature,
        _target_hints(target, path=path),
        path=path,
    )
    return Target(
        target,
        target_path=target_path,
        **normalized_args,
    )


def _resolve_dataloader_config(node: object, *, path: str) -> DataLoaderConfig:
    """Resolve a DataLoader target with nested collator and batch adapter."""
    if not isinstance(node, Mapping):
        raise ConfigResolutionError(path, "DataLoader configuration must be a YAML mapping")

    target_node = dict(node)
    collate_node = target_node.pop("collate_fn", None)
    get_batch_node = target_node.pop("get_batch", None)
    dataloader_type = normalize_value(
        target_node.pop("dataloader_type", "single"),
        Literal["single", "cyclic"],
        path=f"{path}.dataloader_type",
    )
    data_rearrange_map = target_node.pop("data_rearrange_map", None)
    data_sharding = normalize_value(
        target_node.pop("data_sharding", False),
        bool,
        path=f"{path}.data_sharding",
    )
    target = _resolve_target(target_node, path=path)
    collate_fn = (
        None
        if collate_node is None
        else _resolve_target(collate_node, path=f"{path}.collate_fn")
    )
    get_batch = (
        None
        if get_batch_node is None
        else _resolve_target(get_batch_node, path=f"{path}.get_batch")
    )
    return DataLoaderConfig(
        target=target,
        collate_fn=collate_fn,
        get_batch=get_batch,
        dataloader_type=dataloader_type,
        data_rearrange_map=data_rearrange_map,
        data_sharding=data_sharding,
    )


def _resolve_dataset_config(node: object, *, path: str) -> DatasetConfig:
    """Resolve a Dataset target with its assets and sample transform."""
    if not isinstance(node, Mapping):
        raise ConfigResolutionError(path, "Dataset configuration must be a YAML mapping")

    target_node = dict(node)
    model_assets_node = target_node.pop("model_assets", {})
    data_transform_node = target_node.pop("data_transform", None)
    target = _resolve_target(target_node, path=path)
    model_assets = resolve_component(
        model_assets_node,
        annotation=ModelAssetsConfig,
        path=f"{path}.model_assets",
    )
    data_transform = (
        None
        if data_transform_node is None
        else _resolve_target(
            data_transform_node,
            path=f"{path}.data_transform",
        )
    )
    return DatasetConfig(
        target=target,
        model_assets=model_assets,
        data_transform=data_transform,
    )


def _resolve_optimizer_config(node: object, *, path: str) -> OptimizerConfig:
    """Resolve an optimizer target and its fp32 main-parameter policy."""
    if not isinstance(node, Mapping):
        raise ConfigResolutionError(path, "Optimizer configuration must be a YAML mapping")

    target_node = dict(node)
    fp32_main_params = normalize_value(
        target_node.pop("fp32_main_params", False),
        bool,
        path=f"{path}.fp32_main_params",
    )
    return OptimizerConfig(
        target=_resolve_target(target_node, path=path),
        fp32_main_params=fp32_main_params,
    )


def resolve_component(node: object, *, annotation: object, path: str) -> object:
    """Resolve one YAML value according to its declared configuration type.

    Args:
        node: Raw YAML value to resolve.
        annotation: Declared configuration type for the value.
        path: Dotted YAML path used in validation errors.

    Returns:
        The resolved configuration value.
    """
    if node is None:
        return _require_none_allowed(annotation, path=path)
    if _is_union(annotation):
        return _resolve_union(node, annotation, path=path)

    origin = get_origin(annotation)
    if origin is Target or annotation is Target:
        return _resolve_target(node, path=path)
    if annotation is DatasetConfig:
        return _resolve_dataset_config(node, path=path)
    if annotation is DataLoaderConfig:
        return _resolve_dataloader_config(node, path=path)
    if annotation is OptimizerConfig:
        return _resolve_optimizer_config(node, path=path)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _resolve_dataclass(node, annotation, path=path)
    return normalize_value(node, annotation, path=path)


def resolve_config(raw: object) -> TrainerConfig:
    """Resolve YAML root fields and construct ``TrainerConfig``."""
    if not isinstance(raw, Mapping):
        raise ConfigResolutionError("$", "YAML root must be a mapping")

    root_fields = {field.name: field for field in fields(TrainerConfig)}
    unknown = sorted(set(raw) - set(root_fields))
    if unknown:
        raise ConfigResolutionError("$", f"unknown configuration fields: {unknown}")

    required = [
        field.name
        for field in root_fields.values()
        if field.default is MISSING and field.default_factory is MISSING
    ]
    missing = [name for name in required if name not in raw]
    if missing:
        raise ConfigResolutionError("$", f"missing required configuration fields: {missing}")

    root_hints = get_type_hints(TrainerConfig)
    resolved = {
        name: resolve_component(
            node,
            annotation=root_hints[name],
            path=f"$.{name}",
        )
        for name, node in raw.items()
    }
    try:
        return TrainerConfig(**resolved)
    except TypeError as exc:
        raise ConfigResolutionError("$", f"could not construct TrainerConfig: {exc}") from exc
