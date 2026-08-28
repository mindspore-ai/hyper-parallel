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

import dataclasses
import importlib
import inspect
import types
from collections.abc import Mapping
from dataclasses import MISSING, fields
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from hyper_parallel.auto_models.trainer.config import (
    DataLoaderConfig,
    DatasetConfig,
    ModelAssetsConfig,
    Target,
    TrainerConfig,
)


class ConfigResolutionError(ValueError):
    """A target or typed configuration value is invalid."""


def _fail(path: str, message: str) -> ConfigResolutionError:
    return ConfigResolutionError(f"{path}: {message}")


def import_target(target_path: str, *, path: str) -> object:
    """Import a dotted callable, including nested callable attributes."""

    if not isinstance(target_path, str) or not target_path.strip():
        raise _fail(path, "_target_ must be a non-empty dotted path")

    parts = target_path.split(".")
    if any(not part for part in parts):
        raise _fail(path, f"invalid target path {target_path!r}")

    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            target = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                continue
            raise _fail(
                path,
                f"target {target_path!r} failed while importing dependency {exc.name!r}",
            ) from exc
        except ImportError as exc:
            raise _fail(path, f"target {target_path!r} could not be imported: {exc}") from exc

        for attribute in parts[split_at:]:
            if not hasattr(target, attribute):
                raise _fail(
                    path,
                    f"target {target_path!r} has no attribute {attribute!r}",
                )
            target = getattr(target, attribute)

        if not callable(target):
            raise _fail(path, f"target {target_path!r} is not callable")
        return target

    raise _fail(path, f"target {target_path!r} could not be imported")


def _is_union(annotation: object) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)


def _type_name(annotation: object) -> str:
    if annotation is Any:
        return "Any"
    if isinstance(annotation, type):
        return annotation.__qualname__
    return str(annotation).replace("typing.", "")


def _normalize_list(value: object, item_type: object, *, path: str) -> list:
    if not isinstance(value, (list, tuple)):
        raise _fail(path, f"expected list, got {type(value).__name__}")
    return [
        coerce_value(item, item_type, path=f"{path}[{index}]")
        for index, item in enumerate(value)
    ]


def _normalize_tuple(value: object, item_types: tuple, *, path: str) -> tuple:
    """Validate a tuple value and recursively normalize its items."""

    if not isinstance(value, (list, tuple)):
        raise _fail(path, f"expected tuple, got {type(value).__name__}")
    if not item_types:
        return tuple(value)
    if len(item_types) == 2 and item_types[1] is Ellipsis:
        return tuple(
            coerce_value(item, item_types[0], path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if len(value) != len(item_types):
        raise _fail(
            path,
            f"expected tuple of length {len(item_types)}, got {len(value)}",
        )
    return tuple(
        coerce_value(item, item_type, path=f"{path}[{index}]")
        for index, (item, item_type) in enumerate(zip(value, item_types))
    )


def _coerce_none(annotation: object, *, path: str) -> None:
    """Accept ``None`` only when the annotation permits it."""

    if annotation is types.NoneType or (
        _is_union(annotation) and types.NoneType in get_args(annotation)
    ):
        return None
    raise _fail(path, f"expected {_type_name(annotation)}, got None")


def _coerce_union(value: object, annotation: object, *, path: str) -> object:
    """Normalize a value against an ``Optional`` or general union."""

    members = get_args(annotation)
    non_none_members = tuple(member for member in members if member is not types.NoneType)
    if len(non_none_members) == 1 and len(non_none_members) != len(members):
        return coerce_value(value, non_none_members[0], path=path)

    for member in members:
        try:
            return coerce_value(value, member, path=path)
        except ConfigResolutionError:
            continue
    raise _fail(
        path,
        f"expected {_type_name(annotation)}, got {type(value).__name__}",
    )


def _coerce_scalar(value: object, annotation: object, *, path: str) -> object:
    """Normalize one strict bool, integer, float, or string value."""

    if annotation is bool and isinstance(value, bool):
        return value
    if annotation is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if annotation is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if annotation is str and isinstance(value, str):
        return value
    raise _fail(
        path,
        f"expected {_type_name(annotation)}, got {type(value).__name__}",
    )


def _coerce_literal(value: object, choices: tuple, *, path: str) -> object:
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
    raise _fail(path, f"expected one of ({expected}), got {value!r}")


def coerce_value(value: object, annotation: object, *, path: str) -> object:
    """Validate and normalize one target argument or typed CLI override."""

    if annotation in (Any, object):
        return value
    if isinstance(annotation, dataclasses.InitVar):
        return coerce_value(value, annotation.type, path=path)
    if value is None:
        return _coerce_none(annotation, path=path)
    if _is_union(annotation):
        return _coerce_union(value, annotation, path=path)
    if annotation in (bool, int, float, str):
        return _coerce_scalar(value, annotation, path=path)

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return _coerce_literal(value, args, path=path)
    if origin is list or annotation is list:
        return _normalize_list(
            value,
            args[0] if args else Any,
            path=path,
        )
    if origin is tuple or annotation is tuple:
        return _normalize_tuple(value, args, path=path)
    if origin in (dict, Mapping) or annotation in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise _fail(path, f"expected mapping, got {type(value).__name__}")
        return dict(value)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        # Nested dataclass items resolve from mappings in the same way as
        # top-level dataclass components.
        return _resolve_dataclass(value, annotation, path=path)
    if isinstance(annotation, type):
        if isinstance(value, annotation):
            return value
        raise _fail(
            path,
            f"expected {_type_name(annotation)}, got {type(value).__name__}",
        )
    raise _fail(path, f"unsupported type annotation {_type_name(annotation)}")


def _resolve_union(node: object, annotation: object, *, path: str) -> object:
    """Resolve one value against an Optional or general union."""
    non_none_members = [
        member for member in get_args(annotation) if member is not types.NoneType
    ]
    if len(non_none_members) == 1:
        # Single-member union (e.g. Optional[Target]): resolve directly so the
        # member's specific error (e.g. an unexpected target argument) reaches
        # the user instead of being swallowed by the generic union failure.
        return resolve_component(node, expected_type=non_none_members[0], path=path)

    last_error: ConfigResolutionError | None = None
    for member in non_none_members:
        try:
            return resolve_component(node, expected_type=member, path=path)
        except ConfigResolutionError as exc:
            last_error = exc
    raise _fail(
        path,
        f"expected {_type_name(annotation)}, got {type(node).__name__}",
    ) from last_error


def _resolve_dataclass(node: object, config_type: type, *, path: str) -> object:
    """Construct one pure-parameter dataclass from a YAML mapping."""
    if not isinstance(node, Mapping):
        raise _fail(path, "configuration section must be a YAML mapping")

    config_fields = {field.name: field for field in fields(config_type)}
    unknown = sorted(set(node) - set(config_fields))
    if unknown:
        raise _fail(path, f"unknown configuration fields: {unknown}")

    missing = [
        field.name
        for field in config_fields.values()
        if field.name not in node
        and field.default is MISSING
        and field.default_factory is MISSING
    ]
    if missing:
        raise _fail(path, f"missing required configuration fields: {missing}")

    try:
        hints = get_type_hints(config_type)
    except (NameError, TypeError) as exc:
        raise _fail(path, f"could not resolve configuration type annotations: {exc}") from exc

    resolved = {
        name: resolve_component(
            value,
            expected_type=hints[name],
            path=f"{path}.{name}",
        )
        for name, value in node.items()
    }
    try:
        return config_type(**resolved)
    except TypeError as exc:
        raise _fail(path, f"could not construct {config_type.__name__}: {exc}") from exc


def _target_hints(target: object, *, path: str) -> dict[str, object]:
    """Resolve annotations for a target function or class constructor."""
    hint_source = target.__init__ if inspect.isclass(target) else target
    try:
        return get_type_hints(hint_source)
    except (NameError, TypeError) as exc:
        raise _fail(path, f"could not resolve target type annotations: {exc}") from exc


def _normalize_target_args(
    raw_args: Mapping[str, object],
    signature: inspect.Signature,
    hints: Mapping[str, object],
    *,
    path: str,
) -> dict[str, object]:
    """Validate YAML target arguments and apply callable defaults."""
    try:
        signature.bind_partial(**raw_args)
    except TypeError as exc:
        raise _fail(path, f"target arguments are invalid: {exc}") from exc

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
            normalized[name] = coerce_value(
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
        raise _fail(path, "target section must be a YAML mapping")
    if "_target_" not in node:
        raise _fail(path, "target section is missing required _target_")

    target_path = node["_target_"]
    target = import_target(target_path, path=f"{path}._target_")
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise _fail(path, f"target signature is unavailable: {exc}") from exc

    for parameter in signature.parameters.values():
        if (
            parameter.kind is inspect.Parameter.POSITIONAL_ONLY
            and parameter.default is inspect.Signature.empty
        ):
            raise _fail(
                path,
                f"target parameter {parameter.name!r} must be callable by keyword",
            )

    raw_args = {key: value for key, value in node.items() if key != "_target_"}
    normalized_args = _normalize_target_args(
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
        raise _fail(path, "DataLoader configuration must be a YAML mapping")

    target_node = dict(node)
    collate_node = target_node.pop("collate_fn", None)
    get_batch_node = target_node.pop("get_batch", None)
    dataloader_type = coerce_value(
        target_node.pop("dataloader_type", "single"),
        Literal["single", "cyclic"],
        path=f"{path}.dataloader_type",
    )
    data_rearrange_map = target_node.pop("data_rearrange_map", None)
    data_sharding = coerce_value(
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
        raise _fail(path, "Dataset configuration must be a YAML mapping")

    target_node = dict(node)
    model_assets_node = target_node.pop("model_assets", {})
    data_transform_node = target_node.pop("data_transform", None)
    target = _resolve_target(target_node, path=path)
    model_assets = resolve_component(
        model_assets_node,
        expected_type=ModelAssetsConfig,
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


def resolve_component(node: object, *, expected_type: object, path: str) -> object:
    """Resolve one YAML value according to its declared configuration type."""
    if node is None:
        return _coerce_none(expected_type, path=path)
    if _is_union(expected_type):
        return _resolve_union(node, expected_type, path=path)

    origin = get_origin(expected_type)
    if origin is Target or expected_type is Target:
        return _resolve_target(node, path=path)
    if expected_type is DatasetConfig:
        return _resolve_dataset_config(node, path=path)
    if expected_type is DataLoaderConfig:
        return _resolve_dataloader_config(node, path=path)
    if isinstance(expected_type, type) and dataclasses.is_dataclass(expected_type):
        return _resolve_dataclass(node, expected_type, path=path)
    return coerce_value(node, expected_type, path=path)


def resolve_root(raw: object) -> TrainerConfig:
    """Resolve YAML root fields and construct ``TrainerConfig``."""
    if not isinstance(raw, Mapping):
        raise _fail("$", "YAML root must be a mapping")

    root_fields = {field.name: field for field in fields(TrainerConfig)}
    unknown = sorted(set(raw) - set(root_fields))
    if unknown:
        raise _fail("$", f"unknown configuration fields: {unknown}")

    missing = [
        field.name
        for field in root_fields.values()
        if field.name not in raw
        and field.default is MISSING
        and field.default_factory is MISSING
    ]
    if missing:
        raise _fail("$", f"missing required configuration fields: {missing}")

    root_hints = get_type_hints(TrainerConfig)
    resolved = {
        name: resolve_component(
            node,
            expected_type=root_hints[name],
            path=f"$.{name}",
        )
        for name, node in raw.items()
    }
    try:
        return TrainerConfig(**resolved)
    except TypeError as exc:
        raise _fail("$", f"could not construct TrainerConfig: {exc}") from exc


__all__ = [
    "ConfigResolutionError",
    "coerce_value",
    "import_target",
    "resolve_component",
    "resolve_root",
]
