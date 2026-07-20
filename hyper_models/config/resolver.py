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

import importlib
import inspect
import types
from collections.abc import Mapping
from dataclasses import MISSING, fields
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from hyper_models.trainer.config import TrainerConfig


class ConfigResolutionError(ValueError):
    """A target or typed configuration value is invalid."""


def _fail(path: str, message: str) -> ConfigResolutionError:
    return ConfigResolutionError(f"{path}: {message}")


def import_target(target_path: str, *, path: str) -> object:
    """Import a dotted callable, including nested classes such as ``X.Config``."""

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

    if annotation is type(None) or (
        _is_union(annotation) and type(None) in get_args(annotation)
    ):
        return None
    raise _fail(path, f"expected {_type_name(annotation)}, got None")


def _coerce_union(value: object, annotation: object, *, path: str) -> object:
    """Normalize a value against an ``Optional`` or general union."""

    members = get_args(annotation)
    non_none_members = tuple(member for member in members if member is not type(None))
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

    # PyYAML 1.1 parses an unquoted ``off`` scalar as ``False``.
    if value is False and "off" in choices:
        return "off"
    if any(type(value) is type(choice) and value == choice for choice in choices):
        return value
    expected = ", ".join(repr(choice) for choice in choices)
    raise _fail(path, f"expected one of ({expected}), got {value!r}")


def coerce_value(value: object, annotation: object, *, path: str) -> object:
    """Validate and normalize one target argument or typed CLI override."""

    if annotation in (Any, object):
        return value
    if annotation is inspect.Signature.empty:
        raise _fail(path, "target parameter has no type annotation")
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
    if isinstance(annotation, type):
        if isinstance(value, annotation):
            return value
        raise _fail(
            path,
            f"expected {_type_name(annotation)}, got {type(value).__name__}",
        )
    raise _fail(path, f"unsupported type annotation {_type_name(annotation)}")


def _annotation_assignable(source: object, expected: object) -> bool:
    """Return whether a target result annotation fits a root field type."""

    if expected in (Any, object):
        return True
    if source is Any:
        return False
    if _is_union(expected):
        return any(_annotation_assignable(source, item) for item in get_args(expected))
    if _is_union(source):
        return all(_annotation_assignable(item, expected) for item in get_args(source))
    source_origin = get_origin(source) or source
    expected_origin = get_origin(expected) or expected
    if isinstance(source_origin, type) and isinstance(expected_origin, type):
        return issubclass(source_origin, expected_origin)
    return source == expected


def resolve_component(node: object, *, expected_type: object, path: str) -> object:
    """Resolve one top-level YAML group to its declared component type."""

    if not isinstance(node, Mapping):
        raise _fail(path, "component group must be a YAML mapping")
    if "_target_" not in node:
        raise _fail(path, "component group is missing required _target_")

    target = import_target(node["_target_"], path=f"{path}._target_")
    try:
        hints = get_type_hints(target)
    except (NameError, TypeError) as exc:
        raise _fail(path, f"could not resolve target type annotations: {exc}") from exc

    if inspect.isclass(target):
        result_type = target
    else:
        result_type = hints.get("return", inspect.Signature.empty)
        if result_type is inspect.Signature.empty:
            raise _fail(path, "factory target must declare a return annotation")

    if not _annotation_assignable(result_type, expected_type):
        raise _fail(
            path,
            f"target returns {_type_name(result_type)}, expected {_type_name(expected_type)}",
        )

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise _fail(path, f"target signature is unavailable: {exc}") from exc

    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise _fail(
                path,
                f"target parameter {parameter.name!r} must have an explicit name",
            )

    raw_args = {key: value for key, value in node.items() if key != "_target_"}
    try:
        bound = signature.bind(**raw_args)
    except TypeError as exc:
        raise _fail(path, f"target arguments are invalid: {exc}") from exc

    normalized_args = {}
    for name, value in bound.arguments.items():
        parameter = signature.parameters[name]
        annotation = hints.get(name, parameter.annotation)
        normalized_args[name] = coerce_value(
            value,
            annotation,
            path=f"{path}.{name}",
        )

    try:
        result = target(**normalized_args)
    except Exception as exc:
        raise _fail(path, f"target construction failed: {exc}") from exc

    coerce_value(result, result_type, path=path)
    return result


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
