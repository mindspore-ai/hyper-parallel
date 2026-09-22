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


_NONE_TYPE = types.NoneType  # pylint: disable=no-member
_UNION_TYPE = types.UnionType  # pylint: disable=no-member


class ConfigResolutionError(ValueError):
    """A target or typed configuration value is invalid."""

    def __init__(self, location: str, message: str) -> None:
        """Prefix the error message with its configuration location."""
        super().__init__(f"{location}: {message}")


def import_target(target_path: str, *, location: str) -> object:
    """Import a callable from a dotted Python path.

    Args:
        target_path: Module path followed by callable attributes.
        location: Configuration location used in error messages.

    Returns:
        The imported callable, without invoking it.

    Raises:
        ConfigResolutionError: The path is invalid, cannot be imported, or
            does not identify a callable.
    """
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
    """Return whether the annotation is a ``Union`` or a PEP 604 union."""
    return get_origin(annotation) in (Union, _UNION_TYPE)


def _annotation_name(annotation: object) -> str:
    """Format an annotation for configuration error messages."""
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


def _normalize_sequence(
    value: object,
    item_types: tuple,
    *,
    uniform: bool,
    kind: str,
    path: str,
) -> list | tuple:
    """Normalize sequence items against their declared types.

    With ``uniform=True``, one item type applies to every element. Otherwise,
    ``item_types`` must contain one type per element. Empty ``item_types``
    produces a tuple without normalizing its items.
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
    if annotation is _NONE_TYPE or (
        _is_union(annotation) and _NONE_TYPE in get_args(annotation)
    ):
        return None
    raise ConfigResolutionError(path, f"expected {_annotation_name(annotation)}, got None")


def _normalize_union(value: object, annotation: object, *, path: str) -> object:
    """Normalize a value against the first compatible union member."""
    members = get_args(annotation)
    non_none_members = tuple(member for member in members if member is not _NONE_TYPE)
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
    """Validate a scalar's type, allowing integers to normalize to floats."""
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
    """Validate a literal value, restoring YAML boolean aliases when allowed."""
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
    """Normalize a value against a list, tuple, or mapping annotation."""
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
    """Validate that a value is an instance of the annotated class."""
    if not isinstance(annotation, type):
        raise ConfigResolutionError(path, f"unsupported type annotation {_annotation_name(annotation)}")
    if isinstance(value, annotation):
        return value
    raise ConfigResolutionError(
        path,
        f"expected {_annotation_name(annotation)}, got {type(value).__name__}",
    )


def _normalize_override_scalar(value: object, annotation: object) -> object:
    """Convert a CLI string to the numeric type declared by its annotation.

    Supports ``int``, ``float``, and their optional forms. YAML may leave
    numeric CLI input such as ``1e-4`` as a string. Failed conversions leave
    the value unchanged for subsequent validation by ``normalize_value``.
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
    """Validate and normalize a parsed value against its annotation.

    Args:
        value: Parsed value to validate or convert.
        annotation: Type annotation defining the accepted value.
        path: Configuration path used in error messages.

    Returns:
        The validated or converted value.

    Raises:
        ConfigResolutionError: The value does not match the annotation or the
            annotation is unsupported.
    """
    # A nested target is a deferred instance of the annotated runtime type.
    # Its constructor arguments are validated when its own node is resolved;
    # the parent Target materializes it immediately before invocation.
    if isinstance(value, Target):
        return value
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


def _suggestion(name: str, candidates: Iterable[str]) -> str:
    """Return a spelling suggestion, or an empty string if no name is close."""
    matches = difflib.get_close_matches(name, candidates, n=1)
    return f"; did you mean {matches[0]!r}?" if matches else ""


def _replace_target_path(
    config: Target[Any],
    parts: list[str],
    value: object,
    *,
    path: str,
) -> Target[Any]:
    """Return a ``Target`` copy with an argument or nested value replaced."""
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
    """Return a configuration copy with a dotted-path override applied.

    Args:
        config: Resolved configuration object or nested value to update.
        parts: Nonempty sequence of remaining dotted-path components.
        value: Parsed override value to validate and assign.
        path: Path already traversed, or an empty string at the root.

    Returns:
        A copy with the override applied; the input is not modified.

    Raises:
        ConfigResolutionError: The path cannot be updated or the value does
            not match the selected field's annotation.
    """
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


def _resolve_union(node: object, annotation: object, *, path: str) -> object:
    """Resolve a YAML value against a compatible non-``None`` union member."""
    non_none_members = [
        member for member in get_args(annotation) if member is not _NONE_TYPE
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
    """Resolve YAML fields and construct a configuration dataclass."""
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


def _target_hints(target: object, *, path: str) -> dict[str, object]:
    """Resolve annotations for a target function or class constructor."""
    hint_source = target.__init__ if inspect.isclass(target) else target
    try:
        return get_type_hints(hint_source)
    except (NameError, TypeError) as exc:
        raise ConfigResolutionError(path, f"could not resolve target type annotations: {exc}") from exc


def _resolve_nested_target_nodes(value: object, *, path: str) -> object:
    """Resolve reserved ``_target_`` nodes inside a target argument tree."""
    if isinstance(value, Mapping):
        if "_target_" in value:
            return _resolve_target(value, path=path)
        return {
            key: _resolve_nested_target_nodes(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_nested_target_nodes(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _resolve_nested_target_nodes(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    return value


def _resolve_target_args(
    raw_args: Mapping[str, object],
    signature: inspect.Signature,
    hints: Mapping[str, object],
    *,
    path: str,
) -> dict[str, object]:
    """Validate target arguments and fill omitted keyword-callable defaults.

    Positional-only defaults are not added because ``Target`` invokes its
    callable with keyword arguments.
    """
    try:
        signature.bind_partial(**raw_args)
    except TypeError as exc:
        raise ConfigResolutionError(path, f"target arguments are invalid: {exc}") from exc

    normalized = {}
    for name, value in raw_args.items():
        parameter = signature.parameters.get(name)
        value = _resolve_nested_target_nodes(value, path=f"{path}.{name}")
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
    """Resolve a YAML mapping into a ``Target`` without invoking its callable."""
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
    """Resolve a DataLoader target with nested collator and batch runtime."""
    if not isinstance(node, Mapping):
        raise ConfigResolutionError(path, "DataLoader configuration must be a YAML mapping")

    target_node = dict(node)
    if "dataloader_type" in target_node:
        raise ConfigResolutionError(f"{path}.dataloader_type", "renamed to sampler_type")
    if "batch_adapter" in target_node:
        raise ConfigResolutionError(
            f"{path}.batch_adapter",
            "removed by the Omni data lifecycle; use dataset.data_transform for "
            "sample/batch encoding and dataloader.get_batch.runtime_input_adapter "
            "for model-owned forward metadata",
        )
    collate_node = target_node.pop("collate_fn", None)
    get_batch_node = target_node.pop("get_batch", None)
    sampler_type = normalize_value(
        target_node.pop("sampler_type", "single"),
        Literal["single", "cyclic"],
        path=f"{path}.sampler_type",
    )
    data_rearrange_map = target_node.pop("data_rearrange_map", None)
    data_sharding = normalize_value(
        target_node.pop("data_sharding", False),
        bool,
        path=f"{path}.data_sharding",
    )
    use_background_prefetcher = normalize_value(
        target_node.pop("use_background_prefetcher", False),
        bool,
        path=f"{path}.use_background_prefetcher",
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
        sampler_type=sampler_type,
        data_rearrange_map=data_rearrange_map,
        data_sharding=data_sharding,
        use_background_prefetcher=use_background_prefetcher,
    )


def _resolve_dataset_config(node: object, *, path: str) -> DatasetConfig:
    """Resolve a ``DatasetConfig`` with its model assets and sample transform."""
    if not isinstance(node, Mapping):
        raise ConfigResolutionError(path, "Dataset configuration must be a YAML mapping")

    target_node = dict(node)
    model_assets_node = target_node.pop("model_assets", {})
    data_transform_node = target_node.pop("data_transform", None)
    target = _resolve_target(target_node, path=path)
    if isinstance(model_assets_node, Mapping) and "_target_" in model_assets_node:
        model_assets = _resolve_target(
            model_assets_node,
            path=f"{path}.model_assets",
        )
    else:
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
    dataset_config = DatasetConfig(
        target=target,
        model_assets=model_assets,
        data_transform=data_transform,
    )
    return dataset_config


def _resolve_optimizer_config(node: object, *, path: str) -> OptimizerConfig:
    """Resolve an ``OptimizerConfig`` with its FP32 main-parameter policy."""
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
    """Resolve a YAML value according to its configuration annotation.

    Args:
        node: YAML value to resolve.
        annotation: Type annotation defining the configuration component.
        path: Configuration path used in error messages.

    Returns:
        The resolved value, configuration object, or unbuilt ``Target``.

    Raises:
        ConfigResolutionError: The value or target declaration is invalid for
            the annotation.
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
    if isinstance(node, Mapping) and "_target_" in node:
        return _resolve_target(node, path=path)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _resolve_dataclass(node, annotation, path=path)
    return normalize_value(node, annotation, path=path)


def resolve_config(raw: object) -> TrainerConfig:
    """Resolve a YAML root mapping into a ``TrainerConfig``.

    Args:
        raw: Parsed YAML root containing training configuration fields.

    Returns:
        The resolved configuration with unbuilt ``Target`` objects.

    Raises:
        ConfigResolutionError: The root is not a mapping, required fields are
            missing, or a field or target declaration is invalid.
    """
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
