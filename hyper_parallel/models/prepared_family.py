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
"""Replay a concrete family selection from a generated bundle."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from importlib import import_module
from types import MappingProxyType
from typing import Any, Callable

from torch import nn

from hyper_parallel.components.modules.module_family import (
    FamilyRequest,
    ResolvedFamily,
    _axis_sizes,
    _build_family_member,
    get_module_family,
)


PREPARED_FAMILY_RECORD_VERSION = 1


def _load_construction_symbol(path: str) -> Callable:
    """Load the globally importable identity recorded during preparation."""
    module_path, _, qualname = path.partition(":")
    value = import_module(module_path)
    for name in qualname.split("."):
        value = getattr(value, name)
    return value


def _decode_construction_value(value: Any) -> Any:
    """Restore typed construction arguments emitted as Python literals."""
    if not isinstance(value, dict):
        return value
    (kind, data), = value.items()
    if kind == "mapping":
        return {key: _decode_construction_value(item) for key, item in data.items()}
    if kind == "list":
        return [_decode_construction_value(item) for item in data]
    if kind == "tuple":
        return tuple(_decode_construction_value(item) for item in data)
    if kind == "symbol":
        return _load_construction_symbol(data)
    if kind == "partial":
        target, args, kwargs = data
        return partial(_load_construction_symbol(target), *_decode_construction_value(args),
                       **_decode_construction_value(kwargs))
    raise ValueError(f"Unknown prepared family argument kind {kind!r}")


def build_prepared_family_replacement(
    *, module: nn.Module, module_fqn: str, prepared: Mapping[str, Any],
    construct_member: Callable | None = None,
) -> nn.Module:
    """Build the exact source member recorded by shared model preparation.

    Args:
        module: Original module constructed by the source model.
        module_fqn: Exact path of the source module.
        prepared: Serialized request and concrete selection from preparation.
        construct_member: Optional source-emitted call to the same member.

    Returns:
        The prepared member with the same construction provenance as a native build.
    """
    if prepared["version"] != PREPARED_FAMILY_RECORD_VERSION:
        raise ValueError(f"Unsupported prepared family record version for {module_fqn!r}")
    target = _load_construction_symbol(prepared["target"])
    family = get_module_family(target)
    variant = prepared["variant"]
    if family is None or variant not in family.variants:
        raise ValueError(f"Prepared family selection for {module_fqn!r} no longer matches its registration")
    member = _load_construction_symbol(prepared["member"])
    registered_module, _, registered_name = family.variants[variant].rpartition(".")
    if _load_construction_symbol(f"{registered_module}:{registered_name}") is not member:
        raise ValueError(f"Prepared family member for {module_fqn!r} no longer matches variant {variant!r}")
    context = _decode_construction_value(prepared["construction_context"])
    request = FamilyRequest(target, _decode_construction_value(prepared["request_arguments"]))
    resolved = ResolvedFamily(
        family, variant, member,
        MappingProxyType(_decode_construction_value(prepared["resolved_arguments"])),
        MappingProxyType(dict(prepared["selectors"])), _axis_sizes(context),
    )
    return _build_family_member(
        request, resolved, module=module, module_fqn=module_fqn, context=context, construct_member=construct_member,
    )
