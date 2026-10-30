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
"""Emitter-only views reconstructed from frozen snapshot patch records."""

from __future__ import annotations

import importlib
import keyword
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ModuleReplacementPatch:
    """One already-resolved module replacement."""

    module_fqn: str
    source_type: type
    replacement_type: type
    factory: Any = None
    factory_kwargs: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class LocalComputePatch:
    """One already-resolved local-compute source patch."""

    module_fqn: str
    module_type: type
    factory: Any
    factory_kwargs: Mapping[str, Any]
    resolved_compute: Any
    region_dispatch: Optional[bool]
    original_forward: Any
    installed_forward: Any
    boundary: Any = None
    excluded_subtrees: tuple[str, ...] = ()


@dataclass(frozen=True)
class InnerWrapperPatch:
    """One already-resolved inner-wrapper source patch."""

    module_fqn: str
    target_fqn: str
    wrapper: Any
    request: Any
    target_type: Optional[type] = None
    secondary_requests: tuple[Any, ...] = ()


@dataclass(frozen=True)
class EmissionPatchSet:
    """Immutable source-emission view of frozen native observations."""

    module_replacements: tuple[ModuleReplacementPatch, ...] = ()
    local_computes: tuple[LocalComputePatch, ...] = ()
    inner_wrappers: tuple[InnerWrapperPatch, ...] = ()


def emission_patches_from_meta(meta: Any) -> Optional[EmissionPatchSet]:
    """Rehydrate emitter views from JSON-safe snapshot metadata."""
    data = getattr(meta, "emission_patches", None)
    if not data:
        return None
    replacements = tuple(
        ModuleReplacementPatch(
            module_fqn=item["module_fqn"],
            source_type=_resolve_symbol(item["source_type"]),
            replacement_type=_resolve_symbol(item["replacement_type"]),
        )
        for item in data.get("module_replacements", ())
    )
    local_computes = tuple(
        LocalComputePatch(
            module_fqn=item["module_fqn"],
            module_type=_resolve_symbol(item["module_type"]),
            factory=_resolve_symbol(item["factory"]),
            factory_kwargs=MappingProxyType({}),
            resolved_compute=None,
            region_dispatch=item.get("region_dispatch"),
            original_forward=_resolve_symbol(item["original_forward"]),
            installed_forward=None,
        )
        for item in data.get("local_computes", ())
    )
    inner_wrappers = tuple(
        InnerWrapperPatch(
            module_fqn=item["module_fqn"],
            target_fqn=item["target_fqn"],
            wrapper=_resolve_symbol(item["wrapper"]),
            request=SimpleNamespace(target=None),
            target_type=_resolve_symbol(item["target_type"]),
        )
        for item in data.get("inner_wrappers", ())
    )
    return EmissionPatchSet(replacements, local_computes, inner_wrappers)


def local_compute_binding_name(target: Any) -> str:
    """Return the explicit instance attribute used for one compute factory."""
    if isinstance(target, Mapping):
        target = target.get("_target_") or target.get("target") or target.get("name")
    else:
        target = getattr(target, "_target_", target)
    if isinstance(target, str):
        name = target.rsplit(":", 1)[-1].rsplit(".", 1)[-1]
    else:
        name = getattr(target, "__name__", "")
    if not name or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            f"codegen: local-compute target {target!r} has no valid binding name"
        )
    return name


def _resolve_symbol(path: str) -> Any:
    module_name, separator, qualname = path.partition(":")
    if not separator:
        raise ValueError(f"codegen: invalid patch symbol path {path!r}")
    value = importlib.import_module(module_name)
    for segment in qualname.split("."):
        value = getattr(value, segment)
    return value


__all__ = ["EmissionPatchSet", "emission_patches_from_meta", "local_compute_binding_name"]
