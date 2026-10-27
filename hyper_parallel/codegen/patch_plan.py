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
"""Codegen-only observation of mutations made by the native model builder."""

from __future__ import annotations

import fnmatch
import importlib
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any, Iterable, Mapping, Optional

from torch import nn


@dataclass(frozen=True)
class ModuleState:
    """Source-relevant state of one module at a builder phase boundary."""

    module: nn.Module
    module_type: type[nn.Module]
    forward: Any


@dataclass(frozen=True)
class ModuleReplacementPatch:
    """One module replacement observed between builder phase snapshots."""

    module_fqn: str
    source_type: type[nn.Module]
    replacement_type: type[nn.Module]
    factory: Any
    factory_kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class LocalComputePatch:
    """One local-compute forward installed by native Phase C."""

    module_fqn: str
    module_type: type[nn.Module]
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
    """One inner-wrapper forward change observed after native Phase C."""

    module_fqn: str
    target_fqn: str
    wrapper: Any
    request: Any
    secondary_requests: tuple[Any, ...] = ()
    target_type: Optional[type[nn.Module]] = None


@dataclass(frozen=True)
class PatchEmissionPlan:
    """Immutable source-emission view of mutations already made by the builder."""

    module_replacements: tuple[ModuleReplacementPatch, ...] = ()
    local_computes: tuple[LocalComputePatch, ...] = ()
    inner_wrappers: tuple[InnerWrapperPatch, ...] = ()


def snapshot_modules(model: nn.Module) -> dict[str, ModuleState]:
    """Capture module identities and forwards without modifying the model."""
    return {
        fqn: ModuleState(module, type(module), getattr(module, "forward", None))
        for fqn, module in model.named_modules(remove_duplicate=False)
    }


def build_patch_emission_plan(
    model: nn.Module,
    sharding_plan: Any,
    replacement_specs: Iterable[Any],
    before_builder: Mapping[str, ModuleState],
    before_phase_c: Mapping[str, ModuleState],
    after_phase_c: Mapping[str, ModuleState],
) -> PatchEmissionPlan:
    """Describe actual builder mutations from codegen-side phase snapshots."""
    replacements = _replacement_patches(
        model, tuple(replacement_specs), before_builder, before_phase_c
    )
    local_computes: list[LocalComputePatch] = []
    inner_wrappers: list[InnerWrapperPatch] = []
    for module_fqn, spec in sharding_plan.modules.items():
        before = before_phase_c.get(module_fqn)
        after = after_phase_c.get(module_fqn)
        if before is None or after is None:
            continue
        local = getattr(spec, "local_compute_fn", None)
        if local is not None and (
            _forward_identity(before.forward) != _forward_identity(after.forward)
        ):
            local_computes.append(
                LocalComputePatch(
                    module_fqn=module_fqn,
                    module_type=after.module_type,
                    factory=getattr(local, "_target_", local),
                    factory_kwargs=MappingProxyType(
                        dict(getattr(local, "_kwargs", None) or {})
                    ),
                    resolved_compute=getattr(
                        after.module, "__hp_compute__", after.forward
                    ),
                    region_dispatch=getattr(spec, "region_dispatch", None),
                    original_forward=before.forward,
                    installed_forward=after.forward,
                )
            )
        wrapper = getattr(spec, "inner_wrapper", None)
        if wrapper is not None:
            target_name = (
                getattr(spec, "_resolved_inner_target", None)
                or getattr(spec, "inner_target", None)
                or "self"
            )
            target_fqn = module_fqn if target_name == "self" else f"{module_fqn}.{target_name}"
            target_before = before_phase_c.get(target_fqn)
            target_after = after_phase_c.get(target_fqn)
            if target_before is not None and target_after is not None and (
                _forward_identity(target_before.forward)
                != _forward_identity(target_after.forward)
            ):
                inner_wrappers.append(
                    InnerWrapperPatch(
                        module_fqn=module_fqn,
                        target_fqn=target_name,
                        wrapper=getattr(wrapper, "_target_", wrapper),
                        request=SimpleNamespace(target=target_after.module),
                        target_type=target_after.module_type,
                    )
                )
    return PatchEmissionPlan(
        module_replacements=tuple(replacements),
        local_computes=tuple(local_computes),
        inner_wrappers=tuple(inner_wrappers),
    )


def serialize_patch_emission_plan(plan: PatchEmissionPlan) -> dict[str, Any]:
    """Convert a live emission plan into the JSON-safe drift contract."""
    return {
        "module_replacements": [
            {
                "module_fqn": record.module_fqn,
                "source_type": _symbol_path(record.source_type),
                "replacement_type": _symbol_path(record.replacement_type),
                "factory": _symbol_path(record.factory),
            }
            for record in plan.module_replacements
        ],
        "local_computes": [
            {
                "module_fqn": record.module_fqn,
                "module_type": _symbol_path(record.module_type),
                "factory": _symbol_path(record.factory),
                "region_dispatch": record.region_dispatch,
                "original_forward": _symbol_path(_forward_identity(record.original_forward)),
            }
            for record in plan.local_computes
        ],
        "inner_wrappers": [
            {
                "module_fqn": record.module_fqn,
                "target_fqn": record.target_fqn,
                "wrapper": _symbol_path(record.wrapper),
                "target_type": _symbol_path(
                    record.target_type or type(record.request.target)
                ),
            }
            for record in plan.inner_wrappers
        ],
    }


def deserialize_patch_emission_plan(data: Mapping[str, Any]) -> PatchEmissionPlan:
    """Restore the source-emission view used by offline drift checks."""
    replacements = tuple(
        ModuleReplacementPatch(
            module_fqn=item["module_fqn"],
            source_type=_resolve_symbol(item["source_type"]),
            replacement_type=_resolve_symbol(item["replacement_type"]),
            factory=_resolve_symbol(item["factory"]),
            factory_kwargs=MappingProxyType({}),
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
    return PatchEmissionPlan(replacements, local_computes, inner_wrappers)


def patch_emission_plan_from_meta(meta: Any) -> Optional[PatchEmissionPlan]:
    """Return the persisted emission plan, or ``None`` for legacy artifacts."""
    data = getattr(meta, "emission_patches", None)
    return deserialize_patch_emission_plan(data) if data else None


def _replacement_patches(
    model: nn.Module,
    specs: tuple[Any, ...],
    before_builder: Mapping[str, ModuleState],
    before_phase_c: Mapping[str, ModuleState],
) -> list[ModuleReplacementPatch]:
    records = []
    for fqn in tuple(getattr(model, "_hp_replaced_module_fqns", ()) or ()):
        source = before_builder.get(fqn)
        replacement = before_phase_c.get(fqn)
        if source is None or replacement is None or source.module_type is replacement.module_type:
            continue
        matches = [spec for spec in specs if _replacement_spec_matches(spec, fqn, source)]
        if len(matches) != 1:
            raise RuntimeError(
                f"codegen: observed replacement at {fqn!r} maps to {len(matches)} replacement rules"
            )
        records.append(ModuleReplacementPatch(
            module_fqn=fqn,
            source_type=source.module_type,
            replacement_type=replacement.module_type,
            factory=matches[0].factory,
            factory_kwargs=MappingProxyType({"module_fqn": fqn}),
        ))
    return records


def _replacement_spec_matches(spec: Any, fqn: str, source: ModuleState) -> bool:
    if not any(fnmatch.fnmatchcase(fqn, pattern) for pattern in spec.match):
        return False
    expected = getattr(spec, "module_type", None)
    if expected is None:
        return True
    if getattr(spec, "exact_type", False):
        return source.module_type is expected
    return isinstance(source.module, expected)


def _forward_identity(forward: Any) -> Any:
    return getattr(forward, "__func__", forward)


def _symbol_path(symbol: Any) -> str:
    module = getattr(symbol, "__module__", None)
    qualname = getattr(symbol, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname:
        raise RuntimeError(f"codegen: patch symbol {symbol!r} is not importable")
    return f"{module}:{qualname}"


def _resolve_symbol(path: str) -> Any:
    module_name, separator, qualname = path.partition(":")
    if not separator:
        raise ValueError(f"codegen: invalid patch symbol path {path!r}")
    value = importlib.import_module(module_name)
    for segment in qualname.split("."):
        value = getattr(value, segment)
    return value


__all__ = [
    "PatchEmissionPlan",
    "build_patch_emission_plan",
    "deserialize_patch_emission_plan",
    "patch_emission_plan_from_meta",
    "serialize_patch_emission_plan",
    "snapshot_modules",
]
