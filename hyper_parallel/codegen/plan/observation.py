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
"""Observe only native-plan facts required for source emission."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from hyper_parallel.codegen.plan.placements import (
    named_placement_to_dict,
    value_to_data,
)


@dataclass
class EmissionPlanObservation:
    """Plain-data projection consumed by Codegen emitters."""

    param_plan: dict[str, dict[str, Any]] = field(default_factory=dict)
    injections: list[dict[str, Any]] = field(default_factory=list)
    boundary_classes: dict[str, str] = field(default_factory=dict)
    boundary_types: dict[str, str] = field(default_factory=dict)
    nested_components: list[dict[str, str]] = field(default_factory=list)
    mesh_dim_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the mutable mapping consumed by the session boundary."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


def observe_emission_plan(plan: Any, model: Any) -> EmissionPlanObservation:
    """Project a native sharding plan without changing or applying it."""
    if plan is None:
        return EmissionPlanObservation()
    return EmissionPlanObservation(
        param_plan=_observe_boundary_plan(plan),
        injections=_observe_injections(plan),
        boundary_classes=_observe_boundary_classes(plan, model),
        boundary_types=_observe_boundary_types(plan, model),
        nested_components=_observe_nested_components(plan, model),
        mesh_dim_names=tuple(getattr(plan, "mesh_dim_names", ()) or ()),
    )


def _observe_boundary_plan(plan: Any) -> dict[str, dict[str, Any]]:
    """Collect boundary layout contracts required for source emission."""
    result: dict[str, dict[str, Any]] = {}
    for fqn, spec in (getattr(plan, "modules", {}) or {}).items():
        entry: dict[str, Any] = {}
        for name in ("in_src", "in_dst", "out_src", "out_dst"):
            value = getattr(spec, name, None)
            if value is not None:
                entry[name] = named_placement_to_dict(value)
        for name in ("out_names", "_deferred_bias_params"):
            value = getattr(spec, name, None)
            if value:
                key = "deferred_bias_params" if name.startswith("_") else name
                entry[key] = list(value)
        for name in ("is_boundary", "region_dispatch"):
            value = getattr(spec, name, None)
            if value is not None:
                entry[name] = value
        result[fqn] = entry
    return result


def _observe_injections(plan: Any) -> list[dict[str, Any]]:
    """Collect compute and wrapper injections in stable module-path order."""
    result = []
    for fqn, spec in sorted((getattr(plan, "modules", {}) or {}).items()):
        entry: dict[str, Any] = {}
        for name in ("inner_wrapper", "inner_out_src", "local_compute_fn"):
            value = getattr(spec, name, None)
            if value is not None:
                entry[name] = value_to_data(value)
        inner_target = getattr(spec, "inner_target", None)
        if inner_target is not None:
            entry["inner_target"] = inner_target
        if entry:
            result.append({"match": fqn, **entry})
    return result


def _observe_boundary_classes(plan: Any, model: Any) -> dict[str, str]:
    """Map planned module paths to their source class names."""
    if model is None:
        return {}
    return {
        fqn: type(model.get_submodule(fqn)).__name__
        for fqn in (getattr(plan, "modules", {}) or {})
    }


def _observe_boundary_types(plan: Any, model: Any) -> dict[str, str]:
    """Map planned module paths to fully qualified runtime type identities."""
    if model is None:
        return {}
    result = {}
    for fqn in (getattr(plan, "modules", {}) or {}):
        module_type = type(model.get_submodule(fqn))
        result[fqn] = f"{module_type.__module__}:{module_type.__qualname__}"
    return result


def _observe_nested_components(plan: Any, model: Any) -> list[dict[str, str]]:
    """Identify nested model source owners needed for source composition."""
    if model is None:
        return []
    root_module = type(model).__module__
    components: dict[tuple[str, str], dict[str, str]] = {}
    for boundary_fqn in (getattr(plan, "modules", {}) or {}):
        owner_module = type(model.get_submodule(boundary_fqn)).__module__
        if (
            owner_module == root_module
            or owner_module.startswith("torch.")
            or owner_module.startswith("hyper_parallel.")
        ):
            continue
        candidates = []
        parts = boundary_fqn.split(".")
        for index in range(1, len(parts) + 1):
            ancestor_fqn = ".".join(parts[:index])
            ancestor_type = type(model.get_submodule(ancestor_fqn))
            if ancestor_type.__module__ == owner_module:
                candidates.append((ancestor_fqn, ancestor_type))
        if not candidates:
            continue
        component_fqn, component_type = candidates[0]
        identity = f"{component_type.__module__}:{component_type.__qualname__}"
        components[(component_fqn, identity)] = {
            "fqn": component_fqn,
            "type": identity,
            "owner_module": owner_module,
        }
    return [components[key] for key in sorted(components)]


__all__ = ["EmissionPlanObservation", "observe_emission_plan"]
