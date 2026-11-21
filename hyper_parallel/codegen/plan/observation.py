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
import inspect
from typing import Any
from types import MethodType

from hyper_parallel.codegen.inline.component_source import observe_wrapper_source
from hyper_parallel.distributed._builder.applier import (
    _boundary_post_order_key, _family_boundary_state, _get_active_mesh, _get_cp_submesh, _get_tp_submesh,
)
from hyper_parallel.distributed._builder.rule_resolver import _resolve_inner_wrapper
from hyper_parallel.distributed._builder.forward_rewriter import _classify_rewrite_result, validate_wrapped_forward

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


def observe_emission_plan(plan: Any, model: Any, mesh: Any = None) -> EmissionPlanObservation:
    """Project native plan and scoped wrapper results without applying distributed sharding."""
    if plan is None:
        return EmissionPlanObservation()
    injections = _observe_injections(plan)
    if mesh is not None and model is not None:
        _observe_wrapper_results(injections, plan, model, mesh)
    return EmissionPlanObservation(
        param_plan=_observe_boundary_plan(plan),
        injections=injections,
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


def _observe_wrapper_results(injections: list[dict[str, Any]], plan: Any, model: Any, context: Any) -> None:
    """Invoke each native boundary factory once, without installing a distributed runtime."""
    entries = [entry for entry in injections if plan.modules[entry["match"]].is_boundary
               and getattr(plan.modules[entry["match"]], "inner_wrapper", None) is not None]
    if not entries:
        return
    mesh = _get_active_mesh(context.device_mesh, plan.mesh_dim_names)
    cp_mesh = _get_cp_submesh(mesh, plan.mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, plan.mesh_dim_names)
    entries.sort(key=lambda entry: _boundary_post_order_key((entry["match"], None)))
    for entry in entries:
        spec = plan.modules[entry["match"]]
        boundary = model.get_submodule(entry["match"])
        owned, skip = _family_boundary_state(boundary, entry["match"], spec, plan)
        if owned or skip:
            entry["observed_wrapper"] = _observed_target_source(
                boundary, boundary.forward, vars(boundary), MethodType(type(boundary).forward, boundary), cp_mesh,
            )
            continue
        resolved = _resolve_inner_wrapper(
            boundary, spec, cp_mesh, mesh,
            tp_mesh=tp_mesh, ep_mesh=getattr(context, "fsdp_moe_mesh", None),
        )
        if resolved is None:
            continue
        name, target, apply = resolved
        original_forward = target.forward
        original_attrs = dict(vars(target))
        primary, secondaries = _classify_rewrite_result(apply(), target, name)
        if secondaries:
            raise RuntimeError(f"codegen observed secondary rewrites at {entry['match']!r}; not yet supported")
        if primary is not None:
            validate_wrapped_forward(original_forward, primary.forward, owner=f"codegen boundary {entry['match']!r}")
            entry["observed_wrapper"] = _observed_target_source(
                target, primary.forward, primary.companion_attrs, original_forward, cp_mesh,
            )
        elif getattr(target.forward, "__func__", target.forward) is not getattr(
                original_forward, "__func__", original_forward):
            validate_wrapped_forward(original_forward, target.forward, owner=f"codegen boundary {entry['match']!r}")
            companions = {key: value for key, value in vars(target).items()
                          if key not in original_attrs or value is not original_attrs[key]}
            entry["observed_wrapper"] = _observed_target_source(
                target, target.forward, companions, original_forward, cp_mesh,
            )
        _validate_observed_target(entry, target, boundary)


def _observed_target_source(target: Any, forward: Any, companions: dict, original: Any, cp_mesh: Any) -> dict:
    """Keep the exact native target identity alongside its source, not its parent boundary."""
    source = observe_wrapper_source(target, forward, companions, original_forward=original, cp_mesh=cp_mesh,
                                    preserve_signature=True)
    source["host_class"] = type(target).__name__
    source["target_type"] = f"{type(target).__module__}:{type(target).__qualname__}"
    return source



def _validate_observed_target(entry: dict, target: Any, boundary: Any) -> None:
    """A child can use native companions but cannot borrow its parent's boundary field."""
    body = (entry.get("observed_wrapper") or {}).get("body") or ""
    if target is not boundary and "self._hp_boundary" in body:
        raise RuntimeError(f"codegen observed child target at {entry['match']!r} without its own mesh binding")


def observe_final_forwards(model: Any, originals: dict[str, Any], context: Any) -> list[dict[str, Any]]:
    """Read only methods changed after inner construction; never replay their factories."""
    additions = []
    for fqn, original in originals.items():
        target = model.get_submodule(fqn)
        forward = target.forward
        if getattr(forward, "__func__", forward) is getattr(original, "__func__", original):
            continue
        if getattr(original, "__self__", None) is not target:
            raise RuntimeError(f"codegen outer factory replaced observed module {fqn!r}")
        bindings = _final_cp_bindings(forward, context)
        source = observe_wrapper_source(target, forward, vars(target), original_forward=original,
                                        cp_mesh=getattr(context, "cp_mesh", None), source_bindings=bindings,
                                        preserve_signature=True)
        source.update(host_class=type(target).__name__,
                      target_type=f"{type(target).__module__}:{type(target).__qualname__}")
        if "cp_rank" in bindings:
            source["source_imports"] = ["import torch"]
            source["imports"] = [item for item in source["imports"] if item[1] != "torch"]
        function = getattr(forward, "__func__", forward)
        additions.append({"match": fqn, "inner_target": "self", "observed_wrapper": source,
                          "inner_wrapper": f"{function.__globals__['__name__']}.{function.__code__.co_qualname}"})
    return additions


def _final_cp_bindings(forward: Any, context: Any) -> dict[str, str]:
    """Express captured CP coordinates from the native contiguous dp/cp/tp topology."""
    captures = inspect.getclosurevars(forward).nonlocals
    bindings = {}
    if "cp_size" in captures:
        if captures["cp_size"] != context.cp_size:
            raise RuntimeError("codegen outer factory captured a different CP size")
        bindings["cp_size"] = repr(context.cp_size)
    if "cp_rank" in captures:
        mesh = context.device_mesh
        if (tuple(mesh.mesh_dim_names) != ("dp", "cp", "tp")
                or tuple(mesh.rank_list) != tuple(range(mesh.size()))
                or captures["cp_rank"] != context.cp_rank):
            raise RuntimeError("codegen outer factory CP coordinate has no proven source binding")
        bindings["cp_rank"] = f"torch.distributed.get_rank() // {context.tp_size} % {context.cp_size}"
    return bindings
