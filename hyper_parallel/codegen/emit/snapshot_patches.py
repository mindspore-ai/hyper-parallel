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
"""Convert native-build snapshot observations into emitter patch records."""

from __future__ import annotations

import ast
import importlib
import os
from typing import Any, Optional

from hyper_parallel.codegen.inline.targets import import_target_path
from hyper_parallel.codegen.source.resolver import resolve_remote_siblings


def emission_patches_from_snapshot_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Translate resolved native observations into emitter patch records."""
    applications = payload["replacements"]["applications"]
    replacements = [
        {
            "module_fqn": item["primary_fqn"],
            "source_type": item["source_type"],
            "replacement_type": item["replacement_type"],
        }
        for item in applications
    ]
    replacement_types = {
        module_fqn: item["replacement_type"]
        for item in applications
        for module_fqn in item.get("module_fqns", (item["primary_fqn"],))
    }
    source = payload["source"]
    plan = payload.get("plan") or {}
    injections = plan.get("injections") or ()
    boundary_classes = plan.get("boundary_classes") or {}
    boundary_types = plan.get("boundary_types") or {}
    param_plan = plan.get("param_plan") or {}
    local_computes, inner_wrappers = [], []
    for injection in injections:
        local, wrapper = _injection_patches(
            injection,
            source,
            boundary_classes,
            boundary_types,
            replacement_types,
            param_plan,
        )
        if local is not None:
            local_computes.append(local)
        if wrapper is not None:
            inner_wrappers.append(wrapper)
    return {
        "module_replacements": replacements,
        "local_computes": local_computes,
        "inner_wrappers": inner_wrappers,
    }


def _injection_patches(
    injection: dict[str, Any],
    source: dict[str, Any],
    boundary_classes: dict[str, Any],
    boundary_types: dict[str, Any],
    replacement_types: dict[str, str],
    param_plan: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Translate one resolved injection into source-emission patches."""
    module_fqn = injection.get("match")
    class_name = boundary_classes.get(module_fqn)
    module_name = source.get("module_name")
    declared_type = (
        f"{module_name}:{class_name}" if module_name and class_name else None
    )
    source_type = _source_type_identity(source, class_name)
    observed_type = boundary_types.get(module_fqn)
    if source_type is None and _is_remote_boundary_type(observed_type):
        source_type = observed_type
    bound_forward = _bound_forward_identity(source, module_fqn)
    local = _local_compute_patch(
        injection,
        source,
        module_fqn,
        class_name,
        declared_type,
        source_type,
        bound_forward,
        param_plan,
    )
    wrapper = _inner_wrapper_patch(
        injection,
        module_fqn,
        declared_type,
        source_type,
        replacement_types,
    )
    return local, wrapper


def _is_remote_boundary_type(value: Any) -> bool:
    """Return whether an observed type identifies non-framework source."""
    return isinstance(value, str) and not value.startswith(
        ("hyper_parallel.", "torch.")
    )


def _local_compute_patch(
    injection: dict[str, Any],
    source: dict[str, Any],
    module_fqn: str,
    class_name: str,
    declared_type: Optional[str],
    source_type: Optional[str],
    bound_forward: Optional[str],
    param_plan: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Build one local-compute emission record."""
    local_compute = injection.get("local_compute_fn")
    if local_compute is None:
        return None
    if declared_type is None:
        raise ValueError(
            "snapshot local_compute_fn requires a source module and "
            f"boundary class for {module_fqn!r}"
        )
    if source_type is None and bound_forward is None:
        return None
    module_type = (
        (_imported_type_identity(source, class_name) if bound_forward else source_type)
        or source_type
        or declared_type
    )
    return {
        "module_fqn": module_fqn,
        "module_type": module_type,
        "factory": import_target_path(local_compute),
        "region_dispatch": (param_plan.get(module_fqn) or {}).get("region_dispatch"),
        "original_forward": bound_forward or f"{source_type}.forward",
    }


def _inner_wrapper_patch(
    injection: dict[str, Any],
    module_fqn: str,
    declared_type: Optional[str],
    source_type: Optional[str],
    replacement_types: dict[str, str],
) -> Optional[dict[str, Any]]:
    """Build one inner-wrapper emission record."""
    wrapper = injection.get("inner_wrapper")
    if wrapper is None:
        return None
    target_type = replacement_types.get(module_fqn) or source_type or declared_type
    if target_type is None:
        raise ValueError(
            "snapshot inner_wrapper requires an observed replacement or "
            f"boundary class for {module_fqn!r}"
        )
    return {
        "module_fqn": module_fqn,
        "target_fqn": injection.get("inner_target") or "self",
        "wrapper": import_target_path(wrapper),
        "target_type": target_type,
    }


def _source_class_names(source: dict[str, Any]) -> set[str]:
    """Return classes defined by the snapshot's modeling source file."""
    file_path = source.get("file_path")
    if not file_path:
        return set()
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            tree = ast.parse(handle.read(), filename=file_path)
    except (OSError, SyntaxError):
        return set()
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def _source_type_identity(
    source: dict[str, Any], class_name: Optional[str]
) -> Optional[str]:
    """Resolve a class to the remote source unit that actually defines it."""
    if not class_name:
        return None
    module_name = source.get("module_name")
    if not isinstance(module_name, str) or not module_name:
        return None
    if class_name in _source_class_names(source):
        return f"{module_name}:{class_name}"

    package, separator, _ = module_name.rpartition(".")
    candidates = []
    for path in resolve_remote_siblings(source).values():
        sibling_source = {"file_path": path}
        if class_name not in _source_class_names(sibling_source):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        sibling_module = f"{package}.{stem}" if separator else stem
        candidates.append(f"{sibling_module}:{class_name}")
    if len(candidates) > 1:
        raise ValueError(
            f"snapshot class {class_name!r} is defined by multiple bundled source units: "
            f"{sorted(candidates)}"
        )
    return candidates[0] if candidates else None


def _bound_forward_identity(
    source: dict[str, Any], module_fqn: Optional[str]
) -> Optional[str]:
    """Resolve a model-owned function installed through ``MethodType``."""
    file_path = source.get("file_path")
    module_name = source.get("module_name")
    if not file_path or not module_name or not module_fqn:
        return None
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            tree = ast.parse(handle.read(), filename=file_path)
    except (OSError, SyntaxError):
        return None

    boundary_name = module_fqn.rsplit(".", 1)[-1]
    candidates = set()
    for node in ast.walk(tree):
        candidate = _method_type_forward_name(node, boundary_name)
        if candidate:
            candidates.add(candidate)
    if len(candidates) > 1:
        raise ValueError(
            f"snapshot boundary {module_fqn!r} has multiple MethodType forwards: "
            f"{sorted(candidates)}"
        )
    return f"{module_name}:{next(iter(candidates))}" if candidates else None


def _method_type_forward_name(node: ast.AST, boundary_name: str) -> Optional[str]:
    """Return the function assigned as one boundary's MethodType forward."""
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None
    target, call = node.targets[0], node.value
    valid_target = (
        isinstance(target, ast.Attribute)
        and target.attr == "forward"
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == boundary_name
    )
    if not valid_target or not isinstance(call, ast.Call):
        return None
    if _ast_name(call.func) != "MethodType" or not call.args:
        return None
    return call.args[0].id if isinstance(call.args[0], ast.Name) else None


def _imported_type_identity(
    source: dict[str, Any], class_name: Optional[str]
) -> Optional[str]:
    """Resolve an imported boundary class from modules referenced by the source."""
    file_path = source.get("file_path")
    if not file_path or not class_name:
        return None
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            tree = ast.parse(handle.read(), filename=file_path)
    except (OSError, SyntaxError):
        return None

    explicit_modules = _explicit_import_modules(tree, class_name)
    if len(explicit_modules) > 1:
        raise ValueError(
            f"snapshot class {class_name!r} is imported from multiple modules: "
            f"{sorted(explicit_modules)}"
        )
    if explicit_modules:
        return f"{next(iter(explicit_modules))}:{class_name}"
    candidates = _exporting_import_modules(tree, class_name)
    if len(candidates) > 1:
        raise ValueError(
            f"snapshot class {class_name!r} is exported by multiple imported modules: "
            f"{candidates}"
        )
    return candidates[0] if candidates else None


def _explicit_import_modules(tree: ast.AST, class_name: str) -> set[str]:
    """Return modules that explicitly import one class name."""
    return {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module
        and any(
            alias.name == class_name and (alias.asname or alias.name) == class_name
            for alias in node.names
        )
    }


def _exporting_import_modules(tree: ast.AST, class_name: str) -> list[str]:
    """Return imported modules that export one class name."""
    modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    }
    candidates = []
    for module_name in sorted(modules):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, AttributeError):
            continue
        if hasattr(module, class_name):
            candidates.append(f"{module_name}:{class_name}")
    return candidates


def _ast_name(node: ast.AST) -> Optional[str]:
    """Return the terminal name of a callable expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


__all__ = ["emission_patches_from_snapshot_payload"]
