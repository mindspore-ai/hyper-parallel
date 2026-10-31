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
"""Model-agnostic structural checks for generated Python source."""

from __future__ import annotations

import ast
from collections import Counter
from typing import Mapping


def verify_generated_source_structure(files: Mapping[str, str]) -> None:
    """Reject ambiguous forward composition in every generated source unit.

    The check derives ownership from the emitted AST. It has no model-family,
    module-path, or class-name table, so newly onboarded models receive the
    same validation automatically.

    Args:
        files: Complete generated bundle keyed by filename.

    Raises:
        RuntimeError: A generated class has duplicate methods, a recursive
            implementation fallback, or an enable flag not consumed by its
            emitted forward pipeline.
    """
    errors: list[str] = []
    for filename, source_text in sorted(files.items()):
        if not filename.endswith(".py") or filename == "__init__.py":
            continue
        try:
            tree = ast.parse(source_text, filename=filename)
        except SyntaxError as exc:
            errors.append(f"{filename}: invalid Python source ({exc.msg})")
            continue
        for class_node in (
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ):
            _verify_class(filename, class_node, errors)
    if errors:
        raise RuntimeError(
            "codegen: generated forward composition is invalid: "
            + "; ".join(errors)
        )


def _verify_class(
    filename: str, class_node: ast.ClassDef, errors: list[str]
) -> None:
    methods = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    counts = Counter(method.name for method in methods)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    location = f"{filename}:{class_node.name}"
    if duplicates:
        errors.append(f"{location} defines methods more than once: {duplicates}")

    implementations = [
        method for method in methods if method.name == "_forward_impl"
    ]
    for implementation in implementations:
        if _calls_self_method(implementation, "_forward_impl"):
            errors.append(f"{location}._forward_impl calls itself")

    dispatch_methods = [
        method
        for method in methods
        if method.name in {"forward", "_forward_impl"}
    ]
    state_flags = _class_enable_flags(class_node)
    for state_flag in sorted(state_flags):
        if not any(_reads_self_attribute(method, state_flag) for method in dispatch_methods):
            errors.append(
                f"{location} declares {state_flag!r} but its forward pipeline "
                "does not read it"
            )


def _class_enable_flags(class_node: ast.ClassDef) -> set[str]:
    flags: set[str] = set()
    for statement in class_node.body:
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else (statement.target,) if isinstance(statement, ast.AnnAssign) else ()
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id.endswith("_enable"):
                flags.add(target.id)
    return flags


def _calls_self_method(method: ast.AST, method_name: str) -> bool:
    return any(
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == method_name
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
    )


def _reads_self_attribute(method: ast.AST, attribute_name: str) -> bool:
    return any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == attribute_name
        for node in ast.walk(method)
    )


__all__ = ["verify_generated_source_structure"]
