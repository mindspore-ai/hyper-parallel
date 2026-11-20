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
"""Materialize the resolved family member's native parallel forward stages."""

from __future__ import annotations

import ast
from importlib import import_module
import inspect
import textwrap
from typing import Any

from hyper_parallel.codegen.inline.ir import ImportPatch, InlinePatchSet, ModuleSnippetPatch
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set


class _ForwardGlobals(ast.NodeTransformer):
    """Bind copied body references to their original module."""

    def __init__(self, names: set[str], module_alias: str) -> None:
        self.names = names
        self.module_alias = module_alias

    def visit_Name(self, node: ast.Name) -> ast.AST:  # pylint: disable=invalid-name
        """Qualify the native module globals used by the copied forward."""
        if isinstance(node.ctx, ast.Load) and node.id in self.names:
            return ast.copy_location(
                ast.Attribute(ast.parse(self.module_alias, mode="eval").body, node.id, ast.Load()), node,
            )
        return node


def _native_forward_source(member: type, globals_alias: str) -> tuple[str, str]:
    """Copy an executable method without replacing its native stage implementations."""
    forward = member.forward
    closure = inspect.getclosurevars(forward)
    if closure.nonlocals or inspect.unwrap(forward) is not forward:
        raise ValueError(f"Codegen cannot inline the closed or decorated forward of {member.__name__}")
    function, = ast.parse(textwrap.dedent(inspect.getsource(forward))).body
    if not isinstance(function, ast.FunctionDef):
        raise ValueError(f"Codegen cannot inline the forward of {member.__name__}")
    if any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child is not function
           for child in ast.walk(function)):
        raise ValueError(f"Codegen cannot inline nested definitions in {member.__name__}.forward")
    function.returns = None
    for argument in ast.walk(function.args):
        if isinstance(argument, ast.arg):
            argument.annotation = None
    if ast.get_docstring(function) is not None:
        function.body.pop(0)
    globals_import = f"import {forward.__module__}\n" if closure.globals else ""
    transformed = _ForwardGlobals(set(closure.globals), globals_alias).visit(function)
    return ast.unparse(transformed), globals_import


def _stage_calls(host: ast.ClassDef) -> set[str]:
    """Find explicit self-owned stage calls in the emitted forward path."""
    methods = [
        child for child in host.body if isinstance(child, ast.FunctionDef)
        and child.name in ("forward", "_forward_impl")
    ]
    return {
        node.func.attr for method in methods for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
    }


def verify_parallel_family_forwards(tree: ast.AST, records: dict[str, Any]) -> None:
    """Prove each native parallel family has real source-visible forward calls."""
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    for fqn, record in records.items():
        name = record["type"].rsplit(":", 1)[-1]
        host = classes.get(name)
        missing = set(record["stages"]) - (_stage_calls(host) if host is not None else set())
        if missing:
            kind = record["kind"].upper()
            raise ValueError(f"Codegen did not emit {kind} forward stages for {fqn}: {sorted(missing)}")


def materialize_parallel_family_forwards(source: str, records: dict[str, Any]) -> str:
    """Emit a local forward subclass for each concrete member selected by Trainer."""
    if not records:
        return source
    classes = {node.name for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    owners = {item["type"]: item["kind"] for item in records.values()}
    for identity, kind in sorted(owners.items()):
        module_name, name = identity.split(":", 1)
        if name in classes:
            continue
        if not name.isidentifier():
            raise ValueError(f"Codegen cannot emit nested parallel member {identity}")
        member = getattr(import_module(module_name), name)
        body, globals_import = _native_forward_source(member, member.forward.__module__)
        declaration = (
            f"class {name}({module_name}.{name}):\n"
            f"    # {kind.upper()} forward stages selected by native preparation.\n"
            f"{textwrap.indent(body, '    ')}\n"
        )
        patches = InlinePatchSet()
        patches.imports.append(ImportPatch('', (), raw=f"import {module_name}"))
        if globals_import:
            patches.imports.append(ImportPatch("", (), raw=globals_import.strip()))
        patches.module_snippets.append(ModuleSnippetPatch(declaration))
        source = apply_patch_set(source, patches)
        classes.add(name)
    verify_parallel_family_forwards(ast.parse(source), records)
    return source
