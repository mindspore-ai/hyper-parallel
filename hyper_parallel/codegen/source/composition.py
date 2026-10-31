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
"""Compose nested AutoModel sources into one generated modeling unit."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
from typing import Any


def compose_nested_model_sources(source_text: str, meta: Any) -> str:
    """Inline source modules that own boundaries below a nested model root.

    The native model remains unchanged. This is an artifact-only composition:
    it replaces a nested ``AutoModel.from_config`` construction with the
    observed concrete class and appends that class's original source module to
    the same generated file before normal TP/EP lowering runs.

    Args:
        source_text: Primary model source.
        meta: Frozen codegen metadata carrying observed nested components.

    Returns:
        One source string containing the primary and nested model definitions.
    """
    components = getattr(meta, "nested_components", None) or ()
    if not components:
        return source_text
    rendered = source_text
    included_modules: set[str] = set()
    for component in components:
        identity = component.get("type")
        if not isinstance(identity, str) or ":" not in identity:
            raise ValueError(f"invalid nested component identity: {identity!r}")
        module_name, class_name = identity.split(":", 1)
        class_name = class_name.rsplit(".", 1)[-1]
        replaced = _replace_auto_model_construction(
            rendered,
            component_fqn=str(component.get("fqn") or ""),
            class_name=class_name,
        )
        if replaced is None:
            continue
        rendered = replaced
        if module_name not in included_modules:
            nested_source = _read_module_source(module_name)
            rendered = _append_source_module(rendered, nested_source, module_name)
            included_modules.add(module_name)
    return rendered


def _read_module_source(module_name: str) -> str:
    """Read an installed module without importing its runtime dependencies."""
    spec = importlib.util.find_spec(module_name)
    path = None if spec is None else spec.origin
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"codegen: nested model source {module_name!r} is not readable"
        )
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _append_source_module(primary: str, nested: str, module_name: str) -> str:
    """Append one nested module after removing module-only syntax."""
    tree = ast.parse(nested, filename=module_name)
    package = module_name.rpartition(".")[0]
    tree = _AbsoluteImports(package).visit(tree)
    ast.fix_missing_locations(tree)
    primary_tree = ast.parse(primary)
    primary_bindings = _top_level_definitions(primary_tree)
    nested_bindings = _top_level_definitions(tree)
    collisions = {
        name for name in primary_bindings.keys() & nested_bindings.keys()
        if not isinstance(nested_bindings[name], ast.ClassDef)
    }
    if collisions:
        replacements = {name: f"_hp_nested_{name}" for name in collisions}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.name = replacements.get(node.name, node.name)
        tree = _RenameBindings(replacements).visit(tree)
        ast.fix_missing_locations(tree)
    nodes = [
        node for node in tree.body
        if not isinstance(node, ast.Expr) or not _is_docstring(node)
    ]
    nodes = [
        node for node in nodes
        if not isinstance(node, ast.ImportFrom) or node.module != "__future__"
    ]
    body = "\n\n".join(ast.unparse(node) for node in nodes)
    marker = (
        "# [HYPER NESTED MODEL] source merged from "
        f"{module_name}; TP/EP lowering below targets these concrete classes."
    )
    return f"{primary.rstrip()}\n\n\n{marker}\n{body}\n"


def _top_level_definitions(tree: ast.Module) -> dict[str, ast.AST]:
    """Return named functions, classes, and simple assignments."""
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node
    return result


def _is_docstring(node: ast.Expr) -> bool:
    return isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


class _RenameBindings(ast.NodeTransformer):
    """Rename colliding nested-module globals and their references."""

    def __init__(self, names: dict[str, str]) -> None:
        """Create a renamer for the supplied module-global bindings."""
        self._names = names

    def visit_Name(self, node: ast.Name) -> ast.AST:  # pylint: disable=invalid-name
        """Rename a matching load or store while preserving its AST context."""
        replacement = self._names.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node)


class _AbsoluteImports(ast.NodeTransformer):
    """Anchor nested-module relative imports on their original package."""

    def __init__(self, package: str) -> None:
        """Create a transformer for one installed source package."""
        self._package = package

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:  # pylint: disable=invalid-name
        """Convert one relative import into its installed absolute module."""
        if node.level == 0:
            return node
        relative = "." * node.level + (node.module or "")
        module = importlib.util.resolve_name(relative, self._package)
        return ast.copy_location(
            ast.ImportFrom(module=module, names=node.names, level=0),
            node,
        )


def _replace_auto_model_construction(
    source_text: str,
    *,
    component_fqn: str,
    class_name: str,
) -> str | None:
    """Replace an observed AutoModel construction, or reject the candidate."""
    attribute = component_fqn.rsplit(".", 1)[-1]
    if not attribute:
        raise ValueError("nested component FQN must not be empty")
    pattern = re.compile(
        rf"(?m)^(?P<indent>\s*)self\.{re.escape(attribute)}\s*=\s*"
        r"AutoModel\.from_config\((?P<config>[^\n]+)\)\s*$"
    )
    replacement = rf"\g<indent>self.{attribute} = {class_name}(\g<config>)"
    rendered, count = pattern.subn(replacement, source_text, count=1)
    return rendered if count == 1 else None


__all__ = ["compose_nested_model_sources"]
