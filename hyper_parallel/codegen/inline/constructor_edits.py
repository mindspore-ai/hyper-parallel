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
"""Source-position edits for native module construction results."""

from __future__ import annotations

import ast

from hyper_parallel.codegen.astkit.edits import TextEdit
from hyper_parallel.codegen.inline.ir import ConstructorPatch


class _SourceArgument(ast.NodeTransformer):
    """Substitute syntax nodes without altering string literals in arguments."""

    def __init__(self, source: ast.Call) -> None:
        self.source = source

    def visit_Name(self, node: ast.Name) -> ast.AST:  # pylint: disable=invalid-name
        if node.id == '__source_module__':
            return self.source
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:  # pylint: disable=invalid-name
        if len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == '__source_arguments__':
            return ast.copy_location(ast.Call(func=node.func, args=self.source.args,
                                             keywords=self.source.keywords), node)
        return self.generic_visit(node)


def constructor_edits(source: str, patches: list[ConstructorPatch]) -> list[TextEdit]:
    """Edit only an explicitly identified source class and attribute assignment."""
    classes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for patch in patches:
        host = classes.get(patch.owner_class)
        initializers = [] if host is None else [
            method for method in host.body if isinstance(method, ast.FunctionDef) and method.name == "__init__"
        ]
        matches = []
        for method in initializers:
            for node in ast.walk(method):
                targets = node.targets if isinstance(node, ast.Assign) else (
                    [node.target] if isinstance(node, ast.AnnAssign) else []
                )
                if not any(isinstance(target, ast.Attribute) and ast.unparse(target.value) == patch.base
                           and target.attr == patch.attribute for target in targets):
                    continue
                value = node.value
                if isinstance(value, ast.Call) and ast.unparse(value.func).rsplit(".", 1)[-1] == patch.source_class:
                    matches.append(value)
        if len(matches) != 1:
            raise ValueError(
                f"Codegen construction {patch.owner_class}.{patch.attribute}: expected one "
                f"{patch.source_class} constructor, found {len(matches)}"
            )
        value = matches[0]
        start = offsets[value.lineno - 1] + len(lines[value.lineno - 1].encode()[:value.col_offset].decode())
        end = offsets[value.end_lineno - 1] + len(lines[value.end_lineno - 1].encode()[:value.end_col_offset].decode())
        if patch.expression.isidentifier():
            value.func = ast.Name(id=patch.expression, ctx=ast.Load())
            expression = ast.unparse(value)
        else:
            expression = ast.unparse(_SourceArgument(value).visit(ast.parse(patch.expression, mode='eval').body))
        edits.append(TextEdit(start, end, expression))
    return edits
