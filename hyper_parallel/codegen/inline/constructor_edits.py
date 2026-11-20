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
                if not any(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                           and target.value.id == "self" and target.attr == patch.attribute for target in targets):
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
        expression = patch.expression.replace("__source_module__", ast.get_source_segment(source, value))
        edits.append(TextEdit(start, end, expression))
    return edits
