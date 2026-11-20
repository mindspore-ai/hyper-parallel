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
"""Remove copied definitions shadowing native helper imports."""

from __future__ import annotations

import ast
from collections.abc import Iterable

from hyper_parallel.codegen.astkit.edits import TextEdit, apply_edits


def drop_shadowed_definitions(text: str, imported_names: Iterable[str]) -> str:
    """Remove decorated definitions shadowing imported runtime helpers."""
    names = {name for name in imported_names if name}
    if not names:
        return text
    tree = ast.parse(text)
    offsets = _line_offsets(text)
    edits = [
        TextEdit(start, _skip_trailing_blank_lines(text, end), "")
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in names
        and node.decorator_list
        for start, end in (_node_span(node, offsets),)
    ]
    return apply_edits(text, edits) if edits else text


def _node_span(node: ast.AST, offsets: tuple[int, ...]) -> tuple[int, int]:
    """Return a node span including decorators."""
    line, column = node.lineno, node.col_offset
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        first = min(decorators, key=lambda item: item.lineno)
        if first.lineno < line:
            line, column = first.lineno, max(0, first.col_offset - 1)
    return (
        offsets[line - 1] + column,
        offsets[node.end_lineno - 1] + node.end_col_offset,
    )


def _line_offsets(source_text: str) -> tuple[int, ...]:
    """Return absolute offsets for source line starts."""
    offsets = [0]
    for line in source_text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return tuple(offsets)


def _skip_trailing_blank_lines(text: str, end: int) -> int:
    """Advance beyond whitespace following a deleted definition."""
    while end < len(text) and text[end] in " \t":
        end += 1
    while end < len(text) and text[end] == "\n":
        end += 1
        while end < len(text) and text[end] in " \t":
            end += 1
    return end


__all__ = ["drop_shadowed_definitions"]
