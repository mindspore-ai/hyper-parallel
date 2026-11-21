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
"""Render top-level imports and source snippets from inline patch IR."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable

from hyper_parallel.codegen.astkit.edits import TextEdit, insert_after_imports
from hyper_parallel.codegen.inline.ir import ImportPatch, ModuleSnippetPatch


def declaration_edits(
    source_text: str,
    import_end: int | None,
    docstring_end: int | None,
    imports: Iterable[ImportPatch],
    snippets: Iterable[ModuleSnippetPatch],
) -> list[TextEdit]:
    """Build ordered edits for generated imports and top-level snippets."""
    edits = []
    import_nodes = [node for node in ast.parse(source_text).body if isinstance(node, (ast.Import, ast.ImportFrom))]
    if import_nodes:
        # Component bases must follow the complete last import, not its line start.
        import_end = sum(len(line) for line in source_text.splitlines(keepends=True)[:import_nodes[-1].end_lineno])
    lines = _render_imports(source_text, imports)
    if lines:
        edits.append(insert_after_imports(
            source_text,
            import_end,
            "\n" + "\n".join(lines) + "\n",
            default_offset=docstring_end or 0,
        ))
    parts = _render_snippets(source_text, snippets)
    if parts:
        edits.append(insert_after_imports(
            source_text,
            import_end,
            "\n\n" + "\n\n".join(parts) + "\n",
            default_offset=docstring_end or 0,
        ))
    return edits


def _render_snippets(source_text: str, snippets: Iterable[ModuleSnippetPatch]) -> list[str]:
    """Keep shared native definitions once, rejecting conflicting source bindings."""
    definitions = _declared_bindings(source_text)
    parts = []
    for snippet in snippets:
        text = snippet.text.strip("\n")
        if not text or text in parts:
            continue
        current = _declared_bindings(text)
        for name, definition in current.items():
            if name in definitions and definitions[name] != definition:
                raise ValueError(f"codegen: conflicting source definition {name!r}")
        if current and current.keys() <= definitions.keys():
            continue
        definitions.update(current)
        parts.append(text)
    return parts


def _declared_bindings(source: str) -> dict[str, str]:
    """Index native definitions and early-bound aliases without evaluating them."""
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = ast.dump(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                        result[name.id] = ast.dump(node)
    return result


def _render_imports(
    source_text: str, imports: Iterable[ImportPatch]
) -> list[str]:
    """Render import patches as deduplicated, grouped import lines."""
    grouped: dict[str, set[str]] = defaultdict(set)
    lines: list[str] = []
    for patch in imports:
        if patch.raw:
            if patch.raw not in source_text and patch.raw not in lines:
                lines.append(patch.raw)
            continue
        grouped[patch.module].update(patch.names)
    for module in sorted(grouped):
        line = f"from {module} import {', '.join(sorted(grouped[module]))}"
        if line not in source_text:
            lines.append(line)
    return lines


__all__ = ["declaration_edits"]
