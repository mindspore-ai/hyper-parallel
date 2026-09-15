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
"""Sanitize modeling-source imports against the installed transformers.

Generated modeling files embed their origin ``modeling_*.py`` verbatim.  For
trusted-remote-code models that file is the version shipped inside the
checkpoint — it reflects whatever transformers the checkpoint author used, not
the one the artifact runs against.  When that installed transformers is *newer*
in a breaking way, the verbatim source can fail to import even though the
guarded behavior is now redundant.

The compatibility fix handles ``is_torch_fx_available``,
removed from ``transformers.utils.import_utils`` in transformers v5.0.0.  Its
only consumer guard is no longer load-bearing (torch >= 2.x imports ``torch.fx``
unconditionally), so the importing statement is dropped and a module-scope
callable fallback is spliced in its place, keeping the original
``if is_torch_fx_available():`` guard resolving and truthful on transformers v4
and v5 alike, with the guard body byte-identical.

Deliberately narrow — one symbol, AST-located edits, no-op for every source
that does not import it.  Any other stale-import breakage is reported, not
papered over (a new compat rule is an explicit, documented decision).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from hyper_parallel.codegen.astkit.edits import TextEdit, apply_edits

# The symbol transformers v5.0.0 removed.
_REMOVED = "is_torch_fx_available"

# Spliced where the import statement was.  The guard CALLS the symbol, so this
# must be a callable returning truthy — not a bare True (``if True():`` would
# raise ``'bool' object is not callable``).
_FALLBACK = (
    "# Codegen source compatibility: transformers >= 5.0 removed\n"
    "# this name from ``transformers.utils.import_utils``; torch >= 2.x imports\n"
    "# ``torch.fx`` unconditionally, so this (4.x-era) source's guard is always\n"
    "# true.  Keep it callable so the guard resolves and stays truthful here.\n"
    "def is_torch_fx_available():\n"
    "    return True\n"
)


@dataclass
class _Stmt:
    """One ``from`` import statement, resolved to byte spans and symbol names."""

    start: int  # byte offset of the ``from`` keyword
    end: int  # byte offset just past the statement's trailing newline
    module: str
    symbols: list[tuple[str, str | None]]


def sanitize_source_compat(text: str) -> str:
    """Drop ``is_torch_fx_available`` imports from a modeling source.

    Returns ``text`` unchanged when the source does not import the symbol (the
    common case — byte-for-byte).  When a statement imports it, that statement is
    dropped and a module-scope callable fallback is spliced at its position (so
    the symbol is bound before any guard below).  Other imports are untouched.
    """
    offenders = [
        stmt
        for stmt in _import_statements(text)
        if stmt.module in {"transformers.utils", "transformers.utils.import_utils"}
        and any(name == _REMOVED for name, _alias in stmt.symbols)
    ]
    if not offenders:
        return text

    edits = [TextEdit(stmt.start, stmt.end, _replacement(stmt)) for stmt in offenders]
    return apply_edits(text, edits)


def _replacement(stmt: _Stmt) -> str:
    """Render the surviving names from one affected import plus the fallback."""
    remaining = [(name, alias) for name, alias in stmt.symbols if name != _REMOVED]
    if not remaining:
        return _FALLBACK
    rendered = ", ".join(
        f"{name} as {alias}" if alias is not None else name for name, alias in remaining
    )
    return f"from {stmt.module} import {rendered}\n{_FALLBACK}"


# --- parse + locate offending import statements --------------------------------


def _line_offsets(text: str) -> list[int]:
    """Byte offset of each line's start (``offsets[0] == 0``)."""
    offsets = [0]
    for ln in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(ln))
    return offsets


def _import_statements(text: str) -> list[_Stmt]:
    """Return top-level ``from ... import ...`` statements with source spans."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    offsets = _line_offsets(text)
    statements = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        if end < len(text) and text[end] == "\n":
            end += 1
        statements.append(
            _Stmt(
                start=start,
                end=end,
                module=node.module,
                symbols=[(alias.name, alias.asname) for alias in node.names],
            )
        )
    return statements
