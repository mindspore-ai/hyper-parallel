# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Index a modeling source: locate classes, forwards, and import regions.

``lower_forward_boundaries`` (emit/parallel) rewrites a boundary class's
``forward`` in place.  To do that it needs three answers from the source:

1. where a top-level ``class`` starts (and how far it spans) so an edit never
   bleeds into the next class;
2. where that class's ``forward`` sits — its indentation, its first statement,
   and the byte span of its body (the part that gets replaced);
3. where the module's import block ends, so the lowerer can insert the frozen
   per-class boundary constants *before* the first class body but after the
   imports + module docstring (PEP 8: imports first).

Everything is computed with ``ast`` and recorded as 1-based line numbers +
character offsets into the *original* ``source_text``.  Editing happens on the
unmodified text (the index is never reused against a patched string), so the
offset mapping stays valid for the whole call.

Nothing here imports torch or the training stack.
"""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FunctionInfo:
    """Location of one method inside a class body."""

    name: str
    #: 1-based line of the ``def`` line (including decorators if any).
    def_line: int
    #: Byte offset of the first statement in the function body.  ``None`` if the
    #: body is just ``pass`` / ``...`` on the same line as ``def`` (single-line
    #: function) — there is nothing to replace.
    body_start: Optional[int]
    #: Byte offset just past the last statement of the body.
    body_end: Optional[int]
    #: Character offset of the ``def`` keyword (== offset of the first decorator
    #: when one is present); used to delete a decorator along with the body.
    def_offset: int
    #: Positional parameter names of the signature (BOUND to the plan's
    #: ``arg_name`` by ``sharding_applier._bind_input_indices``).  ``self`` is
    #: excluded; ``*args`` / ``**kwargs`` are not part of the positional list.
    param_names: list[str]
    #: Keyword-only parameter names (``def f(self, x, *, y=1)`` -> ``["y"]``).
    kwonly_names: list[str] = field(default_factory=list)
    #: Whether the signature carries ``*args`` / ``**kwargs``.  A statically
    #: rewritten forward must forward every argument it received, so variable
    #: parameter lists disqualify the static form.
    has_var_params: bool = False


@dataclass
class ClassInfo:
    """Location of one top-level class definition."""

    name: str
    #: 1-based line of the ``class`` keyword (decorators excluded).
    class_line: int
    #: Byte offset of the ``class`` keyword (first decorator, if any).
    class_offset: int
    #: 1-based line just past the class body (the next top-level statement or EOF).
    end_line: int
    #: Byte offset just past the class body's last statement.
    end_offset: int
    #: Methods keyed by name (only top-level ``def`` in the class block).
    methods: dict[str, FunctionInfo]


@dataclass
class SourceIndex:
    """Structural map of one modeling source text."""

    source_text: str
    classes: dict[str, ClassInfo]
    #: Byte offsets of the module-level import region (``from ... import`` /
    #: ``import ...``).  ``(start, end)`` is the outer span; ``end`` marks where
    #: the import block ends and code/classes may begin.
    import_start: Optional[int]
    import_end: Optional[int]
    #: Byte offset of the module docstring's closing quote (``None`` if absent).
    docstring_end: Optional[int]

    def find_class(self, name: str) -> Optional[ClassInfo]:
        """Return the ``ClassInfo`` for a top-level class, or ``None``."""
        return self.classes.get(name)

    def find_forward(self, class_name: str) -> Optional[FunctionInfo]:
        """Return the ``FunctionInfo`` for ``class_name.forward``, or ``None``.

        A class without a ``forward`` (not a module) returns ``None``; the
        caller decides whether that is an error.
        """
        info = self.classes.get(class_name)
        if info is None:
            return None
        return info.methods.get("forward")

    def find_imports(self) -> tuple[Optional[int], Optional[int]]:
        """Return ``(import_start, import_end)``, or ``(None, None)``."""
        return self.import_start, self.import_end


def build_source_index(source_text: str) -> SourceIndex:
    """Parse ``source_text`` and answer the three structural questions.

    The index is built against the *original* text.  Caller must apply edits to
    the same unmodified string (edits.py ``apply_edits`` splices by offset).
    """
    tree = ast.parse(source_text)

    lines = source_text.splitlines(keepends=True)
    line_offsets = _line_offsets(lines)

    def offset_for(line_no: int) -> int:
        """Byte offset of the start of ``line_no`` (1-based)."""
        if 1 <= line_no <= len(line_offsets):
            return line_offsets[line_no - 1]
        return len(source_text)

    classes: dict[str, ClassInfo] = {}
    import_start: Optional[int] = None
    import_end: Optional[int] = None
    docstring_end: Optional[int] = None

    body = tree.body
    # Module docstring: the first statement is a constant string (PEP 257).  Its
    # closing quote is the first insertion point for appended constants, so
    # Frozen constants belong after the docstring and before imports.
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str) and body[0].end_lineno is not None:
            docstring_end = offset_for(body[0].end_lineno)

    # Import block: the leading (or maximal) run of Import/ImportFrom at module
    # level.  In a well-formed module imports are first, so we take the span
    # from the first import to the last import before the first non-import
    # statement.  Anything after the docstring but before the first import (a
    # comment, ``__future__`` in a string, a module-level flag) is skipped.
    first_non_import: Optional[ast.AST] = None
    for node in body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            first_non_import = node
            break
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if import_start is None:
                import_start = offset_for(node.lineno)
            if node.end_lineno is not None:
                import_end = offset_for(node.end_lineno)

    for node in body:
        if isinstance(node, ast.ClassDef):
            class_info = _build_class_info(node, source_text, offset_for)
            classes[class_info.name] = class_info

    return SourceIndex(
        source_text=source_text,
        classes=classes,
        import_start=import_start,
        import_end=import_end,
        docstring_end=docstring_end,
    )


def _build_class_info(node: ast.ClassDef, text: str, offset_for) -> ClassInfo:
    methods: dict[str, FunctionInfo] = {}
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[child.name] = _build_function_info(child, text, offset_for)
    end_offset = (
        offset_for(node.end_lineno) if node.end_lineno is not None else len(text)
    )
    return ClassInfo(
        name=node.name,
        class_line=node.lineno,
        class_offset=offset_for(node.lineno),
        end_line=node.end_lineno if node.end_lineno is not None else node.lineno,
        end_offset=end_offset,
        methods=methods,
    )


def _build_function_info(node: ast.AST, text: str, offset_for) -> FunctionInfo:
    decorators = getattr(node, "decorator_list", [])
    first_def_line = node.lineno
    if decorators:
        first_def_line = min(decorators[0].lineno, first_def_line)
    def_offset = offset_for(first_def_line)

    # Positional parameters (``self`` excluded) — the names the boundary's
    # ``arg_name`` binds against.  Only POSITIONAL_ONLY / POSITIONAL_OR_KEYWORD
    # are part of a positional call; ``*args`` / ``**kwargs`` are not.
    params = getattr(node, "args", None)
    positional: list[str] = []
    kwonly: list[str] = []
    has_var_params = False
    if params is not None:
        for arg in params.posonlyargs + params.args:
            if arg.arg != "self":
                positional.append(arg.arg)
        kwonly = [arg.arg for arg in params.kwonlyargs]
        has_var_params = params.vararg is not None or params.kwarg is not None

    body = node.body
    body_start: Optional[int] = None
    body_end: Optional[int] = None
    # A function whose body is a single `pass` / `...` / single constant
    # statement on the same line has no replaceable body span.
    has_body = any(
        not (
            isinstance(stmt, ast.Pass)
            or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis)
        )
        for stmt in body
    )
    if has_body:
        first_stmt = body[0]
        body_start = offset_for(first_stmt.lineno)
        last_stmt = body[-1]
        # End just past the last statement's final character AND its trailing
        # newline (when present).  ``offset_for(end_lineno)`` is only the START
        # of that line — using it here would leave the last statement (e.g.
        # ``return x``) outside the replaceable span, so a body swap would
        # splice a stray leftover statement after the replacement.  The
        # start-of-line offset plus the statement's column + 1 lands just past
        # the line's newline; when the statement is the last line and the file
        # has no trailing newline, that would overshoot EOF, so clamp to
        # ``len(text)``.
        body_end = (
            min(offset_for(last_stmt.end_lineno) + last_stmt.end_col_offset + 1, len(text))
            if last_stmt.end_lineno is not None
            else len(text)
        )
    return FunctionInfo(
        name=node.name,
        def_line=first_def_line,
        body_start=body_start,
        body_end=body_end,
        def_offset=def_offset,
        param_names=positional,
        kwonly_names=kwonly,
        has_var_params=has_var_params,
    )


def returns_single_value(source_text: str, func: FunctionInfo) -> bool:
    """Whether every ``return`` in the function returns a bare single value.

    ``return`` / ``return x`` / ``return f(x)`` are single; ``return a, b`` /
    ``return (a, b)`` / ``return [a, b]`` / ``return *a,`` are not.  The
    compiled boundary plan indexes into a returned sequence by
    ``arg_index``, so a static single-tensor operator call can only replace
    the output redistribution when the return is a single value.

    The function's ``def`` span (decorators skipped) is re-parsed on its own;
    a parse failure fails closed (``False``) — the caller demotes the static
    form rather than guessing.  The span starts at the ``def`` line's first
    byte, which for a class method carries the method indent; ``dedent``
    strips that common prefix so the isolated def parses as a top-level
    statement (an indented first line would raise ``IndentationError`` and
    fail every method closed).
    """
    if func.body_start is None or func.body_end is None:
        return False
    def_line_start = _def_keyword_offset(source_text, func)
    span = textwrap.dedent(source_text[def_line_start:func.body_end])
    try:
        tree = ast.parse(span)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            value = node.value
            if value is None:
                continue
            if isinstance(value, (ast.Tuple, ast.List, ast.Starred)):
                return False
    return True


def _def_keyword_offset(source_text: str, func: FunctionInfo) -> int:
    """Byte offset of the ``def`` keyword line, skipping decorator lines.

    ``func.def_offset`` anchors the FIRST line of the method (a decorator when
    one is present); the span fed to ``ast.parse`` must start at the ``def``
    keyword itself.  Shared logic with ``_build_forward_impl_edit`` in
    emit/parallel (kept local — three lines, not worth a cross-module helper).
    """
    def_line_start = func.def_offset
    for line in source_text[def_line_start:func.body_start].splitlines(keepends=True):
        if line.lstrip().startswith("def "):
            break
        def_line_start += len(line)
    return def_line_start


def _line_offsets(lines: list[str]) -> list[int]:
    """Byte offset of each line's start (1-based index into the result)."""
    offsets: list[int] = []
    acc = 0
    for line in lines:
        offsets.append(acc)
        acc += len(line)
    offsets.append(acc)
    return offsets
