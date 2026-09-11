# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Positional text edits over a modeling source.

The forward lowerer (emit/parallel) turns a collection of structural plan
(index.py) into a list of ``TextEdit``s and replays them.  All offsets are byte
offsets into the *original* text; ``apply_edits`` validates ordering and never
edits the same span twice, so a malformed edit set fails loudly instead of
silently corrupting the file.

Only the edits are positional; every helper that builds an edit from a
``FunctionInfo`` / span does the spelling itself.  Nothing here imports torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class TextEdit:
    """One ``[start, end)`` byte-span replacement in a source string."""

    start: int
    end: int
    replacement: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(
                f"TextEdit has an invalid span: start={self.start} end={self.end}"
            )


def apply_edits(text: str, edits: Sequence[TextEdit]) -> str:
    """Replay ``edits`` on ``text``, earliest-first for stable ordering.

    Edits are authored against the *original* text.  Applying one splice shifts
    every later offset by the edit's length delta, so each edit is re-anchored
    by the cumulative drift of the edits before it.  Edits must be
    non-overlapping and stay within the original text — checked up front.
    """
    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    if ordered:
        prev_end = ordered[0].start
        for edit in ordered:
            if edit.start < prev_end:
                raise ValueError(
                    f"apply_edits: overlapping edits — [{edit.start}, {edit.end}) "
                    f"overlaps the previous edit ending at {prev_end}"
                )
            if edit.end > len(text):
                raise ValueError(
                    f"apply_edits: edit [{edit.start}, {edit.end}) exceeds text "
                    f"length {len(text)}"
                )
            prev_end = edit.end

    out = list(text)
    shift = 0
    for edit in ordered:
        start = edit.start + shift
        end = edit.end + shift
        replacement = edit.replacement
        out[start:end] = replacement
        shift += len(replacement) - (edit.end - edit.start)
    return "".join(out)


def replace_function_body(
    source_text: str,
    func: "FunctionInfo",
    new_body: str,
    *,
    indent: str = "    ",
) -> tuple[TextEdit, ...]:
    """Build the edit that swaps ``func``'s body for ``new_body``.

    ``new_body`` is the replacement body WITHOUT line indentation — every line
    is re-indented to the method's existing indent (``func``'s first body
    statement).  A docstring inside the body is preserved by the caller (the
    lowerer emits it explicitly).  A single-line/pass body has no replaceable
    span: the whole method is rewritten as ``def <name>(...):\n<indented body>``
    from the file text's own signature line.

    Returns an empty tuple when there is nothing to replace (a function with
    no body span and no way to anchor).
    """
    if func.body_start is None or func.body_end is None:
        return ()
    # Re-indent per line, preserving blank lines and inner indentation.  The
    # caller renders ``new_body`` at column-0-relative depth (blank lines carry
    # no indentation), so a statement's relative nesting survives; only the
    # method's own ``indent`` is applied once.
    body_lines = [
        (indent + line) if line.strip() else ""
        for line in new_body.splitlines(keepends=True)
    ]
    replacement = "".join(body_lines)
    return (TextEdit(func.body_start, func.body_end, replacement),)


def insert_after_imports(
    source_text: str,
    import_end: Optional[int],
    insert_text: str,
    *,
    default_offset: int = 0,
) -> Optional[TextEdit]:
    """Build an edit inserting ``insert_text`` right after the import block.

    ``import_end`` is the byte offset just past the last import statement.  A
    ``None`` ``import_end`` means no import block: fall back to
    ``default_offset`` (the docstring end or 0).  Returns ``None`` when there is
    no sensible anchor — caller treats that as "insert at the top".
    """
    anchor = import_end if import_end is not None else default_offset
    if anchor < 0 or anchor > len(source_text):
        return None
    # Keep the inserted text on its own lines, bracketed by newlines, so it
    # never glues onto a trailing comment or the first statement.
    return TextEdit(anchor, anchor, "\n" + insert_text + "\n")


def append_module_footer(
    source_text: str,
    append_text: str,
) -> TextEdit:
    """Build an edit appending ``append_text`` (with a leading newline) at EOF."""
    return TextEdit(len(source_text), len(source_text), "\n" + append_text + "\n")


__all__ = [
    "TextEdit",
    "apply_edits",
    "append_module_footer",
    "insert_after_imports",
    "replace_function_body",
]
