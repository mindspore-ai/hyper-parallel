# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Generate a unified diff between the original and generated modeling files.

Kept separate from ``emit.modeling`` so the diff writer can be reused by a
CLI / review tool without pulling in the file-format logic.
"""
from __future__ import annotations

import difflib
import os


def unified_diff(
    original: str,
    generated: str,
    fromfile: str,
    tofile: str,
) -> str:
    """Return a unified diff from ``original`` to ``generated``.

    Both inputs are the full source text.  ``fromfile``/``tofile`` become the
    diff headers (e.g. ``modeling_deepseek_v3.py`` and
    ``modeling_deepseek_v3_gen_npu.py``).  An empty result means the two files
    are identical.
    """
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def write_diff(diff_text: str, diff_path: str) -> None:
    """Write ``diff_text`` to ``diff_path`` (creating parents as needed)."""
    parent = os.path.dirname(diff_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(diff_path, "w", encoding="utf-8") as handle:
        handle.write(diff_text)


__all__ = ["unified_diff", "write_diff"]
