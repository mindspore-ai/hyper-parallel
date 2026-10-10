# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Generate a unified diff between the original and generated modeling files.

Kept separate from ``emit.modeling`` so the diff writer can be reused by a
CLI / review tool without pulling in the file-format logic.
"""
from __future__ import annotations

import difflib


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


__all__ = ["unified_diff"]
