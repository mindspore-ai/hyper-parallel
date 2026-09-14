# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Transformers-pin verification for codegen artifacts.

The generated modeling file embeds a modeling source written against some
transformers version.  ``verify_transformers_pin`` asks whether the *installed*
transformers satisfies the pin the user expects (``==5.15.0``, ``>=5.14``,
``<6`` ...). The check returns **SKIP** rather than FAIL when the
installed version does NOT satisfy the pin: the artifact is not stale because
of an environment mismatch, it just cannot be validated by the pin — so the
caller exits 0 either way, and reserves exit 1 for "the pin itself cannot be
parsed" / "the installed version cannot be read".
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# The public status the CLI maps to an exit code.
OK = "OK"
SKIP = "SKIP"


def verify_transformers_pin(spec: str) -> Tuple[str, str, str]:
    """Compare the installed transformers to ``spec``.

    Returns ``(status, installed, spec)`` where ``status`` is ``OK`` when the
    installed version satisfies ``spec`` and ``SKIP`` otherwise.  Raises
    ``ValueError`` when ``spec`` cannot be parsed and ``ImportError`` when the
    installed transformers version cannot be read (both are exit-1 conditions
    for the CLI).
    """
    installed = _installed_version()
    if installed is None:
        raise ImportError("codegen check: cannot read installed transformers version")

    from packaging.specifiers import SpecifierSet

    try:
        pin = SpecifierSet(spec)
    except Exception as exc:
        raise ValueError(f"codegen check: invalid transformers pin {spec!r}: {exc}") from exc

    if installed in pin:
        return OK, installed, spec
    logger.warning(
        "codegen check: installed transformers %s does not satisfy pin %s — "
        "SKIP (artifact is not stale, pin is not satisfiable here)",
        installed, spec,
    )
    return SKIP, installed, spec


def _installed_version() -> Optional[str]:
    """Installed transformers version, or ``None`` when not importable."""
    try:
        return _version_from_importlib_metadata()
    except ImportError:
        return None


def _version_from_importlib_metadata() -> str:
    import importlib.metadata

    return importlib.metadata.version("transformers")


__all__ = [
    "OK",
    "SKIP",
    "verify_transformers_pin",
]
