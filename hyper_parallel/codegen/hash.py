# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Signature and file-hash helpers for codegen artifacts.

``signature_from_spec`` is the single source of truth for artifact reuse:
same canonical spec  => same signature => artifact hit.  Any change to the
source modeling file, parallel topology, override set, or checkpoint
transform flags must change the signature.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_file(path: str) -> str:
    """Compute the sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Compute the sha256 of a UTF-8 text payload."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(data: object) -> str:
    """Stable JSON serialization (sorted keys, no whitespace).

    Used for signature inputs — key order must not affect the digest.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def signature_from_spec(spec: object, *, prefix: int = 12) -> str:
    """Compute the artifact signature from a canonical spec, first ``prefix`` hex chars.

    ``spec`` may be any JSON-serializable object (typically a
    :class:`CodegenSpec` ``to_dict()`` result or its canonical projection).
    """
    return sha256_text(canonical_json(spec))[:prefix]


def canonical_snapshot_payload(snapshot: Any) -> str:
    """Serialize the stable identity projection of a build snapshot."""
    signature_payload = getattr(snapshot, "signature_payload", None)
    if not callable(signature_payload):
        raise TypeError("snapshot must provide signature_payload()")
    return canonical_json(signature_payload())


def signature_from_snapshot(snapshot: Any, *, prefix: int = 12) -> str:
    """Compute a short signature from a build snapshot's stable facts."""
    return sha256_text(canonical_snapshot_payload(snapshot))[:prefix]


def file_line_count(path: str) -> int:
    """Count lines in a text file."""
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


__all__ = [
    "canonical_json",
    "canonical_snapshot_payload",
    "file_line_count",
    "sha256_file",
    "sha256_text",
    "signature_from_spec",
    "signature_from_snapshot",
]
