# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Emit a codegen artifact bundle from a frozen plan.

The only public entry is :func:`emit_bundle` — the manager calls it as
``emit_fn(layout, meta)`` (manager.py).  The helper functions are the
building blocks that ``emit_bundle`` composes.
"""
from hyper_parallel.codegen.emit.bundle import (
    emit_bundle,
    emit_diff,
    emit_init_file,
    emit_modeling_file,
    format_generated_filename,
)

__all__ = [
    "emit_bundle",
    "emit_diff",
    "emit_init_file",
    "emit_modeling_file",
    "format_generated_filename",
]
