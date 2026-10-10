# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""YAML-driven inline source patches for generated modeling files."""

from hyper_parallel.codegen.inline.meta_plan import normalize_inline_meta
from hyper_parallel.codegen.inline.pipeline import render_inline_modeling

__all__ = ["normalize_inline_meta", "render_inline_modeling"]
