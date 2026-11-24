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
"""Normalize frozen meta so it matches the inline-generated source shape."""

from __future__ import annotations

import logging
from copy import copy
from typing import Any

logger = logging.getLogger(__name__)


def normalize_inline_meta(meta: Any) -> Any:
    """Copy emission metadata and protect classes shared with non-boundary instances."""
    normalized = _copy_inline_meta(meta)
    _mark_shared_class_boundaries(normalized, normalized.boundary_classes)
    return normalized


def _copy_inline_meta(meta: Any) -> Any:
    """Copy only fields normalized by source emission, preserving all other context."""
    normalized = copy(meta)
    normalized.param_plan = dict(getattr(meta, "param_plan", None) or {})
    normalized.boundary_classes = dict(
        getattr(meta, "boundary_classes", None) or {}
    )
    return normalized


def _mark_shared_class_boundaries(meta: Any, full_class_map: dict[str, str]) -> None:
    """Avoid imposing one instance's boundary on every instance of its class.

    Keep the boundary and class mapping visible, but mark the class-level
    rewrite as skipped. Native per-instance boundary application remains
    Trainer-owned; the review artifact does not install runtime state.
    """
    param_plan = getattr(meta, "param_plan", None) or {}
    boundary_classes = getattr(meta, "boundary_classes", None) or {}

    non_boundary_classes: set[str] = set()
    for fqn, entry in param_plan.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("is_boundary") is False:
            cls = full_class_map.get(fqn)
            if cls is not None:
                non_boundary_classes.add(cls)

    if not non_boundary_classes:
        return

    for fqn, class_name in list(boundary_classes.items()):
        if class_name not in non_boundary_classes:
            continue
        entry = param_plan.get(fqn)
        if isinstance(entry, dict):
            param_plan[fqn] = {**entry, "skip_source_lowering": True}
            logger.info(
                "codegen: boundary %s (class %s) marked skip_source_lowering "
                "-- the class is shared with a non-boundary instance; the "
                "class forward is preserved for native per-instance application",
                fqn, class_name,
            )


__all__ = ["normalize_inline_meta"]
