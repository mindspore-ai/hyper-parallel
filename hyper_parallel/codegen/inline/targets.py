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
"""Normalize callable targets recorded by the native model build."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def projected_target_path(value: Any) -> str | None:
    """Return the dotted target represented by a frozen observation value."""
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, Mapping):
        target = value.get("_target_") or value.get("target")
        return str(target) if target else None
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    return f"{module}.{qualname}" if module and qualname else None


def import_target_path(value: Any) -> str:
    """Return an importable ``module:qualname`` or reject an invalid target."""
    target = projected_target_path(value)
    if not target:
        raise ValueError(f"snapshot injection target is not importable: {target!r}")
    if ":" in target:
        return target
    module_name, separator, name = target.rpartition(".")
    if not separator or not module_name or not name:
        raise ValueError(f"snapshot injection target is not importable: {target!r}")
    return f"{module_name}:{name}"


__all__ = ["import_target_path", "projected_target_path"]
