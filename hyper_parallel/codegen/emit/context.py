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
"""In-memory inputs consumed while emitting one generated bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EmissionContext:
    """Hold frozen native-build facts for a single emission pass."""

    source: dict[str, Any]
    parallel_dims: dict[str, Any]
    status: str
    injections: list[dict[str, Any]] = field(default_factory=list)
    param_plan: dict[str, Any] = field(default_factory=dict)
    boundary_classes: dict[str, str] = field(default_factory=dict)
    nested_components: list[dict[str, str]] = field(default_factory=list)
    module_overrides: list[dict[str, Any]] = field(default_factory=list)
    materialized_patches: list[dict[str, str]] = field(default_factory=list)
    emission_patches: dict[str, Any] = field(default_factory=dict)
    external_state_classes: list[str] = field(default_factory=list)
    mesh_dim_names: Optional[list[str]] = None


__all__ = ["EmissionContext"]
