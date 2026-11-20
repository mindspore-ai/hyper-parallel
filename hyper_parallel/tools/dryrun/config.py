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
"""Dry-run configuration for logical LLM memory analysis."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class DryRunSnapshotConfig:
    """Logical memory snapshot output settings."""

    enabled: bool = True
    path: str = "outputs/dry_run"
    capture_stacks: bool = True
    stack_depth: int = 64


@dataclass
class DryRunConfig:
    """Shape and value-profile configuration for logical LLM memory analysis."""

    enabled: bool = True
    target_device: Literal["npu", "cuda"] = "npu"
    ranks: Optional[list[int]] = None
    snapshot: DryRunSnapshotConfig = field(default_factory=DryRunSnapshotConfig)
    virtual_world_size: Optional[int] = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DryRunConfig":
        """Build a config while resolving the nested snapshot mapping."""
        fields = dict(value)
        snapshot = fields.get("snapshot")
        if isinstance(snapshot, Mapping):
            fields["snapshot"] = DryRunSnapshotConfig(**dict(snapshot))
        return cls(**fields)
