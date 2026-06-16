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
"""Scheduler input model for activation residency planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class AccessKind(Enum):
    """How an op accesses a storage."""

    READ = auto()
    WRITE = auto()


@dataclass
class StorageAccess:
    """A storage access made by an op."""

    op_id: int
    storage_id: int
    kind: AccessKind


@dataclass
class TraceOp:
    """A scheduler-visible execution op."""

    name: str
    duration_ms: float = 0.0
    accesses: list[StorageAccess] = field(default_factory=list)


@dataclass
class ActivationTrace:
    """Scheduler input: ordered ops plus storage metadata."""

    ops: list[TraceOp] = field(default_factory=list)
    storage_sizes: dict[int, int] = field(default_factory=dict)
    retained_sids: set[int] = field(default_factory=set)
    memory_limit_bytes: int | None = None
    d2h_bandwidth_gbps: float = 16.0
    h2d_bandwidth_gbps: float = 16.0
