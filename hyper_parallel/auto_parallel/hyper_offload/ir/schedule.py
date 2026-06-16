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
"""Scheduler output model for residency actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class ResidencyActionType(Enum):
    """Runtime action emitted by the planner."""

    COPY_D2H = auto()
    COPY_H2D = auto()
    RELEASE_DEVICE = auto()
    RELEASE_HOST = auto()


@dataclass(frozen=True)
class ResidencyAction:
    """A scheduled runtime residency action."""

    op_id: int
    storage_id: int
    kind: ResidencyActionType


@dataclass
class ResidencySchedule:
    """Scheduler output indexed by op id."""

    pre: dict[int, list[ResidencyAction]] = field(default_factory=dict)
    post: dict[int, list[ResidencyAction]] = field(default_factory=dict)

    def add_pre(self, op_id: int, kind: ResidencyActionType, storage_id: int) -> None:
        """Add pre."""
        self.pre.setdefault(op_id, []).append(ResidencyAction(op_id, storage_id, kind))

    def add_post(self, op_id: int, kind: ResidencyActionType, storage_id: int) -> None:
        """Add post."""
        self.post.setdefault(op_id, []).append(ResidencyAction(op_id, storage_id, kind))

    def pre_actions(self, op_id: int) -> list[ResidencyAction]:
        """Pre actions."""
        return self.pre.get(op_id, [])

    def post_actions(self, op_id: int) -> list[ResidencyAction]:
        """Post actions."""
        return self.post.get(op_id, [])
