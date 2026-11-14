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
"""Sequence ownership metadata and per-call readiness receipts for CP transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from hyper_parallel.distributed.context_parallel.collectives import AsyncCPCollective


@dataclass(frozen=True)
class SequencePartition:
    """Contiguous, equal-length sequence shards in process-group rank order."""

    lengths: tuple[int, ...]
    rank: int

    @property
    def offset(self) -> int:
        """Global sequence coordinate of the first local token."""
        return sum(self.lengths[:self.rank])

    @property
    def global_length(self) -> int:
        """Number of tokens represented by this partition."""
        return sum(self.lengths)


@dataclass
class PendingSequence:
    """Per-call pending output; consumers obtain tensors only through wait()."""

    collective: AsyncCPCollective
    partition: SequencePartition

    def wait(self) -> tuple[Tensor, SequencePartition]:
        """Wait before exposing the gathered tensor and its coordinate receipt."""
        return self.collective.wait(), self.partition
