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
"""Reusable sequence transport over an existing HyperParallel CP mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from hyper_parallel.distributed.context_parallel.collectives import (
    AsyncCPCollective, async_cp_allgather_launch, ulysses_head_to_seq, ulysses_seq_to_head,
)


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


class CPContext:
    """Transport only: model families retain projection, mask and attention math."""

    def __init__(self, mesh: Any) -> None:
        """Borrow a one-dimensional CP mesh without creating process groups."""
        if mesh is None or mesh.ndim != 1:
            raise ValueError("CPContext requires an existing one-dimensional HP mesh")
        self.mesh = mesh
        self.size = mesh.size()
        self.rank = mesh.get_local_rank()

    def partition(self, local_length: int) -> SequencePartition:
        """Describe equal contiguous source shards without reading activations.

        Args:
            local_length: Sequence length on each rank of the CP group.
        """
        if local_length < 0:
            raise ValueError("local sequence length cannot be negative")
        return SequencePartition((local_length,) * self.size, self.rank)

    def launch_gather_sequence(self, tensor: Tensor, *, dim: int = 1) -> PendingSequence:
        """Launch differentiable sequence gathering with explicit readiness.

        Args:
            tensor: Local activation shard, with equal lengths on all CP ranks.
            dim: Sequence dimension; rank order defines global token coordinates.
        """
        partition = self.partition(tensor.shape[dim])
        return PendingSequence(async_cp_allgather_launch(tensor, dim, self.mesh), partition)

    def gather_sequence(self, tensor: Tensor, *, dim: int = 1) -> tuple[Tensor, SequencePartition]:
        """Gather and wait synchronously; backward sums consumers into source shards.

        Args:
            tensor: Local sequence shard.
            dim: Sequence dimension.
        """
        return self.launch_gather_sequence(tensor, dim=dim).wait()

    def sequence_to_head(self, tensor: Tensor, *, seq_dim: int = 1, head_dim: int = 2) -> Tensor:
        """Exchange sequence-sharded activations into head-sharded activations.

        Args:
            tensor: Local sequence shard with divisible head count.
            seq_dim: Sequence dimension.
            head_dim: Head dimension.
        """
        return ulysses_seq_to_head(tensor, seq_dim=seq_dim, head_dim=head_dim, cp_mesh=self.mesh)

    def head_to_sequence(self, tensor: Tensor, *, seq_dim: int = 1, head_dim: int = 2) -> Tensor:
        """Apply the inverse head-to-sequence exchange.

        Args:
            tensor: Local head shard with divisible sequence length.
            seq_dim: Sequence dimension.
            head_dim: Head dimension.
        """
        return ulysses_head_to_seq(tensor, seq_dim=seq_dim, head_dim=head_dim, cp_mesh=self.mesh)
