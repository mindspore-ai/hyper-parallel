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

from typing import Any

from torch import Tensor

from hyper_parallel.distributed.context_parallel.collectives import (
    AsyncCPCollective, _build_hybrid_cp_submeshes, async_cp_allgather_launch,
    async_ulysses_seq_to_head_launch, ulysses_head_to_seq, ulysses_seq_to_head,
)
from hyper_parallel.distributed.context_parallel.cp_types import PendingSequence, SequencePartition


class CPContext:
    """Transport only: model families retain projection, mask and attention math."""

    def __init__(self, mesh: Any) -> None:
        """Borrow a one-dimensional CP mesh without creating process groups."""
        if mesh is None or getattr(mesh, "ndim", 1) != 1:
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

    def launch_sequence_to_head(self, tensor: Tensor, *, seq_dim: int = 1,
                                head_dim: int = 2) -> AsyncCPCollective:
        """Launch the existing differentiable A2A; wait before consuming its output.

        Args:
            tensor: Input in sequence-partitioned layout.
            seq_dim: Sequence axis.
            head_dim: Head axis.
        """
        return async_ulysses_seq_to_head_launch(tensor, seq_dim, head_dim, self.mesh)

    def launch_head_to_sequence(self, tensor: Tensor, *, seq_dim: int = 1,
                                head_dim: int = 2) -> AsyncCPCollective:
        """Launch the inverse A2A without changing its existing autograd contract.

        Args:
            tensor: Input in head-partitioned layout.
            seq_dim: Sequence axis.
            head_dim: Head axis.
        """
        return async_ulysses_seq_to_head_launch(tensor, head_dim, seq_dim, self.mesh)

    def split(self, ulysses_degree: int) -> tuple[CPContext, CPContext]:
        """Borrow cached Ulysses/query-owner submeshes at bind time, never in forward.

        Args:
            ulysses_degree: Number of CP ranks partitioning attention heads.
        """
        if ulysses_degree < 1 or self.size % ulysses_degree:
            raise ValueError("Ulysses degree must divide the CP mesh size")
        if ulysses_degree in (1, self.size):
            return self, self
        heads, queries = _build_hybrid_cp_submeshes(self.mesh, ulysses_degree)
        return CPContext(heads), CPContext(queries)
