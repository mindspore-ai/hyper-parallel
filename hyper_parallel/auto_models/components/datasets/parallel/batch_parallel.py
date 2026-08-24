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
"""Distribute one DataLoader batch across CP and TP ranks.

The canonical batch exists only on TP rank zero for the active CP coordinate.
``CPBatchSharder`` slices ``input_ids`` and ``labels`` contiguously along the
sequence dimension. ``TPBatchBroadcaster`` then moves the CP-local tensors and
global sequence boundaries to the target device, broadcasts their metadata,
allocates receive tensors on peer TP ranks, and broadcasts the required fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyper_parallel.platform import get_platform
from hyper_parallel.auto_models.components.datasets.parallel.dataloader_parallel import DataLoaderParallelContext


platform = get_platform()


class CPBatchSharder:
    """Select the contiguous token fields owned by the current CP rank."""

    def __init__(
            self,
            parallel_context: DataLoaderParallelContext,
    ) -> None:
        """Initialize context-parallel batch sharding.

        Args:
            parallel_context: DataLoader and TP/CP topology information.
        """
        self.parallel_context = parallel_context

    def shard(
            self,
            canonical_batch: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Shard input IDs and labels along the sequence dim.

        Args:
            canonical_batch: Normalized complete batch on TP rank zero, or ``None`` elsewhere.

        Returns:
            CP-local input IDs and labels on TP rank zero.
        """
        if canonical_batch is None:
            cp_batch = None
            return cp_batch

        input_ids = canonical_batch["input_ids"]
        cp_batch = {
            "input_ids": input_ids,
            "labels": canonical_batch["labels"],
        }

        cp_size = self.parallel_context.cp_world_size
        if cp_size <= 1:
            return cp_batch

        seq_len = input_ids.shape[1]
        if seq_len % cp_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by cp_size ({cp_size})")

        cp_rank = self.parallel_context.cp_rank
        for field, value in cp_batch.items():
            local_value = platform.chunk(value, split_dim=1, split_size=cp_size, index=cp_rank)
            cp_batch[field] = local_value.contiguous()

        return cp_batch


class TPBatchBroadcaster:
    """Broadcast CP-local batch fields within one TP group."""

    def __init__(
            self,
            parallel_context: DataLoaderParallelContext,
            device: Any,
    ) -> None:
        """Initialize tensor-parallel batch transport.

        Args:
            parallel_context: DataLoader and TP/CP topology information.
            device: Device receiving the broadcast batch.
        """
        self.parallel_context = parallel_context
        self.device = device

    def broadcast(
            self,
            cp_local_batch: Mapping[str, Any] | None,
            cu_seq_lens: Any,
    ) -> dict[str, Any]:
        """Broadcast CP-local fields and global boundaries across TP ranks.

        Args:
            cp_local_batch: CP-local fields on TP rank zero, or ``None`` elsewhere.
            cu_seq_lens: Global sequence boundaries, unsharded across CP ranks.

        Returns:
            Local batch populated on every TP rank.
        """
        tp_rank = self.parallel_context.tp_rank
        # TP rank zero owns the CP-local DataLoader fields and moves them to
        # the target device before communication.
        if tp_rank == 0:
            parallel_batch = {
                field: value.to(self.device, dtype=platform.tensor_dtype.int64, non_blocking=True)
                for field, value in cp_local_batch.items()
            }
            local_cu_seq_lens = None
            if cu_seq_lens is not None:
                local_cu_seq_lens = cu_seq_lens.to(self.device, dtype=platform.tensor_dtype.int32, non_blocking=True)
        else:
            parallel_batch = None
            local_cu_seq_lens = None

        tp_size = self.parallel_context.tp_world_size
        if tp_size <= 1:
            parallel_batch["cu_seq_lens"] = local_cu_seq_lens
            return parallel_batch

        # Dynamic Online batches do not have a fixed local sequence length.
        # Broadcast compact shape metadata before receivers allocate tensors.
        if tp_rank == 0:
            batch_size, seq_len = parallel_batch["input_ids"].shape
            num_boundaries = 0 if local_cu_seq_lens is None else local_cu_seq_lens.numel()
            batch_meta = platform.tensor(
                [batch_size, seq_len, num_boundaries],
                dtype=platform.tensor_dtype.int64,
                device=self.device,
            )
        else:
            batch_meta = platform.empty((3,), dtype=platform.tensor_dtype.int64, device=self.device)

        tp_group = self.parallel_context.tp_group
        platform.broadcast(batch_meta, group=tp_group, group_src=0)

        # Non-source TP ranks allocate the same device shapes and dtypes.
        if tp_rank != 0:
            batch_size, seq_len, num_boundaries = [int(value) for value in batch_meta.tolist()]
            shape = (batch_size, seq_len)
            parallel_batch = {
                "input_ids": platform.empty(shape, dtype=platform.tensor_dtype.int64, device=self.device),
                "labels": platform.empty(shape, dtype=platform.tensor_dtype.int64, device=self.device),
            }
            if num_boundaries > 0:
                local_cu_seq_lens = platform.empty(
                    (num_boundaries,), dtype=platform.tensor_dtype.int32, device=self.device
                )

        # Broadcast only model fields that cannot be regenerated locally.
        for field in ("input_ids", "labels"):
            platform.broadcast(parallel_batch[field], group=tp_group, group_src=0)

        # Packed boundaries remain global across CP ranks and variable in size.
        if local_cu_seq_lens is not None:
            platform.broadcast(local_cu_seq_lens, group=tp_group, group_src=0)

        parallel_batch["cu_seq_lens"] = local_cu_seq_lens

        return parallel_batch
