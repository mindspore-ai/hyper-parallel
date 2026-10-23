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
"""Omni-only CP slicing and dtype-preserving TP transport."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# This opt-in transport belongs to the Torch VLMTrainer, not backend-neutral core.
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.parallel.dataloader_parallel import DataLoaderParallelContext


class OmniCPBatchSharder:
    """Slice token-aligned fields; preserve full modality tensors and coordinates."""

    def __init__(self, parallel_context: DataLoaderParallelContext, token_pad_values: dict[str, int]) -> None:
        """Store CP topology and explicit token-field padding rules."""
        self.parallel_context = parallel_context
        self.token_pad_values = token_pad_values

    def shard(self, canonical_batch: Mapping[str, Any]) -> dict[str, Any]:
        """Shard configured token fields and preserve all modality metadata."""
        cp_batch = dict(canonical_batch)
        cp_size = self.parallel_context.cp_world_size
        if cp_size == 1:
            return cp_batch
        input_ids = cp_batch["input_ids"]
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Configured CP sharding requires input_ids with shape [1, sequence]")

        sequence_length = input_ids.shape[1]
        pad_length = (-sequence_length) % (2 * cp_size)
        if "cu_seq_lens" in cp_batch:
            boundaries = cp_batch["cu_seq_lens"].reshape(-1)
            if int(boundaries[0]) != 0 or int(boundaries[-1]) != sequence_length:
                raise ValueError("cu_seq_lens must cover the complete CP input sequence")

            if pad_length:
                boundaries = torch.cat((boundaries, boundaries[-1:] + pad_length))
            cp_batch["cu_seq_lens"] = boundaries

        chunk_length = (sequence_length + pad_length) // cp_size
        local_start = self.parallel_context.cp_rank * chunk_length
        local_end = local_start + chunk_length
        for field, pad_value in self.token_pad_values.items():
            if field not in cp_batch:
                continue

            value = cp_batch[field]
            if not torch.is_tensor(value) or value.ndim != 2 or value.shape != input_ids.shape:
                raise ValueError(f"CP token field {field!r} must match input_ids shape")

            padded = torch.nn.functional.pad(value, (0, pad_length), value=pad_value)
            cp_batch[field] = padded[:, local_start:local_end].contiguous()

        return cp_batch


class OmniTPBatchBroadcaster:
    """Send field metadata before broadcasting heterogeneous tensors."""

    def __init__(self, parallel_context: DataLoaderParallelContext, device: Any) -> None:
        """Store transport topology and receiver device."""
        self.parallel_context = parallel_context
        self.device = device

    def broadcast(self, cp_local_batch: Mapping[str, Any] | None) -> dict[str, Any]:
        """Broadcast variable-shaped Omni tensors and scalar metadata."""
        if self.parallel_context.tp_world_size == 1:
            return self._move_mapping_to_device(cp_local_batch)
        is_source = self.parallel_context.tp_rank == 0
        if is_source:
            parallel_batch = self._move_mapping_to_device(cp_local_batch)
            metadata = []
            for name, value in parallel_batch.items():
                if torch.is_tensor(value):
                    metadata.append((name, tuple(value.shape), value.dtype, None))
                else:
                    metadata.append((name, None, None, value))
        else:
            parallel_batch = {}
            metadata = None
        object_list = [metadata]
        group = self.parallel_context.tp_group
        source_rank = dist.get_process_group_ranks(group)[0]
        dist.broadcast_object_list(
            object_list, group=group, src=source_rank, device=self.device
        )
        for name, shape, dtype, value in object_list[0]:
            if shape is None:
                parallel_batch[name] = value
                continue
            if not is_source:
                parallel_batch[name] = torch.empty(shape, dtype=dtype, device=self.device)
            dist.broadcast(parallel_batch[name], group=group, src=source_rank)
        return parallel_batch

    def _move_mapping_to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Transfer Omni tensors without changing their native dtypes."""
        return {
            field: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for field, value in batch.items()
        }
