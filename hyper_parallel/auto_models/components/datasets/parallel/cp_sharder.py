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
"""Dataset-owned Context Parallel sequence-field sharding."""

from collections.abc import Collection, Mapping
from typing import Any

from hyper_parallel.platform import get_platform
from hyper_parallel.auto_models.components.datasets.parallel.batch_context import (
    BatchParallelContext,
)

platform = get_platform()

_SEQUENCE_LENGTH_FIELDS = {"seq_lens", "seq_lens_padded"}
_SEQUENCE_LENGTH_SENTINEL = -1000


class ContextParallelBatchSharder:
    """Shard Dataset batch sequences using only CP rank and size.

    This component intentionally has no dependency on model-side CP utilities,
    attention implementations, or K/V communication. Dataset CP and model CP
    only agree on the meaning of ``cp_rank`` and ``cp_size``.
    """

    def __init__(self, parallel_context: BatchParallelContext) -> None:
        """Store the Dataset batch CP rank and size."""
        if parallel_context.cp_size <= 0:
            raise ValueError("cp_size must be positive")
        if not 0 <= parallel_context.cp_rank < parallel_context.cp_size:
            raise ValueError(
                f"cp_rank must be in [0, {parallel_context.cp_size}), "
                f"got {parallel_context.cp_rank}"
            )
        self.cp_rank = parallel_context.cp_rank
        self.cp_size = parallel_context.cp_size

    def shard(
            self,
            batch: Mapping[str, Any],
            sequence_fields: Collection[str],
    ) -> dict[str, Any]:
        """Shard selected Dataset fields along the sequence dimension.

        Args:
            batch: Device batch available on the current rank.
            sequence_fields: Dataset fields whose final dimension is a text sequence.

        Returns:
            Batch containing the current CP rank's contiguous sequence interval.
        """
        if self.cp_size <= 1:
            sharded_batch = dict(batch)
            return sharded_batch

        working_batch = self._ensure_position_ids(batch, sequence_fields)
        try:
            input_ids = working_batch["input_ids"]
        except KeyError as exc:
            raise ValueError("Dataset CP sharding requires 'input_ids'") from exc
        if not platform.is_tensor(input_ids) or input_ids.ndim < 2:
            raise ValueError("Dataset CP 'input_ids' must be a batched tensor")

        sequence_length = int(input_ids.shape[-1])
        if sequence_length % self.cp_size != 0:
            raise ValueError(
                f"Dataset sequence length {sequence_length} must be divisible "
                f"by cp_size {self.cp_size}"
            )
        shard_length = sequence_length // self.cp_size
        shard_start = self.cp_rank * shard_length
        shard_stop = shard_start + shard_length
        shard_slice = slice(shard_start, shard_stop)

        sharded_batch = dict(working_batch)
        for field_name in sequence_fields:
            field_value = working_batch.get(field_name)
            if field_value is None or field_name in _SEQUENCE_LENGTH_FIELDS:
                continue
            if not platform.is_tensor(field_value) or field_value.ndim < 1:
                continue
            if int(field_value.shape[-1]) != sequence_length:
                raise ValueError(
                    f"Dataset CP field {field_name!r} must end with sequence "
                    f"length {sequence_length}, got {tuple(field_value.shape)}"
                )
            sharded_batch[field_name] = field_value[..., shard_slice]

        if "seq_lens" in working_batch and "seq_lens_padded" in working_batch:
            local_lengths, local_padded_lengths = self._shard_sequence_lengths(
                working_batch["seq_lens"],
                working_batch["seq_lens_padded"],
                shard_length,
            )
            sharded_batch["seq_lens"] = local_lengths
            sharded_batch["seq_lens_padded"] = local_padded_lengths
        return sharded_batch

    @staticmethod
    def _ensure_position_ids(
            batch: Mapping[str, Any],
            sequence_fields: Collection[str],
    ) -> dict[str, Any]:
        """Generate global position IDs before Dataset CP slicing when absent."""
        normalized_batch = dict(batch)
        if "position_ids" not in sequence_fields or "position_ids" in normalized_batch:
            return normalized_batch
        input_ids = normalized_batch.get("input_ids")
        if not platform.is_tensor(input_ids) or input_ids.ndim < 2:
            return normalized_batch
        batch_size = int(input_ids.shape[0])
        sequence_length = int(input_ids.shape[-1])
        device = getattr(input_ids, "device", None)
        position_range = platform.arange(
            sequence_length,
            dtype=input_ids.dtype,
            device=device,
        ).reshape(1, sequence_length)
        batch_offsets = platform.zeros(
            (batch_size, 1),
            dtype=input_ids.dtype,
            device=device,
        )
        normalized_batch["position_ids"] = batch_offsets + position_range
        return normalized_batch

    def _shard_sequence_lengths(
            self,
            sequence_lengths: Any,
            padded_sequence_lengths: Any,
            shard_length: int,
    ) -> tuple[Any, Any]:
        """Recompute packed sequence lengths in the local CP coordinate system."""
        batch_size = int(sequence_lengths.shape[0])
        shard_start = self.cp_rank * shard_length
        shard_stop = shard_start + shard_length
        local_lengths_by_sample = []
        local_padded_lengths_by_sample = []
        maximum_pack_count = 1

        for sample_index in range(batch_size):
            sample_lengths = sequence_lengths[sample_index].tolist()
            sample_padded_lengths = padded_sequence_lengths[sample_index].tolist()
            local_lengths = []
            local_padded_lengths = []
            offset = 0
            for raw_length, raw_padded_length in zip(
                    sample_lengths,
                    sample_padded_lengths,
            ):
                if raw_length == _SEQUENCE_LENGTH_SENTINEL:
                    break
                pack_start = offset
                pack_stop = offset + raw_padded_length
                offset = pack_stop
                intersection_start = max(pack_start, shard_start)
                intersection_stop = min(pack_stop, shard_stop)
                if intersection_start >= intersection_stop:
                    continue
                actual_stop = min(pack_start + raw_length, shard_stop)
                local_actual_length = max(actual_stop - intersection_start, 0)
                local_padded_length = intersection_stop - intersection_start
                local_lengths.append(local_actual_length)
                local_padded_lengths.append(local_padded_length)
            local_lengths_by_sample.append(local_lengths)
            local_padded_lengths_by_sample.append(local_padded_lengths)
            maximum_pack_count = max(maximum_pack_count, len(local_lengths))

        local_lengths = self._build_length_tensor(
            local_lengths_by_sample,
            sequence_lengths,
            maximum_pack_count,
        )
        local_padded_lengths = self._build_length_tensor(
            local_padded_lengths_by_sample,
            padded_sequence_lengths,
            maximum_pack_count,
        )
        return local_lengths, local_padded_lengths

    @staticmethod
    def _build_length_tensor(
            values_by_sample: list[list[int]],
            reference_tensor: Any,
            maximum_pack_count: int,
    ) -> Any:
        """Build one sentinel-padded packed-length tensor."""
        output_shape = (len(values_by_sample), maximum_pack_count)
        output_tensor = platform.zeros(
            output_shape,
            dtype=reference_tensor.dtype,
            device=getattr(reference_tensor, "device", None),
        )
        output_tensor[...] = _SEQUENCE_LENGTH_SENTINEL
        for sample_index, sample_values in enumerate(values_by_sample):
            for pack_index, pack_value in enumerate(sample_values):
                output_tensor[sample_index, pack_index] = pack_value
        return output_tensor


__all__ = ["ContextParallelBatchSharder"]
