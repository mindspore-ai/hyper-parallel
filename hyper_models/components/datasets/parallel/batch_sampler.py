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
"""Dataset-agnostic DP index slicing and resumable micro-batch sampling."""

from __future__ import annotations

import operator
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, Union, cast

from hyper_parallel.platform import get_platform

IndexMapping = Union[Mapping[int, int], Sequence[int]]
SamplerType = Literal["single", "cyclic"]

platform = get_platform()


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate an integer boundary shared by sampler options."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class _DatasetBatchSampler:
    """Share validation, DP slicing, epoch, and checkpoint state."""

    def __init__(
            self,
            *,
            total_samples: int,
            consumed_samples: int,
            micro_batch_size: int,
            dp_rank: int,
            dp_world_size: int,
            drop_last: bool,
            index_mapping: IndexMapping | None,
    ) -> None:
        """Validate and store DP slicing and checkpoint state."""
        _validate_positive_integer(micro_batch_size, "micro_batch_size")
        _validate_positive_integer(dp_world_size, "dp_world_size")
        if isinstance(total_samples, bool) or not isinstance(total_samples, int) or total_samples <= 0:
            raise ValueError("total_samples must be a positive integer")
        if isinstance(consumed_samples, bool) or not isinstance(consumed_samples, int):
            raise ValueError("consumed_samples must be a non-negative integer")
        self._validate_consumed_samples(consumed_samples, total_samples)
        if isinstance(dp_rank, bool) or not isinstance(dp_rank, int) or not 0 <= dp_rank < dp_world_size:
            raise ValueError(f"dp_rank must be in [0, {dp_world_size}), but got {dp_rank!r}")

        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.drop_last = drop_last
        self.index_mapping = index_mapping
        self.global_micro_batch_size = micro_batch_size * dp_world_size
        self.epoch = 0

    def _validate_consumed_samples(self, consumed_samples: int, total_samples: int) -> None:
        """Require sequential sampling to retain at least one unread sample."""
        if not 0 <= consumed_samples < total_samples:
            raise ValueError("consumed_samples must be in [0, total_samples)")

    def __iter__(self) -> Iterator[list[int]]:
        """Yield sequential rank-local indices for ``single`` mode."""
        while self.consumed_samples < self.total_samples:
            block_start = self.consumed_samples
            block_stop = min(block_start + self.global_micro_batch_size, self.total_samples)
            block_size = block_stop - block_start
            if self.drop_last and block_size < self.global_micro_batch_size:
                self.consumed_samples = self.total_samples
                break

            local_start = block_start + self.dp_rank * self.micro_batch_size
            local_stop = min(local_start + self.micro_batch_size, block_stop)
            self.consumed_samples = block_stop
            if local_start >= block_stop:
                continue

            local_indices = [
                self._resolve_index(index)
                for index in range(local_start, local_stop)
            ]
            yield local_indices

    def __len__(self) -> int:
        """Return the remaining number of micro-batches for this DP rank."""
        remaining_samples = self.total_samples - self.consumed_samples
        full_batches, partial_samples = divmod(remaining_samples, self.global_micro_batch_size)
        if self.drop_last or partial_samples <= self.dp_rank * self.micro_batch_size:
            return full_batches
        return full_batches + 1

    def _resolve_index(self, logical_index: int) -> int:
        """Apply an optional logical-to-physical sample index mapping."""
        if self.index_mapping is None:
            return logical_index
        try:
            resolved_index = self.index_mapping[logical_index]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"index_mapping does not define logical index {logical_index}") from exc
        if isinstance(resolved_index, bool):
            raise ValueError(f"index_mapping[{logical_index}] must be an integer")
        try:
            physical_index = operator.index(resolved_index)
        except TypeError as exc:
            raise ValueError(f"index_mapping[{logical_index}] must be an integer") from exc
        return physical_index

    def state_dict(self) -> dict[str, int]:
        """Return the global sample position needed to resume iteration."""
        sampler_state = {
            "consumed_samples": self.consumed_samples,
            "epoch": self.epoch,
        }
        return sampler_state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the global sample position from a checkpoint.

        Args:
            state_dict: Sampler state produced by :meth:`state_dict`.

        Raises:
            ValueError: If the checkpoint does not contain a valid sample position.
        """
        consumed_samples = state_dict.get("consumed_samples")
        if isinstance(consumed_samples, bool) or not isinstance(consumed_samples, int):
            raise ValueError("sampler state must contain integer 'consumed_samples'")
        self._validate_consumed_samples(consumed_samples, self.total_samples)
        epoch = state_dict.get("epoch", 0)
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler state 'epoch' must be a non-negative integer")
        self.consumed_samples = consumed_samples
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        """Start a new epoch while preserving progress in the restored epoch.

        Args:
            epoch: Zero-based training epoch.

        Raises:
            ValueError: If ``epoch`` is not a non-negative integer.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if epoch != self.epoch:
            self.consumed_samples = 0
            self.epoch = epoch


class _CyclicDatasetBatchSampler(_DatasetBatchSampler):
    """Yield a deterministic shuffled epoch for ``cyclic`` mode."""

    def __init__(self, *, data_sharding: bool, **sampler_options: Any) -> None:
        """Store the cyclic data-sharding policy."""
        super().__init__(**sampler_options)
        if not self.drop_last:
            raise ValueError("cyclic sampling requires drop_last=True")
        self.data_sharding = data_sharding

    def _validate_consumed_samples(self, consumed_samples: int, total_samples: int) -> None:
        """Allow the global consumed position to span multiple epochs."""
        del total_samples
        if consumed_samples < 0:
            raise ValueError("cyclic consumed_samples must be non-negative")

    def __len__(self) -> int:
        """Return the number of local micro-batches remaining in this epoch."""
        active_samples = self._active_samples()
        current_epoch_samples = self.consumed_samples % active_samples
        remaining_samples = active_samples - current_epoch_samples
        return remaining_samples // self.global_micro_batch_size

    def __iter__(self) -> Iterator[list[int]]:
        """Build the epoch permutation and DP slices."""
        active_samples = self._active_samples()
        self.epoch = self.consumed_samples // active_samples
        current_epoch_samples = self.consumed_samples % active_samples

        if current_epoch_samples % self.global_micro_batch_size != 0:
            raise ValueError("cyclic consumed_samples must align to a global micro-batch")

        if self.data_sharding:
            # Shuffle inside one contiguous per-rank bucket so every rank reads
            # only its own Dataset region.
            bucket_size = (self.total_samples // self.global_micro_batch_size) * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.dp_world_size
            bucket_start = self.dp_rank * bucket_size
            random_offsets = platform.random_permutation(bucket_size, self.epoch)
            rank_indices = [
                bucket_start + offset
                for offset in random_offsets[bucket_offset:]
            ]
        else:
            # Shuffle globally, skip restored
            # positions, then distribute indices to DP ranks by striding.
            full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
            shuffled_indices = platform.random_permutation(full_bucket_size, self.epoch)
            active_indices = shuffled_indices[current_epoch_samples:]
            rank_indices = active_indices[self.dp_rank::self.dp_world_size]

        local_batch = []
        for index in rank_indices:
            local_batch.append(index)
            if len(local_batch) == self.micro_batch_size:
                self.consumed_samples += self.global_micro_batch_size
                yield local_batch
                local_batch = []

    def set_epoch(self, epoch: int) -> None:
        """Move to an explicit cyclic epoch without losing in-epoch progress."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        active_samples = self._active_samples()
        current_epoch = self.consumed_samples // active_samples
        if epoch != current_epoch:
            self.consumed_samples = epoch * active_samples
        self.epoch = epoch

    def _active_samples(self) -> int:
        """Return the complete global micro-batch region reused every epoch."""
        active_samples = self.total_samples - self.total_samples % self.global_micro_batch_size
        if active_samples <= 0:
            raise ValueError("cyclic Dataset must contain one complete global micro-batch")
        return active_samples


def _resolve_index_mapping(
        index_mapping: IndexMapping | None,
        data_rearrange_map: IndexMapping | str | os.PathLike[str] | None,
) -> IndexMapping | None:
    """Resolve the optional rearrangement-map configuration."""
    if index_mapping is not None and data_rearrange_map is not None:
        raise ValueError("configure only one of index_mapping and data_rearrange_map")

    mapping_source = data_rearrange_map if data_rearrange_map is not None else index_mapping
    if isinstance(mapping_source, (str, os.PathLike)):
        mapping_path = os.fspath(mapping_source)
        loaded_mapping = platform.load_checkpoint(mapping_path, ckpt_format="torch")
        resolved_mapping = cast(IndexMapping, loaded_mapping)
        return resolved_mapping
    return mapping_source


def build_dataset_batch_sampler(
        *,
        total_samples: int,
        micro_batch_size: int,
        dp_rank: int,
        dp_world_size: int,
        consumed_samples: int = 0,
        drop_last: bool = True,
        index_mapping: IndexMapping | None = None,
        data_rearrange_map: IndexMapping | str | os.PathLike[str] | None = None,
        sampler_type: SamplerType = "single",
        data_sharding: bool = False,
) -> _DatasetBatchSampler:
    """Build a Dataset batch sampler.

    The sampler groups logical indices into global DP blocks, gives each DP rank
    one contiguous micro-batch, and checkpoints the next global sample position.
    It does not know how samples are stored or transformed.

    Supported scenarios:
        - ``single``: Sample sequential Dataset indices.
        - ``single`` with ``data_rearrange_map``: Resolve indices through an
          in-memory mapping or a mapping checkpoint path.
        - ``cyclic``: Use epoch-based ``randperm`` order, with
          ``data_sharding=True/False`` and resumable sampler state.
        - ``external``: Handled by the Trainer because an external dataloader
          does not need a Dataset batch sampler.

    Args:
        total_samples: Number of logical samples in the Dataset.
        micro_batch_size: Samples consumed by one DP rank per micro-step.
        dp_rank: Rank inside the data-parallel group.
        dp_world_size: Number of ranks in the data-parallel group.
        consumed_samples: Global logical sample position restored from a checkpoint.
        drop_last: Whether to omit an incomplete global DP block.
        index_mapping: Optional logical-to-physical sample index mapping.
        data_rearrange_map: Mapping object or checkpoint path.
        sampler_type: ``single`` for sequential sampling or ``cyclic`` for a
            deterministic shuffled epoch.
        data_sharding: Whether cyclic sampling shuffles an independent
            contiguous bucket on each DP rank.

    Returns:
        A resumable iterable of rank-local index lists.
    """
    resolved_mapping = _resolve_index_mapping(index_mapping, data_rearrange_map)
    sampler_options = {
        "total_samples": total_samples,
        "consumed_samples": consumed_samples,
        "micro_batch_size": micro_batch_size,
        "dp_rank": dp_rank,
        "dp_world_size": dp_world_size,
        "drop_last": drop_last,
        "index_mapping": resolved_mapping,
    }

    if sampler_type == "single":
        # Scenario 1 — single without a mapping: sequential Dataset indices.
        # Scenario 2 — single + data_rearrange_map: the same logical sequence
        # is resolved through an in-memory mapping or mapping checkpoint.
        batch_sampler = _DatasetBatchSampler(**sampler_options)
        return batch_sampler
    if sampler_type == "cyclic":
        # Scenario 3 — cyclic: use epoch-based randperm order,
        # support data_sharding=True/False, and checkpoint consumed_samples.
        if resolved_mapping is not None:
            raise ValueError("cyclic sampling does not support a rearrangement map")
        batch_sampler = _CyclicDatasetBatchSampler(
            data_sharding=data_sharding,
            **sampler_options,
        )
        return batch_sampler

    raise ValueError("sampler_type must be one of: single, cyclic")


__all__ = ["build_dataset_batch_sampler"]
