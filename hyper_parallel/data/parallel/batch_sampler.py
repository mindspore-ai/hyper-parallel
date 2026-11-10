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

import copy
import operator
import os
import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, Union, cast

import torch

from hyper_parallel.data.dataset_logging import get_dataset_logger

IndexMapping = Union[Mapping[int, int], Sequence[int]]
SamplerType = Literal["single", "cyclic"]

logger = get_dataset_logger(__name__)


def sampler_state_for_load(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Build a loader-owned checkpoint template without inventing missing progress.

    Args:
        state_dict: Fresh fixed or balanced loader state.

    Returns:
        Loading-only placeholders for the shared cursor and legacy Reader paths.
    """
    state = copy.deepcopy(dict(state_dict))
    balanced = "_num_yielded" not in state and "_snapshot" not in state
    if not balanced and get_sampler_checkpoint_state(state) is None:
        return state
    cursor = dict.fromkeys(("consumed_samples", "epoch", "global_batch_size", "dp_world_size", "optimizer_step_offset"))
    state["_index_sampler_state"] = dict(cursor)
    state["_iterator_finished"] = None
    # Missing torchdata fields must not make a cursor-only checkpoint look like
    # the target's freshly initialized worker snapshot.
    state["_num_yielded"] = None
    snapshot = state.setdefault("_snapshot", {}).setdefault("_main_snapshot", {})
    snapshot["_num_workers"] = None
    if isinstance(snapshot.get("_index_sampler_state"), dict):
        snapshot["_index_sampler_state"]["optimizer_step_offset"] = None
    state["global_rank"] = None
    payload = dict.fromkeys(("dataset_size", "seed"))
    for key in ("dataset_reader", "metadata_reader", "sidecar_reader"):
        state[key] = {"sampler": dict(cursor), "sample_loader": dict(payload)}
    if balanced:
        state.update(epoch=0, step=0, stopped=False, direct_sample_loader=dict(payload))
        state["dataset_state"] = dict(payload)
    return state


def has_torchdata_state(state_dict: Mapping[str, Any]) -> bool:
    """Return whether a loaded checkpoint contains real torchdata iterator state."""
    snapshot = state_dict.get("_snapshot")
    return state_dict.get("_num_yielded") is not None or (
        isinstance(snapshot, Mapping)
        and snapshot.get("_main_snapshot", {}).get("_num_workers") is not None
    )


def get_sampler_checkpoint_state(state_dict: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Read the shared native cursor, accepting legacy Reader-only checkpoints.

    Args:
        state_dict: Saved loader state or a template filled by checkpoint loading.

    Returns:
        Saved sampler state, or None when no cursor was loaded.
    """
    candidates = [state_dict.get("_index_sampler_state")]
    for key in ("dataset_reader", "metadata_reader", "sidecar_reader"):
        reader = state_dict.get(key)
        if isinstance(reader, Mapping):
            candidates.append(reader.get("sampler"))
    return next((state for state in candidates
                 if isinstance(state, Mapping) and state.get("consumed_samples") is not None), None)


def sampler_optimizer_step_progress(state: Mapping[str, Any]) -> int:
    """Return samples accumulated since the saved optimizer-step boundary.

    Args:
        state: Native sampler cursor; old checkpoints use an offset of zero.

    Returns:
        Zero at a complete optimizer step, otherwise the partial sample count.
    """
    consumed, batch_size = state["consumed_samples"], state["global_batch_size"]
    offset = state.get("optimizer_step_offset")
    offset = 0 if offset is None else offset
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (consumed, batch_size, offset)):
        raise ValueError("Sampler cursor, global_batch_size and optimizer_step_offset must be integers")
    if not 0 <= offset <= consumed or offset >= batch_size:
        raise ValueError("Sampler optimizer_step_offset must be within the cursor and global_batch_size")
    return (consumed - offset) % batch_size


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate an integer boundary shared by sampler options."""
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class _DatasetBatchSampler:
    """Share validation, DP slicing, epoch, and checkpoint state.

    BatchSampler splits one global batch across DP ranks
    → each rank receives its logical Dataset index
    → Dataset.__getitem__(index)
    → shuffle_index maps the logical index to a random packed sample
    """

    def __init__(
            self,
            *,
            total_samples: int,
            consumed_samples: int,
            micro_batch_size: int,
            global_batch_size: int,
            dp_rank: int,
            dp_world_size: int,
            drop_last: bool,
            index_mapping: IndexMapping | None,
    ) -> None:
        """Validate and store DP slicing and checkpoint state."""
        _validate_positive_integer(total_samples, "total_samples")
        _validate_positive_integer(micro_batch_size, "micro_batch_size")
        _validate_positive_integer(dp_world_size, "dp_world_size")
        self._validate_consumed_samples(consumed_samples, total_samples)
        if not 0 <= dp_rank < dp_world_size:
            raise ValueError(f"dp_rank must be in [0, {dp_world_size}), but got {dp_rank!r}")

        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.drop_last = drop_last
        self.index_mapping = index_mapping
        self.global_micro_batch_size = micro_batch_size * dp_world_size
        _validate_positive_integer(global_batch_size, "global_batch_size")
        if global_batch_size % self.global_micro_batch_size != 0:
            raise ValueError("global_batch_size must be divisible by micro_batch_size * dp_world_size")

        self.global_batch_size = global_batch_size
        self._optimizer_step_offset = 0
        self.epoch = 0
        self._resume_at_source_batch = False

    @staticmethod
    def _validate_consumed_samples(consumed_samples: int, total_samples: int) -> None:
        """Allow sequential sampling to stop exactly at the epoch boundary."""
        if not 0 <= consumed_samples <= total_samples:
            raise ValueError("consumed_samples must be in [0, total_samples]")

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

        resolved_index = self.index_mapping[logical_index]
        if isinstance(resolved_index, bool):
            raise TypeError(f"index_mapping[{logical_index}] must be an integer")

        physical_index = operator.index(resolved_index)
        return physical_index

    def state_dict(self) -> dict[str, int]:
        """Return the Dataset epoch and position needed to resume iteration."""
        sampler_state = {
            "consumed_samples": self.consumed_samples,
            "epoch": self.epoch,
            "global_batch_size": self.global_batch_size,
            "dp_world_size": self.dp_world_size,
        }
        if self._optimizer_step_offset:
            sampler_state["optimizer_step_offset"] = self._optimizer_step_offset
        return sampler_state

    def enable_source_batch_resume(self) -> None:
        """Restore this sampler at a source-batch boundary for Online batching."""
        self._resume_at_source_batch = True

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the global sample position from a checkpoint.

        Args:
            state_dict: Sampler state produced by :meth:`state_dict`.

        Raises:
            ValueError: If the checkpoint does not contain a valid sample position.
        """
        consumed_samples = state_dict["consumed_samples"]
        self._validate_consumed_samples(consumed_samples, self.total_samples)
        progress = sampler_optimizer_step_progress(state_dict)
        repartition = (state_dict["global_batch_size"] != self.global_batch_size
                       or state_dict["dp_world_size"] != self.dp_world_size)
        if progress and (repartition or not self._resume_at_source_batch):
            raise ValueError("Sampler restoration requires a complete global optimizer batch")
        offset = consumed_samples % self.global_batch_size if repartition else (
            state_dict.get("optimizer_step_offset") or 0
        )
        if (self._resume_at_source_batch and consumed_samples != self.total_samples
                and (consumed_samples - offset) % self.global_micro_batch_size):
            raise ValueError("Online source sampler state must align to a distributed micro-batch")

        self._optimizer_step_offset = offset
        self.consumed_samples = consumed_samples
        self.epoch = state_dict["epoch"]

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
            self._optimizer_step_offset = 0
            self.epoch = epoch


class _CyclicDatasetBatchSampler(_DatasetBatchSampler):
    """Yield one deterministic shuffled epoch for ``cyclic`` mode."""

    def __init__(self, *, data_sharding: bool, seed: int, **sampler_options: Any) -> None:
        """Store the cyclic data-sharding policy."""
        super().__init__(**sampler_options)
        if not self.drop_last:
            raise ValueError("cyclic sampling requires drop_last=True")

        self.data_sharding = data_sharding
        self.seed = seed

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a global cursor, repartitioning only global shuffle order."""
        if self.data_sharding and state_dict["dp_world_size"] != self.dp_world_size:
            raise ValueError("elastic DP resume requires data_sharding=False")

        super().load_state_dict(state_dict)

    def __len__(self) -> int:
        """Return the number of local micro-batches remaining in this epoch."""
        active_samples = self._active_samples()
        remaining_samples = active_samples - self.consumed_samples
        return remaining_samples // self.global_micro_batch_size

    def __iter__(self) -> Iterator[list[int]]:
        """Build the current epoch permutation and DP slices."""
        active_samples = self._active_samples()
        if (self.consumed_samples - self._optimizer_step_offset) % self.global_micro_batch_size != 0:
            raise ValueError("cyclic consumed_samples must align to a distributed micro-batch")

        logger.debug(
            "Starting cyclic Dataset epoch=%d at consumed_samples=%d",
            self.epoch,
            self.consumed_samples,
        )

        if self.data_sharding:
            # Shuffle one contiguous bucket per DP rank.
            bucket_size = active_samples // self.dp_world_size
            bucket_offset = self.consumed_samples // self.dp_world_size
            bucket_start = self.dp_rank * bucket_size
            random_offsets = list(range(bucket_size))
            random.Random(self.seed + self.epoch).shuffle(random_offsets)
            rank_indices = [bucket_start + offset for offset in random_offsets[bucket_offset:]]
        else:
            # Shuffle before dropping the tail so source-file tail samples remain eligible.
            # Legacy checkpoints retain their cursor but not their old truncated permutation.
            shuffled_indices = list(range(self.total_samples))
            random.Random(self.seed + self.epoch).shuffle(shuffled_indices)
            active_indices = shuffled_indices[:active_samples]
            active_indices = active_indices[self.consumed_samples:]
            rank_indices = active_indices[self.dp_rank::self.dp_world_size]

        local_batch = []
        for index in rank_indices:
            local_batch.append(index)
            if len(local_batch) == self.micro_batch_size:
                self.consumed_samples += self.global_micro_batch_size
                yield local_batch
                local_batch = []

    def _active_samples(self) -> int:
        """Return the number of samples fitting complete global micro-batches."""
        remaining = self.total_samples - self.consumed_samples
        active_samples = self.total_samples - remaining % self.global_micro_batch_size
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
        loaded_mapping = torch.load(f=mapping_path)
        resolved_mapping = cast(IndexMapping, loaded_mapping)
        return resolved_mapping
    return mapping_source


def build_dataset_batch_sampler(
        *,
        total_samples: int,
        micro_batch_size: int,
        global_batch_size: int,
        dp_rank: int,
        dp_world_size: int,
        consumed_samples: int = 0,
        drop_last: bool = True,
        index_mapping: IndexMapping | None = None,
        data_rearrange_map: IndexMapping | str | os.PathLike[str] | None = None,
        sampler_type: SamplerType = "single",
        data_sharding: bool = False,
        seed: int = 0,
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

    Args:
        total_samples: Number of logical samples in the Dataset.
        micro_batch_size: Samples consumed by one DP rank per micro-step.
        global_batch_size: Fixed samples consumed globally by one optimizer step.
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
        seed: Base seed used by cyclic epoch shuffling.

    Returns:
        A resumable iterable of rank-local index lists.
    """
    resolved_mapping = _resolve_index_mapping(index_mapping, data_rearrange_map)
    logger.debug(
        "Building Dataset sampler: type=%s, total_samples=%d, consumed_samples=%d, micro_batch_size=%d, "
        "dp_rank=%d, dp_world_size=%d, drop_last=%s, data_sharding=%s, mapped=%s",
        sampler_type, total_samples, consumed_samples, micro_batch_size, dp_rank, dp_world_size, drop_last,
        data_sharding, resolved_mapping is not None,
    )
    sampler_options = {
        "total_samples": total_samples,
        "consumed_samples": consumed_samples,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
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
            seed=seed,
            **sampler_options,
        )
        return batch_sampler

    raise ValueError("sampler_type must be one of: single, cyclic")
