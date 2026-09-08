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
"""Build DataLoaders and calculate distributed micro-batch sizing."""

from __future__ import annotations

import copy
import operator
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.parallel import build_dataset_batch_sampler

logger = get_dataset_logger(__name__)


def calculate_num_micro_batches(
        global_batch_size: int,
        micro_batch_size: int,
        dp_world_size: int,
) -> int:
    """Calculate the number of micro-batches in one optimizer step.

    Args:
        global_batch_size: Number of samples processed by one optimizer step.
        micro_batch_size: Number of samples processed by each DP rank and forward pass.
        dp_world_size: Number of data-parallel ranks.

    Returns:
        Number of forward/backward micro-batches executed by each DP rank.

    Raises:
        ValueError: If the global batch cannot be divided exactly.
    """
    distributed_micro_batch_size = micro_batch_size * dp_world_size
    if global_batch_size % distributed_micro_batch_size != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"micro_batch_size * dp_world_size ({distributed_micro_batch_size})"
        )

    num_micro_batches = global_batch_size // distributed_micro_batch_size
    logger.debug(
        "Resolved micro-batches=%d from global_batch_size=%d, micro_batch_size=%d, dp_world_size=%d",
        num_micro_batches,
        global_batch_size,
        micro_batch_size,
        dp_world_size,
    )
    return num_micro_batches


def _is_iterable_dataset(dataset: Any) -> bool:
    """Return whether a Dataset streams samples without mapping-style access."""
    if isinstance(dataset, IterableDataset):
        is_iterable = True
    else:
        has_iterator = callable(getattr(dataset, "__iter__", None))
        has_index_access = callable(getattr(dataset, "__getitem__", None))
        is_iterable = has_iterator and not has_index_access
    return is_iterable


def _supports_output_index_for_resume(dataset: Any) -> bool:
    """Return whether a Dataset can emit and rebuild stable output indices."""
    get_item = getattr(dataset, "get_item", None)
    supports_output_index = callable(get_item) and hasattr(dataset, "output_index_for_resume")
    return supports_output_index


def _normalize_source_samples(source_item: Any) -> list[Mapping[str, Any]]:
    """Normalize one source output into its ordered ModelSamples."""
    if isinstance(source_item, Mapping):
        model_samples = [source_item]
        return model_samples

    if isinstance(source_item, Sequence) and not isinstance(source_item, (str, bytes)):
        model_samples = []
        for model_sample in source_item:
            if not isinstance(model_sample, Mapping):
                raise ValueError("Every dynamic source sample must be a mapping")

            model_samples.append(model_sample)
        return model_samples

    raise ValueError("A dynamic source item must be a mapping or a sequence of mappings")


def _restore_index_buffer(
        source_dataset: Any,
        saved_buffer: Sequence[Any],
) -> list[tuple[Mapping[str, Any], int]]:
    """Rebuild buffered ModelSamples from output index and sample index."""
    restored_buffer = []
    cached_output_index: Any = object()
    cached_model_samples: list[Mapping[str, Any]] = []
    for output_index_entry in saved_buffer:
        output_index, sample_idx = output_index_entry
        resolved_sample_idx = operator.index(sample_idx)
        if resolved_sample_idx < 0:
            raise ValueError("Buffered sample_idx must be a non-negative integer")

        if output_index != cached_output_index:
            restored_item = source_dataset.get_item(output_index)
            cached_model_samples = _normalize_source_samples(restored_item)
            cached_output_index = output_index
        try:
            model_sample = cached_model_samples[resolved_sample_idx]
        except IndexError as exc:
            raise ValueError(
                f"Buffered sample_idx={resolved_sample_idx} is out of range for output_index={output_index!r}"
            ) from exc

        sample_length = int(model_sample["input_ids"].shape[-1])
        if sample_length <= 0:
            raise ValueError("Dynamic batching samples must contain at least one token")

        restored_buffer.append((model_sample, sample_length))

    return restored_buffer


class _OutputIndexDataset:
    """Attach a stable mapping-Dataset output index to each source item."""

    def __init__(self, dataset: Any) -> None:
        """Store the mapping Dataset used to reconstruct buffered samples."""
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the wrapped Dataset length."""
        dataset_length = len(self.dataset)
        return dataset_length

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Return one source item together with the index that produced it."""
        source_item = self.get_item(index)
        indexed_source_item = (source_item, index)
        return indexed_source_item

    def get_item(self, index: Any) -> Any:
        """Rebuild one source item without advancing the source iterator."""
        resolved_index = operator.index(index)
        get_item = getattr(self.dataset, "get_item", None)
        if callable(get_item):
            source_item = get_item(resolved_index)
        else:
            source_item = self.dataset[resolved_index]

        return source_item


def build_dataloader(
        dataloader_target: Any,
        *,
        datasets: Sequence[Any | None],
        collate_fn: Any,
        mesh_context: Any,
        training_config: Any,
        max_seq_len: int | None = None,
        default_seed: int = 1234,
) -> tuple[tuple[Any | None, ...], tuple[Any | None, ...]]:
    """Build train, validation, and test DataLoaders.

    Mapping Datasets receive a distributed batch sampler. Iterable Datasets
    control their own order. The configured target decides whether selected
    samples use fixed collation or dynamic token batching.

    Args:
        dataloader_target: DataLoader build target.
        datasets: Train, validation, and test datasets.
        collate_fn: Collator applied after fixed or dynamic sample selection.
        mesh_context: Data-parallel mesh context.
        training_config: Batch size and random seed configuration.
        max_seq_len: Maximum sample length used to derive dynamic token budget.
        default_seed: Seed used when no training seed is configured.

    Returns:
        DataLoader and batch-sampler tuples for the three Dataset splits.
    """
    if len(datasets) != 3:
        raise ValueError("datasets must contain train, validation, and test entries")

    if all(dataset is None for dataset in datasets):
        dataloader_splits = (None, None, None)
        batch_sampler_splits = (None, None, None)
        return dataloader_splits, batch_sampler_splits

    if dataloader_target is None:
        raise ValueError("dataloader_target must define a build target")

    micro_batch_size = training_config.micro_batch_size
    global_batch_size = training_config.global_batch_size
    seed = training_config.seed if training_config.seed is not None else default_seed

    dp_rank = mesh_context.dp_rank
    dp_world_size = mesh_context.dp_size

    sampler_type = getattr(dataloader_target, "dataloader_type", "single")
    drop_last = getattr(dataloader_target, "drop_last", True)
    rearrangement_map = getattr(dataloader_target, "data_rearrange_map", None)
    data_sharding = getattr(dataloader_target, "data_sharding", False)

    dataloaders: list[Any | None] = [None] * len(datasets)
    batch_samplers: list[Any | None] = [None] * len(datasets)

    for split_index, (split_name, dataset) in enumerate(zip(("train", "valid", "test"), datasets)):
        if dataset is None:
            logger.debug("Skipping empty Dataset split=%s", split_name)
            continue

        batch_sampler = None
        if not _is_iterable_dataset(dataset):
            total_samples = len(dataset)
            batch_sampler = build_dataset_batch_sampler(
                total_samples=total_samples,
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                drop_last=drop_last,
                data_rearrange_map=rearrangement_map,
                sampler_type=sampler_type,
                data_sharding=data_sharding,
                seed=seed,
            )

        dataloader = dataloader_target.build(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            batch_size=micro_batch_size,
            dp_world_size=dp_world_size,
            max_seq_len=max_seq_len,
            seed=seed,
        )
        logger.debug(
            "Built DataLoader split=%s, dataset=%s, batch_sampler=%s",
            split_name,
            type(dataset).__name__,
            type(batch_sampler).__name__ if batch_sampler is not None else None,
        )
        dataloaders[split_index] = dataloader
        batch_samplers[split_index] = batch_sampler

    dataloader_splits = tuple(dataloaders)
    batch_sampler_splits = tuple(batch_samplers)
    logger.debug("Finished building train/valid/test DataLoaders")
    return dataloader_splits, batch_sampler_splits


class FixedBatchDataLoader(StatefulDataLoader):
    """Build fixed-sample batches while retaining Trainer iterator policy."""

    def __init__(
            self,
            dataset: Any,
            batch_sampler: Any = None,
            collate_fn: Callable[[list[Any]], Any] | None = None,
            *,
            batch_size: int | None = None,
            drop_last: bool = True,
            use_background_prefetcher: bool = False,
            num_workers: int = 0,
            seed: int = 1234,
            pin_memory: bool = False,
            prefetch_factor: int | None = None,
    ) -> None:
        """Initialize the stateful DataLoader."""
        self.drop_last = drop_last
        self.use_background_prefetcher = use_background_prefetcher
        generator = torch.Generator().manual_seed(seed)
        worker_options = {
            "num_workers": num_workers,
            "generator": generator,
            "pin_memory": pin_memory,
        }
        if num_workers > 0 and prefetch_factor is not None:
            worker_options["prefetch_factor"] = prefetch_factor
        if batch_sampler is None:
            resolved_batch_size = 1 if batch_size is None else batch_size
            super().__init__(
                dataset=dataset,
                batch_size=resolved_batch_size,
                collate_fn=collate_fn,
                drop_last=drop_last,
                **worker_options,
            )
        else:
            super().__init__(
                dataset=dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                **worker_options,
            )

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch state when the configured batch sampler supports it."""
        if self.batch_sampler is not None and hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)


@dataclass
class TextTokenBatcher:
    """Buffer Online samples and select K samples within a token budget.

    Runtime batching retains full ModelSamples. ``buffer_output_indices`` stays
    aligned with that buffer and holds replay keys when index checkpoints are enabled.

    Args:
        token_budget: Target packed-token limit for one forward-backward batch.
            A single sample may exceed this limit and forms a batch by itself.
        min_buffered_samples: Minimum candidate samples buffered before batching.
    """

    token_budget: int
    min_buffered_samples: int
    buffer: list[tuple[Mapping[str, Any], int]] = field(default_factory=list, init=False)
    buffer_output_indices: list[Any | None] = field(default_factory=list, init=False)
    buffer_token_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate dynamic batching boundaries."""
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")

        if self.min_buffered_samples <= 0:
            raise ValueError("min_buffered_samples must be positive")

    def put_item(self, model_sample: Mapping[str, Any], output_index_entry: Any | None = None) -> None:
        """Append one ModelSample and its token length."""
        sample_length = int(model_sample["input_ids"].shape[-1])
        if sample_length <= 0:
            raise ValueError("Dynamic batching samples must contain at least one token")

        buffered_sample = (model_sample, sample_length)
        self.buffer.append(buffered_sample)
        self.buffer_output_indices.append(output_index_entry)
        self.buffer_token_count += sample_length

    def is_ready_for_micro_batch(self) -> bool:
        """Report whether both buffer thresholds are satisfied."""
        enough_samples = len(self.buffer) >= self.min_buffered_samples
        enough_tokens = self.buffer_token_count >= self.token_budget
        is_ready = enough_samples and enough_tokens

        return is_ready

    def get_micro_batch(self) -> list[Mapping[str, Any]]:
        """Select fitting samples and retain the remainder."""
        if not self.buffer:
            raise ValueError("Dynamic token buffer is empty")

        selected_samples = []
        remaining_buffer = []
        remaining_buffer_output_indices = []
        selected_token_count = 0
        for (model_sample, sample_length), output_index_entry in zip(self.buffer, self.buffer_output_indices):
            sample_fits = selected_token_count == 0 or selected_token_count + sample_length <= self.token_budget
            if sample_fits:
                selected_samples.append(model_sample)
                selected_token_count += sample_length
            else:
                remaining_buffer.append((model_sample, sample_length))
                remaining_buffer_output_indices.append(output_index_entry)

        self.buffer = remaining_buffer
        self.buffer_output_indices = remaining_buffer_output_indices
        self.buffer_token_count -= selected_token_count

        return selected_samples

    def empty(self) -> bool:
        """Report whether no unconsumed samples remain in the buffer."""
        is_empty = not self.buffer

        return is_empty


class _IndexBufferDynamicBatchRuntime:
    """Checkpoint reconstructable source buffers with output-index entries."""

    save_by_idx = True

    def __init__(self, source_dataset: Any) -> None:
        """Store a Dataset that emits source items with stable output indices."""
        self.source_dataset = source_dataset

    def put_source_item(self, source_item: Any, batcher: TextTokenBatcher) -> None:
        """Flatten one indexed source item and retain each sample index."""
        model_samples_item, output_index = source_item
        model_samples = _normalize_source_samples(model_samples_item)
        for sample_idx, model_sample in enumerate(model_samples):
            batcher.put_item(model_sample, (output_index, sample_idx))

    def get_buffer_state(self, batcher: TextTokenBatcher) -> list[Any]:
        """Return compact output-index entries for buffered ModelSamples."""
        if len(batcher.buffer) != len(batcher.buffer_output_indices):
            raise RuntimeError("Dynamic sample and output-index buffers are inconsistent")

        if any(index is None for index in batcher.buffer_output_indices):
            raise RuntimeError("Index-buffer checkpoint requires a replay key for every buffered sample")

        return batcher.buffer_output_indices

    def restore_buffer(
            self,
            saved_buffer: Sequence[Any],
            saved_by_idx: bool,
    ) -> tuple[list[tuple[Mapping[str, Any], int]], list[Any]]:
        """Rebuild buffered ModelSamples from output index and sample index."""
        if not saved_by_idx:
            if saved_buffer:
                raise ValueError("save_by_idx=True cannot restore a checkpoint containing full samples")

            return [], []

        restored_buffer = _restore_index_buffer(self.source_dataset, saved_buffer)
        restored_indices = list(saved_buffer)
        return restored_buffer, restored_indices


class _FullBufferDynamicBatchRuntime:
    """Checkpoint full buffered samples alongside the upstream source state."""

    save_by_idx = False

    def __init__(self, dataset: Any, replay_dataset: Any | None = None) -> None:
        """Retain the source Dataset and an optional index replay adapter."""
        self.source_dataset = dataset
        self.replay_dataset = replay_dataset

    def put_source_item(self, source_item: Any, batcher: TextTokenBatcher) -> None:
        """Flatten one streaming source item into the runtime buffer."""
        model_samples = _normalize_source_samples(source_item)
        for model_sample in model_samples:
            batcher.put_item(model_sample)

    def get_buffer_state(self, batcher: TextTokenBatcher) -> list[Any]:
        """Return full samples already pulled beyond the streaming cursor."""
        if len(batcher.buffer) != len(batcher.buffer_output_indices):
            raise RuntimeError("Dynamic sample and output-index buffers are inconsistent")

        return batcher.buffer

    def restore_buffer(
            self,
            saved_buffer: Sequence[Any],
            saved_by_idx: bool,
    ) -> tuple[list[tuple[Mapping[str, Any], int]], list[Any]]:
        """Restore full samples that cannot be fetched behind the cursor."""
        if saved_by_idx:
            if saved_buffer and self.replay_dataset is None:
                raise ValueError("An index-buffer checkpoint requires a replayable Dataset")

            restored_buffer = _restore_index_buffer(self.replay_dataset, saved_buffer)
        else:
            restored_buffer = list(saved_buffer)
        restored_indices = [None] * len(restored_buffer)
        return restored_buffer, restored_indices


class DynamicBatchDataLoader:
    """Select K Online samples by token budget, then collate the batch.

    The internal DataLoader preserves source samples. ``TextTokenBatcher``
    selects K samples before the configured ``collate_fn`` builds the batch.

    Args:
        dataset: Online mapping or iterable Dataset producing ModelSamples.
        batch_sampler: Optional mapping-style batch sampler.
        collate_fn: Final batch collator receiving the selected ModelSamples.
        batch_size: Configured micro batch size used to derive the token budget.
        dp_world_size: Number of data-parallel source lanes.
        save_by_idx: Whether to checkpoint buffer entries as output indices.
            Mapping Datasets default to true and iterable Datasets default to false.
        max_seq_len: Maximum sample length used to derive the token budget.
        min_buffered_samples: Minimum candidate samples buffered before batching.
        drop_last: Compatibility option retained from fixed DataLoader configuration.
            Dynamic buffers are always drained when the source is exhausted.
        use_background_prefetcher: Trainer-side prefetch policy.
        num_workers: Number of source DataLoader workers.
        seed: Source DataLoader random seed.
        pin_memory: Whether source samples use pinned host memory.
        prefetch_factor: Number of source batches prefetched by each worker.
    """

    def __init__(
            self,
            dataset: Any,
            batch_sampler: Any = None,
            collate_fn: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]] | None = None,
            *,
            batch_size: int,
            dp_world_size: int | None = None,
            save_by_idx: bool | None = None,
            max_seq_len: int | None,
            min_buffered_samples: int = 200,
            drop_last: bool = True,
            use_background_prefetcher: bool = False,
            num_workers: int = 0,
            seed: int = 1234,
            pin_memory: bool = False,
            prefetch_factor: int | None = None,
    ) -> None:
        """Initialize source reading, token selection, and Online packing."""
        if collate_fn is None:
            raise ValueError("DynamicBatchDataLoader requires a collate_fn")

        if max_seq_len is None:
            raise ValueError("DynamicBatchDataLoader requires data_transform.max_seq_len")

        resolved_dp_world_size = dp_world_size
        if resolved_dp_world_size <= 0:
            raise ValueError("dp_world_size must be positive")

        sampler_dp_world_size = getattr(batch_sampler, "dp_world_size", resolved_dp_world_size)
        if sampler_dp_world_size != resolved_dp_world_size:
            raise ValueError("batch_sampler.dp_world_size must match dp_world_size")

        self.drop_last = drop_last
        self.use_background_prefetcher = use_background_prefetcher
        self.batch_collate_fn = collate_fn
        self.dp_world_size = resolved_dp_world_size
        token_budget = batch_size * max_seq_len
        self.batcher = TextTokenBatcher(
            token_budget=token_budget,
            min_buffered_samples=min_buffered_samples,
        )
        is_iterable = _is_iterable_dataset(dataset)
        if batch_sampler is not None:
            enable_source_resume = getattr(batch_sampler, "enable_source_batch_resume", None)
            if callable(enable_source_resume):
                enable_source_resume()

        resolved_save_by_idx = not is_iterable if save_by_idx is None else save_by_idx
        # Mapping Datasets are replayable through an adapter; iterable sources must opt in.
        replay_dataset = dataset if is_iterable else _OutputIndexDataset(dataset)
        supports_replay = not is_iterable or _supports_output_index_for_resume(dataset)
        if resolved_save_by_idx and not supports_replay:
            raise ValueError("save_by_idx=True requires get_item() and output_index_for_resume")

        if is_iterable and hasattr(dataset, "output_index_for_resume"):
            dataset.output_index_for_resume = resolved_save_by_idx

        if resolved_save_by_idx:
            self._runtime = _IndexBufferDynamicBatchRuntime(replay_dataset)
        else:
            # Retain replay access only to load an earlier index-buffer checkpoint.
            checkpoint_replay_dataset = replay_dataset if supports_replay else None
            self._runtime = _FullBufferDynamicBatchRuntime(dataset, checkpoint_replay_dataset)

        self.source_dataloader = FixedBatchDataLoader(
            dataset=self._runtime.source_dataset,
            batch_sampler=batch_sampler,
            # Keep source samples separate until TextTokenBatcher selects K.
            collate_fn=list,
            batch_size=1 if batch_sampler is None else batch_size,
            drop_last=False,
            num_workers=num_workers,
            seed=seed,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
        self.resume_pending = False

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Start source iteration while retaining restored buffer state."""
        if not self.resume_pending:
            self.batcher.buffer.clear()
            self.batcher.buffer_output_indices.clear()
            self.batcher.buffer_token_count = 0

        source_iterator = iter(self.source_dataloader)
        batch_iterator = self._batch_data_generator(source_iterator)
        self.resume_pending = False

        return batch_iterator

    def _batch_data_generator(
            self,
            source_iterator: Iterator[list[Any]],
    ) -> Iterator[Mapping[str, Any]]:
        """Yield packed batches while continuously filling the token buffer."""
        # Drain batches already available in a restored buffer.
        while self.batcher.is_ready_for_micro_batch():
            model_samples = self.batcher.get_micro_batch()
            batch = self.batch_collate_fn(model_samples)
            yield batch

        # Keep filling the token buffer from the source iterator.
        for source_samples in source_iterator:
            for source_sample in source_samples:
                self._runtime.put_source_item(source_sample, self.batcher)

            while self.batcher.is_ready_for_micro_batch():
                model_samples = self.batcher.get_micro_batch()
                batch = self.batch_collate_fn(model_samples)
                yield batch

        # Preserve every pulled sample by draining below-threshold tail batches.
        while not self.batcher.empty():
            model_samples = self.batcher.get_micro_batch()
            batch = self.batch_collate_fn(model_samples)
            yield batch

    def state_dict(self) -> dict[str, Any]:
        """Capture the future source cursor and unconsumed dynamic buffer."""
        state = {
            "dp_world_size": self.dp_world_size,
            "source_dataloader": self.source_dataloader.state_dict(),
            "save_by_idx": self._runtime.save_by_idx,
            "buffer": self._runtime.get_buffer_state(self.batcher),
            "buffer_token_count": self.batcher.buffer_token_count,
        }
        checkpoint_state = copy.deepcopy(state)

        return checkpoint_state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the future source cursor and unconsumed dynamic buffer."""
        checkpoint_state = copy.deepcopy(dict(state_dict))
        saved_dp_world_size = checkpoint_state.get("dp_world_size", self.dp_world_size)
        if saved_dp_world_size != self.dp_world_size:
            raise ValueError(
                "Online dataloader resume does not support DP world-size changes: "
                f"saved_dp_world_size={saved_dp_world_size}, current_dp_world_size={self.dp_world_size}"
            )

        previous_save_by_idx = bool(checkpoint_state.get("save_by_idx", False))
        saved_buffer = checkpoint_state["buffer"]
        restored_buffer, restored_indices = self._runtime.restore_buffer(
            saved_buffer,
            previous_save_by_idx,
        )

        restored_token_count = sum(sample_length for _, sample_length in restored_buffer)
        if restored_token_count != checkpoint_state["buffer_token_count"]:
            raise ValueError("buffer_token_count does not match the restored dynamic buffer")

        self.source_dataloader.load_state_dict(checkpoint_state["source_dataloader"])
        self.batcher.buffer = restored_buffer
        self.batcher.buffer_output_indices = restored_indices
        self.batcher.buffer_token_count = restored_token_count
        self.resume_pending = True

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch state to the stateful source DataLoader."""
        self.source_dataloader.set_epoch(epoch)
