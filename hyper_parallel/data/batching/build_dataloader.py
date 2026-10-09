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
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from hyper_parallel.data.batching.dynamic_batch import (
    FullSampleCheckpoint,
    IndexReplayCheckpoint,
    MappingReplayDataset,
    _BatchPipeline,
    _OmniPackingPipeline,
    _TokenBatchPipeline,
)
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

    dataloaders: list[Any | None] = [None] * len(datasets)
    batch_samplers: list[Any | None] = [None] * len(datasets)

    for split_index, (split_name, dataset) in enumerate(zip(("train", "valid", "test"), datasets)):
        if dataset is None:
            logger.debug("Skipping empty Dataset split=%s", split_name)
            continue

        batch_sampler = None
        if not _is_iterable_dataset(dataset):
            batch_sampler = build_dataset_batch_sampler(
                total_samples=len(dataset),
                micro_batch_size=training_config.micro_batch_size,
                global_batch_size=training_config.global_batch_size,
                dp_world_size=mesh_context.dp_size,
                dp_rank=mesh_context.dp_rank,
                drop_last=getattr(dataloader_target, "drop_last", True),
                data_rearrange_map=getattr(dataloader_target, "data_rearrange_map", None),
                sampler_type=getattr(dataloader_target, "sampler_type", "single"),
                data_sharding=getattr(dataloader_target, "data_sharding", False),
                seed=training_config.seed if training_config.seed is not None else default_seed,
            )

        dataloaders[split_index] = dataloader_target.build(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            batch_size=training_config.micro_batch_size,
            sampler_type=getattr(dataloader_target, "sampler_type", "single"),
            dp_world_size=mesh_context.dp_size,
            max_seq_len=max_seq_len,
            seed=training_config.seed if training_config.seed is not None else default_seed,
        )
        logger.debug(
            "Built DataLoader split=%s, dataset=%s, batch_sampler=%s",
            split_name,
            type(dataset).__name__,
            type(batch_sampler).__name__ if batch_sampler is not None else None,
        )
        batch_samplers[split_index] = batch_sampler

    logger.debug("Finished building train/valid/test DataLoaders")
    return tuple(dataloaders), tuple(batch_samplers)


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
            sampler_type: str = "single",
            num_workers: int = 0,
            seed: int = 1234,
            pin_memory: bool = False,
            prefetch_factor: int | None = None,
    ) -> None:
        """Initialize the stateful DataLoader."""
        if sampler_type not in ("single", "cyclic"):
            raise ValueError("sampler_type must be 'single' or 'cyclic'")

        self.drop_last = drop_last
        self.sampler_type = sampler_type
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
        """Forward epoch state to the Dataset and configured batch sampler."""
        if self.batch_sampler is not None and hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

        epoch_setter = getattr(self.dataset, "set_epoch", None)
        if callable(epoch_setter):
            dataset_epoch = epoch if self.sampler_type == "cyclic" else 0
            epoch_setter(dataset_epoch)


class _DynamicBatchLoader(ABC):
    """Own common dynamic source iteration, buffering, and checkpoint state.

    Subclasses define how candidates are selected and finalized before the
    configured collator builds a batch.

    Args:
        dataset: Online mapping or iterable Dataset producing ModelSamples.
        batch_sampler: Optional mapping-style batch sampler.
        collate_fn: Final batch collator receiving the selected ModelSamples.
        batch_size: Configured micro batch size used to derive the token budget.
        dp_world_size: Number of data-parallel source lanes.
        save_by_idx: Whether to checkpoint buffer entries as output indices.
            Mapping Datasets default to true and iterable Datasets default to false.
        max_seq_len: Maximum sample length used to derive the token budget.
        token_budget: Optional candidate-selection budget. Values smaller than
            one sample still emit that sample alone instead of truncating it.
        min_buffered_samples: Minimum candidate samples buffered before batching.
        drop_last: Compatibility option retained from fixed DataLoader configuration.
            Dynamic buffers are always drained when the source is exhausted.
        sampler_type: Whether each Dataset epoch reuses one order or advances
            the deterministic cyclic order.
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
            token_budget: int | None = None,
            min_buffered_samples: int = 200,
            drop_last: bool = True,
            sampler_type: str = "single",
            num_workers: int = 0,
            seed: int = 1234,
            pin_memory: bool = False,
            prefetch_factor: int | None = None,
    ) -> None:
        """Initialize source reading, token selection, and Online packing."""
        # pylint: disable=too-many-locals
        if collate_fn is None:
            raise ValueError(f"{type(self).__name__} requires a collate_fn")

        if max_seq_len is None:
            raise ValueError(f"{type(self).__name__} requires data_transform.max_seq_len")
        if token_budget is not None and token_budget <= 0:
            raise ValueError("token_budget must be positive")

        resolved_dp_world_size = dp_world_size
        if resolved_dp_world_size <= 0:
            raise ValueError("dp_world_size must be positive")

        sampler_dp_world_size = getattr(batch_sampler, "dp_world_size", resolved_dp_world_size)
        if sampler_dp_world_size != resolved_dp_world_size:
            raise ValueError("batch_sampler.dp_world_size must match dp_world_size")

        self.drop_last = drop_last
        self.batch_collate_fn = collate_fn
        self.dp_world_size = resolved_dp_world_size

        # build_candidate_buffer
        self._batch_pipeline = self._build_batch_pipeline(dataset)
        resolved_token_budget = batch_size * max_seq_len if token_budget is None else token_budget
        self.candidate_buffer = self._batch_pipeline.build_candidate_buffer(
            token_budget=resolved_token_budget,
            min_buffered_samples=min_buffered_samples,
        )
        if batch_sampler is not None:
            enable_source_resume = getattr(batch_sampler, "enable_source_batch_resume", None)
            if callable(enable_source_resume):
                enable_source_resume()

        # Resuming training from a checkpoint
        self._buffer_checkpoint = self._build_buffer_checkpoint(dataset, save_by_idx)

        self.source_dataloader = FixedBatchDataLoader(
            dataset=self._buffer_checkpoint.source_dataset,
            batch_sampler=batch_sampler,
            # Keep source samples separate until the candidate buffer selects them.
            collate_fn=list,
            batch_size=1 if batch_sampler is None else batch_size,
            drop_last=False,
            sampler_type=sampler_type,
            num_workers=num_workers,
            seed=seed,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
        self.resume_pending = False

    @staticmethod
    def _build_buffer_checkpoint(
            dataset: Any,
            save_by_idx: bool | None,
    ) -> IndexReplayCheckpoint | FullSampleCheckpoint:
        """Build candidate-buffer checkpoint storage for the Dataset access mode."""
        is_iterable = _is_iterable_dataset(dataset)
        resolved_save_by_idx = not is_iterable if save_by_idx is None else save_by_idx

        # Mapping Datasets have stable indices. Iterable Datasets must explicitly
        # provide replay access before their buffer can be checkpointed by index.
        replay_dataset = dataset if is_iterable else MappingReplayDataset(dataset)
        supports_replay = not is_iterable or IndexReplayCheckpoint.supports(dataset)
        if resolved_save_by_idx and not supports_replay:
            raise ValueError("save_by_idx=True requires get_item() and output_index_for_resume")

        if is_iterable and hasattr(dataset, "output_index_for_resume"):
            dataset.output_index_for_resume = resolved_save_by_idx

        if resolved_save_by_idx:
            buffer_checkpoint = IndexReplayCheckpoint(replay_dataset)
        else:
            # Keep replay access only for loading an earlier index-buffer checkpoint.
            checkpoint_replay_dataset = replay_dataset if supports_replay else None
            buffer_checkpoint = FullSampleCheckpoint(dataset, checkpoint_replay_dataset)

        return buffer_checkpoint

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Start source iteration while retaining restored buffer state."""
        if not self.resume_pending:
            self.candidate_buffer.buffer.clear()
            self.candidate_buffer.buffer_output_indices.clear()
            self.candidate_buffer.buffer_token_count = 0

        source_iterator = iter(self.source_dataloader)
        batch_iterator = self._batch_data_generator(source_iterator)
        self.resume_pending = False

        return batch_iterator

    def _batch_data_generator(
            self,
            source_iterator: Iterator[list[Any]],
    ) -> Iterator[Mapping[str, Any]]:
        """Lazily yield micro-batches as source samples fill the candidate buffer."""
        # Stage 1: Use ready checkpointed candidates before advancing the source.
        while self.candidate_buffer.is_ready_for_micro_batch():
            yield self._build_micro_batch()

        # Stage 2: Retain unselected candidates while incrementally reading source batches.
        for source_samples in source_iterator:
            for source_sample in source_samples:
                self._buffer_checkpoint.put_source_item(source_sample, self.candidate_buffer)

            while self.candidate_buffer.is_ready_for_micro_batch():
                yield self._build_micro_batch()

        # Stage 3: At EOF, bypass readiness thresholds so no pulled candidate is lost.
        while not self.candidate_buffer.empty():
            yield self._build_micro_batch()

    @abstractmethod
    def _build_batch_pipeline(self, dataset: Any) -> _BatchPipeline:
        """Build the batching pipeline selected by the concrete Loader."""
        raise NotImplementedError

    def _build_micro_batch(self) -> Mapping[str, Any]:
        """Select, finalize, collate, and encode one micro-batch."""
        selected_samples = self.candidate_buffer.get_micro_batch()
        finalized_samples = self._batch_pipeline.finalize_selected_samples(selected_samples)

        # build batch data
        batch = self.batch_collate_fn(finalized_samples)
        encoded_batch = self._batch_pipeline.encode_batch(batch)
        return encoded_batch

    def state_dict(self) -> dict[str, Any]:
        """Capture the future source cursor and unconsumed dynamic buffer."""
        state = {
            "dp_world_size": self.dp_world_size,
            "source_dataloader": self.source_dataloader.state_dict(),
            "save_by_idx": self._buffer_checkpoint.save_by_idx,
            "buffer": self._buffer_checkpoint.get_buffer_state(self.candidate_buffer),
            "buffer_token_count": self.candidate_buffer.buffer_token_count,
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
        restored_buffer, restored_indices = self._buffer_checkpoint.restore_buffer(
            saved_buffer,
            previous_save_by_idx,
            self.candidate_buffer,
        )

        restored_token_count = sum(sample_length for _, sample_length in restored_buffer)
        if restored_token_count != checkpoint_state["buffer_token_count"]:
            raise ValueError("buffer_token_count does not match the restored dynamic buffer")

        self.source_dataloader.load_state_dict(checkpoint_state["source_dataloader"])
        self.candidate_buffer.buffer = restored_buffer
        self.candidate_buffer.buffer_output_indices = restored_indices
        self.candidate_buffer.buffer_token_count = restored_token_count
        self.resume_pending = True

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch state to the stateful source DataLoader."""
        self.source_dataloader.set_epoch(epoch)


class TokenBatchLoader(_DynamicBatchLoader):
    """Select fully encoded samples within one token budget."""

    def _build_batch_pipeline(self, _dataset: Any) -> _BatchPipeline:
        """Build token-budget batching for fully encoded Dataset samples."""
        batch_pipeline = _TokenBatchPipeline()
        return batch_pipeline


class OmniPackingLoader(_DynamicBatchLoader):
    """Pack selected samples after applying the Dataset encoding lifecycle."""

    def _build_batch_pipeline(self, dataset: Any) -> _BatchPipeline:
        """Build packing around the normalized Dataset encoding lifecycle."""
        batch_pipeline = _OmniPackingPipeline.from_dataset(dataset)
        return batch_pipeline
