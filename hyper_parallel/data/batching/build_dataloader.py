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
from dataclasses import replace
from typing import Any

import torch
from torch.utils.data import IterableDataset, default_collate
from torchdata.stateful_dataloader import StatefulDataLoader

from hyper_parallel.data.batching.dynamic_batch import (
    FullSampleCheckpoint,
    IndexReplayCheckpoint,
    MappingReplayDataset,
    _BatchPipeline,
    _OmniPackingPipeline,
    _TokenBatchPipeline,
)
from hyper_parallel.data.batching.build_collate_fn import TextPackingCollator
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.data.parallel.batch_sampler import (
    get_sampler_checkpoint_state, has_torchdata_state, sampler_optimizer_step_progress, sampler_state_for_load,
)

from hyper_parallel.data.distributed import (
    BalancingAlgorithm,
    CostModel,
    DistributedDatasetConfig,
    SampleMetadata,
    TensorFieldSpec,
    build_distributed_dataloader,
)

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


def _resolve_load_balance(
        dataloader_target: Any, data_config: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Validate the balancing block and reject obsolete Dataset-side settings."""
    removed_options = {
        "load_balance", "distributed_dataloader", "balance_window_size", "device_prefetch_depth",
        "log_balance", "min_balance_gain", "repack_sequences", "use_data_process",
    }.intersection(data_config)

    if removed_options:
        raise ValueError(f"Move data_config balancing options {sorted(removed_options)} to dataloader.load_balance")

    load_balance = getattr(dataloader_target, "load_balance", None)

    if load_balance is not None and not isinstance(load_balance, Mapping):
        raise ValueError("dataloader.load_balance must be a mapping; omit it to disable balancing")

    return load_balance


def _build_distributed_packing_config(
        dataloader_target: Any,
        data_config: Mapping[str, Any],
        *,
        micro_batch_size: int,
        seed: int,
        max_seq_len: int | None = None,
) -> DistributedDatasetConfig:
    """Derive distributed-packing sizes and retain optional runtime tuning."""
    try:
        seq_len = int(data_config.get("max_seq_len", max_seq_len))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Distributed loading requires data_config.max_seq_len or transform.max_seq_len") from exc

    if max_seq_len is not None and seq_len != max_seq_len:
        raise ValueError("data_config.max_seq_len must match transform.max_seq_len for native batch balancing")

    options = dict(dataloader_target.load_balance)
    trainer_managed_options = {
        "seq_len",
        "local_batch_size",
        "dp_dim_names",
        "drop_last",
        "seed",
        "num_workers",
        "pin_memory",
        "prefetch_factor",
        "persistent_workers",
        "dataset_already_sharded",
    }
    conflicting_options = sorted(set(options) & trainer_managed_options)

    if conflicting_options:
        raise ValueError(
            "dataloader.load_balance cannot override Trainer-managed options "
            f"{conflicting_options}; configure DataLoader execution options on dataloader "
            "and sequence length on dataset.data_config"
        )

    return DistributedDatasetConfig(
        seq_len=seq_len,
        local_batch_size=micro_batch_size,
        dp_dim_names=("dp",),
        drop_last=bool(getattr(dataloader_target, "drop_last", True)),
        seed=seed,
        dataset_already_sharded=False,
        **options,
    )


def _native_text_metadata(sample: Mapping[str, Any]) -> SampleMetadata:
    """Describe a complete GPT output without inferring new attention boundaries."""
    seq_len = int(sample["input_ids"].shape[-1])
    return SampleMetadata(pack_tokens=seq_len, features={"P": seq_len, "D": 0})


def _resolve_policy(dataloader_target: Any, name: str, supplied: Any, model_config: Any) -> Any:
    """Prefer an explicit policy, otherwise instantiate its configured target."""

    if supplied is not None:
        return supplied

    configured = getattr(dataloader_target, name, None)

    if configured is None:
        return None

    return configured.build(model_config=model_config)


def _native_dataloader_kwargs(dataloader_target: Any) -> dict[str, Any]:
    """Forward configured execution options without changing distributed packing."""
    return {
        name: getattr(dataloader_target, name)
        for name in (
            "num_workers", "pin_memory", "prefetch_factor", "persistent_workers",
            "timeout", "worker_init_fn", "multiprocessing_context", "pin_memory_device", "in_order",
        )
        if getattr(dataloader_target, name, None) is not None
    }


def _trainer_model_batch_spec(target: Any, collate_fn: Any, data_config: Mapping[str, Any]) -> Any:
    """Resolve declared model fields without reading samples on model peers."""
    configured = getattr(target, "model_batch_spec", None)

    if configured is not None:

        return configured.build() if callable(getattr(configured, "build", None)) else configured

    declared = getattr(collate_fn, "model_batch_spec", None)

    if declared is not None:

        return declared

    if isinstance(collate_fn, TextPackingCollator):

        return {
            "input_ids": TensorFieldSpec((1, None), torch.int64),
            "labels": TensorFieldSpec((1, None), torch.int64),
            "cu_seq_lens": TensorFieldSpec((None,), torch.int32),
        }

    # Instruction and multimodal providers may emit additional model-owned fields.
    # Their layouts must be configured explicitly rather than guessed from a sample.

    if collate_fn is default_collate and not data_config.get("is_instruction_dataset", False):
        spec = {name: TensorFieldSpec((None, None), torch.int64) for name in ("input_ids", "labels")}

        if data_config.get("inter_document_masking", False):
            spec["cu_seq_lens"] = TensorFieldSpec((None, None), torch.int32)

        if data_config.get("create_ltor_fields_in_dataloader", False):
            spec.update({
                "loss_mask": TensorFieldSpec((None, None), torch.float32),
                "position_ids": TensorFieldSpec((None, None), torch.int64),
                "attention_mask": TensorFieldSpec((None, 1, None, None), torch.bool),
            })

        return spec

    return None


def _build_native_sampler_loader(
        dataset: Any,
        batch_sampler: Any,
        collate_fn: Any,
        dataloader_target: Any,
        mesh_context: Any,
        data_config: Mapping[str, Any],
        seed: int,
        *,
        metadata_fn: Callable[[Any], SampleMetadata] | None = None,
        max_seq_len: int | None = None,
        model_config: Any = None,
        cost_model: CostModel | None = None,
        balancing_algorithm: BalancingAlgorithm | None = None,
) -> Any:
    """Reuse native Dataset outputs and the existing collator in collective loading."""

    if batch_sampler is None:
        raise ValueError("Native BatchSampler balancing requires a mapping Dataset")

    if getattr(mesh_context, "device_mesh", None) is None:
        raise ValueError("Native BatchSampler balancing requires mesh_context.device_mesh")

    if int(getattr(mesh_context, "pp_size", 1)) != 1:
        raise ValueError("Native BatchSampler balancing does not yet support pipeline parallelism")

    target_callable = getattr(dataloader_target, "_target_", None)

    if isinstance(target_callable, type) and issubclass(target_callable, _DynamicBatchLoader):
        raise ValueError("Native BatchSampler balancing must not override native batches with dynamic batching")

    config = _build_distributed_packing_config(
        dataloader_target, data_config,
        micro_batch_size=batch_sampler.micro_batch_size, seed=seed,
        max_seq_len=max_seq_len,
    )
    device_type = getattr(mesh_context.device_mesh, "device_type", "cpu")

    if device_type != "npu" and "communication_backend" not in dataloader_target.load_balance:
        config = replace(config, communication_backend="gloo")

    communication_device = torch.device("cpu")

    if device_type != "cpu":
        communication_device = torch.device(device_type, getattr(torch, device_type).current_device())

    effective_metadata_fn = metadata_fn

    if not config.metadata_mode and effective_metadata_fn is None:
        effective_metadata_fn = _native_text_metadata

    effective_cost_model = _resolve_policy(dataloader_target, "cost_model", cost_model, model_config)
    return build_distributed_dataloader(
        dataset, mesh_context.device_mesh, config,
        batch_sampler=batch_sampler, metadata_fn=effective_metadata_fn,
        metadata=getattr(dataset, "metadata", None) if config.metadata_mode else None,
        collate_fn=None if config.repack_sequences else collate_fn,
        model_batch_spec=_trainer_model_batch_spec(
            dataloader_target, None if config.repack_sequences else collate_fn, data_config,
        ),
        dataloader_kwargs=_native_dataloader_kwargs(dataloader_target),
        device=communication_device,
        model_config=model_config,
        cost_model=effective_cost_model,
        balancing_algorithm=_resolve_policy(
            dataloader_target, "balancing_algorithm", balancing_algorithm, model_config,
        ),
    )


def _build_split_batch_sampler(
        dataset: Any, dataloader_target: Any, training_config: Any, mesh_context: Any, seed: int,
) -> Any:
    """Build rank-local sampling for mapping datasets; iterable sources own their order."""

    if _is_iterable_dataset(dataset):
        return None

    return build_dataset_batch_sampler(
        total_samples=len(dataset),
        micro_batch_size=training_config.micro_batch_size,
        global_batch_size=training_config.global_batch_size,
        dp_world_size=mesh_context.dp_size,
        dp_rank=mesh_context.dp_rank,
        data_rearrange_map=getattr(dataloader_target, "data_rearrange_map", None),
        sampler_type=getattr(dataloader_target, "sampler_type", "single"),
        data_sharding=getattr(dataloader_target, "data_sharding", False),
        seed=seed,
    )


def build_dataloader(
        dataloader_target: Any,
        *,
        datasets: Sequence[Any | None],
        collate_fn: Any,
        mesh_context: Any,
        training_config: Any,
        data_config: Mapping[str, Any] | None = None,
        max_seq_len: int | None = None,
        default_seed: int = 1234,
        metadata_fn: Callable[[Any], SampleMetadata] | None = None,
        model_config: Any = None,
        cost_model: CostModel | None = None,
        balancing_algorithm: BalancingAlgorithm | None = None,
) -> tuple[tuple[Any | None, ...], tuple[Any | None, ...]]:
    """Build train, validation, and test DataLoaders.

    Mapping Datasets receive a distributed batch sampler. Iterable Datasets
    control their own order. The configured target decides whether selected
    samples use fixed collation or dynamic token batching.

    Args:
        dataloader_target: DataLoader build target. A load_balance mapping enables
            balancing, including an empty mapping; omission disables it. With balancing, its optional
            model_batch_spec is a tensor layout or a Target that builds one.
            Standard GPT and Online text layouts are provided automatically;
            instruction and multimodal collators require an explicit declaration
            on this target or the collator when model parallelism is enabled.
        datasets: Train, validation, and test datasets.
        collate_fn: Collator applied after fixed or dynamic sample selection.
        mesh_context: Data-parallel mesh context.
        training_config: Batch size and random seed configuration.
        data_config: Dataset options, including the source sequence length.
        max_seq_len: Per-row packing limit and fixed padding length for Online data.
        default_seed: Seed used when no training seed is configured.
        metadata_fn: Native physical sample metadata for online
            ``dataloader.load_balance``. Defaults to complete GPT output
            lengths; VLMTrainer supplies image metadata. Offline mode uses the
            ``dataset.metadata`` instead: an indexable SampleMetadata sequence
            aligned one-to-one with that Dataset split's logical outputs.
        model_config: Effective model dimensions for the default workload estimator.
            Unsupported architectures require an explicit cost model.
        cost_model: Optional workload callback overriding the configured cost-model target.
        balancing_algorithm: Optional assignment policy overriding its configured target.

    Returns:
        DataLoader and batch-sampler tuples for the three Dataset splits.
    """

    if len(datasets) != 3:
        raise ValueError("datasets must contain train, validation, and test entries")

    if all(dataset is None for dataset in datasets):
        return (None, None, None), (None, None, None)

    if dataloader_target is None:
        raise ValueError("dataloader_target must define a build target")

    micro_batch_size = training_config.micro_batch_size
    seed = training_config.seed if training_config.seed is not None else default_seed

    dp_world_size = mesh_context.dp_size

    sampler_type = getattr(dataloader_target, "sampler_type", "single")
    data_config = data_config or {}
    load_balance = _resolve_load_balance(dataloader_target, data_config)
    dataloaders: list[Any | None] = [None] * len(datasets)
    batch_samplers: list[Any | None] = [None] * len(datasets)

    for split_index, (split_name, dataset) in enumerate(zip(("train", "valid", "test"), datasets)):
        if dataset is None:
            logger.debug("Skipping empty Dataset split=%s", split_name)
            continue

        batch_sampler = _build_split_batch_sampler(
            dataset, dataloader_target, training_config, mesh_context, seed,
        )

        if load_balance is not None:
            dataloader = _build_native_sampler_loader(
                dataset, batch_sampler, collate_fn, dataloader_target, mesh_context, data_config, seed,
                metadata_fn=metadata_fn, max_seq_len=max_seq_len,
                model_config=model_config, cost_model=cost_model, balancing_algorithm=balancing_algorithm,
            )

        else:
            dataloader = dataloader_target.build(
                dataset=dataset, collate_fn=collate_fn, batch_sampler=batch_sampler,
                batch_size=micro_batch_size, sampler_type=sampler_type,
                dp_world_size=dp_world_size, max_seq_len=max_seq_len, seed=seed,
            )

        logger.debug(
            "Built DataLoader split=%s, dataset=%s, batch_sampler=%s",
            split_name, type(dataset).__name__,
            type(batch_sampler).__name__ if batch_sampler is not None else None,
        )
        dataloaders[split_index] = dataloader
        # The collective loader owns sampler checkpoint/epoch state; exposing
        # its speculative cursor to the Trainer would skip prefetched batches.
        batch_samplers[split_index] = None if load_balance is not None else batch_sampler

    logger.debug("Finished building train/valid/test DataLoaders")
    return tuple(dataloaders), tuple(batch_samplers)


class RepeatingDataLoader:
    """Repeat Online data from the beginning when the source loader is exhausted.

    Training epoch boundaries reuse the same iterator. Dynamic loaders handle
    repetition internally; this wrapper repeats ordinary loaders' batches.

    Attributes:
        source_dataloader: Loader that reads and batches the data.
        source_repeat_num: Number of restarts after EOF; initially zero.
        _micro_batches_since_restart: Source batches yielded in the current pass,
            used to reject an empty pass even when restoring at EOF.
        _repeating_iterator: Iterator reused across training epochs.
    """

    def __init__(self, source_dataloader: Any) -> None:
        """Wrap a finite data source.

        Args:
            source_dataloader: Loader to restart when exhausted.
        """
        self.source_dataloader = source_dataloader
        self.source_repeat_num = 0
        self._micro_batches_since_restart = 0
        self._repeating_iterator: Iterator[Any] | None = None

    def __iter__(self) -> Iterator[Any]:
        """Reuse the iterator across training epochs."""

        if self._repeating_iterator is None:
            self._repeating_iterator = self._repeat_batches()

        return self._repeating_iterator

    def _repeat_batches(self) -> Iterator[Any]:
        """Repeat at EOF; raise if a full pass produces no batches."""
        while True:
            for micro_batch in self.source_dataloader:
                self._micro_batches_since_restart += 1
                yield micro_batch

            if not self._micro_batches_since_restart:
                raise ValueError("Online repeat=true source produced no batches on this rank")

            self.source_repeat_num += 1
            self._micro_batches_since_restart = 0
            self.source_dataloader.set_epoch(self.source_repeat_num)
            logger.warning(
                "Online data source exhausted; repeating data from the beginning (source_repeat_num=%s).",
                self.source_repeat_num,
            )

    def set_epoch(self, epoch: int) -> None:
        """Preserve the source pass and buffered iterator across training epochs."""
        del epoch

    def state_dict(self) -> dict[str, Any]:
        """Save repeat progress and source loader state."""
        checkpoint_state = {
            "source_epoch": self.source_repeat_num,
            "batches_in_pass": self._micro_batches_since_restart,
            "dataloader": self.source_dataloader.state_dict(),
        }
        return checkpoint_state

    def load_state_dict(self, checkpoint_state: dict[str, Any]) -> None:
        """Restore repeat progress and source loader state."""
        self.source_repeat_num = checkpoint_state["source_epoch"]
        self._micro_batches_since_restart = checkpoint_state["batches_in_pass"]
        self.source_dataloader.load_state_dict(checkpoint_state["dataloader"])
        self._repeating_iterator = None

    def state_dict_for_load(self) -> dict[str, Any]:
        """Build the state template for checkpoint loading."""
        build_source_restore_state = getattr(
            self.source_dataloader, "state_dict_for_load", self.source_dataloader.state_dict
        )
        checkpoint_state = {
            "source_epoch": self.source_repeat_num,
            "batches_in_pass": self._micro_batches_since_restart,
            "dataloader": build_source_restore_state(),
        }
        return checkpoint_state


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

            # Native worker snapshots may precede the delivered optimizer boundary.
            if hasattr(batch_sampler, "enable_source_batch_resume"):
                batch_sampler.enable_source_batch_resume()

    def state_dict(self) -> dict[str, Any]:
        """Save the native delivered cursor alongside torchdata's worker state."""
        state = super().state_dict()

        if "_snapshot" not in state:
            return state

        snapshot = state["_snapshot"]["_main_snapshot"]
        sampler = snapshot.get("_index_sampler_state")

        if isinstance(sampler, dict) and "consumed_samples" in sampler:
            sampler = dict(sampler)
            sampler["consumed_samples"] += (
                state.get("_steps_since_snapshot", 0) * self.batch_sampler.global_micro_batch_size
            )
            state["_index_sampler_state"] = sampler

        return state

    def state_dict_for_load(self) -> dict[str, Any]:
        """Request native worker state and shared or legacy sampler cursors."""
        return sampler_state_for_load(self.state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore torchdata state, or a balanced loader's complete native cursor.

        Args:
            state_dict: State saved by the ordinary or balanced native loader.
        """
        # The training frame may still retain the old iterator during rollback.
        # Stop its producers before torchdata drops the loader's iterator reference.
        iterator = self._iterator
        if self.num_workers > 0 and iterator is not None:
            # torchdata exposes worker shutdown only through its iterator's private API.
            iterator._shutdown_workers()  # pylint: disable=protected-access
        sampler = get_sampler_checkpoint_state(state_dict)
        snapshot = state_dict.get("_snapshot", {}).get("_main_snapshot", {})
        native_sampler = sampler if sampler is not None else snapshot.get("_index_sampler_state")
        repartition = isinstance(native_sampler, Mapping) and any(
            native_sampler.get(name) != getattr(self.batch_sampler, name, native_sampler.get(name))
            for name in ("global_batch_size", "dp_world_size")
        )

        if state_dict.get("global_rank") is None and has_torchdata_state(state_dict) and not repartition:
            if isinstance(native_sampler, Mapping) and native_sampler.get("epoch") is not None:
                self.set_epoch(native_sampler["epoch"])

            super().load_state_dict(state_dict)
            return

        if sampler is None:
            raise ValueError("Balanced checkpoint has no native sampler cursor on this rank")

        if sampler["epoch"] < 0 or sampler_optimizer_step_progress(sampler):
            raise ValueError("Switching loaders requires a complete optimizer-step cursor")

        self.batch_sampler.load_state_dict(sampler)
        self.set_epoch(sampler["epoch"])
        super().load_state_dict({})
        self.next_iter_state = None

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch state to the Dataset and configured batch sampler.

        Args:
            epoch: Epoch index used by the sampler and Dataset.
        """

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
        batch_size: Number of independently packed rows in each micro-batch.
        dp_world_size: Number of data-parallel source lanes.
        save_by_idx: Whether to checkpoint buffer entries as output indices.
            Mapping Datasets default to true and iterable Datasets default to false.
        max_seq_len: Required per-row packing limit and fixed padding length.
            Oversized samples are skipped.
        min_buffered_samples: Minimum candidate samples buffered before batching.
        drop_last: Drop an incomplete group of packed rows at finite EOF.
            Repeating iteration retains candidates across source restarts.
        repeat: Restart the source at EOF while retaining buffered candidates.
            None inherits the Trainer's Online policy; standalone iteration is finite.
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
            max_seq_len: int,
            min_buffered_samples: int = 200,
            drop_last: bool = True,
            repeat: bool | None = None,
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

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if max_seq_len is None or max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        resolved_dp_world_size = dp_world_size

        if resolved_dp_world_size <= 0:
            raise ValueError("dp_world_size must be positive")

        if batch_sampler is not None and batch_sampler.dp_world_size != resolved_dp_world_size:
            raise ValueError("batch_sampler.dp_world_size must match dp_world_size")

        if batch_sampler is not None:
            # Drop only incomplete packed rows, retaining every raw source tail.
            batch_sampler.drop_last = False

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.repeat = repeat is True
        self.source_repeat_num = 0
        self._source_batches_since_restart = 0
        self._batch_iterator: Iterator[Mapping[str, Any]] | None = None
        self.batch_collate_fn = collate_fn
        self.dp_world_size = resolved_dp_world_size

        # build_candidate_buffer
        self._batch_pipeline = self._build_batch_pipeline(dataset)
        self.candidate_buffer = self._batch_pipeline.build_candidate_buffer(
            token_budget=max_seq_len,
            min_buffered_samples=min_buffered_samples,
        )

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
        replay_dataset = (dataset if is_iterable else
                          MappingReplayDataset(dataset, output_index_for_resume=resolved_save_by_idx))
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
            buffer_checkpoint = FullSampleCheckpoint(replay_dataset, checkpoint_replay_dataset)

        return buffer_checkpoint

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Start source iteration while retaining restored buffer state."""

        if self.repeat and self._batch_iterator is not None:
            return self._batch_iterator

        if not self.resume_pending:
            self.candidate_buffer.buffer.clear()
            self.candidate_buffer.buffer_output_indices.clear()
            self.candidate_buffer.buffer_token_count = 0
            self._source_batches_since_restart = 0

        self._batch_iterator = self._batch_data_generator()
        self.resume_pending = False
        return self._batch_iterator

    def _batch_data_generator(self) -> Iterator[Mapping[str, Any]]:
        """Lazily yield micro-batches as source samples fill the candidate buffer."""
        # Use ready checkpointed candidates before advancing the source.
        while self.candidate_buffer.is_ready_for_micro_batch(self.batch_size):
            yield self._build_micro_batch()

        while True:
            for source_samples in self.source_dataloader:
                buffered_samples = len(self.candidate_buffer.buffer)

                for source_sample in source_samples:
                    self._buffer_checkpoint.put_source_item(source_sample, self.candidate_buffer)

                if len(self.candidate_buffer.buffer) > buffered_samples:
                    self._source_batches_since_restart += 1

                while self.candidate_buffer.is_ready_for_micro_batch(self.batch_size):
                    yield self._build_micro_batch()

            if not self.repeat:
                break

            if not self._source_batches_since_restart:
                raise ValueError("Online repeat=true source produced no batches on this rank")

            self.source_repeat_num += 1
            self._source_batches_since_restart = 0
            self.source_dataloader.set_epoch(self.source_repeat_num)
            logger.warning(
                "Online data source exhausted; keeping buffer and repeating (source_repeat_num=%s).",
                self.source_repeat_num,
            )

        # Finite sources emit remaining candidates even below readiness thresholds.
        logger.debug(
            "Dynamic loader source exhausted: buffered_samples=%d buffered_tokens=%d "
            "token_budget=%d drop_last=%s; draining remaining candidates",
            len(self.candidate_buffer.buffer), self.candidate_buffer.buffer_token_count,
            self.candidate_buffer.token_budget, self.drop_last, enabled=True,
        )

        while not self.candidate_buffer.empty():
            batch = self._build_micro_batch()

            if not self.drop_last or batch["input_ids"].shape[0] == self.batch_size:
                yield batch

        logger.debug("Dynamic loader tail drained; no candidates remain", enabled=True)

    @abstractmethod
    def _build_batch_pipeline(self, dataset: Any) -> _BatchPipeline:
        """Build the batching pipeline selected by the concrete Loader."""
        raise NotImplementedError

    def _build_micro_batch(self) -> Mapping[str, Any]:
        """Pack up to B independent rows, pad together, and encode one batch."""
        packed_rows = []

        # Pack rows independently, retaining candidates that do not fit for later rows.
        for _ in range(self.batch_size):
            if self.candidate_buffer.empty():
                break

            selected_samples = self.candidate_buffer.get_micro_batch()
            packed_rows.extend(self._batch_pipeline.finalize_selected_samples(selected_samples))
        logger.debug(
            "Dynamic loader selected: samples=%d token_budget=%d remaining_samples=%d remaining_tokens=%d",
            len(packed_rows), self.candidate_buffer.token_budget, len(self.candidate_buffer.buffer),
            self.candidate_buffer.buffer_token_count, enabled=True,
        )

        for row in packed_rows:
            if row["input_ids"].shape[-1] > self.candidate_buffer.token_budget:
                raise ValueError("Encoded packed row exceeds max_seq_len; packing_length must include encoded tokens")

        # Choose one padding length only after the packed rows are complete.
        batch = self.batch_collate_fn(packed_rows)
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

        if self.repeat:
            # Retain the checkpoint layout previously saved by RepeatingDataLoader.
            state = {
                "source_epoch": self.source_repeat_num,
                "batches_in_pass": self._source_batches_since_restart,
                "dataloader": state,
            }

        checkpoint_state = copy.deepcopy(state)

        return checkpoint_state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the future source cursor and unconsumed dynamic buffer.

        Args:
            state_dict: Saved source progress and unconsumed candidate buffer.
        """
        checkpoint_state = copy.deepcopy(dict(state_dict))
        self.source_repeat_num = checkpoint_state.get("source_epoch", 0)
        self._source_batches_since_restart = checkpoint_state.get("batches_in_pass", 0)
        checkpoint_state = checkpoint_state.get("dataloader", checkpoint_state)
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
        self._batch_iterator = None

    def set_epoch(self, epoch: int) -> None:
        """Forward training epochs only for finite iteration.

        Repeating iteration advances source epochs itself at EOF.

        Args:
            epoch: Epoch index forwarded to the source DataLoader.
        """

        if not self.repeat:
            self.source_dataloader.set_epoch(epoch)


class TokenBatchLoader(_DynamicBatchLoader):
    """Select fully encoded samples within one token budget."""

    def _build_batch_pipeline(self, dataset: Any) -> _BatchPipeline:
        """Build token-budget batching for fully encoded Dataset samples."""
        batch_pipeline = _TokenBatchPipeline()
        return batch_pipeline


class OmniPackingLoader(_DynamicBatchLoader):
    """Pack selected samples after applying the Dataset encoding lifecycle."""

    def _build_batch_pipeline(self, dataset: Any) -> _BatchPipeline:
        """Build packing around the normalized Dataset encoding lifecycle."""
        batch_pipeline = _OmniPackingPipeline.from_dataset(dataset)
        return batch_pipeline


# Compatibility name for callers of the pre-refactor dynamic text loader.
DynamicBatchDataLoader = TokenBatchLoader
