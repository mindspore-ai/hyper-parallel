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
"""Public LLM and Omni dataloader build stage."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from hyper_models.components.datasets.contracts import is_iterable_dataset
from hyper_models.components.datasets.parallel import build_dataset_batch_sampler

from hyper_models.components.datasets.dataset_logging import get_dataset_logger

logger = get_dataset_logger(__name__)


def build_dataloader(
        dataloader_target: Any,
        *,
        datasets: Sequence[Any | None],
        collate_fn: Any,
        mesh_context: Any,
        training_config: Any,
        default_seed: int = 1234,
) -> tuple[tuple[Any | None, ...], tuple[Any | None, ...]]:
    """Build dataloaders and batch samplers for train, validation, and test.

    Args:
        dataloader_target: DataLoader build target.
        datasets: Train, validation, and test datasets.
        collate_fn: Batch collator.
        mesh_context: Data-parallel mesh context.
        training_config: Batch size and random seed configuration.
        default_seed: Seed used when no training seed is configured.

    Returns:
        Dataloaders and batch samplers for each split.
    """
    if len(datasets) != 3:
        raise ValueError("datasets must contain train, validation, and test entries")
    if all(dataset is None for dataset in datasets):
        empty_splits = (None, None, None)
        return empty_splits, empty_splits
    if dataloader_target is None:
        raise ValueError("dataloader_target must define a build target")

    dataloader_type = getattr(dataloader_target, "dataloader_type", "single")
    micro_batch_size = training_config.micro_batch_size
    random_seed = training_config.seed if training_config.seed is not None else default_seed
    drop_last = bool(getattr(dataloader_target, "drop_last", True))
    if dataloader_type == "single" and not drop_last:
        raise ValueError("single sampling currently requires dataloader.drop_last=True")

    data_rearrange_map = getattr(dataloader_target, "data_rearrange_map", None)
    data_sharding = bool(getattr(dataloader_target, "data_sharding", False))
    dataloaders = []
    batch_samplers = []
    logger.debug(
        "Building DataLoaders: type=%s, micro_batch_size=%d, dp_rank=%d, dp_world_size=%d, drop_last=%s",
        dataloader_type, micro_batch_size, mesh_context.dp_rank, mesh_context.dp_size, drop_last, enabled=True
    )

    for split_name, dataset in zip(("train", "valid", "test"), datasets):
        if dataset is None:
            logger.debug("Skipping empty Dataset split=%s", split_name)
            dataloaders.append(None)
            batch_samplers.append(None)
            continue

        if is_iterable_dataset(dataset):
            logger.debug("Building iterable DataLoader split=%s, dataset=%s", split_name, type(dataset).__name__)
            dataloaders.append(dataloader_target.build(dataset=dataset, collate_fn=collate_fn, batch_sampler=None,
                                                       batch_size=micro_batch_size, seed=random_seed, ))
            batch_samplers.append(None)
            continue

        total_samples = len(dataset)
        if total_samples == 0:
            if split_name == "train":
                raise ValueError("train dataset must contain at least one sample")
            dataloaders.append(None)
            batch_samplers.append(None)
            continue

        batch_sampler = build_dataset_batch_sampler(
            total_samples=total_samples,
            micro_batch_size=micro_batch_size,
            global_batch_size=training_config.global_batch_size,
            dp_world_size=mesh_context.dp_size,
            dp_rank=mesh_context.dp_rank,
            drop_last=drop_last,
            data_rearrange_map=data_rearrange_map,
            sampler_type=dataloader_type,
            data_sharding=data_sharding,
            seed=random_seed,
        )
        logger.debug(
            "Built batch sampler split=%s, dataset=%s, total_samples=%d, sampler=%s",
            split_name, type(dataset).__name__, total_samples, type(batch_sampler).__name__,
        )
        dataloaders.append(
            dataloader_target.build(
                dataset=dataset, collate_fn=collate_fn, batch_sampler=batch_sampler, seed=random_seed,
            )
        )
        batch_samplers.append(batch_sampler)

    dataloader_splits = tuple(dataloaders)
    batch_sampler_splits = tuple(batch_samplers)
    logger.debug("Finished building train/valid/test DataLoaders")
    return dataloader_splits, batch_sampler_splits


class DataLoader(StatefulDataLoader):
    """Build a PyTorch dataloader while retaining Trainer iterator policy."""

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
        """Initialize the stateful dataloader.

        Args:
            dataset: Source dataset.
            batch_sampler: Rank-local batch sampler.
            collate_fn: Batch collator.
            batch_size: Batch size for iterable datasets.
            drop_last: Whether to drop incomplete batches.
            use_background_prefetcher: Whether Trainer enables background prefetching.
            num_workers: Number of DataLoader workers.
            seed: DataLoader random seed.
            pin_memory: Whether to use pinned memory.
            prefetch_factor: Batches prefetched by each worker.
        """
        self.drop_last = drop_last
        self.use_background_prefetcher = use_background_prefetcher
        generator = torch.Generator().manual_seed(seed)
        worker_options = {"num_workers": num_workers, "generator": generator, "pin_memory": pin_memory}
        if num_workers > 0 and prefetch_factor is not None:
            worker_options["prefetch_factor"] = prefetch_factor
        if batch_sampler is None:
            resolved_batch_size = 1 if batch_size is None else batch_size
            super().__init__(dataset=dataset, batch_size=resolved_batch_size, collate_fn=collate_fn,
                             drop_last=drop_last, **worker_options)
        else:
            super().__init__(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, **worker_options)

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch state when the configured batch sampler supports it."""
        set_epoch = getattr(self.batch_sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

__all__ = ["DataLoader", "build_dataloader"]
