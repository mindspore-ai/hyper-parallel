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

from collections.abc import Callable
from typing import Any, Optional

import torch
from torchdata.stateful_dataloader import StatefulDataLoader


def build_dataloader(dataloader_target: Any, **dataloader_options: Any) -> Any:
    """Build the shared LLM and Omni training dataloader.

    Args:
        dataloader_target: Resolved DataLoader target that exposes ``build``.
        **dataloader_options: Dataset, collator, batch size, rank, and sampler options.

    Returns:
        The configured training dataloader.

    Raises:
        ValueError: If no DataLoader target is configured.
    """
    if dataloader_target is None:
        raise ValueError("dataloader_target must define a build target")
    dataloader = dataloader_target.build(**dataloader_options)
    return dataloader


class DataLoader(StatefulDataLoader):
    """Build a PyTorch dataloader while retaining Trainer iterator policy."""

    def __init__(
            self,
            dataset: Any,
            batch_sampler: Any = None,
            collate_fn: Optional[Callable[[list[Any]], Any]] = None,
            *,
            batch_size: int | None = None,
            dataloader_type: str = "single",
            data_rearrange_map: Any = None,
            data_sharding: bool = False,
            drop_last: bool = True,
            use_background_prefetcher: bool = False,
            num_workers: int = 0,
            seed: int = 1234,
            dataloader_pin_memory: bool = False,
            prefetch_factor: int | None = None,
    ) -> None:
        """Initialize the training dataloader.

        Args:
            dataset: Runtime training dataset.
            batch_sampler: External component that yields rank-local index lists.
            collate_fn: Runtime collator that creates one model micro-batch.
            batch_size: Direct batch size used by Online iterable Datasets.
            dataloader_type: PanGu sampler scenario selected by the Trainer.
            data_rearrange_map: Rearrangement configuration consumed by the Trainer.
            data_sharding: PanGu cyclic sharding policy consumed by the Trainer.
            drop_last: Sampler policy retained in the target configuration.
            use_background_prefetcher: Trainer iterator policy retained in the
                target configuration.
            num_workers: Number of Dataset worker processes.
            seed: Seed for the DataLoader worker generator.
            dataloader_pin_memory: Whether workers stage batches in pinned memory.
            prefetch_factor: Batches prefetched by each worker, or ``None``.
        """
        self.dataloader_type = dataloader_type
        self.data_rearrange_map = data_rearrange_map
        self.data_sharding = data_sharding
        self.drop_last = drop_last
        self.use_background_prefetcher = use_background_prefetcher
        generator = torch.Generator().manual_seed(seed)
        worker_options = {
            "num_workers": num_workers,
            "generator": generator,
            "pin_memory": dataloader_pin_memory,
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
        set_epoch = getattr(self.batch_sampler, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)

    def __iter__(self) -> Any:
        """Repeat cyclic samplers indefinitely, matching PanGu's iterator wrapper."""
        if self.dataloader_type != "cyclic":
            yield from super().__iter__()
            return

        while True:
            yielded_batch = False
            for batch in super().__iter__():
                yielded_batch = True
                yield batch
            if not yielded_batch:
                raise ValueError("cyclic DataLoader did not produce a complete micro-batch")


__all__ = ["DataLoader", "build_dataloader"]
