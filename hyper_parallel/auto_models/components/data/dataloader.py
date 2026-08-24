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
"""Dataloader components used by Trainer targets."""

from collections.abc import Callable
from typing import Any, Optional

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class DataLoader(TorchDataLoader):
    """Build a PyTorch dataloader while retaining Trainer iterator policy."""

    def __init__(
        self,
        dataset: Dataset,
        collate_fn: Optional[Callable[[list[Any]], Any]] = None,
        *,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
        use_background_prefetcher: bool = False,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        seed: int = 42,
    ) -> None:
        """Initialize the training dataloader.

        Args:
            dataset: Runtime training dataset.
            collate_fn: Runtime collator that creates Trainer micro-batches.
            batch_size: Samples loaded by this rank for one optimizer step.
            shuffle: Whether to shuffle samples.
            drop_last: Whether to drop an incomplete local optimizer batch.
            use_background_prefetcher: Trainer iterator policy retained in the
                target configuration.
            dp_world_size: Number of data-parallel ranks.
            dp_rank: Rank within the data-parallel group.
            seed: Random seed used by the distributed sampler.

        Raises:
            ValueError: If the data-parallel topology is invalid.
        """
        if (
            isinstance(dp_world_size, bool)
            or not isinstance(dp_world_size, int)
            or dp_world_size <= 0
        ):
            raise ValueError("dp_world_size must be a positive integer")
        if (
            isinstance(dp_rank, bool)
            or not isinstance(dp_rank, int)
            or not 0 <= dp_rank < dp_world_size
        ):
            raise ValueError(
                f"dp_rank must be in [0, {dp_world_size}), but got {dp_rank!r}"
            )

        self.use_background_prefetcher = use_background_prefetcher
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=dp_world_size,
                rank=dp_rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
            if dp_world_size > 1
            else None
        )
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to deterministically shuffle distributed data."""
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


__all__ = ["DataLoader"]
