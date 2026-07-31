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
        """
        self.use_background_prefetcher = use_background_prefetcher
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )


__all__ = ["DataLoader"]
