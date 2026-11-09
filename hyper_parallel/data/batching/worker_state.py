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
"""Checkpoint process-local augmentation state through the Dataset protocol."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch  # pylint: disable=forbidden-backend-import


class WorkerRNGDataset:
    """Retain worker CPU RNG alongside a map-style Dataset's private state.

    StatefulDataLoader snapshots the Dataset after a batch, including collator
    execution. Saving here associates RNG with the consumed worker cursor rather
    than the parent process or a later prefetched batch. Private generators must
    still be included in the wrapped Dataset's state_dict/load_state_dict pair.
    """

    def __init__(self, dataset: Any) -> None:
        """Wrap a map-style Dataset without changing its sample generation.

        Args:
            dataset: Dataset providing integer-indexed access.
        """
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the wrapped Dataset's sample count."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        """Return the original sample for the requested index."""
        return self.dataset[index]

    def __getitems__(self, indices: list[int]) -> list[Any]:
        """Preserve optional vectorized fetching before the collator executes."""
        fetch = getattr(self.dataset, "__getitems__", None)
        return fetch(indices) if callable(fetch) else [self.dataset[index] for index in indices]

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch changes to the source Dataset.

        Args:
            epoch: Source epoch selected by the DataLoader.
        """
        setter = getattr(self.dataset, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def state_dict(self) -> dict[str, Any]:
        """Capture weights-only-loadable CPU RNG and optional Dataset state.

        Returns:
            Process-local RNG and Dataset-owned private generator/cursor state.
        """
        dataset_state = getattr(self.dataset, "state_dict", None)
        private_state = dataset_state() if callable(dataset_state) else None
        numpy_state = np.random.get_state()
        return {
            "dataset": private_state,
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch_cpu": torch.get_rng_state(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore private Dataset state before the associated worker RNG.

        Args:
            state_dict: Snapshot produced by state_dict in the same worker topology.
        """
        if state_dict["dataset"] is not None:
            self.dataset.load_state_dict(state_dict["dataset"])
        numpy_state = state_dict["numpy"]
        random.setstate(state_dict["python"])
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        torch.set_rng_state(state_dict["torch_cpu"])
