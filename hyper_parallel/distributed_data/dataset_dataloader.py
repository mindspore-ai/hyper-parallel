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
"""Ordinary iteration over ready model inputs from a declared data contract."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from hyper_parallel.distributed_data.dataset import DistributedDataset
from hyper_parallel.distributed_data.packed_balancing import LocalBalancingDataLoader


class DatasetDataLoader(Iterator[list[Any]]):
    """Deliver device-ready steps without user-side prefetch consumption hooks.

    The underlying loader retains ownership of buffering and communication.
    Device handoff happens on the training thread, so its current stream waits
    for the copy event and owns tensor storage before ordinary iteration returns.
    """

    def __init__(self, dataset: DistributedDataset, loader: LocalBalancingDataLoader, device: Any) -> None:
        """Bind dataset interpretation to one existing local-step loader."""
        self.dataset = dataset
        self.config = loader.config
        self.device = device
        self.last_host_batch: list[Any] | None = None
        self._loader = loader

    def __iter__(self) -> DatasetDataLoader:
        """Start source workers on the caller thread and preserve active progress."""
        iter(self._loader)
        return self

    def __next__(self) -> list[Any]:
        """Return ready microbatches with the dataset's CPU field policy applied."""
        batch = next(self._loader)
        self.last_host_batch = self._loader.last_host_batch
        return batch

    def prefetch_plan(self) -> None:
        """Start next-step planning after compute submission on every rank."""
        self._loader.prefetch_plan()

    def prefetch(self) -> None:
        """Launch next-step exchange before the trainer synchronizes compute."""
        self._loader.prefetch()

    def wait_for_prefetch(self) -> None:
        """Drain pending communication and copies without consuming the next step."""
        self._loader.wait_for_prefetch()

    def __len__(self) -> int:
        """Return the source length or explicit step limit."""
        return len(self._loader)

    @property
    def step(self) -> int:
        """Return the number of steps consumed in the current epoch."""
        return self._loader.step

    @property
    def group_ranks(self) -> tuple[int, ...]:
        """Return the rank-local balancing domain."""
        return self._loader.group_ranks

    @property
    def last_balance_stats(self) -> dict[str, Any] | None:
        """Return diagnostics for the consumed step, not the prefetched step."""
        return self._loader.last_balance_stats

    def set_epoch(self, epoch: int) -> None:
        """Drain pending work and forward the epoch to the original source.

        Args:
            epoch: Source epoch to begin.
        """
        self._loader.set_epoch(epoch)
        self.last_host_batch = None

    def state_dict(self) -> dict[str, Any]:
        """Delegate the local loader's explicit checkpoint support contract."""
        return self._loader.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Delegate checkpoint restoration without hiding unsupported modes.

        Args:
            state: Previously saved loader state, if the underlying mode supports it.
        """
        self._loader.load_state_dict(state)

    def close(self) -> None:
        """Drain pending communication before distributed process-group teardown."""
        self._loader.close()
        self.last_host_batch = None

    def __enter__(self) -> DatasetDataLoader:
        """Support deterministic cleanup on an early training-loop exit."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Finish in-flight producer work without suppressing training errors."""
        self.close()


__all__ = ["DatasetDataLoader"]
