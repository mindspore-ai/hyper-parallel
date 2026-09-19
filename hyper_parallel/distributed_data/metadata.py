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
"""Ahead-of-fetch metadata and plan-aware local sample loading."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch.utils.data import DataLoader, Dataset, Sampler  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.schema import SampleKey
from hyper_parallel.distributed_data.dataset_reader import (
    _IndexedDataset,
    _build_worker_options,
    _identity,
)


class _MutableIndexSampler(Sampler[int]):
    """Expose one finite plan request to a reusable DataLoader."""

    def __init__(self) -> None:
        """Initialize an empty request window."""
        self._indices: tuple[int, ...] = ()

    def __iter__(self) -> Iterator[int]:
        """Iterate over the current immutable request window."""
        return iter(self._indices)

    def __len__(self) -> int:
        """Return the current request count."""
        return len(self._indices)

    def replace(self, indices: Sequence[int]) -> None:
        """Replace the request window before building the next iterator.

        Args:
            indices: Dataset indices for the next request window.
        """
        self._indices = tuple(indices)


class PlannedSampleLoader:
    """Use DataLoader workers to fetch only samples selected by the plan."""

    VERSION = 1

    def __init__(
            self,
            dataset: Any,
            *,
            num_workers: int,
            pin_memory: bool,
            prefetch_factor: int | None,
            persistent_workers: bool,
            seed: int,
            dataloader_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a reusable plan-aware DataLoader.

        Args:
            dataset: Mapping-style Dataset aligned with the metadata sequence.
            num_workers: Native DataLoader worker count.
            pin_memory: Whether workers pin returned sample memory.
            prefetch_factor: Samples prefetched by each worker.
            persistent_workers: Whether workers persist for the loader lifetime.
            seed: Base worker seed.
            dataloader_kwargs: Additional validated DataLoader execution options.
        """
        getitem = getattr(type(dataset), "__getitem__", None)
        if not callable(getitem) or getitem is Dataset.__getitem__ or not hasattr(dataset, "__len__"):
            raise ValueError("Planned reads require a mapping-style Dataset with __len__ and __getitem__.")
        self._dataset = dataset
        self._sampler = _MutableIndexSampler()
        self._seed = seed
        self._epoch = 0
        self._worker_generator = torch.Generator().manual_seed(seed)
        worker_options = _build_worker_options(
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            generator=self._worker_generator,
            dataloader_kwargs=dataloader_kwargs,
        )
        self._data_loader = DataLoader(
            _IndexedDataset(dataset),
            batch_size=None,
            sampler=self._sampler,
            collate_fn=_identity,
            **worker_options,
        )

    def fetch(self, sample_keys: Sequence[SampleKey]) -> dict[SampleKey, Any]:
        """Fetch assigned Dataset indices in deterministic plan order.

        Args:
            sample_keys: Sample keys assigned to this Data Constructor.

        Returns:
            Mapping from planned sample keys to materialized payloads.
        """
        return self.fetch_keys(sample_keys)

    def fetch_keys(self, sample_keys: Sequence[SampleKey]) -> dict[SampleKey, Any]:
        """Fetch an ordered sequence of sample keys from this loader's Dataset.

        Args:
            sample_keys: Keys whose local ``dataset_index`` values belong to
                this loader's Dataset.

        Returns:
            Mapping from every requested key to its materialized payload.
        """
        if any(not isinstance(key, SampleKey) for key in sample_keys):
            raise ValueError("sample_keys must contain SampleKey values.")
        if len(sample_keys) != len(set(sample_keys)):
            raise ValueError("sample_keys must not contain duplicates.")
        dataset_size = len(self._dataset)
        invalid_indices = [key.dataset_index for key in sample_keys if key.dataset_index >= dataset_size]
        if invalid_indices:
            raise ValueError(
                f"Sample plan references Dataset indices outside [0, {dataset_size}): {invalid_indices}."
            )
        self._sampler.replace(tuple(key.dataset_index for key in sample_keys))
        iterator = iter(self._data_loader)
        payloads = {}
        for expected_key in sample_keys:
            try:
                indexed_payload = next(iterator)
            except StopIteration as exc:
                raise ValueError("Plan-aware DataLoader exhausted before all requested samples were read.") from exc
            if indexed_payload.dataset_index != expected_key.dataset_index:
                raise ValueError(
                    f"Plan-aware DataLoader expected index {expected_key.dataset_index}, "
                    f"but returned {indexed_payload.dataset_index}."
                )
            payloads[expected_key] = indexed_payload.payload
        return payloads

    def state_dict(self) -> dict[str, Any]:
        """Return the epoch required to seed future plan-aware workers."""
        return {
            "version": self.VERSION,
            "dataset_size": len(self._dataset),
            "seed": self._seed,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore plan-aware reader state before its first fetch.

        Args:
            state_dict: State produced by :meth:`state_dict` on this constructor.
        """
        try:
            state = dict(state_dict)
        except Exception as exc:
            raise ValueError(f"Plan-aware DataLoader state is not a mapping: {exc}") from exc
        expected = {
            "version": self.VERSION,
            "dataset_size": len(self._dataset),
            "seed": self._seed,
        }
        for name, expected_value in expected.items():
            if state.get(name) != expected_value:
                raise ValueError(
                    f"Plan-aware DataLoader checkpoint {name}={state.get(name)!r} "
                    f"does not match {expected_value!r}."
                )
        epoch = state.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Plan-aware DataLoader checkpoint epoch must be non-negative.")
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        """Record the epoch and reset the future worker-initialization seed.

        Args:
            epoch: Non-negative Dataset epoch.
        """
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative integer, but got {epoch!r}.")
        self._epoch = epoch
        self._worker_generator.manual_seed(self._seed + epoch)


__all__ = ["PlannedSampleLoader"]
