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

from collections.abc import Mapping, Sequence
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch.utils.data import DataLoader, Dataset  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.schema import SampleKey
from hyper_parallel.distributed_data.dataset_reader import (
    _build_worker_options,
    _identity,
)


class PlannedSampleLoader:
    """Use DataLoader workers to fetch only samples selected by the plan."""

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
        self._indices: list[int] = []
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
            dataset,
            batch_size=None,
            sampler=self._indices,
            collate_fn=_identity,
            **worker_options,
        )

    def fetch_samples(self, sample_keys: Sequence[SampleKey]) -> dict[SampleKey, Any]:
        """Fetch Dataset samples for the requested keys, preserving their order.

        Args:
            sample_keys: Keys whose local ``dataset_index`` values belong to
                this loader's Dataset.

        Returns:
            Mapping from every requested key to its Dataset sample.
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
        # DataLoader retains this list, so replace its contents instead of rebinding it.
        self._indices[:] = (key.dataset_index for key in sample_keys)
        iterator = iter(self._data_loader)
        samples_by_key = {}
        for key in sample_keys:
            try:
                payload = next(iterator)
            except StopIteration as exc:
                raise ValueError("Plan-aware DataLoader exhausted before all requested samples were read.") from exc
            samples_by_key[key] = payload
        return samples_by_key

    def state_dict(self) -> dict[str, Any]:
        """Return the epoch required to seed future plan-aware workers."""
        return {
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
            "dataset_size": len(self._dataset),
            "seed": self._seed,
        }
        for name, expected_value in expected.items():
            if state.get(name) != expected_value:
                raise ValueError(
                    f"Plan-aware DataLoader checkpoint {name}={state.get(name)!r} "
                    f"does not match {expected_value!r}."
                )
        self.set_epoch(state.get("epoch"))

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
