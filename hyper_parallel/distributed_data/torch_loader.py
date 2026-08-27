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
"""PyTorch DataLoader backend for local distributed-data preparation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Callable

from hyper_parallel.distributed_data.data_construct import LoadedSample
from hyper_parallel.distributed_data.schema import BatchPlan, SampleMeta
from hyper_parallel.platform import get_platform

platform = get_platform()


def _identity_collate(samples: list[Any]) -> list[Any]:
    return samples


def _single_sample_collate(samples: list[Any]) -> Any:
    if len(samples) != 1:
        raise ValueError(f"Unit DataLoader task expected one sample, but got {len(samples)}.")
    return samples[0]


def _validate_loader_options(
    dataset: Any,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    online_metadata: bool,
) -> None:
    if not hasattr(dataset, "__getitem__"):
        raise ValueError("TorchLocalDataLoader requires a map-style dataset implementing __getitem__.")
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError(f"num_workers must be a non-negative integer, but got {num_workers!r}.")
    if not isinstance(prefetch_factor, int) or isinstance(prefetch_factor, bool) or prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be a positive integer, but got {prefetch_factor!r}.")
    if not isinstance(pin_memory, bool):
        raise ValueError(f"pin_memory must be a boolean, but got {pin_memory!r}.")
    if not isinstance(online_metadata, bool):
        raise ValueError(f"online_metadata must be a boolean, but got {online_metadata!r}.")
    if online_metadata and pin_memory:
        raise ValueError("Online candidate samples must be pinned only after planning and redistribution.")


def _validate_loader_callbacks(
    metadata_fn: Callable[[Any, int], SampleMeta] | None,
    collate_fn: Callable[[list[Any]], Any] | None,
    worker_init_fn: Callable[[int], None] | None,
    online_metadata: bool,
) -> None:
    if online_metadata and metadata_fn is None:
        raise ValueError("Online TorchLocalDataLoader requires metadata_fn.")
    callbacks = (
        ("metadata_fn", metadata_fn),
        ("collate_fn", collate_fn),
        ("worker_init_fn", worker_init_fn),
    )
    for name, callback in callbacks:
        if callback is not None and not callable(callback):
            raise ValueError(f"{name} must be callable or None.")


class _MutableBatchSampler:
    """Expose one finite request window to a reusable DataLoader."""

    def __init__(self) -> None:
        """Initialize an empty request window."""
        self._batches: tuple[tuple[int | str, ...], ...] = ()

    def __iter__(self) -> Iterator[tuple[int | str, ...]]:
        """Iterate over the current immutable request window."""
        return iter(self._batches)

    def __len__(self) -> int:
        """Return the current request-window batch count."""
        return len(self._batches)

    def replace(self, batches: Sequence[Sequence[int | str]]) -> None:
        """Replace the request window before creating the next loader iterator.

        Args:
            batches: Unit-sized sample ID requests for the next iterator.
        """
        self._batches = tuple(tuple(batch) for batch in batches)


class _OnlineMetadataDataset:
    """Read one raw sample and derive metadata inside a DataLoader worker."""

    def __init__(self, dataset: Any, metadata_fn: Callable[[Any, int], SampleMeta]) -> None:
        """Initialize the wrapper around a shared map-style dataset."""
        self._dataset = dataset
        self._metadata_fn = metadata_fn

    def __len__(self) -> int:
        """Return the wrapped dataset size."""
        return len(self._dataset)

    def __getitem__(self, sample_id: int) -> LoadedSample:
        """Return one raw sample together with its validated online metadata."""
        sample = self._dataset[sample_id]
        metadata = self._metadata_fn(sample, sample_id)
        if not isinstance(metadata, SampleMeta):
            raise ValueError(f"metadata_fn must return SampleMeta, but got {type(metadata)}.")
        if metadata.sample_id != sample_id:
            raise ValueError(
                f"metadata_fn must preserve sample_id {sample_id!r}, but returned {metadata.sample_id!r}."
            )
        return LoadedSample(metadata, sample)


class TorchLocalDataLoader:
    """Load unit sample tasks in workers and collate planned batches in the caller."""

    def __init__(
        self,
        dataset: Any,
        *,
        metadata_fn: Callable[[Any, int], SampleMeta] | None,
        collate_fn: Callable[[list[Any]], Any] | None,
        num_workers: int,
        prefetch_factor: int,
        pin_memory: bool,
        worker_init_fn: Callable[[int], None] | None,
        online_metadata: bool,
    ) -> None:
        """Initialize a reusable DataLoader with replaceable finite sample requests."""
        _validate_loader_options(dataset, num_workers, prefetch_factor, pin_memory, online_metadata)
        _validate_loader_callbacks(metadata_fn, collate_fn, worker_init_fn, online_metadata)

        self._online_metadata = online_metadata
        self._collate_fn = collate_fn or _identity_collate
        self._batch_sampler = _MutableBatchSampler()
        loader_dataset = _OnlineMetadataDataset(dataset, metadata_fn) if online_metadata else dataset
        loader_kwargs = {
            "batch_sampler": self._batch_sampler,
            "collate_fn": _single_sample_collate,
            "num_workers": num_workers,
            # Sidecar collation happens after unit worker results are collected.
            # The distributed dataset pins that final batch on its pin thread.
            "pin_memory": False,
            "worker_init_fn": worker_init_fn,
        }
        if num_workers > 0:
            loader_kwargs.update(
                {
                    "multiprocessing_context": "spawn",
                    "persistent_workers": True,
                    "prefetch_factor": prefetch_factor,
                }
            )
        self._loader: Any = platform.create_data_loader(loader_dataset, **loader_kwargs)
        self._iterator: Iterator[Any] | None = None
        self._active_plan_id: str | None = None
        self._next_micro_batch_index = 0
        self._closed = False

    @property
    def pin_memory(self) -> bool:
        """Return whether final sidecar batches are already pinned."""
        return False

    def fetch(self, plan: BatchPlan, data_rank: int, micro_batch_index: int) -> Any:
        """Return one planned sidecar batch from the reusable DataLoader.

        Args:
            plan: Deterministic batch plan for the active optimizer step.
            data_rank: Data-owner coordinate whose samples should be loaded.
            micro_batch_index: Planned microbatch index to consume next.

        Returns:
            Final caller-collated microbatch.
        """
        if self._online_metadata:
            raise ValueError("Online TorchLocalDataLoader cannot fetch sidecar-planned batches.")
        self._raise_if_closed()
        if self._active_plan_id is None:
            if micro_batch_index != plan.micro_batch_start:
                raise ValueError(
                    f"Expected first microbatch {plan.micro_batch_start}, but got {micro_batch_index}."
                )
            sample_tasks = tuple(
                (sample.meta.sample_id,)
                for index in range(plan.micro_batch_start, plan.micro_batch_start + plan.micro_batch_num)
                for sample in plan.samples_for(data_rank, index)
            )
            self._batch_sampler.replace(sample_tasks)
            self._iterator = iter(self._loader)
            self._active_plan_id = plan.plan_id
            self._next_micro_batch_index = plan.micro_batch_start
        if plan.plan_id != self._active_plan_id:
            raise ValueError("Cannot replace an active DataLoader plan before all of its microbatches are consumed.")
        if micro_batch_index != self._next_micro_batch_index:
            raise ValueError(f"Expected microbatch {self._next_micro_batch_index}, but got {micro_batch_index}.")

        planned_samples = plan.samples_for(data_rank, micro_batch_index)
        samples = [self._next_batch() for _ in planned_samples]
        batch = self._collate_fn(samples)
        self._next_micro_batch_index += 1
        if self._next_micro_batch_index == plan.micro_batch_start + plan.micro_batch_num:
            self._active_plan_id = None
            self._iterator = None
        return batch

    def load_online(self, sample_ids: Sequence[int | str]) -> tuple[LoadedSample, ...]:
        """Load one candidate microbatch and derive metadata in DataLoader workers.

        Args:
            sample_ids: Candidate sample IDs for one online-planning microbatch.

        Returns:
            Loaded samples paired with validated online metadata.
        """
        if not self._online_metadata:
            raise ValueError("Sidecar TorchLocalDataLoader cannot derive online metadata.")
        self._raise_if_closed()
        if not sample_ids:
            return ()

        # Unit-sized loader batches let multiple workers participate without
        # reading candidates from later online-planning microbatches.
        self._batch_sampler.replace(tuple((sample_id,) for sample_id in sample_ids))
        self._iterator = iter(self._loader)
        loaded_samples = []
        for sample_id in sample_ids:
            loaded_sample = self._next_batch()
            if not isinstance(loaded_sample, LoadedSample):
                raise RuntimeError("Online DataLoader returned an invalid LoadedSample.")
            if loaded_sample.metadata.sample_id != sample_id:
                raise RuntimeError(
                    f"Online DataLoader expected sample_id {sample_id!r}, "
                    f"but returned {loaded_sample.metadata.sample_id!r}."
                )
            loaded_samples.append(loaded_sample)
        self._iterator = None
        return tuple(loaded_samples)

    def close(self) -> None:
        """Release the DataLoader iterator and any persistent worker processes."""
        if self._closed:
            return
        self._iterator = None
        self._loader = None
        self._active_plan_id = None
        self._closed = True

    def _next_batch(self) -> Any:
        if self._iterator is None:
            raise RuntimeError("TorchLocalDataLoader has no active request window.")
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise RuntimeError("PyTorch DataLoader exhausted before completing the request window.") from exc

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise ValueError("Cannot use a closed TorchLocalDataLoader.")
