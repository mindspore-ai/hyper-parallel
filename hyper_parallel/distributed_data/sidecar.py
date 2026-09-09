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
"""Ahead-of-fetch sidecar metadata and plan-aware local sample loading."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch.utils.data import DataLoader, Dataset, Sampler  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    DataConstructorPlan,
    SampleKey,
    SampleMetadata,
)
from hyper_parallel.distributed_data.dataset_reader import (
    _IndexedDataset,
    _IndexedPayload,
    _ReaderIndexSampler,
    _build_worker_options,
    _commit_reader_buffer,
    _identity,
    _validate_reader_checkpoint,
    _validate_reader_partition,
)


class SidecarMetadataReader:
    """Expose one deterministic reader partition without reading sample payloads."""

    VERSION = 3

    def __init__(
            self,
            metadata: Sequence[SampleMetadata],
            *,
            reader_rank: int,
            reader_idx: int,
            reader_count: int,
            seq_len: int,
            shuffle: bool,
            seed: int,
            dataset_already_sharded: bool = False,
    ) -> None:
        """Initialize a metadata-only Dataset Reader.

        Args:
            metadata: Shared or rank-local sidecar entries aligned one-to-one
                with the corresponding Dataset indices.
            reader_rank: Global rank owning this metadata reader.
            reader_idx: Position in the configured Dataset Reader rank tuple.
            reader_count: Number of metadata Dataset Readers.
            seq_len: Sequence capacity used for buffered-token accounting.
            shuffle: Whether to shuffle this metadata sequence per epoch.
            seed: Base shuffle seed.
            dataset_already_sharded: Whether this Reader already receives only
                its rank-local metadata and must not apply another stride.
        """
        if not hasattr(metadata, "__len__") or not callable(getattr(metadata, "__getitem__", None)):
            raise ValueError("metadata must support __len__ and integer __getitem__ access.")
        metadata_size = len(metadata)
        if not isinstance(metadata_size, int) or isinstance(metadata_size, bool) or metadata_size < 0:
            raise ValueError(f"metadata length must be a non-negative integer, but got {metadata_size!r}.")
        _validate_reader_partition(
            reader_rank=reader_rank,
            reader_idx=reader_idx,
            reader_count=reader_count,
            seq_len=seq_len,
            seed=seed,
            dataset_already_sharded=dataset_already_sharded,
            shuffle=shuffle,
        )
        self._metadata = metadata
        self._reader_rank = reader_rank
        self._reader_idx = reader_idx
        self._reader_count = reader_count
        self._seq_len = seq_len
        self._shuffle = shuffle
        self._seed = seed
        self._dataset_already_sharded = dataset_already_sharded
        self._epoch = 0
        self._next_ordinal = 0
        self._buffer: list[BufferedSampleMetadata] = []
        self._iterator: Iterator[int] | None = None
        self._exhausted = False
        self._error: str | None = None

    @property
    def exhausted(self) -> bool:
        """Return whether this reader has scanned its metadata partition."""
        return self._exhausted

    @property
    def buffer_size(self) -> int:
        """Return the number of uncommitted metadata candidates."""
        return len(self._buffer)

    @property
    def effective_buffer_tokens(self) -> int:
        """Return buffered tokens, capping singleton overflow at ``seq_len``."""
        return sum(min(item.metadata.pack_tokens, self._seq_len) for item in self._buffer)

    def fill(self, *, min_samples: int, min_tokens: int, max_samples: int) -> str | None:
        """Fill the planning buffer without materializing Dataset samples.

        Args:
            min_samples: Minimum buffered candidate count.
            min_tokens: Minimum effective buffered token count.
            max_samples: Hard bound on resident metadata entries.

        Returns:
            Formatted metadata access error, or ``None`` on success/exhaustion.
        """
        for name, value in (("min_samples", min_samples), ("min_tokens", min_tokens), ("max_samples", max_samples)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        if self._error is not None:
            return self._error
        try:
            while (
                    not self._exhausted
                    and len(self._buffer) < max_samples
                    and (len(self._buffer) < min_samples or self.effective_buffer_tokens < min_tokens)
            ):
                dataset_index = self._read_one_index()
                if dataset_index is None:
                    break
                metadata = self._metadata[dataset_index]
                if not isinstance(metadata, SampleMetadata):
                    raise ValueError(
                        f"metadata must contain SampleMetadata, but got {type(metadata)} at index {dataset_index}."
                    )
                self._buffer.append(BufferedSampleMetadata(
                    key=SampleKey(self._reader_rank, dataset_index),
                    metadata=metadata,
                    global_sample_position=(
                        self._reader_idx + self._next_ordinal * self._reader_count
                    ),
                ))
                self._next_ordinal += 1
        except Exception as exc:  # The collective caller propagates the same failure to every rank.
            self._error = f"Sidecar Dataset Reader rank {self._reader_rank} failed: {type(exc).__name__}: {exc}"
            return self._error
        return None

    def metadata(self) -> tuple[BufferedSampleMetadata, ...]:
        """Return the current lightweight planning candidates."""
        return tuple(self._buffer)

    def commit(self, selected_keys: set[SampleKey]) -> None:
        """Remove selected metadata only after construction and delivery succeed.

        Args:
            selected_keys: Successfully consumed sidecar sample keys.
        """
        self._buffer = _commit_reader_buffer(self._buffer, selected_keys, owner="sidecar")

    def state_dict(self) -> dict[str, Any]:
        """Return the metadata cursor and uncommitted planning buffer."""
        state = {
            **self._checkpoint_identity(),
            "epoch": self._epoch,
            "next_ordinal": self._next_ordinal,
            "exhausted": self._exhausted,
            "error": self._error,
            "buffer": self._buffer,
        }
        try:
            return copy.deepcopy(state)
        except Exception as exc:
            raise ValueError(f"Sidecar metadata buffer is not checkpointable: {exc}") from exc

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a checkpoint produced on the same sidecar reader rank.

        Args:
            state_dict: State produced by :meth:`state_dict`.
        """
        try:
            state = copy.deepcopy(dict(state_dict))
        except Exception as exc:
            raise ValueError(f"Sidecar metadata state is not copyable: {exc}") from exc
        epoch, next_ordinal, exhausted, error, buffer = _validate_reader_checkpoint(
            state, self._checkpoint_identity(), BufferedSampleMetadata, owner="Sidecar",
        )
        self._epoch = epoch
        self._next_ordinal = next_ordinal
        self._exhausted = exhausted
        self._error = error
        self._buffer = buffer
        self._iterator = None

    def _checkpoint_identity(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "reader_rank": self._reader_rank,
            "reader_idx": self._reader_idx,
            "reader_count": self._reader_count,
            "dataset_already_sharded": self._dataset_already_sharded,
            "metadata_size": len(self._metadata),
        }

    def set_epoch(self, epoch: int) -> None:
        """Reset this metadata partition for a deterministic epoch.

        Args:
            epoch: Non-negative epoch used in the shuffle seed.
        """
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative integer, but got {epoch!r}.")
        if self._buffer and not self._exhausted:
            raise ValueError("Cannot change epoch while an active sidecar metadata buffer is non-empty.")
        self._buffer.clear()
        self._epoch = epoch
        self._next_ordinal = 0
        self._exhausted = False
        self._error = None
        self._iterator = None

    def _read_one_index(self) -> int | None:
        if self._iterator is None:
            self._iterator = self._build_iterator()
        try:
            return next(self._iterator)
        except StopIteration:
            self._exhausted = True
            return None

    def _build_iterator(self) -> Iterator[int]:
        return iter(_ReaderIndexSampler(
            dataset_size=len(self._metadata),
            reader_idx=0 if self._dataset_already_sharded else self._reader_idx,
            reader_count=1 if self._dataset_already_sharded else self._reader_count,
            start_ordinal=self._next_ordinal,
            shuffle=self._shuffle,
            seed=self._seed,
            epoch=self._epoch,
        ))


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
    """Use DataLoader workers to fetch only requested sidecar samples."""

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
            dataset: Mapping-style Dataset aligned with the sidecar sequence.
            num_workers: Native DataLoader worker count.
            pin_memory: Whether workers pin returned sample memory.
            prefetch_factor: Samples prefetched by each worker.
            persistent_workers: Whether workers persist for the loader lifetime.
            seed: Base worker seed.
            dataloader_kwargs: Additional validated DataLoader execution options.
        """
        getitem = getattr(type(dataset), "__getitem__", None)
        if not callable(getitem) or getitem is Dataset.__getitem__ or not hasattr(dataset, "__len__"):
            raise ValueError("Sidecar planned reads require a mapping-style Dataset with __len__ and __getitem__.")
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

    def fetch(self, plan: DataConstructorPlan) -> dict[SampleKey, Any]:
        """Fetch constructor-assigned Dataset indices in deterministic plan order.

        Args:
            plan: Plan for this rank's Data Constructor.

        Returns:
            Mapping from planned sample keys to materialized payloads.
        """
        if not isinstance(plan, DataConstructorPlan):
            raise ValueError(f"plan must be DataConstructorPlan, but got {type(plan)}.")
        return self.fetch_keys(plan.sample_keys)

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
                f"Sidecar plan references Dataset indices outside [0, {dataset_size}): {invalid_indices}."
            )
        self._sampler.replace(tuple(key.dataset_index for key in sample_keys))
        iterator = iter(self._data_loader)
        payloads = {}
        for expected_key in sample_keys:
            try:
                indexed_payload = next(iterator)
            except StopIteration as exc:
                raise ValueError("Plan-aware DataLoader exhausted before all requested samples were read.") from exc
            if not isinstance(indexed_payload, _IndexedPayload):
                raise ValueError(f"Plan-aware DataLoader returned invalid payload type {type(indexed_payload)}.")
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


__all__ = ["PlannedSampleLoader", "SidecarMetadataReader"]
