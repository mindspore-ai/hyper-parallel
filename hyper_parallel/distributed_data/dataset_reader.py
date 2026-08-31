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
"""Rank-local Dataset Reader with a checkpointable read-ahead sample buffer."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Mapping, Sized
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch  # pylint: disable=forbidden-backend-import
from torch.utils.data import DataLoader, Dataset, Sampler  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.schema import BufferedSampleMetadata, SampleKey, SampleMetadata

_INTERNAL_DATALOADER_OPTIONS = frozenset({
    "batch_sampler",
    "batch_size",
    "collate_fn",
    "dataset",
    "drop_last",
    "generator",
    "num_workers",
    "persistent_workers",
    "pin_memory",
    "prefetch_factor",
    "sampler",
    "shuffle",
})


class _IndexedPayload(NamedTuple):
    """Retain a Dataset index while allowing recursive pin-memory traversal."""

    dataset_index: int
    payload: Any


@dataclass(frozen=True)
class _BufferedSample:
    key: SampleKey
    metadata: SampleMetadata
    payload: Any


def _identity(value: Any) -> Any:
    return value


def _build_worker_options(
        *,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int | None,
        persistent_workers: bool,
        generator: torch.Generator,
        dataloader_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge additional DataLoader options without exposing owned semantics."""
    try:
        options = dict({} if dataloader_kwargs is None else dataloader_kwargs)
    except Exception as exc:
        raise ValueError(f"dataloader_kwargs cannot be copied: {exc}") from exc
    conflicting_options = sorted(set(options) & _INTERNAL_DATALOADER_OPTIONS)
    if conflicting_options:
        raise ValueError(f"dataloader_kwargs contains internally managed options {conflicting_options}.")
    options.update({
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "generator": generator,
    })
    if num_workers > 0 and prefetch_factor is not None:
        options["prefetch_factor"] = prefetch_factor
    return options


class _IndexedDataset(Dataset):
    """Attach the mapping-Dataset index used to materialize each payload."""

    def __init__(self, dataset: Any) -> None:
        """Store the user mapping Dataset."""
        self._dataset = dataset

    def __len__(self) -> int:
        """Return the wrapped Dataset length."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> _IndexedPayload:
        """Materialize one sample while retaining its Dataset index."""
        return _IndexedPayload(index, self._dataset[index])


class _ReaderIndexSampler(Sampler[int]):
    """Deterministically shard one global mapping-Dataset order across readers."""

    def __init__(
            self,
            *,
            dataset_size: int,
            reader_idx: int,
            reader_count: int,
            start_ordinal: int,
            shuffle: bool,
            seed: int,
            epoch: int,
    ) -> None:
        """Store deterministic global-order slicing options."""
        self._dataset_size = dataset_size
        self._reader_idx = reader_idx
        self._reader_count = reader_count
        self._start_ordinal = start_ordinal
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = epoch

    def set_position(self, *, start_ordinal: int, epoch: int) -> None:
        """Update the deterministic cursor before rebuilding an iterator."""
        self._start_ordinal = start_ordinal
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield this Dataset Reader's strided indices."""
        if self._shuffle:
            generator = torch.Generator().manual_seed(self._seed + self._epoch)
            global_indices = torch.randperm(self._dataset_size, generator=generator).tolist()
        else:
            global_indices = range(self._dataset_size)
        reader_indices = global_indices[self._reader_idx::self._reader_count]
        return iter(reader_indices[self._start_ordinal:])

    def __len__(self) -> int:
        """Return remaining indices in this reader's partition."""
        reader_size = max(0, self._dataset_size - self._reader_idx + self._reader_count - 1)
        reader_size //= self._reader_count
        return max(0, reader_size - self._start_ordinal)


class DatasetReader:
    """Read one mapping-Dataset partition and retain unplanned sample payloads.

    The Dataset Reader deliberately receives ``batch_size=None`` and an identity
    collator. User packing and collation happen only after the global plan has
    routed individual samples to a Data Constructor.
    """

    VERSION = 1

    def __init__(
            self,
            dataset: Any,
            metadata_fn: Callable[[Any], SampleMetadata],
            *,
            reader_rank: int,
            reader_idx: int,
            reader_count: int,
            seq_len: int,
            shuffle: bool,
            seed: int,
            num_workers: int,
            pin_memory: bool,
            prefetch_factor: int | None,
            persistent_workers: bool,
            dataloader_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a Dataset Reader.

        Args:
            dataset: Mapping-style Dataset materialized by this reader rank.
            metadata_fn: Callback deriving lightweight metadata from one sample.
            reader_rank: Global rank owning this Dataset Reader.
            reader_idx: Position in the configured Dataset Reader rank tuple.
            reader_count: Number of parallel Dataset Readers.
            seq_len: Sequence capacity used for read-ahead token accounting.
            shuffle: Whether to shuffle the global mapping-Dataset order.
            seed: Base shuffle and DataLoader worker seed.
            num_workers: PyTorch DataLoader worker count.
            pin_memory: Whether workers pin returned sample memory.
            prefetch_factor: Samples prefetched by each worker.
            persistent_workers: Whether DataLoader workers persist.
            dataloader_kwargs: Additional validated DataLoader execution options.
        """
        self._validate_dataset(dataset)
        if not callable(metadata_fn):
            raise ValueError("metadata_fn must be callable.")
        for name, value in (
                ("reader_rank", reader_rank),
                ("reader_idx", reader_idx),
                ("reader_count", reader_count),
                ("seq_len", seq_len),
                ("seed", seed),
                ("num_workers", num_workers),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, but got {value!r}.")
        if reader_count < 1 or reader_idx >= reader_count or seq_len < 1:
            raise ValueError("reader_count and seq_len must be positive, and reader_idx must be in range.")
        if prefetch_factor is not None and (
                not isinstance(prefetch_factor, int) or isinstance(prefetch_factor, bool) or prefetch_factor < 1
        ):
            raise ValueError("prefetch_factor must be a positive integer or None.")
        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0.")
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers=True requires num_workers > 0.")
        self._dataset = dataset
        self._metadata_fn = metadata_fn
        self._reader_rank = reader_rank
        self._reader_idx = reader_idx
        self._reader_count = reader_count
        self._seq_len = seq_len
        self._shuffle = shuffle
        self._seed = seed
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._prefetch_factor = prefetch_factor
        self._persistent_workers = persistent_workers
        self._epoch = 0
        self._next_ordinal = 0
        self._buffer: list[_BufferedSample] = []
        self._iterator: Iterator[_IndexedPayload] | None = None
        self._exhausted = False
        self._error: str | None = None
        self._sampler = _ReaderIndexSampler(
            dataset_size=len(self._dataset),
            reader_idx=self._reader_idx,
            reader_count=self._reader_count,
            start_ordinal=self._next_ordinal,
            shuffle=self._shuffle,
            seed=self._seed,
            epoch=self._epoch,
        )
        self._worker_generator = torch.Generator().manual_seed(self._seed + self._epoch)
        worker_options = _build_worker_options(
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            prefetch_factor=self._prefetch_factor,
            persistent_workers=self._persistent_workers,
            generator=self._worker_generator,
            dataloader_kwargs=dataloader_kwargs,
        )
        self._data_loader = DataLoader(
            _IndexedDataset(self._dataset),
            batch_size=None,
            sampler=self._sampler,
            collate_fn=_identity,
            **worker_options,
        )

    @staticmethod
    def _validate_dataset(dataset: Any) -> None:
        if not callable(getattr(dataset, "__getitem__", None)) or not isinstance(dataset, Sized):
            raise ValueError(
                "Dataset Reader currently requires a mapping-style Dataset with __len__ and __getitem__. "
                "Wrap multiple sources in ConcatDataset/BlendableDataset or a custom mapping Dataset."
            )
        dataset_size = len(dataset)
        if not isinstance(dataset_size, int) or isinstance(dataset_size, bool) or dataset_size < 0:
            raise ValueError(f"Dataset length must be a non-negative integer, but got {dataset_size!r}.")

    @property
    def exhausted(self) -> bool:
        """Return whether this reader has reached the end of its partition."""
        return self._exhausted

    @property
    def buffer_size(self) -> int:
        """Return number of materialized, uncommitted samples."""
        return len(self._buffer)

    @property
    def effective_buffer_tokens(self) -> int:
        """Return read-ahead tokens, capping singleton overflow at seq_len."""
        return sum(min(item.metadata.pack_tokens, self._seq_len) for item in self._buffer)

    def fill(self, *, min_samples: int, min_tokens: int, max_samples: int) -> str | None:
        """Read samples until both planning targets are satisfied.

        Args:
            min_samples: Minimum buffered candidate count.
            min_tokens: Minimum effective buffered token count.
            max_samples: Hard bound on resident payload count.

        Returns:
            Formatted callback/read error, or ``None`` on success/exhaustion.
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
                indexed_payload = self._read_one()
                if indexed_payload is None:
                    break
                metadata = self._metadata_fn(indexed_payload.payload)
                if not isinstance(metadata, SampleMetadata):
                    raise ValueError(
                        f"metadata_fn must return SampleMetadata, but got {type(metadata)} "
                        f"for Dataset index {indexed_payload.dataset_index}."
                    )
                self._buffer.append(_BufferedSample(
                    key=SampleKey(self._reader_rank, indexed_payload.dataset_index),
                    metadata=metadata,
                    payload=indexed_payload.payload,
                ))
                self._next_ordinal += 1
        except Exception as exc:  # The collective caller propagates the same failure to every rank.
            self._error = f"Dataset Reader rank {self._reader_rank} failed: {type(exc).__name__}: {exc}"
            return self._error
        return None

    def metadata(self) -> tuple[BufferedSampleMetadata, ...]:
        """Return lightweight planner candidates without payloads."""
        return tuple(BufferedSampleMetadata(item.key, item.metadata) for item in self._buffer)

    def selected_payloads(self, selected_keys: set[SampleKey]) -> tuple[tuple[SampleKey, Any], ...]:
        """Return selected payloads without removing them from the reader buffer."""
        selected = tuple((item.key, item.payload) for item in self._buffer if item.key in selected_keys)
        selected_key_set = {key for key, _ in selected}
        missing = selected_keys - selected_key_set
        if missing:
            raise ValueError(f"Dataset Reader does not contain planned sample keys {sorted(missing)}.")
        return selected

    def commit(self, selected_keys: set[SampleKey]) -> None:
        """Remove samples only after every Data Constructor succeeds."""
        existing_keys = {item.key for item in self._buffer}
        missing = selected_keys - existing_keys
        if missing:
            raise ValueError(f"Cannot commit missing Dataset Reader sample keys {sorted(missing)}.")
        self._buffer = [item for item in self._buffer if item.key not in selected_keys]

    def state_dict(self) -> dict[str, Any]:
        """Return rank-local state at a completed batch boundary.

        Note:
            Buffered transformed payloads replay exactly. Future mapping
            Dataset indices replay exactly when ``Dataset.__getitem__`` is
            deterministic for a given index and epoch.
        """
        state = {
            "version": self.VERSION,
            "reader_rank": self._reader_rank,
            "reader_idx": self._reader_idx,
            "reader_count": self._reader_count,
            "dataset_size": len(self._dataset),
            "epoch": self._epoch,
            "next_ordinal": self._next_ordinal,
            "exhausted": self._exhausted,
            "error": self._error,
            # Correctness-first checkpointing retains transformed read-ahead
            # payloads. A compact index-replay format can be layered on later.
            "buffer": self._buffer,
        }
        try:
            return copy.deepcopy(state)
        except Exception as exc:
            raise ValueError(f"Dataset Reader buffer is not checkpointable: {exc}") from exc

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the reader cursor and materialized read-ahead payloads.

        Args:
            state_dict: State produced by :meth:`state_dict` on the same rank.
        """
        try:
            state = copy.deepcopy(dict(state_dict))
        except Exception as exc:
            raise ValueError(f"Dataset Reader state is not copyable: {exc}") from exc
        expected = {
            "version": self.VERSION,
            "reader_rank": self._reader_rank,
            "reader_idx": self._reader_idx,
            "reader_count": self._reader_count,
            "dataset_size": len(self._dataset),
        }
        for name, expected_value in expected.items():
            if state.get(name) != expected_value:
                raise ValueError(
                    f"Dataset Reader checkpoint {name}={state.get(name)!r} does not match {expected_value!r}."
                )
        epoch = state.get("epoch")
        next_ordinal = state.get("next_ordinal")
        exhausted = state.get("exhausted")
        error = state.get("error")
        buffer = state.get("buffer")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (epoch, next_ordinal)):
            raise ValueError("Dataset Reader epoch and next_ordinal must be non-negative integers.")
        if not isinstance(exhausted, bool) or not isinstance(buffer, list) or any(
                not isinstance(item, _BufferedSample) for item in buffer
        ):
            raise ValueError("Dataset Reader checkpoint contains invalid exhausted or buffer state.")
        if error is not None and (not isinstance(error, str) or not error):
            raise ValueError("Dataset Reader checkpoint contains an invalid error state.")
        keys = [item.key for item in buffer]
        if len(keys) != len(set(keys)):
            raise ValueError("Dataset Reader checkpoint buffer contains duplicate SampleKey values.")
        self._epoch = epoch
        self._next_ordinal = next_ordinal
        self._exhausted = exhausted
        self._error = error
        self._buffer = buffer
        self._iterator = None

    def set_epoch(self, epoch: int) -> None:
        """Reset this reader partition to a deterministic new epoch.

        Args:
            epoch: Non-negative epoch used in the shuffle seed.
        """
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative integer, but got {epoch!r}.")
        if self._buffer and not self._exhausted:
            raise ValueError("Cannot change epoch while an active Dataset Reader buffer is non-empty.")
        # A synchronized STOP means the remaining read-ahead payloads form the
        # deliberately dropped global tail for the completed epoch.
        self._buffer.clear()
        self._epoch = epoch
        self._next_ordinal = 0
        self._exhausted = False
        self._error = None
        self._iterator = None

    def _read_one(self) -> _IndexedPayload | None:
        if self._iterator is None:
            self._iterator = self._build_iterator()
        try:
            indexed_payload = next(self._iterator)
        except StopIteration:
            self._exhausted = True
            return None
        if not isinstance(indexed_payload, _IndexedPayload):
            raise ValueError(f"Internal Dataset Reader expected _IndexedPayload, but got {type(indexed_payload)}.")
        return indexed_payload

    def _build_iterator(self) -> Iterator[_IndexedPayload]:
        self._sampler.set_position(
            start_ordinal=self._next_ordinal,
            epoch=self._epoch,
        )
        self._worker_generator.manual_seed(self._seed + self._epoch)
        return iter(self._data_loader)


__all__ = ["DatasetReader"]
