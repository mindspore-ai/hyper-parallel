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
"""Generic adapter for externally selected, complete distributed steps."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol

from hyper_parallel.distributed_data.schema import BufferedSampleMetadata, SampleKey, SampleMetadata


class ExternalStepSource(Protocol):
    """Checkpointable iterator that emits one complete local step at a time."""

    def __iter__(self) -> Iterator[Sequence[Sequence[Any]]]:
        """Return an iterator over complete local steps."""

    def state_dict(self) -> dict[str, Any]:
        """Return the source state after the most recently emitted step."""

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the source state represented by ``state``.

        Args:
            state: Source cursor returned by ``state_dict``.
        """

    def set_epoch(self, epoch: int) -> None:
        """Reset the source to the requested epoch.

        Args:
            epoch: Epoch to load from the source.
        """


class ExternalStepAdapter:
    """Add HP metadata, payload, checkpoint, and commit semantics to a source.

    The source emits a sequence of ``local_batch_size`` raw-sample bins. HP
    freezes those bins as the reference plan, and may then move samples across
    data ranks. ``pack_fn`` and ``collate_fn`` are used only to materialize the
    final constructor batch; the source never needs to know HP's placement
    policy.
    """

    def __init__(
            self,
            source: ExternalStepSource,
            *,
            reader_rank: int,
            local_batch_size: int,
            seq_len: int,
            metadata_fn: Callable[[Any], SampleMetadata],
            pack_fn: Callable[[Sequence[Any], int], Any],
            collate_fn: Callable[[Sequence[Any]], Any],
    ) -> None:
        """Validate and store one rank-local external step source."""
        if not callable(getattr(source, "__iter__", None)):
            raise ValueError("external_step_source must be iterable.")
        for name in ("state_dict", "load_state_dict", "set_epoch"):
            if not callable(getattr(source, name, None)):
                raise ValueError(f"external_step_source is missing {name}().")
        if not isinstance(reader_rank, int) or isinstance(reader_rank, bool) or reader_rank < 0:
            raise ValueError("reader_rank must be a non-negative integer.")
        if not isinstance(local_batch_size, int) or isinstance(local_batch_size, bool) or local_batch_size < 1:
            raise ValueError("local_batch_size must be a positive integer.")
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
            raise ValueError("seq_len must be a positive integer.")
        for name, callback in (("metadata_fn", metadata_fn), ("pack_fn", pack_fn), ("collate_fn", collate_fn)):
            if not callable(callback):
                raise ValueError(f"{name} must be callable.")
        self._source = source
        self._iterator = iter(source)
        self._reader_rank = reader_rank
        self._local_batch_size = local_batch_size
        self._seq_len = seq_len
        self._metadata_fn = metadata_fn
        self._pack_fn = pack_fn
        self._collate_fn = collate_fn
        self._epoch = 0
        self._step = 0
        self._sample_ordinal = 0
        self._exhausted = False
        self._buffer: tuple[BufferedSampleMetadata, ...] = ()
        self._payloads: dict[SampleKey, Any] = {}
        self._original_metadatas: tuple[tuple[BufferedSampleMetadata, ...], ...] = ()
        self._original_step_samples: tuple[tuple[Any, ...], ...] = ()

    @property
    def exhausted(self) -> bool:
        """Return whether the external source has reached end of epoch."""
        return self._exhausted

    @property
    def buffer_size(self) -> int:
        """Return the number of pending raw samples."""
        return len(self._buffer)

    @property
    def batch_position(self) -> int:
        """Return the number of committed external steps."""
        return self._step

    @property
    def original_metadatas(self) -> tuple[tuple[BufferedSampleMetadata, ...], ...]:
        """Return metadata grouped by the source step's original packs."""
        return self._original_metadatas

    def prepare_next_step(self) -> None:
        """Read and stage exactly one source step without committing it."""
        if self._buffer or self._exhausted:
            return
        try:
            local_step = next(self._iterator)
        except StopIteration:
            self._exhausted = True
            return
        if not isinstance(local_step, (list, tuple)) or len(local_step) != self._local_batch_size:
            emitted = len(local_step) if isinstance(local_step, (list, tuple)) else type(local_step)
            raise ValueError(
                f"External source emitted {emitted} local packs, expected {self._local_batch_size}."
            )

        metadata = []
        original_metadatas = []
        original_step_samples = []
        for packed_samples in local_step:
            if not isinstance(packed_samples, (list, tuple)) or not packed_samples:
                raise ValueError("External source emitted an empty or non-sequence local pack.")
            original_samples = tuple(packed_samples)
            original_step_samples.append(original_samples)
            bin_metadata = []
            for sample in original_samples:
                key = SampleKey(self._reader_rank, self._sample_ordinal)
                sample_metadata = self._metadata_fn(sample)
                if not isinstance(sample_metadata, SampleMetadata):
                    raise ValueError(
                        "external_step_source metadata_fn must return SampleMetadata, "
                        f"but got {type(sample_metadata)}."
                    )
                entry = BufferedSampleMetadata(key, sample_metadata, self._sample_ordinal)
                metadata.append(entry)
                bin_metadata.append(entry)
                self._payloads[key] = sample
                self._sample_ordinal += 1
            original_metadatas.append(tuple(bin_metadata))
        self._buffer = tuple(metadata)
        self._original_metadatas = tuple(original_metadatas)
        self._original_step_samples = tuple(original_step_samples)

    def metadata(self) -> tuple[BufferedSampleMetadata, ...]:
        """Return metadata for the pending source step."""
        return self._buffer

    def selected_payloads(self, selected_keys: set[SampleKey]) -> tuple[tuple[SampleKey, Any], ...]:
        """Return selected payloads without rereading the external source.

        Args:
            selected_keys: Routing keys requested from the pending step.
        """
        if not selected_keys.issubset(self._payloads):
            missing = selected_keys - set(self._payloads)
            raise ValueError(f"External source is missing selected keys: {sorted(missing)}")
        return tuple((entry.key, self._payloads[entry.key]) for entry in self._buffer if entry.key in selected_keys)

    def commit(self, selected_keys: set[SampleKey]) -> None:
        """Commit exactly the complete pending source step.

        Args:
            selected_keys: Complete set of keys consumed from this Reader.
        """
        expected = {entry.key for entry in self._buffer}
        if selected_keys != expected:
            raise ValueError("External source commit must consume every selected sample exactly once.")
        self._payloads.clear()
        self._buffer = ()
        self._original_metadatas = ()
        self._original_step_samples = ()
        self._step += 1

    def canonical_plan_matches(self, constructor_plan: Any, data_rank: int) -> bool:
        """Return whether HP kept this source step on its original rank/bins.

        Args:
            constructor_plan: Planned bins assigned to the local Constructor.
            data_rank: Data-parallel rank of the local Constructor.
        """
        if getattr(constructor_plan, "target_data_rank", data_rank) != int(data_rank):
            return False
        expected = tuple(tuple(item.key for item in packing_bin) for packing_bin in self._original_metadatas)
        bins = getattr(constructor_plan, "bins", constructor_plan)
        actual = tuple(
            tuple(getattr(item, "key", item) for item in getattr(packing_bin, "sample_keys", packing_bin))
            for packing_bin in bins
        )
        return actual == expected

    def canonical_batch(self) -> Any:
        """Materialize the unchanged source bins through HP's callbacks."""
        if not self._original_step_samples:
            raise RuntimeError("No canonical external source step is buffered.")
        packed = [self._pack_fn(samples, self._seq_len) for samples in self._original_step_samples]
        return self._collate_fn(packed)

    def state_dict(self) -> dict[str, Any]:
        """Save the source's speculative cursor and pending step."""
        return copy.deepcopy({
            "version": 1,
            "reader_rank": self._reader_rank,
            "epoch": self._epoch,
            "step": self._step,
            "sample_ordinal": self._sample_ordinal,
            "exhausted": self._exhausted,
            "source": self._source.state_dict(),
            "buffer": self._buffer,
            "payloads": self._payloads,
            # Preserve serialized keys for existing adapter checkpoints.
            "reference_bins": self._original_metadatas,
            "canonical_step": self._original_step_samples,
        })

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore source state and any already staged step.

        Args:
            state: Adapter checkpoint returned by ``state_dict``.
        """
        state = copy.deepcopy(state)
        if state.get("version") != 1 or state.get("reader_rank") != self._reader_rank:
            raise ValueError("External source checkpoint identity does not match this Reader.")
        self._epoch = int(state["epoch"])
        self._source.load_state_dict(state["source"])
        self._iterator = iter(self._source)
        self._step = int(state["step"])
        self._sample_ordinal = int(state["sample_ordinal"])
        self._exhausted = bool(state["exhausted"])
        self._buffer = state.get("buffer", ())
        self._payloads = state.get("payloads", {})
        self._original_metadatas = state.get("reference_bins", ())
        self._original_step_samples = state.get("canonical_step", ())

    def set_epoch(self, epoch: int) -> None:
        """Reset the source and discard any speculative step.

        Args:
            epoch: Epoch to load from the source.
        """
        self._source.set_epoch(epoch)
        self._iterator = iter(self._source)
        self._epoch = int(epoch)
        self._step = 0
        self._sample_ordinal = 0
        self._exhausted = False
        self._buffer = ()
        self._payloads = {}
        self._original_metadatas = ()
        self._original_step_samples = ()


__all__ = ["ExternalStepAdapter", "ExternalStepSource"]
