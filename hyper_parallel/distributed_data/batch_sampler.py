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
"""Transactional Dataset reads driven by HP's native DP BatchSampler."""

from __future__ import annotations

import copy
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from hyper_parallel.data.parallel.batch_sampler import sampler_optimizer_step_progress
from hyper_parallel.distributed_data.schema import SampleKey, SampleMetadata
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader


def validate_native_sampler(sampler: Any, *, data_rank: int, dp_size: int, local_batch_size: int) -> None:
    """Validate the native HP sampler interface, DP layout, and cursor alignment.

    Args:
        sampler: Native HP batch sampler to validate.
        data_rank: Expected DP coordinate for this rank.
        dp_size: Expected number of DP replicas.
        local_batch_size: Expected samples per local forward/backward round.

    Raises:
        ValueError: If the sampler interface, DP settings, or cursor is incompatible.
    """
    for name in ("__iter__", "state_dict", "load_state_dict", "set_epoch", "enable_source_batch_resume"):
        if not callable(getattr(sampler, name, None)):
            raise ValueError(f"batch_sampler must implement HP's native sampler contract: missing {name}.")
    expected = {"dp_rank": data_rank, "dp_world_size": dp_size, "micro_batch_size": local_batch_size,
                "drop_last": True}
    for name, value in expected.items():
        if getattr(sampler, name, None) != value:
            raise ValueError(f"batch_sampler.{name} must equal {value!r}.")
    if sampler_optimizer_step_progress(sampler.state_dict()):
        raise ValueError("batch_sampler.consumed_samples must align to a complete optimizer step.")


def preserve_sample(samples: Sequence[Any], seq_len: int) -> Any:
    """Pass one native Dataset output unchanged to the original collator.

    Args:
        samples: Source samples for one native sequence bin.
        seq_len: Sequence capacity, unused when preserving a Dataset output.

    Returns:
        The sole source sample, unchanged.
    """
    del seq_len
    if len(samples) != 1:
        raise ValueError("Native BatchSampler mode must keep one Dataset output per sequence bin.")
    return samples[0]


@dataclass(frozen=True)
class _ReaderPlanInput:
    """Transfer metadata and source keys, grouped by micro-batch then original pack."""

    metadata: tuple[SampleMetadata, ...] = ()
    original_batches: tuple[tuple[tuple[SampleKey, ...], ...], ...] = ()


@dataclass
class _ReaderWindow:
    """Keep one window's data and cursor independent of speculative sampling."""

    metadata: tuple[SampleMetadata, ...]
    samples: dict[SampleKey, Any]
    sampler_state: dict[str, Any]
    original_batches: tuple[tuple[tuple[SampleKey, ...], ...], ...]
    delivered_rounds: int = 0


class BatchSamplerReader:
    """Read a bounded window of native batches within one optimizer step.

    Only the DP Constructor owns this reader. Payload loading is deferred until
    after planning when metadata is available; otherwise it precedes metadata
    extraction. The sampler's speculative cursor is never a checkpoint cursor.
    """

    def __init__(
            self,
            sampler: Any,
            *,
            reader_rank: int,
            metadata: Sequence[SampleMetadata] | None,
            metadata_fn: Callable[[Any], SampleMetadata] | None,
            sample_loader: PlannedSampleLoader | None,
            split_fn: Callable[[Any], Sequence[Any]] | None = None,
    ) -> None:
        """Store the validated native sampler and metadata/payload sources."""
        if metadata is None and (sample_loader is None or not callable(metadata_fn)):
            raise ValueError("Online BatchSampler Reader requires payload loading and metadata_fn.")
        if split_fn is not None and (not callable(split_fn) or metadata is not None):
            raise ValueError("split_fn requires online metadata and must be callable.")
        self._sampler = sampler
        self._reader_rank = reader_rank
        self._metadata = metadata
        self._metadata_fn = metadata_fn
        self._sample_loader = sample_loader
        self._split_fn = split_fn
        self._window: _ReaderWindow | None = None
        self._sampler.enable_source_batch_resume()
        self._committed_state = copy.deepcopy(sampler.state_dict())
        self._iterator = None
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """Return whether the native sampler has exhausted its epoch."""
        return self._exhausted

    @property
    def buffer_size(self) -> int:
        """Return the number of pending native sample occurrences."""
        return len(self.metadata())

    @property
    def batch_position(self) -> int:
        """Return the native global cursor before this pending round."""
        return self._committed_state["consumed_samples"]

    def prepare_next_step(self, window_micro_batch_num: int) -> None:
        """Read the window requested by the owning loader without source lookahead.

        Args:
            window_micro_batch_num: Positive number of source rounds to read,
                already clipped to the optimizer boundary by the loader.
        """
        if isinstance(window_micro_batch_num, bool) or not isinstance(window_micro_batch_num, int):
            raise ValueError("window_micro_batch_num must be a positive integer.")
        if window_micro_batch_num < 1:
            raise ValueError("window_micro_batch_num must be a positive integer.")
        if self._window is not None or self._exhausted:
            return
        self._window = self._read_window(window_micro_batch_num)
        self._exhausted = self._window is None

    def _read_window(self, window_micro_batch_num: int) -> _ReaderWindow | None:
        """Read a bounded native window without changing the active routing window."""
        # Only one read runs at a time. It must not mutate the active window
        # while that window's samples are being routed and committed.
        try:
            if self._iterator is None:
                self._iterator = iter(self._sampler)
            source_batches = self._read_window_keys(window_micro_batch_num)
            if not source_batches:
                return None
            return self._load_window(source_batches)
        except StopIteration as exc:
            # A user callback's StopIteration must not masquerade as dataset EOF.
            raise RuntimeError(f"Native BatchSampler callback raised StopIteration: {exc}") from exc

    def _load_window(self, source_batches: tuple[tuple[SampleKey, ...], ...]) -> _ReaderWindow:
        """Materialize the selected occurrences and freeze their original pack boundaries."""
        keys = tuple(key for batch in source_batches for key in batch)
        original_batches = ()
        samples = {}
        if self._metadata is None:
            samples = self._sample_loader.fetch_samples(keys)
            if self._split_fn is not None:
                samples, original_batches = self._split_payloads(samples, source_batches)
                keys = tuple(samples)
            metadata = [self._metadata_fn(samples[key]) for key in keys]
        else:
            metadata = [self._metadata[key.dataset_index] for key in keys]
        if self._split_fn is None:
            original_batches = tuple(tuple((key,) for key in batch) for batch in source_batches)
        if any(not isinstance(item, SampleMetadata) for item in metadata):
            raise ValueError("BatchSampler metadata must contain SampleMetadata entries.")
        return _ReaderWindow(
            metadata=tuple(replace(item, key=key) for key, item in zip(keys, metadata)),
            samples=samples,
            sampler_state=copy.deepcopy(self._sampler.state_dict()),
            original_batches=original_batches,
        )

    def _read_window_keys(self, window_micro_batch_num: int) -> tuple[tuple[SampleKey, ...], ...]:
        source_batches = []
        for _ in range(window_micro_batch_num):
            position = self._sampler.consumed_samples
            try:
                indices = next(self._iterator)
            except StopIteration:
                break
            source_batches.append(self._sample_keys(indices, position))
        return tuple(source_batches)

    def _split_payloads(
            self, samples: Mapping[SampleKey, Any], source_batches: tuple[tuple[SampleKey, ...], ...],
    ) -> tuple[dict[SampleKey, Any], tuple[tuple[tuple[SampleKey, ...], ...], ...]]:
        """Split records into keyed sequences while retaining their original per-round bins."""
        payloads = {}
        original_batches = []
        for batch in source_batches:
            original_bins = []
            for key in batch:
                sequences = tuple(self._split_fn(samples[key]))
                if not sequences:
                    raise ValueError("split_fn must return at least one independent sequence per Dataset output.")
                sequence_keys = tuple(replace(key, sequence_index=index) for index in range(len(sequences)))
                payloads.update(zip(sequence_keys, sequences))
                original_bins.append(sequence_keys)
            original_batches.append(tuple(original_bins))
        return payloads, tuple(original_batches)

    def close(self) -> None:
        """Release source buffers after the owning loader has drained its producer."""
        self._window = None
        self._iterator = None

    def _sample_keys(self, indices: Sequence[int], batch_position: int) -> tuple[SampleKey, ...]:
        """Bind local Dataset indices to distinct global sampling occurrences."""
        if len(indices) != self._sampler.micro_batch_size:
            raise ValueError("Native BatchSampler must yield exactly local_batch_size indices on every DP rank.")
        # A native sampler exposes local indices, not its internal permutation
        # positions. Stable DP-major occurrence slots preserve duplicates.
        start = batch_position + self._sampler.dp_rank * self._sampler.micro_batch_size
        keys = []
        for offset, index in enumerate(indices):
            if isinstance(index, bool):
                raise ValueError("Native BatchSampler indices must be non-negative integers.")
            keys.append(SampleKey(self._reader_rank, operator.index(index), start + offset))
        return tuple(keys)

    def metadata(self) -> tuple[SampleMetadata, ...]:
        """Return metadata for the undelivered occurrences in the frozen window."""
        return () if self._window is None else self._window.metadata

    def planning_input(self) -> _ReaderPlanInput:
        """Return the window's metadata and original layout without payloads."""
        if self._window is None:
            return _ReaderPlanInput()
        return _ReaderPlanInput(self._window.metadata, self._window.original_batches)

    def selected_payloads(self, selected_keys: set[SampleKey]) -> tuple[tuple[SampleKey, Any], ...]:
        """Return already-read payloads without rereading their Dataset indices.

        Args:
            selected_keys: Sample occurrences selected from this Reader's buffer.

        Returns:
            Selected key-payload pairs in buffered sample order.
        """
        return tuple(
            (item.key, self._window.samples[item.key]) for item in self.metadata() if item.key in selected_keys
        )

    def commit(self, selected_keys: set[SampleKey]) -> None:
        """Retire delivered occurrences and commit the cursor at the window end.

        Args:
            selected_keys: Source-owned occurrences delivered in this distributed round.
        """
        window = self._window
        remaining = {item.key for item in self.metadata()}
        if window is None or not selected_keys <= remaining:
            raise ValueError("Native BatchSampler commit must consume the complete pending local batch.")
        last_round = window.delivered_rounds + 1 == len(window.original_batches)
        if last_round and selected_keys != remaining:
            raise ValueError("Native BatchSampler commit must consume the complete pending local batch.")
        window.metadata = tuple(item for item in window.metadata if item.key not in selected_keys)
        for key in selected_keys:
            window.samples.pop(key, None)
        window.delivered_rounds += 1
        if last_round:
            self._committed_state = window.sampler_state
            self._window = None

    def state_dict(self) -> dict[str, Any]:
        """Snapshot the delivered native cursor, excluding speculative prefetch."""
        state = {
            "epoch": self._committed_state["epoch"],
            "sampler": self._committed_state,
            "sample_loader": self._sample_loader.state_dict() if self._sample_loader is not None else None,
        }
        return copy.deepcopy(state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the original native cursor before the next sampler yield.

        Args:
            state: Reader checkpoint containing the committed native sampler state.

        Raises:
            ValueError: If the checkpoint epoch or payload reader is incompatible.
        """
        sampler_state = state["sampler"]
        if sampler_state.get("epoch") != state.get("epoch"):
            raise ValueError("Native BatchSampler checkpoint epoch is inconsistent.")
        if (state.get("sample_loader") is None) != (self._sample_loader is None):
            raise ValueError("Native BatchSampler checkpoint payload reader ownership changed.")
        if "window" in state or sampler_optimizer_step_progress(sampler_state):
            raise ValueError("Native BatchSampler checkpoint must align to a complete optimizer step.")
        self.close()
        self._sampler.load_state_dict(sampler_state)
        if self._sample_loader is not None:
            self._sample_loader.load_state_dict(state["sample_loader"])
        self._committed_state = copy.deepcopy(self._sampler.state_dict())
        self._exhausted = False

    def set_epoch(self, epoch: int) -> None:
        """Delegate epoch changes to the native sampler without changing its policy.

        Args:
            epoch: Epoch to apply to the sampler and sample loader.
        """
        self.close()
        self._sampler.load_state_dict(self._committed_state)
        self._sampler.set_epoch(epoch)
        if self._sample_loader is not None:
            self._sample_loader.set_epoch(epoch)
        self._committed_state = copy.deepcopy(self._sampler.state_dict())
        self._exhausted = False
