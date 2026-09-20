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
"""Candidate buffering and finalization used by dynamic DataLoaders."""

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from hyper_parallel.data.batching.packing import (
    FirstFitPackingSelector,
    PackingSelector,
    SamplePacker,
)


@dataclass
class BaseCandidateBuffer(ABC):
    """Store candidates and expose the next selected sample group."""

    token_budget: int
    min_buffered_samples: int
    buffer: list[tuple[Mapping[str, Any], int]] = field(default_factory=list, init=False)
    buffer_output_indices: list[Any | None] = field(default_factory=list, init=False)
    buffer_token_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate the candidate-buffer limits."""
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")

        if self.min_buffered_samples <= 0:
            raise ValueError("min_buffered_samples must be positive")

    @abstractmethod
    def build_buffer_entry(self, model_sample: Mapping[str, Any]) -> tuple[Mapping[str, Any], int]:
        """Attach the scheduling cost used to select this sample."""
        raise NotImplementedError

    def put_item(self, model_sample: Mapping[str, Any], output_index_entry: Any | None = None) -> None:
        """Append one candidate and its optional checkpoint replay index."""
        buffered_sample = self.build_buffer_entry(model_sample)
        self.buffer.append(buffered_sample)
        self.buffer_output_indices.append(output_index_entry)
        self.buffer_token_count += buffered_sample[1]

    def is_ready_for_micro_batch(self) -> bool:
        """Return whether both configured candidate limits are reached."""
        enough_samples = len(self.buffer) >= self.min_buffered_samples
        enough_tokens = self.buffer_token_count >= self.token_budget
        is_ready = enough_samples and enough_tokens
        return is_ready

    @abstractmethod
    def get_micro_batch(self) -> list[Mapping[str, Any]]:
        """Select and remove the next sample group."""
        raise NotImplementedError

    def empty(self) -> bool:
        """Return whether no candidates remain."""
        is_empty = not self.buffer
        return is_empty


@dataclass
class TokenBudgetCandidateBuffer(BaseCandidateBuffer):
    """Select encoded samples in source order within one token budget."""

    def build_buffer_entry(self, model_sample: Mapping[str, Any]) -> tuple[Mapping[str, Any], int]:
        """Attach the encoded token length to one candidate."""
        sample_length = int(model_sample["input_ids"].shape[-1])
        if sample_length <= 0:
            raise ValueError("Dynamic batching samples must contain at least one token")

        buffered_sample = (model_sample, sample_length)
        return buffered_sample

    def get_micro_batch(self) -> list[Mapping[str, Any]]:
        """Select fitting candidates and retain the remainder."""
        if not self.buffer:
            raise ValueError("Dynamic token buffer is empty")

        selected_samples = []
        remaining_buffer = []
        remaining_output_indices = []
        selected_token_count = 0
        for (model_sample, sample_length), output_index_entry in zip(
                self.buffer,
                self.buffer_output_indices,
        ):
            sample_fits = selected_token_count == 0 or selected_token_count + sample_length <= self.token_budget
            if sample_fits:
                selected_samples.append(model_sample)
                selected_token_count += sample_length
            else:
                remaining_buffer.append((model_sample, sample_length))
                remaining_output_indices.append(output_index_entry)

        self.buffer = remaining_buffer
        self.buffer_output_indices = remaining_output_indices
        self.buffer_token_count -= selected_token_count
        return selected_samples


@dataclass
class PackingCandidateBuffer(BaseCandidateBuffer):
    """Delegate candidate-group selection to a configured packing strategy."""

    packing_selector: PackingSelector

    def build_buffer_entry(self, model_sample: Mapping[str, Any]) -> tuple[Mapping[str, Any], int]:
        """Attach selector-defined scheduling cost to one candidate."""
        sample_cost = self.packing_selector.get_sample_cost(model_sample)
        buffered_sample = (model_sample, sample_cost)
        return buffered_sample

    def get_micro_batch(self) -> list[Mapping[str, Any]]:
        """Select and remove the next candidate group."""
        if not self.buffer:
            raise ValueError("Packing candidate buffer is empty")

        candidates = []
        for model_sample, _ in self.buffer:
            candidates.append(model_sample)

        selected_indices = self.packing_selector.select_samples_to_pack(
            candidates,
            self.token_budget,
        )
        resolved_indices = self._validate_selected_indices(selected_indices)
        selected_index_set = set(resolved_indices)

        selected_samples = []
        selected_cost = 0
        for selected_index in resolved_indices:
            model_sample, sample_cost = self.buffer[selected_index]
            selected_samples.append(model_sample)
            selected_cost += sample_cost

        remaining_buffer = []
        remaining_output_indices = []
        for sample_index, (buffered_sample, output_index) in enumerate(
                zip(self.buffer, self.buffer_output_indices)
        ):
            if sample_index not in selected_index_set:
                remaining_buffer.append(buffered_sample)
                remaining_output_indices.append(output_index)

        self.buffer = remaining_buffer
        self.buffer_output_indices = remaining_output_indices
        self.buffer_token_count -= selected_cost
        return selected_samples

    def _validate_selected_indices(self, selected_indices: Sequence[int]) -> list[int]:
        """Validate selector output before mutating the candidate buffer."""
        resolved_indices = []
        for selected_index in selected_indices:
            if isinstance(selected_index, bool):
                raise TypeError("PackingSelector indices must be integers")
            try:
                resolved_index = operator.index(selected_index)
            except TypeError as exc:
                raise TypeError("PackingSelector indices must be integers") from exc

            if resolved_index < 0 or resolved_index >= len(self.buffer):
                raise IndexError("PackingSelector returned an out-of-range candidate index")
            resolved_indices.append(resolved_index)

        if not resolved_indices:
            raise ValueError("PackingSelector must select at least one candidate")
        if len(set(resolved_indices)) != len(resolved_indices):
            raise ValueError("PackingSelector must not select a candidate more than once")

        return resolved_indices


def _normalize_source_samples(source_item: Any) -> list[Mapping[str, Any]]:
    """Normalize one source output into its ordered model samples."""
    if isinstance(source_item, Mapping):
        model_samples = [source_item]
        return model_samples

    if isinstance(source_item, Sequence) and not isinstance(source_item, (str, bytes)):
        model_samples = []
        for model_sample in source_item:
            if not isinstance(model_sample, Mapping):
                raise ValueError("Every dynamic source sample must be a mapping")

            model_samples.append(model_sample)
        return model_samples

    raise ValueError("A dynamic source item must be a mapping or a sequence of mappings")


def _restore_index_buffer(
        source_dataset: Any,
        saved_buffer: Sequence[Any],
        candidate_buffer: BaseCandidateBuffer,
) -> list[tuple[Mapping[str, Any], int]]:
    """Rebuild buffered samples from their output and sample indices."""
    restored_buffer = []
    cached_output_index: Any = object()
    cached_model_samples: list[Mapping[str, Any]] = []
    for output_index_entry in saved_buffer:
        output_index, sample_index = output_index_entry
        resolved_sample_index = operator.index(sample_index)
        if resolved_sample_index < 0:
            raise ValueError("Buffered sample index must be a non-negative integer")

        if output_index != cached_output_index:
            restored_item = source_dataset.get_item(output_index)
            cached_model_samples = _normalize_source_samples(restored_item)
            cached_output_index = output_index
        try:
            model_sample = cached_model_samples[resolved_sample_index]
        except IndexError as exc:
            raise ValueError(
                f"Buffered sample index {resolved_sample_index} is out of range "
                f"for output index {output_index!r}"
            ) from exc

        buffer_entry = candidate_buffer.build_buffer_entry(model_sample)
        restored_buffer.append(buffer_entry)

    return restored_buffer


class MappingReplayDataset:
    """Attach a stable replay index to each mapping Dataset output."""

    def __init__(self, dataset: Any) -> None:
        """Store the mapping Dataset used to reconstruct buffered samples."""
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the wrapped Dataset length."""
        dataset_length = len(self.dataset)
        return dataset_length

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Return one source item and the index that can reproduce it."""
        source_item = self.get_item(index)
        indexed_source_item = (source_item, index)
        return indexed_source_item

    def get_item(self, index: Any) -> Any:
        """Rebuild one source item without advancing the source iterator."""
        resolved_index = operator.index(index)
        get_item = getattr(self.dataset, "get_item", None)
        if callable(get_item):
            source_item = get_item(resolved_index)
        else:
            source_item = self.dataset[resolved_index]

        return source_item


class IndexReplayCheckpoint:
    """Checkpoint candidates as compact mapping Dataset replay indices."""

    save_by_idx = True

    def __init__(self, source_dataset: Any) -> None:
        """Store a Dataset that emits source items with stable replay indices."""
        self.source_dataset = source_dataset

    @staticmethod
    def supports(dataset: Any) -> bool:
        """Return whether an iterable Dataset supports index replay."""
        get_item = getattr(dataset, "get_item", None)
        supports_index_replay = callable(get_item) and hasattr(dataset, "output_index_for_resume")
        return supports_index_replay

    @staticmethod
    def put_source_item(source_item: Any, candidate_buffer: BaseCandidateBuffer) -> None:
        """Append an indexed source item and retain each sample index."""
        model_samples_item, output_index = source_item
        model_samples = _normalize_source_samples(model_samples_item)
        for sample_index, model_sample in enumerate(model_samples):
            candidate_buffer.put_item(model_sample, (output_index, sample_index))

    @staticmethod
    def get_buffer_state(candidate_buffer: BaseCandidateBuffer) -> list[Any]:
        """Return replay indices for buffered model samples."""
        if len(candidate_buffer.buffer) != len(candidate_buffer.buffer_output_indices):
            raise RuntimeError("Dynamic sample and output-index buffers are inconsistent")

        if any(index is None for index in candidate_buffer.buffer_output_indices):
            raise RuntimeError("Index-buffer checkpoint requires a replay key for every buffered sample")

        return candidate_buffer.buffer_output_indices

    def restore_buffer(
            self,
            saved_buffer: Sequence[Any],
            saved_by_idx: bool,
            candidate_buffer: BaseCandidateBuffer,
    ) -> tuple[list[tuple[Mapping[str, Any], int]], list[Any]]:
        """Rebuild buffered samples from their replay indices."""
        if not saved_by_idx:
            if saved_buffer:
                raise ValueError("save_by_idx=True cannot restore a checkpoint containing full samples")

            return [], []

        restored_buffer = _restore_index_buffer(self.source_dataset, saved_buffer, candidate_buffer)
        restored_indices = list(saved_buffer)
        return restored_buffer, restored_indices


class FullSampleCheckpoint:
    """Checkpoint full candidate samples alongside the source state."""

    save_by_idx = False

    def __init__(self, dataset: Any, replay_dataset: Any | None = None) -> None:
        """Retain the source Dataset and optional mapping replay Dataset."""
        self.source_dataset = dataset
        self.replay_dataset = replay_dataset

    @staticmethod
    def put_source_item(source_item: Any, candidate_buffer: BaseCandidateBuffer) -> None:
        """Append one streaming source item to the candidate buffer."""
        model_samples = _normalize_source_samples(source_item)
        for model_sample in model_samples:
            candidate_buffer.put_item(model_sample)

    @staticmethod
    def get_buffer_state(candidate_buffer: BaseCandidateBuffer) -> list[Any]:
        """Return full samples already pulled beyond the source cursor."""
        if len(candidate_buffer.buffer) != len(candidate_buffer.buffer_output_indices):
            raise RuntimeError("Dynamic sample and output-index buffers are inconsistent")

        return candidate_buffer.buffer

    def restore_buffer(
            self,
            saved_buffer: Sequence[Any],
            saved_by_idx: bool,
            candidate_buffer: BaseCandidateBuffer,
    ) -> tuple[list[tuple[Mapping[str, Any], int]], list[Any]]:
        """Restore full samples or an earlier index-buffer checkpoint."""
        if saved_by_idx:
            if saved_buffer and self.replay_dataset is None:
                raise ValueError("An index-buffer checkpoint requires a replayable Dataset")

            restored_buffer = _restore_index_buffer(self.replay_dataset, saved_buffer, candidate_buffer)
        else:
            restored_buffer = list(saved_buffer)
        restored_indices = [None] * len(restored_buffer)
        return restored_buffer, restored_indices


class _BatchPipeline(ABC):
    """Define candidate selection and finalization for one dynamic Loader."""

    @abstractmethod
    def build_candidate_buffer(
            self,
            *,
            token_budget: int,
            min_buffered_samples: int,
    ) -> BaseCandidateBuffer:
        """Build the candidate buffer used by this pipeline."""
        raise NotImplementedError

    @abstractmethod
    def finalize_selected_samples(
            self,
            selected_samples: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]:
        """Finish selected samples before final collation."""
        raise NotImplementedError

    def encode_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply an optional final batch encoding stage."""
        return batch


class _TokenBatchPipeline(_BatchPipeline):
    """Build batches directly from fully encoded token samples."""

    def build_candidate_buffer(
            self,
            *,
            token_budget: int,
            min_buffered_samples: int,
    ) -> BaseCandidateBuffer:
        """Build the token-budget candidate buffer."""
        candidate_buffer = TokenBudgetCandidateBuffer(
            token_budget=token_budget,
            min_buffered_samples=min_buffered_samples,
        )
        return candidate_buffer

    def finalize_selected_samples(
            self,
            selected_samples: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]:
        """Return fully encoded samples for final collation."""
        return selected_samples


@dataclass(frozen=True)
class _OmniPackingPipeline(_BatchPipeline):
    """Compose selection, selected-sample encoding, packing, and batch encoding."""

    selector: PackingSelector
    selected_sample_encoder: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    packer: SamplePacker
    batch_encoder: Callable[[Mapping[str, Any]], Mapping[str, Any]]

    @classmethod
    def from_dataset(cls, dataset: Any) -> "_OmniPackingPipeline":
        """Build the packing pipeline from one normalized Omni Dataset interface."""
        selected_sample_encoder = getattr(dataset, "encode_selected_sample", None)
        batch_encoder = getattr(dataset, "encode_batch", None)
        if not callable(selected_sample_encoder) or not callable(batch_encoder):
            raise TypeError("OmniPackingLoader requires encode_selected_sample() and encode_batch()")

        # Hook—a criterion used to select data objects; this criterion can be extended to support load balancing.
        selector = getattr(dataset, "packing_selector", None)
        if selector is None:
            selector = FirstFitPackingSelector()
        if not isinstance(selector, PackingSelector):
            raise TypeError("dataset.packing_selector must implement PackingSelector")

        # This is a hook function used to package data into different fields;
        # you need to verify that the keyword mappings match.
        packer = getattr(dataset, "sample_packer", None)
        if packer is None:
            packer = SamplePacker()
        if not isinstance(packer, SamplePacker):
            raise TypeError("dataset.sample_packer must implement SamplePacker")

        packing_pipeline = cls(
            selector=selector,
            selected_sample_encoder=selected_sample_encoder,
            packer=packer,
            batch_encoder=batch_encoder,
        )
        return packing_pipeline

    def build_candidate_buffer(
            self,
            *,
            token_budget: int,
            min_buffered_samples: int,
    ) -> BaseCandidateBuffer:
        """Build candidate selection around the composed selector."""
        candidate_buffer = PackingCandidateBuffer(
            token_budget=token_budget,
            min_buffered_samples=min_buffered_samples,
            packing_selector=self.selector,
        )
        return candidate_buffer

    def finalize_selected_samples(
            self,
            selected_samples: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]:
        """Encode and pack one selected sample group."""
        encoded_samples = []
        for selected_sample in selected_samples:
            encoded_sample = self.selected_sample_encoder(selected_sample)
            if not isinstance(encoded_sample, Mapping):
                raise TypeError("postencode_sample must return a mapping")
            encoded_samples.append(encoded_sample)

        packed_sample = self.packer.pack_selected_samples(encoded_samples)
        if not isinstance(packed_sample, Mapping):
            raise TypeError("SamplePacker must return one mapping")

        packed_samples = [packed_sample]
        return packed_samples

    def encode_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply the composed final batch encoder."""
        encoded_batch = self.batch_encoder(batch)
        return encoded_batch


__all__ = [
    "BaseCandidateBuffer",
    "FullSampleCheckpoint",
    "IndexReplayCheckpoint",
    "MappingReplayDataset",
    "PackingCandidateBuffer",
    "TokenBudgetCandidateBuffer",
]
