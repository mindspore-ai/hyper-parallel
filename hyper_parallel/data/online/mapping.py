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
"""Build finite integer-indexed Online sources."""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from hyper_parallel.data.constants import ONLINE_SOURCE_PATH_KEY, ONLINE_SPLIT_COUNT, ONLINE_SPLIT_NAMES
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.online.source import (
    OnlineDataPath,
    OnlinePathSpec,
    _OnlineSourceBuilder,
)

logger = get_dataset_logger(__name__)

SampleTransform = Callable[[Any], list[Any]]


def build_online_mapping_source(
    *,
    data_config: Mapping[str, Any],
    data_path: OnlineDataPath | None = None,
    sample_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    train_valid_test_num_samples: Sequence[int] | None = None,
) -> Any:
    """Build a finite integer-indexed Online source.

    Args:
        data_config: Online source options.
        data_path: Local path, Hub dataset ID, or pre-split path mapping. Use
            ``data_config.sources`` instead for multiple sources.
        sample_filter: Optional raw-sample predicate applied before indexing.
        train_valid_test_num_samples: Optional target sample count for each
            train, validation, and test split.

    Returns:
        One Mapping Dataset or a train-valid-test tuple.
    """
    dataset_builder = _OnlineMappingSourceBuilder(data_path, data_config, train_valid_test_num_samples)
    mapping_source = dataset_builder.build_source(sample_filter=sample_filter)
    return mapping_source


class MappingTransformDataset:
    """Apply a sample transform lazily to an integer-indexed source Dataset."""

    def __init__(self, source_dataset: Any, transform: SampleTransform) -> None:
        """Store the source Dataset and its lazy sample transform."""
        self.source_dataset = source_dataset
        self.transform = transform

    @classmethod
    def apply(cls, source_dataset: Any, transform: SampleTransform) -> Any:
        """Transform one Mapping Dataset or each available split.

        Args:
            source_dataset: One Mapping Dataset or a train-valid-test tuple.
            transform: Transform from one RawSample to zero, one, or many ModelSamples.

        Returns:
            A transformed Mapping Dataset or transformed split tuple.
        """
        if isinstance(source_dataset, tuple) and len(source_dataset) == ONLINE_SPLIT_COUNT:
            transformed_splits = []
            for split_dataset in source_dataset:
                transformed_dataset = None
                if split_dataset is not None:
                    transformed_dataset = cls(split_dataset, transform)
                transformed_splits.append(transformed_dataset)

            mapping_splits = (transformed_splits[0], transformed_splits[1], transformed_splits[2])
            return mapping_splits

        if source_dataset is None:
            return None

        mapping_dataset = cls(source_dataset, transform)
        return mapping_dataset

    def __len__(self) -> int:
        """Return the source Dataset length."""
        dataset_length = len(self.source_dataset)
        return dataset_length

    def get_item(self, index: int) -> list[Any]:
        """Transform one source index into zero, one, or many output samples."""
        dataset_length = len(self.source_dataset)
        if index < 0:
            index += dataset_length
        if index < 0 or index >= dataset_length:
            raise IndexError("Online Mapping Dataset index out of range")

        raw_sample = self.source_dataset[index]
        transformed_samples = self.transform(raw_sample)
        if not isinstance(transformed_samples, list):
            raise TypeError("Online Mapping sample transform must return a list")

        return transformed_samples

    def __getitem__(self, index: int) -> Any:
        """Return exactly one sample for fixed-size DataLoader consumers."""
        transformed_samples = self.get_item(index)
        if not transformed_samples:
            raise ValueError(
                "An empty transform result requires source filtering or TokenBatchLoader"
            )
        if len(transformed_samples) != 1:
            raise ValueError("A multi-sample transform result requires TokenBatchLoader")

        transformed_sample = transformed_samples[0]
        return transformed_sample


class _JsonlMappingSource:
    """Read heterogeneous JSONL records without imposing an Arrow schema."""

    def __init__(
        self,
        data_files: Sequence[str],
        sample_filter: Callable[[Mapping[str, Any]], bool] | None,
    ) -> None:
        """Index non-empty JSONL records by file path and byte offset."""
        self._record_locations: list[tuple[str, int]] = []
        self._record_start = 0
        self._record_end = 0
        self._file_handles: dict[str, Any] = {}
        self._reader_process_id = os.getpid()

        for data_file in data_files:
            self._index_file(data_file, sample_filter)
        self._record_end = len(self._record_locations)

    def __len__(self) -> int:
        """Return the number of indexed JSONL records."""
        record_count = self._record_end - self._record_start
        return record_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Deserialize one record and attach its source file context."""
        record_count = len(self)
        if index < 0:
            index += record_count
        if index < 0 or index >= record_count:
            raise IndexError("Online JSONL source index out of range")

        record_index = self._record_start + index
        data_file, byte_offset = self._record_locations[record_index]
        file_handle = self._get_file_handle(data_file)
        file_handle.seek(byte_offset)
        raw_record = file_handle.readline()
        source_record = self._parse_record(raw_record, data_file)
        source_record[ONLINE_SOURCE_PATH_KEY] = data_file
        return source_record

    def split_by_percentages(
        self,
        percentages: Sequence[int],
    ) -> tuple[_JsonlMappingSource | None, _JsonlMappingSource | None, _JsonlMappingSource | None]:
        """Create train, validation, and test views over the filtered index."""
        record_count = len(self)
        train_end = int(round(percentages[0] * record_count / 100.0))
        valid_end = int(round((percentages[0] + percentages[1]) * record_count / 100.0))
        split_ranges = ((0, train_end), (train_end, valid_end), (valid_end, record_count))

        split_sources = []
        for percentage, (split_start, split_end) in zip(percentages, split_ranges):
            split_source = None
            if percentage > 0:
                split_source = self._create_view(split_start, split_end)
            split_sources.append(split_source)
        jsonl_splits = (split_sources[0], split_sources[1], split_sources[2])
        return jsonl_splits

    def __getstate__(self) -> dict[str, Any]:
        """Exclude process-local file handles when DataLoader workers pickle the source."""
        source_state = dict(self.__dict__)
        source_state["_file_handles"] = {}
        source_state["_reader_process_id"] = None
        return source_state

    def _index_file(
        self,
        data_file: str,
        sample_filter: Callable[[Mapping[str, Any]], bool] | None,
    ) -> None:
        """Record byte offsets and evaluate an optional raw-sample filter."""
        with open(data_file, "rb") as file_handle:
            while True:
                byte_offset = file_handle.tell()
                raw_record = file_handle.readline()
                if not raw_record:
                    break
                if not raw_record.strip():
                    continue
                if sample_filter is not None:
                    source_record = self._parse_record(raw_record, data_file)
                    if not sample_filter(source_record):
                        continue
                self._record_locations.append((data_file, byte_offset))

    def _create_view(self, start: int, end: int) -> _JsonlMappingSource:
        """Create a zero-copy view that shares this source's record locations."""
        split_source = self.__class__.__new__(self.__class__)
        split_source._record_locations = self._record_locations
        split_source._record_start = self._record_start + start
        split_source._record_end = self._record_start + end
        split_source._file_handles = {}
        split_source._reader_process_id = os.getpid()
        return split_source

    def _get_file_handle(self, data_file: str) -> Any:
        """Return a binary reader owned by the current DataLoader process."""
        process_id = os.getpid()
        if self._reader_process_id != process_id:
            self._close_file_handles()
            self._reader_process_id = process_id

        file_handle = self._file_handles.get(data_file)
        if file_handle is None:
            file_handle = open(data_file, "rb")
            self._file_handles[data_file] = file_handle
        return file_handle

    def _close_file_handles(self) -> None:
        """Close readers inherited from another process or opened locally."""
        for file_handle in self._file_handles.values():
            file_handle.close()
        self._file_handles = {}

    @staticmethod
    def _parse_record(raw_record: bytes, data_file: str) -> dict[str, Any]:
        """Decode one JSON object while preserving heterogeneous nested values."""
        try:
            source_record = json.loads(raw_record)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid JSONL record in {data_file}") from error
        if not isinstance(source_record, dict):
            raise ValueError(f"Online JSONL records must be objects: {data_file}")
        return source_record


class _OnlineBlendedMappingDataset:
    """Expose Mapping sources through Megatron-style deterministic indices."""

    def __init__(
        self,
        source_datasets: Sequence[Any],
        weights: Sequence[float],
        size: int,
        random_seed: int,
    ) -> None:
        """Build the source schedule and source-local shuffle indices."""
        if not source_datasets or len(source_datasets) != len(weights):
            raise ValueError("Mapping blend sources and weights must be non-empty and aligned")
        if len(source_datasets) >= np.iinfo(np.int16).max:
            raise ValueError("Mapping blend supports fewer than 32767 sources")
        if size <= 0:
            raise ValueError("Mapping blend size must be positive")

        source_lengths = []
        for source_dataset in source_datasets:
            source_length = len(source_dataset)
            if source_length <= 0:
                raise ValueError("Mapping blend sources must not be empty")
            source_lengths.append(source_length)

        self.source_datasets = tuple(source_datasets)
        self.weights = self._normalize_weights(weights)
        self.size = size
        self.source_index, self.source_sample_index = self._build_blend_indices()
        self.source_shuffle_indices = self._build_source_shuffle_indices(
            source_lengths,
            random_seed,
        )

    def __len__(self) -> int:
        """Return the requested global sample count."""
        return self.size

    def __getitem__(self, index: int) -> Any:
        """Resolve one global index to its source and shuffled source sample."""
        source_id = int(self.source_index[index])
        source_sample_id = int(self.source_sample_index[index])
        shuffled_sample_id = int(self.source_shuffle_indices[source_id][source_sample_id])
        source_sample = self.source_datasets[source_id][shuffled_sample_id]
        return source_sample

    @staticmethod
    def _normalize_weights(weights: Sequence[float]) -> tuple[float, ...]:
        """Validate and normalize source weights."""
        if not weights or any(weight <= 0.0 for weight in weights):
            raise ValueError("Mapping blend weights must be positive")

        weight_sum = sum(weights)
        normalized_weights = []
        for weight in weights:
            normalized_weights.append(float(weight) / weight_sum)
        return tuple(normalized_weights)

    def _build_blend_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Build Megatron's deterministic largest-deficit source schedule."""
        source_index = np.empty(self.size, dtype=np.int16)
        source_logical_index = np.empty(self.size, dtype=np.int64)
        source_sample_counts = [0] * len(self.source_datasets)

        for global_sample_index in range(self.size):
            scheduled_samples = max(global_sample_index, 1)
            selected_source = 0
            largest_deficit = self.weights[0] * scheduled_samples - source_sample_counts[0]
            for source_id in range(1, len(self.source_datasets)):
                source_deficit = (
                    self.weights[source_id] * scheduled_samples
                    - source_sample_counts[source_id]
                )
                if source_deficit > largest_deficit:
                    selected_source = source_id
                    largest_deficit = source_deficit

            source_index[global_sample_index] = selected_source
            source_logical_index[global_sample_index] = source_sample_counts[selected_source]
            source_sample_counts[selected_source] += 1

        return source_index, source_logical_index

    def _build_source_shuffle_indices(
        self,
        source_lengths: Sequence[int],
        random_seed: int,
    ) -> tuple[np.ndarray, ...]:
        """Map source-local counters to repeated deterministic sample permutations."""
        source_sample_counts = np.bincount(
            self.source_index,
            minlength=len(self.source_datasets),
        )
        source_random = np.random.RandomState(random_seed)
        source_shuffle_indices = []
        for source_length, requested_samples in zip(source_lengths, source_sample_counts):
            repeated_epochs = int(np.ceil(int(requested_samples) / source_length))
            source_sample_index = np.tile(
                np.arange(source_length, dtype=np.int64),
                repeated_epochs,
            )
            source_random.shuffle(source_sample_index)
            source_shuffle_indices.append(source_sample_index[:requested_samples])
        return tuple(source_shuffle_indices)


class _OnlineMappingSourceBuilder(_OnlineSourceBuilder):
    """Build split-local, DP-global Mapping sample indices."""

    def __init__(
        self,
        data_path: OnlineDataPath | None,
        data_config: Mapping[str, Any],
        train_valid_test_num_samples: Sequence[int] | None,
    ) -> None:
        """Store source configuration and optional training sample targets."""
        super().__init__(data_path, data_config)
        self.train_valid_test_num_samples = self._validate_split_sizes(train_valid_test_num_samples)

    def build_source(
        self,
        *,
        sample_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> Any:
        """Load Mapping sources and compose their global split graphs."""
        source_loaders, weights = self._create_source_loaders()
        source_results = []
        for source_loader in source_loaders:
            source_result = self._build_leaf_source(source_loader, sample_filter)
            source_results.append(source_result)

        # Example: source A returns Dataset, while source B returns (train, valid, test).
        # The combined result follows B's split structure, and A contributes only to train.
        has_split_result = False
        for source_result in source_results:
            if isinstance(source_result, tuple):
                has_split_result = True
                break

        if not has_split_result:
            target_size = self._get_target_size(0, source_results)
            mapping_dataset = self._build_blended_split(source_results, weights, target_size)
            return mapping_dataset

        # source of train, valid, test => each source blend
        blended_splits = []
        for split_index in range(ONLINE_SPLIT_COUNT):
            split_datasets = []
            split_weights = []
            for source_result, weight in zip(source_results, weights):
                if isinstance(source_result, tuple):
                    split_dataset = source_result[split_index]
                elif split_index == 0:
                    split_dataset = source_result
                else:
                    split_dataset = None

                if split_dataset is not None:
                    split_datasets.append(split_dataset)
                    split_weights.append(weight)

            target_size = self._get_target_size(split_index, split_datasets)
            blended_split = self._build_blended_split(
                split_datasets,
                split_weights,
                target_size,
            )
            blended_splits.append(blended_split)

        # train, valid, test
        mapping_splits = (blended_splits[0], blended_splits[1], blended_splits[2])
        return mapping_splits

    @classmethod
    def _build_leaf_source(
        cls,
        source_loader: Any,
        sample_filter: Callable[[Mapping[str, Any]], bool] | None,
    ) -> Any:
        """Build one leaf through native JSONL indexing or the general loader."""
        data_path = source_loader.data_path
        split_config = source_loader.data_config.get("split")
        if isinstance(data_path, Mapping) and split_config is None:
            jsonl_split_sources = cls._try_build_presplit_jsonl_sources(
                data_path,
                sample_filter,
            )
            if jsonl_split_sources is not None:
                return jsonl_split_sources

        if not isinstance(data_path, Mapping):
            jsonl_files = cls._resolve_native_jsonl_files(data_path)
            logger.debug("Resolved native JSONL files: %s", jsonl_files)
            if jsonl_files is not None:
                jsonl_source = _JsonlMappingSource(jsonl_files, sample_filter)
                if split_config is not None:
                    split_percentages = source_loader._parse_split_percentages(split_config)
                    jsonl_splits = jsonl_source.split_by_percentages(split_percentages)
                    return jsonl_splits
                return jsonl_source

        mapping_source = source_loader.load(
            streaming=False,
            sample_filter=sample_filter,
        )
        return mapping_source

    @classmethod
    def _try_build_presplit_jsonl_sources(
        cls,
        data_path: Mapping[str, OnlinePathSpec],
        sample_filter: Callable[[Mapping[str, Any]], bool] | None,
    ) -> tuple[Any, Any, Any] | None:
        """Try to preserve configured splits with native JSONL sources."""
        if not data_path:
            return None

        resolved_split_files = {}
        for split_name, split_path in data_path.items():
            if split_name not in ONLINE_SPLIT_NAMES:
                return None
            jsonl_files = cls._resolve_native_jsonl_files(split_path)
            if jsonl_files is None:
                return None
            resolved_split_files[split_name] = jsonl_files

        split_sources = []
        for split_name in ONLINE_SPLIT_NAMES:
            split_source = None
            jsonl_files = resolved_split_files.get(split_name)
            if jsonl_files is not None:
                split_source = _JsonlMappingSource(jsonl_files, sample_filter)
            split_sources.append(split_source)
        jsonl_split_sources = (split_sources[0], split_sources[1], split_sources[2])
        return jsonl_split_sources

    @staticmethod
    def _resolve_native_jsonl_files(data_path: OnlinePathSpec) -> list[str] | None:
        """Resolve a local JSONL file, glob, directory, or ordered file list."""
        configured_paths = [data_path] if isinstance(data_path, str) else list(data_path)
        data_files = []
        for configured_path in configured_paths:
            matched_paths = sorted(glob.glob(configured_path))
            if os.path.isdir(configured_path):
                matched_paths = sorted(glob.glob(os.path.join(configured_path, "*.jsonl")))
            if not matched_paths:
                return None
            data_files.extend(matched_paths)

        if not data_files:
            return None
        for data_file in data_files:
            if not os.path.isfile(data_file) or os.path.splitext(data_file)[1].lower() != ".jsonl":
                return None
        return data_files

    def _get_target_size(self, split_index: int, source_datasets: Sequence[Any]) -> int:
        """Return the configured sample target or one natural blended epoch."""
        if self.train_valid_test_num_samples is not None:
            return self.train_valid_test_num_samples[split_index]

        natural_size = 0
        for source_dataset in source_datasets:
            natural_size += len(source_dataset)
        return natural_size

    def _build_blended_split(
        self,
        source_datasets: Sequence[Any],
        weights: Sequence[float],
        target_size: int,
    ) -> Any:
        """Build one deterministic source and sample index graph."""
        nonempty_datasets = []
        nonempty_weights = []
        for source_dataset, weight in zip(source_datasets, weights):
            if len(source_dataset) == 0:
                continue
            nonempty_datasets.append(source_dataset)
            nonempty_weights.append(weight)

        if not nonempty_datasets or target_size == 0:
            return None

        blended_dataset = _OnlineBlendedMappingDataset(
            nonempty_datasets,
            nonempty_weights,
            target_size,
            self._random_seed,
        )
        return blended_dataset

    @staticmethod
    def _validate_split_sizes(split_sizes: Sequence[int] | None) -> tuple[int, int, int] | None:
        """Validate optional train, validation, and test sample targets."""
        if split_sizes is None:
            return None
        if len(split_sizes) != ONLINE_SPLIT_COUNT:
            raise ValueError("Online Mapping sample targets must contain train, validation, and test sizes")

        normalized_sizes = (int(split_sizes[0]), int(split_sizes[1]), int(split_sizes[2]))
        if any(size < 0 for size in normalized_sizes):
            raise ValueError("Online Mapping sample targets must be non-negative")
        return normalized_sizes


__all__ = [
    "build_online_mapping_source",
    "MappingTransformDataset",
]
