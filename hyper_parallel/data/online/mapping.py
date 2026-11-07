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
import hashlib
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from copy import copy
from dataclasses import replace
from typing import Any, BinaryIO, Literal

import numpy as np

from hyper_parallel.data.constants import ONLINE_SOURCE_PATH_KEY, ONLINE_SPLIT_COUNT, ONLINE_SPLIT_NAMES
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.online.mapping_config import OnlineMappingConfig
from hyper_parallel.data.online.mapping_dataset import _OnlineMappingBlendIndex, _OnlineMappingSampleDataset
from hyper_parallel.data.online.source import (
    OnlineDataPath,
    OnlinePathSpec,
    _OnlineLeafSourceLoader,
    _OnlineSourceBuilder,
)
from hyper_parallel.data.parallel.dataloader_parallel import DataLoaderParallelContext, build_dataset_for_dataloader

logger = get_dataset_logger(__name__)

SampleTransform = Callable[[Any], list[Any]]
_MAPPING_INDEX_VERSION = 3


class _RecordReader(ABC):
    """Locate -> read -> validate externally -> export failures.

    Position: (file_id, byte_offset for JSONL or row_index for row-based readers).
    """

    data_files: tuple[str, ...]

    @abstractmethod
    def iter_positions(self) -> Iterator[tuple[int, int]]:
        """Yield original record positions.

        Example:
            JSONL: tell() -> readline() -> (0, 0), (0, 372), (0, 773), ...
        """

    @abstractmethod
    def read_record(self, position: tuple[int, int]) -> Mapping[str, Any]:
        """Read all fields of one record.

        Raises:
            ValueError: If the record is malformed and may be filtered by the source.

        Example:
            JSONL: (0, 372) -> data_files[0] -> seek(372) -> readline() -> json.loads()
        """

    @abstractmethod
    def write_failed_record(self, position: tuple[int, int], output_file: BinaryIO) -> None:
        """Export the complete failed record; keep JSONL bytes unchanged.

        Example:
            JSONL: (0, 372) -> seek(372) -> readline() -> output_file.write(raw_line)
        """

    @staticmethod
    def _write_bytes(raw_line: bytes, output_file: BinaryIO) -> None:
        """raw bytes -> output_file.write -> verify byte count."""
        bytes_written = output_file.write(raw_line)
        if bytes_written != len(raw_line):
            raise OSError(f"Incomplete failure-record write: expected {len(raw_line)}, wrote {bytes_written}")


class _JsonlRecordReader(_RecordReader):
    """Read large JSONL manifests with one record per line.

    Example:
        manifests/000014.jsonl -> (file_id, byte_offset) -> original line
    """

    def __init__(self, data_files: Sequence[str]) -> None:
        """Keep file order: [a.jsonl, b.jsonl] -> file IDs 0, 1."""
        self.data_files: tuple[str, ...] = tuple(data_files)
        self._file_handles: dict[int, BinaryIO] = {}
        self._file_stack = ExitStack()
        self._reader_process_id: int | None = os.getpid()

    def iter_positions(self) -> Iterator[tuple[int, int]]:
        """Yield nonblank JSONL line starts without parsing.

        Example:
            data_files[0] -> open("rb") -> tell()/readline() -> (0, 0), (0, 372), ...
        """
        for file_id, data_file in enumerate(self.data_files):
            with open(data_file, "rb") as file_handle:
                while True:
                    byte_offset = file_handle.tell()
                    raw_line = file_handle.readline()
                    if not raw_line:
                        break

                    if not raw_line.strip():
                        continue

                    position = (file_id, byte_offset)
                    yield position

    def read_record(self, position: tuple[int, int]) -> Mapping[str, Any]:
        """Read one complete JSON object; preserve every field.

        Example:
            (0, 372) -> data_files[0] -> seek(372) -> readline() -> json.loads()
        """
        raw_line = self._read_raw_record(position)
        file_id, byte_offset = position
        data_file = self.data_files[file_id]
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid JSONL record: {data_file} at byte {byte_offset}") from error

        if not isinstance(record, dict):
            raise ValueError(f"JSONL record must be an object: {data_file} at byte {byte_offset}")

        return record

    def write_failed_record(self, position: tuple[int, int], output_file: BinaryIO) -> None:
        """Copy the entire original line without parsing or adding metadata.

        Example:
            (0, 372) -> _read_raw_record() -> output_file.write(raw_line)
        """
        raw_line = self._read_raw_record(position)
        self._write_bytes(raw_line, output_file)

    def __getstate__(self) -> dict[str, Any]:
        """reader -> pickle without open files -> worker opens its own files."""
        reader_state = dict(self.__dict__)
        reader_state["_file_handles"] = {}
        reader_state["_file_stack"] = ExitStack()
        reader_state["_reader_process_id"] = None
        return reader_state

    def _read_raw_record(self, position: tuple[int, int]) -> bytes:
        """position -> process-local file handle -> seek -> readline -> raw bytes."""
        file_id, byte_offset = position
        if not 0 <= file_id < len(self.data_files) or byte_offset < 0:
            raise ValueError(f"Invalid JSONL record position: {position}")

        process_id = os.getpid()
        if self._reader_process_id != process_id:
            self._file_stack.close()
            self._file_stack = ExitStack()
            self._file_handles = {}
            self._reader_process_id = process_id

        file_handle = self._file_handles.get(file_id)
        if file_handle is None:
            data_file = self.data_files[file_id]
            file_handle = self._file_stack.enter_context(open(data_file, "rb"))
            self._file_handles[file_id] = file_handle

        file_handle.seek(byte_offset)
        raw_line = file_handle.readline()
        return raw_line


class _DatasetRecordReader(_RecordReader):
    """Read JSON arrays, CSV, Parquet or Arrow through the existing Dataset loader.

    Example:
        train.parquet -> Dataset -> (file_id, row_index) -> logical record
    """

    def __init__(self, source_loader: _OnlineLeafSourceLoader) -> None:
        """Local source paths -> ordered files; keep ratio splitting outside the reader."""
        if isinstance(source_loader.data_path, Mapping):
            raise ValueError("Dataset reader requires one local source; resolve named splits first")

        file_paths, _ = source_loader._collect_local_data_files(source_loader.data_path)
        data_files = []
        for file_path in file_paths:
            matched_files = sorted(glob.glob(os.path.expanduser(file_path)))
            if not matched_files:
                raise FileNotFoundError(f"Dataset source does not exist: {file_path}")

            data_files.extend(os.path.abspath(matched_file) for matched_file in matched_files)

        self.data_files: tuple[str, ...] = tuple(data_files)
        self._data_config = dict(source_loader.data_config)
        self._data_config.pop("split", None)
        self._datasets: dict[int, Any] = {}

    def iter_positions(self) -> Iterator[tuple[int, int]]:
        """Enumerate unfiltered logical rows in file order.

        Example:
            [a.parquet: 2 rows, b.parquet: 1 row] -> (0, 0), (0, 1), (1, 0)
        """
        for file_id in range(len(self.data_files)):
            dataset = self._load_dataset(file_id)
            for row_index in range(len(dataset)):
                position = (file_id, row_index)
                yield position

    def read_record(self, position: tuple[int, int]) -> Mapping[str, Any]:
        """Read all stored fields without decoding image or audio features.

        Example:
            (0, 1) -> datasets[0] -> row 1 -> complete record
        """
        file_id, row_index = position
        dataset = self._load_dataset(file_id)
        if not 0 <= row_index < len(dataset):
            raise ValueError(f"Invalid Dataset record position: {position}")

        record = dataset[row_index].to_pylist()[0]
        if not isinstance(record, Mapping):
            raise ValueError(f"Dataset record must be an object: {self.data_files[file_id]} row {row_index}")

        return record

    def write_failed_record(self, position: tuple[int, int], output_file: BinaryIO) -> None:
        """Export all fields as one JSONL record without adding metadata.

        Example:
            (0, 1) -> read_record() -> json.dumps() -> UTF-8 + newline -> output_file
        """
        record = self.read_record(position)
        try:
            record_json = json.dumps(record, ensure_ascii=False, allow_nan=False)
            raw_line = record_json.encode("utf-8") + b"\n"
        except (TypeError, ValueError) as error:
            file_id, row_index = position
            raise ValueError(
                f"Cannot export complete record as JSONL: {self.data_files[file_id]} row {row_index}: {error}"
            ) from error

        self._write_bytes(raw_line, output_file)

    def _load_dataset(self, file_id: int) -> Any:
        """file_id -> load once without filtering/splitting -> reuse Dataset."""
        if not 0 <= file_id < len(self.data_files):
            raise ValueError(f"Invalid Dataset file ID: {file_id}")

        if file_id not in self._datasets:
            file_loader = _OnlineLeafSourceLoader(self.data_files[file_id], self._data_config)
            dataset = file_loader.load(streaming=False, sample_filter=None)
            self._datasets[file_id] = dataset.with_format("arrow")

        dataset = self._datasets[file_id]
        return dataset


def build_online_mapping_source(
    *,
    data_config: Mapping[str, Any] | OnlineMappingConfig,
    data_path: OnlineDataPath | None = None,
    sample_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    train_valid_test_num_samples: list[int] | tuple[int, int, int] | None = None,
    mode: Literal["production", "debug"] = "production",
    dataloader_context: DataLoaderParallelContext | None = None,
) -> Any:
    """Build a finite integer-indexed Online source.

    Args:
        data_config: Existing source options, or an internally assembled OnlineMappingConfig.
        data_path: Local path, Hub dataset ID, or pre-split path mapping. Use
            ``data_config.sources`` instead for multiple sources.
        sampler_type: Cyclic leaves shuffle and repetition across epochs to the sampler.
        sample_filter: Optional raw-sample predicate applied before indexing.
        train_valid_test_num_samples: Optional target sample count for each
            train, validation, and test split.
        mode: Failure policy for sources using the filtered index.
        dataloader_context: Internal rank ownership and shared-cache synchronization.
            None selects standalone construction.

    Returns:
        One Mapping Dataset or a train-valid-test tuple.

    Note:
        A Config explicitly controls persistent filtering and owns its callback, mode
        and sample targets. Plain dictionaries retain the legacy source path and use
        the separate callback/mode/target arguments. Callback attributes are never inspected.
    """
    if isinstance(data_config, OnlineMappingConfig):
        if sample_filter is not None or train_valid_test_num_samples is not None or mode != "production":
            raise ValueError("Configure the filter, mode and sample targets inside OnlineMappingConfig")

        config = data_config
    else:
        config = OnlineMappingConfig(
            source_options=data_config, sample_filter=sample_filter, mode=mode,
            train_valid_test_num_samples=train_valid_test_num_samples,
        )
    context = dataloader_context or DataLoaderParallelContext()

    def dataset_factory() -> Any:
        """Keep configuration, cache and report errors inside the coordinated build."""
        writer = not context.distributed_enabled or context.build_cache_on_rank()
        dataset_builder = _OnlineMappingSourceBuilder(data_path, config, writer=writer)
        source = dataset_builder.build_source()
        return source

    if not config.use_filtered_cache:
        mapping_source = dataset_factory()
        return mapping_source

    mapping_source = build_dataset_for_dataloader(
        dataset_factory, context, barrier_needed=True, mode=config.mode,
    )
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
        logger.debug(
            "Online sample source_index=%s sample_id=%s", index, raw_sample.get("id"),
            enabled=True,
        )
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

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch selection while retaining lazy sample transformation."""
        epoch_setter = getattr(self.source_dataset, "set_epoch", None)
        if callable(epoch_setter):
            epoch_setter(epoch)


class _FilteredMappingSource:
    """reader + valid positions -> integer-indexed training records.

    Example:
        [(0, 0), (0, 240)] -> source[1] -> reader.read_record((0, 240))
    """

    def __init__(
        self,
        reader: _RecordReader,
        config: OnlineMappingConfig,
        *,
        writer: bool = True,
    ) -> None:
        """reader + normalized Config -> source awaiting a valid-record index."""
        self.config = config
        self._reader = reader
        self.data_files = reader.data_files
        self._sample_filter = config.sample_filter
        self._data_config = config.reading_options
        self._mode = config.mode
        self._is_cache_writer = writer
        self._record_positions: np.ndarray | None = None
        self._record_start = 0
        self._record_end = 0
        self._scanned_count = 0
        self._failed_count = 0
        self._failure_directory: str | None = None
        self._index_fingerprint: str | None = None

    @property
    def unique_identifiers(self) -> Mapping[str, Any]:
        """Describe one valid-record view without reading its original records.

        Example:
            One file cache with split bounds [0, 8] identifies its train view;
            bounds [8, 10] identify validation without duplicating the valid index.
        """
        if self._index_fingerprint is None:
            raise RuntimeError("Filtered Mapping source index has not been built or loaded")

        identifiers = {
            "class": type(self).__name__,
            "index_fingerprint": self._index_fingerprint,
            "record_range": [self._record_start, self._record_end],
        }
        return identifiers

    def __len__(self) -> int:
        """valid positions[start:end] -> number of records in this view."""
        if self._record_positions is None:
            raise RuntimeError("Filtered Mapping source index has not been built or loaded")

        record_count = self._record_end - self._record_start
        return record_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        """source[1] -> valid position -> complete record + source file path."""
        record_count = len(self)
        if index < 0:
            index += record_count
        if index < 0 or index >= record_count:
            raise IndexError("Filtered Mapping source index out of range")

        record_index = self._record_start + index
        file_id, record_position = self._record_positions[record_index]
        position = (int(file_id), int(record_position))
        source_record = dict(self._reader.read_record(position))
        source_record[ONLINE_SOURCE_PATH_KEY] = self._reader.data_files[position[0]]
        return source_record

    def split_by_percentages(
        self,
        percentages: Sequence[int],
    ) -> tuple[_FilteredMappingSource | None, _FilteredMappingSource | None, _FilteredMappingSource | None]:
        """Share the valid index and reader; change only each view's bounds.

        Example:
            10 valid records + (80, 10, 10) -> [0:8], [8:9], [9:10]
        """
        if (
            len(percentages) != ONLINE_SPLIT_COUNT
            or any(percentage < 0 for percentage in percentages)
            or sum(percentages) != 100
        ):
            raise ValueError("Split requires three non-negative percentages summing to 100")

        record_count = len(self)
        train_end = int(round(percentages[0] * record_count / 100.0))
        valid_end = int(round((percentages[0] + percentages[1]) * record_count / 100.0))
        split_ranges = ((0, train_end), (train_end, valid_end), (valid_end, record_count))

        split_sources = []
        for percentage, (split_start, split_end) in zip(percentages, split_ranges):
            split_source = None
            if percentage > 0:
                split_source = copy(self)
                split_source._record_start = self._record_start + split_start
                split_source._record_end = self._record_start + split_end
                split_source._is_cache_writer = False
            split_sources.append(split_source)

        mapping_splits = (split_sources[0], split_sources[1], split_sources[2])
        return mapping_splits

    def _describe_valid_record_index(self) -> str:
        """Describe the ordered source snapshot and rules as deterministic JSON.

        Config provides the explicit filter identity; callback object representations
        and their owning objects are not inspected.
        Image changes require explicitly clearing the corresponding cache.

        Example:
            [a.jsonl, b.jsonl] + reader options + {"name": "local_images", "version": 1}
            -> stable JSON; reversing the files or changing a rule changes the result.
        """
        filter_identifiers = self.config.filter_identifiers
        if self._sample_filter is not None and not filter_identifiers:
            raise ValueError("Mapping index requires explicit filter_identifiers in Config")

        source_files = []
        for data_file in self._reader.data_files:
            file_stat = os.stat(data_file)
            source_files.append({
                "path": os.path.abspath(data_file),
                "size": file_stat.st_size,
                "mtime_ns": file_stat.st_mtime_ns,
            })

        reader_type = type(self._reader)
        index_description = {
            "version": _MAPPING_INDEX_VERSION,
            "reader": f"{reader_type.__module__}.{reader_type.__qualname__}",
            "data_files": source_files,
            "data_config": self._data_config,
            "filtered": self._sample_filter is not None,
            "sample_filter": dict(filter_identifiers) if filter_identifiers is not None else None,
        }
        try:
            description_json = json.dumps(
                index_description, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Mapping index reading and validation options must be JSON-serializable") from error

        return description_json

    def _build_or_load_valid_record_index(self) -> None:
        """Select a fingerprint directory and build or load its effective index.

        Config supplies the existing path_to_cache root and explicit filter identity;
        mode controls failure handling independently.

        Example:
            First call: snapshot + rules -> indices/<hash> -> scan -> publish.
            Same snapshot and rules: indices/<same hash> -> read-only mmap.
            Changed manifest or rules: indices/<new hash> -> rebuild by writer.
        """
        cache_root = self.config.path_to_cache
        if not isinstance(cache_root, str) or not cache_root.strip():
            raise ValueError("Filtered Mapping sources require data_config.path_to_cache")

        description_json = self._describe_valid_record_index()
        fingerprint = hashlib.sha256(description_json.encode("utf-8")).hexdigest()
        cache_directory = os.path.join(cache_root, "indices", fingerprint)
        index_path = os.path.join(cache_directory, "record_index.npy")
        description_path = os.path.join(cache_directory, "description.json")
        self._failure_directory = os.path.join(cache_directory, "failures")
        index_description = json.loads(description_json)

        if os.path.exists(description_path):
            self._load_valid_record_index(index_path, description_path, index_description)
        else:
            if not self._is_cache_writer:
                raise RuntimeError(f"Mapping index cache is not ready: {cache_directory}")

            logger.info(
                "Building Mapping index: index_file=%s metadata_file=%s failures_dir=%s",
                os.path.abspath(index_path), os.path.abspath(description_path),
                os.path.abspath(self._failure_directory),
            )
            os.makedirs(cache_directory, exist_ok=True)
            if os.path.exists(self._failure_directory):
                shutil.rmtree(self._failure_directory)
            self._scan_and_filter_records(self._failure_directory)
            if self._describe_valid_record_index() != description_json:
                self._record_positions = None
                self._record_end = 0
                raise RuntimeError("Source files or validation rules changed while building the Mapping index")

            description = {
                "index_description": index_description,
                "scanned_count": self._scanned_count,
                "valid_count": len(self),
                "failed_count": self._failed_count,
            }
            with open(index_path + ".tmp", "wb") as index_file:
                np.save(index_file, self._record_positions, allow_pickle=False)
            os.replace(index_path + ".tmp", index_path)
            with open(description_path + ".tmp", "w", encoding="utf-8") as description_file:
                json.dump(description, description_file, ensure_ascii=False, sort_keys=True)
            # Publish the completion marker only after the index and failures are ready.
            os.replace(description_path + ".tmp", description_path)
            logger.info(
                "Built Mapping index: scanned=%d valid=%d failed=%d index_file=%s metadata_file=%s failures_dir=%s",
                self._scanned_count, len(self), self._failed_count,
                os.path.abspath(index_path), os.path.abspath(description_path),
                os.path.abspath(self._failure_directory),
            )

        self._index_fingerprint = fingerprint

    def _load_valid_record_index(
        self,
        index_path: str,
        description_path: str,
        index_description: Mapping[str, Any],
    ) -> None:
        """Validate a published cache before exposing its readonly record positions.

        Example:
            scanned=10, valid=8, failed=2 and record_index.shape=(8, 2):
            debug reopens the 8 records; production rejects the cached failures.
            A corrupt completed cache raises instead of silently rescanning its source.
        """
        cache_directory = os.path.dirname(description_path)
        with open(description_path, "r", encoding="utf-8") as description_file:
            description = json.load(description_file)
        if not isinstance(description, dict) or description.get("index_description") != index_description:
            raise ValueError(f"Incompatible Mapping index cache: {cache_directory}")

        scanned_count = description.get("scanned_count")
        valid_count = description.get("valid_count")
        failed_count = description.get("failed_count")
        counts = (scanned_count, valid_count, failed_count)
        if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
            raise ValueError(f"Invalid Mapping index counts: {cache_directory}")

        if scanned_count != valid_count + failed_count:
            raise ValueError(f"Inconsistent Mapping index counts: {cache_directory}")

        record_positions = np.load(index_path, mmap_mode="r", allow_pickle=False)
        if record_positions.dtype != np.int64 or record_positions.shape != (valid_count, 2):
            raise ValueError(f"Invalid Mapping record index: {index_path}")

        if not os.path.isdir(self._failure_directory):
            raise ValueError(f"Missing Mapping failure records: {self._failure_directory}")

        if self._mode == "production" and failed_count:
            message = f"Mapping index cache contains {failed_count} invalid samples: {cache_directory}"
            logger.error(message)
            raise ValueError(message)

        self._record_positions = record_positions
        self._record_start = 0
        self._record_end = valid_count
        self._scanned_count = scanned_count
        self._failed_count = failed_count
        log_level = logging.INFO if self._is_cache_writer else logging.DEBUG
        logger.log(
            log_level,
            "Loaded Mapping index: scanned=%d valid=%d failed=%d index_file=%s metadata_file=%s failures_dir=%s",
            scanned_count, valid_count, failed_count,
            os.path.abspath(index_path), os.path.abspath(description_path),
            os.path.abspath(self._failure_directory),
        )

    def _scan_and_filter_records(self, failure_directory: str) -> None:
        """Scan into a new failure directory; publish positions only after success.

        ValueError from reading or validating a record denotes invalid sample data.
        Index configuration, cache writes and failure export are outside that handler.

        Example:
            (0, 0): valid -> keep
            (0, 100): invalid -> 0.jsonl -> debug: warn/skip, production: error/stop
            (0, 240): valid -> keep -> index = [(0, 0), (0, 240)]
        """
        if not self._is_cache_writer:
            raise RuntimeError("Only the index writer may scan source records")

        self._record_positions = None
        self._record_start = 0
        self._record_end = 0
        self._scanned_count = 0
        self._failed_count = 0
        os.makedirs(failure_directory, exist_ok=False)

        valid_record_positions = []
        failure_files: dict[int, BinaryIO] = {}
        with ExitStack() as file_stack:
            for position in self._reader.iter_positions():
                self._scanned_count += 1
                file_id, record_position = position
                data_file = self._reader.data_files[file_id]
                try:
                    source_record = dict(self._reader.read_record(position))
                    source_record[ONLINE_SOURCE_PATH_KEY] = data_file
                    if self._sample_filter is not None and not self._sample_filter(source_record):
                        raise ValueError("sample_filter rejected the record")

                except ValueError as error:
                    self._failed_count += 1
                    if file_id not in failure_files:
                        failure_path = os.path.join(failure_directory, f"{file_id}.jsonl")
                        failure_files[file_id] = file_stack.enter_context(open(failure_path, "wb"))
                    self._reader.write_failed_record(position, failure_files[file_id])
                    message = f"Invalid sample: source={data_file} position={record_position} reason={error}"
                    if self._mode == "production":
                        logger.error(message)
                        raise ValueError(message) from error

                    logger.warning(message)
                    continue

                valid_record_positions.append(position)

        self._record_positions = np.asarray(valid_record_positions, dtype=np.int64).reshape(-1, 2)
        self._record_end = len(valid_record_positions)


def _publish_failure_reports(
    reader_failure_directories: Sequence[tuple[_RecordReader, str]],
    report_directory: str,
) -> None:
    """Export failures from all sources; replace reports without appending old records.

    Example:
        [(reader, "index/failures")] -> failures/000014.jsonl + failures.jsonl
        a/train.jsonl + b/train.jsonl -> train-<path hash>.jsonl
    """
    failed_files = []
    file_name_counts: Counter[str] = Counter()
    for reader, failure_directory in reader_failure_directories:
        failure_file_names = set(os.listdir(failure_directory))
        for file_id, data_file in enumerate(reader.data_files):
            report_name = os.path.splitext(os.path.basename(data_file))[0] + ".jsonl"
            file_name_counts[report_name] += 1
            failure_name = f"{file_id}.jsonl"
            if failure_name in failure_file_names:
                failure_path = os.path.join(failure_directory, failure_name)
                failed_files.append((data_file, failure_path, report_name))

    report_failures = os.path.join(report_directory, "failures")
    if os.path.exists(report_failures):
        shutil.rmtree(report_failures)
    os.makedirs(report_failures)

    summary_path = os.path.join(report_directory, "failures.jsonl")
    with open(summary_path, "wb") as summary_file:
        for data_file, failure_path, report_name in failed_files:
            if file_name_counts[report_name] > 1:
                path_hash = hashlib.sha256(os.path.abspath(data_file).encode("utf-8")).hexdigest()[:12]
                report_name = os.path.splitext(report_name)[0] + f"-{path_hash}.jsonl"
            report_path = os.path.join(report_failures, report_name)
            with open(failure_path, "rb") as failure_file, open(report_path, "xb") as report_file:
                for raw_line in failure_file:
                    report_file.write(raw_line)
                    if not raw_line.endswith(b"\n"):
                        raw_line += b"\n"
                    summary_file.write(raw_line)
            logger.info("Published Mapping failure records: source=%s report_file=%s", data_file,
                        os.path.abspath(report_path))
    logger.info("Published Mapping failure summary: files=%d summary_file=%s", len(failed_files),
                os.path.abspath(summary_path))


class _JsonlMappingSource:
    """Read heterogeneous JSONL records without imposing an Arrow schema."""

    def __init__(
        self,
        data_files: Sequence[str],
        sample_filter: Callable[[Mapping[str, Any]], bool] | None,
    ) -> None:
        """Index non-empty JSONL records by file path and byte offset."""
        self.data_files = tuple(os.path.abspath(data_file) for data_file in data_files)
        self._record_locations: list[tuple[str, int]] = []
        self._record_start = 0
        self._record_end = 0
        self._file_handles: dict[str, Any] = {}
        self._file_stack = ExitStack()
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
                split_source = self._create_view(
                    self._record_locations,
                    self._record_start + split_start,
                    self._record_start + split_end,
                )
            split_sources.append(split_source)
        jsonl_splits = (split_sources[0], split_sources[1], split_sources[2])
        return jsonl_splits

    def __getstate__(self) -> dict[str, Any]:
        """Exclude process-local file handles when DataLoader workers pickle the source."""
        source_state = dict(self.__dict__)
        source_state["_file_handles"] = {}
        source_state["_file_stack"] = ExitStack()
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
                    source_record[ONLINE_SOURCE_PATH_KEY] = data_file
                    if not sample_filter(source_record):
                        continue
                self._record_locations.append((data_file, byte_offset))

    @classmethod
    def _create_view(
        cls, record_locations: list[tuple[str, int]], start: int, end: int,
    ) -> _JsonlMappingSource:
        """Create a zero-copy view over indexed records with independent readers."""
        split_source = cls.__new__(cls)
        split_source._record_locations = record_locations
        split_source._record_start = start
        split_source._record_end = end
        split_source._file_handles = {}
        split_source._file_stack = ExitStack()
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
            file_handle = self._file_stack.enter_context(open(data_file, "rb"))
            self._file_handles[data_file] = file_handle
        return file_handle

    def _close_file_handles(self) -> None:
        """Close readers inherited from another process or opened locally."""
        self._file_stack.close()
        self._file_stack = ExitStack()
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
        source_index: np.ndarray,
        source_sample_index: np.ndarray,
    ) -> None:
        """Retain prepared mid-level Datasets and their global source schedule.

        Example:
            source_index=[0, 1, 0], source_sample_index=[0, 0, 1]
            reads A[0], B[0], A[1]; each mid Dataset owns its actual record order.
        """
        if not source_datasets or len(source_datasets) != len(weights):
            raise ValueError("Mapping blend sources and weights must be non-empty and aligned")

        if source_index.ndim != 1 or source_sample_index.shape != source_index.shape or len(source_index) == 0:
            raise ValueError("Mapping blend indices must be nonempty aligned one-dimensional arrays")

        self.source_datasets = tuple(source_datasets)
        self.weights = tuple(weights)
        self.source_index = source_index
        self.source_sample_index = source_sample_index
        self.size = len(source_index)

    def __len__(self) -> int:
        """Return the requested global sample count."""
        return self.size

    def __getitem__(self, index: int) -> Any:
        """Resolve one global index to its source and source-local logical sample."""
        source_id = int(self.source_index[index])
        source_sample_id = int(self.source_sample_index[index])
        source_sample = self.source_datasets[source_id][source_sample_id]
        return source_sample


class _OnlineMappingSourceBuilder(_OnlineSourceBuilder):
    """Build file Datasets, preserve source weights, and blend each split before DP sampling.

    Example:
        Input: one source containing [a.jsonl, b.jsonl], split="50, 50, 0".
        Config selects whether each file's valid-record cache is built or loaded first.
        Output: (blend(a_train, b_train), blend(a_valid, b_valid), None).
        Each split has one global sample order; the downstream sampler owns DP sharding.
    """

    def __init__(
        self,
        data_path: OnlineDataPath | None,
        config: OnlineMappingConfig,
        *,
        writer: bool = True,
    ) -> None:
        """Store build policy without reading records or building indexes.

        Example:
            Input: data_path="/data/manifests", Config with path_to_cache="/cache/data"
            and train_valid_test_num_samples=[100, 10, 0].
            Output: a builder configured for train/valid/test targets of 100/10/0.
            writer=False permits loading published caches but forbids record scanning.
        """
        super().__init__(data_path, config.source_options)
        self.config = config
        self._uses_filtered_index = config.use_filtered_cache
        self._is_cache_writer = writer
        self._cache_root = config.path_to_cache
        self._reader_failure_directories: list[tuple[_RecordReader, str]] = []

    def build_source(self) -> Any:
        """Build configured source groups, blend their splits, and publish failure reports.

        Example:
            Input: one source [a.jsonl, b.jsonl], each with 4 valid records,
            split="50, 50, 0", and no explicit target sample counts.
            Per-file results: [(a_train, a_valid, None), (b_train, b_valid, None)].
            Output: (train with 4 samples, valid with 4 samples, None).
            Without any split configuration, return one blended training Dataset instead.
        """
        self._reader_failure_directories = []
        sample_filter = self.config.sample_filter
        try:
            source_loaders, weights = self._create_source_loaders()
            source_groups = []
            for source_loader in source_loaders:
                if self._uses_filtered_index:
                    source_group = self._build_filtered_file_sources(source_loader)
                else:
                    source_group = [self._build_leaf_source(source_loader, sample_filter)]
                source_groups.append(source_group)

            # Preserve configured source weights when a directory expands into several file Datasets.
            has_split_result = any(
                isinstance(source_result, tuple)
                for source_group in source_groups for source_result in source_group
            )

            blended_splits = []
            split_count = ONLINE_SPLIT_COUNT if has_split_result else 1
            for split_index in range(split_count):
                split_datasets, split_weights = self._collect_split_datasets_and_weights(
                    source_groups, weights, split_index,
                )
                if self.config.train_valid_test_num_samples is None:
                    target_size = sum(len(dataset) for dataset in split_datasets)
                else:
                    target_size = self.config.train_valid_test_num_samples[split_index]
                blended_split = self._build_blended_split(
                    split_datasets, split_weights, target_size, split_index=split_index,
                )
                blended_splits.append(blended_split)

            if self._uses_filtered_index and blended_splits[0] is None:
                raise ValueError("No valid training records remain after filtering and splitting")

            if not has_split_result:
                mapping_dataset = blended_splits[0]
                return mapping_dataset

            mapping_splits = (blended_splits[0], blended_splits[1], blended_splits[2])
            return mapping_splits

        finally:
            if self._is_cache_writer and self._reader_failure_directories:
                _publish_failure_reports(self._reader_failure_directories, self._cache_root)

    @staticmethod
    def _collect_split_datasets_and_weights(
        source_groups: Sequence[Sequence[Any]],
        weights: Sequence[float],
        split_index: int,
    ) -> tuple[list[Any], list[float]]:
        """Allocate each configured source's weight to its nonempty file splits.

        Example:
            Input: source_groups=[[A, B], [C]], weights=[0.6, 0.4], split_index=0,
            with train lengths len(A)=1, len(B)=3, len(C)=2.
            Output: ([A, B, C], [0.15, 0.45, 0.4]).
            Empty file splits are excluded before dividing their source's weight.
            A plain Dataset contributes only to train; tuples contribute their selected split.
        """
        split_datasets = []
        split_weights = []
        for source_group, source_weight in zip(source_groups, weights):
            group_datasets = []
            for source_result in source_group:
                if isinstance(source_result, tuple):
                    split_dataset = source_result[split_index]
                elif split_index == 0:
                    split_dataset = source_result
                else:
                    split_dataset = None
                if split_dataset is not None and len(split_dataset) > 0:
                    group_datasets.append(split_dataset)

            group_size = sum(len(dataset) for dataset in group_datasets)
            for dataset in group_datasets:
                split_datasets.append(dataset)
                split_weights.append(source_weight * len(dataset) / group_size)

        components = (split_datasets, split_weights)
        return components

    def _build_leaf_source(
        self,
        source_loader: Any,
        sample_filter: Callable[[Mapping[str, Any]], bool] | None,
    ) -> Any:
        """Keep the legacy source-loading path when Config disables persistent filtering.

        Example:
            Input: source_loader.data_path=["a.jsonl", "b.jsonl"], no ratio split.
            Output: one _JsonlMappingSource spanning both files, without a persistent valid index.
            Non-native sources fall back to source_loader.load(streaming=False).
        """
        data_path = source_loader.data_path
        split_config = source_loader.data_config.get("split")
        if isinstance(data_path, Mapping) and split_config is None:
            jsonl_split_sources = self._try_build_presplit_jsonl_sources(
                data_path,
                sample_filter,
            )
            if jsonl_split_sources is not None:
                return jsonl_split_sources

        if not isinstance(data_path, Mapping):
            jsonl_files = self._resolve_native_jsonl_files(data_path)
            logger.debug("Resolved native JSONL files: %s", jsonl_files)
            if jsonl_files is not None:
                jsonl_source = _JsonlMappingSource(jsonl_files, sample_filter)
                if split_config is not None:
                    split_percentages = source_loader.parse_split_percentages(split_config)
                    jsonl_splits = jsonl_source.split_by_percentages(split_percentages)
                    return jsonl_splits

                return jsonl_source

        mapping_source = source_loader.load(
            streaming=False,
            sample_filter=sample_filter,
        )
        return mapping_source

    def _build_filtered_file_sources(
        self,
        source_loader: _OnlineLeafSourceLoader,
    ) -> list[Any]:
        """Build one independently cached Dataset per local file, then split each.

        Example:
            Input: ["a.jsonl", "b.jsonl"], each with 4 valid records, split="50, 50, 0".
            Output: [(a_train, a_valid, None), (b_train, b_valid, None)],
            where each nonempty split view has 2 records and shares its file's valid index.
            With no split, output [a_dataset, b_dataset].

            Named split input: {"train": ["a.jsonl", "b.jsonl"], "test": "c.csv"}.
            Output: [(a_dataset, None, None), (b_dataset, None, None), (None, None, c_dataset)].
            Named splits retain their ownership and cannot be combined with a ratio split.
        """
        data_path = source_loader.data_path
        split_config = source_loader.data_config.get("split")
        if isinstance(data_path, Mapping):
            if split_config is not None:
                raise ValueError("Online ratio split cannot be combined with pre-split paths")

            if not data_path or any(split_name not in ONLINE_SPLIT_NAMES for split_name in data_path):
                raise ValueError(f"Online split paths must use nonempty entries from {ONLINE_SPLIT_NAMES}")

            source_results = []
            for split_index, split_name in enumerate(ONLINE_SPLIT_NAMES):
                if split_name in data_path:
                    split_loader = _OnlineLeafSourceLoader(data_path[split_name], source_loader.data_config)
                    file_sources = self._build_filtered_file_sources(split_loader)
                    for file_source in file_sources:
                        split_sources = [None] * ONLINE_SPLIT_COUNT
                        split_sources[split_index] = file_source
                        source_results.append(tuple(split_sources))
            return source_results

        # a.jsonl → Dataset A → cache A → A_train / A_valid
        # b.jsonl → Dataset B → cache B → B_train / B_valid
        jsonl_files = self._resolve_native_jsonl_files(data_path)
        data_files = jsonl_files if jsonl_files is not None else _DatasetRecordReader(source_loader).data_files
        source_results = []
        for data_file in data_files:
            file_loader = _OnlineLeafSourceLoader(data_file, source_loader.data_config)
            reader = _JsonlRecordReader([data_file]) if jsonl_files is not None else _DatasetRecordReader(file_loader)

            source = self._build_filtered_file_source(reader, source_loader.data_config)
            if split_config is not None:
                split_percentages = source_loader.parse_split_percentages(split_config)
                source = source.split_by_percentages(split_percentages)
            source_results.append(source)

        return source_results

    def _build_filtered_file_source(
        self,
        reader: _RecordReader,
        data_config: Mapping[str, Any],
    ) -> _FilteredMappingSource:
        """Build or reopen one file's valid index independently of sampling options.

        Example:
            Input: reader for a.jsonl with 3 records, of which 1 fails validation in debug mode.
            Output: a _FilteredMappingSource with len(source)=2 and one cached failed record.
            Changing only split or random_seed reuses this valid index without revalidation.
            An all-invalid file yields len(source)=0; the final training-set check owns rejection.
        """
        file_config = replace(self.config, source_options=data_config)
        source = _FilteredMappingSource(reader, file_config, writer=self._is_cache_writer)
        # Production can stop during scanning; its failed original rows still need publication.
        try:
            source._build_or_load_valid_record_index()
        finally:
            failure_directory = source._failure_directory
            if (failure_directory is not None and os.path.isdir(failure_directory)
                    and all(directory != failure_directory for _, directory in self._reader_failure_directories)):
                self._reader_failure_directories.append((reader, failure_directory))

        return source

    @classmethod
    def _try_build_presplit_jsonl_sources(
        cls,
        data_path: Mapping[str, OnlinePathSpec],
        sample_filter: Callable[[Mapping[str, Any]], bool] | None,
    ) -> tuple[Any, Any, Any] | None:
        """Try to preserve named splits in the legacy native-JSONL loading path.

        Example:
            Input: {"train": "a.jsonl", "test": "b.jsonl"}.
            Output: (a_source, None, b_source).
            A non-JSONL file or unsupported split name returns None to select the general loader.
        """
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
        """Resolve a local JSONL file, glob, directory, or ordered file list.

        Example:
            Input: "/data/manifests" containing b.jsonl and a.jsonl.
            Output: ["/data/manifests/a.jsonl", "/data/manifests/b.jsonl"].
            Explicit input ["b.jsonl", "a.jsonl"] retains that order; a.csv selects the fallback.
        """
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

    def _build_blended_split(
        self,
        source_datasets: Sequence[Any],
        weights: Sequence[float],
        target_size: int,
        *,
        split_index: int = 0,
    ) -> Any:
        """Build one deterministic blend from the nonempty components of a split.

        Example:
            Input: Datasets [A, B], weights=[1, 1], target_size=4, len(A)=len(B)=2.
            Output: one Dataset of length 4 with source_index=[0, 1, 0, 1];
            Build/load the Online-only global schedule, then size the mid Datasets from cached counts;
            then each mid Dataset owns its independently cached shuffle mapping.
            Empty components are omitted; no components or target_size=0 returns None.
        """
        if not source_datasets or target_size == 0:
            return None

        sample_cache = self._cache_root if self._uses_filtered_index else None
        blend_index = _OnlineMappingBlendIndex(
            source_datasets, weights, target_size, split_index=split_index,
            path_to_cache=sample_cache, writer=self._is_cache_writer,
        )
        sample_datasets = []
        for dataset, requested_count in zip(source_datasets, blend_index.source_sample_counts):
            sample_dataset = _OnlineMappingSampleDataset(
                dataset, int(requested_count), self.config.random_seed, split_index=split_index,
                path_to_cache=sample_cache, writer=self._is_cache_writer,
            )
            sample_datasets.append(sample_dataset)

        blended_dataset = _OnlineBlendedMappingDataset(
            sample_datasets, blend_index.weights, blend_index.source_index, blend_index.source_sample_index,
        )
        return blended_dataset


__all__ = [
    "build_online_mapping_source",
    "MappingTransformDataset",
]
