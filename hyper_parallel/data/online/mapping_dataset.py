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
"""Cache Online Mapping source-local sample orders and global blend schedules."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from hyper_parallel.data.dataset_logging import get_dataset_logger

logger = get_dataset_logger(__name__)
_MAPPING_SAMPLE_INDEX_VERSION = 1
_MAPPING_BLEND_INDEX_VERSION = 1


class _OnlineMappingBlendIndex:
    """Plan source requests before constructing the source-local sample Datasets.

    Example:
        Input: sources [A, B], weights=[1, 1], size=4.
        Output: source_index=[0, 1, 0, 1], source_sample_index=[0, 0, 1, 1],
        source_sample_counts=[2, 2]. Neither source records nor images are read.
    """

    def __init__(
        self,
        source_datasets: Sequence[Any],
        weights: Sequence[float],
        size: int,
        *,
        split_index: int = 0,
        path_to_cache: str | None = None,
        writer: bool = True,
    ) -> None:
        """Build in memory or reopen the published Online blend schedule.

        Example:
            Cached identities include the ordered valid-source views, weights, size and split.
            Changing only the source shuffle seed reuses this schedule; mid owns shuffling.
            writer=False may load a complete cache but cannot publish a missing one.
        """
        if not source_datasets or len(source_datasets) != len(weights):
            raise ValueError("Mapping blend sources and weights must be nonempty and aligned")

        if len(source_datasets) >= np.iinfo(np.int16).max:
            raise ValueError("Mapping blend supports fewer than 32767 sources")

        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("Mapping blend size must be a positive integer")

        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.ndim != 1 or not np.all(np.isfinite(weight_array)) or np.any(weight_array <= 0):
            raise ValueError("Mapping blend weights must be finite positive numbers")

        weight_sum = float(weight_array.sum())
        if not np.isfinite(weight_sum):
            raise ValueError("Mapping blend weight sum must be finite")

        source_identifiers = [getattr(source, "unique_identifiers", None) for source in source_datasets]
        if path_to_cache is not None and any(identity is None for identity in source_identifiers):
            raise ValueError("Cached Mapping blend indices require stable source unique_identifiers")

        self.weights = tuple((weight_array / weight_sum).tolist())
        self.size = size
        self.unique_identifiers = {
            "class": type(self).__name__,
            "version": _MAPPING_BLEND_INDEX_VERSION,
            "sources": source_identifiers,
            "weights": self.weights,
            "size": size,
            "split_index": split_index,
        }
        self.source_sample_counts: list[int] = []
        if path_to_cache is None:
            self.source_index, self.source_sample_index = self._build_blend_indices()
        else:
            self.source_index, self.source_sample_index = self._build_or_load_blend_index(path_to_cache, writer=writer)

    def _build_blend_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Assign the next sample to the source with the largest weighted deficit.

        Example:
            weights=[0.75, 0.25], size=4 -> source IDs [0, 1, 0, 0]
            and source-local IDs [0, 0, 1, 2]. Source-local IDs are never shuffled here.
        """
        source_index = np.empty(self.size, dtype=np.int16)
        source_sample_index = np.empty(self.size, dtype=np.int64)
        source_sample_counts = [0] * len(self.weights)
        for global_sample_index in range(self.size):
            scheduled_samples = max(global_sample_index, 1)
            selected_source = 0
            largest_deficit = self.weights[0] * scheduled_samples - source_sample_counts[0]
            for source_id in range(1, len(self.weights)):
                source_deficit = self.weights[source_id] * scheduled_samples - source_sample_counts[source_id]
                if source_deficit > largest_deficit:
                    selected_source = source_id
                    largest_deficit = source_deficit

            source_index[global_sample_index] = selected_source
            source_sample_index[global_sample_index] = source_sample_counts[selected_source]
            source_sample_counts[selected_source] += 1

        self.source_sample_counts = source_sample_counts
        blend_indices = (source_index, source_sample_index)
        return blend_indices

    def _build_or_load_blend_index(self, cache_root: str, *, writer: bool) -> tuple[np.ndarray, np.ndarray]:
        """Publish the two arrays first and metadata last; reopen arrays as readonly mmap.

        Example:
            blends/<hash>/{source_index.npy,source_sample_index.npy,description.json}.
            Metadata stores per-source sample counts so readers size their sample Datasets without scanning arrays.
        """
        description_json = json.dumps(
            self.unique_identifiers, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        fingerprint = hashlib.sha256(description_json.encode("utf-8")).hexdigest()
        cache_directory = os.path.join(cache_root, "blends", fingerprint)
        source_index_path = os.path.join(cache_directory, "source_index.npy")
        sample_index_path = os.path.join(cache_directory, "source_sample_index.npy")
        description_path = os.path.join(cache_directory, "description.json")
        index_description = json.loads(description_json)
        cache_hit = os.path.exists(description_path)
        if cache_hit:
            with open(description_path, "r", encoding="utf-8") as description_file:
                description = json.load(description_file)
            if not isinstance(description, dict) or description.get("index_description") != index_description:
                raise ValueError(f"Incompatible Mapping blend index cache: {cache_directory}")

            source_sample_counts = description.get("sample_counts")
            if (not isinstance(source_sample_counts, list)
                    or len(source_sample_counts) != len(self.weights)
                    or any(not isinstance(count, int) or isinstance(count, bool) or count < 0
                           for count in source_sample_counts)
                    or sum(source_sample_counts) != self.size):
                raise ValueError(f"Invalid Mapping blend sample counts: {cache_directory}")

            self.source_sample_counts = source_sample_counts
        else:
            if not writer:
                raise RuntimeError(f"Mapping blend index cache is not ready: {cache_directory}")

            source_index, sample_index = self._build_blend_indices()
            os.makedirs(cache_directory, exist_ok=True)
            for path, array in ((source_index_path, source_index), (sample_index_path, sample_index)):
                with open(path + ".tmp", "wb") as index_file:
                    np.save(index_file, array, allow_pickle=False)
                os.replace(path + ".tmp", path)
            with open(description_path + ".tmp", "w", encoding="utf-8") as description_file:
                json.dump(
                    {"index_description": index_description, "sample_counts": self.source_sample_counts},
                    description_file, ensure_ascii=False, sort_keys=True,
                )
            os.replace(description_path + ".tmp", description_path)

        source_index = np.load(source_index_path, mmap_mode="r", allow_pickle=False)
        sample_index = np.load(sample_index_path, mmap_mode="r", allow_pickle=False)
        if (source_index.dtype != np.int16 or source_index.shape != (self.size,)
                or sample_index.dtype != np.int64 or sample_index.shape != (self.size,)):
            raise ValueError(f"Invalid Mapping blend index arrays: {cache_directory}")

        logger.log(
            logging.INFO if writer else logging.DEBUG,
            "%s Mapping blend index: sources=%d samples=%d source_index_file=%s "
            "source_sample_index_file=%s metadata_file=%s",
            "Loaded" if cache_hit else "Built", len(self.weights), self.size,
            os.path.abspath(source_index_path), os.path.abspath(sample_index_path), os.path.abspath(description_path),
        )
        blend_indices = (source_index, sample_index)
        return blend_indices


class _OnlineMappingSampleDataset:
    """Expose a source split through independently shuffled, optionally repeated indices.

    Example:
        Input: a source split with records [A, B, C], requesting 5 samples.
        Two independently shuffled epochs may produce [C, A, B, B, C].
        Only record indices are cached; original records and images remain in their source.
    """

    def __init__(
        self,
        source_dataset: Any,
        num_samples: int,
        random_seed: int,
        *,
        split_index: int = 0,
        path_to_cache: str | None = None,
        writer: bool = True,
    ) -> None:
        """Build or load one source split's deterministic access order.

        Example:
            Same source, split and seed -> the same per-epoch permutations.
            Increasing num_samples extends their prefix rather than reshuffling it.
            path_to_cache=None keeps legacy, non-persistent sources in memory.
        """
        self.source_dataset = source_dataset
        self.source_length = len(source_dataset)
        if self.source_length <= 0 or num_samples < 0:
            raise ValueError("Mapping samples require a nonempty source and a nonnegative sample count")

        self.num_samples = int(num_samples)
        self.data_files = tuple(getattr(source_dataset, "data_files", ()))
        source_identifiers = getattr(source_dataset, "unique_identifiers", None)
        if path_to_cache is not None and source_identifiers is None:
            raise ValueError("Cached Mapping samples require stable source unique_identifiers")

        if source_identifiers is None:
            # Unmigrated sources keep in-memory indexing; HF's fingerprint distinguishes their source content.
            source_identifiers = {
                "class": type(source_dataset).__name__,
                "data_files": self.data_files,
                "fingerprint": getattr(source_dataset, "_fingerprint", None),
                "length": self.source_length,
            }
        seed_description = json.dumps(
            {"source": source_identifiers, "split_index": split_index, "random_seed": int(random_seed)},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        self._shuffle_seed = int(hashlib.sha256(seed_description.encode("utf-8")).hexdigest()[:8], 16)
        self.unique_identifiers = {
            "class": type(self).__name__,
            "version": _MAPPING_SAMPLE_INDEX_VERSION,
            "source": source_identifiers,
            "split_index": split_index,
            "random_seed": int(random_seed),
            "num_samples": self.num_samples,
        }
        num_source_passes = (self.num_samples + self.source_length - 1) // self.source_length
        if self.num_samples > self.source_length and writer:
            logger.warning(
                "Mapping source has insufficient samples; repeating records: "
                "files=%s split_index=%d available=%d requested=%d source_passes=%d",
                self.data_files, split_index, self.source_length, self.num_samples, num_source_passes,
            )
        if path_to_cache is None:
            self.shuffle_index = self._build_shuffle_index()
        else:
            self.shuffle_index = self._build_or_load_shuffle_index(path_to_cache, writer=writer)

    def __len__(self) -> int:
        """Return the logical sample count without reading source records."""
        return self.num_samples

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Map one logical sample to its source record.

        Example:
            shuffle_index=[2, 0, 1] -> dataset[0] reads source_dataset[2].
        """
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError("Online Mapping sample index out of range")

        source_index = int(self.shuffle_index[index])
        sample = self.source_dataset[source_index]
        return sample

    def _build_shuffle_index(self) -> np.ndarray:
        """Shuffle each complete epoch before taking the requested prefix.

        Example:
            Three valid records and 5 requests -> one full permutation, then
            the first 2 entries of another permutation; no full epoch repeats a record.
        """
        random_state = RandomState(self._shuffle_seed)
        shuffle_index = np.empty(self.num_samples, dtype=np.int64)
        for start in range(0, self.num_samples, self.source_length):
            permutation = random_state.permutation(self.source_length)
            count = min(self.source_length, self.num_samples - start)
            shuffle_index[start:start + count] = permutation[:count]

        return shuffle_index

    def _build_or_load_shuffle_index(self, cache_root: str, *, writer: bool) -> np.ndarray:
        """Publish or reopen a source-local sample cache after its valid index is ready.

        Example:
            samples/<hash>/shuffle_index.npy stores source-local record indices.
            description.json is published last; readers never build a missing cache.
        """
        description_json = json.dumps(
            self.unique_identifiers, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        fingerprint = hashlib.sha256(description_json.encode("utf-8")).hexdigest()
        cache_directory = os.path.join(cache_root, "samples", fingerprint)
        index_path = os.path.join(cache_directory, "shuffle_index.npy")
        description_path = os.path.join(cache_directory, "description.json")
        index_description = json.loads(description_json)
        cache_hit = os.path.exists(description_path)
        if cache_hit:
            with open(description_path, "r", encoding="utf-8") as description_file:
                description = json.load(description_file)
            if description != index_description:
                raise ValueError(f"Incompatible Mapping sample index cache: {cache_directory}")

        else:
            if not writer:
                raise RuntimeError(f"Mapping sample index cache is not ready: {cache_directory}")

            shuffle_index = self._build_shuffle_index()
            os.makedirs(cache_directory, exist_ok=True)
            with open(index_path + ".tmp", "wb") as index_file:
                np.save(index_file, shuffle_index, allow_pickle=False)
            os.replace(index_path + ".tmp", index_path)
            with open(description_path + ".tmp", "w", encoding="utf-8") as description_file:
                json.dump(index_description, description_file, ensure_ascii=False, sort_keys=True)
            os.replace(description_path + ".tmp", description_path)

        shuffle_index = np.load(index_path, mmap_mode="r", allow_pickle=False)
        if shuffle_index.dtype != np.int64 or shuffle_index.shape != (self.num_samples,):
            raise ValueError(f"Invalid Mapping sample index: {index_path}")

        logger.log(
            logging.INFO if writer else logging.DEBUG,
            "%s Mapping sample index: available=%d samples=%d index_file=%s metadata_file=%s",
            "Loaded" if cache_hit else "Built", self.source_length, self.num_samples,
            os.path.abspath(index_path), os.path.abspath(description_path),
        )
        return shuffle_index
