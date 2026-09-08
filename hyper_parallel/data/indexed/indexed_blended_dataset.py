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
"""Standard deterministic weighted Dataset blending."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.indexed.indexed_data_config import GPTDatasetConfig
from hyper_parallel.data.indexed.indexed_helpers import build_blending_indices

logger = get_dataset_logger(__name__)


class BlendedDataset:
    """Expose several Datasets through deterministic weighted sample indices."""

    def __init__(
            self,
            datasets: Sequence[Any],
            weights: Sequence[float],
            size: int,
            config: GPTDatasetConfig,
    ) -> None:
        """Validate the inputs and create weighted sample indices."""
        if not datasets or len(datasets) != len(weights):
            raise ValueError("datasets and weights must be non-empty and have the same length")
        if len(datasets) >= np.iinfo(np.int16).max:
            raise ValueError("number of datasets must be less than 32767")
        if not all(isinstance(dataset, type(datasets[0])) for dataset in datasets):
            raise ValueError("all datasets must be of the same type")
        self.datasets = list(datasets)
        self.weights = _normalize_weights(weights)
        self.size = size
        self.config = config
        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["datasets"] = self._collect_dataset_identifiers()
        self.unique_identifiers["weights"] = self.weights
        self.unique_identifiers["size"] = self.size
        self.unique_description = json.dumps(
            self.unique_identifiers,
            indent=4,
            default=lambda value: value.unique_identifiers,
        )
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8")
        ).hexdigest()
        self.dataset_index, self.dataset_sample_index = self._build_indices()

    def _collect_dataset_identifiers(self) -> list[Any]:
        """Collect component Dataset cache identities."""
        identifiers = [dataset.unique_identifiers for dataset in self.datasets]
        return identifiers

    def __len__(self) -> int:
        """Return the requested blended sample count."""
        return self.size

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Read one sample from the Dataset selected by the blend indices."""
        dataset_id = int(self.dataset_index[index])
        sample_id = int(self.dataset_sample_index[index])
        sample = {"dataset_id": dataset_id, **self.datasets[dataset_id][sample_id]}
        return sample

    def _build_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Load or build the top-level blend index cache."""
        cache_directory = self.config.path_to_cache
        if cache_directory is None:
            logger.debug(
                "Building blended indices in memory: size=%d, sources=%d, weights are %s",
                self.size, len(self.weights), self.weights[:3],
            )
            indices = self._build_indices_in_memory()
            return indices

        cache_prefix = os.path.join(
            cache_directory,
            f"{self.unique_description_hash}-{type(self).__name__}",
        )
        description_path = f"{cache_prefix}-description.txt"
        dataset_index_path = f"{cache_prefix}-dataset_index.npy"
        sample_index_path = f"{cache_prefix}-dataset_sample_index.npy"
        cache_paths = (description_path, dataset_index_path, sample_index_path)
        if all(os.path.isfile(path) for path in cache_paths):
            logger.debug("Loading blended index cache: prefix=%s", cache_prefix)
            dataset_index = np.load(dataset_index_path, allow_pickle=True, mmap_mode="r")
            sample_index = np.load(sample_index_path, allow_pickle=True, mmap_mode="r")
            return dataset_index, sample_index

        dataset_index, sample_index = self._build_indices_in_memory()
        os.makedirs(cache_directory, exist_ok=True)
        with open(description_path, "w", encoding="utf-8") as description_file:
            description_file.write(self.unique_description)
        np.save(dataset_index_path, dataset_index, allow_pickle=True)
        np.save(sample_index_path, sample_index, allow_pickle=True)
        logger.debug("Saved blended index cache: prefix=%s", cache_prefix)
        return dataset_index, sample_index

    def _build_indices_in_memory(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a deterministic weighted source schedule."""
        dataset_index = np.empty(self.size, dtype=np.int16)
        sample_index = np.empty(self.size, dtype=np.int64)
        build_blending_indices(
            dataset_index=dataset_index,
            dataset_sample_index=sample_index,
            weights=self.weights,
        )
        requested_counts = np.bincount(dataset_index, minlength=len(self.datasets))
        for dataset_id, requested_count in enumerate(requested_counts):
            dataset_size = len(self.datasets[dataset_id])
            if requested_count > dataset_size:
                raise ValueError(
                    f"Dataset {dataset_id} has only {dataset_size} samples, "
                    f"but the blend requested {requested_count} samples"
                )
        return dataset_index, sample_index


def _normalize_weights(weights: Sequence[float]) -> list[float]:
    """Validate and normalize positive blend weights."""
    if not weights or any(weight <= 0.0 for weight in weights):
        raise ValueError("Dataset blend weights must be positive")
    weight_array = np.asarray(weights, dtype=np.float64)
    normalized_weights = (weight_array / np.sum(weight_array)).tolist()
    return normalized_weights
