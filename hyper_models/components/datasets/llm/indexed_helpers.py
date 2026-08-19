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
"""Local index-building helpers for indexed LLM Datasets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from hyper_models.components.datasets.llm import _indexed_helpers_cpp
except ImportError:  # The native helper is optional for source-only and test environments.
    _indexed_helpers_cpp = None


def build_blending_indices(
    dataset_index: np.ndarray,
    dataset_sample_index: np.ndarray,
    weights: Sequence[float],
    num_datasets: int,
    size: int,
    verbose: bool = False,
) -> None:
    """Build deterministic weighted blend indices.

    Args:
        dataset_index: Output source-Dataset indices with shape ``(size,)``.
        dataset_sample_index: Output per-source sample indices with shape
            ``(size,)``.
        weights: Normalized source-Dataset weights.
        num_datasets: Number of source Datasets.
        size: Number of blended samples to schedule.
        verbose: Compatibility argument from the compiled helper.

    Raises:
        ValueError: If input sizes or output buffers are inconsistent.
    """
    if num_datasets <= 0 or len(weights) != num_datasets:
        raise ValueError("num_datasets must be positive and match weights")
    if size < 0:
        raise ValueError("size must be non-negative")
    if dataset_index.shape != (size,) or dataset_sample_index.shape != (size,):
        raise ValueError("blend index output buffers must have shape (size,)")
    if dataset_index.dtype != np.int16 or dataset_sample_index.dtype != np.int64:
        raise ValueError("blend index output buffers must use int16 and int64 dtypes")

    if _indexed_helpers_cpp is not None:
        weight_array = np.asarray(weights, dtype=np.float64)
        _indexed_helpers_cpp.build_blending_indices(
            dataset_index=dataset_index,
            dataset_sample_index=dataset_sample_index,
            weights=weight_array,
            num_datasets=num_datasets,
            size=size,
            verbose=verbose,
        )
        return

    # Extended precision preserves the same tie-breaking observed from
    # the compiled helper for decimal weights such as 0.1/0.2/0.7.
    weight_array = np.asarray(weights, dtype=np.longdouble)
    current_samples = np.zeros(num_datasets, dtype=np.int64)
    for sample_index in range(size):
        sample_position = max(float(sample_index), 1.0)
        sampling_errors = weight_array * sample_position - current_samples
        dataset_id = int(np.argmax(sampling_errors))
        dataset_index[sample_index] = dataset_id
        dataset_sample_index[sample_index] = current_samples[dataset_id]
        current_samples[dataset_id] += 1
