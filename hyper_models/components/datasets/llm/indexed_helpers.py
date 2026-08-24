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

from hyper_models.components.datasets.llm._indexed_helpers_cpp import (
    build_blending_indices as _build_blending_indices_cpp,
    build_sample_index_int32,
    build_sample_index_int64,
)


def build_sample_index(
    sequence_lengths: np.ndarray,
    document_index: np.ndarray,
    sequence_length: int,
    num_epochs: int,
    num_tokens_per_epoch: int,
    drop_last_partial_sequence: bool,
    add_extra_token_to_sequence: bool,
) -> np.ndarray:
    """Build the 2-D sample index using the properly typed templated C++ helper.

    Args:
        sequence_lengths (np.ndarray): The 1-D array of document lengths.

        document_index (np.ndarray): The 1-D array of document indices.

        sequence_length (int): The sequence length.

        num_epochs (int): The number of epochs.

        num_tokens_per_epoch (int): The number of tokens per epoch.

        drop_last_partial_sequence (bool): Whether to omit the last partial sequence in the sample
            index should it exist.

        add_extra_token_to_sequence (bool): Whether to build samples with sequence length
            ``sequence_length + 1``.

    Returns:
        np.ndarray: The 2-D sample index.
    """
    sample_index_max = max(document_index.shape[0], int(sequence_lengths.max()))
    sample_index_builder = (
        build_sample_index_int32 if sample_index_max <= np.iinfo(np.int32).max else build_sample_index_int64
    )
    sample_index = sample_index_builder(
        sequence_lengths,
        document_index,
        sequence_length,
        num_epochs,
        num_tokens_per_epoch,
        drop_last_partial_sequence,
        add_extra_token_to_sequence,
    )
    return sample_index


def build_blending_indices(
    dataset_index: np.ndarray,
    dataset_sample_index: np.ndarray,
    weights: Sequence[float],
) -> None:
    """Build deterministic weighted blend indices.

    Args:
        dataset_index: Output source-Dataset ID for each blended sample.
        dataset_sample_index: Output per-source sample ID for each blended sample.
        weights: Normalized source-Dataset weights.
    """
    weight_array = np.asarray(weights, dtype=np.float64)
    _build_blending_indices_cpp(
        dataset_index=dataset_index,
        dataset_sample_index=dataset_sample_index,
        weights=weight_array,
    )
