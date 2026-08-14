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
"""Build and cache document, sample, and shuffle indices for GPT sampling."""

import os
import time

import numpy as np
from numpy.random import RandomState

from hyper_models.components.datasets.dataset_logging import get_dataset_logger

logger = get_dataset_logger(__name__)


def build_document_sample_shuffle_indices(
        *,
        sequence_lengths: np.ndarray,
        indices: np.ndarray,
        num_samples: int,
        sequence_length: int,
        random_seed: int,
        cache_directory: str,
        cache_key: str,
        cache_description: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or construct the three indices used by a GPT Dataset.

    Args:
        sequence_lengths: Token length for every low-level sequence.
        indices: Low-level sequence indices exposed by the current split.
        num_samples: Requested number of training samples.
        sequence_length: Number of input tokens per sample.
        random_seed: Seed used for document and sample shuffling.
        cache_directory: Directory containing reusable index arrays.
        cache_key: Stable identifier for the Dataset configuration.
        cache_description: Human-readable Dataset identity stored with the indices.

    Returns:
        Document, sample, and shuffle index arrays.
    """
    paths = {
        "description": os.path.join(cache_directory, f"{cache_key}-description.txt"),
        "document": os.path.join(cache_directory, f"{cache_key}-document_index.npy"),
        "sample": os.path.join(cache_directory, f"{cache_key}-sample_index.npy"),
        "shuffle": os.path.join(cache_directory, f"{cache_key}-shuffle_index.npy"),
    }
    if all(os.path.isfile(path) for path in paths.values()):
        logger.debug("Loading indexed sample cache: key=%s, directory=%s", cache_key, cache_directory, enabled=True)
        cached_indices = (
            np.load(paths["document"], allow_pickle=True, mmap_mode="r"),
            np.load(paths["sample"], allow_pickle=True, mmap_mode="r"),
            np.load(paths["shuffle"], allow_pickle=True, mmap_mode="r"),
        )
        return cached_indices

    start_time = time.time()
    num_tokens_per_epoch = int(np.sum(sequence_lengths[indices]))
    if num_tokens_per_epoch <= 1:
        raise ValueError("The selected indexed split must contain at least two tokens")

    num_epochs = 0
    accumulated_tokens = 0
    requested_tokens = num_samples * sequence_length + 1
    while accumulated_tokens < requested_tokens:
        num_epochs += 1
        accumulated_tokens += num_tokens_per_epoch
    if num_epochs == 1:
        separate_final_epoch = False
        samples_without_final_epoch = 0
    else:
        samples_without_final_epoch = (
                                              (num_epochs - 1) * num_tokens_per_epoch - 1
                                      ) // sequence_length
        samples_from_final_epoch = num_samples - samples_without_final_epoch
        samples_per_epoch = (num_tokens_per_epoch - 1) // sequence_length
        if samples_from_final_epoch < 0:
            raise ValueError("The calculated final-epoch sample count must not be negative")
        if samples_from_final_epoch > samples_per_epoch + 1:
            raise ValueError("The final epoch contains more samples than one complete epoch")
        separate_final_epoch = samples_from_final_epoch < int(0.80 * samples_per_epoch)

    random_state = RandomState(random_seed)
    document_index = _build_document_index(
        indices,
        num_epochs,
        random_state,
        separate_final_epoch,
    )
    sample_index = _build_sample_index(
        sequence_lengths,
        document_index,
        sequence_length,
        num_epochs,
        num_tokens_per_epoch,
    )
    if separate_final_epoch:
        shuffle_index = _build_shuffle_index(
            samples_without_final_epoch,
            sample_index.shape[0] - 1,
            random_state,
        )
    else:
        shuffle_index = _build_shuffle_index(
            sample_index.shape[0] - 1,
            sample_index.shape[0] - 1,
            random_state,
        )

    os.makedirs(cache_directory, exist_ok=True)
    with open(paths["description"], "w", encoding="utf-8") as description_file:
        description_file.write(cache_description)
    np.save(paths["document"], document_index, allow_pickle=True)
    np.save(paths["sample"], sample_index, allow_pickle=True)
    np.save(paths["shuffle"], shuffle_index, allow_pickle=True)
    logger.debug(
        "Built indexed sample cache: split_elements=%d, samples=%d, epochs=%d, elapsed=%.4f seconds, directory=%s",
        len(indices), num_samples, num_epochs, time.time() - start_time, cache_directory,
    )
    return document_index, sample_index, shuffle_index


def _build_document_index(
        documents: np.ndarray,
        num_epochs: int,
        random_state: RandomState,
        separate_final_epoch: bool,
) -> np.ndarray:
    """Repeat and shuffle the exposed sequence indices for every epoch."""
    if not separate_final_epoch or num_epochs == 1:
        document_index = np.mgrid[0:num_epochs, 0:len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1).astype(np.int32)
        random_state.shuffle(document_index)
        return document_index

    first_epochs = _build_document_index(documents, num_epochs - 1, random_state, False)
    final_epoch = _build_document_index(documents, 1, random_state, False)
    return np.concatenate((first_epochs, final_epoch))


def _build_sample_index(
        sequence_lengths: np.ndarray,
        document_index: np.ndarray,
        sequence_length: int,
        num_epochs: int,
        num_tokens_per_epoch: int,
) -> np.ndarray:
    """Mark the start document and offset of every fixed-length sample."""
    num_samples = (num_epochs * num_tokens_per_epoch - 1) // sequence_length
    sample_index = np.empty((num_samples + 1, 2), dtype=np.int32)
    sample_position = 0
    document_position = 0
    document_offset = 0
    sample_index[sample_position] = (document_position, document_offset)
    sample_position += 1

    while sample_position <= num_samples:
        remaining_length = sequence_length + 1
        while remaining_length != 0:
            document_id = document_index[document_position]
            document_length = int(sequence_lengths[document_id]) - document_offset
            remaining_length -= document_length
            if remaining_length <= 0:
                document_offset += remaining_length + document_length - 1
                remaining_length = 0
            else:
                document_position += 1
                document_offset = 0
        sample_index[sample_position] = (document_position, document_offset)
        sample_position += 1
    return sample_index


def _build_shuffle_index(
        num_samples: int,
        total_size: int,
        random_state: RandomState,
) -> np.ndarray:
    """Shuffle the main epoch range separately from a small final epoch."""
    dtype = np.uint32 if total_size < np.iinfo(np.uint32).max - 1 else np.int64
    first_range = np.arange(0, num_samples, dtype=dtype)
    random_state.shuffle(first_range)
    if num_samples == total_size:
        return first_range

    final_range = np.arange(num_samples, total_size, dtype=dtype)
    random_state.shuffle(final_range)
    return np.concatenate((first_range, final_range))
