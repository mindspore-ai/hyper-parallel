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
"""Runtime Dataset types selected by the indexed pretraining builder."""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.random import RandomState

from hyper_models.components.datasets.dataset_logging import get_dataset_logger
from hyper_models.components.datasets.llm.indexed_data_config import GPTDatasetConfig
from hyper_models.components.datasets.llm.indexed_data_reader import IndexedDataReader

logger = get_dataset_logger(__name__)
_PAD_TOKEN_ID = -1


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
    document_index = np.concatenate((first_epochs, final_epoch))
    return document_index


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
    shuffle_index = np.concatenate((first_range, final_range))
    return shuffle_index


class _IndexedPretrainDataset(ABC):
    """Store the common state used by all indexed pretraining Datasets."""

    def __init__(
        self,
        dataset: IndexedDataReader | None,
        dataset_path: str | None,
        indices: np.ndarray | None,
        num_samples: int,
        index_split: Any,
        config: GPTDatasetConfig,
    ) -> None:
        """Store common construction inputs and finalize the concrete Dataset."""
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.indices = indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config

        # Handle pad token id provided by the tokenizer.
        try:
            pad_token_id = self.config.tokenizer.pad
            self._pad_token_id = _PAD_TOKEN_ID if pad_token_id is None else int(pad_token_id)
        except (AttributeError, NotImplementedError):
            self._pad_token_id = _PAD_TOKEN_ID

        try:
            special_token_ids = [
                token_id
                for token_name, token_id in self.config.tokenizer.special_tokens_dict.items()
                if token_name != "pad_token"
            ]
        except (AttributeError, IndexError, ValueError):
            special_token_ids = []
        if not special_token_ids:
            for token_name in ("eos", "eod"):
                try:
                    special_token_ids.append(getattr(self.config.tokenizer, token_name))
                except (AttributeError, NotImplementedError):
                    pass

        if not self.config.mock and self._pad_token_id in special_token_ids:
            if self.config.allow_ambiguous_pad_tokens:
                logger.warning(
                    "The tokenizer PAD ID collides with another special token ID; matching tokens will also be "
                    "masked as padding"
                )
            else:
                self._pad_token_id = _PAD_TOKEN_ID
                logger.warning(
                    "The tokenizer PAD ID collides with another special token ID; using an internal padding sentinel"
                )

        if not self.config.mock:
            self.unique_identifiers = type(self).build_unique_identifiers(
                self.dataset_path,
                self.num_samples,
                self.index_split,
                self.config,
            )

            self.unique_description = json.dumps(
                self.unique_identifiers,
                indent=4,
                default=lambda value: value.unique_identifiers,
            )
            self.unique_description_hash = hashlib.md5(
                self.unique_description.encode("utf-8")
            ).hexdigest()

        self._finalize()

    def _finalize(self) -> None:
        """Build indices or validate the low-level Dataset."""

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataReader) -> int:
        """Return the number of elements used to construct split indices."""
        del low_level_dataset
        raise NotImplementedError

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str,
        config: GPTDatasetConfig,
    ) -> IndexedDataReader:
        """Build the low-level Dataset used by a concrete pretraining Dataset."""
        del dataset_path, config
        raise NotImplementedError

    @staticmethod
    def _key_config_attributes() -> list[str]:
        """Return config fields that contribute to the Dataset cache identity."""
        return [
            "random_seed",
            "sequence_length",
            "split",
            "split_matrix",
            "tokenizer",
            "drop_last_partial_validation_sequence",
            "add_extra_token_to_sequence",
        ]

    @classmethod
    def build_unique_identifiers(
        cls,
        dataset_path: str | None,
        num_samples: int,
        index_split: Any,
        config: GPTDatasetConfig,
    ) -> OrderedDict[str, Any]:
        """Build the stable cache identity without constructing a Dataset.

        Args:
            dataset_path: Prefix of the indexed Dataset files.
            num_samples: Requested number of samples.
            index_split: Train, validation, or test split identifier.
            config: Indexed GPT Dataset configuration.

        Returns:
            Ordered cache identity matching an instantiated Dataset.
        """
        identifiers = OrderedDict()
        identifiers["class"] = cls.__name__
        identifiers["dataset_path"] = dataset_path
        identifiers["num_samples"] = num_samples
        identifiers["index_split"] = index_split.name
        for attribute in cls._key_config_attributes():
            identifiers[attribute] = getattr(config, attribute)
        return identifiers

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples exposed by the Dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Return one raw pretraining sample."""


class _MockDataset(_IndexedPretrainDataset):
    """Base class for mock Datasets that do not use a low-level reader."""

    def __init__(
        self,
        dataset: IndexedDataReader | None,
        dataset_path: str | None,
        indices: np.ndarray | None,
        num_samples: int,
        index_split: Any,
        config: GPTDatasetConfig,
    ) -> None:
        """Validate mock mode and initialize the common Dataset state."""
        self.config = config
        if not self.config.mock:
            raise ValueError("config.mock must be True for MockDataset")
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    def __len__(self) -> int:
        """Return the configured number of mock samples."""
        return self.num_samples


class MockGPTDataset(_MockDataset):
    """Generate deterministic, fixed-length mock GPT samples."""

    def __init__(
        self,
        dataset: IndexedDataReader | None,
        dataset_path: str | None,
        indices: np.ndarray | None,
        num_samples: int,
        index_split: Any,
        config: GPTDatasetConfig,
    ) -> None:
        """Initialize the mock Dataset with the standard constructor contract."""
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)
        self.masks_and_position_ids_are_cacheable = not any(
            (
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
            )
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Generate a deterministic mock sample for one split and index."""
        token = 1
        pad = 2
        eod = self.config.tokenizer.eod

        if index >= self.num_samples:
            raise ValueError(
                f"Exceeded the available number of samples ({self.num_samples}); got {index}"
            )

        rng = np.random.default_rng(seed=[self.index_split.value, index])
        length = rng.integers(low=0, high=self.config.sequence_length)
        sample_tokens = np.zeros(length) + token
        sample_pads = np.zeros(self.config.sequence_length - length - 1) + pad
        sample = np.int64(np.concatenate([[length], sample_tokens, [eod], sample_pads]))
        tokens = sample[:-1]
        labels = sample[1:]

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        raw_sample = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        if self.config.create_attention_mask:
            raw_sample["attention_mask"] = attention_mask
        return raw_sample


class GPTDataset(_IndexedPretrainDataset):
    """Build fixed-length GPT samples from an indexed token stream."""

    def __init__(
        self,
        dataset: IndexedDataReader,
        dataset_path: str,
        indices: np.ndarray,
        num_samples: int,
        index_split: Any,
        config: GPTDatasetConfig,
    ) -> None:
        """Initialize the Dataset and build or load its three sample indices."""
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)
        self.masks_and_position_ids_are_cacheable = not any(
            (
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
            )
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

    def _finalize(self) -> None:
        if self.dataset is None or self.indices is None:
            raise ValueError("GPTDataset requires a low-level Dataset and split indices")

        self.document_index, self.sample_index, self.shuffle_index = self._build_document_sample_shuffle_indices()

    def _build_document_sample_shuffle_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build or load the document, sample, and shuffle indices.

        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            The document, sample, and shuffle index arrays.
        """
        cache_directory = self.config.path_to_cache
        if cache_directory is None:
            cache_directory = os.path.join(self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices")
        cache_key = f"{self.unique_description_hash}-{type(self).__name__}"

        # drop last for train/test, valid drop depended on config
        if self.index_split.name == "valid":
            drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence
        else:
            drop_last_partial_sequence = True

        cache_paths = {
            "description": os.path.join(cache_directory, f"{cache_key}-description.txt"),
            "document": os.path.join(cache_directory, f"{cache_key}-document_index.npy"),
            "sample": os.path.join(cache_directory, f"{cache_key}-sample_index.npy"),
            "shuffle": os.path.join(cache_directory, f"{cache_key}-shuffle_index.npy"),
        }

        # read from cache
        cache_hit = all(os.path.isfile(path) for path in cache_paths.values())
        if cache_hit:
            logger.debug("Loading indexed sample cache: key=%s, directory=%s", cache_key, cache_directory, enabled=True)
            document_index = np.load(cache_paths["document"], mmap_mode="r")
            sample_index = np.load(cache_paths["sample"], mmap_mode="r")
            shuffle_index = np.load(cache_paths["shuffle"], mmap_mode="r")

            return document_index, sample_index, shuffle_index

        # build cache
        start_time = time.time()
        num_tokens_per_epoch = int(np.sum(self.dataset.sequence_lengths[self.indices]))
        if num_tokens_per_epoch <= 1:
            raise ValueError("The selected indexed split must contain at least two tokens")

        # repeate documents for multi-epcho
        num_epochs = 0
        accumulated_tokens = 0
        extra_token = int(self.config.add_extra_token_to_sequence)
        requested_tokens = self.num_samples * self.config.sequence_length + extra_token
        while accumulated_tokens < requested_tokens:
            num_epochs += 1
            accumulated_tokens += num_tokens_per_epoch

        if num_epochs == 1:
            separate_final_epoch = False
            samples_without_final_epoch = 0
        else:
            samples_without_final_epoch = (
                (num_epochs - 1) * num_tokens_per_epoch - extra_token
            ) // self.config.sequence_length
            samples_from_final_epoch = self.num_samples - samples_without_final_epoch
            samples_per_epoch = (num_tokens_per_epoch - extra_token) // self.config.sequence_length

            # Keep a small final epoch separate from the main shuffled range.
            threshold = 0.80
            separate_final_epoch = samples_from_final_epoch < int(threshold * samples_per_epoch)

        random_state = RandomState(self.config.random_seed)

        # Shuffle the repeated document IDs, optionally isolating the final epoch.
        document_index = _build_document_index(self.indices, num_epochs, random_state, separate_final_epoch)

        # Convert the shuffled token stream into fixed-length sample boundaries in C++.
        # Keep the optional C++ extension out of the pre-packed MindRecord path.
        from hyper_models.components.datasets.llm.indexed_helpers import build_sample_index  # pylint: disable=C0415

        sample_index = build_sample_index(
            self.dataset.sequence_lengths,
            document_index,
            self.config.sequence_length,
            num_epochs,
            num_tokens_per_epoch,
            drop_last_partial_sequence,
            self.config.add_extra_token_to_sequence,
        )

        # Shuffle samples without mixing an isolated final epoch into the main range.
        if separate_final_epoch:
            shuffle_index = _build_shuffle_index(samples_without_final_epoch, sample_index.shape[0] - 1, random_state)
        else:
            shuffle_index = _build_shuffle_index(sample_index.shape[0] - 1, sample_index.shape[0] - 1, random_state)

        os.makedirs(cache_directory, exist_ok=True)
        np.save(cache_paths["document"], document_index)
        np.save(cache_paths["sample"], sample_index)
        np.save(cache_paths["shuffle"], shuffle_index)
        # Write the description last so readers treat it as the cache completion marker.
        with open(cache_paths["description"], "w", encoding="utf-8") as description_file:
            description_file.write(self.unique_description)

        logger.debug(
            "Built indexed sample cache: samples=%d, epochs=%d, elapsed=%.4f seconds",
            self.num_samples, num_epochs, time.time() - start_time,
        )
        return document_index, sample_index, shuffle_index

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataReader) -> int:
        """Return the number of low-level sequences available for splitting."""
        return int(low_level_dataset.sequence_lengths.shape[0])

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str,
        config: GPTDatasetConfig,
    ) -> IndexedDataReader:
        """Open the low-level indexed token reader."""
        return IndexedDataReader(dataset_path, mmap=config.mmap_bin_files)

    def __len__(self) -> int:
        """Return the number of complete fixed-length samples."""
        return int(self.sample_index.shape[0] - 1)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Read, shift, and mask one shuffled GPT sample."""
        text, _, _ = self._query_document_sample_shuffle_indices(index)

        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].copy()
            labels = text[1:].copy()
        else:
            tokens = text.copy()
            labels = np.roll(text, shift=-1)
            labels[-1] = self._pad_token_id

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        # For padded sequences, mask the loss
        loss_mask = loss_mask.copy()
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        if np.any(tokens < 0) or np.any(tokens >= len(self.config.tokenizer)):
            raise ValueError("An input token is out of bounds of the tokenizer vocabulary")

        raw_sample = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

        if self.config.create_attention_mask:
            raw_sample["attention_mask"] = attention_mask

        return raw_sample

    def _query_document_sample_shuffle_indices(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            The text ids, document ids, and per-document token counts (before any padding).
        """
        shuffled_index = int(self.shuffle_index[index])
        begin_document, begin_offset = self.sample_index[shuffled_index]
        end_document, end_offset = self.sample_index[shuffled_index + 1]

        document_ids = []
        sample_parts = []

        # Read one contiguous slice when the sample stays within a document.
        if begin_document == end_document:
            document_id = int(self.document_index[begin_document])
            document_ids.append(document_id)
            sample_parts.append(
                self.dataset.get(
                    document_id,
                    offset=int(begin_offset),
                    length=int(end_offset - begin_offset + self.config.add_extra_token_to_sequence),
                )
            )

        # Concatenate slices when the sample crosses document boundaries.
        else:
            for document_position in range(int(begin_document), int(end_document) + 1):
                document_id = int(self.document_index[document_position])
                document_ids.append(document_id)
                offset = 0 if document_position > begin_document else int(begin_offset)
                length = (
                    None
                    if document_position < end_document
                    else int(end_offset + self.config.add_extra_token_to_sequence)
                )
                sample_parts.append(self.dataset.get(document_id, offset=offset, length=length))

        # Retained partial samples still need the configured fixed length.
        length = sum(map(len, sample_parts))
        document_lengths = [len(p) for p in sample_parts]

        if length < (self.config.sequence_length + self.config.add_extra_token_to_sequence):
            sample_parts.append(
                [self._pad_token_id]
                * (self.config.sequence_length + self.config.add_extra_token_to_sequence - length)
            )

        text = np.asarray(np.concatenate(sample_parts), dtype=np.int64)

        return text, np.asarray(document_ids, dtype=np.int64), document_lengths


class GPTFromMRDataset(_IndexedPretrainDataset):
    """GPT Dataset for pre-packed samples converted from MindRecord.

    Each entry in the underlying indexed Dataset represents a complete pre-packed
    training sample. Samples are read directly without rebuilding document, sample,
    or shuffle indices.

    Args:
        dataset: The indexed Dataset containing pre-packed samples converted from MindRecord.

        dataset_path: The path to the underlying indexed Dataset.

        indices: The indices of the pre-packed samples exposed by this Dataset.

        num_samples: The number of pre-packed samples to draw from the indexed Dataset.

        index_split: The Dataset split associated with ``indices``.

        config: The GPT-specific Dataset configuration.
    """

    def __init__(
        self,
        dataset: IndexedDataReader,
        dataset_path: str | None,
        indices: np.ndarray,
        num_samples: int,
        index_split: Any,
        config: GPTDatasetConfig,
    ) -> None:
        """Initialize the Dataset and validate pre-cut record lengths."""
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)
        self.masks_and_position_ids_are_cacheable = not any(
            (
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
            )
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

    def _finalize(self) -> None:
        if self.dataset is None or self.indices is None:
            raise ValueError("GPTFromMRDataset requires a low-level Dataset and split indices")

        if not isinstance(self.config, GPTDatasetConfig):
            raise ValueError("GPTFromMRDataset requires GPTDatasetConfig")

        if not self.config.skip_data_check:
            expected_length = self.config.sequence_length + self.config.add_extra_token_to_sequence
            if not np.all(self.dataset.sequence_lengths == expected_length):
                raise ValueError(
                    "All pre-cut records must match sequence_length + add_extra_token_to_sequence; "
                    f"expected {expected_length}, got {self.dataset.sequence_lengths.tolist()}"
                )
        if self.num_samples > len(self.indices):
            raise ValueError(
                "Requested samples exceed the pre-cut records in this split: "
                f"requested {self.num_samples}, available {len(self.indices)}"
            )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataReader) -> int:
        """Return the number of pre-cut records available for splitting."""
        return int(low_level_dataset.sequence_lengths.shape[0])

    @staticmethod
    def is_multimodal() -> bool:
        """Return whether records contain multimodal sequence modes."""
        return False

    @staticmethod
    def is_split_by_sequence() -> bool:
        """Return whether train/validation/test boundaries use sequence indices."""
        return True

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str,
        config: GPTDatasetConfig,
    ) -> IndexedDataReader:
        """Open a low-level reader for pre-cut records."""
        return IndexedDataReader(
            dataset_path,
            mmap=config.mmap_bin_files,
            reuse_index=config.reuse_idx,
        )

    def __len__(self) -> int:
        """Return the requested number of pre-cut records in this split."""
        return self.num_samples

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Read and shift one pre-cut record without dynamic concatenation."""
        sequence_id = int(self.indices[index])
        text = np.asarray(self.dataset[sequence_id], dtype=np.int64)
        if np.any(text == self._pad_token_id):
            raise ValueError("Pre-cut indexed Dataset records must be packed without padding")

        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].copy()
            labels = text[1:].copy()
        else:
            tokens = text.copy()
            labels = np.roll(text, shift=-1)
            labels[-1] = self._pad_token_id

        if np.any(tokens < 0) or np.any(tokens >= self.config.tokenizer.vocab_size):
            raise ValueError("An input token is out of bounds of the tokenizer vocabulary")

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        loss_mask = loss_mask.copy()
        loss_mask[labels == self._pad_token_id] = 0.0
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        raw_sample = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        if self.config.create_attention_mask:
            raw_sample["attention_mask"] = attention_mask
        return raw_sample


def _get_ltor_masks_and_position_ids(
    tokens: np.ndarray,
    config: GPTDatasetConfig,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """
    Build causal attention, loss, and position arrays for one token sequence.

    Args:
        tokens (torch.Tensor): The data tenor that holds the tokens from the dataset
        config (GPTDatasetConfig): The dataset config.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    seq_length = tokens.size
    eod_token = config.tokenizer.eod
    attention_mask = None
    if config.create_attention_mask:
        attention_mask = np.tril(np.ones((seq_length, seq_length)))[None, :, :]

    # Loss mask.
    loss_mask = np.ones(seq_length, dtype=np.float32)
    if config.eod_mask_loss:
        loss_mask[tokens == eod_token] = 0.0

    # Position ids.
    position_ids = np.arange(seq_length, dtype=np.int64)
    # We need to clone as the ids will be modifed based on batch index.
    if config.reset_position_ids:
        position_ids = position_ids.copy()

    if config.reset_position_ids or config.reset_attention_mask:
        # Find indices where EOD token is.
        eod_indices = position_ids[tokens == eod_token]
        # Detach indices from positions if going to modify positions.
        if config.reset_position_ids:
            eod_indices = eod_indices.copy()

        # Loop through EOD indices:
        prev_index = 0
        for eod_index in eod_indices:
            next_index = int(eod_index) + 1
            # Mask attention loss.
            if config.reset_attention_mask and attention_mask is not None:
                attention_mask[0, next_index:, :next_index] = 0
            # Reset positions.
            if config.reset_position_ids:
                position_ids[next_index:] -= next_index - prev_index
                prev_index = next_index

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids
