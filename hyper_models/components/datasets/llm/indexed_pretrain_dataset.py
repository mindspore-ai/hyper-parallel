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

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np

from hyper_models.components.datasets.llm.indexed_data_config import GPTDatasetConfig
from hyper_models.components.datasets.llm.indexed_data_reader import IndexedDataReader
from hyper_models.components.datasets.llm.indexed_sample_index import (
    build_document_sample_shuffle_indices,
)


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

        if not self.config.mock:
            self.unique_identifiers = type(self).build_unique_identifiers(
                self.dataset_path,
                self.num_samples,
                self.index_split,
                self.config,
            )

            # The original tokenizer exposes unique_identifiers. The fallback
            # keeps the current Hugging Face tokenizer usable until its wrapper
            # provides the same cache identity contract.
            self.unique_description = json.dumps(
                self.unique_identifiers,
                indent=4,
                default=lambda value: getattr(value, "unique_identifiers", repr(value)),
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
        return ["random_seed", "sequence_length", "split", "split_matrix", "tokenizer"]

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
        eod = 0

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
            attention_mask, loss_mask, position_ids = _build_ltor_masks_and_position_ids(
                tokens,
                eod,
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
        if self.dataset.sequence_lengths.dtype != np.int32:
            raise TypeError("Indexed sequence lengths must use int32")
        if self.indices.dtype != np.int32:
            raise TypeError("GPTDataset split indices must use int32")

        cache_directory = self.config.path_to_cache
        if cache_directory is None:
            cache_directory = os.path.join(
                self.dataset.path_prefix,
                "cache",
                f"{type(self).__name__}_indices",
            )
        cache_key = f"{self.unique_description_hash}-{type(self).__name__}"
        self.document_index, self.sample_index, self.shuffle_index = (
            build_document_sample_shuffle_indices(
                sequence_lengths=self.dataset.sequence_lengths,
                indices=self.indices,
                num_samples=self.num_samples,
                sequence_length=self.config.sequence_length,
                random_seed=self.config.random_seed,
                cache_directory=cache_directory,
                cache_key=cache_key,
                cache_description=self.unique_description,
            )
        )

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
        text, _ = self._query_document_sample_shuffle_indices(index)
        tokens = text[:-1]
        labels = text[1:]
        if np.any(tokens >= self.config.tokenizer.vocab_size):
            raise ValueError("An input token is out of bounds of the tokenizer vocabulary")

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _build_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
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

    def _query_document_sample_shuffle_indices(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble one sample, which may span several indexed sequences."""
        shuffled_index = int(self.shuffle_index[index])
        begin_document, begin_offset = self.sample_index[shuffled_index]
        end_document, end_offset = self.sample_index[shuffled_index + 1]

        document_ids = []
        sample_parts = []
        if begin_document == end_document:
            document_id = int(self.document_index[begin_document])
            document_ids.append(document_id)
            sample_parts.append(
                self.dataset.get(
                    document_id,
                    offset=int(begin_offset),
                    length=int(end_offset - begin_offset + 1),
                )
            )
        else:
            for document_position in range(int(begin_document), int(end_document) + 1):
                document_id = int(self.document_index[document_position])
                document_ids.append(document_id)
                offset = 0 if document_position > begin_document else int(begin_offset)
                length = None if document_position < end_document else int(end_offset + 1)
                sample_parts.append(self.dataset.get(document_id, offset=offset, length=length))

        return (
            np.asarray(np.concatenate(sample_parts), dtype=np.int64),
            np.asarray(document_ids, dtype=np.int64),
        )


class GPTFromMRDataset(_IndexedPretrainDataset):
    """Read pre-cut ``sequence_length + 1`` GPT records directly."""

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
        if self.dataset is None:
            raise ValueError("GPTFromMRDataset requires a low-level Dataset")
        if not isinstance(self.config, GPTDatasetConfig):
            raise ValueError("GPTFromMRDataset requires GPTDatasetConfig")
        if not self.config.skip_data_check:
            expected_length = self.config.sequence_length + 1
            if not np.all(self.dataset.sequence_lengths == expected_length):
                raise ValueError(
                    "All pre-cut records must have sequence_length + 1 tokens; "
                    f"expected {expected_length}, got {self.dataset.sequence_lengths.tolist()}"
                )
        if self.num_samples > len(self.dataset):
            raise ValueError(
                "Requested samples exceed the available pre-cut records: "
                f"requested {self.num_samples}, available {len(self.dataset)}"
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
        """Return the number of pre-cut records in the low-level Dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Read and shift one pre-cut record without dynamic concatenation."""
        text = np.asarray(self.dataset[index], dtype=np.int64)
        tokens = text[:-1]
        labels = text[1:]
        if np.any(tokens >= self.config.tokenizer.vocab_size):
            raise ValueError("An input token is out of bounds of the tokenizer vocabulary")

        if not self.config.create_attention_mask:
            return {"tokens": tokens, "labels": labels}

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _build_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
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
        return {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }


def _build_ltor_masks_and_position_ids(
    tokens: np.ndarray,
    eod_token: int,
    config: GPTDatasetConfig,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Build causal attention, loss, and position arrays for one token sequence."""
    sequence_length = tokens.size
    attention_mask = None
    if config.create_attention_mask:
        attention_mask = np.tril(np.ones((sequence_length, sequence_length)))[None, :, :]

    loss_mask = np.ones(sequence_length, dtype=np.float32)
    if config.eod_mask_loss:
        loss_mask[tokens == eod_token] = 0.0

    position_ids = np.arange(sequence_length, dtype=np.int64)
    if config.reset_position_ids:
        position_ids = position_ids.copy()

    if config.reset_position_ids or config.reset_attention_mask:
        eod_indices = position_ids[tokens == eod_token]
        if config.reset_position_ids:
            eod_indices = eod_indices.copy()

        previous_index = 0
        for eod_index in eod_indices:
            next_index = int(eod_index) + 1
            if config.reset_attention_mask and attention_mask is not None:
                attention_mask[0, next_index:, :next_index] = 0
            if config.reset_position_ids:
                position_ids[next_index:] -= next_index - previous_index
                previous_index = next_index

    if attention_mask is not None:
        attention_mask = attention_mask < 0.5
    return attention_mask, loss_mask, position_ids
