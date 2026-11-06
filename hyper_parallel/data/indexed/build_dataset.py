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
"""Build indexed Dataset splits with optional lazy initialization.

IndexedPretrainDatasetBuilder selects the Dataset type and build options;
IndexedDatasetSplitBuilder assembles train, validation, and test splits;
LazyDatasetProxy defers construction until a split is accessed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from functools import partial
from typing import Any, Literal, TypeAlias, cast

import numpy as np

from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.indexed.blend_dataset import BlendedDataset, SimpleBlendedDataset
from hyper_parallel.data.indexed.indexed_data_config import (
    GPTDatasetConfig,
    build_gpt_dataset_config,
    resolve_data_paths,
)
from hyper_parallel.data.indexed.indexed_pretrain_dataset import (
    GPTDataset,
    GPTFromMRDataset,
    MockGPTDataset,
)
from hyper_parallel.data.parallel import (
    DataLoaderParallelContext,
    build_dataset_for_dataloader,
)

logger = get_dataset_logger(__name__)

DataPath: TypeAlias = str | Sequence[str]
DatasetSplits: TypeAlias = tuple[Any | None, Any | None, Any | None]
InstructionSplitBuilder: TypeAlias = Callable[..., Sequence[Any | None]]
BlendMode: TypeAlias = Literal["no", "inter", "intra"]
SplitRange: TypeAlias = tuple[float, float]
SplitMatrix: TypeAlias = Sequence[SplitRange | None]


def build_indexed_dataset(
        *,
        data_path: DataPath | None = None,
        data_config: Mapping[str, Any],
        train_valid_test_num_samples: list[int] | tuple[int, int, int],
        dataloader_context: DataLoaderParallelContext | None = None,
) -> DatasetSplits:
    """Build indexed pretraining datasets through the dedicated builder.

    Args:
        data_path: Indexed prefix, directory, or weighted paths. May be omitted
            when mock data is enabled.
        data_config: Indexed pretraining build options.
        train_valid_test_num_samples: Trainer-derived train, validation, and test target sizes.
        dataloader_context: DataLoader ownership and synchronization policy.

    Returns:
        Train, validation, and test datasets.
    """
    builder = IndexedPretrainDatasetBuilder(
        data_path=data_path,
        data_config=data_config,
        train_valid_test_num_samples=train_valid_test_num_samples,
        dataloader_context=dataloader_context,
    )
    datasets = builder.build()
    return datasets


class IndexedPretrainDatasetBuilder:
    """Resolve indexed sources and assemble their pretraining or instruction splits."""

    def __init__(
            self,
            *,
            data_path: DataPath | None,
            data_config: Mapping[str, Any],
            train_valid_test_num_samples: Sequence[int],
            dataloader_context: DataLoaderParallelContext | None = None,
    ) -> None:
        """Store the inputs required by the indexed pretraining build.

        Args:
            data_path: Indexed prefix, directory, or weighted paths. May be
                omitted when mock data is enabled.
            data_config: Indexed pretraining build options.
            train_valid_test_num_samples: Trainer-derived target sizes.
            dataloader_context: DataLoader ownership and synchronization policy.
        """
        self.data_path = data_path
        self.data_config = data_config
        self.train_valid_test_num_samples = train_valid_test_num_samples
        self.dataloader_context = dataloader_context or DataLoaderParallelContext(
            data_index_cache=bool(data_config.get("data_index_cache", False))
        )
        self.split_builder = IndexedDatasetSplitBuilder(self.dataloader_context)

    def build(self) -> DatasetSplits:
        """Build train, validation, and test indexed datasets.

        Returns:
            Train, validation, and test datasets.
        """
        # Mock Datasets generate samples without filesystem discovery.
        mock_data = bool(self.data_config.get("mock_data", False))
        logger.debug("Building indexed Dataset: mock=%s, data_path=%s", mock_data, self.data_path)
        if mock_data:
            data_paths = []
        else:
            if self.data_path is None:
                raise ValueError("data_path is required when mock_data is false")

            data_paths = resolve_data_paths(
                self.data_path,
                distributed_walk=bool(self.data_config.get("distributed_walk", True)),
            )

        config = build_gpt_dataset_config(data_paths, self.data_config)
        logger.debug(
            "Indexed Dataset config: simple_blend=%s, lazy=%s, shared_blend=%s, independent_blends=%s",
            config.simple_blend, config.data_lazy_load, bool(config.blend), bool(config.blend_per_split),
        )

        if bool(self.data_config.get("is_instruction_dataset", False)):
            logger.debug("Selecting instruction Dataset pipeline")
            dataset_splits = self._build_instruction_dataset_splits(config)
        else:
            dataset_type = self._select_dataset_type(config)
            logger.debug("Selected indexed Dataset: type=%s, split_mode=%s", dataset_type.__name__, config.simple_blend)
            dataset_splits = self.split_builder.build(
                dataset_type=dataset_type,
                train_valid_test_num_samples=self.train_valid_test_num_samples,
                config=config,
                blend_mode=config.simple_blend,
            )
        split_types = tuple(type(split).__name__ if split is not None else None for split in dataset_splits)
        logger.debug("Built indexed Dataset splits=%s", split_types)
        return dataset_splits

    def _build_instruction_dataset_splits(self, config: GPTDatasetConfig) -> DatasetSplits:
        """Build packed instruction splits through the configured implementation.

        The concrete packed-dataset implementation remains outside this provider.

        Raises:
            NotImplementedError: If no instruction Dataset builder is configured.
            ValueError: If the configured builder does not return three splits.
        """
        instruction_builder = self.data_config.get("instruction_dataset_builder")
        if not callable(instruction_builder):
            raise NotImplementedError(
                "is_instruction_dataset=True requires an instruction_dataset_builder"
            )

        instruction_split_builder: InstructionSplitBuilder = instruction_builder

        data_prefix = config.blend if config.blend is not None else self.data_path
        datasets = instruction_split_builder(
            data_prefix=data_prefix,
            splits_string=config.split,
            train_valid_test_num_samples=self.train_valid_test_num_samples,
            seq_length=config.sequence_length,
            seed=config.random_seed,
            tokenizer=config.tokenizer,
        )
        if not isinstance(datasets, Sequence) or len(datasets) != 3:
            raise ValueError("instruction_dataset_builder must return train, validation, and test")

        dataset_splits = (datasets[0], datasets[1], datasets[2])
        return dataset_splits

    @staticmethod
    def _select_dataset_type(config: GPTDatasetConfig) -> type:
        """Select GPT, Mock GPT, or MR GPT Dataset."""
        if config.mock:
            return MockGPTDataset

        if config.is_dataset_from_mr:
            return GPTFromMRDataset

        return GPTDataset


class IndexedDatasetSplitBuilder:
    """Build indexed Dataset splits with one shared distributed context."""

    def __init__(self, dataloader_context: DataLoaderParallelContext) -> None:
        """Store the DataLoader ownership policy once for all build steps."""
        self.dataloader_context = dataloader_context

    def build(
        self,
        dataset_type: type,
        train_valid_test_num_samples: Sequence[int] | None,
        config: GPTDatasetConfig,
        blend_mode: BlendMode,
    ) -> DatasetSplits:
        """Build mock, shared-blend, or independent-blend Dataset splits."""
        if train_valid_test_num_samples is None:
            raise ValueError("train_valid_test_num_samples must be configured for indexed Datasets")

        if len(train_valid_test_num_samples) != len(_Split):
            raise ValueError("train_valid_test_num_samples must contain exactly three values")

        if any(size < 0 for size in train_valid_test_num_samples):
            raise ValueError("train_valid_test_num_samples must be non-negative")

        if blend_mode not in {"no", "inter", "intra"}:
            raise ValueError(f"Unsupported blend mode: {blend_mode!r}")

        if blend_mode != "no" and not config.is_dataset_from_mr:
            raise ValueError("inter/intra blending requires is_dataset_from_mr=True")

        logger.debug(
            "Building indexed splits: dataset=%s, num_samples=%s, mock=%s, blend_mode=%s, lazy=%s",
            dataset_type.__name__,
            tuple(train_valid_test_num_samples),
            config.mock,
            blend_mode,
            config.data_lazy_load,
        )
        if config.mock:
            dataset_splits = self._build_mock_dataset_splits(dataset_type, train_valid_test_num_samples, config)
            return dataset_splits

        if config.blend:
            if config.split_matrix is None:
                raise ValueError("A shared blend requires blend and split_matrix")

            dataset_splits = self._build_splits_from_blend(
                dataset_type, config.blend, config.split_matrix, train_valid_test_num_samples, config, blend_mode
            )
            return dataset_splits

        if config.blend_per_split:
            dataset_splits = self._build_splits_from_independent_blends(
                dataset_type, train_valid_test_num_samples, config, blend_mode
            )
            return dataset_splits

        raise ValueError("One of blend or blend_per_split must be configured")

    def _build_mock_dataset_splits(
        self,
        dataset_type: type,
        sizes: Sequence[int],
        config: GPTDatasetConfig,
    ) -> DatasetSplits:
        """Build generated train, validation, and test Datasets."""
        datasets = []
        for split, size in zip(_Split, sizes):
            if size == 0:
                datasets.append(None)
                continue

            dataset_factory = partial(dataset_type, None, None, None, int(size), split, config)
            mock_dataset = self._build_dataset_for_dataloader(dataset_factory, barrier_needed=False)
            datasets.append(mock_dataset)
        dataset_splits = cast(DatasetSplits, tuple(datasets))
        return dataset_splits

    def _build_splits_from_blend(
        self,
        dataset_type: type,
        blend: Sequence[str],
        split_matrix: SplitMatrix,
        sizes: Sequence[int],
        config: GPTDatasetConfig,
        blend_mode: BlendMode,
    ) -> DatasetSplits:
        """
        Parse one blend and dispatch it to the single- or multiple-source path.
        Low level: Index -> Mid level: GPTDataset -> Final: Blend.
        """
        if len(blend) == 1:
            dataset_splits = self._build_mid_level_dataset_splits(dataset_type, blend[0], split_matrix, sizes, config)
            return dataset_splits

        prefixes, weights, sizes_per_dataset = _get_prefixes_weights_and_sizes_for_blend(
            blend, sizes, blend_mode, config.dataset_margin
        )
        logger.debug(
            "Resolved blend: mode=%s, sources=%d, first_weight=%s, first_source_sizes=%s",
            blend_mode,
            len(prefixes),
            weights[0],
            sizes_per_dataset[0],
        )
        dataset_splits = self._build_multiple_source_splits(
            dataset_type, prefixes, weights, sizes_per_dataset, split_matrix, config, blend_mode
        )
        return dataset_splits

    def _build_splits_from_independent_blends(
        self,
        dataset_type: type,
        sizes: Sequence[int],
        config: GPTDatasetConfig,
        blend_mode: BlendMode,
    ) -> DatasetSplits:
        """Build each split from its own data-source distribution."""
        if config.blend_per_split is None or len(config.blend_per_split) != len(_Split):
            raise ValueError("Independent blends require one blend per split")

        datasets = []
        for split_index, blend in enumerate(config.blend_per_split):
            if not blend:
                datasets.append(None)
                continue

            split_matrix: list[SplitRange | None] = [None] * len(_Split)
            split_matrix[split_index] = (0.0, 1.0)
            split_sizes = [0] * len(_Split)
            split_sizes[split_index] = int(sizes[split_index])
            split_datasets = self._build_splits_from_blend(
                dataset_type, blend, split_matrix, split_sizes, config, blend_mode
            )
            datasets.append(split_datasets[split_index])
        dataset_splits = cast(DatasetSplits, tuple(datasets))
        return dataset_splits

    def _build_multiple_source_splits(
        self,
        dataset_type: type,
        prefixes: Sequence[str],
        weights: Sequence[float],
        sizes_per_dataset: Sequence[Sequence[int]],
        split_matrix: SplitMatrix,
        config: GPTDatasetConfig,
        blend_mode: BlendMode,
    ) -> DatasetSplits:
        """Build each source's mid-level splits and assemble top-level blends.

        Example with two sources::

            prefixes = ["/data/books", "/data/code"]
            sizes_per_dataset = [[80, 10, 0], [20, 5, 0]]

            Offline preparation, tokenize and optionally pre-pack raw records:
              books.jsonl -> tokenizer -> token IDs -> pack to 2049-token records
                          -> /data/books.idx + /data/books.bin
              code.jsonl  -> tokenizer -> token IDs -> pack to 2049-token records
                          -> /data/code.idx + /data/code.bin

            Low-level, open one IndexedDataReader per prefix and read one sequence (uint16 example):
              books.idx: sequence_lengths[1] = 2049, sequence_pointers[1] = 4098
              books_reader[1] -> books.bin[offset=4098, count=2049] -> text token IDs

            Mid-level, organize tokens into training records (tokens, labels, and masks),
            with one GPTDataset/GPTFromMRDataset per source and split:
              books_train[0] -> split indices[0] = 1 -> books_reader[1]
                             -> tokens = text[:-1], labels = text[1:]
              datasets_per_source = [
                  (books_train[80], books_valid[10], None),
                  (code_train[20],  code_valid[5],   None),
              ]

            Transpose source-major tuples into split-major components:
              train components = [books_train, code_train], total size = 100
              valid components = [books_valid, code_valid], total size = 15
              test components  = [], size = 0

            Blend-level, select the source and source-local sample for each train index, for example:
              dataset_index        = [0, 1, ...]
              dataset_sample_index = [0, 0, ...]
              global sample 0 -> (dataset_id=0, sample_id=0) -> books_train[0]
              global sample 1 -> (dataset_id=1, sample_id=0) -> code_train[0]

            ``BlendedDataset[i]`` reads ``dataset_index[i]`` and ``dataset_sample_index[i]``;
            ``SimpleBlendedDataset[i]`` reads ``_sample_locations[i] = (dataset_id, sample_id)``.
            Both return ``datasets[dataset_id][sample_id]``.

            final result = (blended_train[100], blended_valid[15], None)
        """
        # pylint: disable=too-many-locals
        datasets_per_source = []
        logger.debug("Building %d indexed Dataset sources", len(prefixes))
        # TODO: Add a separate cache-prebuild phase that assigns source_index % world_size to each rank,
        # synchronizes failures, and atomically writes a completion manifest before Dataset construction.

        # books -> (books_train, books_valid, None), then code -> (code_train, code_valid, None).
        for prefix, sizes_for_source in zip(prefixes, sizes_per_dataset):
            source_datasets = self._build_mid_level_dataset_splits(
                dataset_type, prefix, split_matrix, sizes_for_source, config
            )
            datasets_per_source.append(source_datasets)

        blended_sizes = [sum(split_sizes) for split_sizes in zip(*sizes_per_dataset)]
        blended_splits = []
        for split_index, (split_range, size) in enumerate(zip(split_matrix, blended_sizes)):
            if split_range is None or size == 0:
                blended_splits.append(None)
                continue

            component_datasets = [datasets[split_index] for datasets in datasets_per_source]
            missing_datasets = [dataset is None for dataset in component_datasets]
            if any(missing_datasets) and not all(missing_datasets):
                raise ValueError("Enabled sources must either all build or all remain None on a rank")

            component_datasets = [dataset for dataset in component_datasets if dataset is not None]
            if blend_mode == "no":
                dataset_factory = partial(BlendedDataset, component_datasets, weights, int(size), config)
            else:
                dataset_factory = partial(SimpleBlendedDataset, component_datasets, int(size), blend_mode)
            blended_dataset = self._build_dataset_for_dataloader(dataset_factory, barrier_needed=True)
            blended_splits.append(blended_dataset)

        dataset_splits = cast(DatasetSplits, tuple(blended_splits))
        split_types = tuple(type(split).__name__ if split is not None else None for split in dataset_splits)
        logger.debug("Built blended splits=%s", split_types)
        return dataset_splits

    def _build_mid_level_dataset_splits(
        self,
        dataset_type: type,
        dataset_path: str,
        split_matrix: SplitMatrix,
        sizes_for_source: Sequence[int],
        config: GPTDatasetConfig,
    ) -> DatasetSplits:
        """Build each mid-level split from one low-level indexed Dataset."""
        # pylint: disable=too-many-locals
        if (
            config.data_lazy_load
            and self.dataloader_context.distributed_enabled
            and not (self.dataloader_context.data_index_cache or self.dataloader_context.build_on_rank())
        ):
            empty_splits = (None, None, None)
            return empty_splits

        low_level_dataset = None
        split_indices: list[np.ndarray | None] = [None] * len(split_matrix)
        if not config.data_lazy_load:
            low_level_dataset, split_indices = _build_low_level_dataset_and_split_indices(
                dataset_type, dataset_path, split_matrix, config
            )

        datasets = []
        for split, split_range, indices, size in zip(_Split, split_matrix, split_indices, sizes_for_source):
            if split_range is None or size == 0:
                datasets.append(None)
                continue

            size = int(size)
            if config.data_lazy_load:
                lazy_builder = self._build_lazy_mid_level_dataset_split
                dataset_factory = partial(lazy_builder, dataset_type, dataset_path, split, split_range, size, config)
                unique_identifiers = dataset_type.build_unique_identifiers(dataset_path, size, split, config)
                mid_level_dataset = LazyDatasetProxy(
                    dataset_factory,
                    unique_identifiers=unique_identifiers,
                    length=size if dataset_type is GPTFromMRDataset else None,
                )
            else:
                dataset_factory = partial(dataset_type, low_level_dataset, dataset_path, indices, size, split, config)
                mid_level_dataset = self._build_dataset_for_dataloader(
                    dataset_factory, barrier_needed=not config.is_dataset_from_mr
                )
            datasets.append(mid_level_dataset)
        dataset_splits = cast(DatasetSplits, tuple(datasets))
        return dataset_splits

    def _build_lazy_mid_level_dataset_split(
        self,
        dataset_type: type,
        dataset_path: str,
        split: _Split,
        split_range: SplitRange,
        size: int,
        config: GPTDatasetConfig,
    ) -> Any | None:
        """Apply distributed construction when a lazy split is first accessed."""

        def _dataset_factory() -> Any:
            """Build the low-level Dataset and one lazy split on first access."""
            low_level_dataset, split_indices = _build_low_level_dataset_and_split_indices(
                dataset_type, dataset_path, [split_range], config
            )
            mid_level_dataset = dataset_type(low_level_dataset, dataset_path, split_indices[0], size, split, config)
            return mid_level_dataset

        local_dataset = self._build_dataset_for_dataloader(_dataset_factory, barrier_needed=False)
        return local_dataset

    def _build_dataset_for_dataloader(
        self,
        dataset_factory: Callable[[], Any],
        *,
        barrier_needed: bool,
    ) -> Any | None:
        """Build one Dataset with the shared DataLoader context."""
        local_dataset = build_dataset_for_dataloader(
            dataset_factory,
            self.dataloader_context,
            barrier_needed=barrier_needed,
        )
        return local_dataset


class LazyDatasetProxy:
    """Construct and cache a Dataset when it is first accessed."""

    def __init__(
            self,
            dataset_factory: Callable[[], Any],
            *,
            unique_identifiers: Any,
            length: int | None = None,
    ) -> None:
        """Store lightweight metadata without constructing the Dataset.

        Args:
            dataset_factory: Callable that constructs the real Dataset on first
                sample access.
            unique_identifiers: Stable Dataset cache identity assembled from
                construction inputs.
            length: Known logical sample count, if available without construction.
        """
        self._dataset_factory = dataset_factory
        self._unique_identifiers = unique_identifiers
        self._length = length
        self._dataset = None

    def __len__(self) -> int:
        """Return a known length, constructing the Dataset only when necessary."""
        if self._length is not None:
            return self._length
        dataset = self._get_or_build_dataset()
        dataset_length = len(dataset)
        return dataset_length

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Construct the Dataset if necessary and read one sample."""
        dataset = self._get_or_build_dataset()
        sample = dataset[index]
        return sample

    @property
    def unique_identifiers(self) -> Any:
        """Return the cache identity assembled before Dataset construction."""
        return self._unique_identifiers

    def _get_or_build_dataset(self) -> Any:
        """Construct the Dataset once and reuse it for later accesses."""
        if self._dataset is None:
            identifiers = self._unique_identifiers if isinstance(self._unique_identifiers, Mapping) else {}
            logger.debug(
                "Initializing lazy Dataset: class=%s, path=%s, split=%s, samples=%s",
                identifiers.get("class"), identifiers.get("dataset_path"), identifiers.get("index_split"),
                identifiers.get("num_samples"),
            )
            self._dataset = self._dataset_factory()
        return self._dataset


class _Split(Enum):
    """Stable train, validation, and test identifiers used by Dataset caches."""

    TRAIN = 0
    VALID = 1
    TEST = 2


def _get_prefixes_weights_and_sizes_for_blend(
    blend: Sequence[str],
    target_num_samples_per_split: Sequence[int],
    blend_mode: BlendMode,
    dataset_margin: float,
) -> tuple[list[str], list[float], list[list[int]]]:
    """Determine each source's contribution to the blended Dataset splits.

     Args:
         blend: Alternating ``weight, prefix`` pairs for each data source.
         target_num_samples_per_split: Requested ``[train, validation, test]`` sample counts.
         blend_mode: ``"no"`` for normalized weighted blending; ``"inter"`` or ``"intra"`` for simple blending.
         dataset_margin: Extra sample margin applied only to normalized weighted blending.

     Returns:
         ``(prefixes, weights, sizes_per_dataset)``, where ``sizes_per_dataset[source][split]`` is the number of
         samples requested from one source for the train, validation, or test split.

    Example:
        Regular: ``["30", "/data/a", "70", "/data/b"]`` with sizes ``[1000, 100, 0]`` and margin ``1.005``
        -> ``(["/data/a", "/data/b"], [0.3, 0.7], [[302, 31, 0], [704, 71, 0]])``.
        Simple: ``["1", "/data/a", "1", "/data/b"]`` with sizes ``[1000, 100, 0]``
        -> ``(["/data/a", "/data/b"], [1, 1], [[500, 50, 0], [500, 50, 0]])``.
    """
    if len(blend) % 2 != 0:
        raise ValueError("Dataset blend must contain alternating weight/prefix pairs")

    # standard blend
    if blend_mode == "no":
        weights, prefixes = zip(*[(float(blend[index]), blend[index + 1].strip()) for index in range(0, len(blend), 2)])
        weight_sum = float(sum(weights))
        weights = [weight / weight_sum for weight in weights]
        sizes_per_dataset = [
            [
                int(math.ceil(target_num_samples * weight * dataset_margin))
                for target_num_samples in target_num_samples_per_split
            ]
            for weight in weights
        ]
        return prefixes, weights, sizes_per_dataset

    # inter/intra blend
    weights, prefixes = zip(*[(int(blend[index]), blend[index + 1].strip()) for index in range(0, len(blend), 2)])
    sizes_per_dataset = [
        [int(math.ceil(target_num_samples / len(weights))) for target_num_samples in target_num_samples_per_split]
        for _ in weights
    ]
    return prefixes, weights, sizes_per_dataset


def _build_low_level_dataset_and_split_indices(
    dataset_type: type,
    dataset_path: str,
    split_matrix: SplitMatrix,
    config: GPTDatasetConfig,
) -> tuple[Any, list[np.ndarray | None]]:
    """Build one low-level Dataset and calculate its requested split indices."""
    low_level_dataset = dataset_type.build_low_level_dataset(dataset_path, config)
    num_elements = dataset_type.numel_low_level_dataset(low_level_dataset)

    split_indices = []
    for split_range in split_matrix:
        if split_range is None:
            split_indices.append(None)
            continue

        begin = int(round(split_range[0] * num_elements))
        end = int(round(split_range[1] * num_elements))
        split_indice = np.arange(start=begin, stop=end, step=1, dtype=np.int32)
        split_indices.append(split_indice)

    # (low_level_dataset, [train_indices, valid_indices, test_indices]).
    dataset_and_split_indices = (low_level_dataset, split_indices)
    return dataset_and_split_indices
