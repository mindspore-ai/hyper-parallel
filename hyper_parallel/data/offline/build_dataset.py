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
"""Build offline training Datasets from indexed tokens and prepared records."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from functools import partial
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import torch.distributed as dist

from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.offline.blended_dataset import BlendedDataset, SimpleBlendedDataset
from hyper_parallel.data.offline.data_config import (
    GPTDatasetConfig,
    build_gpt_dataset_config,
    resolve_dataset_type,
)
from hyper_parallel.data.offline.dataset import GPTDataset, MockGPTDataset, SFTFromMRDataset
from hyper_parallel.data.parallel import (
    DataLoaderParallelContext,
    build_dataset_for_dataloader,
)

DataPath: TypeAlias = str | Sequence[str]
DatasetSplits: TypeAlias = tuple[Any | None, Any | None, Any | None]
SplitBuilder: TypeAlias = Callable[..., DatasetSplits]
InstructionSplitBuilder: TypeAlias = Callable[..., Sequence[Any | None]]
BlendMode: TypeAlias = Literal["no", "inter", "intra"]
SplitRange: TypeAlias = tuple[float, float]
SplitMatrix: TypeAlias = Sequence[SplitRange | None]

logger = get_dataset_logger(__name__)


def build_indexed_dataset(
        *,
        data_path: DataPath = None,
        data_config: Mapping[str, Any],
        train_valid_test_num_samples: Sequence[int],
        dataloader_context: DataLoaderParallelContext | None = None,
) -> DatasetSplits:
    """Build indexed pretraining datasets through the dedicated builder.

    Args:
        data_path: Indexed prefix, directory, or weighted paths. May be omitted
            when mock data is enabled.
        data_config: Indexed pretraining build options.
        train_valid_test_num_samples: Trainer-derived target sizes.
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


class _Split(Enum):
    """Stable train, validation, and test identifiers used by Dataset caches."""

    TRAIN = 0
    VALID = 1
    TEST = 2


def _walk_data_directories(directories: Sequence[str], file_suffix: str) -> list[str]:
    """Recursively find source files below the assigned directories."""
    file_paths = []
    for directory in directories:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(file_suffix):
                    path = os.path.abspath(os.path.join(root, filename))
                    file_paths.append(os.path.splitext(path)[0] if file_suffix == ".bin" else path)
    return file_paths


def _discover_indexed_paths(
    current_path: str,
    rank: int,
    world_size: int,
    distributed_walk: bool,
    file_suffix: str,
) -> list[str]:
    """Discover indexed prefixes under one directory."""
    if not distributed_walk:
        return _walk_data_directories([current_path], file_suffix)

    root_layout = None
    if rank == 0:
        subdirectories = []
        top_files = []
        for name in os.listdir(current_path):
            path = os.path.join(current_path, name)
            if os.path.isdir(path):
                subdirectories.append(path)
            elif name.endswith(file_suffix):
                top_files.append(os.path.splitext(path)[0] if file_suffix == ".bin" else path)
        root_layout = (subdirectories, top_files)

    gathered_layouts: list[tuple[list[str], list[str]] | None] = [None] * world_size
    dist.all_gather_object(gathered_layouts, root_layout)
    rank_zero_layout = gathered_layouts[0]
    if rank_zero_layout is None:
        raise ValueError("Rank 0 did not provide the indexed directory layout")
    subdirectories, file_paths = rank_zero_layout
    local_files = _walk_data_directories(subdirectories[rank::world_size], file_suffix)
    gathered_files: list[list[str] | None] = [None] * world_size
    dist.all_gather_object(gathered_files, local_files)
    for rank_files in gathered_files:
        if rank_files:
            file_paths.extend(rank_files)
    return file_paths


def resolve_data_paths(
    data_path: str | Sequence[str],
    *,
    distributed_walk: bool = True,
    file_suffix: str = ".bin",
) -> list[str]:
    """Discover independent data files and return source paths with optional weights.

    Args:
        data_path: One indexed prefix, Arrow file, or sequence of directories/source paths.
        distributed_walk: Whether initialized ranks divide directory walking.
        file_suffix: Dataset storage suffix: .bin returns prefixes, .arrow keeps full file paths.

    Returns:
        One prefix directly, or alternating weight/prefix values for multiple sources. If ``data_path`` is not a
        collection of discoverable indexed paths, return it unchanged so an explicit blend can be parsed.
    """
    # 1. Normalize one path and multiple paths to the same traversal form.
    data_paths = [data_path] if isinstance(data_path, str) else list(data_path)

    # 2. Use distributed walking only after the process group is available.
    # Before initialization, one process walks all configured directories.
    try:
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
    except (RuntimeError, ValueError):
        rank, world_size = 0, 1
    use_distributed_walk = distributed_walk and world_size > 1

    # 3. Accept a direct prefix, or recursively discover prefixes in a directory.
    file_paths: list[str] = []
    for current_path in data_paths:
        if file_suffix == ".arrow" and current_path.endswith(file_suffix) and os.path.isfile(current_path):
            file_paths.append(current_path)
            continue
        if file_suffix == ".bin" and os.path.isfile(current_path + ".bin"):
            file_paths.append(current_path)
            continue
        if not os.path.isdir(current_path):
            # Keep args.data_path unchanged when filesystem discovery
            # returns no files. This preserves explicit forms such as
            # ["30", prefix_a, "70", prefix_b] for the blend parser.
            original_paths = list(data_paths)
            return original_paths

        file_paths.extend(_discover_indexed_paths(current_path, rank, world_size, use_distributed_walk, file_suffix))

    if not file_paths:
        raise ValueError(f"data_path {data_paths!r} has no data file ending with {file_suffix}")

    if len(file_paths) == 1:
        return file_paths

    if file_suffix == ".arrow":
        # Every rank must blend files in the same order after directory discovery.
        file_paths.sort()
    else:
        # 4. Prefer the numeric shard order encoded by the corpus filenames.
        # Unknown naming schemes remain usable in their original walk order.
        try:
            def numeric_shard_key(path: str) -> tuple[int, int]:
                """Extract numeric corpus and shard identifiers from a path."""
                path_parts = os.path.basename(path).split("_")
                return int(path_parts[2]), int(path_parts[-1])

            file_paths.sort(key=numeric_shard_key)
        except (IndexError, TypeError, ValueError):
            logger.warning("Cannot sort indexed files with the numeric filename rule; keeping walk order")

    # 5. Automatically discovered sources use equal blend weights.
    weighted_paths = [value for path in file_paths for value in ("1", path)]
    return weighted_paths


class LazyDatasetProxy:
    """Construct and cache a Dataset when it is first accessed."""

    def __init__(
            self,
            dataset_factory: Callable[[], Any],
            *,
            unique_identifiers: Any,
    ) -> None:
        """Store lightweight metadata without constructing the Dataset.

        Args:
            dataset_factory: Callable that constructs the real Dataset on first
                sample access.
            unique_identifiers: Stable Dataset cache identity assembled from
                construction inputs.
        """
        self._dataset_factory = dataset_factory
        self._unique_identifiers = unique_identifiers
        self._dataset = None

    def __len__(self) -> int:
        """Initialize the Dataset and return its length."""
        return len(self._get_dataset())

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Construct the Dataset if necessary and read one sample."""
        dataset = self._get_dataset()
        sample = dataset[index]
        return sample

    @property
    def unique_identifiers(self) -> Any:
        """Initialize the Dataset and return its cache identity."""
        return self._get_dataset().unique_identifiers

    def _get_dataset(self) -> Any:
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

        if config.mock != (dataset_type is MockGPTDataset):
            raise ValueError("mock_data must match the selected MockGPTDataset class")

        if not config.mock and (blend_mode == "no") != (dataset_type is GPTDataset):
            raise ValueError("simple_blend='no' is only for GPTDataset; other Datasets require inter/intra")

        if dataset_type is GPTDataset:
            # Reject invalid options before cache builders and readers can diverge.
            GPTDataset.validate_config(config)

        logger.debug(
            "Building indexed splits: dataset=%s, num_samples=%s, mock=%s, blend_mode=%s, lazy=%s",
            dataset_type.__name__,
            tuple(train_valid_test_num_samples),
            config.mock,
            blend_mode,
            config.data_lazy_load,
        )
        if config.mock:
            return self._build_mock_dataset_splits(dataset_type, train_valid_test_num_samples, config)

        if config.blend:
            if config.split_matrix is None:
                raise ValueError("A shared blend requires blend and split_matrix")

            return self._build_splits_from_blend(
                dataset_type, config.blend, config.split_matrix, train_valid_test_num_samples, config, blend_mode
            )

        if config.blend_per_split:
            return self._build_splits_from_independent_blends(
                dataset_type, train_valid_test_num_samples, config, blend_mode
            )
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
        return cast(DatasetSplits, tuple(datasets))

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
            return self._build_mid_level_dataset_splits(dataset_type, blend[0], split_matrix, sizes, config)

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
        return cast(DatasetSplits, tuple(datasets))

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

            Mid-level, organize tokens into training records (input_ids, labels, and masks),
            with one GPTDataset/GPTFromMRDataset per source and split:
              books_train[0] -> split indices[0] = 1 -> books_reader[1]
                             -> input_ids = text[:-1], labels = text[1:]
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
            ``SimpleBlendedDataset[i]`` reads ``_locations[i] = (dataset_id, sample_id)``.
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
            and not (
                self.dataloader_context.data_index_cache
                or self.dataloader_context.build_on_rank()
            )
        ):
            return cast(DatasetSplits, (None, None, None))

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
                mid_level_dataset = LazyDatasetProxy(dataset_factory, unique_identifiers=unique_identifiers)
            else:
                dataset_factory = partial(dataset_type, low_level_dataset, dataset_path, indices, size, split, config)
                mid_level_dataset = self._build_dataset_for_dataloader(
                    dataset_factory, barrier_needed=dataset_type is GPTDataset
                )
            datasets.append(mid_level_dataset)
        return cast(DatasetSplits, tuple(datasets))

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


class IndexedPretrainDatasetBuilder:
    """Build GPT pretraining datasets from indexed token files.

    Its ``build`` method keeps the indexed provider flow visible while each
    private step can be implemented and reviewed independently.
    """

    def __init__(
            self,
            *,
            data_path: DataPath,
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
        # Discover indexed files and add their default blend weights.
        # Mock Datasets generate their samples and do not need filesystem
        # discovery. Real indexed Datasets still require an explicit path.
        mock_data = bool(self.data_config.get("mock_data", False))
        logger.debug("Building indexed Dataset: mock=%s, data_path=%s", mock_data, self.data_path)
        if mock_data:
            data_paths = []
        else:
            if self.data_path is None:
                raise ValueError("data_path is required when mock_data is false")

            data_paths = self._resolve_data_paths()

        # Normalize Dataset options. Sample counts remain separate and are
        # passed directly to the split builder.
        config = self._build_gpt_dataset_config(data_paths)
        logger.debug(
            "Indexed Dataset config: simple_blend=%s, lazy=%s, shared_blend=%s, independent_blends=%s",
            config.simple_blend, config.data_lazy_load, bool(config.blend), bool(config.blend_per_split),
        )

        # Follow the Provider boundary: instruction and pretraining datasets
        # share one public entry but own separate construction pipelines.
        dataset_splits = self._build_dataset_splits(config)
        split_types = tuple(type(split).__name__ if split is not None else None for split in dataset_splits)
        logger.debug("Built indexed Dataset splits=%s", split_types)
        return dataset_splits

    def _build_dataset_splits(self, config: GPTDatasetConfig) -> DatasetSplits:
        """Select the instruction or indexed-pretraining Dataset pipeline."""
        if bool(self.data_config.get("is_instruction_dataset", False)):
            logger.debug("Selecting instruction Dataset pipeline")
            dataset_splits = self._build_instruction_dataset_splits(config)
            return dataset_splits

        dataset_splits = self._build_pretrain_dataset_splits(config)
        return dataset_splits

    def _build_pretrain_dataset_splits(self, config: GPTDatasetConfig) -> DatasetSplits:
        """Build GPT pretraining splits through the selected Dataset and blend types."""
        # Select GPTDataset / MockGPTDataset / GPTFromMRDataset.
        dataset_type = self._select_dataset_type(config)

        # Select the standard or simple blended Dataset builder.
        split_builder = self._select_split_builder(config)
        logger.debug("Selected indexed Dataset: type=%s, split_mode=%s", dataset_type.__name__, config.simple_blend)

        # Build train, validation, and test datasets.
        datasets = split_builder(
            dataset_type=dataset_type,
            train_valid_test_num_samples=self.train_valid_test_num_samples,
            config=config,
        )
        return datasets

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

    def _resolve_data_paths(self) -> list[str]:
        """Discover source paths for the selected offline Dataset type."""
        if self.data_path is None:
            raise ValueError("data_path is required when mock_data is false")

        dataset_type = resolve_dataset_type(self.data_config["low_level_dataset"])
        data_paths = resolve_data_paths(
            self.data_path,
            distributed_walk=bool(self.data_config["distributed_walk"]),
            file_suffix=".arrow" if issubclass(dataset_type, SFTFromMRDataset) else ".bin",
        )
        return data_paths

    def _build_gpt_dataset_config(self, data_paths: Sequence[str]) -> GPTDatasetConfig:
        """Delegate GPT Dataset option normalization."""
        config = build_gpt_dataset_config(data_paths, self.data_config)
        return config

    @staticmethod
    def _select_dataset_type(config: GPTDatasetConfig) -> type:
        """Select GPT, Mock GPT, or MR GPT Dataset."""
        return config.low_level_dataset

    def _select_split_builder(self, config: GPTDatasetConfig) -> SplitBuilder:
        """
        Select the standard or simple train/valid/test builder.
            "no": BlendedDataset
            "inter/intra": SimpleBlendedDataset
        """
        if config.simple_blend == "no":
            split_builder = partial(self.split_builder.build, blend_mode="no")
            return split_builder

        if config.simple_blend not in {"inter", "intra"}:
            raise ValueError("simple_blend must be one of 'no', 'inter', or 'intra'; "
                             f"got {config.simple_blend!r}"
                             )

        split_builder = partial(self.split_builder.build, blend_mode=config.simple_blend)
        return split_builder
