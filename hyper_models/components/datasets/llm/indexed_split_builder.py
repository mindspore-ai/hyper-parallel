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
"""Build train, validation, and test Datasets from indexed data blends."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from enum import Enum
from functools import partial
from typing import Any, Literal, TypeAlias, cast

import numpy as np

from hyper_models.components.datasets.dataset_logging import get_dataset_logger
from hyper_models.components.datasets.llm.indexed_blended_dataset import BlendedDataset
from hyper_models.components.datasets.llm.indexed_data_config import GPTDatasetConfig
from hyper_models.components.datasets.llm.indexed_lazy_dataset import LazyDatasetProxy
from hyper_models.components.datasets.llm.indexed_simple_blended_dataset import SimpleBlendedDataset
from hyper_models.components.datasets.parallel import DatasetParallelContext, build_distributed_dataset

logger = get_dataset_logger(__name__)

BlendMode: TypeAlias = Literal["no", "inter", "intra"]
DatasetSplits: TypeAlias = tuple[Any | None, Any | None, Any | None]
SplitRange: TypeAlias = tuple[float, float]
SplitMatrix: TypeAlias = Sequence[SplitRange | None]


class _Split(Enum):
   """Stable train, validation, and test identifiers used by Dataset caches."""

   train = 0
   valid = 1
   test = 2


def _get_prefixes_weights_and_sizes_for_blend(
       blend: Sequence[str], target_num_samples_per_split: Sequence[int],
       blend_mode: BlendMode, dataset_margin: float,
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
       returns ``(["/data/a", "/data/b"], [0.3, 0.7], [[302, 31, 0], [704, 71, 0]])``.
       Simple: ``["1", "/data/a", "1", "/data/b"]`` with sizes ``[1000, 100, 0]``
       returns ``(["/data/a", "/data/b"], [1, 1], [[500, 50, 0], [500, 50, 0]])``.
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
       [
           int(math.ceil(target_num_samples / len(weights)))
           for target_num_samples in target_num_samples_per_split
       ]
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

   dataset_and_split_indices = (low_level_dataset, split_indices)
   return dataset_and_split_indices


class IndexedDatasetSplitBuilder:
   """Build indexed Dataset splits with one shared distributed context."""

   def __init__(self, parallel_context: DatasetParallelContext) -> None:
       """Store the distributed construction policy once for all build steps."""
       self.parallel_context = parallel_context

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
           dataset_type.__name__, tuple(train_valid_test_num_samples), config.mock, blend_mode, config.data_lazy_load,
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
           self, dataset_type: type, sizes: Sequence[int], config: GPTDatasetConfig,
   ) -> DatasetSplits:
       """Build generated train, validation, and test Datasets."""
       datasets = []
       for split, size in zip(_Split, sizes):
           if size == 0:
               datasets.append(None)
               continue

           dataset_factory = partial(dataset_type, None, None, None, int(size), split, config)
           mock_dataset = self._build_distributed_dataset(dataset_factory, barrier_needed=False)
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
           blend_mode, len(prefixes), weights[0], sizes_per_dataset[0],
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
       """Build each source's mid-level splits and assemble top-level blends."""
       datasets_per_source = []
       logger.debug("Building %d indexed Dataset sources", len(prefixes))
       # TODO: Add a separate cache-prebuild phase that assigns source_index % world_size to each rank,
       # synchronizes failures, and atomically writes a completion manifest before Dataset construction.
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
           blended_splits.append(self._build_distributed_dataset(dataset_factory, barrier_needed=True))

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
       """ Build each mid-level split from one low-level indexed Dataset. """
       if (config.data_lazy_load and self.parallel_context.distributed_enabled and not (
           self.parallel_context.data_index_cache or self.parallel_context.build_on_rank())
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
               dataset_factory = partial(
                   self._build_lazy_mid_level_dataset_split,
                   dataset_type,
                   dataset_path,
                   split,
                   split_range,
                   size,
                   config,
               )
               unique_identifiers = dataset_type.build_unique_identifiers(dataset_path, size, split, config)
               mid_level_dataset = LazyDatasetProxy(dataset_factory, unique_identifiers=unique_identifiers)
           else:
               dataset_factory = partial(dataset_type, low_level_dataset, dataset_path, indices, size, split, config)
               mid_level_dataset = self._build_distributed_dataset(
                   dataset_factory, barrier_needed=not config.is_dataset_from_mr
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
           low_level_dataset, split_indices = _build_low_level_dataset_and_split_indices(
               dataset_type, dataset_path, [split_range], config
           )
           mid_level_dataset = dataset_type(low_level_dataset, dataset_path, split_indices[0], size, split, config)
           return mid_level_dataset

       return self._build_distributed_dataset(_dataset_factory, barrier_needed=False)

   def _build_distributed_dataset(self, dataset_factory: Callable[[], Any], *, barrier_needed: bool) -> Any | None:
       """Build one Dataset with the shared distributed context."""
       return build_distributed_dataset(dataset_factory, self.parallel_context, barrier_needed=barrier_needed)


def build_dataset_splits(
       *,
       dataset_type: type,
       train_valid_test_num_samples: Sequence[int] | None,
       parallel_context: DatasetParallelContext,
       config: GPTDatasetConfig,
       blend_mode: BlendMode,
) -> DatasetSplits:
   """Build indexed Dataset splits through the compatibility function API."""
   split_builder = IndexedDatasetSplitBuilder(parallel_context)
   dataset_splits = split_builder.build(dataset_type, train_valid_test_num_samples, config, blend_mode)
   return dataset_splits
