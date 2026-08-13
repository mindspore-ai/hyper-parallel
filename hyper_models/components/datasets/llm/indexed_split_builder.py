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

import logging
import math
import time
from collections.abc import Sequence
from enum import Enum
from functools import partial
from typing import Any, Literal, TypeAlias, cast

import numpy as np

from hyper_parallel.platform import get_platform
from hyper_models.components.datasets.llm.indexed_data_config import GPTDatasetConfig
from hyper_models.components.datasets.llm.indexed_blended_dataset import BlendedDataset
from hyper_models.components.datasets.llm.indexed_lazy_dataset import LazyDatasetProxy
from hyper_models.components.datasets.parallel import (
    DatasetParallelContext,
    build_distributed_dataset,
)
from hyper_models.components.datasets.llm.indexed_simple_blended_dataset import (
    SimpleBlendedDataset,
)

logger = logging.getLogger(__name__)
platform = get_platform()

BlendMode: TypeAlias = Literal["no", "inter", "intra"]
DatasetSplits: TypeAlias = tuple[Any | None, Any | None, Any | None]
SplitRange: TypeAlias = tuple[float, float]
SplitMatrix: TypeAlias = Sequence[SplitRange | None]


class _Split(Enum):
    """Stable train, validation, and test identifiers used by Dataset caches."""

    train = 0
    valid = 1
    test = 2


def _get_rank() -> int:
    """Return the current distributed rank for progress logging."""
    try:
        return int(platform.get_rank())
    except (RuntimeError, ValueError):
        return 0


def _should_log_progress(index: int, total: int) -> bool:
    """Return whether a source index is a useful progress milestone."""
    interval = max(1, total // 10)
    completed = index + 1
    return index == 0 or completed == total or completed % interval == 0


def build_dataset_splits(
        *,
        dataset_type: type,
        train_valid_test_num_samples: Sequence[int] | None,
        parallel_context: DatasetParallelContext,
        config: GPTDatasetConfig,
        blend_mode: BlendMode,
) -> DatasetSplits:
    """Build standard or simple train, validation, and test Dataset splits.
        ├── config.mock
        ├── config.blend
        └── config.blend_per_split

    Args:
        dataset_type: Concrete GPT Dataset type to construct.
        train_valid_test_num_samples: Requested sample count for each split.
        parallel_context: Rank ownership, cache ordering, and barrier callbacks.
        config: Normalized indexed GPT Dataset configuration.
        blend_mode: ``no`` for standard blending, or ``inter``/``intra`` for
            index-free MR blending.

    Returns:
        Train, validation, and test datasets.
    """
    _validate_build_inputs(train_valid_test_num_samples, config, blend_mode)
    # logger.info(
    #     "Rank %d starting indexed split build: targets=%s, mode=%s, lazy=%s",
    #     _get_rank(),
    #     tuple(train_valid_test_num_samples),
    #     blend_mode,
    #     config.data_lazy_load,
    # )

    if config.mock:
        datasets = _build_mock_dataset_splits(
            dataset_type,
            train_valid_test_num_samples,
            config,
            parallel_context,
        )
        return datasets

    if config.blend:
        datasets = _build_splits_from_shared_blend(
            dataset_type,
            train_valid_test_num_samples,
            config,
            blend_mode,
            parallel_context,
        )
        return datasets

    if config.blend_per_split:
        datasets = _build_splits_from_independent_blends(
            dataset_type,
            train_valid_test_num_samples,
            config,
            blend_mode,
            parallel_context,
        )
        return datasets
    raise ValueError("One of blend or blend_per_split must be configured")


def _validate_build_inputs(
        sizes: Sequence[int] | None,
        config: GPTDatasetConfig,
        blend_mode: BlendMode,
) -> None:
    """Validate the common split-builder inputs."""
    if sizes is None:
        raise ValueError(
            "train_valid_test_num_samples must be configured for indexed Datasets"
        )
    if len(sizes) != len(_Split):
        raise ValueError("train_valid_test_num_samples must contain exactly three values")
    if blend_mode not in {"no", "inter", "intra"}:
        raise ValueError(f"Unsupported blend mode: {blend_mode!r}")
    if blend_mode != "no" and not config.is_dataset_from_mr:
        raise ValueError("inter/intra blending requires is_dataset_from_mr=True")


def _build_mock_dataset_splits(
        dataset_type: type,
        sizes: Sequence[int],
        config: GPTDatasetConfig,
        parallel_context: DatasetParallelContext,
) -> DatasetSplits:
    """Build generated train, validation, and test Datasets."""
    datasets = []
    for split, size in zip(_Split, sizes):
        dataset_factory = partial(
            dataset_type,
            None,
            None,
            None,
            int(size),
            split,
            config,
        )
        mock_dataset = build_distributed_dataset(
            dataset_factory,
            parallel_context,
            barrier_needed=False,
        )
        datasets.append(mock_dataset)
    dataset_splits = cast(DatasetSplits, tuple(datasets))
    return dataset_splits


def _build_splits_from_shared_blend(
        dataset_type: type,
        sizes: Sequence[int],
        config: GPTDatasetConfig,
        blend_mode: BlendMode,
        parallel_context: DatasetParallelContext,
) -> DatasetSplits:
    """Build all splits from one shared data-source distribution.

    ``config.split_matrix`` divides every source into train, validation, and test,
    while ``config.blend`` defines the sources and their weights.
    """
    if config.blend is None or config.split_matrix is None:
        raise ValueError("A shared blend requires blend and split_matrix")

    # Shared mode can build all enabled splits through one common blend request.
    dataset_splits = _build_splits_from_blend(
        dataset_type,
        config.blend,
        config.split_matrix,
        sizes,
        config,
        blend_mode,
        parallel_context,
    )
    return dataset_splits


def _build_splits_from_independent_blends(
        dataset_type: type,
        sizes: Sequence[int],
        config: GPTDatasetConfig,
        blend_mode: BlendMode,
        parallel_context: DatasetParallelContext,
) -> DatasetSplits:
    """Build each split from its own data-source distribution.

    Each entry in ``config.blend_per_split`` owns the complete data range of that
    split; it is not divided again by the shared split matrix.
    """
    if config.blend_per_split is None or len(config.blend_per_split) != len(_Split):
        raise ValueError("Independent blends require one blend per split")

    datasets = []
    for split_index, blend in enumerate(config.blend_per_split):
        if not blend:
            datasets.append(None)
            continue

        # Present the current independent source as one full-range split. The
        # other two positions stay disabled so the common builder can be reused.
        split_matrix: list[SplitRange | None] = [None] * len(_Split)
        split_matrix[split_index] = (0.0, 1.0)
        split_sizes = [0] * len(_Split)
        split_sizes[split_index] = int(sizes[split_index])

        split_datasets = _build_splits_from_blend(
            dataset_type,
            blend,
            split_matrix,
            split_sizes,
            config,
            blend_mode,
            parallel_context,
        )
        datasets.append(split_datasets[split_index])
    dataset_splits = cast(DatasetSplits, tuple(datasets))
    return dataset_splits


def _build_splits_from_blend(
        dataset_type: type,
        blend: Sequence[str],
        split_matrix: SplitMatrix,
        sizes: Sequence[int],
        config: GPTDatasetConfig,
        blend_mode: BlendMode,
        parallel_context: DatasetParallelContext,
) -> DatasetSplits:
    """Parse one blend and dispatch it to the single- or multiple-source path."""
    # PanGu selects the direct Mid-level path from the external blend shape,
    # before parsing weights. Only [prefix] bypasses Top-level blending;
    # ["1", prefix], including one automatically discovered source, retains
    # the standard margin and BlendedDataset cache protocol.
    if len(blend) == 1:
        dataset_splits = _build_mid_level_dataset_splits(
            dataset_type,
            blend[0],
            split_matrix,
            sizes,
            config,
            parallel_context,
        )
        return dataset_splits

    # Convert the external weight/path representation into normalized source
    # metadata and the number of samples required from each source and split.
    prefixes, weights, source_sizes = _get_prefixes_weights_and_sizes_for_blend(
        blend,
        sizes,
        config.dataset_margin,
        blend_mode,
    )

    # Multiple sources first produce their own mid-level splits, then each group
    # of corresponding splits is assembled into one top-level blended Dataset.
    dataset_splits = _build_multiple_source_splits(
        dataset_type,
        prefixes,
        weights,
        source_sizes,
        split_matrix,
        config,
        blend_mode,
        parallel_context,
    )
    return dataset_splits


def _build_multiple_source_splits(
        dataset_type: type,
        prefixes: Sequence[str],
        weights: Sequence[float],
        source_sizes: Sequence[Sequence[int]],
        split_matrix: SplitMatrix,
        config: GPTDatasetConfig,
        blend_mode: BlendMode,
        parallel_context: DatasetParallelContext,
) -> DatasetSplits:
    """Build each source's mid-level splits and assemble top-level blends."""
    # Stage 1: build train/validation/test Mid-level Datasets independently for
    # every source. The resulting layout is grouped by source:
    # [[A_train, A_valid, A_test], [B_train, B_valid, B_test], ...].
    # total_sources = len(prefixes)
    rank = _get_rank()
    # started_at = time.monotonic()
    # logger.info(
    #     "Rank %d building %d mid-level indexed sources",
    #     rank,
    #     total_sources,
    # )
    datasets_per_source = []
    for source_index, (prefix, sizes_for_source) in enumerate(zip(prefixes, source_sizes)):
        source_datasets = _build_mid_level_dataset_splits(
            dataset_type,
            prefix,
            split_matrix,
            sizes_for_source,
            config,
            parallel_context,
        )
        datasets_per_source.append(source_datasets)
        # if _should_log_progress(source_index, total_sources):
        #     completed = source_index + 1
        #     logger.info(
        #         "Rank %d mid-level source progress: %d/%d (%.1f%%), elapsed=%.1fs",
        #         rank,
        #         completed,
        #         total_sources,
        #         completed * 100.0 / total_sources,
        #         time.monotonic() - started_at,
        #     )

    # logger.info(
    #     "Rank %d completed %d mid-level indexed sources in %.1fs",
    #     rank,
    #     total_sources,
    #     time.monotonic() - started_at,
    # )

    # Stage 2: regroup the Mid-level Datasets by split, for example
    # [A_train, B_train, ...], and wrap each enabled group in a Top-level
    # BlendedDataset or SimpleBlendedDataset according to blend_mode.
    # PanGu exposes the sum of the rounded per-source requests, including the
    # standard blend margin, so the top-level Dataset cannot undersupply the
    # training schedule after weighted rounding.
    blended_sizes = [sum(split_sizes) for split_sizes in zip(*source_sizes)]
    blended_splits = []
    for split_index, (split_range, size) in enumerate(zip(split_matrix, blended_sizes)):
        # A disabled split must remain None and must not construct a blend.
        if split_range is None:
            blended_splits.append(None)
            continue

        component_datasets = [datasets[split_index] for datasets in datasets_per_source]
        # logger.info(
        #     "Rank %d assembling %s blend from %d sources with target size %d",
        #     rank,
        #     _Split(split_index).name,
        #     len(component_datasets),
        #     int(size),
        # )
        blended_dataset = _assemble_blended_split(
            component_datasets,
            weights,
            int(size),
            blend_mode,
            config,
            parallel_context,
        )
        blended_splits.append(blended_dataset)
        logger.info("Rank %d completed %s blend assembly", rank, _Split(split_index).name)
    dataset_splits = cast(DatasetSplits, tuple(blended_splits))
    return dataset_splits


def _build_mid_level_dataset_splits(
        dataset_type: type,
        dataset_path: str,
        split_matrix: SplitMatrix,
        sizes: Sequence[int],
        config: GPTDatasetConfig,
        parallel_context: DatasetParallelContext,
) -> DatasetSplits:
    """Build each mid-level split from one low-level indexed Dataset.
        一个 Dataset 路径
        → 构建一个 Low-level Dataset
        → 生成 train/valid/test indices
        → 构建三个 Mid-level Dataset
    """
    low_level_dataset = None
    split_indices: list[np.ndarray | None] = [None] * len(split_matrix)
    if not config.data_lazy_load:
        low_level_dataset, split_indices = _build_low_level_dataset_and_split_indices(
            dataset_type,
            dataset_path,
            split_matrix,
            config,
        )

    datasets = []
    for split, split_range, indices, size in zip(
            _Split,
            split_matrix,
            split_indices,
            sizes,
    ):
        if split_range is None:
            datasets.append(None)
            continue
        if config.data_lazy_load:
            dataset_factory = partial(
                _build_lazy_mid_level_dataset_split,
                dataset_type,
                dataset_path,
                split,
                split_range,
                int(size),
                config,
                parallel_context,
            )
            unique_identifiers = dataset_type.build_unique_identifiers(
                dataset_path,
                int(size),
                split,
                config,
            )
            mid_level_dataset = LazyDatasetProxy(
                dataset_factory,
                dataset_length=int(size),
                unique_identifiers=unique_identifiers,
            )
        else:
            dataset_factory = partial(
                dataset_type,
                low_level_dataset,
                dataset_path,
                indices,
                int(size),
                split,
                config,
            )
            mid_level_dataset = build_distributed_dataset(
                dataset_factory,
                parallel_context,
                barrier_needed=not config.is_dataset_from_mr,
            )
        datasets.append(mid_level_dataset)
    dataset_splits = cast(DatasetSplits, tuple(datasets))
    return dataset_splits


def _build_lazy_mid_level_dataset_split(
        dataset_type: type,
        dataset_path: str,
        split: _Split,
        split_range: SplitRange,
        size: int,
        config: GPTDatasetConfig,
        parallel_context: DatasetParallelContext,
) -> Any | None:
    """Apply distributed construction when a lazy split is first accessed."""
    dataset_factory = partial(
        _build_mid_level_dataset_split,
        dataset_type,
        dataset_path,
        split,
        split_range,
        size,
        config,
    )
    mid_level_dataset = build_distributed_dataset(
        dataset_factory,
        parallel_context,
        barrier_needed=False,
    )
    return mid_level_dataset


def _build_mid_level_dataset_split(
        dataset_type: type,
        dataset_path: str,
        split: _Split,
        split_range: SplitRange,
        size: int,
        config: GPTDatasetConfig,
) -> Any:
    """Build one mid-level Dataset when its lazy proxy is first accessed."""
    low_level_dataset, split_indices = _build_low_level_dataset_and_split_indices(
        dataset_type,
        dataset_path,
        [split_range],
        config,
    )
    indices = split_indices[0]
    mid_level_dataset = dataset_type(
        low_level_dataset,
        dataset_path,
        indices,
        size,
        split,
        config,
    )
    return mid_level_dataset


def _build_low_level_dataset_and_split_indices(
        dataset_type: type,
        dataset_path: str,
        split_matrix: SplitMatrix,
        config: GPTDatasetConfig,
) -> tuple[Any, list[np.ndarray | None]]:
    """Build one low-level Dataset and calculate its requested split indices."""
    low_level_dataset = dataset_type.build_low_level_dataset(dataset_path, config)
    element_count = dataset_type.numel_low_level_dataset(low_level_dataset)
    split_indices = _build_split_indices(element_count, split_matrix)
    return low_level_dataset, split_indices


def _build_split_indices(
        element_count: int,
        split_matrix: SplitMatrix,
) -> list[np.ndarray | None]:
    """Convert normalized split ranges to low-level sequence indices."""
    split_indices = []
    for split_range in split_matrix:
        if split_range is None:
            split_indices.append(None)
            continue
        begin = int(round(split_range[0] * element_count))
        end = int(round(split_range[1] * element_count))
        indices = np.arange(begin, end, dtype=np.int32)
        split_indices.append(indices)
    return split_indices


def _compute_source_sample_counts(
        weights: Sequence[float],
        target_sizes: Sequence[int],
        dataset_margin: float,
        blend_mode: BlendMode,
) -> list[list[int]]:
    """Calculate the requested sample count for every source and split."""
    margin = dataset_margin if blend_mode == "no" else 1.0
    sizes_per_dataset = [
        [int(math.ceil(size * weight * margin)) for size in target_sizes]
        for weight in weights
    ]
    return sizes_per_dataset


def _get_prefixes_weights_and_sizes_for_blend(
        blend: Sequence[str],
        target_sizes: Sequence[int],
        dataset_margin: float,
        blend_mode: BlendMode,
) -> tuple[list[str], list[float], list[list[int]]]:
    """Parse one blend and calculate each source's normalized sample contribution."""
    if blend_mode == "no":
        prefixes, weights = _parse_blend(blend)
    else:
        prefixes, weights = _parse_simple_blend(blend)
    normalized_weights = _normalize_weights(weights)
    source_sizes = _compute_source_sample_counts(
        normalized_weights,
        target_sizes,
        dataset_margin,
        blend_mode,
    )
    return prefixes, normalized_weights, source_sizes


def _assemble_blended_split(
        datasets: Sequence[Any | None],
        weights: Sequence[float],
        size: int,
        blend_mode: BlendMode,
        config: GPTDatasetConfig,
        parallel_context: DatasetParallelContext,
) -> Any | None:
    """Wrap mid-level Datasets in the selected top-level blend."""
    missing_datasets = [dataset is None for dataset in datasets]
    if any(missing_datasets) and not all(missing_datasets):
        raise ValueError("Enabled sources must either all build or all remain None on a rank")
    component_datasets = [dataset for dataset in datasets if dataset is not None]
    if blend_mode == "no":
        dataset_factory = partial(BlendedDataset, component_datasets, weights, size, config)
    else:
        dataset_factory = partial(SimpleBlendedDataset, component_datasets, size, blend_mode)
    blended_dataset = build_distributed_dataset(
        dataset_factory,
        parallel_context,
        barrier_needed=True,
    )
    return blended_dataset


def _parse_blend(blend: Sequence[str]) -> tuple[list[str], list[float]]:
    """Parse PanGu's alternating floating-point weight/prefix values."""
    values = list(blend)
    if not values:
        raise ValueError("Dataset blend must not be empty")
    if len(values) % 2 != 0:
        raise ValueError("Dataset blend must contain alternating weight/prefix pairs")
    try:
        weights = [float(values[index]) for index in range(0, len(values), 2)]
    except ValueError as exc:
        raise ValueError("Dataset blend weights must be numeric") from exc
    prefixes = [values[index].strip() for index in range(1, len(values), 2)]
    return prefixes, weights


def _parse_simple_blend(blend: Sequence[str]) -> tuple[list[str], list[float]]:
    """Parse the integer unit weights required by PanGu SimpleBlendedDataset."""
    values = list(blend)
    if not values or len(values) % 2 != 0:
        raise ValueError("Simple blend must contain alternating weight/prefix pairs")
    try:
        integer_weights = [int(values[index]) for index in range(0, len(values), 2)]
    except ValueError as exc:
        raise ValueError("Simple blend weights must be integers") from exc
    if not all(weight == 1 for weight in integer_weights):
        raise ValueError("SimpleBlendedDataset requires all weights to be 1")
    prefixes = [values[index].strip() for index in range(1, len(values), 2)]
    weights = [float(weight) for weight in integer_weights]
    return prefixes, weights


def _normalize_weights(weights: Sequence[float]) -> list[float]:
    """Validate and normalize positive blend weights."""
    if not weights or any(weight <= 0.0 for weight in weights):
        raise ValueError("Dataset blend weights must be positive")
    total = float(sum(weights))
    normalized_weights = [float(weight) / total for weight in weights]
    return normalized_weights
