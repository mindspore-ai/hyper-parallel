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
"""Resolve indexed data paths and normalize GPT Dataset build options."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hyper_parallel.platform import get_platform

logger = logging.getLogger(__name__)
platform = get_platform()


@dataclass
class BlendedMegatronDatasetConfig:
    """Configuration shared by blended Megatron-style Datasets."""

    random_seed: int
    sequence_length: int
    blend: list[str] | None = None
    blend_per_split: list[list[str] | None] | None = None
    split: str | None = None
    split_matrix: list[tuple[float, float] | None] | None = field(init=False, default=None)
    path_to_cache: str | None = None
    mmap_bin_files: bool = True
    mock: bool = False
    tokenizer: Any = None

    def __post_init__(self) -> None:
        """Validate the configured data distribution and derive split ranges."""
        if self.mock:
            return

        if self.blend_per_split is not None and any(self.blend_per_split):
            if self.blend is not None:
                raise ValueError("blend and blend_per_split are incompatible")
            if self.split is not None:
                raise ValueError("split and blend_per_split are incompatible")
            if len(self.blend_per_split) != 3:
                raise ValueError("blend_per_split must contain train, validation, and test blends")
            return

        if self.blend is None:
            raise ValueError("one of either blend or blend_per_split must be provided")
        if self.split is None:
            raise ValueError("both blend and split must be provided")

        # Parse at most three train/validation/test ratios, padding omitted
        # trailing splits with zero before normalization.
        split_values = [float(value) for value in re.findall(r"[.0-9]+", self.split)]
        split_values.extend([0.0] * (3 - len(split_values)))
        if len(split_values) != 3:
            raise ValueError(f"split length {len(split_values)} does not match expected 3")

        if not all(value >= 0.0 for value in split_values):
            raise ValueError("all split values must be non-negative")

        normalized_values = (
                np.asarray(split_values, dtype=np.float64) / np.sum(split_values)
        ).tolist()
        canonical_values = []
        for value in normalized_values:
            rounded_value = round(value, 12)
            precision = 2 if rounded_value == round(rounded_value, 2) else 12
            canonical_values.append(f"{rounded_value:.{precision}f}")
        self.split = ", ".join(canonical_values)

        # Convert normalized ratios into non-overlapping ranges. A zero-sized
        # split is represented by None.
        boundaries = [0.0]
        for value in normalized_values:
            boundaries.append(boundaries[-1] + value)
        self.split_matrix = []
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            self.split_matrix.append(None if end <= begin else (begin, end))


@dataclass
class GPTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration required to build indexed GPT Dataset splits."""

    reset_position_ids: bool = False
    reset_attention_mask: bool = False
    eod_mask_loss: bool = False
    create_attention_mask: bool = True
    dataset_margin: float = 1.005
    skip_data_check: bool = False
    reuse_idx: bool = False
    data_lazy_load: bool = False

    is_dataset_from_mr: bool = False
    simple_blend: str = "no"

    def __post_init__(self) -> None:
        """Validate the blended Dataset fields followed by GPT-specific fields."""
        super().__post_init__()

        if self.tokenizer is None and not self.mock:
            raise ValueError("Attribute 'tokenizer' must not be None")


def resolve_data_paths(
        data_path: str | Sequence[str],
        *,
        distributed_walk: bool = True,
) -> list[str]:
    """Find ``.bin`` files and return indexed prefixes with optional weights.

    Args:
        data_path: One indexed prefix or a sequence of directories/prefixes.
        distributed_walk: Whether initialized ranks divide directory walking.

    Returns:
        One prefix directly, or alternating weight/prefix values for multiple sources. If ``data_path`` is not a
        collection of discoverable indexed paths, return it unchanged so an explicit blend can be parsed.
    """
    # 1. Normalize one path and multiple paths to the same traversal form.
    data_paths = [data_path] if isinstance(data_path, str) else list(data_path)

    # 2. Use distributed walking only after the process group is available.
    # Before initialization, one process walks all configured directories.
    try:
        rank = int(platform.get_rank())
        world_size = int(platform.get_world_size())
    except (RuntimeError, ValueError):
        rank, world_size = 0, 1
    use_distributed_walk = distributed_walk and world_size > 1

    # 3. Accept a direct prefix, or recursively discover prefixes in a directory.
    file_paths: list[str] = []
    for current_path in data_paths:
        if os.path.isfile(current_path + ".bin"):
            file_paths.append(current_path)
            continue
        if not os.path.isdir(current_path):
            # Keep args.data_path unchanged when filesystem discovery
            # returns no files. This preserves explicit forms such as
            # ["30", prefix_a, "70", prefix_b] for the blend parser.
            original_paths = list(data_paths)
            return original_paths

        if use_distributed_walk:
            # Rank 0 discovers the first level, then all ranks use that same
            # directory order when taking their rank-strided share.
            root_layout = None
            if rank == 0:
                subdirectories = []
                top_files = []
                for name in os.listdir(current_path):
                    path = os.path.join(current_path, name)
                    if os.path.isdir(path):
                        subdirectories.append(path)
                    elif name.endswith(".bin"):
                        top_files.append(os.path.splitext(path)[0])
                root_layout = (subdirectories, top_files)

            gathered_layouts: list[tuple[list[str], list[str]] | None] = [None] * world_size
            platform.all_gather_object(gathered_layouts, root_layout)
            rank_zero_layout = gathered_layouts[0]
            if rank_zero_layout is None:
                raise ValueError("Rank 0 did not provide the indexed directory layout")
            subdirectories, top_files = rank_zero_layout

            local_files = _walk_bin_directories(subdirectories[rank::world_size])
            gathered_files: list[list[str] | None] = [None] * world_size
            platform.all_gather_object(gathered_files, local_files)

            # Reconstruct the complete and identical prefix list on every rank.
            file_paths.extend(top_files)
            for rank_files in gathered_files:
                if rank_files:
                    file_paths.extend(rank_files)
        else:
            file_paths.extend(_walk_bin_directories([current_path]))

    if not file_paths:
        raise ValueError(f"data_path {data_paths!r} has no data file ending with .bin")

    if len(file_paths) == 1:
        return file_paths

    # 4. Prefer the numeric shard order encoded by the corpus filenames.
    # Unknown naming schemes remain usable in their original walk order.
    try:
        file_paths.sort(
            key=lambda path: (
                int(os.path.basename(path).split("_")[2]),
                int(os.path.basename(path).split("_")[-1]),
            )
        )
    except (IndexError, TypeError, ValueError):
        logger.warning("Cannot sort indexed files with the numeric filename rule; keeping walk order")

    # 5. Automatically discovered sources use equal blend weights.
    weighted_paths = [value for path in file_paths for value in ("1", path)]
    return weighted_paths


def build_gpt_dataset_config(
        data_paths: Sequence[str],
        data_config: Mapping[str, Any],
) -> GPTDatasetConfig:
    """Build a GPT Dataset config from provider fields.

    Args:
        data_paths: Weighted prefixes returned by :func:`resolve_data_paths`.
        data_config: Indexed pretraining configuration supplied by the caller.

    Returns:
        Configuration required by GPT Dataset and split construction.

    Raises:
        ValueError: If a required build field is missing.
    """
    # The source argument parser owns defaults and validation. This stage only
    # renames fields where the Dataset implementation expects another name.
    blend_per_split = [
        data_config.get("train_data_path"),
        data_config.get("valid_data_path"),
        data_config.get("test_data_path"),
    ]
    has_independent_blends = any(blend_per_split)
    try:
        mock_data = bool(data_config["mock_data"])
        tokenizer = data_config.get("tokenizer")
        if tokenizer is None and not mock_data:
            raise ValueError("tokenizer is required when mock_data is false")

        config = GPTDatasetConfig(
            random_seed=data_config["random_seed"],
            sequence_length=data_config["seq_length"],
            tokenizer=tokenizer,
            mock=mock_data,
            blend=None if has_independent_blends else list(data_paths),
            blend_per_split=blend_per_split if has_independent_blends else None,
            split=None if has_independent_blends else data_config["split"],
            path_to_cache=data_config.get("data_cache_path"),
            mmap_bin_files=data_config.get("mmap_bin_files", True),
            reuse_idx=data_config.get("reuse_idx", False),
            data_lazy_load=data_config["data_lazy_load"],
            reset_position_ids=data_config.get("reset_position_ids", False),
            reset_attention_mask=data_config.get("reset_attention_mask", False),
            eod_mask_loss=data_config.get("eod_mask_loss", False),
            create_attention_mask=data_config.get("create_attention_mask_in_dataloader", True),
            dataset_margin=data_config.get("dataset_margin", 1.005),
            skip_data_check=data_config.get("skip_data_check", False),
            is_dataset_from_mr=data_config["is_dataset_from_mr"],
            simple_blend=data_config["simple_blend"],
        )
    except KeyError as error:
        raise ValueError(f"Missing indexed Dataset build option: {error.args[0]}") from error
    source_paths = list(data_paths) if len(data_paths) <= 1 else list(data_paths[1::2])
    logger.debug("Resolved indexed sources=%d, first four=%s", len(source_paths), source_paths[:4])
    return config


def _walk_bin_directories(directories: Sequence[str]) -> list[str]:
    """Recursively find ``.bin`` files below the assigned directories."""
    file_paths = []
    for directory in directories:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(".bin"):
                    path = os.path.abspath(os.path.join(root, filename))
                    file_paths.append(os.path.splitext(path)[0])
    return file_paths
