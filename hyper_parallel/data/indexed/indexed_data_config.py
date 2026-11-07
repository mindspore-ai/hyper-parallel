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

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import numpy as np

import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_store

from hyper_parallel.core.fault_tolerance.groups import is_recovery_enabled, is_recovery_reboot
from hyper_parallel.data.dataset_logging import get_dataset_logger

logger = get_dataset_logger(__name__)


@dataclass
class BlendedMegatronDatasetConfig:
    """Configuration shared by blended indexed datasets."""

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

        normalized_values = (np.asarray(split_values, dtype=np.float64) / np.sum(split_values)).tolist()
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
    create_ltor_fields: bool = False
    drop_last_partial_validation_sequence: bool = True
    add_extra_token_to_sequence: bool = True
    allow_ambiguous_pad_tokens: bool = False
    dataset_margin: float = 1.005
    skip_data_check: bool = False
    reuse_idx: bool = False
    data_lazy_load: bool = False

    low_level_dataset: type | None = None
    simple_blend: str | None = None
    dataset_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the blended Dataset fields followed by GPT-specific fields."""
        super().__post_init__()

        if self.low_level_dataset is None:
            raise ValueError("low_level_dataset must be explicitly configured")
        if self.simple_blend is None:
            raise ValueError("simple_blend must be explicitly configured")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.tokenizer is None and not self.mock:
            raise ValueError("Attribute 'tokenizer' must not be None")


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
    if not is_recovery_enabled() or not dist.is_initialized():
        return _resolve_data_paths(data_path, distributed_walk=distributed_walk)
    source = [data_path] if isinstance(data_path, str) else list(data_path)
    identity = json.dumps((source, distributed_walk), ensure_ascii=False).encode("utf-8")
    key = f"hyper_parallel/recovery_data_paths/{dist.get_rank()}/{hashlib.sha256(identity).hexdigest()}"
    store = _get_default_store()
    if is_recovery_reboot():
        # Rediscovery could change blend ordering even if it avoided collectives.
        if not store.check([key]):
            raise RuntimeError("original worker did not publish indexed data paths; recovery cannot rediscover them")
        paths = json.loads(store.get(key).decode("utf-8"))
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise RuntimeError("invalid indexed data path manifest in recovery store")
        return paths
    paths = _resolve_data_paths(source, distributed_walk=distributed_walk)
    store.set(key, json.dumps(paths, ensure_ascii=False))
    return paths


def _resolve_data_paths(data_path: str | Sequence[str], *, distributed_walk: bool) -> list[str]:
    """Discover indexed prefixes using the ordinary initial-training path."""
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
        mock_data = bool(data_config.get("mock_data", False))
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
            split=None if has_independent_blends else data_config.get("split", "1, 0, 0"),
            path_to_cache=data_config.get("data_cache_path"),
            mmap_bin_files=data_config.get("mmap_bin_files", True),
            reuse_idx=data_config.get("reuse_idx", False),
            data_lazy_load=data_config.get("data_lazy_load", False),
            reset_position_ids=data_config.get("reset_position_ids", False),
            reset_attention_mask=data_config.get("reset_attention_mask", False),
            eod_mask_loss=data_config.get("eod_mask_loss", False),
            create_ltor_fields=data_config.get("create_ltor_fields_in_dataloader", False),
            drop_last_partial_validation_sequence=data_config.get("drop_last_partial_validation_sequence", True),
            add_extra_token_to_sequence=data_config.get("add_extra_token_to_sequence", True),
            allow_ambiguous_pad_tokens=data_config.get("allow_ambiguous_pad_tokens", False),
            dataset_margin=data_config.get("dataset_margin", 1.005),
            skip_data_check=data_config.get("skip_data_check", False),
            low_level_dataset=resolve_dataset_type(data_config["low_level_dataset"]),
            dataset_options=(
                {key: value for key, value in data_config["low_level_dataset"].items() if key != "_target_"}
                if isinstance(data_config["low_level_dataset"], Mapping) else {}
            ),
            simple_blend=data_config["simple_blend"],
        )
    except KeyError as error:
        raise ValueError(f"Missing indexed Dataset build option: {error.args[0]}") from error
    source_paths = list(data_paths) if len(data_paths) <= 1 else list(data_paths[1::2])
    logger.debug("Resolved indexed sources=%d, first four=%s", len(source_paths), source_paths[:4])
    return config


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


def resolve_dataset_type(spec: Mapping[str, Any] | type) -> type:
    """Resolve an explicitly selected class without constructing a Dataset."""
    if isinstance(spec, type):
        return spec
    if not isinstance(spec, Mapping) or "_target_" not in spec:
        raise ValueError("low_level_dataset requires an explicit Dataset _target_")
    module, _, name = spec["_target_"].rpartition(".")
    dataset_type = getattr(import_module(module), name)
    if not isinstance(dataset_type, type):
        raise ValueError("low_level_dataset._target_ must identify a Dataset class")
    return dataset_type
