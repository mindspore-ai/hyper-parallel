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
"""Shared file and optional Hugging Face helpers for Online LLM sources."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from hyper_models.components.datasets.dataset_logging import get_dataset_logger
from hyper_models.components.datasets.parallel import DatasetParallelContext

logger = get_dataset_logger(__name__)

_ONLINE_FILE_FORMATS = {
    ".arrow": "arrow",
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
}


def resolve_online_data_files(data_path: str | Sequence[str]) -> tuple[list[str], str]:
    """Resolve local Online files and their Hugging Face loader format.

    Args:
        data_path: One file/directory, comma-separated paths, or ordered paths.

    Returns:
        Deterministically ordered files and their common loader format.

    Raises:
        FileNotFoundError: If a configured local path does not exist.
        ValueError: If no supported files exist or formats are mixed.
    """
    if isinstance(data_path, str):
        configured_paths = [path.strip() for path in data_path.split(",") if path.strip()]
    else:
        configured_paths = list(data_path)

    data_files = []
    for configured_path in configured_paths:
        if os.path.isdir(configured_path):
            directory_files = [
                os.path.join(configured_path, filename)
                for filename in sorted(os.listdir(configured_path))
                if os.path.splitext(filename)[1].lower() in _ONLINE_FILE_FORMATS
            ]
            data_files.extend(directory_files)
        elif os.path.isfile(configured_path):
            data_files.append(configured_path)
        else:
            raise FileNotFoundError(f"Online Dataset path does not exist: {configured_path}")
    if not data_files:
        raise ValueError("Online data_path must contain at least one supported data file")

    loader_formats = {
        _ONLINE_FILE_FORMATS.get(os.path.splitext(data_file)[1].lower())
        for data_file in data_files
    }
    if None in loader_formats:
        raise ValueError("Online Dataset supports only JSON/JSONL/Parquet/CSV/Arrow files")
    if len(loader_formats) != 1:
        raise ValueError("All Online Dataset files must use the same format")
    loader_format = loader_formats.pop()
    return data_files, loader_format


def load_online_hf_dataset(
        *,
        data_path: str | Sequence[str] | None = None,
        data_config: Mapping[str, Any],
        streaming: bool = False,
) -> Any:
    """Load one local-file or Hugging Face Online Dataset.

    Args:
        data_path: Optional local source paths. Required unless
            ``hf_dataset_name`` is set.
        data_config: Namespace, cache, and optional HF Dataset identifiers.
        streaming: Whether to request the Hugging Face streaming implementation.

    Returns:
        A Hugging Face mapping or iterable Dataset.

    Raises:
        ImportError: If the optional ``datasets`` package is unavailable.
        ValueError: If neither ``data_path`` nor ``hf_dataset_name`` is set.
    """
    try:
        from datasets import (  # pylint: disable=C0415
            disable_progress_bars,
            enable_progress_bars,
            load_dataset,
        )
    except ImportError as error:
        raise ImportError(
            "Online LLM Dataset requires the optional 'datasets' package"
        ) from error

    namespace = str(data_config.get("namespace", "train"))
    cache_directory = data_config.get("cache_dir")
    if bool(data_config.get("show_progress", True)):
        enable_progress_bars()
    else:
        disable_progress_bars()
    hf_dataset_name = data_config.get("hf_dataset_name")
    if hf_dataset_name is not None:
        hf_config_name = data_config.get("hf_config_name")
        logger.debug(
            "Loading Hugging Face Dataset %s (config=%s, split=%s, streaming=%s)",
            hf_dataset_name,
            hf_config_name,
            namespace,
            streaming,
        )
        dataset = load_dataset(
            str(hf_dataset_name),
            name=hf_config_name,
            split=namespace,
            streaming=streaming,
            cache_dir=cache_directory,
        )
        return dataset

    if data_path is None:
        raise ValueError(
            "data_path is required when hf_dataset_name is not configured"
        )
    data_files, loader_format = resolve_online_data_files(data_path)
    logger.debug(
        "Loading %d Online Dataset files (format=%s, split=%s, streaming=%s)",
        len(data_files),
        loader_format,
        namespace,
        streaming,
    )
    dataset = load_dataset(
        loader_format,
        data_files=data_files,
        split=namespace,
        streaming=streaming,
        cache_dir=cache_directory,
    )
    return dataset


def split_online_dataset_by_dp(dataset: Any, parallel_context: DatasetParallelContext) -> Any:
    """Split one Hugging Face iterable source across data-parallel ranks."""
    if parallel_context.dp_world_size <= 1:
        return dataset
    try:
        from datasets.distributed import split_dataset_by_node  # pylint: disable=C0415
    except ImportError as error:
        raise ImportError(
            "Distributed Online iterable Dataset requires the optional 'datasets' package"
        ) from error
    split_dataset = split_dataset_by_node(
        dataset,
        rank=parallel_context.dp_rank,
        world_size=parallel_context.dp_world_size,
    )
    return split_dataset


def normalize_online_parallel_context(
        parallel_context: DatasetParallelContext | None,
) -> DatasetParallelContext:
    """Keep Online IO on TP rank zero even when indexed caches are enabled."""
    dataset_context = parallel_context or DatasetParallelContext()
    if dataset_context.data_index_cache:
        dataset_context = replace(dataset_context, data_index_cache=False)
    return dataset_context
