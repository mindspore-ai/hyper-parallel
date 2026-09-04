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

from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.parallel import DataLoaderParallelContext

logger = get_dataset_logger(__name__)

ONLINE_PLAINTEXT_TEXT_KEYS_KEY = "_online_plaintext_text_keys"

_ONLINE_FILE_FORMATS = {
    ".arrow": "arrow",
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
}


def _is_nonempty_plaintext_sample(sample: Mapping[str, Any], *, text_keys: str | Sequence[str]) -> bool:
    """Return whether a raw plaintext sample contains non-whitespace text."""
    if isinstance(text_keys, str):
        text = sample[text_keys]
    else:
        for text_key in text_keys:
            if text_key in sample:
                text = sample[text_key]
                break
        else:
            raise ValueError(f"Sample does not contain any configured text fields: {list(text_keys)!r}")

    if not isinstance(text, str):
        raise ValueError("Plaintext sample text must be a string")

    is_nonempty = text.strip() != ""
    return is_nonempty


def _filter_online_plaintext_dataset(dataset: Any, data_config: Mapping[str, Any], *, streaming: bool) -> Any:
    """Filter empty plaintext records before shuffling, sharding, or tokenization."""
    text_keys = data_config.get(ONLINE_PLAINTEXT_TEXT_KEYS_KEY)
    if text_keys is None:
        return dataset

    source_records = None if streaming else len(dataset)
    filtered_dataset = dataset.filter(_is_nonempty_plaintext_sample, fn_kwargs={"text_keys": text_keys})

    # log
    if streaming:
        logger.debug("Enabled lazy empty-plaintext filtering for Online iterable Dataset")
    else:
        filtered_records = len(filtered_dataset)
        logger.debug(
            "Filtered empty Online plaintext records: source=%d, filtered=%d, removed=%d",
            source_records, filtered_records, source_records - filtered_records,
        )

    return filtered_dataset


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
        if os.path.isfile(configured_path):
            data_files.append(configured_path)
            continue
        if not os.path.isdir(configured_path):
            raise FileNotFoundError(f"Online Dataset path does not exist: {configured_path}")

        for filename in sorted(os.listdir(configured_path)):
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in _ONLINE_FILE_FORMATS:
                data_files.append(os.path.join(configured_path, filename))

    if not data_files:
        raise ValueError("Online data_path must contain at least one supported data file")

    loader_formats = set()
    for data_file in data_files:
        file_extension = os.path.splitext(data_file)[1].lower()
        file_format = _ONLINE_FILE_FORMATS.get(file_extension)
        loader_formats.add(file_format)

    if None in loader_formats or len(loader_formats) != 1:
        raise ValueError("Online Dataset files must use one supported format: JSON/JSONL/Parquet/CSV/Arrow")

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
        data_config: Cache options and optional HF Dataset identifiers.
        streaming: Whether to request the Hugging Face streaming implementation.

    Returns:
        A Hugging Face mapping or iterable Dataset.

    Raises:
        ImportError: If the optional ``datasets`` package is unavailable.
        ValueError: If neither ``data_path`` nor ``hf_dataset_name`` is set.
    """
    try:
        from datasets import load_dataset  # pylint: disable=C0415
    except ImportError as error:
        raise ImportError("Online LLM Dataset requires the optional 'datasets' package") from error

    cache_directory = data_config.get("cache_dir")
    hf_dataset_name = data_config.get("hf_dataset_name")
    if hf_dataset_name is not None:
        hf_config_name = data_config.get("hf_config_name")
        logger.debug(
            "Loading Hugging Face Dataset %s (config=%s, split=%s, streaming=%s)",
            hf_dataset_name, hf_config_name, "train", streaming,
        )
        dataset = load_dataset(
            str(hf_dataset_name), name=hf_config_name, split="train",
            streaming=streaming, cache_dir=cache_directory,
        )
    else:
        if data_path is None:
            raise ValueError("data_path is required when hf_dataset_name is not configured")

        data_files, loader_format = resolve_online_data_files(data_path)
        logger.debug(
            "Loading %d Online Dataset files (format=%s, split=%s, streaming=%s)",
            len(data_files), loader_format, "train", streaming,
        )
        dataset = load_dataset(
            loader_format, data_files=data_files, split="train",
            streaming=streaming, cache_dir=cache_directory,
        )

    return _filter_online_plaintext_dataset(dataset, data_config, streaming=streaming)


def normalize_online_dataloader_context(
        dataloader_context: DataLoaderParallelContext | None,
) -> DataLoaderParallelContext:
    """Keep Online IO on TP rank zero even when indexed caches are enabled."""
    normalized_context = dataloader_context or DataLoaderParallelContext()
    if normalized_context.data_index_cache:
        normalized_context = replace(normalized_context, data_index_cache=False)
    return normalized_context
