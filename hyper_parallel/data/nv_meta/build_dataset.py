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
"""Build decoded datasets from nv-meta metadata."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from typing import Any

from hyper_parallel.data.nv_meta.reader import NvMetaDataset
from hyper_parallel.data.nv_meta.sample_adapter import NvMetaSampleAdapter
from hyper_parallel.data.online.source_views import (
    AccessMode, IndexedAccessContext, IndexedBlendMappingView, SampleAdapter,
    build_indexed_view, resolve_read_balance,
)
from hyper_parallel.data.parallel import DataLoaderParallelContext, build_dataset_for_dataloader


def build_nv_meta_dataset(
    *,
    data_config: Mapping[str, Any] | None = None,
    data_path: str | os.PathLike[str] | None = None,
    access_mode: AccessMode = "mapping",
    is_valid_sample: Callable[[Mapping[str, Any]], bool] | None = None,
    dataloader_context: DataLoaderParallelContext | None = None,
    training_config: Any = None,
) -> Any:
    """Build decoded nv-meta samples on DataLoader-owning ranks.

    Args:
        data_config: Reader and access options. Without ``sample_adapter``,
            ``record_part`` selects the built-in decoder's part (default json;
            None is unsupported). A custom ``sample_adapter`` owns decoding and
            ignores this shortcut. See NvMetaSampleAdapter for supported parts.
        data_path: Dataset root or metadata directory; mutually exclusive with ``data_config.sources``.
        access_mode: Either ``"mapping"`` or ``"iterable"``; blends require Mapping.
        is_valid_sample: Optional check after decoding. Invalid samples raise by
            default; ``data_config.filter_samples`` explicitly enables filtering.
        dataloader_context: DataLoader ownership derived by the caller; None builds locally.
        training_config: Optional Trainer plan used for seed and target sizes.

    Returns:
        A decoded source implementing the requested access mode, or None on non-building ranks.

    Note:
        Training configurations use an existing Text/Omni Dataset entry with
        ``data_config.format: nv_meta`` to retain model-owned encoding stages.
        This function also supports direct use without a model transform.
        Use NvMetaDataset directly when raw part bytes are required.

    Raises:
        ValueError: If the access mode or source configuration is invalid.
    """
    config = dict(data_config or {})
    if access_mode not in ("mapping", "iterable"):
        raise ValueError("access_mode must be 'mapping' or 'iterable'")
    _validate_source_options(config)
    sample_adapter = config.get("sample_adapter")
    if sample_adapter is None:
        sample_adapter = NvMetaSampleAdapter(record_part=config.get("record_part", "json"))

    weights = None
    if config.get("sources") is None:
        sources = [(_validate_path(data_path), config)]
    else:
        if data_path is not None:
            raise ValueError("Dataset cannot combine data_path with data_config.sources")
        if access_mode != "mapping":
            raise ValueError("nv-meta data_config.sources requires mapping access; use data_path for Iterable")
        sources, weights = _configure_blend_sources(config)

    training_seed = getattr(training_config, "seed", None)
    context = IndexedAccessContext(
        random_seed=42 if training_seed is None else int(training_seed),
        split_sizes=_get_train_valid_test_num_samples(training_config),
        dataloader_context=dataloader_context,
        micro_batch_size=max(1, int(getattr(training_config, "micro_batch_size", 1))),
    )
    factory = partial(_build_sources, sources, weights, split=str(config.get("split", "train")),
                      context=context, access_mode=access_mode,
                      sample_adapter=sample_adapter, is_valid_sample=is_valid_sample)
    if dataloader_context is None:
        return factory()
    if access_mode == "iterable":
        dataloader_context = replace(dataloader_context, data_index_cache=False)
    return build_dataset_for_dataloader(factory, dataloader_context, barrier_needed=False)


def _validate_path(data_path: Any) -> str:
    """Validate a prepared metadata path without accessing storage."""
    path = os.fspath(data_path) if isinstance(data_path, (str, os.PathLike)) else None
    if not isinstance(path, str) or not path.strip():
        raise ValueError(".nv-meta Dataset requires a non-empty data_path")
    return path


def _validate_source_options(config: Mapping[str, Any]) -> None:
    """Reject conflicting formats, misplaced paths and unsupported option names."""
    if config.get("format", "nv_meta") != "nv_meta":
        raise ValueError("nv-meta builder requires data_config.format: nv_meta")
    if "path" in config or "data_path" in config:
        raise ValueError("Use dataset.data_path, not data_config.path or data_config.data_path")
    for alias, name in (("split_name", "split"), ("parts", "required_parts")):
        if alias in config:
            raise ValueError(f"Use data_config.{name}, not data_config.{alias}")


def _configure_blend_sources(
    config: Mapping[str, Any],
) -> tuple[list[tuple[str, Mapping[str, Any]]], list[float]]:
    """Combine shared options with each source's path, weight, split and exclusions."""
    sources = config["sources"]
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError(".nv-meta sources must be a non-empty sequence")
    _, balance_policy = resolve_read_balance(config)
    if balance_policy != "none":
        raise ValueError("For nv-meta sources, set read_balance: none; use one data_path to enable read balancing")
    shared_config = dict(config)
    shared_config.pop("sources")
    configured_sources = []
    weights = []
    for source_index, source in enumerate(sources):
        if not isinstance(source, Mapping) or source.keys() - {"data_path", "weight", "split", "exclude"}:
            raise ValueError(
                f"data_config.sources[{source_index}] accepts data_path and optional weight, split, exclude; "
                "put shared reader, decoder and access options in data_config"
            )
        child_config = shared_config | dict(source)
        path = _validate_path(child_config.pop("data_path", None))
        weight = float(child_config.pop("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(".nv-meta source weights must be finite and positive")
        configured_sources.append((path, child_config))
        weights.append(weight)
    scale = max(weights)
    scaled_weights = [weight / scale for weight in weights]
    if any(weight == 0.0 for weight in scaled_weights):
        raise ValueError(".nv-meta source weight ratios exceed floating-point precision")
    return configured_sources, scaled_weights


def _build_sources(
    sources: Sequence[tuple[str, Mapping[str, Any]]],
    weights: Sequence[float] | None,
    *,
    split: str,
    context: IndexedAccessContext,
    access_mode: AccessMode,
    sample_adapter: SampleAdapter,
    is_valid_sample: Callable[[Mapping[str, Any]], bool] | None,
) -> Any:
    """Open readers after ownership selection and release all on build failure."""
    child_context = context if weights is None else replace(context, split_sizes=None)
    with ExitStack() as resources:
        datasets = []
        for path, config in sources:
            required_parts = config.get("required_parts")
            if required_parts is None:
                required_parts = getattr(sample_adapter, "required_parts", None)
            reader_options = {name: config[name] for name in (
                "split", "exclude", "metadata_cache_size", "max_open_shards",
                "cache_dir", "cache_timeout", "read_buffer_size",
            ) if name in config}
            reader = NvMetaDataset(path, required_parts=required_parts, **reader_options)
            resources.callback(reader.close)
            datasets.append(build_indexed_view(
                reader, access_mode=access_mode, context=child_context,
                data_config=config, split=reader.split,
                sample_adapter=sample_adapter, is_valid_sample=is_valid_sample,
            ))
        result = datasets[0]
        if weights is not None:
            target_size = sum(len(dataset) for dataset in datasets)
            if context.split_sizes is not None:
                split_name = {"val": "valid", "validation": "valid"}.get(split, split)
                target_size = context.split_sizes[("train", "valid", "test").index(split_name)]
            result = IndexedBlendMappingView(datasets, weights, target_size, context.random_seed)
        resources.pop_all()
        return result


def _get_train_valid_test_num_samples(training_config: Any) -> tuple[int, int, int] | None:
    """Derive source target sizes from the Trainer plan when available."""
    if training_config is None:
        return None
    global_batch_size = int(training_config.global_batch_size)
    if global_batch_size <= 0:
        raise ValueError("training.global_batch_size must be positive")
    if training_config.train_iters is not None:
        train_iters = int(training_config.train_iters)
    elif training_config.train_samples is not None:
        train_iters = int(training_config.train_samples) // global_batch_size
    else:
        raise ValueError("training.train_iters and training.train_samples cannot both be None")
    if train_iters <= 0:
        raise ValueError("training configuration must produce at least one train iteration")
    train_samples = (
        int(training_config.train_samples)
        if training_config.train_samples is not None else train_iters * global_batch_size
    )
    eval_iters = int(training_config.eval_iters or 0)
    valid_iters = (train_iters // eval_iters + 1) * eval_iters if eval_iters else 0
    return train_samples, valid_iters * global_batch_size, eval_iters * global_batch_size
