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
"""Compose prepared nv-meta readers with Online access and blend policies."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from hyper_parallel.data.nv_meta.reader import NvMetaDataset
from hyper_parallel.data.nv_meta.sample_adapter import NvMetaSampleAdapter
from hyper_parallel.data.online.provider import (
    AccessMode, SourceBuildContext, SampleAdapter, SourceCapabilities, validate_access_mode,
)
from hyper_parallel.data.online.source_views import IndexedBlendMappingView, build_indexed_view, resolve_read_balance

__all__ = ["NvMetaSource"]


class NvMetaSource:
    """Configure one prepared source or a Mapping blend without opening files."""

    capabilities = SourceCapabilities(supports_mapping=True, supports_iterable=True)

    def __init__(
        self,
        *,
        data_path: str | os.PathLike[str] | None = None,
        data_config: Mapping[str, Any] | None = None,
        record_part: str = "json",
    ) -> None:
        """Store the source used by an existing Text/Omni Dataset builder.

        Args:
            data_path: Prepared dataset root or metadata directory.
            data_config: Split, reader and access options; sources selects a blend
                and is mutually exclusive with data_path.
            record_part: Built-in sample record, default json. TXT maps to text,
                NPY to input_ids, and JSON/NPZ/PT provide field dictionaries.
        """
        self.data_config = dict(data_config or {})
        if {"path", "data_path"}.intersection(self.data_config):
            raise ValueError("Use source.data_path, not data_config.path or data_config.data_path")
        for alias, name in (("split_name", "split"), ("parts", "required_parts")):
            if alias in self.data_config:
                raise ValueError(f"Use source.data_config.{name}, not source.data_config.{alias}")
        self._sample_adapter = NvMetaSampleAdapter(record_part=record_part)
        if self.data_config.get("sources") is not None:
            if data_path is not None:
                raise ValueError("Dataset cannot combine data_path with data_config.sources")
            self.data_path = None
        else:
            path = os.fspath(data_path) if isinstance(data_path, (str, os.PathLike)) else None
            if not isinstance(path, str) or not path.strip():
                raise ValueError(".nv-meta Dataset requires a non-empty data_path")
            self.data_path = path

    def build(
        self,
        *,
        access_mode: AccessMode,
        context: SourceBuildContext,
        sample_adapter: SampleAdapter | None = None,
        sample_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> Any:
        """Build raw access views on a DataLoader-owning rank.

        Args:
            access_mode: Mapping or Iterable access, implemented by Online views.
            context: Trainer seed, target sizes and DataLoader ownership.
            sample_adapter: Optional replacement for the built-in record decoder.
                Cannot be combined with a non-default source.record_part.
            sample_filter: Validate each canonical sample on access. With
                filter_samples=True, Mapping scans candidates and Iterable skips invalid records.

        Returns:
            An adapted source or blend, without model transforms.
        """
        validate_access_mode(access_mode, self.capabilities)
        if sample_adapter is None:
            sample_adapter = self._sample_adapter
        elif self._sample_adapter.record_part != "json":
            raise ValueError("Configure source.record_part or sample_adapter, not both")
        if self.data_config.get("sources") is not None:
            if access_mode != "mapping":
                raise ValueError(
                    "nv-meta source.data_config.sources requires mapping access; "
                    "use source.data_path for a single Iterable source"
                )
            return _build_mapping_blend(
                self.data_config, context, sample_adapter, sample_filter,
            )
        reader = NvMetaDataset(
            self.data_path,
            split=str(self.data_config.get("split", "train")),
            exclude=self.data_config.get("exclude"),
            required_parts=_resolve_required_parts(self.data_config, sample_adapter),
            metadata_cache_size=int(self.data_config.get("metadata_cache_size", 4096)),
            max_open_shards=int(self.data_config.get("max_open_shards", 64)),
            cache_dir=self.data_config.get("cache_dir"),
            cache_timeout=float(self.data_config.get("cache_timeout", 600)),
            read_buffer_size=int(self.data_config.get("read_buffer_size", 8 * 1024 * 1024)),
        )
        try:
            return build_indexed_view(
                reader, access_mode=access_mode, context=context,
                data_config=self.data_config, split=reader.split,
                sample_adapter=sample_adapter, sample_filter=sample_filter,
            )
        except Exception:
            reader.close()
            raise


def _resolve_required_parts(config: Mapping[str, Any], adapter: SampleAdapter | None) -> list[str] | None:
    """Prefer explicit reader parts, then the adapter's declaration."""
    configured = config.get("required_parts")
    if configured is None and adapter is not None:
        configured = getattr(adapter, "required_parts", None)
    if configured is None:
        return None
    if isinstance(configured, str):
        configured = [configured]
    if not isinstance(configured, (list, tuple, set)):
        raise TypeError(".nv-meta required_parts must be a sequence of extensions")
    return [str(part).lstrip(".") for part in configured]


def _configure_blend_sources(
    sources: Sequence[Mapping[str, Any]],
    base_config: Mapping[str, Any],
) -> tuple[list[NvMetaSource], list[float]]:
    """Validate every leaf configuration before opening the first reader."""
    configured_sources = []
    weights = []
    shared_config = dict(base_config)
    shared_config.pop("sources", None)
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("Each .nv-meta source must contain data_path and weight")
        if "path" in source:
            raise ValueError("Each source must use data_path, not path")
        path = source.get("data_path")
        if path is None:
            raise ValueError("Each .nv-meta source must define data_path")
        weight = float(source.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(".nv-meta source weights must be finite and positive")
        child_config = dict(source)
        child_config.pop("data_path")
        child_config.pop("weight", None)
        child_config = shared_config | child_config
        if resolve_read_balance(child_config)[1] != "none":
            raise ValueError("nv-meta blend children cannot enable read_balance before source selection")
        configured_sources.append(NvMetaSource(data_path=path, data_config=child_config))
        weights.append(weight)

    scale = max(weights)
    scaled_weights = [weight / scale for weight in weights]
    if any(weight == 0.0 for weight in scaled_weights):
        raise ValueError(".nv-meta source weight ratios exceed floating-point precision")
    return configured_sources, scaled_weights


def _build_mapping_blend(
    config: Mapping[str, Any],
    context: SourceBuildContext,
    sample_adapter: SampleAdapter | None,
    sample_filter: Callable[[Mapping[str, Any]], bool] | None,
) -> Any:
    """Build a weighted global index for HP's existing Mapping sampler."""
    sources = config["sources"]
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError(".nv-meta sources must be a non-empty sequence")
    _, balance_policy = resolve_read_balance(config)
    if balance_policy != "none":
        raise ValueError(
            ".nv-meta read scheduling requires one source so the planner can see "
            "all sample metadata; blend weights remain sample-level"
        )
    configured_sources, weights = _configure_blend_sources(sources, config)
    child_context = replace(context, split_sizes=None)
    datasets = [
        source.build(
            access_mode="mapping", context=child_context,
            sample_adapter=sample_adapter, sample_filter=sample_filter,
        )
        for source in configured_sources
    ]
    target_size = sum(len(dataset) for dataset in datasets)
    if context.split_sizes is not None:
        split_name = str(config.get("split", "train"))
        split_name = {"val": "valid", "validation": "valid"}.get(split_name, split_name)
        target_size = context.split_sizes[("train", "valid", "test").index(split_name)]
    return IndexedBlendMappingView(datasets, weights, target_size, context.random_seed)
