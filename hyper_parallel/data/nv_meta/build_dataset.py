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
"""Convenience builder for adapted nv-meta sources without model processing."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from hyper_parallel.data.nv_meta.provider import NvMetaSource
from hyper_parallel.data.online.provider import AccessMode, SampleAdapter, build_provider_source


def build_nv_meta_dataset(
    *,
    data_config: Mapping[str, Any] | None = None,
    data_path: str | os.PathLike[str] | None = None,
    access_mode: AccessMode = "mapping",
    record_part: str = "json",
    sample_adapter: SampleAdapter | None = None,
    sample_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    mesh_context: Any = None,
    dataloader_context: Any = None,
    training_config: Any = None,
) -> Any:
    """Build an adapted source for direct use outside the Text/Omni builders.

    Args:
        data_config: Optional provider, split, balancing, and blend configuration.
            An omitted value uses the provider defaults.
        data_path: Dataset root or metadata directory; mutually exclusive with ``data_config.sources``.
        access_mode: Either ``"mapping"`` or ``"iterable"``; blends require Mapping.
        record_part: JSON/NPZ/PT fields, TXT text, or NPY token IDs; defaults to json.
        sample_adapter: Optional replacement for the built-in sample decoder.
        sample_filter: Optional predicate applied after adaptation.
        mesh_context: Optional mesh used to derive DataLoader ownership.
        dataloader_context: Optional precomputed ownership context.
        training_config: Optional Trainer plan used for seed and target sizes.

    Returns:
        A decoded source implementing the requested access mode.

    Note:
        Training configurations should use an existing Text/Omni Dataset entry
        with ``source=NvMetaSource(...)`` to retain model-owned encoding stages.
        Use NvMetaDataset directly when raw part bytes are required.

    Raises:
        ValueError: If the access mode or provider configuration is invalid.
    """
    config = dict(data_config or {})
    return build_provider_source(
        source=NvMetaSource(data_path=data_path, data_config=config, record_part=record_part),
        access_mode=access_mode,
        sample_adapter=sample_adapter,
        sample_filter=sample_filter,
        data_config={key: config[key] for key in ("data_index_cache", "no_shared_storage") if key in config},
        mesh_context=mesh_context,
        dataloader_context=dataloader_context,
        training_config=training_config,
    )
