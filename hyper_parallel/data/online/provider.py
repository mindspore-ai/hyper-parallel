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
"""Source contracts and ownership-aware construction for Online builders.

This boundary describes sources and passes Trainer context to them. It does
not import storage backends or access views; providers choose their own readers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal, Protocol, TypeAlias

from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.parallel import (
    DataLoaderParallelContext,
    build_dataset_for_dataloader,
    create_dataloader_parallel_context,
)

logger = get_dataset_logger(__name__)

__all__ = [
    "AccessMode", "SourceBuildContext", "SampleAdapter", "SourceCapabilities",
    "SOURCE_INFO_KEY", "SourceInfo", "SourceProvider", "resolve_dataloader_context", "build_provider_source",
    "validate_access_mode",
]

AccessMode: TypeAlias = Literal["mapping", "iterable"]
SOURCE_INFO_KEY = "__source_info__"


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Stable provenance attached to one raw source record."""

    provider: str
    sample_key: str | None = None
    source_path: str | None = None
    shard_path: str | None = None


@dataclass(frozen=True)
class SourceBuildContext:
    """Trainer-derived state shared by every source provider."""

    random_seed: int = 42
    split_sizes: tuple[int, int, int] | None = None
    dataloader_context: DataLoaderParallelContext | None = None
    micro_batch_size: int = 1


@dataclass(frozen=True)
class SourceCapabilities:
    """Capabilities used to select an access view without format branches."""

    supports_mapping: bool
    supports_iterable: bool


def validate_access_mode(
    access_mode: AccessMode,
    capabilities: SourceCapabilities | None = None,
) -> None:
    """Reject unknown access modes and modes unsupported by a provider."""
    if access_mode not in ("mapping", "iterable"):
        raise ValueError("access_mode must be 'mapping' or 'iterable'")
    if capabilities is not None:
        supported = capabilities.supports_mapping if access_mode == "mapping" else capabilities.supports_iterable
        if not supported:
            raise ValueError(f"Source provider does not support {access_mode!r} access")


class SampleAdapter(Protocol):
    """Convert one provider-specific raw record into a canonical HP sample."""

    def __call__(self, raw_sample: Any) -> Mapping[str, Any]:
        """Return one mapping accepted by the configured data transform."""


class SourceProvider(Protocol):
    """Build raw access views independently of model semantics.

    Constructors store serializable configuration only. Online's provider
    builder supplies runtime context to ``build`` after checking ownership.
    Providers adapt records before filtering and return no model transform
    wrapper; Text/Omni builders retain their own transform lifecycle.
    """

    capabilities: SourceCapabilities

    def build(
        self,
        *,
        access_mode: AccessMode,
        context: SourceBuildContext,
        sample_adapter: SampleAdapter | None = None,
        sample_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> Any:
        """Build a source view and apply the adapter before filtering."""


def build_provider_source(
    source: SourceProvider,
    *,
    access_mode: AccessMode = "mapping",
    data_path: Any = None,
    data_config: Mapping[str, Any] | None = None,
    sample_adapter: SampleAdapter | None = None,
    sample_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    mesh_context: Any = None,
    dataloader_context: DataLoaderParallelContext | None = None,
    training_config: Any = None,
) -> Any:
    """Build an explicitly configured provider for an Online Dataset builder.

    Args:
        source: Lightweight provider; construction must not open a reader.
        access_mode: Shared access protocol, defaulting to integer indexing.
        data_path: Existing Dataset path, which must be absent with a provider.
        data_config: Dataset ownership options; storage options belong to source.
        sample_adapter: Optional raw-record to canonical-sample conversion.
        sample_filter: Predicate applied after adaptation.
        mesh_context: Trainer mesh used to determine DataLoader ownership.
        dataloader_context: Explicit ownership context, overriding mesh inference.
        training_config: Trainer seed and target sample counts.

    Returns:
        A raw source or split tuple, or None on a non-owning rank.

    Raises:
        TypeError: If the provider, adapter or filter contract is invalid.
        ValueError: If source selection or the requested access mode is invalid.
    """
    if not callable(getattr(source, "build", None)) or not hasattr(source, "capabilities"):
        raise TypeError("source must implement SourceProvider: capabilities and build")
    validate_access_mode(access_mode, source.capabilities)
    if sample_adapter is not None and not callable(sample_adapter):
        raise TypeError("sample_adapter must be callable")
    if sample_filter is not None and not callable(sample_filter):
        raise TypeError("sample_filter must be callable")
    config = dict(data_config or {})
    if data_path is not None or {"path", "data_path", "sources"}.intersection(config):
        raise ValueError("Configure paths only on source when dataset.source is present")
    training_seed = getattr(training_config, "seed", None)
    context = SourceBuildContext(
        random_seed=42 if training_seed is None else int(training_seed),
        split_sizes=_get_train_valid_test_num_samples(training_config),
        dataloader_context=resolve_dataloader_context(mesh_context, config, dataloader_context),
        micro_batch_size=max(1, int(getattr(training_config, "micro_batch_size", 1))),
    )
    factory = partial(source.build, access_mode=access_mode, context=context,
                      sample_adapter=sample_adapter, sample_filter=sample_filter)
    if context.dataloader_context is None:
        return factory()
    owner_context = context.dataloader_context
    if access_mode == "iterable":
        owner_context = replace(owner_context, data_index_cache=False)
    return build_dataset_for_dataloader(factory, owner_context, barrier_needed=False)


def resolve_dataloader_context(
        mesh_context: Any,
        data_config: Mapping[str, Any],
        explicit_context: DataLoaderParallelContext | None = None,
) -> DataLoaderParallelContext | None:
    """Resolve one shared Dataset ownership context."""
    if explicit_context is not None:
        return explicit_context
    if mesh_context is None:
        return None
    return create_dataloader_parallel_context(
        mesh_context,
        data_index_cache=bool(data_config.get("data_index_cache", False)),
        shared_storage=not bool(data_config.get("no_shared_storage", False)),
    )


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
        if training_config.train_samples is not None
        else train_iters * global_batch_size
    )
    eval_iters = int(training_config.eval_iters or 0)
    valid_iters = (train_iters // eval_iters + 1) * eval_iters if eval_iters else 0
    split_sizes = (
        train_samples,
        valid_iters * global_batch_size,
        eval_iters * global_batch_size,
    )
    logger.debug("Dataset target sizes: train=%d, validation=%d, test=%d", *split_sizes)
    return split_sizes
