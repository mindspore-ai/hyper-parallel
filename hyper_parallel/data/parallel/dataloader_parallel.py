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
"""DataLoader rank ownership, Dataset cache synchronization, and DP topology."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from hyper_parallel.platform import get_platform
from hyper_parallel.data.dataset_logging import get_dataset_logger

platform = get_platform()
logger = get_dataset_logger(__name__)


def _always_build() -> bool:
    """Enable Dataset construction in a standalone process."""
    return True


def _no_barrier() -> None:
    """Provide a no-op barrier for standalone Dataset construction."""


@dataclass(frozen=True)
class DataLoaderParallelContext:
    """Describe DataLoader ownership and its DP/TP/CP topology.

    ``dp_rank`` and ``dp_world_size`` are consumed only by online iterable
    sources. Mapping sources leave DP sample ownership to their batch sampler.
    Every CP coordinate uses its TP rank zero as the DataLoader source rank.
    """

    build_on_rank: Callable[[], bool] = _always_build
    build_cache_on_rank: Callable[[], bool] = _always_build
    barrier: Callable[[], None] = _no_barrier
    distributed_enabled: bool = False
    data_index_cache: bool = False
    dp_rank: int = 0
    dp_world_size: int = 1
    tp_rank: int = 0
    tp_world_size: int = 1
    tp_group: Any = None
    cp_rank: int = 0
    cp_world_size: int = 1
    cp_mesh: Any = None


def _is_dataloader_rank(mesh_context: Any) -> bool:
    """Return whether this CP coordinate owns its DataLoader input."""
    tp_rank = int(getattr(mesh_context, "tp_rank", 0))
    owns_dataloader = tp_rank == 0
    return owns_dataloader


def _is_global_cache_builder_rank(shared_storage: bool) -> bool:
    """Select global rank zero as writer when Dataset caches are shared."""
    if not shared_storage:
        return True

    try:
        global_rank = int(platform.get_rank())
    except (RuntimeError, ValueError):
        global_rank = 0
    builds_shared_cache = global_rank == 0
    return builds_shared_cache


def create_dataloader_parallel_context(
        mesh_context: Any,
        *,
        data_index_cache: bool = False,
        shared_storage: bool = True,
        barrier: Callable[[], None] | None = None,
) -> DataLoaderParallelContext:
    """Create DataLoader ownership and synchronization policy from the Trainer mesh.

    Args:
        mesh_context: Trainer mesh state that provides TP, CP, and DP ranks.
        data_index_cache: Whether every Dataset rank may consume an existing index cache.
        shared_storage: Whether Dataset index caches are visible to every process.
        barrier: Optional Dataset-specific long-wait synchronization callback.

    Returns:
        DataLoader rank ownership, cache ownership, and synchronization callbacks.
    """
    device_mesh = getattr(mesh_context, "device_mesh", None)
    try:
        world_size = int(platform.get_world_size())
    except (RuntimeError, ValueError):
        world_size = 1
    distributed_enabled = device_mesh is not None and world_size > 1
    if not distributed_enabled:
        dataloader_context = DataLoaderParallelContext(data_index_cache=data_index_cache)
        logger.debug("Created standalone DataLoader context: data_index_cache=%s", data_index_cache)
        return dataloader_context

    build_on_rank = partial(_is_dataloader_rank, mesh_context)
    build_cache_on_rank = partial(_is_global_cache_builder_rank, shared_storage)
    mesh_dim_names = tuple(getattr(device_mesh, "mesh_dim_names", ()) or ())
    tp_mesh = device_mesh["tp"] if "tp" in mesh_dim_names else None
    cp_mesh = device_mesh["cp"] if "cp" in mesh_dim_names else None
    dataloader_context = DataLoaderParallelContext(
        build_on_rank=build_on_rank,
        build_cache_on_rank=build_cache_on_rank,
        barrier=barrier or platform.barrier,
        distributed_enabled=True,
        data_index_cache=data_index_cache,
        dp_rank=int(getattr(mesh_context, "dp_rank", 0)),
        dp_world_size=int(getattr(mesh_context, "dp_size", 1)),
        tp_rank=int(getattr(mesh_context, "tp_rank", 0)),
        tp_world_size=int(getattr(mesh_context, "tp_size", 1)),
        tp_group=tp_mesh.get_group() if tp_mesh is not None else None,
        cp_rank=int(getattr(mesh_context, "cp_rank", 0)),
        cp_world_size=int(getattr(mesh_context, "cp_size", 1)),
        cp_mesh=cp_mesh,
    )
    logger.debug(
        "Created distributed DataLoader context: dp_rank=%d, dp_world_size=%d, "
        "tp_rank=%d, tp_world_size=%d, cp_rank=%d, cp_world_size=%d, "
        "data_index_cache=%s, shared_storage=%s",
        dataloader_context.dp_rank,
        dataloader_context.dp_world_size,
        dataloader_context.tp_rank,
        dataloader_context.tp_world_size,
        dataloader_context.cp_rank,
        dataloader_context.cp_world_size,
        data_index_cache,
        shared_storage,
    )
    return dataloader_context


def split_iterable_dataset_by_dp(
        dataset: Any,
        dataloader_context: DataLoaderParallelContext,
) -> Any:
    """Split a Hugging Face iterable Dataset across data-parallel ranks."""
    if dataloader_context.dp_world_size <= 1:
        split_dataset = dataset
        return split_dataset

    from datasets.distributed import split_dataset_by_node  # pylint: disable=C0415

    split_dataset = split_dataset_by_node(
        dataset,
        rank=dataloader_context.dp_rank,
        world_size=dataloader_context.dp_world_size,
    )
    return split_dataset


def build_dataset_for_dataloader(
        dataset_factory: Callable[[], Any],
        dataloader_context: DataLoaderParallelContext,
        *,
        barrier_needed: bool,
) -> Any | None:
    """Build a Dataset on the ranks that own DataLoader input.

    Args:
        dataset_factory: Deferred Dataset construction callable.
        dataloader_context: DataLoader ownership and cache synchronization policy.
        barrier_needed: Whether cache writers and readers must synchronize.

    Returns:
        Dataset used by the local DataLoader, or ``None`` on non-owning ranks.

    Note:
        With shared storage, the cache-builder rank calls ``dataset_factory``
        first. Other DataLoader ranks wait for the cache and then reopen it.
    """
    if not dataloader_context.distributed_enabled:
        local_dataset = dataset_factory()
        return local_dataset

    local_dataset = None
    builds_cache_first = dataloader_context.build_cache_on_rank()
    owns_dataset = dataloader_context.data_index_cache or dataloader_context.build_on_rank()
    logger.debug(
        "DataLoader Dataset synchronization: owns_dataset=%s, builds_cache_first=%s",
        owns_dataset,
        builds_cache_first,
        enabled=owns_dataset,
    )

    if builds_cache_first and owns_dataset:
        local_dataset = dataset_factory()

    if barrier_needed:
        dataloader_context.barrier()

    if not builds_cache_first and owns_dataset:
        local_dataset = dataset_factory()

    return local_dataset
