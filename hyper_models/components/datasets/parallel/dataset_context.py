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
"""Distributed Dataset construction context and lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from hyper_parallel.platform import get_platform

platform = get_platform()


def _always_build() -> bool:
    """Enable Dataset construction in a standalone process."""
    return True


def _no_barrier() -> None:
    """Provide a no-op barrier for standalone Dataset construction."""


@dataclass(frozen=True)
class DatasetParallelContext:
    """Callbacks and topology required by distributed Dataset construction.

    ``dp_rank`` and ``dp_world_size`` are consumed only by Online iterable
    sources. Mapping sources leave DP sample ownership to their batch sampler.
    """

    build_on_rank: Callable[[], bool] = _always_build
    build_cache_on_rank: Callable[[], bool] = _always_build
    barrier: Callable[[], None] = _no_barrier
    distributed_enabled: bool = False
    data_index_cache: bool = False
    dp_rank: int = 0
    dp_world_size: int = 1


def _is_tp_rank_zero(mesh_context: Any) -> bool:
    """Return whether the current rank is rank zero in its TP group."""
    tp_rank = int(getattr(mesh_context, "tp_rank", 0))
    return tp_rank == 0


def _is_global_cache_builder_rank(shared_storage: bool) -> bool:
    """Select global rank zero as writer when Dataset caches are shared."""
    if not shared_storage:
        return True
    try:
        global_rank = int(platform.get_rank())
    except (RuntimeError, ValueError):
        global_rank = 0
    return global_rank == 0


def create_dataset_parallel_context(
        mesh_context: Any,
        *,
        data_index_cache: bool = False,
        shared_storage: bool = True,
) -> DatasetParallelContext:
    """Create Dataset construction policy from the Trainer topology.

    Args:
        mesh_context: Trainer mesh state that provides TP rank and device mesh.
        data_index_cache: Whether every Dataset rank may consume an existing index cache.
        shared_storage: Whether Dataset index caches are visible to every process.

    Returns:
        Rank ownership, cache ownership, and synchronization callbacks.
    """
    device_mesh = getattr(mesh_context, "device_mesh", None)
    try:
        world_size = int(platform.get_world_size())
    except (RuntimeError, ValueError):
        world_size = 1
    distributed_enabled = device_mesh is not None and world_size > 1
    if not distributed_enabled:
        parallel_context = DatasetParallelContext(data_index_cache=data_index_cache)
        return parallel_context

    build_on_rank = partial(_is_tp_rank_zero, mesh_context)
    build_cache_on_rank = partial(_is_global_cache_builder_rank, shared_storage)
    parallel_context = DatasetParallelContext(
        build_on_rank=build_on_rank,
        build_cache_on_rank=build_cache_on_rank,
        barrier=platform.barrier,
        distributed_enabled=True,
        data_index_cache=data_index_cache,
        dp_rank=int(getattr(mesh_context, "dp_rank", 0)),
        dp_world_size=int(getattr(mesh_context, "dp_size", 1)),
    )
    return parallel_context


def build_distributed_dataset(
        dataset_factory: Callable[[], Any],
        parallel_context: DatasetParallelContext,
        *,
        barrier_needed: bool,
) -> Any | None:
    """Construct one Dataset according to rank ownership and cache order.

    Args:
        dataset_factory: Deferred Dataset construction callable.
        parallel_context: Rank ownership and cache synchronization policy.
        barrier_needed: Whether cache writers and readers must synchronize.

    Returns:
        The Dataset on owning ranks, otherwise ``None``.
    """
    if not parallel_context.distributed_enabled:
        dataset = dataset_factory()
        return dataset

    dataset = None
    builds_cache_first = parallel_context.build_cache_on_rank()
    should_build = parallel_context.data_index_cache or parallel_context.build_on_rank()
    if builds_cache_first and should_build:
        dataset = dataset_factory()

    if barrier_needed:
        parallel_context.barrier()

    if not builds_cache_first and should_build:
        dataset = dataset_factory()
    return dataset


__all__ = ["DatasetParallelContext", "build_distributed_dataset", "create_dataset_parallel_context"]
