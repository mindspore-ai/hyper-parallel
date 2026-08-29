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
"""Public construction API for PyTorch distributed datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from hyper_parallel.distributed_data.cost_model import CostModel
from hyper_parallel.distributed_data.distributed_dataset import DistributedDataset
from hyper_parallel.distributed_data.distributor import (
    IdentityLocalBatchRedistributor,
    IdentityModelParallelLocalBatchDistributor,
    LocalMetadataSynchronizer,
    TorchMetadataAllGather,
    TorchModelParallelLocalBatchDistributor,
    TorchPackedBytesLocalBatchRedistributor,
    TorchTensorLocalBatchRedistributor,
)
from hyper_parallel.distributed_data.data_construct import (
    LocalBatchMetadataView,
    LocalBatchSource,
    OnlineLocalBatchSource,
    OnlineLocalBatchView,
    RankLocalDataLoaderSource,
    SidecarLocalBatchFetcher,
    SidecarLocalBatchSource,
)
from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import LocalBatchMeta, TensorShardSpec, WorkloadCost
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType

platform = get_platform()

_PAYLOAD_TRANSPORTS = ("packed_bytes_a2a", "direct_tensor_a2a")


def _create_data_groups(topology: DataTopology) -> tuple[Any, Any]:
    """Create dedicated data communicators in one globally deterministic order."""
    layout_signature = (
        topology.mesh_shape,
        topology.mesh_dim_names,
        topology.rank_list,
        topology.dp_dim_names,
    )
    namespace = hashlib.sha256(repr(layout_signature).encode("utf-8")).hexdigest()[:12]

    created_groups = []
    metadata_group = None
    if len(topology.data_owner_ranks) > 1:
        metadata_group = platform.create_named_group(
            topology.data_owner_ranks,
            f"hp_data_metadata_{namespace}",
        )
        created_groups.append((topology.data_owner_ranks, metadata_group))

    model_parallel_group = None
    for data_rank, rank_group in enumerate(topology.model_parallel_rank_groups):
        if len(rank_group) == 1:
            continue
        group = platform.create_named_group(
            rank_group,
            f"hp_data_model_{namespace}_{data_rank}",
        )
        created_groups.append((rank_group, group))
        if data_rank == topology.data_rank:
            model_parallel_group = group

    # HCCL initializes communicators lazily. Prewarm them serially before the
    # data producer thread can issue collectives on multiple fresh groups.
    for rank_group, group in created_groups:
        if topology.global_rank in rank_group:
            platform.barrier(group)
        platform.barrier()
    return metadata_group, model_parallel_group


@dataclass(frozen=True)
class DistributedDatasetConfig:
    """Configuration for whole-step local-batch planning and distribution.

    The single-card source owns reading, workers, preprocessing, packing, and
    collation. This configuration only controls distributed planning, bounded
    step prefetch, payload transport, and model-parallel delivery.
    """

    micro_batch_num: int
    prefetch_steps: int = 2
    payload_transport: str = "packed_bytes_a2a"
    dp_dim_names: tuple[str, ...] | None = None
    cp_shards: tuple[TensorShardSpec, ...] = ()
    double_buffer: bool = False

    def __post_init__(self) -> None:
        for name in ("micro_batch_num", "prefetch_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        if not isinstance(self.double_buffer, bool):
            raise ValueError(f"double_buffer must be a boolean, but got {self.double_buffer!r}.")
        if self.payload_transport not in _PAYLOAD_TRANSPORTS:
            raise ValueError(
                f"payload_transport must be one of {_PAYLOAD_TRANSPORTS}, "
                f"but got {self.payload_transport!r}."
            )


def _validate_platform() -> None:
    if platform.platform_type != PlatformType.PYTORCH:
        raise ValueError("The distributed dataset MVP currently supports only PyTorch.")


def _complete_local_entries(
    source: LocalBatchSource,
    planner: DistributedBatchPlanner,
) -> int:
    complete_steps = len(source) // planner.global_local_batches_per_step
    return complete_steps * planner.local_batches_per_step


def _create_local_batch_views(
    source: LocalBatchSource,
    planner: DistributedBatchPlanner,
    topology: DataTopology,
) -> tuple[LocalBatchMetadataView | None, OnlineLocalBatchView | None]:
    max_entries = _complete_local_entries(source, planner)
    if isinstance(source, SidecarLocalBatchSource):
        return (
            LocalBatchMetadataView(
                source,
                topology.data_rank,
                topology.data_parallel_size,
                max_entries=max_entries,
            ),
            None,
        )
    if not isinstance(source, OnlineLocalBatchSource):
        raise ValueError(f"Unsupported local-batch source type {type(source)}.")
    return (
        None,
        OnlineLocalBatchView(
            source,
            topology.data_rank,
            topology.data_parallel_size,
            max_entries=max_entries,
        ),
    )


def _data_owner_loader_size(
    data_loader: Any,
    topology: DataTopology,
    metadata_group: Any,
    local_batches_per_step: int,
) -> int | None:
    if not topology.is_data_owner:
        return None
    if data_loader is None:
        raise ValueError("A data-owner rank must provide its existing rank-local DataLoader.")
    if not hasattr(data_loader, "__iter__"):
        raise ValueError("source must be an iterable rank-local DataLoader.")
    try:
        local_size = len(data_loader)
    except TypeError as exc:
        raise ValueError("A rank-local DataLoader must implement __len__ for bounded planning.") from exc

    owner_sizes = [local_size]
    if topology.data_parallel_size > 1:
        topology.validate_metadata_group(metadata_group)
        owner_sizes = [None] * topology.data_parallel_size
        platform.all_gather_object(owner_sizes, local_size, metadata_group)
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in owner_sizes):
        raise ValueError(f"Data owners reported invalid DataLoader sizes {owner_sizes}.")
    common_size = min(owner_sizes)
    return common_size - common_size % local_batches_per_step


def _share_data_owner_size(
    owner_size: int | None,
    topology: DataTopology,
    model_parallel_group: Any,
) -> int:
    if len(topology.model_parallel_ranks) == 1:
        if owner_size is None:
            raise ValueError("A single-rank data group must have a DataLoader size.")
        return owner_size
    topology.validate_model_parallel_group(model_parallel_group)
    group_ranks = tuple(platform.get_process_group_ranks(model_parallel_group))
    gathered_sizes = [None] * len(group_ranks)
    platform.all_gather_object(gathered_sizes, owner_size, model_parallel_group)
    sizes_by_rank = dict(zip(group_ranks, gathered_sizes, strict=True))
    shared_size = sizes_by_rank[topology.data_owner_rank]
    if not isinstance(shared_size, int) or isinstance(shared_size, bool) or shared_size < 0:
        raise ValueError(
            f"Data owner {topology.data_owner_rank} reported invalid DataLoader size {shared_size!r}."
        )
    return shared_size


def _create_data_sources(
    source: Any,
    metadata_fn: Callable[[Any, int | str], LocalBatchMeta | WorkloadCost] | None,
    planner: DistributedBatchPlanner,
    topology: DataTopology,
    metadata_group: Any,
    model_parallel_group: Any,
) -> tuple[
    LocalBatchSource,
    LocalBatchMetadataView | None,
    OnlineLocalBatchView | RankLocalDataLoaderSource | None,
]:
    if isinstance(source, (SidecarLocalBatchSource, OnlineLocalBatchSource)):
        if metadata_fn is not None:
            raise ValueError("metadata_fn is only valid when source is a rank-local DataLoader.")
        metadata_source, online_source = _create_local_batch_views(source, planner, topology)
        return source, metadata_source, online_source

    if metadata_fn is not None and not callable(metadata_fn):
        raise ValueError("metadata_fn must be callable or None.")
    owner_size = _data_owner_loader_size(
        source,
        topology,
        metadata_group,
        planner.local_batches_per_step,
    )
    source_size = _share_data_owner_size(owner_size, topology, model_parallel_group)
    rank_local_source = RankLocalDataLoaderSource(
        source if topology.is_data_owner else None,
        metadata_fn,
        data_rank=topology.data_rank,
        source_size=source_size,
    )
    return rank_local_source, None, rank_local_source


def _create_local_batch_redistributor(
    topology: DataTopology,
    config: DistributedDatasetConfig,
    metadata_group: Any,
    communication_device: Any,
) -> Any:
    if topology.data_parallel_size == 1 or not topology.is_data_owner:
        return IdentityLocalBatchRedistributor()
    topology.validate_metadata_group(metadata_group)
    if config.payload_transport == "packed_bytes_a2a":
        return TorchPackedBytesLocalBatchRedistributor(
            metadata_group,
            communication_device=communication_device,
        )
    return TorchTensorLocalBatchRedistributor(
        metadata_group,
        communication_device=communication_device,
    )


def _create_metadata_synchronizer(topology: DataTopology, metadata_group: Any) -> Any:
    if topology.data_parallel_size == 1 or not topology.is_data_owner:
        return LocalMetadataSynchronizer()
    topology.validate_metadata_group(metadata_group)
    return TorchMetadataAllGather(metadata_group)


def _create_model_parallel_distributor(
    topology: DataTopology,
    model_parallel_group: Any,
    communication_device: Any,
) -> Any:
    if len(topology.model_parallel_ranks) == 1:
        return IdentityModelParallelLocalBatchDistributor()
    topology.validate_model_parallel_group(model_parallel_group)
    return TorchModelParallelLocalBatchDistributor(
        model_parallel_group,
        communication_device=communication_device,
    )


def build_distributed_dataset(
    source: Any,
    mesh: Any,
    config: DistributedDatasetConfig,
    *,
    metadata_fn: Callable[[Any, int | str], LocalBatchMeta | WorkloadCost] | None = None,
    communication_device: Any = None,
    prepare_local_batch: Callable[[Any], Any] | None = None,
    cost_model: CostModel | None = None,
) -> DistributedDataset:
    """Wrap an existing rank-local DataLoader with distributed planning.

    The builder creates dedicated ``metadata_group`` and model-parallel data
    groups from ``mesh``. Only data-owner ranks iterate the user DataLoader;
    every iteration result is treated as one opaque, complete local batch. The
    existing ``SidecarLocalBatchSource`` and ``OnlineLocalBatchSource`` inputs
    remain supported for globally indexable sources.

    Args:
        source: Existing rank-local DataLoader on data-owner ranks, ``None`` on
            other ranks, or a prebuilt sidecar/global-map source.
        mesh: Named root training mesh.
        config: Distributed planning, prefetch, and payload transport options.
        metadata_fn: Optional rank-local callback receiving ``(local_batch,
            local_batch_id)`` and returning ``LocalBatchMeta`` or
            ``WorkloadCost``. Omit it for uniform-cost planning.
        communication_device: Local collective device for A2A and MP delivery.
        prepare_local_batch: Optional move to the communication or model device.
        cost_model: Optional calibrated workload cost model.

    Returns:
        Configured :class:`DistributedDataset` iterator.
    """
    _validate_platform()
    topology = DataTopology.from_mesh(mesh, dp_dim_names=config.dp_dim_names)
    metadata_group, model_parallel_group = _create_data_groups(topology)
    planner = DistributedBatchPlanner(
        topology.data_parallel_size,
        config.micro_batch_num,
        cost_model=cost_model,
        cp_shards=config.cp_shards,
    )
    source, metadata_source, online_source = _create_data_sources(
        source,
        metadata_fn,
        planner,
        topology,
        metadata_group,
        model_parallel_group,
    )
    local_batch_redistributor = None
    sidecar_fetcher = None
    if online_source is not None:
        local_batch_redistributor = _create_local_batch_redistributor(
            topology, config, metadata_group, communication_device
        )
    else:
        if not isinstance(source, SidecarLocalBatchSource):
            raise ValueError("Sidecar metadata view requires SidecarLocalBatchSource.")
        sidecar_fetcher = SidecarLocalBatchFetcher(source)
    metadata_synchronizer = _create_metadata_synchronizer(topology, metadata_group)
    model_parallel_distributor = _create_model_parallel_distributor(
        topology, model_parallel_group, communication_device
    )
    data_stream = platform.new_stream() if config.double_buffer and communication_device is not None else None
    return DistributedDataset(
        topology=topology,
        source=source,
        metadata_source=metadata_source,
        online_source=online_source,
        planner=planner,
        sidecar_fetcher=sidecar_fetcher,
        metadata_synchronizer=metadata_synchronizer,
        local_batch_redistributor=local_batch_redistributor,
        model_parallel_distributor=model_parallel_distributor,
        prefetch_steps=config.prefetch_steps,
        double_buffer=config.double_buffer,
        prepare_local_batch=prepare_local_batch,
        data_stream=data_stream,
    )
