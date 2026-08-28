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
    SidecarLocalBatchFetcher,
    SidecarLocalBatchSource,
)
from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import TensorShardSpec
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


def _validate_builder_source(source: LocalBatchSource) -> None:
    if platform.platform_type != PlatformType.PYTORCH:
        raise ValueError("The distributed dataset MVP currently supports only PyTorch.")
    if not isinstance(source, (SidecarLocalBatchSource, OnlineLocalBatchSource)):
        raise ValueError(
            "source must be a SidecarLocalBatchSource or OnlineLocalBatchSource, "
            f"but got {type(source)}."
        )


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
    source: LocalBatchSource,
    mesh: Any,
    config: DistributedDatasetConfig,
    *,
    communication_device: Any = None,
    prepare_local_batch: Callable[[Any], Any] | None = None,
    cost_model: CostModel | None = None,
) -> DistributedDataset:
    """Wrap a single-card local-batch source with distributed planning.

    The builder creates dedicated ``metadata_group`` and model-parallel data
    groups from ``mesh``. A sidecar source plans before target-rank fetches. An
    online source materializes one complete optimizer-step window, derives
    metadata, and then redistributes whole local batches. The distributed layer
    does not inspect or construct the contents of a local batch.

    Args:
        source: Single-card source producing complete local batches.
        mesh: Named root training mesh.
        config: Distributed planning, prefetch, and payload transport options.
        communication_device: Local collective device for A2A and MP delivery.
        prepare_local_batch: Optional move to the communication or model device.
        cost_model: Optional calibrated workload cost model.

    Returns:
        Configured :class:`DistributedDataset` iterator.
    """
    _validate_builder_source(source)
    topology = DataTopology.from_mesh(mesh, dp_dim_names=config.dp_dim_names)
    metadata_group, model_parallel_group = _create_data_groups(topology)
    planner = DistributedBatchPlanner(
        topology.data_parallel_size,
        config.micro_batch_num,
        cost_model=cost_model,
        cp_shards=config.cp_shards,
    )
    metadata_source, online_source = _create_local_batch_views(source, planner, topology)
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
