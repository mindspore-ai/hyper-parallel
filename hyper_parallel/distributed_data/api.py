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

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from hyper_parallel.distributed_data.cost_model import CostModel
from hyper_parallel.distributed_data.distributed_dataset import DistributedDataset
from hyper_parallel.distributed_data.distributor import (
    LocalMetadataSynchronizer,
    LocalOwnerPayloadRedistributor,
    LocalPayloadDistributor,
    TorchMetadataAllGather,
    TorchPackedBytesRedistributor,
    TorchPayloadDistributor,
    TorchTensorRedistributor,
)
from hyper_parallel.distributed_data.materializer import (
    MapDatasetMaterializer,
    RankMaterializer,
    StridedMetadataSource,
    StridedOnlineSampleSource,
)
from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import SampleMeta, TensorShardSpec
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType

platform = get_platform()

_OWNER_PAYLOAD_TRANSPORTS = ("packed_bytes_a2a", "direct_tensor_a2a")


@dataclass(frozen=True)
class DistributedDatasetConfig:
    """Configuration for global-step planning and bounded prefetch.

    ``packed_bytes_a2a`` accepts nested byte records and JSON scalar values.
    ``direct_tensor_a2a`` requires every owner sample to be one tensor with a
    globally identical shape and dtype. ``prefetch_steps`` bounds lightweight
    step-plan look-ahead; payload look-ahead is always one microbatch.
    ``pin_memory`` copies collated Host tensor leaves into pinned memory on a
    dedicated data-owner thread.
    """

    micro_batch_size: int
    micro_batch_count: int
    prefetch_steps: int = 2
    owner_payload_transport: str = "packed_bytes_a2a"
    dp_dim_names: tuple[str, ...] | None = None
    cp_shards: tuple[TensorShardSpec, ...] = ()
    pin_memory: bool = False

    def __post_init__(self) -> None:
        for name in ("micro_batch_size", "micro_batch_count", "prefetch_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        if not isinstance(self.pin_memory, bool):
            raise ValueError(f"pin_memory must be a boolean, but got {self.pin_memory!r}.")
        if self.owner_payload_transport not in _OWNER_PAYLOAD_TRANSPORTS:
            raise ValueError(
                f"owner_payload_transport must be one of {_OWNER_PAYLOAD_TRANSPORTS}, "
                f"but got {self.owner_payload_transport!r}."
            )


def build_distributed_dataset(
    dataset: Any,
    mesh: Any,
    config: DistributedDatasetConfig,
    *,
    metadata_fn: Callable[[Any, int], SampleMeta] | None = None,
    metadata: Sequence[SampleMeta] | None = None,
    collate_fn: Callable[[list[Any]], Any] | None = None,
    metadata_group: Any = None,
    payload_group: Any = None,
    communication_device: Any = None,
    prepare_payload: Callable[[Any], Any] | None = None,
    cost_model: CostModel | None = None,
) -> DistributedDataset:
    """Build a PyTorch-only online-planned distributed dataset.

    Groups must come from the training mesh; this API never creates an
    independent communication world. ``metadata_group`` contains exactly one
    data owner per DP coordinate. ``payload_group`` contains all model peers
    sharing the current DP coordinate. Online metadata is the default path and
    balances one global microbatch at a time. An explicit sidecar ``metadata``
    sequence enables whole-step inter-microbatch planning before payload reads.

    Args:
        dataset: Shared map-style dataset. Online mode reads only one local
            microbatch of raw Host candidates at a time and redistributes them
            before heavyweight target-rank decode.
        mesh: Named root training mesh.
        config: Batch planning, prefetch, and owner A2A transport configuration.
        metadata_fn: Derive ``SampleMeta`` from ``(raw_sample, data_ref)`` for
            microbatch-local online planning. Required without ``metadata``.
        metadata: Optional shared lightweight sidecar for whole-step planning.
        collate_fn: Target-owner transform and collation function. In online
            mode it receives the raw samples retained or received after planning.
        metadata_group: Existing process group containing all data owners.
            Online mode also uses it for owner payload A2A.
        payload_group: Existing process group for this DP coordinate's peers.
        communication_device: Local collective device. Required for packed-byte
            owner A2A and tensor reception over HCCL/NCCL groups.
        prepare_payload: Optional owner-side move/packing before communication.
        cost_model: Optional calibrated workload cost model.

    Returns:
        Configured :class:`DistributedDataset` iterator.
    """
    if platform.platform_type != PlatformType.PYTORCH:
        raise ValueError("The distributed dataset MVP currently supports only PyTorch.")
    if metadata is None and metadata_fn is None:
        raise ValueError("Online metadata requires metadata_fn when no sidecar metadata is provided.")
    if metadata is not None and metadata_fn is not None:
        raise ValueError("Provide either online metadata_fn or sidecar metadata, but not both.")

    topology = DataTopology.from_mesh(mesh, dp_dim_names=config.dp_dim_names)
    planner = DistributedBatchPlanner(
        topology.data_world_size,
        config.micro_batch_size,
        config.micro_batch_count,
        cost_model=cost_model,
        cp_shards=config.cp_shards,
    )
    online_sample_source = None
    owner_payload_redistributor = None
    if metadata is None:
        if not hasattr(dataset, "__len__"):
            raise ValueError("Online metadata requires a map-style dataset implementing __len__.")
        complete_steps = len(dataset) // planner.global_samples_per_step
        metadata_source = None
        online_sample_source = StridedOnlineSampleSource(
            dataset,
            metadata_fn,
            topology.data_rank,
            topology.data_world_size,
            max_entries=complete_steps * planner.local_samples_per_step,
        )
        if topology.data_world_size == 1 or not topology.is_data_owner:
            owner_payload_redistributor = LocalOwnerPayloadRedistributor()
        else:
            topology.validate_metadata_group(metadata_group)
            if config.owner_payload_transport == "packed_bytes_a2a":
                owner_payload_redistributor = TorchPackedBytesRedistributor(
                    metadata_group,
                    communication_device=communication_device,
                )
            else:
                owner_payload_redistributor = TorchTensorRedistributor(
                    metadata_group,
                    communication_device=communication_device,
                )
    else:
        complete_steps = len(metadata) // planner.global_samples_per_step
        metadata_source = StridedMetadataSource(
            metadata,
            topology.data_rank,
            topology.data_world_size,
            max_entries=complete_steps * planner.local_samples_per_step,
        )
    rank_materializer = RankMaterializer(MapDatasetMaterializer(dataset), collate_fn)

    if topology.data_world_size == 1:
        metadata_synchronizer = LocalMetadataSynchronizer()
    else:
        if topology.is_data_owner:
            topology.validate_metadata_group(metadata_group)
            metadata_synchronizer = TorchMetadataAllGather(metadata_group)
        else:
            metadata_synchronizer = LocalMetadataSynchronizer()

    if len(topology.consumer_ranks) == 1:
        payload_distributor = LocalPayloadDistributor()
    else:
        topology.validate_payload_group(payload_group)
        payload_distributor = TorchPayloadDistributor(
            payload_group,
            communication_device=communication_device,
        )

    return DistributedDataset(
        topology=topology,
        metadata_source=metadata_source,
        planner=planner,
        rank_materializer=rank_materializer,
        metadata_synchronizer=metadata_synchronizer,
        payload_distributor=payload_distributor,
        prefetch_steps=config.prefetch_steps,
        pin_memory=config.pin_memory,
        prepare_payload=prepare_payload,
        online_sample_source=online_sample_source,
        owner_payload_redistributor=owner_payload_redistributor,
    )
