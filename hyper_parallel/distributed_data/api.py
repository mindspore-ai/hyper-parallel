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
from typing import Any, Callable, Sequence

from hyper_parallel.distributed_data.cost_model import CostModel
from hyper_parallel.distributed_data.distributed_dataset import DistributedDataset
from hyper_parallel.distributed_data.distributor import (
    LocalMetadataSynchronizer,
    LocalMicroBatchDistributor,
    LocalSampleRedistributor,
    TorchMicroBatchDistributor,
    TorchMetadataAllGather,
    TorchPackedBytesRedistributor,
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

_SAMPLE_TRANSPORTS = ("packed_bytes_a2a", "direct_tensor_a2a")


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
    """Configuration for global-step planning and bounded prefetch.

    ``packed_bytes_a2a`` accepts nested byte records and JSON scalar values.
    ``direct_tensor_a2a`` requires every owner sample to be one tensor with a
    globally identical shape and dtype. ``prefetch_steps`` bounds lightweight
    step-plan look-ahead. ``double_buffer`` keeps exactly one fully prepared
    microbatch ahead, including online metadata collectives and sample
    distribution. One prefetched step enables overlap within an optimizer
    step; two or more also enable overlap across optimizer-step boundaries.
    ``pin_memory`` copies collated Host tensor leaves into pinned memory on a
    dedicated data-owner thread.
    """

    micro_batch_size: int
    micro_batch_num: int
    prefetch_steps: int = 2
    sample_transport: str = "packed_bytes_a2a"
    dp_dim_names: tuple[str, ...] | None = None
    cp_shards: tuple[TensorShardSpec, ...] = ()
    pin_memory: bool = False
    double_buffer: bool = False

    def __post_init__(self) -> None:
        for name in ("micro_batch_size", "micro_batch_num", "prefetch_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        if not isinstance(self.pin_memory, bool):
            raise ValueError(f"pin_memory must be a boolean, but got {self.pin_memory!r}.")
        if not isinstance(self.double_buffer, bool):
            raise ValueError(f"double_buffer must be a boolean, but got {self.double_buffer!r}.")
        if self.sample_transport not in _SAMPLE_TRANSPORTS:
            raise ValueError(
                f"sample_transport must be one of {_SAMPLE_TRANSPORTS}, "
                f"but got {self.sample_transport!r}."
            )


def build_distributed_dataset(
    dataset: Any,
    mesh: Any,
    config: DistributedDatasetConfig,
    *,
    metadata_fn: Callable[[Any, int], SampleMeta] | None = None,
    metadata: Sequence[SampleMeta] | None = None,
    collate_fn: Callable[[list[Any]], Any] | None = None,
    communication_device: Any = None,
    prepare_micro_batch: Callable[[Any], Any] | None = None,
    cost_model: CostModel | None = None,
) -> DistributedDataset:
    """Build a PyTorch-only online-planned distributed dataset.

    The builder creates dedicated ``metadata_group`` and model-parallel data
    groups from ``mesh``. All ranks must therefore call this function in the
    same control flow. Online metadata is the default path and balances one
    global microbatch at a time. An explicit sidecar ``metadata`` sequence
    enables whole-step inter-microbatch planning before sample reads. When
    ``config.double_buffer`` is enabled, every rank submits data collectives
    from one ordered producer thread.

    Args:
        dataset: Shared map-style dataset. Online mode reads only one local
            microbatch of raw Host candidates at a time and redistributes them
            before heavyweight target-rank decode.
        mesh: Named root training mesh.
        config: Batch planning, double buffering, and sample A2A transport configuration.
        metadata_fn: Derive ``SampleMeta`` from ``(raw_sample, data_ref)`` for
            microbatch-local online planning. Required without ``metadata``.
        metadata: Optional shared lightweight sidecar for whole-step planning.
        collate_fn: Target-owner transform and collation function. In online
            mode it receives the raw samples retained or received after planning.
        communication_device: Local collective device. Required for packed-byte
            owner A2A and tensor reception over HCCL/NCCL groups.
        prepare_micro_batch: Optional owner-side move/packing before communication.
            Double buffering invokes it on the data producer thread.
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
    metadata_group, model_parallel_group = _create_data_groups(topology)
    planner = DistributedBatchPlanner(
        topology.data_parallel_size,
        config.micro_batch_size,
        config.micro_batch_num,
        cost_model=cost_model,
        cp_shards=config.cp_shards,
    )
    online_sample_source = None
    sample_redistributor = None
    if metadata is None:
        if not hasattr(dataset, "__len__"):
            raise ValueError("Online metadata requires a map-style dataset implementing __len__.")
        complete_steps = len(dataset) // planner.global_samples_per_step
        metadata_source = None
        online_sample_source = StridedOnlineSampleSource(
            dataset,
            metadata_fn,
            topology.data_rank,
            topology.data_parallel_size,
            max_entries=complete_steps * planner.local_samples_per_step,
        )
        if topology.data_parallel_size == 1 or not topology.is_data_owner:
            sample_redistributor = LocalSampleRedistributor()
        else:
            topology.validate_metadata_group(metadata_group)
            if config.sample_transport == "packed_bytes_a2a":
                sample_redistributor = TorchPackedBytesRedistributor(
                    metadata_group,
                    communication_device=communication_device,
                )
            else:
                sample_redistributor = TorchTensorRedistributor(
                    metadata_group,
                    communication_device=communication_device,
                )
    else:
        complete_steps = len(metadata) // planner.global_samples_per_step
        metadata_source = StridedMetadataSource(
            metadata,
            topology.data_rank,
            topology.data_parallel_size,
            max_entries=complete_steps * planner.local_samples_per_step,
        )
    rank_materializer = RankMaterializer(MapDatasetMaterializer(dataset), collate_fn)

    if topology.data_parallel_size == 1:
        metadata_synchronizer = LocalMetadataSynchronizer()
    else:
        if topology.is_data_owner:
            topology.validate_metadata_group(metadata_group)
            metadata_synchronizer = TorchMetadataAllGather(metadata_group)
        else:
            metadata_synchronizer = LocalMetadataSynchronizer()

    if len(topology.model_parallel_ranks) == 1:
        micro_batch_distributor = LocalMicroBatchDistributor()
    else:
        topology.validate_model_parallel_group(model_parallel_group)
        micro_batch_distributor = TorchMicroBatchDistributor(
            model_parallel_group,
            communication_device=communication_device,
        )

    data_stream = platform.new_stream() if config.double_buffer and communication_device is not None else None
    return DistributedDataset(
        topology=topology,
        metadata_source=metadata_source,
        planner=planner,
        rank_materializer=rank_materializer,
        metadata_synchronizer=metadata_synchronizer,
        micro_batch_distributor=micro_batch_distributor,
        prefetch_steps=config.prefetch_steps,
        pin_memory=config.pin_memory,
        double_buffer=config.double_buffer,
        prepare_micro_batch=prepare_micro_batch,
        online_sample_source=online_sample_source,
        sample_redistributor=sample_redistributor,
        data_stream=data_stream,
    )
