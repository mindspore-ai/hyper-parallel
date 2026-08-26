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
from hyper_parallel.distributed_data.fetcher import (
    MapDatasetFetcher,
    MicroBatchFetcher,
    StridedMetadataSource,
    StridedOnlineSampleSource,
)
from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import SampleMeta, TensorShardSpec
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.torch_loader import TorchLocalDataLoader
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
    ``direct_tensor_a2a`` requires every owner sample to be one tensor and one
    dtype per global microbatch; tensor shapes may vary. With sidecar metadata,
    ``prefetch_steps`` bounds complete optimizer steps that are read, collated,
    and made ready in Host memory. Online metadata preserves microbatch-local
    planning, so the same option bounds Host-ready microbatches instead.
    ``double_buffer`` independently keeps at most one device/communication-ready
    microbatch ahead. ``prefetch_factor`` controls native PyTorch worker-task
    look-ahead, while ``num_workers`` controls the worker count. ``pin_memory``
    pins each final collated batch on a dedicated thread. Online candidates are
    pinned only after planning and redistribution. Distributed collectives
    always remain in the training rank process. ``raw_sample_size`` is the
    number of dataset records assigned to each data owner per microbatch.
    """

    raw_sample_size: int
    micro_batch_num: int
    prefetch_steps: int = 2
    sample_transport: str = "packed_bytes_a2a"
    dp_dim_names: tuple[str, ...] | None = None
    cp_shards: tuple[TensorShardSpec, ...] = ()
    pin_memory: bool = False
    double_buffer: bool = False
    num_workers: int = 0
    prefetch_factor: int = 2

    def __post_init__(self) -> None:
        for name in ("raw_sample_size", "micro_batch_num", "prefetch_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        if not isinstance(self.pin_memory, bool):
            raise ValueError(f"pin_memory must be a boolean, but got {self.pin_memory!r}.")
        if not isinstance(self.double_buffer, bool):
            raise ValueError(f"double_buffer must be a boolean, but got {self.double_buffer!r}.")
        if not isinstance(self.num_workers, int) or isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError(f"num_workers must be a non-negative integer, but got {self.num_workers!r}.")
        if (
            not isinstance(self.prefetch_factor, int)
            or isinstance(self.prefetch_factor, bool)
            or self.prefetch_factor < 1
        ):
            raise ValueError(f"prefetch_factor must be a positive integer, but got {self.prefetch_factor!r}.")
        if self.sample_transport not in _SAMPLE_TRANSPORTS:
            raise ValueError(
                f"sample_transport must be one of {_SAMPLE_TRANSPORTS}, "
                f"but got {self.sample_transport!r}."
            )


def _validate_builder_metadata(
    metadata_fn: Callable[[Any, int], SampleMeta] | None,
    metadata: Sequence[SampleMeta] | None,
) -> None:
    if platform.platform_type != PlatformType.PYTORCH:
        raise ValueError("The distributed dataset MVP currently supports only PyTorch.")
    if metadata is None and metadata_fn is None:
        raise ValueError("Online metadata requires metadata_fn when no sidecar metadata is provided.")
    if metadata is not None and metadata_fn is not None:
        raise ValueError("Provide either online metadata_fn or sidecar metadata, but not both.")


def _create_local_data_loader(
    dataset: Any,
    metadata_fn: Callable[[Any, int], SampleMeta] | None,
    metadata: Sequence[SampleMeta] | None,
    collate_fn: Callable[[list[Any]], Any] | None,
    worker_init_fn: Callable[[int], None] | None,
    topology: DataTopology,
    config: DistributedDatasetConfig,
) -> TorchLocalDataLoader | None:
    if not topology.is_data_owner:
        return None
    return TorchLocalDataLoader(
        dataset,
        metadata_fn=metadata_fn,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory if metadata is not None else False,
        worker_init_fn=worker_init_fn,
        online_metadata=metadata is None,
    )


def _create_online_sample_source(
    dataset: Any,
    metadata_fn: Callable[[Any, int], SampleMeta],
    planner: DistributedBatchPlanner,
    topology: DataTopology,
    local_data_loader: TorchLocalDataLoader | None,
) -> StridedOnlineSampleSource:
    if not hasattr(dataset, "__len__"):
        raise ValueError("Online metadata requires a map-style dataset implementing __len__.")
    complete_steps = len(dataset) // planner.global_samples_per_step
    return StridedOnlineSampleSource(
        dataset,
        metadata_fn,
        topology.data_rank,
        topology.data_parallel_size,
        max_entries=complete_steps * planner.local_samples_per_step,
        local_data_loader=local_data_loader,
    )


def _create_sidecar_metadata_source(
    metadata: Sequence[SampleMeta],
    planner: DistributedBatchPlanner,
    topology: DataTopology,
) -> StridedMetadataSource:
    complete_steps = len(metadata) // planner.global_samples_per_step
    return StridedMetadataSource(
        metadata,
        topology.data_rank,
        topology.data_parallel_size,
        max_entries=complete_steps * planner.local_samples_per_step,
    )


def _create_sample_redistributor(
    topology: DataTopology,
    config: DistributedDatasetConfig,
    metadata_group: Any,
    communication_device: Any,
) -> Any:
    if topology.data_parallel_size == 1 or not topology.is_data_owner:
        return LocalSampleRedistributor()
    topology.validate_metadata_group(metadata_group)
    if config.sample_transport == "packed_bytes_a2a":
        return TorchPackedBytesRedistributor(metadata_group, communication_device=communication_device)
    return TorchTensorRedistributor(metadata_group, communication_device=communication_device)


def _create_metadata_synchronizer(topology: DataTopology, metadata_group: Any) -> Any:
    if topology.data_parallel_size == 1 or not topology.is_data_owner:
        return LocalMetadataSynchronizer()
    topology.validate_metadata_group(metadata_group)
    return TorchMetadataAllGather(metadata_group)


def _create_micro_batch_distributor(
    topology: DataTopology,
    model_parallel_group: Any,
    communication_device: Any,
) -> Any:
    if len(topology.model_parallel_ranks) == 1:
        return LocalMicroBatchDistributor()
    topology.validate_model_parallel_group(model_parallel_group)
    return TorchMicroBatchDistributor(model_parallel_group, communication_device=communication_device)


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
    worker_init_fn: Callable[[int], None] | None = None,
) -> DistributedDataset:
    """Build an online-planned distributed dataset.

    The builder creates dedicated ``metadata_group`` and model-parallel data
    groups from ``mesh``. All ranks must therefore call this function in the
    same control flow. Online metadata is the default path and balances one
    global microbatch at a time. An explicit sidecar ``metadata`` sequence
    enables whole-step inter-microbatch planning before sample reads. Host
    prefetch performs real dataset reads and CPU preprocessing; device double
    buffering remains a separate one-microbatch stage. Every rank submits data
    collectives from one ordered producer thread.

    Args:
        dataset: Shared map-style dataset. Online mode reads a bounded number
            of local raw Host-candidate microbatches and redistributes each one
            before heavyweight target-rank decode.
        mesh: Named root training mesh.
        config: Batch planning, double buffering, and sample A2A transport configuration.
        metadata_fn: Derive ``SampleMeta`` from ``(raw_sample, sample_id)`` for
            microbatch-local online planning. Required without ``metadata``.
        metadata: Optional shared lightweight sidecar for whole-step planning.
        collate_fn: Target-owner transform and collation function. In online
            mode it receives the raw samples retained or received after planning.
        communication_device: Local collective device. Required for packed-byte
            owner A2A and tensor reception over HCCL/NCCL groups.
        prepare_micro_batch: Optional owner-side move or packing in the
            device/communication stage. Double buffering runs this stage ahead.
        cost_model: Optional calibrated workload cost model.
        worker_init_fn: Optional callback forwarded to the native data loader
            and invoked once in each local worker process.

    Returns:
        Configured :class:`DistributedDataset` iterator.
    """
    _validate_builder_metadata(metadata_fn, metadata)
    topology = DataTopology.from_mesh(mesh, dp_dim_names=config.dp_dim_names)
    metadata_group, model_parallel_group = _create_data_groups(topology)
    planner = DistributedBatchPlanner(
        topology.data_parallel_size,
        config.raw_sample_size,
        config.micro_batch_num,
        cost_model=cost_model,
        cp_shards=config.cp_shards,
    )
    local_data_loader = _create_local_data_loader(
        dataset,
        metadata_fn,
        metadata,
        collate_fn,
        worker_init_fn,
        topology,
        config,
    )
    online_sample_source = None
    sample_redistributor = None
    if metadata is None:
        metadata_source = None
        online_sample_source = _create_online_sample_source(
            dataset, metadata_fn, planner, topology, local_data_loader
        )
        sample_redistributor = _create_sample_redistributor(
            topology, config, metadata_group, communication_device
        )
    else:
        metadata_source = _create_sidecar_metadata_source(metadata, planner, topology)
    sidecar_data_loader = local_data_loader if metadata is not None else None
    micro_batch_fetcher = MicroBatchFetcher(MapDatasetFetcher(dataset), collate_fn, sidecar_data_loader)
    metadata_synchronizer = _create_metadata_synchronizer(topology, metadata_group)
    micro_batch_distributor = _create_micro_batch_distributor(
        topology, model_parallel_group, communication_device
    )
    data_stream = platform.new_stream() if config.double_buffer and communication_device is not None else None
    return DistributedDataset(
        topology=topology,
        metadata_source=metadata_source,
        planner=planner,
        micro_batch_fetcher=micro_batch_fetcher,
        metadata_synchronizer=metadata_synchronizer,
        micro_batch_distributor=micro_batch_distributor,
        prefetch_steps=config.prefetch_steps,
        pin_memory=config.pin_memory,
        double_buffer=config.double_buffer,
        prepare_micro_batch=prepare_micro_batch,
        online_sample_source=online_sample_source,
        sample_redistributor=sample_redistributor,
        data_stream=data_stream,
        local_data_loader=local_data_loader,
    )
