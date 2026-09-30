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
"""One publication transaction composed with a data strategy and an IPC/HCCL transport."""

from __future__ import annotations

__all__ = [
    "WeightPublisher",
    "build_weight_transfer",
]


import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

import torch
import torch.distributed as dist

from rl.weight_sync.config import validate_weight_sync_support
from rl.weight_sync.hccl import HCCLWeightTransport
from rl.weight_sync.ipc import IPCContext, IPCWeightTransport
from rl.weight_sync.layout import (
    DestinationTensorLayout,
    DirectReshardPlan,
    SourceTensorLayout,
    TensorRegion,
    TransferEntry,
    resolve_destination_layouts,
    resolve_source_layouts,
)
from rl.weight_sync.model_adapter import alias_tied_embeddings, build_model_weight_adapter
from rl.weight_sync.packed_weight import (
    PackedWeightBucket,
    build_direct_reshard_buckets,
    build_packed_weight_buckets,
    materialize_packed_weight_bucket,
)
from rl.weight_sync.sync import PolicySnapshot, coordinator_call, synchronized_call
from rl.weight_sync.vllm_client import (
    VLLMWeightSyncClientMixin,
    committed_policy_version,
    direct_reshard_workers,
)

# Annotation-only references must not initialize roles while weight_sync is importing.
if TYPE_CHECKING:
    from rl.roles.model_setup import VLLMModelRegistration

logger = logging.getLogger(__name__)
WeightTransport = Union[IPCWeightTransport, HCCLWeightTransport]


def _local_state_dict(payload: Any, *, operation: str) -> dict[str, Any]:
    """Extract FSDP-local model shards and validate their tensor-only contract."""
    state_dict = (
        dict(payload)
        if isinstance(payload, Mapping)
        else payload.state_dict()
    )
    invalid = next(
        ((name, value) for name, value in state_dict.items() if not torch.is_tensor(value)),
        None,
    )
    if invalid is not None:
        name, value = invalid
        state_dict.clear()
        raise ValueError(
            f"{operation} state entry {name!r} must be a tensor, got {type(value)!r}"
        )
    return state_dict


class WeightSource:
    """Map one Actor snapshot into the names used by both strategies."""

    def __init__(
        self,
        model: VLLMModelRegistration,
        bucket_size_bytes: int,
    ) -> None:
        """Store model-owned naming and the shared bucket setting."""
        self.model = model
        self.bucket_size_bytes = bucket_size_bytes
        self.adapter = build_model_weight_adapter(model)

    def local_state(self, payload: Any) -> dict[str, Any]:
        """Extract only local Actor shards and expose existing tied aliases."""
        self.adapter.bind_source(payload)
        state = _local_state_dict(payload, operation="weight publication")
        return alias_tied_embeddings(self.adapter.map_local_state_dict(state), self.model)


def _validate_plan_sources(names: frozenset[str], state: Mapping[str, Any]) -> None:
    missing = sorted(names - state.keys())
    if missing:
        raise ValueError(f"Actor state changed after layout planning; missing={missing}")


def _intersect_regions(
    source: TensorRegion,
    destination: TensorRegion,
) -> Optional[TensorRegion]:
    """Return the overlap between two tensor regions, if any."""
    starts = tuple(max(left, right) for left, right in zip(source.starts, destination.starts))
    ends = tuple(min(left, right) for left, right in zip(source.ends, destination.ends))
    lengths = tuple(end - start for start, end in zip(starts, ends))
    if any(length <= 0 for length in lengths):
        return None
    return TensorRegion(starts, lengths)


def _transfer_entry(
    source: SourceTensorLayout,
    destination: DestinationTensorLayout,
    intersection: TensorRegion,
) -> TransferEntry:
    """Translate one global intersection into source and destination offsets."""
    source_starts = tuple(
        local + start - base
        for local, start, base in zip(
            source.local_starts,
            intersection.starts,
            source.region.starts,
        )
    )
    destination_offsets = tuple(
        start - base
        for start, base in zip(
            intersection.starts,
            destination.region.starts,
        )
    )
    destination_starts = tuple(
        local + destination_offsets[axis]
        for local, axis in zip(
            destination.local_starts,
            destination.physical_permutation,
        )
    )
    return TransferEntry(
        name=source.name,
        dtype_name=source.dtype_name,
        element_size=source.element_size,
        source_starts=source_starts,
        destination_starts=destination_starts,
        lengths=intersection.lengths,
        destination_name=destination.target_name,
        source_name=source.source_key,
        destination_permutation=destination.physical_permutation,
        destination_dtype_name=destination.dtype_name,
        destination_element_size=destination.element_size,
    )


def _validate_coverage(
    name: str,
    destinations: Sequence[DestinationTensorLayout],
    coverage: Mapping[tuple[str, int], int],
) -> None:
    """Require every destination region to receive each value exactly once."""
    for destination in destinations:
        actual = coverage.get((name, destination.route_rank), 0)
        if actual != destination.region.numel:
            raise ValueError(
                f"Direct reshard plan covers {actual} values for {name!r} destination route "
                f"{destination.route_rank}, expected {destination.region.numel}"
            )


def _validate_transfer_contract(source, destination):
    """Require identical logical tensors or an explicitly accepted source dtype."""
    name = source.name
    dtype_compatible = (
        source.dtype_name == destination.dtype_name
        and source.element_size == destination.element_size
    ) or source.dtype_name in destination.accepted_source_dtypes
    if source.global_shape != destination.global_shape or not dtype_compatible:
        raise ValueError(
            f"Direct reshard tensor contract mismatch for {name!r}: "
            f"source={(source.global_shape, source.dtype_name)}, "
            f"destination={(destination.global_shape, destination.dtype_name)}"
        )


def _destination_worker_size(destinations):
    """Return the physical routing extent only for worker-specific layouts."""
    worker_ranks = [destination.worker_rank for destination in destinations if destination.worker_rank is not None]
    return max(worker_ranks) + 1 if worker_ranks else None


class DirectReshardStrategy:
    """Plan source-to-target intersections and execute them through either transport."""
    name = "direct_reshard"

    def __init__(
        self,
        source: WeightSource,
        *,
        data_parallel_size: int,
        tensor_parallel_size: int,
    ) -> None:
        """Own the direct plan cache."""
        self.source = source
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self._plan: Optional[DirectReshardPlan] = None
        self._parameter_names: frozenset[str] = frozenset()

    @staticmethod
    def build_direct_reshard_plan(
        sources: Sequence[SourceTensorLayout],
        destinations: Sequence[DestinationTensorLayout],
        *,
        source_world_size: int,
        bucket_size_bytes: int,
    ) -> DirectReshardPlan:
        """Compile global source/destination regions into bounded broadcast routes."""
        if bucket_size_bytes <= 0:
            raise ValueError("Direct reshard bucket_size_bytes must be positive")
        sources_by_name: dict[str, list[SourceTensorLayout]] = {}
        destinations_by_name: dict[str, list[DestinationTensorLayout]] = {}
        for source in sources:
            sources_by_name.setdefault(source.name, []).append(source)
        for destination in destinations:
            destinations_by_name.setdefault(destination.name, []).append(destination)
        if set(sources_by_name) != set(destinations_by_name):
            raise ValueError(
                "Direct reshard source/destination parameter mismatch: "
                f"source_only={sorted(set(sources_by_name) - set(destinations_by_name))}, "
                f"destination_only={sorted(set(destinations_by_name) - set(sources_by_name))}"
            )
        route_entries: dict[tuple[int, int], list[TransferEntry]] = {}
        coverage: dict[tuple[str, int], int] = {}
        for name in sorted(sources_by_name):
            for source in sources_by_name[name]:
                for destination in destinations_by_name[name]:
                    _validate_transfer_contract(source, destination)
                    intersection = _intersect_regions(
                        source.region,
                        destination.region,
                    )
                    if intersection is None:
                        continue
                    entry = _transfer_entry(source, destination, intersection)
                    route_entries.setdefault((source.source_rank, destination.route_rank), []).append(entry)
                    coverage[(name, destination.route_rank)] = (
                        coverage.get((name, destination.route_rank), 0) + entry.numel
                    )
            _validate_coverage(name, destinations_by_name[name], coverage)
        tp_sizes = {destination.tp_size for destination in destinations}
        if len(tp_sizes) != 1:
            raise ValueError(f"Direct reshard destination TP sizes differ: {sorted(tp_sizes)}")
        buckets = {
            route: build_direct_reshard_buckets(entries, bucket_size_bytes)
            for route, entries in route_entries.items()
        }
        return DirectReshardPlan(
            source_world_size=source_world_size,
            destination_tp_size=tp_sizes.pop(),
            bucket_size_bytes=bucket_size_bytes,
            buckets=buckets,
            destination_worker_size=_destination_worker_size(destinations),
        )

    def _layouts(
        self,
        client: VLLMWeightSyncClientMixin,
        state: Mapping[str, Any],
    ) -> tuple[
        tuple[SourceTensorLayout, ...],
        tuple[DestinationTensorLayout, ...],
    ]:
        """Resolve Actor and rollout metadata into physical layouts."""
        descriptions = self.source.adapter.direct_source_descriptions(
            state,
            dist.get_rank(),
        )
        rank_descriptions = [None] * dist.get_world_size()
        dist.all_gather_object(rank_descriptions, descriptions)
        sources = resolve_source_layouts(rank_descriptions)

        def query_destinations() -> Any:
            """Query every rollout worker through the synchronized coordinator."""
            return direct_reshard_workers(
                client,
                data_parallel_size=self.data_parallel_size,
                tensor_parallel_size=self.tensor_parallel_size,
            )

        destinations = resolve_destination_layouts(
            coordinator_call("direct reshard rollout layout query", query_destinations),
            {source.name: source.global_shape for source in sources},
        )
        return sources, destinations

    def prepare(
        self, client: VLLMWeightSyncClientMixin, state: Mapping[str, Any], transport: WeightTransport,
    ) -> None:
        """Build the direct plan once and validate its source contract."""
        del transport
        if self._plan is None:
            sources, destinations = self._layouts(client, state)
            self._plan = self.build_direct_reshard_plan(
                sources, destinations, source_world_size=dist.get_world_size(),
                bucket_size_bytes=self.source.bucket_size_bytes,
            )
            self._parameter_names = frozenset(source.source_key for source in sources)
            if dist.get_rank() == 0:
                logger.info(
                    "direct-reshard plan: family=%s source_world_size=%s destination_tp=%s "
                    "destination_workers=%s routes=%s fragments=%s max_bucket_bytes=%s",
                    self.source.model.family, self._plan.source_world_size, self._plan.destination_tp_size,
                    self.data_parallel_size * self.tensor_parallel_size,
                    self._plan.route_count, self._plan.fragment_count,
                    max(bucket.total_bytes for buckets in self._plan.buckets.values() for bucket in buckets),
                )
        _validate_plan_sources(self._parameter_names, state)

    def execute(
        self, client: VLLMWeightSyncClientMixin, state: Mapping[str, Any], transport: WeightTransport,
        version: int,
    ) -> Optional[dict[str, Any]]:
        """Transfer direct buckets while the shared publisher owns start and finish."""
        if self._plan is None:
            raise RuntimeError("Direct reshard strategy has no prepared plan")
        synchronized_call("direct producer synchronization", torch.get_device_module().current_stream().synchronize)
        transport.transfer_direct(client, state, self._plan, version)


class FullGatherStrategy:
    """Gather complete HF parameters and let vLLM load packed buckets."""
    name = "full_gather"

    def __init__(self, source: WeightSource) -> None:
        """Own the immutable whole-parameter bucket plan."""
        self.source = source
        self._buckets: Optional[tuple[PackedWeightBucket, ...]] = None
        self._parameter_names: frozenset[str] = frozenset()
        self._context: Optional[Union[IPCContext, str]] = None

    def prepare(self, client: VLLMWeightSyncClientMixin, state: Mapping[str, Any], transport: WeightTransport) -> None:
        """Build whole-parameter buckets and initialize one full-gather route."""
        if self._buckets is None:
            skip_names = (
                frozenset(("lm_head.weight",))
                if self.source.model.model.tie_word_embeddings
                else frozenset()
            )
            self._buckets = build_packed_weight_buckets(
                state,
                self.source.bucket_size_bytes,
                skip_names=skip_names,
            )
            self._parameter_names = frozenset(
                entry.name for bucket in self._buckets for entry in bucket.entries
            )
            if dist.get_rank() == 0:
                logger.info(
                    "packed full-gather plan: family=%s buckets=%s parameters=%s "
                    "bytes=%s max_bucket_bytes=%s",
                    self.source.model.family,
                    len(self._buckets),
                    sum(len(bucket.entries) for bucket in self._buckets),
                    sum(bucket.total_bytes for bucket in self._buckets),
                    max(bucket.total_bytes for bucket in self._buckets),
                )
        _validate_plan_sources(self._parameter_names, state)
        signature = tuple(
            (
                entry.name,
                entry.dtype_name,
                entry.shape,
                entry.buffer_offset,
                entry.num_bytes,
            )
            for bucket in self._buckets
            for entry in bucket.entries
        )
        signatures = [None] * dist.get_world_size()
        dist.all_gather_object(signatures, signature)
        if any(candidate != signature for candidate in signatures):
            raise RuntimeError(
                "Packed full-gather parameter order differs across Trainer ranks"
            )
        self._context = transport.prepare_packed(client)

    def execute(
        self, client: VLLMWeightSyncClientMixin, state: Mapping[str, Any], transport: WeightTransport,
        version: int,
    ) -> Optional[dict[str, Any]]:
        """Materialize and send one complete-parameter bucket at a time."""
        if self._buckets is None or self._context is None:
            raise RuntimeError("Full gather strategy has no prepared buckets")
        max_bucket_bytes = 0
        for bucket_index, bucket in enumerate(self._buckets):
            def materialize(selected: Any = bucket) -> Any:
                """Materialize the selected bucket under synchronized failure handling."""
                return materialize_packed_weight_bucket(state, selected)

            packed = synchronized_call(
                "full-gather bucket materialization",
                materialize,
            )

            def send_bucket(selected: Any = bucket, payload: Any = packed, index: int = bucket_index) -> Any:
                """Send the selected payload and retain its bucket identity."""
                return transport.send_packed_bucket(
                    client,
                    self._context,
                    index,
                    self.source.adapter.packed_metadata(selected.worker_metadata()),
                    selected.total_bytes,
                    payload,
                    version,
                )

            ack = synchronized_call(
                "full-gather bucket transfer",
                send_bucket,
            )
            if (
                ack.bucket_index != bucket_index
                or ack.total_bytes != bucket.total_bytes
                or ack.worker_count <= 0
            ):
                raise RuntimeError(
                    "Packed full-gather ACK differs from its bucket: "
                    f"bucket={bucket_index}, ack={ack}"
                )
            max_bucket_bytes = max(max_bucket_bytes, bucket.total_bytes)
            del packed
        synchronized_call(
            "full-gather post-release synchronization", torch.get_device_module().current_stream().synchronize,
        )
        bucket_count = len(self._buckets)
        total_bytes = sum(bucket.total_bytes for bucket in self._buckets)
        parameter_count = sum(len(bucket.entries) for bucket in self._buckets)
        return {
            "bucket_count": bucket_count,
            "fragment_count": parameter_count,
            "total_bytes": total_bytes,
            "max_gathered_bytes": max_bucket_bytes,
            "max_packed_bytes": max_bucket_bytes,
            "max_ipc_shared_bytes": max_bucket_bytes if transport.name == "ipc" else 0,
            "max_hccl_bytes": max_bucket_bytes if transport.name == "hccl" else 0,
            "max_transport_bytes": max_bucket_bytes * 2,
            "transport_buffer_count": 2 if bucket_count else 0,
            "max_inflight_buckets": 1 if bucket_count else 0,
            "acked_buckets": bucket_count,
            "released_buckets": bucket_count,
        }


class WeightPublisher:
    """Run one publication transaction through the configured strategy and transport."""

    def __init__(
        self,
        strategy: Union[DirectReshardStrategy, FullGatherStrategy],
        transport: WeightTransport,
    ) -> None:
        """Own the selected strategy, transport, and latest result."""
        self.strategy = strategy
        self.transport = transport
        self.configured_strategy = strategy.name
        self.last_strategy: Optional[str] = None
        self.last_streaming_stats: Optional[dict[str, Any]] = None

    def publish(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Publish once; any failure propagates and terminates the training run."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError("Weight publication requires the vLLM HTTP control client")
        self.last_streaming_stats = None
        state = synchronized_call(
            "weight-sync local-state extraction",
            lambda: self.strategy.source.local_state(snapshot.payload),
        )
        try:
            synchronized_call(
                "weight-sync layout preparation",
                lambda: self.strategy.prepare(client, state, self.transport),
            )
            coordinator_call("weight-sync pause", client.pause)
            coordinator_call("weight-sync start", client.start_weight_update)

            def execute_transfer() -> Optional[dict[str, Any]]:
                """Execute the prepared publication strategy on every rank."""
                return self.strategy.execute(client, state, self.transport, snapshot.version)

            streaming_stats = synchronized_call(
                "weight-sync data transfer",
                execute_transfer,
            )
            coordinator_call("weight-sync finish", client.finish_weight_update)
            committed_version = coordinator_call(
                "weight-sync committed version",
                lambda: committed_policy_version(client),
            )
            if committed_version != snapshot.version:
                raise RuntimeError(
                    "vLLM committed a different policy version: "
                    f"expected={snapshot.version}, actual={committed_version}"
                )
            self.last_strategy = self.strategy.name
            self.last_streaming_stats = streaming_stats
            if dist.get_rank() == 0:
                logger.info(
                    "weight-sync publication complete: strategy=%s version=%s stats=%s",
                    self.strategy.name,
                    snapshot.version,
                    streaming_stats,
                )
        finally:
            state.clear()

    def close(self) -> None:
        """Release communication resources after rollout shutdown."""
        self.transport.close()


def build_weight_transfer(
    deployment: str, model: VLLMModelRegistration, *, tensor_parallel_size: int = 1,
    data_parallel_size: int = 1, bucket_size_bytes: int = 128 * 2**20,
    strategy: str = "full_gather",
) -> WeightPublisher:
    """Compose one data strategy with the deployment transport."""
    validate_weight_sync_support(
        deployment=deployment, model_family=model.family, rollout_tp=tensor_parallel_size,
        strategy=strategy,
    )
    if data_parallel_size <= 0 or bucket_size_bytes <= 0:
        raise ValueError("Weight-sync DP size and bucket bytes must be positive")
    source = WeightSource(model, bucket_size_bytes)
    transport_type = IPCWeightTransport if deployment == "colocated" else HCCLWeightTransport
    transport = transport_type(data_parallel_size=data_parallel_size, tensor_parallel_size=tensor_parallel_size)
    primary = (
        DirectReshardStrategy(
            source,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        if strategy == "direct_reshard"
        else FullGatherStrategy(source)
    )
    return WeightPublisher(primary, transport)
