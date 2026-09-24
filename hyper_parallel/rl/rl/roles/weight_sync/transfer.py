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

__all__ = [
    "WeightPublisher",
    "build_weight_transfer",
]


import logging
from typing import Any, Mapping, Optional, Union

import torch
import torch.distributed as dist

from rl.roles.model_setup import VLLMModelRegistration
from rl.roles.weight_sync.config import validate_weight_sync_support
from rl.roles.weight_sync.hccl import HCCLWeightTransport
from rl.roles.weight_sync.ipc import IPCContext, IPCWeightTransport
from rl.roles.weight_sync.layout import (
    DestinationTensorLayout,
    DirectReshardPlan,
    SourceTensorLayout,
    build_direct_reshard_plan,
    resolve_destination_layouts,
    resolve_source_layouts,
)
from rl.roles.weight_sync.model_adapter import alias_tied_embeddings, build_model_weight_adapter
from rl.roles.weight_sync.packed_weight import (
    PackedWeightBucket,
    build_packed_weight_buckets,
    materialize_packed_weight_bucket,
)
from rl.roles.weight_sync.sync import PolicySnapshot, coordinator_call, synchronized_call
from rl.roles.weight_sync.vllm_client import (
    VLLMWeightSyncClientMixin,
    committed_policy_version,
    direct_reshard_workers,
)

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
        destinations = resolve_destination_layouts(
            coordinator_call(
                "direct reshard rollout layout query",
                lambda: direct_reshard_workers(
                    client,
                    data_parallel_size=self.data_parallel_size,
                    tensor_parallel_size=self.tensor_parallel_size,
                ),
            ),
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
            self._plan = build_direct_reshard_plan(
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
            packed = synchronized_call(
                "full-gather bucket materialization",
                lambda selected=bucket: materialize_packed_weight_bucket(
                    state,
                    selected,
                ),
            )
            ack = synchronized_call(
                "full-gather bucket transfer",
                lambda selected=bucket, payload=packed, index=bucket_index: (
                    transport.send_packed_bucket(
                        client,
                        self._context,
                        index,
                        self.source.adapter.packed_metadata(selected.worker_metadata()),
                        selected.total_bytes,
                        payload,
                        version,
                    )
                ),
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
            streaming_stats = synchronized_call(
                "weight-sync data transfer",
                lambda: self.strategy.execute(
                    client,
                    state,
                    self.transport,
                    snapshot.version,
                ),
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
