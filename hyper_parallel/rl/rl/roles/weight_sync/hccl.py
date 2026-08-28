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
"""Actor rank-0 HCCL broadcast to rollout DP replicas."""

from concurrent.futures import ThreadPoolExecutor
import logging
import socket
import time
from typing import Any, Mapping, Optional, Union

from rl.roles.weight_sync.layout import DirectReshardPlan, TransferBucket
from rl.roles.weight_sync.sync import VLLMWeightSyncClientMixin, synchronize_error

from hyper_parallel import get_platform


platform = get_platform()
logger = logging.getLogger(__name__)


def _open_port() -> int:
    """Return an unused loopback port for one stateless HCCL group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class BroadcastDirectReshardHCCLTransport:
    """Broadcast only the FSDP fragments required by each rollout TP rank."""

    def __init__(
        self,
        *,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
    ) -> None:
        """Defer stateless route creation until both layouts are available."""
        if data_parallel_size <= 0:
            raise ValueError("Direct HCCL data_parallel_size must be positive")
        if tensor_parallel_size <= 0:
            raise ValueError("Direct HCCL tensor_parallel_size must be positive")
        self._data_parallel_size = int(data_parallel_size)
        self._tensor_parallel_size = int(tensor_parallel_size)
        self._groups: dict[tuple[int, int], Any] = {}
        self._group_ids: dict[tuple[int, int], str] = {}
        self._endpoint: Optional[str] = None
        self.last_metrics: dict[str, Union[float, int]] = {}

    @staticmethod
    def _trainer_init(init_info: Mapping[str, Any]) -> Any:
        """Create rank 0 of one source-to-target-TP stateless group."""
        from vllm_ascend.distributed.weight_transfer.hccl_engine import (  # pylint: disable=C0415
            HCCLWeightTransferEngine,
        )

        device = int(platform.get_device_handle(platform.device_type()).current_device())
        return HCCLWeightTransferEngine._stateless_init_process_group(  # pylint: disable=W0212
            init_info["master_address"],
            int(init_info["master_port"]),
            0,
            int(init_info["world_size"]),
            device=device,
        )

    @staticmethod
    def _shared_endpoint(client: VLLMWeightSyncClientMixin) -> str:
        """Require every Trainer rank to use one shared rollout endpoint."""
        world_size = platform.get_world_size()
        endpoints = [""] * world_size
        platform.all_gather_object(endpoints, client.base_url)
        unique = tuple(sorted(set(endpoints)))
        if len(unique) != 1 or any(not endpoint for endpoint in unique):
            raise RuntimeError(
                "Direct reshard requires one shared rollout endpoint, "
                f"got {endpoints}"
            )
        return unique[0]

    @staticmethod
    def _group_id(
        source_rank: int,
        tp_rank: int,
        replica_count: int,
        master_port: int,
    ) -> str:
        return (
            f"hyper-direct-s{source_rank}-t{tp_rank}-d{replica_count}"
            f"-p{master_port}"
        )

    def _initialize_route(
        self,
        client: VLLMWeightSyncClientMixin,
        endpoint: str,
        source_rank: int,
        tp_rank: int,
    ) -> None:
        """Join one source and all matching shared-deployment DP workers."""
        route = (source_rank, tp_rank)
        if route in self._group_ids:
            return
        local_rank = platform.get_rank()
        local_route_identity = None
        if local_rank == 0:
            master_port = _open_port()
            local_route_identity = (
                self._group_id(
                    source_rank,
                    tp_rank,
                    self._data_parallel_size,
                    master_port,
                ),
                master_port,
            )
        route_identities: list[Any] = [None] * platform.get_world_size()
        platform.all_gather_object(route_identities, local_route_identity)
        route_identity = route_identities[0]
        if (
            not isinstance(route_identity, tuple)
            or len(route_identity) != 2
            or any(identity is not None for identity in route_identities[1:])
        ):
            raise RuntimeError(
                f"Direct reshard route {route} rendezvous identity is invalid: {route_identities}"
            )
        group_id, master_port = route_identity
        world_size = 1 + self._data_parallel_size
        init_info = {
            "master_address": "127.0.0.1",
            "master_port": master_port,
            "world_size": world_size,
        }
        request = None
        executor = None
        worker_results = None
        group = None
        local_error = None
        try:
            if local_rank == 0:
                executor = ThreadPoolExecutor(max_workers=1)
                request = executor.submit(
                    client.collective_rpc,
                    "init_direct_reshard_group",
                    {
                        "group_id": group_id,
                        "target_tp_rank": tp_rank,
                        "master_address": "127.0.0.1",
                        "master_port": master_port,
                        "world_size": world_size,
                        "expected_data_parallel_size": self._data_parallel_size,
                        "expected_tensor_parallel_size": self._tensor_parallel_size,
                    },
                    endpoint,
                )
            if local_rank == source_rank:
                group = self._trainer_init(init_info)
            if request is not None:
                worker_results = request.result(timeout=180)
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        finally:
            if executor is not None:
                executor.shutdown(wait=False)
        synchronize_error(local_error, f"direct reshard group source={source_rank} tp={tp_rank}")
        gathered_results: list[Any] = [None] * platform.get_world_size()
        platform.all_gather_object(gathered_results, worker_results if local_rank == 0 else None)
        worker_results = gathered_results[0]
        if not isinstance(worker_results, list) or not worker_results:
            raise RuntimeError(
                f"Direct reshard group {group_id!r} returned invalid workers: {worker_results}"
            )
        for result in worker_results:
            if not isinstance(result, Mapping):
                raise RuntimeError(
                    f"Direct reshard group {group_id!r} returned invalid ACK: {result!r}"
                )
            result_tp_rank = int(result["tp_rank"])
            result_dp_rank = int(result["dp_rank"])
            result_group_rank = result.get("group_rank")
            if bool(result.get("joined")):
                expected_group_rank = 1 + result_dp_rank
                if result_tp_rank != tp_rank or int(result_group_rank) != expected_group_rank:
                    raise RuntimeError(
                        f"Direct reshard group {group_id!r} returned invalid ACK: {result}"
                    )
            elif result_tp_rank == tp_rank or result_group_rank is not None:
                raise RuntimeError(
                    f"Direct reshard group {group_id!r} returned invalid skip ACK: {result}"
                )
        if local_rank == source_rank:
            self._groups[route] = group
        self._group_ids[route] = group_id

    def ensure_groups(
        self,
        client: VLLMWeightSyncClientMixin,
        plan: DirectReshardPlan,
    ) -> str:
        """Create routes in deterministic source/TP order to avoid rendezvous races."""
        endpoint = self._shared_endpoint(client)
        if self._endpoint is not None and endpoint != self._endpoint:
            raise RuntimeError(
                f"Direct reshard rollout endpoint changed: {self._endpoint} -> {endpoint}"
            )
        if plan.destination_tp_size != self._tensor_parallel_size:
            raise RuntimeError(
                "Direct reshard plan TP size differs from configured topology: "
                f"expected={self._tensor_parallel_size}, actual={plan.destination_tp_size}"
            )
        self._endpoint = endpoint
        for source_rank in range(plan.source_world_size):
            for tp_rank in range(plan.destination_tp_size):
                if not plan.for_route(source_rank, tp_rank):
                    continue
                self._initialize_route(
                    client,
                    endpoint,
                    source_rank,
                    tp_rank,
                )
        return endpoint

    @staticmethod
    def _local_tensor(value: Any) -> Any:
        """Return a DTensor's local shard or the original plain tensor."""
        to_local = getattr(value, "to_local", None)
        return to_local() if callable(to_local) else value

    @classmethod
    def _pack_bucket(
        cls,
        state_dict: Mapping[str, Any],
        bucket: TransferBucket,
        device: Any,
    ) -> Any:
        """Pack source-local rectangular slices into one bounded NPU buffer."""
        import torch  # pylint: disable=C0415,forbidden-backend-import

        packed = torch.empty(bucket.total_bytes, dtype=torch.uint8, device=device)
        for entry in bucket.entries:
            value = state_dict.get(entry.name)
            if value is None:
                raise ValueError(f"Direct reshard source parameter {entry.name!r} is missing")
            local_tensor = cls._local_tensor(value)
            source_slice = tuple(
                slice(start, start + length)
                for start, length in zip(entry.source_starts, entry.lengths)
            )
            fragment = local_tensor[source_slice].detach().contiguous()
            if str(fragment.device) != str(device):
                fragment = fragment.to(device)
            raw = fragment.view(torch.uint8).view(-1)
            if int(raw.numel()) != entry.num_bytes:
                raise ValueError(
                    f"Direct reshard source fragment {entry.name!r} has "
                    f"{raw.numel()} bytes, expected {entry.num_bytes}"
                )
            packed.narrow(0, entry.buffer_offset, entry.num_bytes).copy_(raw)
        return packed

    def _broadcast_route(
        self,
        client: VLLMWeightSyncClientMixin,
        endpoint: str,
        state_dict: Mapping[str, Any],
        plan: DirectReshardPlan,
        source_rank: int,
        tp_rank: int,
        policy_version: int,
    ) -> tuple[int, int]:
        """Broadcast all bounded buckets for one source-to-TP route."""
        route = (source_rank, tp_rank)
        buckets = plan.for_route(source_rank, tp_rank)
        group_id = self._group_ids.get(route)
        if group_id is None:
            raise RuntimeError(f"Direct reshard route {route} has no HCCL group identity")
        receiver_payload = [bucket.worker_metadata() for bucket in buckets]
        local_rank = platform.get_rank()
        request = None
        executor = None
        worker_results = None
        sent_bytes = 0
        local_error = None
        try:
            if local_rank == 0:
                executor = ThreadPoolExecutor(max_workers=1)
                request = executor.submit(
                    client.collective_rpc,
                    "receive_direct_reshard",
                    {
                        "group_id": group_id,
                        "target_tp_rank": tp_rank,
                        "buckets": receiver_payload,
                        "policy_version": policy_version,
                        "expected_data_parallel_size": self._data_parallel_size,
                        "expected_tensor_parallel_size": self._tensor_parallel_size,
                    },
                    endpoint,
                )
            if local_rank == source_rank:
                group = self._groups.get(route)
                if group is None:
                    raise RuntimeError(f"Direct reshard route {route} has no HCCL group")
                for bucket in buckets:
                    packed = self._pack_bucket(state_dict, bucket, group.device)
                    platform.get_current_stream().synchronize()
                    group.broadcast(packed, src=0)
                    platform.get_current_stream().synchronize()
                    sent_bytes += bucket.total_bytes
                    del packed
            if request is not None:
                worker_results = request.result(timeout=600)
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        finally:
            if executor is not None:
                executor.shutdown(wait=False)
        synchronize_error(local_error, f"direct reshard transfer source={source_rank} tp={tp_rank}")
        expected_fragment_bytes = sum(
            entry.num_bytes for bucket in buckets for entry in bucket.entries
        )
        gathered_results: list[Any] = [None] * platform.get_world_size()
        platform.all_gather_object(gathered_results, worker_results if local_rank == 0 else None)
        worker_results = gathered_results[0]
        if not isinstance(worker_results, list) or not worker_results:
            raise RuntimeError(
                f"Direct reshard route {route} returned invalid results: {worker_results}"
            )
        for result in worker_results:
            if not isinstance(result, Mapping):
                raise RuntimeError(
                    f"Direct reshard route {route} returned invalid receive ACK: {result!r}"
                )
            result_tp_rank = int(result["tp_rank"])
            result_dp_rank = int(result["dp_rank"])
            if not 0 <= result_dp_rank < self._data_parallel_size:
                raise RuntimeError(
                    f"Direct reshard route {route} returned invalid DP rank: {result}"
                )
            if bool(result.get("received")):
                if result_tp_rank != tp_rank or int(result["bytes"]) != expected_fragment_bytes:
                    raise RuntimeError(
                        f"Direct reshard route {route} returned invalid receive ACK: {result}"
                    )
            elif result_tp_rank == tp_rank or int(result["bytes"]) != 0:
                raise RuntimeError(
                    f"Direct reshard route {route} returned invalid skip ACK: {result}"
                )
        return sent_bytes, expected_fragment_bytes

    def transfer(
        self,
        client: VLLMWeightSyncClientMixin,
        state_dict: Mapping[str, Any],
        plan: DirectReshardPlan,
        policy_version: int,
    ) -> None:
        """Execute every route without ever materializing one full Actor tensor."""
        group_started = time.perf_counter()
        endpoint = self.ensure_groups(client, plan)
        group_seconds = time.perf_counter() - group_started
        transfer_started = time.perf_counter()
        local_rank = platform.get_rank()
        sent_bytes = 0
        fragment_bytes = 0
        for source_rank in range(plan.source_world_size):
            for tp_rank in range(plan.destination_tp_size):
                if not plan.for_route(source_rank, tp_rank):
                    continue
                result = self._broadcast_route(
                    client,
                    endpoint,
                    state_dict,
                    plan,
                    source_rank,
                    tp_rank,
                    policy_version,
                )
                if local_rank == source_rank:
                    route_sent, route_fragments = result
                    sent_bytes += route_sent
                    fragment_bytes += route_fragments
        transfer_seconds = time.perf_counter() - transfer_started
        metric_values: list[Optional[dict[str, Union[float, int]]]] = [
            None
        ] * plan.source_world_size
        platform.all_gather_object(
            metric_values,
            {
                "sent_bytes": sent_bytes,
                "fragment_bytes": fragment_bytes,
            },
        )
        total_sent = sum(int(value["sent_bytes"]) for value in metric_values if value)
        total_fragments = sum(int(value["fragment_bytes"]) for value in metric_values if value)
        self.last_metrics = {
            "group_init_seconds": group_seconds,
            "transfer_seconds": transfer_seconds,
            "sent_bytes": total_sent,
            "fragment_bytes": total_fragments,
            "delivered_bytes": total_fragments * self._data_parallel_size,
            "route_count": plan.route_count,
            "fragment_count": plan.fragment_count,
        }
        if local_rank == 0:
            logger.info(
                "direct reshard completed: group_init=%.6fs transfer=%.6fs "
                "sent_gib=%.6f delivered_gib=%.6f routes=%d fragments=%d",
                group_seconds,
                transfer_seconds,
                total_sent / 2**30,
                total_fragments * self._data_parallel_size / 2**30,
                plan.route_count,
                plan.fragment_count,
            )

    def close(self) -> None:
        """Release trainer-side communicator references."""
        self._groups.clear()
        self._group_ids.clear()
        self._endpoint = None


__all__ = ["BroadcastDirectReshardHCCLTransport"]
