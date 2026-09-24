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
"""HCCL routes from Trainer producers to rollout workers."""

__all__ = ["HCCLWeightTransport"]


import logging
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import torch
import torch.distributed as dist

from rl.roles.weight_sync.layout import DirectReshardPlan, pack_direct_bucket
from rl.roles.weight_sync.packed_weight import PackedWeightAck
from rl.roles.weight_sync.sync import coordinator_call, synchronize_error, synchronized_call
from rl.roles.weight_sync.vllm_client import VLLMWeightSyncClientMixin, shared_endpoint

logger = logging.getLogger(__name__)


def _open_port() -> int:
    """Return an unused loopback port for one stateless HCCL group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _rpc_with_collective(
    client: VLLMWeightSyncClientMixin, endpoint: str, method: str,
    kwargs: Mapping[str, Any], local_action: Callable[[], Any], *, timeout: int,
) -> tuple[Any, Any]:
    """Run worker RPC concurrently with Trainer collectives, then synchronize errors."""
    rank = dist.get_rank()
    executor = None
    request = None
    results = None
    local_result = None
    error = None
    try:
        if rank == 0:
            executor = ThreadPoolExecutor(max_workers=1)
            request = executor.submit(client.collective_rpc, method, kwargs, endpoint)
        local_result = local_action()
        if request is not None:
            results = request.result(timeout=timeout)
    # Every local failure must be exchanged before peers enter the next collective.
    except Exception as caught:  # pylint: disable=broad-exception-caught
        error = caught
    finally:
        if executor is not None:
            # A failed collective must not wait indefinitely for its peer RPC.
            executor.shutdown(wait=False)
    synchronize_error(error, f"{method} group={kwargs['group_id']}")
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, results if rank == 0 else None)
    return local_result, gathered[0]


class HCCLWeightTransport:
    """Own HCCL routes and broadcasts for both direct and gather publication."""
    name = "hccl"

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
        self._packed_group: Optional[Any] = None
        self._packed_group_id: Optional[str] = None
        self._endpoint: Optional[str] = None
        self._failed_buffers: list[Any] = []

    @staticmethod
    def _trainer_init(init_info: Mapping[str, Any]) -> Any:
        """Create rank 0 of one stateless weight-transfer group."""
        from vllm_ascend.distributed.weight_transfer.hccl_engine import (  # pylint: disable=C0415
            HCCLWeightTransferEngine,
        )

        device = int(torch.get_device_module().current_device())
        return HCCLWeightTransferEngine._stateless_init_process_group(  # pylint: disable=W0212
            init_info["master_address"],
            int(init_info["master_port"]),
            0,
            int(init_info["world_size"]),
            device=device,
        )

    def _resolve_endpoint(self, client: VLLMWeightSyncClientMixin) -> str:
        """Bind both transmission paths to one immutable shared endpoint."""
        endpoint = shared_endpoint(client)
        if self._endpoint is not None and endpoint != self._endpoint:
            raise RuntimeError(f"Weight-sync rollout endpoint changed: {self._endpoint} -> {endpoint}")
        self._endpoint = endpoint
        return endpoint

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
        local_rank = dist.get_rank()
        master_port = coordinator_call("HCCL rendezvous port", _open_port)
        group_id = self._group_id(source_rank, tp_rank, self._data_parallel_size, master_port)
        init_info = {
            "master_address": "127.0.0.1",
            "master_port": master_port,
            "world_size": 1 + self._data_parallel_size,
        }

        def join() -> Any:
            """Only this route's producer joins the stateless group."""
            return self._trainer_init(init_info) if local_rank == source_rank else None

        group, worker_results = _rpc_with_collective(
            client, endpoint, "init_direct_reshard_group",
            {"group_id": group_id, "target_tp_rank": tp_rank, **init_info,
             "expected_data_parallel_size": self._data_parallel_size,
             "expected_tensor_parallel_size": self._tensor_parallel_size},
            join, timeout=180,
        )
        self._validate_route_workers(worker_results, group_id, tp_rank)
        if local_rank == source_rank:
            self._groups[route] = group
        self._group_ids[route] = group_id

    @staticmethod
    def _validate_route_workers(worker_results: Any, group_id: str, tp_rank: int) -> None:
        """Validate joined and skipped worker acknowledgements for one route."""
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

    def ensure_groups(
        self,
        client: VLLMWeightSyncClientMixin,
        plan: DirectReshardPlan,
    ) -> str:
        """Create routes in deterministic source/TP order to avoid rendezvous races."""
        endpoint = self._resolve_endpoint(client)
        if plan.destination_tp_size != self._tensor_parallel_size:
            raise RuntimeError(
                "Direct reshard plan TP size differs from configured topology: "
                f"expected={self._tensor_parallel_size}, actual={plan.destination_tp_size}"
            )
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

    def _broadcast_route(
        self, client: VLLMWeightSyncClientMixin, endpoint: str, state_dict: Mapping[str, Any],
        plan: DirectReshardPlan, source_rank: int, tp_rank: int, policy_version: int,
    ) -> tuple[int, int]:
        """Pack and broadcast the original ordered buckets for one direct route."""
        buckets = plan.for_route(source_rank, tp_rank)
        def materialize(index: int) -> Any:
            """Only the route's source rank evaluates the producer callback."""
            return pack_direct_bucket(state_dict, buckets[index], self._groups[(source_rank, tp_rank)].device)
        return self._broadcast_buffers(
            client, endpoint, source_rank, tp_rank,
            [bucket.worker_metadata() for bucket in buckets], materialize, policy_version,
        )

    def transfer_direct(
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
        local_rank = dist.get_rank()
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
        dist.all_gather_object(
            metric_values,
            {
                "sent_bytes": sent_bytes,
                "fragment_bytes": fragment_bytes,
            },
        )
        total_sent = sum(int(value["sent_bytes"]) for value in metric_values if value)
        total_fragments = sum(int(value["fragment_bytes"]) for value in metric_values if value)
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

    def _broadcast_buffers(
        self, client: VLLMWeightSyncClientMixin, endpoint: str, source_rank: int, tp_rank: int,
        metadata: Sequence[Mapping[str, Any]], materialize: Callable[[int], Any], policy_version: int,
    ) -> tuple[int, int]:
        """Receive one route asynchronously while its source broadcasts bounded buffers."""
        route = (source_rank, tp_rank)
        group_id = self._group_ids.get(route)
        if group_id is None:
            raise RuntimeError(f"Weight-sync HCCL route {route} has no group identity")

        def broadcast() -> int:
            """Pack and broadcast each bucket only on its source rank."""
            if dist.get_rank() != source_rank:
                return 0
            group = self._groups.get(route)
            if group is None:
                raise RuntimeError(f"Weight-sync HCCL route {route} has no producer group")
            packed = None
            sent_bytes = 0
            try:
                for index, description in enumerate(metadata):
                    packed = materialize(index)
                    if int(packed.numel()) != int(description["total_bytes"]):
                        raise ValueError("HCCL packed buffer differs from bucket metadata")
                    torch.get_device_module().current_stream().synchronize()
                    group.broadcast(packed, src=0)
                    torch.get_device_module().current_stream().synchronize()
                    sent_bytes += int(description["total_bytes"])
                    packed = None
            except Exception:
                if packed is not None:
                    self._failed_buffers.append(packed)
                raise
            return sent_bytes

        sent_bytes, results = _rpc_with_collective(
            client, endpoint, "receive_direct_reshard",
            {"group_id": group_id, "target_tp_rank": tp_rank, "buckets": list(metadata),
             "policy_version": policy_version, "expected_data_parallel_size": self._data_parallel_size,
             "expected_tensor_parallel_size": self._tensor_parallel_size},
            broadcast, timeout=600,
        )
        copied_bytes = sum(int(entry["num_bytes"]) for bucket in metadata for entry in bucket["entries"])
        self._validate_receive_results(results, tp_rank, copied_bytes)
        return sent_bytes, copied_bytes

    def _validate_receive_results(self, results: Any, tp_rank: int, expected_bytes: int) -> None:
        """Validate representative DP results after workers check their coordinates."""
        if not isinstance(results, list) or not results:
            raise RuntimeError(f"HCCL returned invalid receive results: {results}")
        coordinates = set()
        received = set()
        for result in results:
            if not isinstance(result, Mapping):
                raise RuntimeError(f"HCCL returned an invalid receive ACK: {result}")
            coordinate = (int(result["dp_rank"]), int(result["tp_rank"]))
            if (coordinate in coordinates or not 0 <= coordinate[0] < self._data_parallel_size
                    or not 0 <= coordinate[1] < self._tensor_parallel_size):
                raise RuntimeError(f"HCCL receive ACK has an invalid or duplicate rank: {result}")
            coordinates.add(coordinate)
            intended = coordinate[1] == tp_rank
            if result.get("received") is not intended or int(result["bytes"]) != (expected_bytes if intended else 0):
                raise RuntimeError(f"HCCL receive ACK differs from its route: {result}")
            if intended:
                received.add(coordinate)
        if not received:
            raise RuntimeError(f"HCCL returned no representative ACK for TP{tp_rank}: {results}")

    def prepare_packed(self, client: VLLMWeightSyncClientMixin) -> str:
        """Create one rank-zero producer group containing every rollout worker."""
        endpoint = self._resolve_endpoint(client)
        if self._packed_group_id is not None:
            return endpoint
        rank = dist.get_rank()
        master_port = coordinator_call("HCCL rendezvous port", _open_port)
        group_id = f"hyper-packed-d{self._data_parallel_size}-t{self._tensor_parallel_size}-p{master_port}"
        init_info = {
            "master_address": "127.0.0.1",
            "master_port": master_port,
            "world_size": 1 + self._data_parallel_size * self._tensor_parallel_size,
        }

        def join() -> Any:
            """Only Trainer rank zero joins the full-gather producer group."""
            return self._trainer_init(init_info) if rank == 0 else None

        group, worker_results = _rpc_with_collective(
            client, endpoint, "init_packed_weight_group",
            {"group_id": group_id, **init_info,
             "expected_data_parallel_size": self._data_parallel_size,
             "expected_tensor_parallel_size": self._tensor_parallel_size},
            join, timeout=180,
        )
        if not isinstance(worker_results, list) or not worker_results:
            raise RuntimeError(
                f"Packed full-gather returned invalid workers: {worker_results}"
            )
        coordinates = set()
        for result in worker_results:
            if not isinstance(result, Mapping) or result.get("joined") is not True:
                raise RuntimeError(
                    f"Packed full-gather returned an invalid group ACK: {result}"
                )
            coordinate = (int(result["dp_rank"]), int(result["tp_rank"]))
            expected_rank = 1 + coordinate[0] * self._tensor_parallel_size + coordinate[1]
            if (
                coordinate in coordinates
                or not 0 <= coordinate[0] < self._data_parallel_size
                or not 0 <= coordinate[1] < self._tensor_parallel_size
                or int(result["group_rank"]) != expected_rank
            ):
                raise RuntimeError(
                    f"Packed full-gather returned an invalid worker rank: {result}"
                )
            coordinates.add(coordinate)
        if rank == 0:
            self._packed_group = group
        self._packed_group_id = str(group_id)
        return endpoint

    def send_packed_bucket(
        self,
        client: VLLMWeightSyncClientMixin,
        context: str,
        bucket_index: int,
        metadata: list[Mapping[str, Any]],
        total_bytes: int,
        packed: Optional[Any],
        policy_version: int,
    ) -> PackedWeightAck:
        """Broadcast one complete-parameter bucket to every rollout worker."""
        if context != self._endpoint or self._packed_group_id is None:
            raise RuntimeError("Packed full-gather HCCL group is not initialized")
        rank = dist.get_rank()

        def validate_buffer() -> None:
            """Reject invalid local buffers before starting a worker receive RPC."""
            nonlocal packed
            if rank != 0:
                if packed is not None:
                    raise ValueError("Only Trainer rank 0 may own a packed HCCL buffer")
                return
            if packed is None or int(packed.numel()) != total_bytes:
                raise ValueError("Packed HCCL producer buffer differs from its metadata")
            if self._packed_group is None:
                raise RuntimeError("Packed HCCL producer group is missing")
            # CPU-offloaded weights must be staged before peers enter HCCL.
            packed = packed.to(self._packed_group.device)

        def broadcast() -> None:
            """Only rank zero broadcasts the complete-parameter buffer."""
            if rank != 0:
                return
            torch.get_device_module().current_stream().synchronize()
            self._packed_group.broadcast(packed, src=0)
            torch.get_device_module().current_stream().synchronize()

        try:
            synchronized_call("Packed HCCL buffer validation", validate_buffer)
            _, results = _rpc_with_collective(
                client, context, "receive_packed_weights",
                {"group_id": self._packed_group_id, "metadata": list(metadata),
                 "total_bytes": int(total_bytes), "policy_version": int(policy_version)},
                broadcast, timeout=600,
            )
        except Exception:
            if rank == 0 and packed is not None:
                self._failed_buffers.append(packed)
            raise
        if not isinstance(results, list) or not results:
            raise RuntimeError(f"Packed HCCL returned invalid ACKs: {results}")
        coordinates = set()
        for result in results:
            if (
                not isinstance(result, Mapping)
                or result.get("received") is not True
                or int(result.get("bytes", -1)) != total_bytes
            ):
                raise RuntimeError(f"Packed HCCL returned an invalid ACK: {result}")
            coordinate = (int(result["dp_rank"]), int(result["tp_rank"]))
            if (
                coordinate in coordinates
                or not 0 <= coordinate[0] < self._data_parallel_size
                or not 0 <= coordinate[1] < self._tensor_parallel_size
            ):
                raise RuntimeError(
                    f"Packed HCCL returned an invalid worker coordinate: {result}"
                )
            coordinates.add(coordinate)
        return PackedWeightAck(
            bucket_index,
            total_bytes,
            self._data_parallel_size * self._tensor_parallel_size,
        )

    def close(self) -> None:
        """Release trainer-side communicator references."""
        self._failed_buffers.clear()
        self._groups.clear()
        self._group_ids.clear()
        self._packed_group = None
        self._packed_group_id = None
        self._endpoint = None
