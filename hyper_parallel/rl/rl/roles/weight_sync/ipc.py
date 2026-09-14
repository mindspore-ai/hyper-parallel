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
"""Same-device IPC delivery shared by direct and streaming weight publication."""

import base64
import os
import pickle
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.distributed as dist

from rl.roles.weight_sync.layout import (
    DirectReshardPlan,
    pack_direct_bucket,
)
from rl.roles.weight_sync.packed_weight import PackedWeightAck
from rl.roles.weight_sync.sync import coordinator_call, synchronized_call
from rl.roles.weight_sync.vllm_client import VLLMWeightSyncClientMixin, shared_endpoint


@dataclass(frozen=True)
class PhysicalRolloutWorker:
    """Bind one rollout DP x TP worker to a colocated physical device."""

    dp_rank: int
    tp_rank: int
    physical_device_id: Any


def resolve_physical_worker_topology(
    physical_device_ids: Sequence[Any],
    *,
    data_parallel_size: int,
    tensor_parallel_size: int,
) -> tuple[PhysicalRolloutWorker, ...]:
    """Resolve vLLM's DP-major worker order into explicit IPC ownership."""
    if data_parallel_size <= 0 or tensor_parallel_size <= 0:
        raise ValueError(
            "Rollout data_parallel_size and tensor_parallel_size must be positive"
        )
    expected = data_parallel_size * tensor_parallel_size
    if len(physical_device_ids) != expected:
        raise ValueError(
            "Rollout physical devices must match DP x TP: "
            f"expected={expected}, got={len(physical_device_ids)}"
        )
    if len(set(physical_device_ids)) != expected:
        raise ValueError("Rollout physical device identities must be unique")
    return tuple(
        PhysicalRolloutWorker(
            dp_rank=index // tensor_parallel_size,
            tp_rank=index % tensor_parallel_size,
            physical_device_id=device_id,
        )
        for index, device_id in enumerate(physical_device_ids)
    )


@dataclass(frozen=True)
class IPCContext:
    """The shared endpoint and each Trainer's same-device rollout ownership."""
    endpoint: str
    physical_device_id: Any
    local_worker: PhysicalRolloutWorker
    workers: tuple[PhysicalRolloutWorker, ...]


def tensor_ipc_rebuild_args(tensor: Any) -> tuple[Any, ...]:
    """Export Torch storage without copying the packed weight buffer."""
    # The vLLM IPC wire protocol specifically uses Torch storage reconstruction.
    from torch.multiprocessing.reductions import reduce_tensor  # pylint: disable=C0415,forbidden-backend-import

    _, rebuild_args = reduce_tensor(tensor)
    return rebuild_args


class IPCWeightTransport:
    """Own IPC handles, receiver acknowledgements, and unsafe producer buffers."""
    name = "ipc"

    def __init__(self, *, data_parallel_size: int = 1, tensor_parallel_size: int = 1) -> None:
        """Defer physical-device discovery until rollout weights are awake."""
        if data_parallel_size <= 0 or tensor_parallel_size <= 0:
            raise ValueError("IPC DP and TP sizes must be positive")
        self._data_parallel_size = int(data_parallel_size)
        self._tensor_parallel_size = int(tensor_parallel_size)
        self._context: Optional[IPCContext] = None
        self._failed_buffers: list[Any] = []

    @property
    def failed_buffer_count(self) -> int:
        """Return the number of producers retained after an uncertain receive."""
        return len(self._failed_buffers)

    @staticmethod
    def _device_order(physical_device_ids: list[Any]) -> tuple[Any, ...]:
        """Order worker UUIDs by the launcher's physical visible-device contract."""
        visible = tuple(value.strip() for value in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
                        if value.strip())
        by_id = {str(value).rsplit("-", maxsplit=1)[-1]: value for value in physical_device_ids}
        if not visible or set(visible) != set(by_id) or len(by_id) != len(physical_device_ids):
            raise RuntimeError(f"IPC devices differ from ASCEND_RT_VISIBLE_DEVICES: {visible}, {physical_device_ids}")
        return tuple(by_id[value] for value in visible)

    def prepare(self, client: VLLMWeightSyncClientMixin, destination_tp_size: int) -> IPCContext:
        """Resolve a single endpoint and explicit same-device ownership once."""
        if destination_tp_size != self._tensor_parallel_size:
            raise ValueError("IPC destination size differs from rollout TP")
        endpoint = shared_endpoint(client)
        if self._context is not None:
            if endpoint != self._context.endpoint:
                raise RuntimeError("IPC rollout endpoint changed after transport initialization")
            return self._context
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (  # pylint: disable=C0415
            npu_generate_uuid,
        )

        device_id = npu_generate_uuid()
        physical_ids = [None] * dist.get_world_size()
        dist.all_gather_object(physical_ids, device_id)
        workers = resolve_physical_worker_topology(
            self._device_order(physical_ids), data_parallel_size=self._data_parallel_size,
            tensor_parallel_size=self._tensor_parallel_size,
        )
        local_worker = next(worker for worker in workers if worker.physical_device_id == device_id)
        self._context = IPCContext(endpoint, device_id, local_worker, workers)
        return self._context

    def _validate_results(
        self, results: Any, context: IPCContext, target_tp_rank: int, copied_bytes: int,
    ) -> None:
        """Check every returned DP replica without assuming vLLM returns all replicas."""
        if not isinstance(results, list) or not results:
            raise RuntimeError(f"IPC returned invalid receive acknowledgements: {results}")
        expected = {(worker.dp_rank, worker.tp_rank): worker.physical_device_id for worker in context.workers}
        received = set()
        for result in results:
            if not isinstance(result, Mapping) or result.get("received") is not True:
                raise RuntimeError(f"IPC returned invalid receive acknowledgement: {result}")
            coordinate = (int(result["dp_rank"]), int(result["tp_rank"]))
            expected_bytes = (
                copied_bytes
                if target_tp_rank < 0 or coordinate[1] == target_tp_rank
                else 0
            )
            if (coordinate in received or coordinate not in expected
                    or result.get("physical_device_id") != expected[coordinate]
                    or int(result.get("bytes", -1)) != expected_bytes):
                raise RuntimeError(f"IPC acknowledgement differs from the target bucket: {result}")
            received.add(coordinate)
        for dp_rank in {dp_rank for dp_rank, _ in received}:
            if {tp_rank for dp, tp_rank in received if dp == dp_rank} != set(range(self._tensor_parallel_size)):
                raise RuntimeError(f"IPC returned an incomplete TP replica: {results}")

    @staticmethod
    def _collect_handles(
        context: IPCContext, packed: Any, expected_devices: set[Any],
    ) -> dict[Any, Any]:
        """Synchronize export errors before gathering the intended device handles."""
        def export() -> dict[Any, Any]:
            """Only devices participating in this delivery export their buffer."""
            if context.physical_device_id not in expected_devices:
                return {}
            return {context.physical_device_id: tensor_ipc_rebuild_args(packed)}

        local_handles = synchronized_call("IPC handle export", export)
        handles_by_rank = [None] * dist.get_world_size()
        dist.all_gather_object(handles_by_rank, local_handles)
        handles = {}
        for rank_handles in handles_by_rank:
            if not isinstance(rank_handles, Mapping) or handles.keys() & rank_handles.keys():
                raise RuntimeError(f"IPC received duplicate or invalid producer handles: {handles_by_rank}")
            handles.update(rank_handles)
        if handles.keys() != expected_devices:
            raise RuntimeError(
                f"IPC handles do not cover the target devices: actual={set(handles)}, expected={expected_devices}"
            )
        return handles

    def _receive(
        self, client: VLLMWeightSyncClientMixin, context: IPCContext, method: str,
        payload: Mapping[str, Any], policy_version: int, target_tp_rank: int, copied_bytes: int,
    ) -> None:
        """Serialize once on the coordinator and validate receipt before release."""
        def receive() -> None:
            """Send one IPC payload through the shared endpoint."""
            results = client.collective_rpc(
                method,
                {"payload_pickled": base64.b64encode(pickle.dumps(payload)).decode("ascii"),
                 "policy_version": int(policy_version)},
                context.endpoint,
            )
            self._validate_results(results, context, target_tp_rank, copied_bytes)

        coordinator_call("IPC bucket receive", receive)
        synchronized_call(
            "IPC acknowledged producer synchronization", torch.get_device_module().current_stream().synchronize,
        )

    def send_bucket(
        self, client: VLLMWeightSyncClientMixin, context: IPCContext,
        target_tp_rank: int, bucket_index: int, metadata: Mapping[str, Any], packed: Any, policy_version: int,
    ) -> None:
        """Export one buffer per target NPU and retain it until receipt is known."""
        expected_devices = {worker.physical_device_id for worker in context.workers
                            if worker.tp_rank == target_tp_rank}
        try:
            handles = self._collect_handles(context, packed, expected_devices)
            payload = {
                "buckets_by_target": {target_tp_rank: [{
                    "target_rank": target_tp_rank, "bucket_index": bucket_index,
                    "metadata": dict(metadata), "ipc_handles": handles,
                }]},
                "tensor_parallel_size": self._tensor_parallel_size,
                "worker_topology": [vars(worker) for worker in context.workers],
            }
            self._receive(
                client, context, "receive_ipc_direct_reshard", payload, policy_version,
                target_tp_rank, sum(int(entry["num_bytes"]) for entry in metadata["entries"]),
            )
        except Exception:
            if context.physical_device_id in expected_devices:
                self._failed_buffers.append(packed)
            raise

    def prepare_packed(self, client: VLLMWeightSyncClientMixin) -> IPCContext:
        """Resolve every same-device worker that will load complete parameters."""
        return self.prepare(client, self._tensor_parallel_size)

    def send_packed_bucket(
        self,
        client: VLLMWeightSyncClientMixin,
        context: IPCContext,
        bucket_index: int,
        metadata: list[Mapping[str, Any]],
        total_bytes: int,
        packed: Optional[Any],
        policy_version: int,
    ) -> PackedWeightAck:
        """Replicate one producer buffer, then export one handle per NPU."""

        def allocate() -> Any:
            """Validate the producer or allocate a local receive buffer before broadcast."""
            device_handle = torch.get_device_module()
            device_type = (torch.accelerator.current_accelerator() or torch.device("cpu")).type
            device = torch.device("cpu") if device_type == "cpu" else torch.device(
                device_type, device_handle.current_device()
            )
            if dist.get_rank() == 0:
                if packed is None or int(packed.numel()) != total_bytes:
                    raise ValueError("Packed IPC producer buffer differs from its metadata")
                # FSDP CPU offload can leave full-gather output on CPU. Every
                # participant must enter the same device/backend broadcast.
                return packed.to(device)
            return torch.empty(
                total_bytes, dtype=torch.uint8,
                device=device,
            )

        local_buffer = synchronized_call("IPC packed buffer allocation", allocate)
        try:
            dist.broadcast(local_buffer, src=0)
            synchronized_call("IPC packed broadcast completion", torch.get_device_module().current_stream().synchronize)
            expected_devices = {worker.physical_device_id for worker in context.workers}
            handles = self._collect_handles(context, local_buffer, expected_devices)
            payload = {
                "ipc_handles": handles,
                "metadata": list(metadata),
                "total_bytes": int(total_bytes),
                "worker_topology": [vars(worker) for worker in context.workers],
            }
            self._receive(
                client, context, "receive_ipc_packed_weights", payload, policy_version, -1, total_bytes,
            )
        except Exception:
            self._failed_buffers.append(local_buffer)
            raise
        return PackedWeightAck(
            bucket_index, total_bytes, self._data_parallel_size * self._tensor_parallel_size,
        )

    def transfer_direct(
        self, client: VLLMWeightSyncClientMixin, state_dict: Mapping[str, Any], plan: DirectReshardPlan,
        policy_version: int,
    ) -> None:
        """Redistribute one source bucket at a time before same-device IPC delivery."""

        context = self.prepare(client, plan.destination_tp_size)
        device_handle = torch.get_device_module()
        device_type = (torch.accelerator.current_accelerator() or torch.device("cpu")).type
        device = torch.device(device_type, device_handle.current_device())
        rank = dist.get_rank()
        for source_rank in range(plan.source_world_size):
            for target_rank in range(plan.destination_tp_size):
                for bucket_index, bucket in enumerate(plan.for_route(source_rank, target_rank)):
                    packed = synchronized_call(
                        "direct IPC bucket packing",
                        lambda current_bucket=bucket, current_source=source_rank: (
                            pack_direct_bucket(state_dict, current_bucket, device) if rank == current_source
                            else torch.empty(current_bucket.total_bytes, dtype=torch.uint8, device=device)
                        ),
                    )
                    torch.get_device_module().current_stream().synchronize()
                    dist.broadcast(packed, src=source_rank)
                    torch.get_device_module().current_stream().synchronize()
                    try:
                        self.send_bucket(client, context, target_rank, bucket_index, bucket.worker_metadata(),
                                         packed, policy_version)
                    finally:
                        # The transport owns any exported buffer whose receive was uncertain.
                        del packed

    def close(self) -> None:
        """Release resources after the rollout process has shut down."""
        self._failed_buffers.clear()
        self._context = None
