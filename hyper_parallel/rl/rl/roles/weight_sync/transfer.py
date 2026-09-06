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
"""Actor-to-rollout weight preparation, transport, and verification."""
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import logging
import os
from pathlib import Path
import pickle
import socket
from typing import Any, Mapping, Optional, Protocol

from torch.distributed.checkpoint.state_dict import StateDictOptions  # pylint: disable=forbidden-backend-import
from torch.multiprocessing.reductions import reduce_tensor  # pylint: disable=forbidden-backend-import

from rl.roles.model import VLLMModelRegistration
from rl.roles.weight_sync.hccl import BroadcastDirectReshardHCCLTransport
from rl.roles.weight_sync.layout import (
    DirectReshardPlan,
    TransferBucket,
    build_direct_reshard_plan,
    describe_source_tensor,
    resolve_destination_layouts,
    resolve_physical_worker_topology,
    resolve_source_layouts,
)
from rl.roles.weight_sync.sync import (
    POLICY_FINGERPRINT_ALGORITHM,
    PolicySnapshot,
    VLLMWeightSyncClientMixin,
    aggregate_policy_fingerprint,
    canonical_policy_weight_name,
    coordinator_call,
    is_policy_fingerprint_weight,
    policy_fingerprint_header,
    policy_tensor_fingerprint,
    policy_weight_fingerprint,
    synchronized_call,
    synchronize_error,
    verify_policy_fingerprints,
)

from hyper_parallel import get_platform
platform = get_platform()
logger = logging.getLogger(__name__)


def _tensor_ipc_rebuild_args(tensor: Any) -> tuple[Any, ...]:
    """Share tensor storage and return Torch multiprocessing rebuild arguments."""
    _, rebuild_args = reduce_tensor(tensor)
    return rebuild_args


def _write_rollout_parameter_manifest(
    client: VLLMWeightSyncClientMixin,
    *,
    strategy: str,
    policy_version: int,
    data_parallel_size: int = 1,
) -> None:
    """Persist exact rank-local rollout hashes for an explicit verification run."""
    output_dir = os.environ.get("HYPER_RL_WEIGHT_MANIFEST_DIR")
    if not output_dir:
        return
    if not bool(getattr(client, "is_server_owner", True)):
        return
    oracle_run_id = os.environ.get("HYPER_RL_WEIGHT_ORACLE_RUN_ID")
    if not oracle_run_id:
        raise RuntimeError("HYPER_RL_WEIGHT_ORACLE_RUN_ID must identify the verification run")
    oracle_dir = (
        os.environ.get("HYPER_RL_WEIGHT_MANIFEST_ORACLE_DIR")
        if strategy != "full_gather"
        else None
    )
    expected_dir = os.environ.get("HYPER_RL_WEIGHT_EXPECTED_MANIFEST_DIR")
    results = client.collective_rpc(
        "write_parameter_manifest",
        {
            "output_dir": output_dir,
            "strategy": strategy,
            "policy_version": int(policy_version),
            "rollout_replica_rank": int(platform.get_rank()),
            "expected_data_parallel_size": int(data_parallel_size),
            "oracle_run_id": oracle_run_id,
            "oracle_dir": oracle_dir,
            "oracle_strategy": "full_gather" if oracle_dir else None,
            "expected_dir": expected_dir,
        },
    )
    if not results or not all(
        isinstance(result, Mapping) and bool(result.get("written"))
        for result in results
    ):
        raise RuntimeError(f"Invalid rollout parameter manifest results: {results}")


def _local_state_dict(payload: Any, *, operation: str) -> dict[str, Any]:
    """Extract FSDP-local model shards and validate their tensor-only contract."""
    state_dict = (
        dict(payload)
        if isinstance(payload, Mapping)
        else platform.get_model_state_dict(
            payload,
            options=StateDictOptions(
                full_state_dict=False,
                cpu_offload=False,
            ),
        )
    )
    invalid = next(
        ((name, value) for name, value in state_dict.items() if not platform.is_tensor(value)),
        None,
    )
    if invalid is not None:
        name, value = invalid
        state_dict.clear()
        raise ValueError(
            f"{operation} state entry {name!r} must be a tensor, got {type(value)!r}"
        )
    return state_dict


def _full_state_dict(payload: Any, *, operation: str) -> dict[str, Any]:
    """All-gather complete Actor tensors onto every FSDP rank."""
    state_dict = (
        dict(payload)
        if isinstance(payload, Mapping)
        else platform.get_model_state_dict(
            payload,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=False,
            ),
        )
    )
    invalid = next(
        ((name, value) for name, value in state_dict.items() if not platform.is_tensor(value)),
        None,
    )
    if invalid is not None:
        name, value = invalid
        state_dict.clear()
        raise ValueError(
            f"{operation} state entry {name!r} must be a tensor, got {type(value)!r}"
        )
    return state_dict


def _gather_selected_state_dict(
    state_dict: Mapping[str, Any],
    *,
    cpu_offload: bool,
) -> dict[str, Any]:
    """Materialize only the selected DTensor values using master semantics."""
    is_rank_zero = platform.get_rank() == 0
    gathered = {}
    for name, value in state_dict.items():
        full_tensor = getattr(value, "full_tensor", None)
        if callable(full_tensor):
            value = full_tensor()
        if cpu_offload:
            if not is_rank_zero:
                continue
            value = value.to("cpu")
        gathered[name] = value
    return gathered


def map_actor_state_dict(
    state_dict: Mapping[str, Any],
    model: VLLMModelRegistration,
) -> dict[str, Any]:
    """Map policy Actor names to the selected rollout model namespace."""
    mapped = {}
    for name, tensor in state_dict.items():
        mapped_name = model.actor_weight_name(name)
        if mapped_name is None:
            continue
        if mapped_name in mapped:
            raise ValueError(
                f"vLLM policy-name mapping collision: {name!r} maps to {mapped_name!r}"
            )
        mapped[mapped_name] = tensor
    return mapped


def _alias_tied_embeddings(
    state_dict: dict[str, Any],
    model: VLLMModelRegistration,
) -> dict[str, Any]:
    """Expose both tied checkpoint names without allocating another tensor."""
    if not model.model.tie_word_embeddings:
        return state_dict
    embedding_name = "model.embed_tokens.weight"
    lm_head_name = "lm_head.weight"
    if embedding_name in state_dict and lm_head_name not in state_dict:
        state_dict[lm_head_name] = state_dict[embedding_name]
    elif lm_head_name in state_dict and embedding_name not in state_dict:
        state_dict[embedding_name] = state_dict[lm_head_name]
    return state_dict


class WeightTransfer(Protocol):
    """Publish one policy Actor snapshot into an existing rollout model."""

    def publish(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Atomically publish one policy snapshot to rollout."""


class _RefitCompatibleTransfer:
    """Expose historical verbs as adapters to the canonical publication operation."""

    def publish(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Atomically publish one policy snapshot to rollout."""
        raise NotImplementedError

    def transfer(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Route the historical transfer verb to publication."""
        self.publish(client, snapshot)

    def refit(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Route the historical refit verb to publication."""
        self.publish(client, snapshot)


class DirectReshardHCCLWeightTransfer(_RefitCompatibleTransfer):
    """Broadcast FSDP-local fragments directly into Qwen3 rollout TP shards."""

    def __init__(
        self,
        model: VLLMModelRegistration,
        *,
        bucket_size_bytes: int = 128 * 2**20,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
    ) -> None:
        """Store the fixed model contract and defer layout-plan construction."""
        if model.family != "qwen3":
            raise ValueError(
                "Direct reshard currently supports Qwen3 rollout models"
            )
        if bucket_size_bytes <= 0:
            raise ValueError("Direct reshard bucket_size_bytes must be positive")
        if data_parallel_size <= 0:
            raise ValueError("Direct reshard data_parallel_size must be positive")
        if tensor_parallel_size <= 0:
            raise ValueError("Direct reshard tensor_parallel_size must be positive")
        self._model = model
        self._bucket_size_bytes = int(bucket_size_bytes)
        self._data_parallel_size = int(data_parallel_size)
        self._tensor_parallel_size = int(tensor_parallel_size)
        self._transport = BroadcastDirectReshardHCCLTransport(
            data_parallel_size=self._data_parallel_size,
            tensor_parallel_size=self._tensor_parallel_size,
        )
        self._plan: Optional[DirectReshardPlan] = None
        self._parameter_names: Optional[frozenset[str]] = None
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None
        self.configured_strategy = "direct_reshard"
        self.last_strategy: Optional[str] = None
        self.fallback_count = 0
        self.direct_success_count = 0

    @staticmethod
    def _local_state_dict(payload: Any) -> dict[str, Any]:
        """Return DTensor local shards without any FSDP all-gather."""
        return _local_state_dict(
            payload,
            operation="vLLM direct reshard",
        )

    def _mapped_local_state_dict(self, payload: Any) -> dict[str, Any]:
        """Map Actor names and alias tied embeddings without copying their shard."""
        return _alias_tied_embeddings(
            map_actor_state_dict(
                self._local_state_dict(payload),
                self._model,
            ),
            self._model,
        )

    def _query_destination_workers(
        self,
        client: VLLMWeightSyncClientMixin,
    ) -> list[Mapping[str, Any]]:
        """Query and collapse the DP-engine layouts returned by vLLM."""
        expected_world_size = self._data_parallel_size * self._tensor_parallel_size

        def query_workers() -> list[Any]:
            """Query every physical rollout worker after validating world size."""
            actual_world_size = client.get_world_size()
            if actual_world_size != expected_world_size:
                raise RuntimeError(
                    "Direct reshard rollout world size differs from configured DP x TP: "
                    f"expected={expected_world_size}, actual={actual_world_size}"
                )
            return client.collective_rpc("get_direct_reshard_layout")

        workers = coordinator_call("direct reshard rollout layout query", query_workers)
        if not isinstance(workers, list) or not workers or not all(
            isinstance(worker, Mapping) for worker in workers
        ):
            raise RuntimeError(
                f"Direct reshard rollout returned invalid layouts: {workers}"
            )
        by_identity: dict[tuple[int, int], Mapping[str, Any]] = {}
        for worker in workers:
            identity = (int(worker["dp_rank"]), int(worker["tp_rank"]))
            worker_dp_size = int(worker["dp_size"])
            # Non-MoE vLLM engines may expose engine-local DP size 1 while
            # retaining the deployment-global data_parallel_index.
            if worker_dp_size not in (1, self._data_parallel_size):
                raise RuntimeError(
                    "Direct reshard worker DP size differs from configured topology: "
                    f"worker={identity}, expected={self._data_parallel_size}, "
                    f"actual={worker_dp_size}"
                )
            if int(worker["tp_size"]) != self._tensor_parallel_size:
                raise RuntimeError(
                    "Direct reshard worker TP size differs from configured topology: "
                    f"worker={identity}, expected={self._tensor_parallel_size}, "
                    f"actual={worker['tp_size']}"
                )
            if (
                not 0 <= identity[0] < self._data_parallel_size
                or not 0 <= identity[1] < self._tensor_parallel_size
            ):
                raise RuntimeError(f"Direct reshard worker identity is out of range: {identity}")
            if identity in by_identity:
                raise RuntimeError(f"Direct reshard returned duplicate worker identity {identity}")
            by_identity[identity] = worker
        returned_dp_ranks = sorted({dp_rank for dp_rank, _ in by_identity})
        expected_tp_ranks = set(range(self._tensor_parallel_size))
        for dp_rank in returned_dp_ranks:
            actual_tp_ranks = {
                tp_rank
                for worker_dp_rank, tp_rank in by_identity
                if worker_dp_rank == dp_rank
            }
            if actual_tp_ranks != expected_tp_ranks:
                raise RuntimeError(
                    "Direct reshard layout query returned an incomplete TP engine: "
                    f"dp_rank={dp_rank}, expected={sorted(expected_tp_ranks)}, "
                    f"actual={sorted(actual_tp_ranks)}"
                )
        # vLLM internal-DP utilities fan out to every engine but may expose only
        # one engine's return value. Destination layouts are identical across DP,
        # while each worker validates its own global identity before receiving.
        representative_dp_rank = returned_dp_ranks[0]
        representatives = []
        for tp_rank in range(self._tensor_parallel_size):
            representative = by_identity[(representative_dp_rank, tp_rank)]
            expected_tensors = representative.get("tensors")
            for dp_rank in returned_dp_ranks[1:]:
                replica_tensors = by_identity[(dp_rank, tp_rank)].get("tensors")
                if replica_tensors != expected_tensors:
                    raise RuntimeError(
                        "Direct reshard layouts differ across same-TP DP replicas: "
                        f"tp_rank={tp_rank}, dp_rank={dp_rank}"
                    )
            representatives.append(representative)
        return representatives

    def _build_plan(
        self,
        client: VLLMWeightSyncClientMixin,
        state_dict: Mapping[str, Any],
    ) -> DirectReshardPlan:
        """Compile metadata-only source and destination layouts once."""
        destination_workers = self._query_destination_workers(client)
        parameter_names = frozenset(
            str(tensor["name"])
            for worker in destination_workers
            for tensor in worker["tensors"]
        )
        missing = sorted(parameter_names - set(state_dict))
        if missing:
            raise ValueError(
                "Direct reshard Actor state is missing rollout parameters: "
                + ", ".join(missing)
            )
        source_rank = platform.get_rank()
        local_descriptions = [
            describe_source_tensor(name, state_dict[name], source_rank)
            for name in sorted(parameter_names)
        ]
        source_world_size = platform.get_world_size()
        rank_descriptions: list[Any] = [None] * source_world_size
        platform.all_gather_object(rank_descriptions, local_descriptions)
        source_layouts = resolve_source_layouts(rank_descriptions)
        global_shapes = {
            source.name: source.global_shape for source in source_layouts
        }
        destination_layouts = resolve_destination_layouts(
            destination_workers,
            global_shapes,
        )
        trace_dir = os.environ.get("HYPER_RL_DIRECT_PLAN_TRACE_DIR")
        if trace_dir and source_rank == 0:
            trace = {
                "rank_descriptions": rank_descriptions,
                "source_layouts": [
                    {
                        "name": source.name,
                        "source_rank": source.source_rank,
                        "global_shape": list(source.global_shape),
                        "starts": list(source.region.starts),
                        "lengths": list(source.region.lengths),
                    }
                    for source in source_layouts
                ],
                "destination_layouts": [
                    {
                        "name": destination.name,
                        "tp_rank": destination.tp_rank,
                        "placement": destination.placement,
                        "shard_dim": destination.shard_dim,
                        "global_shape": list(destination.global_shape),
                        "starts": list(destination.region.starts),
                        "lengths": list(destination.region.lengths),
                    }
                    for destination in destination_layouts
                ],
            }
            directory = Path(trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "layout.json"
            temporary = directory / f".{target.name}.{os.getpid()}.tmp"
            temporary.write_text(
                json.dumps(trace, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        self._parameter_names = parameter_names
        return build_direct_reshard_plan(
            source_layouts,
            destination_layouts,
            source_world_size=source_world_size,
            bucket_size_bytes=self._bucket_size_bytes,
        )

    def _ensure_plan(
        self,
        client: VLLMWeightSyncClientMixin,
        state_dict: Mapping[str, Any],
    ) -> DirectReshardPlan:
        if self._plan is None:
            self._plan = self._build_plan(client, state_dict)
        if self._parameter_names is None:
            raise RuntimeError("Direct reshard plan has no parameter contract")
        missing = sorted(self._parameter_names - set(state_dict))
        if missing:
            raise ValueError(
                "Direct reshard Actor state changed after planning; missing="
                + ", ".join(missing)
            )
        return self._plan

    @staticmethod
    def _distributed_policy_fingerprint(
        state_dict: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Gather only small norm tensors needed by publication verification."""
        fingerprint_shards = {
            name: tensor
            for name, tensor in state_dict.items()
            if is_policy_fingerprint_weight(name)
        }
        if not fingerprint_shards:
            raise RuntimeError("Direct reshard found no Actor norm tensors to verify")
        full_norms = _gather_selected_state_dict(
            fingerprint_shards,
            cpu_offload=True,
        )
        local_fingerprint = (
            policy_weight_fingerprint(full_norms)
            if platform.get_rank() == 0
            else None
        )
        fingerprints: list[Any] = [None] * platform.get_world_size()
        platform.all_gather_object(fingerprints, local_fingerprint)
        expected = fingerprints[0]
        if not isinstance(expected, Mapping) or any(
            fingerprint is not None for fingerprint in fingerprints[1:]
        ):
            raise RuntimeError(
                f"Direct reshard Actor fingerprint publication is invalid: {fingerprints}"
            )
        return dict(expected)

    def publish(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Plan, broadcast local shards, commit, and verify one policy version."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError(
                "DirectReshardHCCLWeightTransfer requires an external vLLM HTTP client"
            )
        local_state_dict = synchronized_call(
            "direct reshard local-state extraction",
            lambda: self._mapped_local_state_dict(snapshot.payload),
        )
        try:
            plan = synchronized_call(
                "direct reshard layout planning",
                lambda: self._ensure_plan(client, local_state_dict),
            )
            coordinator_call("direct reshard pause", client.pause)
            coordinator_call("direct reshard start", client.start_weight_update)
            synchronized_call(
                "direct reshard producer synchronization",
                platform.get_current_stream().synchronize,
            )
            self._transport.transfer(
                client,
                local_state_dict,
                plan,
                snapshot.version,
            )
            coordinator_call("direct reshard finish", client.finish_weight_update)
            coordinator_call(
                "direct reshard rollout parameter manifest",
                lambda: _write_rollout_parameter_manifest(
                    client,
                    strategy="direct_reshard",
                    policy_version=snapshot.version,
                    data_parallel_size=self._data_parallel_size,
                ),
            )
            expected_fingerprint = synchronized_call(
                "direct reshard Actor policy fingerprint",
                lambda: self._distributed_policy_fingerprint(local_state_dict),
            )

            def verify_worker_fingerprints() -> None:
                """Verify every worker loaded the published policy version."""
                client.verify_policy_weight_identity(
                    snapshot.version,
                    expected_fingerprint,
                )

            coordinator_call("direct reshard policy fingerprint", verify_worker_fingerprints)
            self.last_policy_fingerprint = expected_fingerprint
            self.last_strategy = "direct_reshard"
            self.direct_success_count += 1
        finally:
            local_state_dict.clear()

    def close(self) -> None:
        """Release cached stateless HCCL route references."""
        self._transport.close()


class ColocatedDirectReshardWeightTransfer(DirectReshardHCCLWeightTransfer):
    """Redistribute FSDP fragments among trainers, then use same-NPU IPC."""

    def __init__(
        self,
        model: VLLMModelRegistration,
        *,
        bucket_size_bytes: int = 128 * 2**20,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
    ) -> None:
        """Initialize direct resharding and retain failed IPC producers safely."""
        super().__init__(
            model,
            bucket_size_bytes=bucket_size_bytes,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        if data_parallel_size <= 0:
            raise ValueError("Direct reshard data_parallel_size must be positive")
        self._data_parallel_size = int(data_parallel_size)
        self._failed_buffers: list[Any] = []

    @property
    def weights_awake(self) -> bool:
        """Return whether this transfer successfully restored vLLM weights."""
        return bool(getattr(self, "_weights_awake", False))

    @staticmethod
    def _run_control(
        operation: str,
        client: VLLMWeightSyncClientMixin,
        callback: Any,
    ) -> None:
        """Run a mutating endpoint call once per colocated rollout replica."""
        synchronized_call(
            operation,
            lambda: callback()
            if bool(getattr(client, "is_server_owner", True))
            else None,
        )

    @staticmethod
    def _gather_endpoints(client: VLLMWeightSyncClientMixin) -> tuple[str, ...]:
        """Return every unique colocated rollout endpoint."""
        endpoints = [""] * platform.get_world_size()
        platform.all_gather_object(endpoints, client.base_url)
        unique = tuple(sorted(set(endpoints)))
        if not unique or any(not endpoint for endpoint in unique):
            raise RuntimeError(f"Colocated direct endpoints are invalid: {endpoints}")
        return unique

    @staticmethod
    def _local_tensor(value: Any) -> Any:
        """Return a DTensor's local shard or its original plain tensor."""
        to_local = getattr(value, "to_local", None)
        return to_local() if callable(to_local) else value

    @staticmethod
    def _rollout_device_order(physical_device_ids: list[Any]) -> tuple[Any, ...]:
        """Order gathered UUIDs by the rollout's explicit visible-device contract."""
        visible_devices = tuple(
            device.strip()
            for device in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
            if device.strip()
        )
        by_physical_id = {
            str(device_id).rsplit("-", maxsplit=1)[-1]: device_id
            for device_id in physical_device_ids
        }
        if not visible_devices or set(visible_devices) != set(by_physical_id):
            raise RuntimeError(
                "Colocated direct physical devices do not match ASCEND_RT_VISIBLE_DEVICES: "
                f"visible={visible_devices}, workers={sorted(by_physical_id)}"
            )
        return tuple(by_physical_id[device] for device in visible_devices)

    @classmethod
    def _pack_bucket(
        cls,
        state_dict: Mapping[str, Any],
        bucket: TransferBucket,
        device: Any,
    ) -> Any:
        """Pack one planned source route into a bounded NPU byte buffer."""
        # Torch is optional outside the Torch-NPU RL runtime.
        import torch  # pylint: disable=C0415,forbidden-backend-import

        packed = torch.empty(bucket.total_bytes, dtype=torch.uint8, device=device)
        for entry in bucket.entries:
            value = state_dict.get(entry.name)
            if value is None:
                raise ValueError(
                    f"Colocated direct source parameter {entry.name!r} is missing"
                )
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
                    f"Colocated direct source fragment {entry.name!r} has "
                    f"{raw.numel()} bytes, expected {entry.num_bytes}"
                )
            packed.narrow(0, entry.buffer_offset, entry.num_bytes).copy_(raw)
        return packed

    @classmethod
    def _redistribute_and_export(
        cls,
        state_dict: Mapping[str, Any],
        plan: DirectReshardPlan,
        data_parallel_size: int = 1,
    ) -> tuple[dict[str, Any], list[Any]]:
        """Broadcast planned fragments and export local destination buffers."""
        # Torch and vLLM-Ascend are optional outside the Torch-NPU RL runtime.
        import torch  # pylint: disable=C0415,forbidden-backend-import
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (  # pylint: disable=C0415
            npu_generate_uuid,
        )

        device_handle = platform.get_device_handle(platform.device_type())
        device = torch.device(
            platform.device_type(),
            device_handle.current_device(),
        )
        rank = platform.get_rank()
        local_records = []
        retained_buffers = []
        npu_uuid = npu_generate_uuid()
        physical_device_ids: list[Any] = [None] * platform.get_world_size()
        platform.all_gather_object(physical_device_ids, npu_uuid)
        ordered_device_ids = cls._rollout_device_order(physical_device_ids)
        workers = resolve_physical_worker_topology(
            ordered_device_ids,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=plan.destination_tp_size,
        )
        local_worker = next(
            worker for worker in workers if worker.physical_device_id == npu_uuid
        )

        for source_rank in range(plan.source_world_size):
            for target_tp_rank in range(plan.destination_tp_size):
                for bucket_index, bucket in enumerate(
                    plan.for_route(source_rank, target_tp_rank)
                ):
                    if rank == source_rank:
                        packed = cls._pack_bucket(state_dict, bucket, device)
                    else:
                        packed = torch.empty(
                            bucket.total_bytes,
                            dtype=torch.uint8,
                            device=device,
                        )
                    platform.get_current_stream().synchronize()
                    platform.broadcast(packed, src=source_rank)
                    platform.get_current_stream().synchronize()
                    if local_worker.tp_rank == target_tp_rank:
                        retained_buffers.append(packed)
                        local_records.append(
                            {
                                "key": (source_rank, target_tp_rank, bucket_index),
                                "tp_rank": target_tp_rank,
                                "metadata": bucket.worker_metadata(),
                                "ipc_handles": {
                                    npu_uuid: _tensor_ipc_rebuild_args(packed)
                                },
                            }
                        )
                    else:
                        del packed

        gathered_records: list[Any] = [None] * platform.get_world_size()
        platform.all_gather_object(gathered_records, local_records)
        merged: dict[tuple[int, int, int], dict[str, Any]] = {}
        for rank_records in gathered_records:
            for record in rank_records:
                key = tuple(record["key"])
                existing = merged.get(key)
                if existing is None:
                    merged[key] = {
                        "tp_rank": int(record["tp_rank"]),
                        "metadata": record["metadata"],
                        "ipc_handles": dict(record["ipc_handles"]),
                    }
                else:
                    if existing["metadata"] != record["metadata"]:
                        raise RuntimeError(
                            f"Colocated direct bucket metadata differs for route {key}"
                        )
                    existing["ipc_handles"].update(record["ipc_handles"])

        buckets_by_tp = {
            tp_rank: [
                merged[key]
                for key in sorted(merged)
                if int(merged[key]["tp_rank"]) == tp_rank
            ]
            for tp_rank in range(plan.destination_tp_size)
        }
        for tp_rank, buckets in buckets_by_tp.items():
            expected_devices = {
                worker.physical_device_id
                for worker in workers
                if worker.tp_rank == tp_rank
            }
            for bucket in buckets:
                actual_devices = set(bucket["ipc_handles"])
                if actual_devices != expected_devices:
                    raise RuntimeError(
                        "Colocated direct IPC handles do not cover the physical workers: "
                        f"tp_rank={tp_rank}, expected={expected_devices}, actual={actual_devices}"
                    )
        worker_topology = [
            {
                "dp_rank": worker.dp_rank,
                "tp_rank": worker.tp_rank,
                "physical_device_id": worker.physical_device_id,
            }
            for worker in workers
        ]
        return {
            "buckets_by_tp": buckets_by_tp,
            "worker_topology": worker_topology,
        }, retained_buffers

    @staticmethod
    def _send_payload(
        client: VLLMWeightSyncClientMixin,
        endpoints: tuple[str, ...],
        payload: Mapping[str, Any],
        policy_version: int,
        destination_tp_size: int,
    ) -> None:
        """Send IPC handles once and validate every colocated TP worker."""
        payload_pickled = base64.b64encode(pickle.dumps(payload)).decode("ascii")
        send_error = None
        if platform.get_rank() == 0:
            try:
                with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
                    requests = [
                        executor.submit(
                            client.collective_rpc,
                            "receive_ipc_direct_reshard",
                            {
                                "payload_pickled": payload_pickled,
                                "policy_version": policy_version,
                            },
                            endpoint,
                        )
                        for endpoint in endpoints
                    ]
                    for endpoint, request in zip(endpoints, requests):
                        results = request.result(timeout=600)
                        expected_workers = sorted(
                            (
                                int(worker["dp_rank"]),
                                int(worker["tp_rank"]),
                                str(worker["physical_device_id"]),
                            )
                            for worker in payload["worker_topology"]
                        )
                        expected_by_identity = {
                            (dp_rank, tp_rank): physical_device_id
                            for dp_rank, tp_rank, physical_device_id in expected_workers
                        }
                        expected_dp_ranks = sorted({worker[0] for worker in expected_workers})
                        expected_tp_ranks = set(range(destination_tp_size))
                        if not expected_dp_ranks or any(
                            {
                                tp_rank
                                for worker_dp_rank, tp_rank in expected_by_identity
                                if worker_dp_rank == dp_rank
                            }
                            != expected_tp_ranks
                            for dp_rank in expected_dp_ranks
                        ):
                            raise RuntimeError(
                                f"Colocated direct payload has invalid TP workers: {expected_workers}"
                            )
                        if not isinstance(results, list) or not all(
                            isinstance(result, Mapping) and result.get("received") is True
                            for result in results
                        ):
                            raise RuntimeError(
                                f"Colocated direct endpoint {endpoint} returned {results}"
                            )
                        received_workers = sorted(
                            (
                                int(result["dp_rank"]),
                                int(result["tp_rank"]),
                                str(result["physical_device_id"]),
                            )
                            for result in results
                        )
                        received_identities = {
                            (dp_rank, tp_rank): physical_device_id
                            for dp_rank, tp_rank, physical_device_id in received_workers
                        }
                        returned_dp_ranks = sorted(
                            {dp_rank for dp_rank, _tp_rank in received_identities}
                        )
                        if (
                            len(received_identities) != len(received_workers)
                            or not returned_dp_ranks
                            or any(
                                {
                                    tp_rank
                                    for worker_dp_rank, tp_rank in received_identities
                                    if worker_dp_rank == dp_rank
                                }
                                != expected_tp_ranks
                                for dp_rank in returned_dp_ranks
                            )
                            or any(
                                expected_by_identity.get(identity) != physical_device_id
                                for identity, physical_device_id in received_identities.items()
                            )
                        ):
                            raise RuntimeError(
                                f"Colocated direct endpoint {endpoint} returned {results}"
                            )
            except Exception as error:  # pylint: disable=W0718
                send_error = error
        synchronize_error(send_error, "colocated direct IPC transfer")

    def publish(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Publish one FSDP policy without materializing full Actor tensors."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError("Colocated direct reshard requires a vLLM HTTP client")
        self._weights_awake = False
        local_state_dict = synchronized_call(
            "colocated direct local-state extraction",
            lambda: self._mapped_local_state_dict(snapshot.payload),
        )
        retained_buffers: list[Any] = []
        transfer_complete = False
        try:
            self._run_control(
                "colocated direct weight wake",
                client,
                lambda: client.wake_up(("weights",)),
            )
            self._weights_awake = True
            plan = synchronized_call(
                "colocated direct layout planning",
                lambda: self._ensure_plan(client, local_state_dict),
            )
            self._run_control("colocated direct pause", client, client.pause)
            self._run_control(
                "colocated direct start",
                client,
                client.start_weight_update,
            )
            payload, retained_buffers = synchronized_call(
                "colocated direct redistribution",
                lambda: self._redistribute_and_export(
                    local_state_dict,
                    plan,
                    self._data_parallel_size,
                ),
            )
            endpoints = synchronized_call(
                "colocated direct endpoints",
                lambda: self._gather_endpoints(client),
            )
            self._send_payload(
                client,
                endpoints,
                payload,
                snapshot.version,
                plan.destination_tp_size,
            )
            synchronized_call(
                "colocated direct producer synchronization",
                platform.get_current_stream().synchronize,
            )
            self._run_control(
                "colocated direct finish",
                client,
                client.finish_weight_update,
            )
            synchronized_call(
                "colocated direct rollout parameter manifest",
                lambda: _write_rollout_parameter_manifest(
                    client,
                    strategy="direct_reshard",
                    policy_version=snapshot.version,
                    data_parallel_size=self._data_parallel_size,
                ),
            )
            expected_fingerprint = synchronized_call(
                "colocated direct Actor fingerprint",
                lambda: self._distributed_policy_fingerprint(local_state_dict),
            )

            def verify_local_fingerprint() -> None:
                """Compare every rollout worker against the transferred policy."""
                client.verify_policy_weight_identity(
                    snapshot.version,
                    expected_fingerprint,
                )

            synchronized_call(
                "colocated direct policy fingerprint",
                verify_local_fingerprint,
            )
            self.last_policy_fingerprint = expected_fingerprint
            self.last_strategy = "direct_reshard"
            self.direct_success_count += 1
            transfer_complete = True
        finally:
            if transfer_complete:
                retained_buffers.clear()
            else:
                self._failed_buffers.extend(retained_buffers)
                retained_buffers.clear()
            local_state_dict.clear()

    def release_failed_buffers(self) -> None:
        """Release failed IPC producers after fallback fully overwrote the transaction."""
        self._failed_buffers.clear()

    def close(self) -> None:
        """Release route metadata and any failed IPC producer buffers."""
        super().close()
        self._failed_buffers.clear()


def _open_port() -> int:
    """Return an unused loopback port for one HCCL rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class FullGatherHCCLWeightTransfer(_RefitCompatibleTransfer):
    """All-gather FSDP weights, then send them from Actor rank 0 over HCCL."""

    def __init__(
        self,
        model: VLLMModelRegistration,
        *,
        bucket_size_bytes: int = 128 * 2**20,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
    ) -> None:
        """Store the rollout contract and defer the shared HCCL group."""
        if bucket_size_bytes <= 0:
            raise ValueError("Full-gather bucket_size_bytes must be positive")
        if data_parallel_size <= 0:
            raise ValueError("Full-gather data_parallel_size must be positive")
        if tensor_parallel_size <= 0:
            raise ValueError("Full-gather tensor_parallel_size must be positive")
        self._model = model
        self._bucket_size_bytes = int(bucket_size_bytes)
        self._data_parallel_size = int(data_parallel_size)
        self._tensor_parallel_size = int(tensor_parallel_size)
        self._group: Any = None
        self._endpoint: Optional[str] = None
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None
        self.configured_strategy = "full_gather"
        self.last_strategy: Optional[str] = None
        self.fallback_count = 0
        self.direct_success_count = 0

    def _mapped_full_state_dict(self, payload: Any) -> dict[str, Any]:
        """Gather complete tensors and map them to checkpoint-format names."""
        mapped = map_actor_state_dict(
            _full_state_dict(payload, operation="vLLM full-gather refit"),
            self._model,
        )
        mapped = _alias_tied_embeddings(mapped, self._model)
        return {name: mapped[name] for name in sorted(mapped)}

    @staticmethod
    def _gather_endpoints(client: VLLMWeightSyncClientMixin) -> tuple[str, ...]:
        """Require every Trainer rank to use one shared rollout endpoint."""
        world_size = platform.get_world_size()
        endpoints = [""] * world_size
        platform.all_gather_object(endpoints, client.base_url)
        unique = tuple(sorted(set(endpoints)))
        if len(unique) != 1 or any(not endpoint for endpoint in unique):
            raise RuntimeError(
                "Full-gather HCCL requires one shared rollout endpoint, "
                f"got {endpoints}"
            )
        return unique

    def _ensure_group(
        self,
        client: VLLMWeightSyncClientMixin,
        endpoint: str,
    ) -> Any:
        """Create one Actor-rank-0-to-all-rollout-workers HCCL group."""
        if self._group is not None:
            if endpoint != self._endpoint:
                raise RuntimeError(
                    f"Full-gather rollout endpoint changed: {self._endpoint} -> {endpoint}"
                )
            return self._group
        expected_world_size = self._data_parallel_size * self._tensor_parallel_size
        inference_world_size = client.get_world_size(base_url=endpoint)
        if inference_world_size != expected_world_size:
            raise RuntimeError(
                "Full-gather rollout world size differs from configured DP x TP: "
                f"expected={expected_world_size}, actual={inference_world_size}"
            )
        init_info = {
            "master_address": "127.0.0.1",
            "master_port": _open_port(),
            "world_size": expected_world_size + 1,
        }
        from vllm_ascend.distributed.weight_transfer.hccl_engine import (  # pylint: disable=C0415
            HCCLWeightTransferEngine,
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            server_init = executor.submit(
                client.collective_rpc,
                "init_full_gather_group",
                {
                    **init_info,
                    "expected_data_parallel_size": self._data_parallel_size,
                    "expected_tensor_parallel_size": self._tensor_parallel_size,
                },
                endpoint,
            )
            group = HCCLWeightTransferEngine.trainer_init(init_info)
            try:
                worker_results = server_init.result(timeout=180)
            except Exception:
                group = None
                raise
        if not isinstance(worker_results, list) or not worker_results:
            raise RuntimeError(
                f"Full-gather HCCL group returned invalid workers: {worker_results}"
            )
        for result in worker_results:
            if not isinstance(result, Mapping) or not bool(result.get("joined")):
                raise RuntimeError(f"Full-gather HCCL group returned invalid ACK: {result!r}")
            expected_rank = (
                1
                + int(result["dp_rank"]) * self._tensor_parallel_size
                + int(result["tp_rank"])
            )
            if int(result["group_rank"]) != expected_rank:
                raise RuntimeError(f"Full-gather HCCL group returned invalid rank: {result}")
        self._group = group
        self._endpoint = endpoint
        return self._group

    @staticmethod
    def _device_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
        """Keep complete tensors contiguous on the current Actor NPU."""
        device_handle = platform.get_device_handle(platform.device_type())
        target_device = f"{platform.device_type()}:{device_handle.current_device()}"
        result = {}
        for name, tensor in state_dict.items():
            if str(tensor.device).startswith("cpu"):
                tensor = tensor.to(target_device)
            result[name] = tensor.detach().contiguous()
        return result

    def _send_shared(
        self,
        client: VLLMWeightSyncClientMixin,
        endpoint: str,
        state_dict: Mapping[str, Any],
        policy_version: int,
    ) -> None:
        """Send one complete checkpoint to all shared rollout workers."""
        names = list(state_dict)
        dtype_names = [
            str(state_dict[name].dtype).rsplit(".", maxsplit=1)[-1]
            for name in names
        ]
        shapes = [list(state_dict[name].shape) for name in names]
        total_bytes = sum(
            state_dict[name].numel() * state_dict[name].element_size()
            for name in names
        )
        packed_buffer_size_bytes = total_bytes + self._bucket_size_bytes
        update_info = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "packed": True,
            "packed_buffer_size_bytes": packed_buffer_size_bytes,
            "packed_num_buffers": 1,
        }
        from vllm_ascend.distributed.weight_transfer.hccl_engine import (  # pylint: disable=C0415
            HCCLTrainerSendWeightsArgs,
            HCCLWeightTransferEngine,
        )

        group = self._ensure_group(client, endpoint)
        with ThreadPoolExecutor(max_workers=1) as executor:
            server_update = executor.submit(
                client.receive_weights,
                update_info,
                policy_version,
                endpoint,
            )
            HCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(state_dict.items()),
                trainer_args=HCCLTrainerSendWeightsArgs(
                    group=group,
                    packed=True,
                    packed_buffer_size_bytes=packed_buffer_size_bytes,
                    packed_num_buffers=1,
                ),
            )
            server_update.result(timeout=600)

    def publish(
        self,
        client: Any,
        snapshot: PolicySnapshot,
        *,
        manifest_strategy: str = "full_gather",
    ) -> None:
        """Publish one full Actor checkpoint to every disjoint TP replica."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError("Full-gather HCCL requires an external vLLM HTTP client")
        state_dict = synchronized_call(
            "full-gather state extraction",
            lambda: self._mapped_full_state_dict(snapshot.payload),
        )
        try:
            state_dict = synchronized_call(
                "full-gather NPU staging",
                lambda: self._device_state_dict(state_dict),
            )
            endpoints = synchronized_call(
                "full-gather HCCL endpoints",
                lambda: self._gather_endpoints(client),
            )
            coordinator_call("full-gather pause", client.pause)
            coordinator_call("full-gather start", client.start_weight_update)
            synchronized_call(
                "full-gather producer synchronization",
                platform.get_current_stream().synchronize,
            )
            coordinator_call(
                "full-gather HCCL transfer",
                lambda: self._send_shared(
                    client,
                    endpoints[0],
                    state_dict,
                    snapshot.version,
                ),
            )
            coordinator_call("full-gather finish", client.finish_weight_update)
            coordinator_call(
                "full-gather rollout parameter manifest",
                lambda: _write_rollout_parameter_manifest(
                    client,
                    strategy=manifest_strategy,
                    policy_version=snapshot.version,
                    data_parallel_size=self._data_parallel_size,
                ),
            )
            expected_fingerprint = synchronized_call(
                "full-gather Actor fingerprint",
                lambda: policy_weight_fingerprint(state_dict),
            )

            def verify_worker_fingerprints() -> None:
                """Require every shared rollout worker to match the full Actor."""
                client.verify_policy_weight_identity(
                    snapshot.version,
                    expected_fingerprint,
                )

            coordinator_call("full-gather policy fingerprint", verify_worker_fingerprints)
            self.last_policy_fingerprint = expected_fingerprint
            self.last_strategy = "full_gather"
        finally:
            state_dict.clear()

    def close(self) -> None:
        """Release cached trainer-side HCCL group references."""
        self._group = None
        self._endpoint = None


class ColocatedFullGatherWeightTransfer(_RefitCompatibleTransfer):
    """All-gather FSDP weights and expose complete tensors through NPU IPC."""

    def __init__(
        self,
        model: VLLMModelRegistration,
        data_parallel_size: int = 1,
    ) -> None:
        """Store the model contract and failed asynchronous buffer ownership."""
        if data_parallel_size <= 0:
            raise ValueError("Full-gather data_parallel_size must be positive")
        self._model = model
        self._data_parallel_size = int(data_parallel_size)
        self._failed_state_dict: dict[str, Any] = {}
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None
        self.configured_strategy = "full_gather"
        self.last_strategy: Optional[str] = None
        self.fallback_count = 0
        self.direct_success_count = 0

    @staticmethod
    def _run_control(
        operation: str,
        client: VLLMWeightSyncClientMixin,
        callback: Any,
    ) -> None:
        """Run one mutating call per colocated rollout replica."""
        synchronized_call(
            operation,
            lambda: callback()
            if bool(getattr(client, "is_server_owner", True))
            else None,
        )

    def _mapped_full_state_dict(self, payload: Any) -> dict[str, Any]:
        """Validate local shards, gather them, and stage a complete checkpoint."""
        local_state_dict = _local_state_dict(
            payload,
            operation="vLLM colocated full-gather refit",
        )
        try:
            self._validate_metadata(local_state_dict)
        finally:
            local_state_dict.clear()
        full_state_dict = _full_state_dict(
            payload,
            operation="vLLM colocated full-gather refit",
        )
        try:
            mapped = map_actor_state_dict(full_state_dict, self._model)
            mapped = _alias_tied_embeddings(mapped, self._model)
        finally:
            full_state_dict.clear()
        device_handle = platform.get_device_handle(platform.device_type())
        target_device = f"{platform.device_type()}:{device_handle.current_device()}"
        result = {}
        for name in sorted(mapped):
            tensor = mapped[name]
            if str(tensor.device).startswith("cpu"):
                tensor = tensor.to(target_device)
            result[name] = tensor.detach().contiguous()
        return result

    @staticmethod
    def _validate_metadata(state_dict: Mapping[str, Any]) -> None:
        """Require every FSDP rank to expose identical tensor metadata."""
        world_size = platform.get_world_size()
        if world_size <= 1:
            return
        metadata = [
            (
                name,
                str(
                    (
                        tensor.to_local()
                        if callable(getattr(tensor, "to_local", None))
                        else tensor
                    ).dtype
                ),
                tuple(tensor.shape),
                tuple(repr(placement) for placement in getattr(tensor, "placements", ()) or ()),
                tuple(
                    getattr(getattr(tensor, "device_mesh", None), "mesh_shape", ())
                    or ()
                ),
                tuple(
                    getattr(getattr(tensor, "device_mesh", None), "mesh_dim_names", ())
                    or ()
                ),
            )
            for name, tensor in state_dict.items()
        ]
        gathered: list[Any] = [None] * world_size
        platform.all_gather_object(gathered, metadata)
        if any(rank_metadata != metadata for rank_metadata in gathered):
            raise RuntimeError("NPU IPC refit metadata differs across FSDP ranks")

    @staticmethod
    def _transfer_endpoints(client: VLLMWeightSyncClientMixin) -> tuple[str, ...]:
        """Return the one shared colocated endpoint."""
        return (client.base_url,)

    @staticmethod
    def _build_local_handles(
        state_dict: Mapping[str, Any],
        generate_uuid: Any,
    ) -> list[dict[Any, Any]]:
        """Export one local IPC handle for every full parameter tensor."""
        npu_uuid = generate_uuid()
        return [
            {npu_uuid: _tensor_ipc_rebuild_args(tensor)}
            for tensor in state_dict.values()
        ]

    @staticmethod
    def _merge_handles(local_handles: list[dict[Any, Any]]) -> list[dict[Any, Any]]:
        """Combine same-parameter handles from every colocated Actor NPU."""
        gathered_handles: list[Any] = [None] * platform.get_world_size()
        platform.all_gather_object(gathered_handles, local_handles)
        merged_handles = []
        for parameter_index in range(len(local_handles)):
            merged_handle = {}
            for rank_handles in gathered_handles:
                merged_handle.update(rank_handles[parameter_index])
            merged_handles.append(merged_handle)
        return merged_handles

    @staticmethod
    def _send_payload(
        client: VLLMWeightSyncClientMixin,
        endpoints: tuple[str, ...],
        update_info: Any,
        policy_version: int,
    ) -> None:
        """Send merged IPC handles once to every colocated rollout replica."""
        send_error = None
        if platform.get_rank() == 0:
            try:
                with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
                    requests = [
                        executor.submit(
                            client.receive_ipc_weights,
                            endpoint,
                            update_info,
                            policy_version,
                        )
                        for endpoint in endpoints
                    ]
                    for request in requests:
                        request.result(timeout=600)
            except Exception as error:  # pylint: disable=W0718
                send_error = error
        synchronize_error(send_error, "colocated full-gather IPC transfer")

    def publish(
        self,
        client: Any,
        snapshot: PolicySnapshot,
        *,
        weights_already_awake: bool = False,
        manifest_strategy: str = "full_gather",
    ) -> None:
        """Publish complete Actor tensors to every colocated TP replica."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError("Colocated full-gather requires a vLLM HTTP client")
        state_dict = synchronized_call(
            "colocated full-gather state extraction",
            lambda: self._mapped_full_state_dict(snapshot.payload),
        )
        try:
            if not weights_already_awake:
                self._run_control(
                    "colocated full-gather weight wake",
                    client,
                    lambda: client.wake_up(("weights",)),
                )
            endpoints = synchronized_call(
                "colocated full-gather endpoints",
                lambda: self._transfer_endpoints(client),
            )
            self._run_control("colocated full-gather pause", client, client.pause)
            self._run_control(
                "colocated full-gather start",
                client,
                client.start_weight_update,
            )

            def setup_transport() -> tuple[Any, Any]:
                """Synchronize producers before importing optional IPC helpers."""
                platform.get_current_stream().synchronize()
                from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (  # pylint: disable=C0415
                    NPUIPCWeightTransferUpdateInfo,
                    npu_generate_uuid,
                )
                return NPUIPCWeightTransferUpdateInfo, npu_generate_uuid

            update_info_class, generate_uuid = synchronized_call(
                "colocated full-gather IPC setup",
                setup_transport,
            )
            local_handles = synchronized_call(
                "colocated full-gather IPC handles",
                lambda: self._build_local_handles(state_dict, generate_uuid),
            )
            merged_handles = synchronized_call(
                "colocated full-gather IPC handle merge",
                lambda: self._merge_handles(local_handles),
            )
            update_info = synchronized_call(
                "colocated full-gather IPC update construction",
                lambda: update_info_class(
                    names=list(state_dict),
                    dtype_names=[
                        str(tensor.dtype).rsplit(".", maxsplit=1)[-1]
                        for tensor in state_dict.values()
                    ],
                    shapes=[list(tensor.shape) for tensor in state_dict.values()],
                    ipc_handles=merged_handles,
                    packed=False,
                ),
            )
            try:
                self._send_payload(
                    client,
                    endpoints,
                    update_info,
                    snapshot.version,
                )
            except Exception:
                self._failed_state_dict = state_dict
                state_dict = {}
                raise
            synchronized_call(
                "colocated full-gather producer synchronization",
                platform.get_current_stream().synchronize,
            )
            self._run_control(
                "colocated full-gather finish",
                client,
                client.finish_weight_update,
            )
            synchronized_call(
                "colocated full-gather rollout parameter manifest",
                lambda: _write_rollout_parameter_manifest(
                    client,
                    strategy=manifest_strategy,
                    policy_version=snapshot.version,
                    data_parallel_size=self._data_parallel_size,
                ),
            )
            expected_fingerprint = synchronized_call(
                "colocated full-gather Actor fingerprint",
                lambda: policy_weight_fingerprint(state_dict),
            )

            def verify_local_fingerprint() -> None:
                """Compare every rollout worker against the transferred policy."""
                client.verify_policy_weight_identity(
                    snapshot.version,
                    expected_fingerprint,
                )

            self._run_control(
                "colocated full-gather policy fingerprint",
                client,
                verify_local_fingerprint,
            )
            self.last_policy_fingerprint = expected_fingerprint
            self.last_strategy = "full_gather"
        finally:
            state_dict.clear()

    def close(self) -> None:
        """Release tensors retained after a failed asynchronous IPC update."""
        self._failed_state_dict.clear()


class FallbackWeightTransfer(_RefitCompatibleTransfer):
    """Try direct reshard first and recover with a complete-model transfer."""

    def __init__(self, primary: WeightTransfer, fallback: WeightTransfer) -> None:
        """Store both strategies and expose the successful publication identity."""
        self._primary = primary
        self._fallback = fallback
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None
        self.configured_strategy = "direct_reshard"
        self.last_strategy: Optional[str] = None
        self.last_primary_error: Optional[str] = None
        self.fallback_count = 0
        self.direct_success_count = 0

    @staticmethod
    def _current_policy_version(client: VLLMWeightSyncClientMixin) -> int:
        """Read committed versions without touching potentially sleeping weights."""
        versions = {
            int(result["version"])
            for result in client.collective_rpc("get_policy_version")
        }
        if len(versions) != 1:
            raise RuntimeError(
                "Rollout versions differ before direct-reshard fallback: "
                f"{sorted(versions)}"
            )
        return versions.pop()

    @staticmethod
    def _abort_direct_update(
        client: VLLMWeightSyncClientMixin,
        restore_policy_version: int,
    ) -> None:
        """Discard pending identity so a full update can overwrite partial weights."""
        def abort_and_validate() -> None:
            """Require every worker to acknowledge transaction recovery."""
            results = client.collective_rpc(
                "abort_weight_update",
                kwargs={"restore_policy_version": restore_policy_version},
            )
            if results and all(
                isinstance(result, Mapping) and bool(result.get("aborted"))
                for result in results
            ):
                return
            raise RuntimeError(f"vLLM rejected direct-update abort: {results}")

        coordinator_call("direct-reshard fallback abort", abort_and_validate)

    def publish(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Use full gather only when the direct transaction raises an error."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError("Fallback weight transfer requires a vLLM HTTP client")
        previous_version = coordinator_call(
            "direct-reshard fallback baseline",
            lambda: self._current_policy_version(client),
        )
        try:
            self._primary.publish(client, snapshot)
        except Exception as direct_error:  # pylint: disable=W0718
            self.last_primary_error = repr(direct_error)
            if platform.get_rank() == 0:
                logger.warning(
                    "direct reshard failed; falling back to full gather: %r",
                    direct_error,
                )
            try:
                self._abort_direct_update(client, previous_version)
            except Exception as abort_error:
                raise RuntimeError(
                    "Direct reshard failed and its transaction could not be aborted: "
                    f"direct={direct_error!r}, abort={abort_error!r}"
                ) from abort_error
            try:
                if isinstance(self._fallback, ColocatedFullGatherWeightTransfer):
                    self._fallback.publish(
                        client,
                        snapshot,
                        weights_already_awake=bool(
                            getattr(self._primary, "weights_awake", False)
                        ),
                        manifest_strategy="full_gather_fallback",
                    )
                elif isinstance(self._fallback, FullGatherHCCLWeightTransfer):
                    self._fallback.publish(
                        client,
                        snapshot,
                        manifest_strategy="full_gather_fallback",
                    )
                else:
                    self._fallback.publish(client, snapshot)
            except Exception as fallback_error:
                try:
                    self._abort_direct_update(client, previous_version)
                except Exception as abort_error:
                    raise RuntimeError(
                        "Direct reshard and full-gather fallback failed, then recovery "
                        "also failed: "
                        f"direct={direct_error!r}, fallback={fallback_error!r}, "
                        f"abort={abort_error!r}"
                    ) from abort_error
                raise RuntimeError(
                    "Both direct reshard and full-gather fallback failed: "
                    f"direct={direct_error!r}, fallback={fallback_error!r}"
                ) from fallback_error
            release_failed_buffers = getattr(self._primary, "release_failed_buffers", None)
            if callable(release_failed_buffers):
                release_failed_buffers()
            self.last_strategy = "full_gather"
            self.fallback_count += 1
            self.last_policy_fingerprint = getattr(
                self._fallback,
                "last_policy_fingerprint",
                None,
            )
            return
        self.last_strategy = "direct_reshard"
        self.direct_success_count += 1
        self.last_primary_error = None
        self.last_policy_fingerprint = getattr(
            self._primary,
            "last_policy_fingerprint",
            None,
        )

    def close(self) -> None:
        """Release resources owned by both transfer strategies."""
        for transfer in (self._primary, self._fallback):
            close = getattr(transfer, "close", None)
            if callable(close):
                close()


def build_weight_transfer(
    deployment: str,
    model: VLLMModelRegistration,
    *,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    bucket_size_bytes: int = 128 * 2**20,
    strategy: str = "full_gather",
    fallback_strategy: str = "none",
) -> WeightTransfer:
    """Build full-weight DP sync or TP-aware direct reshard with recovery."""
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be positive")
    if strategy not in ("direct_reshard", "full_gather"):
        raise ValueError(f"Unsupported weight-sync strategy: {strategy!r}")
    if fallback_strategy not in ("full_gather", "none"):
        raise ValueError(f"Unsupported weight-sync fallback: {fallback_strategy!r}")
    # A TP1 rollout replica owns every parameter in full.  Building a source-to-TP
    # redistribution plan only adds metadata, packing, and routing work without
    # changing the destination layout, so pure rollout DP always uses full weights.
    if tensor_parallel_size == 1:
        strategy = "full_gather"
        fallback_strategy = "none"
    if deployment == "colocated":
        full = ColocatedFullGatherWeightTransfer(
            model,
            data_parallel_size=data_parallel_size,
        )
    elif deployment == "disjoint":
        full = FullGatherHCCLWeightTransfer(
            model,
            bucket_size_bytes=bucket_size_bytes,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
    else:
        raise ValueError(f"Unsupported rollout deployment: {deployment!r}")
    if strategy == "full_gather":
        return full
    if deployment == "colocated":
        direct = ColocatedDirectReshardWeightTransfer(
            model,
            bucket_size_bytes=bucket_size_bytes,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
    else:
        direct = DirectReshardHCCLWeightTransfer(
            model,
            bucket_size_bytes=bucket_size_bytes,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
    if fallback_strategy == "full_gather":
        return FallbackWeightTransfer(direct, full)
    return direct


__all__ = [
    "ColocatedFullGatherWeightTransfer",
    "ColocatedDirectReshardWeightTransfer",
    "DirectReshardHCCLWeightTransfer",
    "FallbackWeightTransfer",
    "FullGatherHCCLWeightTransfer",
    "POLICY_FINGERPRINT_ALGORITHM",
    "WeightTransfer",
    "aggregate_policy_fingerprint",
    "build_weight_transfer",
    "canonical_policy_weight_name",
    "is_policy_fingerprint_weight",
    "map_actor_state_dict",
    "policy_fingerprint_header",
    "policy_tensor_fingerprint",
    "policy_weight_fingerprint",
    "verify_policy_fingerprints",
]
