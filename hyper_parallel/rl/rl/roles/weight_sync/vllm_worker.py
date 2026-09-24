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
"""vLLM worker hooks used by Actor-to-rollout weight synchronization."""
import base64
import os
import pickle
from dataclasses import dataclass
from typing import Any, Mapping, Optional

# Installed worker hooks share RL-owned version state on foreign vLLM worker instances.
# vLLM has no public policy-version API for these transaction fields.
# pylint: disable=protected-access
import torch

from rl.roles.model_setup import (
    HYPER_QWEN3_ARCHITECTURE,
    NATIVE_QWEN3_ARCHITECTURE,
)
from rl.roles.weight_sync.model_adapter import rollout_tensor_descriptions
from rl.roles.weight_sync.packed_weight import unpack_packed_weights
from rl.roles.weight_sync.vllm_client import KEEP_SCHEDULER_PAUSED_TAG

_SUPPORTED_ARCHITECTURES = frozenset(
    (
        HYPER_QWEN3_ARCHITECTURE,
        NATIVE_QWEN3_ARCHITECTURE,
    )
)


@dataclass
class _PatchState:
    """Track process-local idempotent vLLM patch installation."""

    ascend_lifecycle: bool = False
    engine_core_wake: bool = False


_patch_state = _PatchState()


def _rollout_worker_topology(worker: Any) -> dict[str, Any]:
    """Return this worker's explicit DP, TP, and physical-device identity."""
    from vllm.distributed import get_tp_group  # pylint: disable=C0415
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (  # pylint: disable=C0415
        npu_generate_uuid,
    )

    parallel_config = worker.parallel_config
    tp_rank = int(get_tp_group().rank_in_group)
    tp_size = int(get_tp_group().world_size)
    physical_device_id = npu_generate_uuid()
    visible_devices = tuple(
        device.strip()
        for device in os.environ.get("HYPER_RL_ROLLOUT_VISIBLE_DEVICES", "").split(",")
        if device.strip()
    )
    if visible_devices:
        physical_index = str(physical_device_id).rsplit("-", maxsplit=1)[-1]
        if physical_index not in visible_devices:
            raise RuntimeError(
                "Rollout worker physical device is absent from the shared deployment: "
                f"device={physical_device_id}, visible={visible_devices}"
            )
        worker_index = visible_devices.index(physical_index)
        if len(visible_devices) % tp_size != 0 or worker_index % tp_size != tp_rank:
            raise RuntimeError(
                "Rollout worker physical order differs from its TP rank: "
                f"device={physical_device_id}, index={worker_index}, tp_rank={tp_rank}, "
                f"tp_size={tp_size}, visible={visible_devices}"
            )
        dp_rank = worker_index // tp_size
        dp_size = len(visible_devices) // tp_size
    else:
        dp_rank = int(parallel_config.data_parallel_index)
        dp_size = int(parallel_config.data_parallel_size)
    return {
        "dp_rank": dp_rank,
        "dp_size": dp_size,
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "physical_device_id": physical_device_id,
    }


def _validate_topology(
    topology: Mapping[str, Any],
    *,
    expected_data_parallel_size: int,
    expected_tensor_parallel_size: int,
) -> tuple[int, int]:
    """Validate one worker against the controller-owned rollout topology."""
    expected_dp_size = int(expected_data_parallel_size)
    expected_tp_size = int(expected_tensor_parallel_size)
    if expected_dp_size <= 0 or expected_tp_size <= 0:
        raise ValueError("Expected rollout DP and TP sizes must be positive")
    actual_dp_size = int(topology["dp_size"])
    actual_tp_size = int(topology["tp_size"])
    dp_rank = int(topology["dp_rank"])
    tp_rank = int(topology["tp_rank"])
    # Dense vLLM engines may expose engine-local DP size 1 while retaining
    # the deployment-global data_parallel_index used below.
    if actual_dp_size not in (1, expected_dp_size) or actual_tp_size != expected_tp_size:
        raise ValueError(
            "Rollout worker topology differs from the configured DP x TP: "
            f"expected=({expected_dp_size}, {expected_tp_size}), "
            f"actual=({actual_dp_size}, {actual_tp_size})"
        )
    if not 0 <= dp_rank < expected_dp_size or not 0 <= tp_rank < expected_tp_size:
        raise ValueError(
            "Rollout worker rank is outside the configured DP x TP: "
            f"rank=({dp_rank}, {tp_rank}), size=({expected_dp_size}, {expected_tp_size})"
        )
    return dp_rank, tp_rank


def get_policy_version(worker: Any) -> dict[str, int]:
    """Return the committed version without touching sleeping model tensors."""
    return {"version": int(getattr(worker, "_hyper_loaded_policy_version", 0))}


def _receive_ack(
    topology: Mapping[str, Any],
    num_bytes: int,
    **details: Any,
) -> dict[str, Any]:
    """Return the common successful worker receive acknowledgement."""
    return {
        "received": True,
        "dp_rank": int(topology["dp_rank"]),
        "tp_rank": int(topology["tp_rank"]),
        "bytes": int(num_bytes),
        **details,
    }


def get_direct_reshard_layout(worker: Any) -> dict[str, Any]:
    """Describe local Qwen3 storage and the worker's physical ownership."""
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    if not _is_supported_worker(worker):
        raise ValueError("Direct reshard requires a supported Qwen3 rollout model")
    topology = _rollout_worker_topology(worker)
    model = worker.model_runner.get_model()
    return {
        **topology,
        "tensors": rollout_tensor_descriptions(
            model, getattr(worker.model_config, "hf_config", None),
            is_hyper=_is_hyper_worker(worker),
            tp_rank=int(topology["tp_rank"]), tp_size=int(topology["tp_size"]),
        ),
    }


def _join_hccl_group(
    worker: Any,
    *,
    attribute: str,
    group_id: str,
    master_address: str,
    master_port: int,
    receiver_rank: int,
    world_size: int,
) -> None:
    """Create and retain one worker-side stateless HCCL group."""
    groups = getattr(worker, attribute, None)
    if groups is None:
        groups = {}
        setattr(worker, attribute, groups)
    if group_id in groups:
        return
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (  # pylint: disable=C0415
        HCCLWeightTransferEngine,
    )

    device = int(torch.get_device_module().current_device())
    groups[group_id] = HCCLWeightTransferEngine._stateless_init_process_group(  # pylint: disable=W0212
        master_address,
        int(master_port),
        int(receiver_rank),
        int(world_size),
        device=device,
    )


def _target_tp_rank(value: int, tensor_parallel_size: int) -> int:
    """Validate and return one direct destination TP rank."""
    target = int(value)
    if not 0 <= target < int(tensor_parallel_size):
        raise ValueError(
            "Direct reshard target TP rank is outside the configured topology: "
            f"rank={target}, size={tensor_parallel_size}"
        )
    return target


def _direct_worker(
    worker: Any,
    target_tp_rank: int,
    data_parallel_size: int,
    tensor_parallel_size: int,
) -> tuple[dict[str, Any], int, int, int]:
    """Resolve one direct RPC's worker and target coordinates."""
    topology = _rollout_worker_topology(worker)
    dp_rank, tp_rank = _validate_topology(
        topology,
        expected_data_parallel_size=data_parallel_size,
        expected_tensor_parallel_size=tensor_parallel_size,
    )
    target = _target_tp_rank(target_tp_rank, tensor_parallel_size)
    return topology, dp_rank, tp_rank, target


def init_direct_reshard_group(
    worker: Any,
    *,
    group_id: str,
    target_tp_rank: int,
    master_address: str,
    master_port: int,
    world_size: int,
    expected_data_parallel_size: int,
    expected_tensor_parallel_size: int,
) -> dict[str, Any]:
    """Join one source-rank-to-target-TP stateless HCCL broadcast group."""
    unused_topology, dp_rank, tp_rank, target_tp_rank = _direct_worker(
        worker,
        target_tp_rank,
        expected_data_parallel_size,
        expected_tensor_parallel_size,
    )
    expected_world_size = 1 + int(expected_data_parallel_size)
    if int(world_size) != expected_world_size:
        raise ValueError(
            "Direct reshard HCCL group world size differs from configured rollout DP: "
            f"expected={expected_world_size}, actual={world_size}"
        )
    if tp_rank != target_tp_rank:
        return {
            "joined": False,
            "dp_rank": dp_rank,
            "tp_rank": tp_rank,
            "group_rank": None,
        }
    receiver_rank = 1 + dp_rank
    _join_hccl_group(
        worker,
        attribute="_hyper_direct_reshard_groups",
        group_id=group_id,
        master_address=master_address,
        master_port=master_port,
        receiver_rank=receiver_rank,
        world_size=world_size,
    )
    return {
        "joined": True,
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "group_rank": receiver_rank,
        "group_id": group_id,
    }


def init_packed_weight_group(
    worker: Any,
    *,
    group_id: str,
    master_address: str,
    master_port: int,
    world_size: int,
    expected_data_parallel_size: int,
    expected_tensor_parallel_size: int,
) -> dict[str, Any]:
    """Join one producer-to-all-workers HCCL group for packed weights."""
    topology = _rollout_worker_topology(worker)
    dp_rank, tp_rank = _validate_topology(
        topology,
        expected_data_parallel_size=expected_data_parallel_size,
        expected_tensor_parallel_size=expected_tensor_parallel_size,
    )
    expected_world_size = (
        1 + int(expected_data_parallel_size) * int(expected_tensor_parallel_size)
    )
    if int(world_size) != expected_world_size:
        raise ValueError(
            "Packed HCCL group world size differs from rollout DP x TP: "
            f"expected={expected_world_size}, actual={world_size}"
        )
    receiver_rank = 1 + dp_rank * int(expected_tensor_parallel_size) + tp_rank
    _join_hccl_group(
        worker,
        attribute="_hyper_packed_weight_groups",
        group_id=group_id,
        master_address=master_address,
        master_port=master_port,
        receiver_rank=receiver_rank,
        world_size=world_size,
    )
    return {
        "joined": True,
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "group_rank": receiver_rank,
        "group_id": group_id,
    }


def _validate_update(worker: Any, policy_version: int, *, transport: str) -> int:
    """Validate transaction and version preconditions for one bucket."""
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    if not _is_supported_worker(worker):
        raise ValueError(f"{transport} requires a supported rollout worker")
    if not bool(getattr(worker, "_weight_update_active", False)):
        raise RuntimeError(f"{transport} requires an active vLLM weight update")
    version = int(policy_version)
    loaded_version = int(getattr(worker, "_hyper_loaded_policy_version", 0))
    pending_version = getattr(worker, "_hyper_pending_policy_version", None)
    if version <= loaded_version:
        raise ValueError(
            f"{transport} policy version must increase: "
            f"loaded={loaded_version}, received={version}"
        )
    if pending_version is not None and int(pending_version) != version:
        raise ValueError(
            "One weight update cannot mix policy versions: "
            f"pending={pending_version}, received={version}"
        )
    return version


def _apply_direct_bucket(
    parameters: Mapping[str, Any],
    packed: Any,
    metadata: Mapping[str, Any],
    *,
    transport: str,
) -> int:
    """Scatter one packed direct bucket into rollout-local parameters."""

    received_bytes = 0
    for entry in metadata["entries"]:
        name = str(entry["name"])
        parameter = parameters.get(name)
        if parameter is None:
            raise ValueError(f"{transport} parameter {name!r} is missing")
        source_dtype = getattr(torch, str(entry["dtype_name"]))
        destination_dtype = getattr(
            torch,
            str(entry.get("destination_dtype_name", entry["dtype_name"])),
        )
        destination_element_size = int(
            entry.get("destination_element_size", entry["element_size"])
        )
        if (
            int(parameter.element_size()) != destination_element_size
            or parameter.dtype != destination_dtype
        ):
            raise ValueError(
                f"{transport} parameter {name!r} dtype mismatch: "
                f"parameter={parameter.dtype}, destination={destination_dtype}"
            )
        lengths = tuple(
            int(value)
            for value in entry.get("destination_lengths", entry["lengths"])
        )
        starts = tuple(int(value) for value in entry["destination_starts"])
        num_bytes = int(entry["num_bytes"])
        offset = int(entry["buffer_offset"])
        fragment = packed.narrow(0, offset, num_bytes).view(source_dtype).view(lengths)
        destination_slice = tuple(
            slice(start, start + length) for start, length in zip(starts, lengths)
        )
        target = parameter[destination_slice]
        if tuple(target.shape) != lengths:
            raise ValueError(
                f"{transport} destination {name!r} has shape "
                f"{tuple(target.shape)}, expected {lengths}"
            )
        with torch.no_grad():
            target.copy_(fragment)
        received_bytes += num_bytes
    return received_bytes


def receive_direct_reshard(
    worker: Any,
    *,
    group_id: str,
    target_tp_rank: int,
    buckets: list[Mapping[str, Any]],
    policy_version: int,
    expected_data_parallel_size: int,
    expected_tensor_parallel_size: int,
) -> dict[str, Any]:
    """Receive bounded source fragments and write them into local TP parameters."""
    topology, dp_rank, tp_rank, target_tp_rank = _direct_worker(
        worker,
        target_tp_rank,
        expected_data_parallel_size,
        expected_tensor_parallel_size,
    )
    if tp_rank != target_tp_rank:
        return {
            "received": False,
            "dp_rank": dp_rank,
            "tp_rank": tp_rank,
            "bytes": 0,
        }
    version = _validate_update(worker, policy_version, transport="Direct reshard")
    groups = getattr(worker, "_hyper_direct_reshard_groups", {})
    group = groups.get(group_id)
    if group is None:
        raise RuntimeError(f"Direct reshard HCCL group {group_id!r} is not initialized")

    received_bytes = _receive_direct_buckets(worker, group, buckets)
    worker._hyper_pending_policy_version = version
    return _receive_ack(
        topology,
        received_bytes,
        bucket_count=len(buckets),
    )


def _receive_direct_buckets(worker: Any, group: Any, buckets: list[Mapping[str, Any]]) -> int:
    """Receive, synchronize and apply each bucket in publication order."""
    parameters = dict(worker.model_runner.get_model().named_parameters())
    received_bytes = 0
    for bucket in buckets:
        total_bytes = int(bucket["total_bytes"])
        packed = torch.empty(total_bytes, dtype=torch.uint8, device=group.device)
        group.broadcast(packed, src=0)
        torch.npu.current_stream().synchronize()
        received_bytes += _apply_direct_bucket(
            parameters,
            packed,
            bucket,
            transport="Direct reshard rollout",
        )
        del packed
    return received_bytes


def _ipc_worker(
    worker: Any,
    worker_topology: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], Any]:
    """Match this worker's DP/TP coordinate to its physical NPU."""
    topology = _rollout_worker_topology(worker)
    physical_npu_id = topology["physical_device_id"]
    expected = {
        description["physical_device_id"]: description
        for description in worker_topology
    }.get(physical_npu_id)
    if expected is None:
        raise ValueError(
            f"IPC worker {physical_npu_id} is absent from the publication topology"
        )
    coordinate = (int(topology["dp_rank"]), int(topology["tp_rank"]))
    expected_coordinate = (int(expected["dp_rank"]), int(expected["tp_rank"]))
    if coordinate != expected_coordinate:
        raise ValueError(
            "IPC worker coordinate differs from its physical device: "
            f"expected={expected_coordinate}, actual={coordinate}"
        )
    return topology, physical_npu_id


def _import_ipc_buffer(handles: Mapping[Any, Any], physical_npu_id: Any) -> Any:
    """Rebuild one same-device tensor from its serialized IPC handle."""
    from torch_npu.multiprocessing.reductions import rebuild_npu_tensor  # pylint: disable=C0415

    if physical_npu_id not in handles:
        raise ValueError(
            f"IPC handle not found for {physical_npu_id}; available={list(handles)}"
        )
    rebuild_args = list(handles[physical_npu_id])
    rebuild_args[6] = torch.accelerator.current_device_index()
    return rebuild_npu_tensor(*rebuild_args)


def receive_ipc_direct_reshard(
    worker: Any,
    *,
    payload_pickled: str,
    policy_version: int,
) -> dict[str, Any]:
    """Import same-NPU packed buffers and scatter them into TP-local weights."""
    # Torch and vLLM-Ascend are optional outside the Torch-NPU RL runtime.

    version = _validate_update(worker, policy_version, transport="IPC direct")

    payload = pickle.loads(base64.b64decode(payload_pickled.encode("ascii")))
    topology, physical_npu_id = _ipc_worker(worker, payload["worker_topology"])
    tp_rank = int(topology["tp_rank"])
    worker_tp_size = int(topology.get("tp_size", 1))
    tensor_parallel_size = int(payload["tensor_parallel_size"])
    if tensor_parallel_size != worker_tp_size:
        raise ValueError(
            "IPC direct payload TP size differs from the worker topology: "
            f"payload={tensor_parallel_size}, worker={worker_tp_size}"
        )
    buckets = payload["buckets_by_target"].get(tp_rank, ())
    parameters = dict(worker.model_runner.get_model().named_parameters())
    received_bytes = 0
    imported_buffers = []

    try:
        for bucket in buckets:
            handles = bucket["ipc_handles"]
            packed = _import_ipc_buffer(handles, physical_npu_id)
            imported_buffers.append(packed)
            metadata = bucket["metadata"]
            if int(packed.numel()) != int(metadata["total_bytes"]):
                raise ValueError(
                    "IPC direct reshard packed-buffer size mismatch: "
                    f"tensor={packed.numel()}, metadata={metadata['total_bytes']}"
                )
            received_bytes += _apply_direct_bucket(
                parameters,
                packed,
                metadata,
                transport="IPC direct reshard",
            )
    finally:
        if imported_buffers:
            torch.npu.current_stream().synchronize()
            imported_buffers.clear()

    worker._hyper_pending_policy_version = version
    return _receive_ack(
        topology,
        received_bytes,
        physical_device_id=physical_npu_id,
        bucket_count=len(buckets),
    )


def _load_packed_weights(
    worker: Any,
    packed: Any,
    metadata: list[Mapping[str, Any]],
) -> None:
    """Load one complete-parameter bucket through the model's vLLM API."""
    weights = unpack_packed_weights(packed, metadata)
    model = worker.model_runner.get_model()
    if _is_hyper_worker(worker):
        loaded = model.load_weights(weights, require_all=False)
    else:
        loaded = model.load_weights(weights)
    if weights and not loaded:
        raise RuntimeError(
            "vLLM load_weights did not accept any parameter from a packed bucket"
        )


def receive_packed_weights(
    worker: Any,
    *,
    group_id: str,
    metadata: list[Mapping[str, Any]],
    total_bytes: int,
    policy_version: int,
) -> dict[str, Any]:
    """Receive one HCCL bucket and load complete parameters through vLLM."""
    version = _validate_update(worker, policy_version, transport="Packed HCCL")
    groups = getattr(worker, "_hyper_packed_weight_groups", {})
    group = groups.get(group_id)
    if group is None:
        raise RuntimeError(f"Packed HCCL group {group_id!r} is not initialized")

    packed = torch.empty(int(total_bytes), dtype=torch.uint8, device=group.device)
    group.broadcast(packed, src=0)
    torch.npu.current_stream().synchronize()
    _load_packed_weights(worker, packed, metadata)
    torch.npu.current_stream().synchronize()
    worker._hyper_pending_policy_version = version
    topology = _rollout_worker_topology(worker)
    return _receive_ack(topology, total_bytes)


def receive_ipc_packed_weights(
    worker: Any,
    *,
    payload_pickled: str,
    policy_version: int,
) -> dict[str, Any]:
    """Import one same-device buffer and load complete parameters through vLLM."""

    version = _validate_update(worker, policy_version, transport="Packed IPC")
    payload = pickle.loads(base64.b64decode(payload_pickled.encode("ascii")))
    ipc_handles = payload["ipc_handles"]
    metadata = payload["metadata"]
    total_bytes = int(payload["total_bytes"])
    worker_topology = payload["worker_topology"]
    topology, physical_npu_id = _ipc_worker(worker, worker_topology)
    packed = _import_ipc_buffer(ipc_handles, physical_npu_id)
    try:
        if int(packed.numel()) != int(total_bytes):
            raise ValueError(
                "Packed IPC buffer differs from its metadata: "
                f"tensor={packed.numel()}, metadata={total_bytes}"
            )
        _load_packed_weights(worker, packed, metadata)
    finally:
        torch.npu.current_stream().synchronize()
        del packed
    worker._hyper_pending_policy_version = version
    return _receive_ack(
        topology,
        total_bytes,
        physical_device_id=physical_npu_id,
    )


def _clear_pending_update(worker: Any) -> None:
    """Reset the pending version before a new update starts."""
    worker._hyper_pending_policy_version = None


def _worker_architectures(worker: Any) -> frozenset[str]:
    """Return the worker's declared Hugging Face model architectures."""
    model_config = getattr(worker, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", ())
    return frozenset(architectures or ())


def _is_hyper_worker(worker: Any) -> bool:
    """Return whether the worker hosts a Hyper-registered model."""
    return HYPER_QWEN3_ARCHITECTURE in _worker_architectures(worker)


def _is_supported_worker(worker: Any) -> bool:
    """Return whether the worker hosts a supported Qwen3 rollout model."""
    return bool(_SUPPORTED_ARCHITECTURES.intersection(_worker_architectures(worker)))


def _finish_custom_weight_update(worker: Any) -> None:
    """Commit a custom worker transaction only after versioned weights arrived."""
    worker._check_weight_transfer_engine()  # pylint: disable=W0212
    if not worker._weight_update_active:  # pylint: disable=W0212
        raise RuntimeError("start_weight_update must be called before finish_weight_update")
    pending_version = getattr(worker, "_hyper_pending_policy_version", None)
    if pending_version is None:
        raise RuntimeError(
            "finish_weight_update requires received weights with a pending policy version"
        )
    worker._weight_update_active = False  # pylint: disable=W0212
    worker._is_checkpoint_format = True  # pylint: disable=W0212
    worker._hyper_loaded_policy_version = pending_version
    _clear_pending_update(worker)


def _patch_ascend_weight_update_lifecycle() -> None:
    """Bypass vLLM's layerwise wrapper for supported direct weight updates."""
    if _patch_state.ascend_lifecycle:
        return
    try:
        from vllm_ascend.worker.worker import NPUWorker  # pylint: disable=C0415
    except ImportError:
        return
    original_start = NPUWorker.start_weight_update
    original_finish = NPUWorker.finish_weight_update

    def start_weight_update(worker: Any, is_checkpoint_format: bool = True) -> None:
        """Start one model-owned direct weight-update transaction."""
        if not _is_supported_worker(worker):
            original_start(worker, is_checkpoint_format=is_checkpoint_format)
            worker._hyper_pending_policy_version = None
            return
        if not is_checkpoint_format:
            raise ValueError("Direct weight transfer requires checkpoint-format names")
        worker._check_weight_transfer_engine()  # pylint: disable=W0212
        if worker._weight_update_active:  # pylint: disable=W0212
            raise RuntimeError(
                "start_weight_update called while a weight update is already active"
            )
        worker._check_nz_disabled()  # pylint: disable=W0212
        _clear_pending_update(worker)
        worker._is_checkpoint_format = True  # pylint: disable=W0212
        worker._weight_update_active = True  # pylint: disable=W0212

    def finish_weight_update(worker: Any) -> None:
        """Commit the worker version only after the native receiver finishes."""
        if not _is_supported_worker(worker):
            original_finish(worker)
        else:
            _finish_custom_weight_update(worker)
            return
        pending_version = getattr(worker, "_hyper_pending_policy_version", None)
        if pending_version is not None:
            worker._hyper_loaded_policy_version = pending_version
        worker._hyper_pending_policy_version = None
    NPUWorker.start_weight_update = start_weight_update
    NPUWorker.finish_weight_update = finish_weight_update
    _patch_state.ascend_lifecycle = True


def _patch_engine_core_wake_lifecycle() -> None:
    """Wake executor memory while keeping the fixed vLLM scheduler paused."""
    if _patch_state.engine_core_wake:
        return
    from vllm.v1.engine.core import EngineCore  # pylint: disable=C0415
    original_wake_up = EngineCore.wake_up

    def wake_up(engine_core: Any, tags: Optional[list[str]] = None) -> Any:
        """Handle the Hyper sentinel before vLLM's unconditional scheduler resume."""
        if tags is None or KEEP_SCHEDULER_PAUSED_TAG not in tags:
            return original_wake_up(engine_core, tags)
        memory_tags = [tag for tag in tags if tag != KEEP_SCHEDULER_PAUSED_TAG]
        if memory_tags:
            engine_core.model_executor.wake_up(memory_tags)
        return None

    EngineCore.wake_up = wake_up
    _patch_state.engine_core_wake = True


def install_vllm_weight_sync_hooks(*, private_lifecycle: bool = True) -> None:
    """Install stable worker RPCs and optionally pinned private lifecycle patches."""
    from vllm.v1.worker.worker_base import WorkerBase  # pylint: disable=C0415
    if not hasattr(WorkerBase, "get_policy_version"):
        setattr(WorkerBase, "get_policy_version", get_policy_version)
    for name, method in (
        ("get_direct_reshard_layout", get_direct_reshard_layout),
        ("init_direct_reshard_group", init_direct_reshard_group),
        ("init_packed_weight_group", init_packed_weight_group),
        ("receive_direct_reshard", receive_direct_reshard),
        ("receive_ipc_direct_reshard", receive_ipc_direct_reshard),
        ("receive_ipc_packed_weights", receive_ipc_packed_weights),
        ("receive_packed_weights", receive_packed_weights),
    ):
        if not hasattr(WorkerBase, name):
            setattr(WorkerBase, name, method)
    if private_lifecycle:
        _patch_ascend_weight_update_lifecycle()
        _patch_engine_core_wake_lifecycle()
