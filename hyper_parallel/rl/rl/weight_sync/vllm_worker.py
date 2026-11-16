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
import asyncio
import base64
import gc
from importlib.metadata import version as package_version
import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

# Installed worker hooks share RL-owned version state on foreign vLLM worker instances.
# vLLM has no public policy-version API for these transaction fields.
# pylint: disable=protected-access
import torch
from safetensors import safe_open

from rl.workers.engine.model import (
    NATIVE_QWEN3_ARCHITECTURE,
    NATIVE_QWEN3_MOE_ARCHITECTURE,
    NATIVE_QWEN3_5_ARCHITECTURE,
)
from rl.weight_sync.model_adapter import rollout_tensor_descriptions
from rl.weight_sync.packed_weight import unpack_packed_weights
from rl.weight_sync.vllm_client import KEEP_SCHEDULER_PAUSED_TAG

_SUPPORTED_ARCHITECTURES = frozenset(
    (
            NATIVE_QWEN3_ARCHITECTURE,
        NATIVE_QWEN3_MOE_ARCHITECTURE,
        NATIVE_QWEN3_5_ARCHITECTURE,
    )
)


@dataclass
class _PatchState:
    """Track process-local idempotent vLLM patch installation."""

    ascend_lifecycle: bool = False
    engine_core_wake: bool = False
    dp_rpc: bool = False


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
    return {"version": int(getattr(worker, "_rl_loaded_policy_version", 0))}


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


def _native_weight_layout(worker: Any) -> dict[str, Any]:
    """Describe local Qwen3 storage and the worker's physical ownership."""
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    if not _is_supported_worker(worker):
        raise ValueError("Direct reshard requires a supported Qwen3 rollout model")
    if NATIVE_QWEN3_5_ARCHITECTURE in _worker_architectures(worker):
        raise ValueError("Qwen3.5/Qwen3.8 supports full_gather only")
    topology = _rollout_worker_topology(worker)
    model = worker.model_runner.get_model()
    return {
        **topology,
        "model_type": getattr(getattr(worker.model_config, "hf_config", None), "model_type", "qwen3"),
        "tensors": rollout_tensor_descriptions(
            model, getattr(worker.model_config, "hf_config", None),
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
        attribute="_rl_packed_weight_groups",
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
    loaded_version = int(getattr(worker, "_rl_loaded_policy_version", 0))
    pending_version = getattr(worker, "_rl_pending_policy_version", None)
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






def _load_packed_weights(
    worker: Any,
    packed: Any,
    metadata: list[Mapping[str, Any]],
) -> None:
    """Load one complete-parameter bucket through the model's vLLM API."""
    weights = unpack_packed_weights(packed, metadata)
    model = worker.model_runner.get_model()
    if NATIVE_QWEN3_MOE_ARCHITECTURE in _worker_architectures(worker):
        _load_moe_checkpoint_weights(worker, model, weights)
        return
    if NATIVE_QWEN3_5_ARCHITECTURE in _worker_architectures(worker):
        _load_qwen3_5_checkpoint_weights(worker, model, weights)
        return
    loaded = model.load_weights(weights)
    if weights and not loaded:
        raise RuntimeError(
            "vLLM load_weights did not accept any parameter from a packed bucket"
        )


def _qwen3_5_text_checkpoint_names(worker: Any) -> set[str]:
    """Read the checkpoint's text policy keys without loading weight tensors."""
    checkpoint = Path(worker.model_config.model)
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        names = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])
    else:
        with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as weights:
            names = set(weights.keys())
    text_names = {name for name in names if name.startswith("model.language_model.")}
    if not text_names:
        raise ValueError("Qwen3.5 checkpoint contains no text policy parameters")
    if not worker.model_config.hf_text_config.tie_word_embeddings:
        if "lm_head.weight" not in names:
            raise ValueError("Qwen3.5 checkpoint is missing lm_head.weight")
        text_names.add("lm_head.weight")
    return text_names


def _load_qwen3_5_checkpoint_weights(worker: Any, model: Any, weights: list[tuple[str, Any]]) -> None:
    """Publish only complete text policies through the native fused-weight loader."""
    if not getattr(worker, "_rl_expected_weights", set()):
        worker._rl_expected_weights = _qwen3_5_text_checkpoint_names(worker)
        worker._rl_received_weights = set()
    for name, weight in weights:
        if name not in worker._rl_expected_weights:
            raise ValueError(f"Unexpected Qwen3.5 text policy weight {name!r}")
        if name in worker._rl_received_weights:
            raise ValueError(f"Duplicate Qwen3.5 text policy weight {name!r}")
        # A native fused loader may retain an input until another projection arrives.
        # The IPC producer can release its bucket as soon as this RPC acknowledges.
        if not model.load_weights([(name, weight.clone())]):
            raise RuntimeError(f"Native Qwen3.5 loader did not accept weight {name!r}")
        worker._rl_received_weights.add(name)


def _load_moe_checkpoint_weights(worker: Any, model: Any, weights: list[tuple[str, Any]]) -> None:
    """Reload checkpoint tensors with Native layerwise processing and logical coverage."""
    # Native reload restores loader attributes lost during Ascend expert postprocessing.
    from vllm.model_executor.model_loader.reload import initialize_layerwise_reload  # pylint: disable=C0415

    if not getattr(worker, "_rl_layerwise_reload", False):
        descriptions = _native_weight_layout(worker)["tensors"]
        worker._rl_expected_weights = {tensor["name"] for tensor in descriptions}
        worker._rl_received_weights = set()
        with torch.device(worker.device):
            initialize_layerwise_reload(model)
        worker._rl_layerwise_reload = True
    expected = worker._rl_expected_weights
    received = worker._rl_received_weights
    expert_prefixes = {name.split(".experts.", maxsplit=1)[0] + ".experts."
                       for name in expected if ".experts." in name}
    for name, weight in weights:
        if name not in expected:
            match = re.fullmatch(r"(.+\.experts\.)(\d+)(\.(?:gate|up|down)_proj\.weight)", name)
            if (match and 0 <= int(match[2]) < int(worker.model_config.hf_config.num_experts)
                    and match[1] in expert_prefixes):
                continue
            raise ValueError(f"Unexpected MoE checkpoint weight {name!r}")
        if name in received:
            raise ValueError(f"Duplicate MoE checkpoint weight {name!r}")
        # Layerwise loaders may retain inputs until the other fused projections arrive.
        # Own their storage before acknowledging and releasing the IPC producer buffer.
        loaded = model.load_weights([(name, weight.clone())])
        if not loaded:
            raise RuntimeError(f"Native MoE loader did not accept local weight {name!r}")
        received.add(name)


def _finalize_moe_checkpoint_reload(worker: Any) -> None:
    """Require every local logical weight before finalizing Native reload once."""
    from vllm.model_executor.model_loader.reload import finalize_layerwise_reload  # pylint: disable=C0415

    missing = worker._rl_expected_weights - worker._rl_received_weights
    if missing:
        raise RuntimeError(f"MoE checkpoint update is incomplete: missing={sorted(missing)}")
    with torch.device(worker.device):
        finalize_layerwise_reload(worker.model_runner.get_model(), worker.model_config)
    torch.npu.current_stream().synchronize()


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
    groups = getattr(worker, "_rl_packed_weight_groups", {})
    group = groups.get(group_id)
    if group is None:
        raise RuntimeError(f"Packed HCCL group {group_id!r} is not initialized")

    packed = torch.empty(int(total_bytes), dtype=torch.uint8, device=group.device)
    group.broadcast(packed, src=0)
    torch.npu.current_stream().synchronize()
    _load_packed_weights(worker, packed, metadata)
    torch.npu.current_stream().synchronize()
    worker._rl_pending_policy_version = version
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
    worker._rl_pending_policy_version = version
    return _receive_ack(
        topology,
        total_bytes,
        physical_device_id=physical_npu_id,
    )


def _clear_pending_update(worker: Any) -> None:
    """Reset the pending version before a new update starts."""
    worker._rl_pending_policy_version = None
    worker._rl_layerwise_reload = False
    worker._rl_expected_weights = set()
    worker._rl_received_weights = set()


def _worker_architectures(worker: Any) -> frozenset[str]:
    """Return the worker's declared Hugging Face model architectures."""
    model_config = getattr(worker, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", ())
    return frozenset(architectures or ())




def _is_supported_worker(worker: Any) -> bool:
    """Return whether the worker hosts a supported Qwen3 rollout model."""
    return bool(_SUPPORTED_ARCHITECTURES.intersection(_worker_architectures(worker)))


def _finish_custom_weight_update(worker: Any) -> None:
    """Commit a custom worker transaction only after versioned weights arrived."""
    worker._check_weight_transfer_engine()  # pylint: disable=W0212
    if not worker._weight_update_active:  # pylint: disable=W0212
        raise RuntimeError("start_weight_update must be called before finish_weight_update")
    pending_version = getattr(worker, "_rl_pending_policy_version", None)
    if pending_version is None:
        raise RuntimeError(
            "finish_weight_update requires received weights with a pending policy version"
        )
    if getattr(worker, "_rl_layerwise_reload", False):
        _finalize_moe_checkpoint_reload(worker)
    if NATIVE_QWEN3_5_ARCHITECTURE in _worker_architectures(worker):
        expected = getattr(worker, "_rl_expected_weights", set())
        missing = expected - getattr(worker, "_rl_received_weights", set())
        if not expected or missing:
            raise RuntimeError(f"Qwen3.5 text policy update is incomplete: missing={sorted(missing)}")
    worker._weight_update_active = False  # pylint: disable=W0212
    worker._is_checkpoint_format = True  # pylint: disable=W0212
    worker._rl_loaded_policy_version = pending_version
    if NATIVE_QWEN3_5_ARCHITECTURE in _worker_architectures(worker):
        worker._rl_graph_refresh_pending = not getattr(worker.model_config, "enforce_eager", True)
    _clear_pending_update(worker)


def _retire_decode_graphs(wrappers: list[Any], params: Any) -> tuple[list, list]:
    """Retain old pools while clearing capture and attention metadata for recapture."""
    retired_entries = [list(wrapper.concrete_aclgraph_entries.values()) for wrapper in wrappers]
    retired_metadata = [list(field.values()) for field in (
        params.events, params.handles, params.attn_params, params.workspaces,
    )]
    for wrapper in wrappers:
        wrapper.concrete_aclgraph_entries.clear()
    for field in (params.events, params.handles, params.attn_params):
        for size in field:
            field[size] = []
    for size in params.workspaces:
        params.workspaces[size] = None
    return retired_entries, retired_metadata


def _refresh_ascend_decode_graphs(worker: Any) -> None:
    """Recapture the pinned text decode graphs with new weights and awake KV memory.

    The owned wake operation runs while admission is paused. Ascend 0.23 has no
    public graph invalidation API, so both its capture entries and attention-task
    metadata must be released together before using the runner's capture API.
    """
    if not getattr(worker, "_rl_graph_refresh_pending", False):
        return
    runtime = tuple(package_version(name).split("+", maxsplit=1)[0] for name in ("vllm", "vllm-ascend"))
    if runtime != ("0.23.0", "0.23.0.post1"):
        raise RuntimeError("Decode graph publication requires the pinned vLLM/Ascend 0.23.0.post1 runtime")
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled  # pylint: disable=C0415
    from vllm.config import CUDAGraphMode  # pylint: disable=C0415
    from vllm_ascend.compilation import acl_graph  # pylint: disable=C0415

    runner = worker.model_runner
    if runner.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        raise RuntimeError("Policy publication only supports FULL_DECODE_ONLY graphs")
    wrappers = [wrapper for wrapper in acl_graph._acl_graph_wrappers
                if wrapper.vllm_config is runner.vllm_config]
    descriptors = [set(wrapper.concrete_aclgraph_entries) for wrapper in wrappers]
    params = acl_graph.get_graph_params()
    if not wrappers or not any(descriptors) or params is None:
        raise RuntimeError("Decode graph publication requires initialized capture entries and attention metadata")
    torch.npu.synchronize()
    # Keep the shared pool alive until its replacement captures exist. Retiring
    # its final graph first invalidates the handle; minting pools grows residency.
    retired_entries, retired_metadata = _retire_decode_graphs(wrappers, params)
    gc.collect()
    try:
        runner.capture_model()
    finally:
        set_cudagraph_capturing_enabled(False)
    torch.npu.synchronize()
    for entries in retired_entries:
        for entry in entries:
            if entry.aclgraph is not None:
                entry.aclgraph.reset()
    retired_entries.clear()
    retired_metadata.clear()
    gc.collect()
    if any(set(wrapper.concrete_aclgraph_entries) != expected
           or any(entry.aclgraph is None for entry in wrapper.concrete_aclgraph_entries.values())
           for wrapper, expected in zip(wrappers, descriptors)):
        raise RuntimeError("Decode graph recapture did not restore every original capture descriptor")
    worker._rl_graph_refresh_pending = False
    logging.getLogger(__name__).info("Recaptured decode graphs for policy version %s (%s descriptors)",
                                     get_policy_version(worker)["version"], sum(map(len, descriptors)))


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
    original_wake = NPUWorker.wake_up

    def wake_up(worker: Any, tags: Optional[list[str]] = None) -> None:
        """Keep MoE kernel layout intact until the explicit checkpoint transaction."""
        if (NATIVE_QWEN3_MOE_ARCHITECTURE not in _worker_architectures(worker)
                or (tags is not None and "weights" not in tags)):
            original_wake(worker, tags)
            if tags is None or "kv_cache" in tags:
                _refresh_ascend_decode_graphs(worker)
            return
        model = worker.model_runner.get_model()
        experts = [
            (model.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1], parameter)
            for name, parameter in model.named_parameters()
            if name.endswith((".w13_weight", ".w2_weight"))
        ]
        try:
            original_wake(worker, tags)
        finally:
            # Native wake replaces experts with transposed views. Retain the original
            # kernel Parameters: DP pause consensus can execute dummy batches here.
            for module, name, parameter in experts:
                setattr(module, name, parameter)

    def start_weight_update(worker: Any, is_checkpoint_format: bool = True) -> None:
        """Start one model-owned direct weight-update transaction."""
        if not _is_supported_worker(worker):
            original_start(worker, is_checkpoint_format=is_checkpoint_format)
            worker._rl_pending_policy_version = None
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
        pending_version = getattr(worker, "_rl_pending_policy_version", None)
        if pending_version is not None:
            worker._rl_loaded_policy_version = pending_version
        worker._rl_pending_policy_version = None
    NPUWorker.wake_up = wake_up
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
        """Handle the RL sentinel before vLLM's unconditional scheduler resume."""
        if tags is None or KEEP_SCHEDULER_PAUSED_TAG not in tags:
            return original_wake_up(engine_core, tags)
        memory_tags = [tag for tag in tags if tag != KEEP_SCHEDULER_PAUSED_TAG]
        if memory_tags:
            engine_core.model_executor.wake_up(memory_tags)
        return None

    EngineCore.wake_up = wake_up
    _patch_state.engine_core_wake = True


def _patch_dp_weight_rpc() -> None:
    """Preserve every DP engine result for physical weight publication RPCs."""
    if _patch_state.dp_rpc:
        return
    from vllm.v1.engine.core_client import DPLBAsyncMPClient  # pylint: disable=C0415

    original = DPLBAsyncMPClient.collective_rpc_async
    weight_methods = frozenset((
        "get_policy_version",
        "receive_ipc_packed_weights",
    ))

    async def collective_rpc_async(
        client: Any, method: str, timeout: Optional[float] = None,
        args: tuple = (), kwargs: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        """Aggregate only weight-control results across the fixed engine group."""
        if method not in weight_methods:
            return await original(client, method, timeout, args, kwargs)
        results = await asyncio.gather(*(
            client._call_utility_async("collective_rpc", method, timeout, args, kwargs, engine=engine)
            for engine in client.core_engines
        ))
        return [worker for engine_results in results for worker in engine_results]

    DPLBAsyncMPClient.collective_rpc_async = collective_rpc_async
    _patch_state.dp_rpc = True


def install_vllm_weight_sync_hooks(*, private_lifecycle: bool = True) -> None:
    """Install stable worker RPCs and optionally pinned private lifecycle patches."""
    from vllm.v1.worker.worker_base import WorkerBase  # pylint: disable=C0415
    if not hasattr(WorkerBase, "get_policy_version"):
        setattr(WorkerBase, "get_policy_version", get_policy_version)
    for name, method in (
        ("init_packed_weight_group", init_packed_weight_group),
        ("receive_ipc_packed_weights", receive_ipc_packed_weights),
        ("receive_packed_weights", receive_packed_weights),
    ):
        if not hasattr(WorkerBase, name):
            setattr(WorkerBase, name, method)
    if private_lifecycle:
        _patch_ascend_weight_update_lifecycle()
        _patch_engine_core_wake_lifecycle()
        _patch_dp_weight_rpc()
