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
from dataclasses import dataclass
import base64
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any, Mapping, Optional
from rl.roles.model import (
    HYPER_QWEN3_ARCHITECTURE,
    NATIVE_QWEN3_ARCHITECTURE,
)
from rl.roles.weight_sync.sync import (
    KEEP_SCHEDULER_PAUSED_TAG,
    aggregate_policy_fingerprint,
    is_policy_fingerprint_weight,
    policy_tensor_fingerprint,
    verify_policy_fingerprints,
)
from hyper_parallel import get_platform

platform = get_platform()
_HYPER_ARCHITECTURES = frozenset((HYPER_QWEN3_ARCHITECTURE,))
_DIRECT_RESHARD_ARCHITECTURES = frozenset(
    (HYPER_QWEN3_ARCHITECTURE, NATIVE_QWEN3_ARCHITECTURE)
)
_POLICY_VERSION_FIELD = "_hyper_policy_version"


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


def _validate_direct_reshard_topology(
    topology: Mapping[str, Any],
    *,
    expected_data_parallel_size: int,
    expected_tensor_parallel_size: int,
) -> tuple[int, int]:
    """Validate one worker against the controller-owned rollout topology."""
    expected_dp_size = int(expected_data_parallel_size)
    expected_tp_size = int(expected_tensor_parallel_size)
    if expected_dp_size <= 0 or expected_tp_size <= 0:
        raise ValueError("Direct reshard expected DP and TP sizes must be positive")
    actual_dp_size = int(topology["dp_size"])
    actual_tp_size = int(topology["tp_size"])
    dp_rank = int(topology["dp_rank"])
    tp_rank = int(topology["tp_rank"])
    # Non-MoE vLLM engines may expose engine-local DP size 1 while retaining
    # the deployment-global data_parallel_index used below.
    if actual_dp_size not in (1, expected_dp_size) or actual_tp_size != expected_tp_size:
        raise ValueError(
            "Direct reshard worker topology differs from the configured DP x TP: "
            f"expected=({expected_dp_size}, {expected_tp_size}), "
            f"actual=({actual_dp_size}, {actual_tp_size})"
        )
    if not 0 <= dp_rank < expected_dp_size or not 0 <= tp_rank < expected_tp_size:
        raise ValueError(
            "Direct reshard worker rank is outside the configured DP x TP: "
            f"rank=({dp_rank}, {tp_rank}), size=({expected_dp_size}, {expected_tp_size})"
        )
    return dp_rank, tp_rank


def get_policy_version(worker: Any) -> dict[str, int]:
    """Return committed worker identity without touching sleeping model tensors."""
    return {"version": int(getattr(worker, "_hyper_loaded_policy_version", 0))}


def get_policy_weight_fingerprint(
    worker: Any,
    version: Optional[int] = None,
) -> dict[str, Any]:
    """Hash replicated language-model norms for post-transfer verification."""
    del version  # Retain the old RPC signature without trusting caller-owned identity.
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    tensor_digests = {}
    value_count = 0
    model = worker.model_runner.get_model()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not is_policy_fingerprint_weight(name):
            continue
        values = platform.tensor_type_cast(
            parameter.detach().to(device="cpu").contiguous(),
            "float32",
        )
        canonical_name, tensor_digest = policy_tensor_fingerprint(
            name,
            tuple(values.shape),
            platform.tensor_to_numpy(values).tobytes(),
        )
        if canonical_name in tensor_digests:
            raise RuntimeError(
                f"vLLM policy fingerprint has duplicate tensor {canonical_name!r}"
            )
        tensor_digests[canonical_name] = tensor_digest
        value_count += int(values.numel())
    if not tensor_digests:
        raise RuntimeError("vLLM policy fingerprint found no language-model norm tensors")
    hf_config = getattr(worker.model_config, "hf_config", None)
    architectures = tuple(getattr(hf_config, "architectures", ()) or ())
    try:
        rank = platform.get_rank()
    except (RuntimeError, ValueError):
        rank = 0
    fingerprint = aggregate_policy_fingerprint(tensor_digests, value_count)
    fingerprint.update(
        {
            "version": int(getattr(worker, "_hyper_loaded_policy_version", 0)),
            "rank": rank,
            "architecture": architectures[0] if architectures else None,
        }
    )
    return fingerprint


def verify_policy_weight_identity(
    worker: Any,
    expected_version: int,
    expected_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail locally unless this worker owns the expected policy identity."""
    actual = get_policy_weight_fingerprint(worker)
    verify_policy_fingerprints(
        expected_fingerprint,
        [actual],
        expected_version=int(expected_version),
    )
    return {
        "version": actual["version"],
        "digest": actual["digest"],
    }


def get_all_parameter_manifest(worker: Any) -> dict[str, Any]:
    """Return exact byte hashes for every rank-local rollout parameter.

    This diagnostic is intentionally separate from the lightweight publication
    fingerprint: it copies one parameter at a time to CPU and is therefore only
    used by explicit direct-reshard verification runs.
    """
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    import torch  # pylint: disable=C0415,forbidden-backend-import
    model = worker.model_runner.get_model()
    tensors: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_parameters = model.named_parameters()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        raw = (
            parameter.detach().contiguous().view(-1).view(torch.uint8).to(device="cpu")
        )
        payload = platform.tensor_to_numpy(raw).tobytes()
        num_bytes = len(payload)
        tensors[name] = {
            "dtype": str(parameter.dtype).rsplit(".", maxsplit=1)[-1],
            "shape": [int(size) for size in parameter.shape],
            "num_bytes": num_bytes,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        total_bytes += num_bytes
        del raw, payload
    manifest_payload = json.dumps(
        tensors,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result = {
        "parameter_count": len(tensors),
        "total_bytes": total_bytes,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "tensors": tensors,
    }
    result.update(_rollout_worker_topology(worker))
    return result


def write_parameter_manifest(
    worker: Any,
    *,
    output_dir: str,
    strategy: str,
    policy_version: int,
    rollout_replica_rank: int,
    expected_data_parallel_size: int,
    oracle_run_id: str,
    oracle_dir: Optional[str] = None,
    oracle_strategy: Optional[str] = None,
    expected_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Persist and optionally compare one worker's exact parameter manifest."""
    if not output_dir:
        raise ValueError("Parameter manifest output_dir must be non-empty")
    if not strategy:
        raise ValueError("Parameter manifest strategy must be non-empty")
    if not oracle_run_id:
        raise ValueError("Parameter manifest oracle_run_id must be non-empty")
    version = int(policy_version)
    expected_dp_size = int(expected_data_parallel_size)
    if expected_dp_size <= 0:
        raise ValueError("Parameter manifest expected_data_parallel_size must be positive")
    loaded_version = int(getattr(worker, "_hyper_loaded_policy_version", 0))
    if loaded_version != version:
        raise RuntimeError(
            "Parameter manifest policy version is not committed: "
            f"loaded={loaded_version}, expected={version}"
        )
    manifest = get_all_parameter_manifest(worker)
    if not 0 <= int(manifest["dp_rank"]) < expected_dp_size:
        raise RuntimeError(
            "Parameter manifest DP rank is outside the publication topology: "
            f"rank={manifest['dp_rank']}, size={expected_dp_size}"
        )
    manifest.update(
        {
            "dp_size": expected_dp_size,
            "strategy": strategy,
            "oracle_run_id": oracle_run_id,
            "policy_version": version,
            "rollout_replica_rank": int(rollout_replica_rank),
        }
    )
    filename = (
        f"{strategy}-version{version}-replica{int(rollout_replica_rank)}-"
        f"dp{manifest['dp_rank']}-tp{manifest['tp_rank']}.json"
    )
    expected = None
    if expected_dir:
        expected_path = (
            Path(expected_dir)
            / f"version{version}-dp{manifest['dp_rank']}-tp{manifest['tp_rank']}.json"
        )
        if not expected_path.is_file():
            raise RuntimeError(
                f"Trainer-derived expected parameter manifest is missing: {expected_path}"
            )
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        expected_identity = {
            "oracle_run_id": oracle_run_id,
            "policy_version": version,
            "dp_rank": manifest["dp_rank"],
            "tp_rank": manifest["tp_rank"],
            "dp_size": expected_dp_size,
            "tp_size": manifest["tp_size"],
            "physical_device_id": manifest["physical_device_id"],
        }
        identity_mismatches = {
            key: (value, expected.get(key))
            for key, value in expected_identity.items()
            if expected.get(key) != value
        }
        if identity_mismatches:
            raise RuntimeError(
                "Trainer-derived expected parameter identity differs from rollout: "
                f"{identity_mismatches}"
            )
        expected_tensors = expected.get("tensors", {})
        actual_tensors = manifest["tensors"]
        if expected_tensors != actual_tensors:
            changed_tensors = sorted(
                name
                for name in set(expected_tensors) & set(actual_tensors)
                if expected_tensors[name] != actual_tensors[name]
            )
            raise RuntimeError(
                "Rollout parameter manifest differs from Trainer-derived expectation: "
                f"missing={sorted(set(expected_tensors) - set(actual_tensors))}, "
                f"unexpected={sorted(set(actual_tensors) - set(expected_tensors))}, "
                f"changed={changed_tensors}"
            )
        manifest["source_manifest_sha256"] = expected.get("source_manifest_sha256")
        manifest["expected_manifest_sha256"] = expected.get("manifest_sha256")
        manifest["source_match"] = True
    if oracle_dir:
        if not oracle_strategy:
            raise ValueError("Parameter manifest oracle_strategy must be non-empty")
        oracle_filename = (
            f"{oracle_strategy}-version{version}-replica{int(rollout_replica_rank)}-"
            f"dp{manifest['dp_rank']}-tp{manifest['tp_rank']}.json"
        )
        oracle_path = Path(oracle_dir) / oracle_filename
        if not oracle_path.is_file():
            raise RuntimeError(f"Parameter manifest oracle is missing: {oracle_path}")
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        if oracle.get("oracle_run_id") != oracle_run_id:
            raise RuntimeError(
                "Parameter manifest oracle run mismatch: "
                f"expected={oracle_run_id!r}, actual={oracle.get('oracle_run_id')!r}"
            )
        oracle_source = oracle.get("source_manifest_sha256")
        expected_source = None if expected is None else expected.get("source_manifest_sha256")
        comparable = expected_source is None or oracle_source in (None, expected_source)
        manifest["oracle_comparable"] = comparable
        manifest["oracle_source_manifest_sha256"] = oracle_source
        if comparable and oracle.get("tensors") != manifest["tensors"]:
            expected_tensors = oracle.get("tensors", {})
            actual_tensors = manifest["tensors"]
            changed_tensors = sorted(
                name
                for name in set(expected_tensors) & set(actual_tensors)
                if expected_tensors[name] != actual_tensors[name]
            )
            raise RuntimeError(
                "Rollout parameter manifest differs from full-gather oracle: "
                f"missing={sorted(set(expected_tensors) - set(actual_tensors))}, "
                f"unexpected={sorted(set(actual_tensors) - set(expected_tensors))}, "
                f"changed={changed_tensors}"
            )
        if comparable:
            manifest["oracle_manifest_sha256"] = oracle.get("manifest_sha256")
            manifest["oracle_match"] = True
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    temporary = directory / f".{filename}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return {
        "written": True,
        "dp_rank": manifest["dp_rank"],
        "tp_rank": manifest["tp_rank"],
        "manifest_sha256": manifest["manifest_sha256"],
        "oracle_match": manifest.get("oracle_match"),
        "oracle_comparable": manifest.get("oracle_comparable"),
        "source_match": manifest.get("source_match"),
    }


def _direct_tensor_description(
    source_name: str,
    destination_name: str,
    parameter: Any,
    local_shape: tuple[int, ...],
    placement: str,
    shard_dim: Optional[int],
    destination_starts: tuple[int, ...],
) -> dict[str, Any]:
    """Describe one logical Actor tensor region inside a physical parameter."""
    if len(local_shape) != len(parameter.shape):
        raise ValueError(
            f"Native direct tensor {source_name!r} rank mismatch: "
            f"logical={local_shape}, destination={tuple(parameter.shape)}"
        )
    if len(destination_starts) != len(parameter.shape):
        raise ValueError(
            f"Native direct tensor {source_name!r} offset rank mismatch: "
            f"offset={destination_starts}, destination={tuple(parameter.shape)}"
        )
    if any(
        start < 0 or start + length > int(limit)
        for start, length, limit in zip(
            destination_starts,
            local_shape,
            parameter.shape,
        )
    ):
        raise ValueError(
            f"Native direct tensor {source_name!r} exceeds {destination_name!r}: "
            f"offset={destination_starts}, logical={local_shape}, "
            f"destination={tuple(parameter.shape)}"
        )
    return {
        "name": source_name,
        "destination_name": destination_name,
        "dtype_name": str(parameter.dtype).rsplit(".", maxsplit=1)[-1],
        "element_size": int(parameter.element_size()),
        "local_shape": list(local_shape),
        "placement": placement,
        "shard_dim": shard_dim,
        "destination_starts": list(destination_starts),
    }


def _native_qwen3_qkv_descriptions(
    name: str,
    parameter: Any,
    hf_config: Any,
    tp_size: int,
) -> list[dict[str, Any]]:
    """Map native fused QKV storage to the three canonical Actor tensors."""
    num_heads = int(hf_config.num_attention_heads)
    num_kv_heads = int(hf_config.num_key_value_heads)
    hidden_size = int(hf_config.hidden_size)
    head_dim = int(getattr(hf_config, "head_dim", hidden_size // num_heads))
    if num_heads % tp_size != 0:
        raise ValueError(
            f"Native Qwen3 query heads {num_heads} are not divisible by TP {tp_size}"
        )
    q_size = num_heads * head_dim
    q_local_size = q_size // tp_size
    kv_size = num_kv_heads * head_dim
    if num_kv_heads < tp_size:
        if tp_size % num_kv_heads != 0:
            raise ValueError(
                f"Native Qwen3 TP {tp_size} cannot replicate {num_kv_heads} KV heads"
            )
        raise ValueError(
            "Native Qwen3 direct reshard does not support grouped KV-head replication: "
            f"kv_heads={num_kv_heads}, tp_size={tp_size}"
        )
    if num_kv_heads % tp_size != 0:
        raise ValueError(
            f"Native Qwen3 KV heads {num_kv_heads} are not divisible by TP {tp_size}"
        )
    kv_local_size = kv_size // tp_size
    kv_placement = "shard"
    kv_shard_dim = 0
    tail_shape = tuple(int(size) for size in parameter.shape[1:])
    expected_shape = (q_local_size + 2 * kv_local_size,) + tail_shape
    if tuple(int(size) for size in parameter.shape) != expected_shape:
        raise ValueError(
            f"Native Qwen3 fused QKV parameter {name!r} has shape "
            f"{tuple(parameter.shape)}, expected {expected_shape}"
        )
    source_suffixes = ("q_proj", "k_proj", "v_proj")
    local_sizes = (q_local_size, kv_local_size, kv_local_size)
    placements = ("shard", kv_placement, kv_placement)
    shard_dims = (0, kv_shard_dim, kv_shard_dim)
    descriptions = []
    destination_offset = 0
    for source_suffix, local_size, placement, shard_dim in zip(
        source_suffixes,
        local_sizes,
        placements,
        shard_dims,
    ):
        source_name = name.replace("qkv_proj", source_suffix)
        local_shape = (local_size,) + tail_shape
        destination_starts = (destination_offset,) + (0,) * len(tail_shape)
        descriptions.append(
            _direct_tensor_description(
                source_name,
                name,
                parameter,
                local_shape,
                placement,
                shard_dim,
                destination_starts,
            )
        )
        destination_offset += local_size
    return descriptions


def _native_qwen3_gate_up_descriptions(
    name: str,
    parameter: Any,
    hf_config: Any,
    tp_size: int,
) -> list[dict[str, Any]]:
    """Map native fused gate/up storage to canonical Actor MLP tensors."""
    intermediate_size = int(hf_config.intermediate_size)
    if intermediate_size % tp_size != 0:
        raise ValueError(
            f"Native Qwen3 intermediate size {intermediate_size} is not divisible by TP {tp_size}"
        )
    local_size = intermediate_size // tp_size
    tail_shape = tuple(int(size) for size in parameter.shape[1:])
    expected_shape = (2 * local_size,) + tail_shape
    if tuple(int(size) for size in parameter.shape) != expected_shape:
        raise ValueError(
            f"Native Qwen3 fused gate/up parameter {name!r} has shape "
            f"{tuple(parameter.shape)}, expected {expected_shape}"
        )
    descriptions = []
    for source_suffix, destination_offset in (
        ("gate_proj", 0),
        ("up_proj", local_size),
    ):
        descriptions.append(
            _direct_tensor_description(
                name.replace("gate_up_proj", source_suffix),
                name,
                parameter,
                (local_size,) + tail_shape,
                "shard",
                0,
                (destination_offset,) + (0,) * len(tail_shape),
            )
        )
    return descriptions


def _native_qwen3_direct_tensors(
    model: Any,
    hf_config: Any,
    tp_rank: int,
    tp_size: int,
) -> list[dict[str, Any]]:
    """Describe native vLLM Qwen3 storage in canonical Actor coordinates."""
    tensors = []
    vocab_size = int(hf_config.vocab_size)
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if ".qkv_proj." in name:
            tensors.extend(
                _native_qwen3_qkv_descriptions(
                    name,
                    parameter,
                    hf_config,
                    tp_size,
                )
            )
            continue
        if ".gate_up_proj." in name:
            tensors.extend(
                _native_qwen3_gate_up_descriptions(
                    name,
                    parameter,
                    hf_config,
                    tp_size,
                )
            )
            continue
        parameter_shape = tuple(int(size) for size in parameter.shape)
        destination_starts = (0,) * len(parameter_shape)
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            partition_size = parameter_shape[0]
            source_start = tp_rank * partition_size
            local_size = max(0, min(partition_size, vocab_size - source_start))
            if local_size <= 0:
                raise ValueError(
                    f"Native Qwen3 vocabulary shard {tp_rank} contains no Actor rows"
                )
            local_shape = (local_size,) + parameter_shape[1:]
            placement = "shard"
            shard_dim = 0
        elif name.endswith((".self_attn.o_proj.weight", ".mlp.down_proj.weight")):
            local_shape = parameter_shape
            placement = "shard"
            shard_dim = 1
        else:
            local_shape = parameter_shape
            placement = "replicate"
            shard_dim = None
        tensors.append(
            _direct_tensor_description(
                name,
                name,
                parameter,
                local_shape,
                placement,
                shard_dim,
                destination_starts,
            )
        )
    return tensors


def get_direct_reshard_layout(worker: Any) -> dict[str, Any]:
    """Describe one Hyper or native Qwen3 worker's local TP parameters."""
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    model = worker.model_runner.get_model()
    from vllm.distributed import get_tp_group  # pylint: disable=C0415

    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    if _is_native_qwen3_worker(worker):
        hf_config = getattr(worker.model_config, "hf_config", None)
        if hf_config is None:
            raise ValueError("Native Qwen3 direct reshard requires an HF config")
        result = {
            "tensors": _native_qwen3_direct_tensors(
                model,
                hf_config,
                tp_rank,
                tp_size,
            ),
        }
        result.update(_rollout_worker_topology(worker))
        return result
    if not _is_hyper_worker(worker) or not hasattr(model, "_tp_placements"):
        raise ValueError(
            "Direct reshard requires a Hyper or native Qwen3 rollout model"
        )
    placements_by_name = getattr(model, "_tp_placements")
    tensors = []
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        placements = tuple(placements_by_name.get(name, ()))
        if tp_size == 1 and not placements:
            placement_name = "replicate"
            shard_dim = None
        else:
            if len(placements) != 1:
                raise ValueError(
                    f"Direct reshard parameter {name!r} requires one TP placement, "
                    f"got {placements}"
                )
            placement = placements[0]
            if callable(getattr(placement, "is_shard", None)) and placement.is_shard():
                placement_name = "shard"
                shard_dim = int(placement.dim)
            elif callable(getattr(placement, "is_replicate", None)) and placement.is_replicate():
                placement_name = "replicate"
                shard_dim = None
            else:
                raise ValueError(
                    f"Direct reshard parameter {name!r} has unsupported placement {placement!r}"
                )
        tensors.append(
            {
                "name": name,
                "dtype_name": str(parameter.dtype).rsplit(".", maxsplit=1)[-1],
                "element_size": int(parameter.element_size()),
                "local_shape": list(parameter.shape),
                "placement": placement_name,
                "shard_dim": shard_dim,
            }
        )
    result = {"tensors": tensors}
    result.update(_rollout_worker_topology(worker))
    return result


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
    topology = _rollout_worker_topology(worker)
    dp_rank, tp_rank = _validate_direct_reshard_topology(
        topology,
        expected_data_parallel_size=expected_data_parallel_size,
        expected_tensor_parallel_size=expected_tensor_parallel_size,
    )
    target_tp_rank = int(target_tp_rank)
    if not 0 <= target_tp_rank < int(expected_tensor_parallel_size):
        raise ValueError(
            "Direct reshard target TP rank is outside the configured topology: "
            f"rank={target_tp_rank}, size={expected_tensor_parallel_size}"
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
    groups = getattr(worker, "_hyper_direct_reshard_groups", None)
    if groups is None:
        groups = {}
        worker._hyper_direct_reshard_groups = groups
    if group_id not in groups:
        from vllm_ascend.distributed.weight_transfer.hccl_engine import (  # pylint: disable=C0415
            HCCLWeightTransferEngine,
        )

        device = int(
            platform.get_device_handle(platform.device_type()).current_device()
        )
        groups[group_id] = HCCLWeightTransferEngine._stateless_init_process_group(  # pylint: disable=W0212
            master_address,
            int(master_port),
            int(receiver_rank),
            int(world_size),
            device=device,
        )
    return {
        "joined": True,
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "group_rank": receiver_rank,
        "group_id": group_id,
    }


def init_full_gather_group(
    worker: Any,
    *,
    master_address: str,
    master_port: int,
    world_size: int,
    expected_data_parallel_size: int,
    expected_tensor_parallel_size: int,
) -> dict[str, Any]:
    """Create one full-gather group with a unique rank for every DP x TP worker."""
    topology = _rollout_worker_topology(worker)
    dp_rank, tp_rank = _validate_direct_reshard_topology(
        topology,
        expected_data_parallel_size=expected_data_parallel_size,
        expected_tensor_parallel_size=expected_tensor_parallel_size,
    )
    expected_world_size = 1 + int(expected_data_parallel_size) * int(expected_tensor_parallel_size)
    if int(world_size) != expected_world_size:
        raise ValueError(
            "Full-gather HCCL group world size differs from configured rollout DP x TP: "
            f"expected={expected_world_size}, actual={world_size}"
        )
    worker._check_weight_transfer_engine()  # pylint: disable=W0212
    transfer_engine = worker.weight_transfer_engine
    group_rank = 1 + dp_rank * int(expected_tensor_parallel_size) + tp_rank
    device = int(
        platform.get_device_handle(platform.device_type()).current_device()
    )
    transfer_engine.model_update_group = transfer_engine._stateless_init_process_group(  # pylint: disable=W0212
        master_address,
        int(master_port),
        group_rank,
        int(world_size),
        device=device,
    )
    return {
        "joined": True,
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "group_rank": group_rank,
    }


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
    topology = _rollout_worker_topology(worker)
    dp_rank, tp_rank = _validate_direct_reshard_topology(
        topology,
        expected_data_parallel_size=expected_data_parallel_size,
        expected_tensor_parallel_size=expected_tensor_parallel_size,
    )
    target_tp_rank = int(target_tp_rank)
    if not 0 <= target_tp_rank < int(expected_tensor_parallel_size):
        raise ValueError(
            "Direct reshard target TP rank is outside the configured topology: "
            f"rank={target_tp_rank}, size={expected_tensor_parallel_size}"
        )
    if tp_rank != target_tp_rank:
        return {
            "received": False,
            "dp_rank": dp_rank,
            "tp_rank": tp_rank,
            "bytes": 0,
        }
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    if not _is_direct_reshard_worker(worker):
        raise ValueError("Direct reshard requires a Hyper or native Qwen3 worker")
    groups = getattr(worker, "_hyper_direct_reshard_groups", {})
    group = groups.get(group_id)
    if group is None:
        raise RuntimeError(f"Direct reshard HCCL group {group_id!r} is not initialized")
    if not bool(getattr(worker, "_weight_update_active", False)):
        raise RuntimeError("Direct reshard requires an active vLLM weight update")
    version = int(policy_version)
    loaded_version = int(getattr(worker, "_hyper_loaded_policy_version", 0))
    pending_version = getattr(worker, "_hyper_pending_policy_version", None)
    if version <= loaded_version:
        raise ValueError(
            "Direct reshard policy version must increase: "
            f"loaded={loaded_version}, received={version}"
        )
    if pending_version is not None and int(pending_version) != version:
        raise ValueError(
            "One direct reshard update cannot mix policy versions: "
            f"pending={pending_version}, received={version}"
        )
    import torch  # pylint: disable=C0415,forbidden-backend-import

    parameters = dict(worker.model_runner.get_model().named_parameters())
    received_bytes = 0
    for bucket in buckets:
        total_bytes = int(bucket["total_bytes"])
        packed = torch.empty(total_bytes, dtype=torch.uint8, device=group.device)
        group.broadcast(packed, src=0)
        torch.npu.current_stream().synchronize()
        for entry in bucket["entries"]:
            name = str(entry["name"])
            parameter = parameters.get(name)
            if parameter is None:
                raise ValueError(f"Direct reshard rollout parameter {name!r} is missing")
            dtype = getattr(torch, str(entry["dtype_name"]))
            element_size = int(entry["element_size"])
            if int(parameter.element_size()) != element_size or parameter.dtype != dtype:
                raise ValueError(
                    f"Direct reshard rollout parameter {name!r} dtype mismatch: "
                    f"parameter={parameter.dtype}, transfer={dtype}"
                )
            lengths = tuple(int(value) for value in entry["lengths"])
            starts = tuple(int(value) for value in entry["destination_starts"])
            num_bytes = int(entry["num_bytes"])
            offset = int(entry["buffer_offset"])
            fragment = packed.narrow(0, offset, num_bytes).view(dtype).view(lengths)
            destination_slice = tuple(
                slice(start, start + length) for start, length in zip(starts, lengths)
            )
            target = parameter[destination_slice]
            if tuple(target.shape) != lengths:
                raise ValueError(
                    f"Direct reshard destination slice for {name!r} has shape "
                    f"{tuple(target.shape)}, expected {lengths}"
                )
            with torch.no_grad():
                target.copy_(fragment)
            received_bytes += num_bytes
        del packed
    worker._hyper_pending_policy_version = version
    return {
        "received": True,
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "bytes": received_bytes,
        "bucket_count": len(buckets),
    }


def receive_ipc_direct_reshard(
    worker: Any,
    *,
    payload_pickled: str,
    policy_version: int,
) -> dict[str, Any]:
    """Import same-NPU packed buffers and scatter them into TP-local weights."""
    # Torch and vLLM-Ascend are optional outside the Torch-NPU RL runtime.
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (  # pylint: disable=C0415
        npu_generate_uuid,
    )
    from torch_npu.multiprocessing.reductions import rebuild_npu_tensor  # pylint: disable=C0415
    import torch  # pylint: disable=C0415,forbidden-backend-import

    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    if not _is_direct_reshard_worker(worker):
        raise ValueError("IPC direct reshard requires a Hyper or native Qwen3 worker")
    if not bool(getattr(worker, "_weight_update_active", False)):
        raise RuntimeError("IPC direct reshard requires an active vLLM weight update")

    version = int(policy_version)
    loaded_version = int(getattr(worker, "_hyper_loaded_policy_version", 0))
    pending_version = getattr(worker, "_hyper_pending_policy_version", None)
    if version <= loaded_version:
        raise ValueError(
            "IPC direct reshard policy version must increase: "
            f"loaded={loaded_version}, received={version}"
        )
    if pending_version is not None and int(pending_version) != version:
        raise ValueError(
            "One IPC direct reshard update cannot mix policy versions: "
            f"pending={pending_version}, received={version}"
        )

    payload = pickle.loads(base64.b64decode(payload_pickled.encode("ascii")))
    topology = _rollout_worker_topology(worker)
    tp_rank = int(topology["tp_rank"])
    expected_workers = {
        worker_description["physical_device_id"]: worker_description
        for worker_description in payload["worker_topology"]
    }
    buckets = payload["buckets_by_tp"].get(tp_rank, ())
    device_index = torch.accelerator.current_device_index()
    physical_npu_id = npu_generate_uuid()
    expected_worker = expected_workers.get(physical_npu_id)
    if expected_worker is None:
        raise ValueError(
            f"IPC direct worker {physical_npu_id} is absent from the publication topology"
        )
    actual_identity = (int(topology["dp_rank"]), tp_rank)
    expected_identity = (
        int(expected_worker["dp_rank"]),
        int(expected_worker["tp_rank"]),
    )
    if actual_identity != expected_identity:
        raise ValueError(
            "IPC direct physical worker topology mismatch: "
            f"physical_device_id={physical_npu_id}, expected={expected_identity}, "
            f"actual={actual_identity}"
        )
    parameters = dict(worker.model_runner.get_model().named_parameters())
    received_bytes = 0
    imported_buffers = []

    try:
        for bucket in buckets:
            handles = bucket["ipc_handles"]
            if physical_npu_id not in handles:
                raise ValueError(
                    f"IPC direct reshard handle not found for {physical_npu_id}; "
                    f"available={list(handles)}"
                )
            rebuild_args = list(handles[physical_npu_id])
            rebuild_args[6] = device_index
            packed = rebuild_npu_tensor(*rebuild_args)
            imported_buffers.append(packed)
            metadata = bucket["metadata"]
            if int(packed.numel()) != int(metadata["total_bytes"]):
                raise ValueError(
                    "IPC direct reshard packed-buffer size mismatch: "
                    f"tensor={packed.numel()}, metadata={metadata['total_bytes']}"
                )
            for entry in metadata["entries"]:
                name = str(entry["name"])
                parameter = parameters.get(name)
                if parameter is None:
                    raise ValueError(f"IPC direct reshard parameter {name!r} is missing")
                dtype = getattr(torch, str(entry["dtype_name"]))
                element_size = int(entry["element_size"])
                if int(parameter.element_size()) != element_size or parameter.dtype != dtype:
                    raise ValueError(
                        f"IPC direct reshard parameter {name!r} dtype mismatch: "
                        f"parameter={parameter.dtype}, transfer={dtype}"
                    )
                lengths = tuple(int(value) for value in entry["lengths"])
                starts = tuple(int(value) for value in entry["destination_starts"])
                num_bytes = int(entry["num_bytes"])
                offset = int(entry["buffer_offset"])
                fragment = packed.narrow(0, offset, num_bytes).view(dtype).view(lengths)
                destination_slice = tuple(
                    slice(start, start + length) for start, length in zip(starts, lengths)
                )
                target = parameter[destination_slice]
                if tuple(target.shape) != lengths:
                    raise ValueError(
                        f"IPC direct reshard destination {name!r} has shape "
                        f"{tuple(target.shape)}, expected {lengths}"
                    )
                with torch.no_grad():
                    target.copy_(fragment)
                received_bytes += num_bytes
    finally:
        if imported_buffers:
            torch.npu.current_stream().synchronize()
            imported_buffers.clear()

    worker._hyper_pending_policy_version = version
    return {
        "received": True,
        "dp_rank": int(topology["dp_rank"]),
        "tp_rank": tp_rank,
        "physical_device_id": physical_npu_id,
        "bytes": received_bytes,
        "bucket_count": len(buckets),
    }


def reload_weights(
    worker: Any,
    weights_iterator: Any = None,
    weights_path: Optional[str] = None,
    is_checkpoint_format: bool = True,
    policy_version: Optional[int] = None,
) -> None:
    """Reload a Hyper checkpoint without vLLM's layerwise wrapper."""
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    if not is_checkpoint_format:
        raise ValueError("Hyper vLLM refit requires checkpoint-format weights")
    consistency_profile = os.environ.get("HYPER_RL_CONSISTENCY_PROFILE")
    if policy_version is None and consistency_profile not in (None, "", "off"):
        raise ValueError("Consistency-profile CPU reload requires a worker policy version")
    normalized_version = None
    if policy_version is not None:
        normalized_version = int(policy_version)
        loaded_version = int(getattr(worker, "_hyper_loaded_policy_version", 0))
        if normalized_version <= loaded_version:
            raise ValueError(
                "vLLM worker policy version must increase: "
                f"loaded={loaded_version}, received={normalized_version}"
            )
        pending_version = getattr(worker, "_hyper_pending_policy_version", None)
        if pending_version is not None:
            raise RuntimeError(
                "vLLM worker has an uncommitted CPU reload: "
                f"pending={pending_version}, received={normalized_version}"
            )
    model_runner = worker.model_runner
    model = model_runner.get_model()
    if weights_iterator is not None:
        model.load_weights(weights_iterator)
        if normalized_version is not None:
            worker._hyper_pending_policy_version = normalized_version
        return
    if weights_path is None:
        raise ValueError("Hyper vLLM refit requires weights_iterator or weights_path")
    from vllm.model_executor.model_loader import get_model_loader  # pylint: disable=C0415
    original_model_path = model_runner.model_config.model
    try:
        model_runner.model_config.model = weights_path
        model_loader = get_model_loader(model_runner.load_config)
        model.load_weights(model_loader.get_all_weights(model_runner.model_config, model))
        if normalized_version is not None:
            worker._hyper_pending_policy_version = normalized_version
    finally:
        model_runner.model_config.model = original_model_path


def commit_reloaded_weights(worker: Any, policy_version: int) -> None:
    """Commit worker identity after every CPU reload RPC has completed."""
    normalized_version = int(policy_version)
    pending_version = getattr(worker, "_hyper_pending_policy_version", None)
    if pending_version != normalized_version:
        raise RuntimeError(
            "vLLM CPU reload version does not match its pending weights: "
            f"pending={pending_version}, received={normalized_version}"
        )
    worker._hyper_loaded_policy_version = normalized_version
    worker._hyper_pending_policy_version = None


def abort_weight_update(worker: Any, restore_policy_version: int) -> dict[str, Any]:
    """Clear a failed update transaction before a full-checkpoint retry."""
    was_active = bool(getattr(worker, "_weight_update_active", False))
    pending_version = getattr(worker, "_hyper_pending_policy_version", None)
    worker._weight_update_active = False
    worker._is_checkpoint_format = True
    worker._hyper_pending_policy_version = None
    worker._hyper_loaded_policy_version = int(restore_policy_version)
    return {
        "aborted": True,
        "was_active": was_active,
        "pending_version": pending_version,
        "restored_version": int(restore_policy_version),
    }


def _worker_architectures(worker: Any) -> frozenset[str]:
    """Return the worker's declared Hugging Face model architectures."""
    hf_config = getattr(worker.model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", ())
    return frozenset(architectures or ())


def _is_hyper_worker(worker: Any) -> bool:
    """Return whether the worker hosts a Hyper-registered model."""
    return bool(_HYPER_ARCHITECTURES.intersection(_worker_architectures(worker)))


def _is_native_qwen3_worker(worker: Any) -> bool:
    """Return whether the worker hosts native vLLM Qwen3."""
    return NATIVE_QWEN3_ARCHITECTURE in _worker_architectures(worker)


def _is_direct_reshard_worker(worker: Any) -> bool:
    """Return whether the worker supports direct-reshard weight updates."""
    return bool(
        _DIRECT_RESHARD_ARCHITECTURES.intersection(_worker_architectures(worker))
    )


def _uses_custom_weight_update_lifecycle(worker: Any) -> bool:
    """Return whether Hyper owns this worker's update transaction lifecycle."""
    return _is_hyper_worker(worker) or _is_native_qwen3_worker(worker)


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
    worker._hyper_pending_policy_version = None


def _patch_ascend_weight_update_lifecycle() -> None:
    """Bypass vLLM's layerwise wrapper for direct Qwen3 weight updates."""
    if _patch_state.ascend_lifecycle:
        return
    try:
        from vllm_ascend.worker.worker import NPUWorker  # pylint: disable=C0415
    except ImportError:
        return
    original_start = NPUWorker.start_weight_update
    original_update = NPUWorker.update_weights
    original_finish = NPUWorker.finish_weight_update

    def start_weight_update(worker: Any, is_checkpoint_format: bool = True) -> None:
        """Start one direct Qwen3 weight-update transaction."""
        if not _uses_custom_weight_update_lifecycle(worker):
            original_start(worker, is_checkpoint_format=is_checkpoint_format)
            worker._hyper_pending_policy_version = None
            return
        if not is_checkpoint_format:
            raise ValueError("Direct Qwen3 weight transfer requires checkpoint-format names")
        worker._check_weight_transfer_engine()  # pylint: disable=W0212
        if worker._weight_update_active:  # pylint: disable=W0212
            raise RuntimeError(
                "start_weight_update called while a weight update is already active"
            )
        worker._check_nz_disabled()  # pylint: disable=W0212
        worker._hyper_pending_policy_version = None
        worker._is_checkpoint_format = True  # pylint: disable=W0212
        worker._weight_update_active = True  # pylint: disable=W0212

    def update_weights(worker: Any, update_info: dict[str, Any]) -> None:
        """Receive weights while retaining worker-owned pending identity."""
        versioned_update = dict(update_info)
        version = versioned_update.pop(_POLICY_VERSION_FIELD, None)
        if not _uses_custom_weight_update_lifecycle(worker):
            original_update(worker, versioned_update)
            if version is not None:
                worker._hyper_pending_policy_version = int(version)
            return
        if version is None:
            raise ValueError("Direct Qwen3 weight update requires a worker policy version")
        version = int(version)
        loaded_version = int(getattr(worker, "_hyper_loaded_policy_version", 0))
        pending_version = getattr(worker, "_hyper_pending_policy_version", None)
        if version <= loaded_version:
            raise ValueError(
                "vLLM worker policy version must increase: "
                f"loaded={loaded_version}, received={version}"
            )
        if pending_version is not None and version != pending_version:
            raise ValueError(
                "One vLLM weight update cannot mix policy versions: "
                f"pending={pending_version}, received={version}"
            )
        original_update(worker, versioned_update)
        worker._hyper_pending_policy_version = version

    def finish_weight_update(worker: Any) -> None:
        """Commit worker identity only after the native receiver finishes."""
        if not _uses_custom_weight_update_lifecycle(worker):
            original_finish(worker)
        else:
            _finish_custom_weight_update(worker)
            return
        pending_version = getattr(worker, "_hyper_pending_policy_version", None)
        if pending_version is not None:
            worker._hyper_loaded_policy_version = pending_version
        worker._hyper_pending_policy_version = None
    NPUWorker.start_weight_update = start_weight_update
    NPUWorker.update_weights = update_weights
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
    if not hasattr(WorkerBase, "reload_weights"):
        setattr(WorkerBase, "reload_weights", reload_weights)
    if not hasattr(WorkerBase, "get_policy_weight_fingerprint"):
        setattr(
            WorkerBase,
            "get_policy_weight_fingerprint",
            get_policy_weight_fingerprint,
        )
    if not hasattr(WorkerBase, "get_policy_version"):
        setattr(WorkerBase, "get_policy_version", get_policy_version)
    if not hasattr(WorkerBase, "get_all_parameter_manifest"):
        setattr(
            WorkerBase,
            "get_all_parameter_manifest",
            get_all_parameter_manifest,
        )
    if not hasattr(WorkerBase, "write_parameter_manifest"):
        setattr(WorkerBase, "write_parameter_manifest", write_parameter_manifest)
    if not hasattr(WorkerBase, "commit_reloaded_weights"):
        setattr(WorkerBase, "commit_reloaded_weights", commit_reloaded_weights)
    if not hasattr(WorkerBase, "verify_policy_weight_identity"):
        setattr(
            WorkerBase,
            "verify_policy_weight_identity",
            verify_policy_weight_identity,
        )
    for name, method in (
        ("abort_weight_update", abort_weight_update),
        ("get_direct_reshard_layout", get_direct_reshard_layout),
        ("init_direct_reshard_group", init_direct_reshard_group),
        ("init_full_gather_group", init_full_gather_group),
        ("receive_direct_reshard", receive_direct_reshard),
        ("receive_ipc_direct_reshard", receive_ipc_direct_reshard),
    ):
        if not hasattr(WorkerBase, name):
            setattr(WorkerBase, name, method)
    if private_lifecycle:
        _patch_ascend_weight_update_lifecycle()
        _patch_engine_core_wake_lifecycle()
__all__ = [
    "abort_weight_update",
    "commit_reloaded_weights",
    "get_direct_reshard_layout",
    "get_all_parameter_manifest",
    "get_policy_weight_fingerprint",
    "get_policy_version",
    "init_direct_reshard_group",
    "init_full_gather_group",
    "install_vllm_weight_sync_hooks",
    "reload_weights",
    "verify_policy_weight_identity",
    "write_parameter_manifest",
    "receive_direct_reshard",
    "receive_ipc_direct_reshard",
]
