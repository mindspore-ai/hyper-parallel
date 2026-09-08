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
import resource
from typing import Any, Mapping, Optional
from rl.roles.model import (
    HYPER_DEEPSEEK_V3_ARCHITECTURE,
    HYPER_QWEN3_ARCHITECTURE,
    HYPER_QWEN3_MOE_ARCHITECTURE,
    NATIVE_DEEPSEEK_V3_ARCHITECTURE,
    NATIVE_QWEN3_ARCHITECTURE,
    NATIVE_QWEN3_MOE_ARCHITECTURE,
)
from rl.roles.weight_sync.model_adapter import (
    aggregate_direct_content_identity,
    direct_fragment_record,
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
_HYPER_ARCHITECTURES = frozenset(
    (
        HYPER_DEEPSEEK_V3_ARCHITECTURE,
        HYPER_QWEN3_ARCHITECTURE,
        HYPER_QWEN3_MOE_ARCHITECTURE,
    )
)
_DIRECT_RESHARD_ARCHITECTURES = frozenset(
    (
        HYPER_QWEN3_ARCHITECTURE,
        HYPER_QWEN3_MOE_ARCHITECTURE,
        HYPER_DEEPSEEK_V3_ARCHITECTURE,
        NATIVE_DEEPSEEK_V3_ARCHITECTURE,
        NATIVE_QWEN3_ARCHITECTURE,
        NATIVE_QWEN3_MOE_ARCHITECTURE,
    )
)


@dataclass
class _PatchState:
    """Track process-local idempotent vLLM patch installation."""

    ascend_lifecycle: bool = False
    engine_core_wake: bool = False


_patch_state = _PatchState()


def _current_process_rss_bytes() -> int:
    """Return current Linux resident memory, falling back to the process peak."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


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
    result = {
        "dp_rank": dp_rank,
        "dp_size": dp_size,
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "physical_device_id": physical_device_id,
    }
    if bool(getattr(parallel_config, "enable_expert_parallel", False)):
        from vllm.distributed import get_ep_group  # pylint: disable=C0415

        group = get_ep_group()
        result.update(ep_rank=int(group.rank_in_group), ep_size=int(group.world_size))
        if result["ep_size"] != dp_size * tp_size or result["ep_rank"] != dp_rank * tp_size + tp_rank:
            raise ValueError(f"Rollout EP ownership requires flattened DP x TP coordinates: {result}")
    return result


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


def get_weight_sync_memory_stats(worker: Any) -> dict[str, Any]:
    """Return worker peak device and host memory for sync acceptance logs."""
    handle = platform.get_device_handle(platform.device_type())
    allocated_fn = getattr(handle, "max_memory_allocated", None)
    reserved_fn = getattr(handle, "max_memory_reserved", None)
    current_allocated_fn = getattr(handle, "memory_allocated", None)
    current_reserved_fn = getattr(handle, "memory_reserved", None)
    result = _rollout_worker_topology(worker)
    result.update(
        {
            "max_memory_allocated_bytes": int(allocated_fn()) if allocated_fn else 0,
            "max_memory_reserved_bytes": int(reserved_fn()) if reserved_fn else 0,
            "current_memory_allocated_bytes": (
                int(current_allocated_fn()) if current_allocated_fn else 0
            ),
            "current_memory_reserved_bytes": (
                int(current_reserved_fn()) if current_reserved_fn else 0
            ),
            "current_host_rss_bytes": _current_process_rss_bytes(),
            # Linux reports ru_maxrss in KiB.
            "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
        }
    )
    return result


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
    model_config = getattr(worker, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
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


def _record_direct_content_fragment(
    worker: Any,
    version: int,
    entry: Mapping[str, Any],
    target: Any,
) -> None:
    """Hash one copied destination slice back in canonical axis order."""
    import torch  # pylint: disable=C0415,forbidden-backend-import

    canonical_lengths = tuple(int(value) for value in entry["lengths"])
    permutation = tuple(
        int(value)
        for value in entry.get(
            "destination_permutation",
            range(len(canonical_lengths)),
        )
    )
    inverse_permutation = tuple(
        permutation.index(canonical_axis)
        for canonical_axis in range(len(permutation))
    )
    canonical = target.detach().permute(inverse_permutation).contiguous()
    source_dtype = getattr(torch, str(entry["dtype_name"]))
    if canonical.dtype != source_dtype:
        canonical = canonical.to(dtype=source_dtype)
    raw = canonical.view(torch.uint8).view(-1).to(device="cpu")
    payload = platform.tensor_to_numpy(raw).tobytes()
    key, record = direct_fragment_record(
        str(entry.get("canonical_name", entry["name"])),
        tuple(
            int(value)
            for value in entry.get(
                "canonical_starts",
                entry["destination_starts"],
            )
        ),
        canonical_lengths,
        str(entry["dtype_name"]),
        payload,
    )
    fragments_by_version = getattr(
        worker,
        "_hyper_pending_content_fragments",
        None,
    )
    if fragments_by_version is None:
        fragments_by_version = {}
        worker._hyper_pending_content_fragments = fragments_by_version
    fragments = fragments_by_version.setdefault(int(version), {})
    if key in fragments:
        if fragments[key] != record:
            raise RuntimeError(
                f"Direct content destination changed duplicate fragment {key!r}"
            )
        return
    fragments[key] = record
    del canonical, raw, payload


def verify_direct_content_identity(
    worker: Any,
    expected_version: int,
    expected_by_tp_rank: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify one committed worker against its source-derived TP identity."""
    version = int(expected_version)
    loaded_version = int(getattr(worker, "_hyper_loaded_policy_version", 0))
    if loaded_version != version:
        raise RuntimeError(
            "Direct content identity policy version mismatch: "
            f"loaded={loaded_version}, expected={version}"
        )
    tp_rank = getattr(worker, "_hyper_loaded_content_tp_rank", None)
    actual = getattr(worker, "_hyper_loaded_content_identity", None)
    if tp_rank is None or not isinstance(actual, Mapping):
        raise RuntimeError("Direct content identity was not committed on this worker")
    expected = expected_by_tp_rank.get(str(int(tp_rank)))
    if not isinstance(expected, Mapping):
        raise RuntimeError(
            f"Direct content identity has no source expectation for TP rank {tp_rank}"
        )
    if dict(actual) != dict(expected):
        raise RuntimeError(
            "Direct content identity differs from Trainer source: "
            f"tp_rank={tp_rank}, expected_digest={expected.get('digest')}, "
            f"actual_digest={actual.get('digest')}"
        )
    return {
        "verified": True,
        "version": version,
        "tp_rank": int(tp_rank),
        "digest": actual["digest"],
        "fragment_count": int(actual["fragment_count"]),
        "total_bytes": int(actual["total_bytes"]),
    }


def get_deepseek_v3_runtime_contract(worker: Any) -> dict[str, Any]:
    """Describe the vLLM leaves retained by one DeepSeek-V3 worker.

    Absorbed MLA owns latent paged-KV-cache execution and FusedMoE owns the
    Ascend routed-expert storage/kernel contract.  This diagnostic makes those
    two deliberate Hyper adapter boundaries observable without depending on
    their private Python class names in the control plane.
    """
    if not _is_deepseek_v3_worker(worker):
        raise ValueError("DeepSeek-V3 runtime diagnostics require a DeepSeek worker")
    if worker.model_runner is None:
        raise RuntimeError("DeepSeek-V3 runtime diagnostics require a model runner")
    model = worker.model_runner.get_model()
    hf_config = worker.model_config.hf_config
    attention_leaves = {}
    fused_moe_leaves = {}
    hyper_mla_count = 0
    for module in model.modules():
        mla_attention = getattr(module, "mla_attn", None)
        process_mla_weights = getattr(
            mla_attention,
            "process_weights_after_loading",
            None,
        )
        if mla_attention is not None and callable(process_mla_weights):
            attention_leaves[id(mla_attention)] = (
                f"{type(mla_attention).__module__}.{type(mla_attention).__name__}"
            )
            mla_module = type(mla_attention).__module__
            if mla_module.startswith(("hyper_parallel", "rl.")):
                hyper_mla_count += 1
        if (
            getattr(module, "w13_weight", None) is not None
            and getattr(module, "w2_weight", None) is not None
        ):
            fused_moe_leaves[id(module)] = (
                f"{type(module).__module__}.{type(module).__name__}"
            )
    expected_attention_layers = int(hf_config.num_hidden_layers)
    expected_moe_layers = expected_attention_layers - int(
        getattr(hf_config, "first_k_dense_replace", 0)
    )
    if len(attention_leaves) != expected_attention_layers:
        raise RuntimeError(
            "DeepSeek-V3 absorbed MLA coverage is incomplete: "
            f"expected={expected_attention_layers}, actual={len(attention_leaves)}"
        )
    if len(fused_moe_leaves) != expected_moe_layers:
        raise RuntimeError(
            "DeepSeek-V3 FusedMoE coverage is incomplete: "
            f"expected={expected_moe_layers}, actual={len(fused_moe_leaves)}"
        )
    if hyper_mla_count:
        raise RuntimeError(
            "Moonlight q_lora_rank=None must not use the incompatible Hyper MLA: "
            f"count={hyper_mla_count}"
        )
    enable_expert_parallel = bool(
        getattr(worker.parallel_config, "enable_expert_parallel", False)
    )
    result = {
        "use_mla": bool(worker.model_config.use_mla),
        "q_lora_rank": getattr(hf_config, "q_lora_rank", None),
        "kv_lora_rank": int(hf_config.kv_lora_rank),
        "absorbed_mla_layer_count": len(attention_leaves),
        "absorbed_mla_classes": sorted(set(attention_leaves.values())),
        "hyper_mla_layer_count": hyper_mla_count,
        "fused_moe_layer_count": len(fused_moe_leaves),
        "fused_moe_classes": sorted(set(fused_moe_leaves.values())),
        "enable_expert_parallel": enable_expert_parallel,
    }
    if _is_hyper_deepseek_v3_worker(worker):
        ownership = getattr(model, "hyper_component_ownership", None)
        if not isinstance(ownership, Mapping):
            raise RuntimeError("Hyper DeepSeek-V3 omitted its component ownership contract")
        result["component_ownership"] = dict(ownership)
    return result


def _policy_destination_tensors(model: Any) -> dict[str, Any]:
    """Return refittable parameters and persistent DeepSeek router buffers."""
    tensors = dict(model.named_parameters())
    named_buffers = getattr(model, "named_buffers", lambda: ())
    for name, buffer in named_buffers():
        if not name.endswith(".mlp.gate.e_score_correction_bias"):
            continue
        if name in tensors:
            raise RuntimeError(f"Duplicate rollout policy tensor {name!r}")
        tensors[name] = buffer
    return tensors


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
        named_parameters = list(model.named_parameters(remove_duplicate=False))
    except TypeError:
        named_parameters = list(model.named_parameters())
    existing_names = {name for name, _parameter in named_parameters}
    iter_named_buffers = getattr(model, "named_buffers", lambda: ())
    named_buffers = [
        (name, buffer)
        for name, buffer in iter_named_buffers()
        if name.endswith(".mlp.gate.e_score_correction_bias")
        and name not in existing_names
    ]
    for name, parameter in sorted(
        named_parameters + named_buffers,
        key=lambda item: item[0],
    ):
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
    model_config = getattr(worker, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    architectures = tuple(getattr(hf_config, "architectures", ()) or ())
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
            "architecture": architectures[0] if architectures else None,
            "model_type": getattr(hf_config, "model_type", None),
        }
    )
    if _is_deepseek_v3_worker(worker):
        manifest["deepseek_runtime"] = get_deepseek_v3_runtime_contract(worker)
    if _is_native_deepseek_v3_worker(worker) or _is_native_qwen3_moe_worker(worker):
        manifest["native_moe_ownership"] = _native_moe_ownership_manifest(worker, manifest)
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
    destination_permutation: Optional[tuple[int, ...]] = None,
    accepted_source_dtypes: tuple[str, ...] = (),
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
    permutation = destination_permutation or tuple(range(len(local_shape)))
    if sorted(permutation) != list(range(len(local_shape))):
        raise ValueError(
            f"Native direct tensor {source_name!r} has invalid permutation {permutation}"
        )
    physical_lengths = tuple(local_shape[axis] for axis in permutation)
    if any(
        start < 0 or start + length > int(limit)
        for start, length, limit in zip(
            destination_starts,
            physical_lengths,
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
        "destination_permutation": list(permutation),
        "accepted_source_dtypes": list(accepted_source_dtypes),
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
    intermediate_size = (
        int(hf_config.moe_intermediate_size) * int(hf_config.n_shared_experts)
        if ".shared_experts." in name else int(hf_config.intermediate_size)
    )
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
    *,
    parameters: Optional[list[tuple[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Describe native vLLM Qwen3 storage in canonical Actor coordinates."""
    tensors = []
    vocab_size = int(hf_config.vocab_size)
    entries = model.named_parameters() if parameters is None else parameters
    for name, parameter in sorted(entries, key=lambda item: item[0]):
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
        elif name.endswith((".self_attn.q_proj.weight", ".self_attn.kv_b_proj.weight")):
            local_shape = parameter_shape
            placement = "shard"
            shard_dim = 0
        elif name.endswith((".self_attn.o_proj.weight", ".mlp.down_proj.weight",
                            ".mlp.shared_experts.down_proj.weight")):
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


def _fused_moe_expert_descriptions(
    name: str,
    parameter: Any,
    *,
    family: str,
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    flattened_size: int,
) -> list[dict[str, Any]]:
    """Map canonical expert projections into Ascend FusedMoE storage."""
    if flattened_size <= 0 or intermediate_size % flattened_size != 0:
        raise ValueError(
            f"{family} routed intermediate size must divide the "
            f"flattened MoE size: intermediate={intermediate_size}, size={flattened_size}"
        )
    local_intermediate_size = intermediate_size // flattened_size
    placement = "shard" if flattened_size > 1 else "replicate"
    shard_dim = 1 if flattened_size > 1 else None
    if name.endswith(".w13_weight"):
        checkpoint_shape = (num_experts, 2 * local_intermediate_size, hidden_size)
        runtime_shape = (num_experts, hidden_size, 2 * local_intermediate_size)
        parameter_shape = tuple(int(size) for size in parameter.shape)
        if parameter_shape == checkpoint_shape:
            destination_starts = ((0, 0, 0), (0, local_intermediate_size, 0))
            permutation = None
        elif parameter_shape == runtime_shape:
            destination_starts = ((0, 0, 0), (0, 0, local_intermediate_size))
            permutation = (0, 2, 1)
        else:
            raise ValueError(
                f"{family} w13 parameter {name!r} has shape "
                f"{parameter_shape}, expected checkpoint/runtime layouts "
                f"{checkpoint_shape}/{runtime_shape}"
            )
        prefix = name.removesuffix(".w13_weight")
        canonical_shape = (num_experts, local_intermediate_size, hidden_size)
        return [
            _direct_tensor_description(
                f"{prefix}.{projection}.weight",
                name,
                parameter,
                canonical_shape,
                placement,
                shard_dim,
                destination_start,
                permutation,
            )
            for projection, destination_start in zip(
                ("gate_proj", "up_proj"),
                destination_starts,
            )
        ]
    checkpoint_shape = (num_experts, hidden_size, local_intermediate_size)
    runtime_shape = (num_experts, local_intermediate_size, hidden_size)
    parameter_shape = tuple(int(size) for size in parameter.shape)
    if parameter_shape == checkpoint_shape:
        permutation = None
    elif parameter_shape == runtime_shape:
        permutation = (0, 2, 1)
    else:
        raise ValueError(
            f"{family} w2 parameter {name!r} has shape "
            f"{parameter_shape}, expected checkpoint/runtime layouts "
            f"{checkpoint_shape}/{runtime_shape}"
        )
    prefix = name.removesuffix(".w2_weight")
    return [
        _direct_tensor_description(
            f"{prefix}.down_proj.weight",
            name,
            parameter,
            (num_experts, hidden_size, local_intermediate_size),
            placement,
            2 if flattened_size > 1 else None,
            (0, 0, 0),
            permutation,
        )
    ]


def _native_moe_gate_up_descriptions(
    name: str,
    parameter: Any,
    hf_config: Any,
) -> list[dict[str, Any]]:
    """Map one replicated dense/shared gate/up linear in a native TP1 MoE model."""
    if ".shared_experts." in name:
        intermediate_size = int(hf_config.moe_intermediate_size) * int(
            hf_config.n_shared_experts
        )
    else:
        intermediate_size = int(hf_config.intermediate_size)
    tail_shape = tuple(int(size) for size in parameter.shape[1:])
    expected_shape = (2 * intermediate_size,) + tail_shape
    if tuple(int(size) for size in parameter.shape) != expected_shape:
        raise ValueError(
            f"Native MoE gate/up parameter {name!r} has shape "
            f"{tuple(parameter.shape)}, expected {expected_shape}"
        )
    return [
        _direct_tensor_description(
            name.replace("gate_up_proj", projection),
            name,
            parameter,
            (intermediate_size,) + tail_shape,
            "replicate",
            None,
            (destination_offset,) + (0,) * len(tail_shape),
        )
        for projection, destination_offset in (
            ("gate_proj", 0),
            ("up_proj", intermediate_size),
        )
    ]


def _native_moe_local_experts(module: Any, num_experts: int, ep_size: int, ep_rank: int) -> int:
    """Validate actual static native ownership before using contiguous planner slices."""
    if bool(getattr(module, "dynamic_eplb", False)) or bool(getattr(module, "enable_eplb", False)):
        raise ValueError("Native MoE weight synchronization requires EPLB disabled")
    if ep_size <= 0 or not 0 <= ep_rank < ep_size or num_experts % ep_size:
        raise ValueError("Native MoE requires evenly divisible static expert ownership")
    local_experts = num_experts // ep_size
    expert_map = getattr(module, "expert_map", None)
    if ep_size > 1:
        if expert_map is None:
            raise ValueError("Native EP requires an actual global-to-local expert map")
        actual = expert_map.detach().cpu().tolist()
        expected = [-1] * num_experts
        start = ep_rank * local_experts
        expected[start:start + local_experts] = list(range(local_experts))
        if actual != expected:
            raise ValueError("Native MoE expert map is not the supported static contiguous placement")
    elif expert_map is not None:
        raise ValueError("Native EP-off unexpectedly exposes an expert map")
    if int(module.w13_weight.shape[0]) != local_experts or int(module.w2_weight.shape[0]) != local_experts:
        raise ValueError("Native MoE physical expert count differs from the verified ownership")
    return local_experts


def _native_moe_ownership_manifest(worker: Any, topology: Mapping[str, Any]) -> dict[str, Any]:
    """Persist actual original-leaf ownership only for explicitly requested manifests."""
    config = worker.model_config.hf_config
    num_experts = int(config.n_routed_experts if _is_native_deepseek_v3_worker(worker) else config.num_experts)
    ep_size, ep_rank = int(topology.get("ep_size", 1)), int(topology.get("ep_rank", 0))
    ownership = {}
    for name, module in worker.model_runner.get_model().named_modules():
        if not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
            continue
        local_experts = _native_moe_local_experts(module, num_experts, ep_size, ep_rank)
        expert_map = getattr(module, "expert_map", None)
        ownership[name] = {
            "class": f"{type(module).__module__}.{type(module).__name__}",
            "global_to_local": None if expert_map is None else expert_map.detach().cpu().tolist(),
            "local_experts": local_experts,
            "w13_shape": list(module.w13_weight.shape),
            "w2_shape": list(module.w2_weight.shape),
        }
    return ownership


def _native_moe_direct_tensors(
    model: Any,
    hf_config: Any,
    tp_size: int,
    dp_size: int = 1,
    *,
    family: str,
    ep_size: int = 1,
    ep_rank: int = 0,
    tp_rank: int = 0,
) -> list[dict[str, Any]]:
    """Describe original vLLM MoE storage without replacing its compute components."""
    if tp_size != 1 and not (
        family in ("qwen3_moe", "deepseek_v3") and tp_size == 2 and ep_size == dp_size * tp_size
    ):
        raise ValueError(
            "Native MoE direct reshard supports rollout TP1 only, "
            f"got TP{tp_size}"
        )
    num_experts = int(hf_config.n_routed_experts if family == "deepseek_v3" else hf_config.num_experts)
    modules = dict(getattr(model, "named_modules", lambda: ())())
    ownership = {}
    tensors = []
    if tp_size > 1:
        dense_parameters = [
            (name, parameter) for name, parameter in model.named_parameters()
            if not name.endswith((".experts.w13_weight", ".experts.w2_weight"))
        ]
        tensors.extend(_native_qwen3_direct_tensors(
            model, hf_config, tp_rank, tp_size, parameters=dense_parameters,
        ))
        for description in tensors:
            if description["name"].endswith(".mlp.gate.weight") and description["dtype_name"] == "float32":
                description["accepted_source_dtypes"] = ["bfloat16"]
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if name.endswith((".experts.w13_weight", ".experts.w2_weight")):
            module_name = name.rsplit(".", 1)[0]
            if module_name not in ownership:
                module = modules.get(module_name)
                # EP-off storage contains every expert; EP-local views require the actual ownership map.
                ownership[module_name] = (
                    _native_moe_local_experts(module, num_experts, ep_size, ep_rank)
                    if module is not None or ep_size > 1 else num_experts
                )
            descriptions = _fused_moe_expert_descriptions(
                name,
                parameter,
                family=family,
                num_experts=ownership[module_name],
                intermediate_size=int(hf_config.moe_intermediate_size),
                hidden_size=int(hf_config.hidden_size),
                flattened_size=1 if ep_size > 1 else dp_size * tp_size,
            )
            if ep_size > 1:
                for description in descriptions:
                    description.update(placement="shard", shard_dim=0)
            tensors.extend(descriptions)
            continue
        if tp_size > 1:
            continue
        if family == "qwen3_moe" and ".qkv_proj." in name:
            descriptions = _native_qwen3_qkv_descriptions(name, parameter, hf_config, tp_size)
            for description in descriptions:
                description.update(placement="replicate", shard_dim=None)
            tensors.extend(descriptions)
            continue
        if ".gate_up_proj." in name:
            tensors.extend(
                _native_moe_gate_up_descriptions(
                    name,
                    parameter,
                    hf_config,
                )
            )
            continue
        shape = tuple(int(size) for size in parameter.shape)
        if name in {"model.embed_tokens.weight", "lm_head.weight"}:
            shape = (int(hf_config.vocab_size),) + shape[1:]
        tensors.append(
            _direct_tensor_description(
                name,
                name,
                parameter,
                shape,
                "replicate",
                None,
                (0,) * len(shape),
                accepted_source_dtypes=(
                    ("bfloat16",)
                    if name.endswith(".mlp.gate.weight")
                    and str(parameter.dtype).rsplit(".", maxsplit=1)[-1] == "float32"
                    else ()
                ),
            )
        )
    return tensors


def _hyper_tp_placement(model: Any, name: str, tp_size: int) -> tuple[str, Optional[int]]:
    """Read the public apply pass's parameter placement for either Hyper family."""
    placements = tuple(getattr(model, "_tp_placements", {}).get(name, ()))
    if tp_size == 1 and not placements:
        return "replicate", None
    if len(placements) != 1:
        raise ValueError(f"Direct reshard parameter {name!r} requires one TP placement, got {placements}")
    placement = placements[0]
    if callable(getattr(placement, "is_shard", None)) and placement.is_shard():
        return "shard", int(placement.dim)
    if callable(getattr(placement, "is_replicate", None)) and placement.is_replicate():
        return "replicate", None
    raise ValueError(f"Direct reshard parameter {name!r} has unsupported placement {placement!r}")


def _hyper_moe_direct_tensors(
    model: Any,
    *,
    family: str,
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    tp_size: int,
) -> list[dict[str, Any]]:
    """Describe an HF outer model with common local FusedMoE leaves."""
    if tp_size != 1 and not (tp_size == 2 and family in ("Qwen3-MoE", "qwen3_moe", "DeepSeek-V3")):
        raise ValueError(
            f"Hyper {family} weight synchronization supports rollout TP1 only, "
            f"got TP{tp_size}"
        )
    tensors = []
    policy_tensors = _policy_destination_tensors(model)
    named_modules = getattr(model, "named_modules", lambda: ())
    modules = dict(named_modules())
    for name, parameter in sorted(policy_tensors.items(), key=lambda item: item[0]):
        if name.endswith((".experts.w13_weight", ".experts.w2_weight")):
            expert_module_name = name.rsplit(".", maxsplit=1)[0]
            expert_module = modules.get(expert_module_name)
            if not bool(getattr(expert_module, "hyper_local_expert_leaf", False)):
                raise RuntimeError(
                    f"Hyper {family} expert {expert_module_name!r} is not the common local leaf"
                )
            local_count = int(getattr(expert_module, "local_expert_count", num_experts))
            if local_count <= 0 or num_experts % local_count:
                raise ValueError(f"Invalid Hyper {family} local expert count {local_count}")
            descriptions = _fused_moe_expert_descriptions(
                name,
                parameter,
                family=family,
                num_experts=local_count,
                intermediate_size=intermediate_size,
                hidden_size=hidden_size,
                flattened_size=1,
            )
            if local_count != num_experts:
                for description in descriptions:
                    description.update(placement="shard", shard_dim=0)
            tensors.extend(descriptions)
            continue
        shape = tuple(int(size) for size in parameter.shape)
        placement_name, shard_dim = _hyper_tp_placement(model, name, tp_size)
        tensors.append(
            _direct_tensor_description(
                name,
                name,
                parameter,
                shape,
                placement_name,
                shard_dim,
                (0,) * len(shape),
                accepted_source_dtypes=(
                    ("bfloat16",)
                    if name.endswith(".mlp.gate.weight")
                    and str(parameter.dtype).rsplit(".", maxsplit=1)[-1] == "float32"
                    else ()
                ),
            )
        )
    return tensors


def _hyper_deepseek_v3_direct_tensors(
    model: Any,
    hf_config: Any,
    tp_size: int,
    dp_size: int = 1,
) -> list[dict[str, Any]]:
    """Describe the Hyper DeepSeek-V3 EP1 destination layout."""
    del dp_size
    return _hyper_moe_direct_tensors(
        model,
        family="DeepSeek-V3",
        num_experts=int(hf_config.n_routed_experts),
        intermediate_size=int(hf_config.moe_intermediate_size),
        hidden_size=int(hf_config.hidden_size),
        tp_size=tp_size,
    )


def _hyper_qwen3_moe_direct_tensors(
    model: Any,
    hf_config: Any,
    tp_size: int,
) -> list[dict[str, Any]]:
    """Describe the Hyper Qwen3-MoE EP1 destination layout."""
    return _hyper_moe_direct_tensors(
        model,
        family="Qwen3-MoE",
        num_experts=int(hf_config.num_experts),
        intermediate_size=int(hf_config.moe_intermediate_size),
        hidden_size=int(hf_config.hidden_size),
        tp_size=tp_size,
    )


def get_direct_reshard_layout(worker: Any) -> dict[str, Any]:
    """Describe one supported worker's local parameters."""
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    model = worker.model_runner.get_model()
    from vllm.distributed import get_tp_group  # pylint: disable=C0415

    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    if _is_hyper_qwen3_moe_worker(worker):
        hf_config = getattr(worker.model_config, "hf_config", None)
        if hf_config is None:
            raise ValueError("Qwen3-MoE direct reshard requires an HF config")
        result = {
            "tensors": _hyper_qwen3_moe_direct_tensors(
                model,
                hf_config,
                tp_size,
            ),
        }
        result.update(_rollout_worker_topology(worker))
        return result
    if _is_hyper_deepseek_v3_worker(worker):
        hf_config = getattr(worker.model_config, "hf_config", None)
        if hf_config is None:
            raise ValueError("DeepSeek-V3 direct reshard requires an HF config")
        topology = _rollout_worker_topology(worker)
        result = {
            "tensors": _hyper_deepseek_v3_direct_tensors(
                model,
                hf_config,
                tp_size,
                int(topology["dp_size"]),
            ),
        }
        result.update(topology)
        return result
    if _is_native_deepseek_v3_worker(worker) or _is_native_qwen3_moe_worker(worker):
        hf_config = getattr(worker.model_config, "hf_config", None)
        if hf_config is None:
            raise ValueError("DeepSeek-V3 direct reshard requires an HF config")
        topology = _rollout_worker_topology(worker)
        result = {
            "tensors": _native_moe_direct_tensors(
                model,
                hf_config,
                tp_size,
                int(topology["dp_size"]),
                family="deepseek_v3" if _is_native_deepseek_v3_worker(worker) else "qwen3_moe",
                ep_size=int(topology.get("ep_size", 1)),
                ep_rank=int(topology.get("ep_rank", 0)),
                tp_rank=tp_rank,
            ),
        }
        result.update(topology)
        return result
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
            "Direct reshard requires a supported Hyper or native rollout model"
        )
    tensors = []
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        placement_name, shard_dim = _hyper_tp_placement(model, name, tp_size)
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


def _validate_direct_update(worker: Any, policy_version: int, *, transport: str) -> int:
    """Validate the shared transaction preconditions for one direct bucket."""
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    if not _is_direct_reshard_worker(worker):
        raise ValueError(f"{transport} reshard requires a supported rollout worker")
    if not bool(getattr(worker, "_weight_update_active", False)):
        raise RuntimeError(f"{transport} reshard requires an active vLLM weight update")
    version = int(policy_version)
    loaded_version = int(getattr(worker, "_hyper_loaded_policy_version", 0))
    pending_version = getattr(worker, "_hyper_pending_policy_version", None)
    if version <= loaded_version:
        raise ValueError(
            f"{transport} reshard policy version must increase: "
            f"loaded={loaded_version}, received={version}"
        )
    if pending_version is not None and int(pending_version) != version:
        raise ValueError(
            "One direct reshard update cannot mix policy versions: "
            f"pending={pending_version}, received={version}"
        )
    return version


def _apply_direct_bucket(
    worker: Any,
    parameters: Mapping[str, Any],
    packed: Any,
    metadata: Mapping[str, Any],
    policy_version: int,
    *,
    transport: str,
) -> int:
    """Scatter one packed direct bucket into rollout-local parameters."""
    import torch  # pylint: disable=C0415,forbidden-backend-import

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
        _record_direct_content_fragment(worker, policy_version, entry, target)
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
    version = _validate_direct_update(worker, policy_version, transport="Direct")
    groups = getattr(worker, "_hyper_direct_reshard_groups", {})
    group = groups.get(group_id)
    if group is None:
        raise RuntimeError(f"Direct reshard HCCL group {group_id!r} is not initialized")
    import torch  # pylint: disable=C0415,forbidden-backend-import

    parameters = _policy_destination_tensors(worker.model_runner.get_model())
    received_bytes = 0
    for bucket in buckets:
        total_bytes = int(bucket["total_bytes"])
        packed = torch.empty(total_bytes, dtype=torch.uint8, device=group.device)
        group.broadcast(packed, src=0)
        torch.npu.current_stream().synchronize()
        received_bytes += _apply_direct_bucket(
            worker,
            parameters,
            packed,
            bucket,
            version,
            transport="Direct reshard rollout",
        )
        del packed
    worker._hyper_pending_policy_version = version
    worker._hyper_pending_content_tp_rank = tp_rank
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

    version = _validate_direct_update(worker, policy_version, transport="IPC direct")

    payload = pickle.loads(base64.b64decode(payload_pickled.encode("ascii")))
    topology = _rollout_worker_topology(worker)
    tp_rank = int(topology["tp_rank"])
    expected_workers = {
        worker_description["physical_device_id"]: worker_description
        for worker_description in payload["worker_topology"]
    }
    delivery_mode = str(payload.get("delivery_mode", "replicated_tp"))
    worker_tp_size = int(topology.get("tp_size", 1))
    tensor_parallel_size = int(payload.get("tensor_parallel_size", worker_tp_size))
    if tensor_parallel_size != worker_tp_size:
        raise ValueError(
            "IPC direct payload TP size differs from the worker topology: "
            f"payload={tensor_parallel_size}, worker={worker_tp_size}"
        )
    if delivery_mode == "replicated_tp":
        target_rank = tp_rank
    elif delivery_mode == "flattened_dp_tp":
        target_rank = int(topology["dp_rank"]) * tensor_parallel_size + tp_rank
    else:
        raise ValueError(f"Unsupported IPC direct delivery mode {delivery_mode!r}")
    buckets_by_target = payload.get("buckets_by_target", payload.get("buckets_by_tp", {}))
    buckets = buckets_by_target.get(target_rank, ())
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
    parameters = _policy_destination_tensors(worker.model_runner.get_model())
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
            received_bytes += _apply_direct_bucket(
                worker,
                parameters,
                packed,
                metadata,
                version,
                transport="IPC direct reshard",
            )
    finally:
        if imported_buffers:
            torch.npu.current_stream().synchronize()
            imported_buffers.clear()

    worker._hyper_pending_policy_version = version
    worker._hyper_pending_content_tp_rank = target_rank
    return {
        "received": True,
        "dp_rank": int(topology["dp_rank"]),
        "tp_rank": tp_rank,
        "physical_device_id": physical_npu_id,
        "bytes": received_bytes,
        "bucket_count": len(buckets),
    }


def abort_weight_update(worker: Any, restore_policy_version: int) -> dict[str, Any]:
    """Clear a failed update transaction before a full-checkpoint retry."""
    was_active = bool(getattr(worker, "_weight_update_active", False))
    pending_version = getattr(worker, "_hyper_pending_policy_version", None)
    worker._weight_update_active = False
    worker._is_checkpoint_format = True
    worker._hyper_pending_policy_version = None
    worker._hyper_pending_content_fragments = {}
    worker._hyper_pending_content_tp_rank = None
    worker._hyper_loaded_policy_version = int(restore_policy_version)
    worker._hyper_loaded_content_identity = None
    worker._hyper_loaded_content_tp_rank = None
    return {
        "aborted": True,
        "was_active": was_active,
        "pending_version": pending_version,
        "restored_version": int(restore_policy_version),
    }


def _worker_architectures(worker: Any) -> frozenset[str]:
    """Return the worker's declared Hugging Face model architectures."""
    model_config = getattr(worker, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", ())
    return frozenset(architectures or ())


def _is_hyper_worker(worker: Any) -> bool:
    """Return whether the worker hosts a Hyper-registered model."""
    return bool(_HYPER_ARCHITECTURES.intersection(_worker_architectures(worker)))


def _is_native_qwen3_worker(worker: Any) -> bool:
    """Return whether the worker hosts native vLLM Qwen3."""
    return NATIVE_QWEN3_ARCHITECTURE in _worker_architectures(worker)


def _is_hyper_qwen3_moe_worker(worker: Any) -> bool:
    """Return whether the worker uses the Hyper Qwen3-MoE adapter."""
    return HYPER_QWEN3_MOE_ARCHITECTURE in _worker_architectures(worker)


def _is_native_qwen3_moe_worker(worker: Any) -> bool:
    """Return whether the worker uses the original vLLM Qwen3-MoE model."""
    return NATIVE_QWEN3_MOE_ARCHITECTURE in _worker_architectures(worker)


def _is_hyper_deepseek_v3_worker(worker: Any) -> bool:
    """Return whether the worker uses the HF-outer Hyper DeepSeek adapter."""
    return HYPER_DEEPSEEK_V3_ARCHITECTURE in _worker_architectures(worker)


def _is_native_deepseek_v3_worker(worker: Any) -> bool:
    """Return whether the worker uses vLLM's native DeepSeek model."""
    return NATIVE_DEEPSEEK_V3_ARCHITECTURE in _worker_architectures(worker)


def _is_deepseek_v3_worker(worker: Any) -> bool:
    """Return whether the worker hosts either DeepSeek-V3 runtime."""
    return _is_hyper_deepseek_v3_worker(worker) or _is_native_deepseek_v3_worker(worker)


def _is_direct_reshard_worker(worker: Any) -> bool:
    """Return whether the worker supports direct-reshard weight updates."""
    return bool(
        _DIRECT_RESHARD_ARCHITECTURES.intersection(_worker_architectures(worker))
    )


def _uses_custom_weight_update_lifecycle(worker: Any) -> bool:
    """Return whether Hyper owns this worker's update transaction lifecycle."""
    return _is_direct_reshard_worker(worker)


def _refresh_native_deepseek_v3_derived_weights(worker: Any) -> int:
    """Refresh absorbed MLA leaves in place after their source weights change."""
    if not _is_deepseek_v3_worker(worker):
        return 0
    if worker.model_runner is None:
        raise RuntimeError("Native DeepSeek-V3 MLA refresh requires a model runner")
    model = worker.model_runner.get_model()
    act_dtype = getattr(worker.model_config, "dtype", None)
    if act_dtype is None:
        raise ValueError("Native DeepSeek-V3 MLA refresh requires model_config.dtype")
    refreshed = set()
    for module in model.modules():
        mla_attention = getattr(module, "mla_attn", None)
        process_weights = getattr(
            mla_attention,
            "process_weights_after_loading",
            None,
        )
        if mla_attention is None or not callable(process_weights):
            continue
        if id(mla_attention) in refreshed:
            continue
        process_weights(act_dtype)
        refreshed.add(id(mla_attention))
    expected_layers = int(worker.model_config.hf_config.num_hidden_layers)
    if len(refreshed) != expected_layers:
        raise RuntimeError(
            "Native DeepSeek-V3 MLA refresh did not cover every layer: "
            f"expected={expected_layers}, actual={len(refreshed)}"
        )
    return len(refreshed)


def _refresh_fused_moe_weights(worker: Any) -> int:
    """Restore supported FusedMoE weights to Ascend runtime layout."""
    is_deepseek = _is_deepseek_v3_worker(worker)
    if not is_deepseek and not _is_hyper_qwen3_moe_worker(worker) and not _is_native_qwen3_moe_worker(worker):
        return 0
    if worker.model_runner is None:
        raise RuntimeError("FusedMoE refresh requires a model runner")
    model = worker.model_runner.get_model()
    hf_config = worker.model_config.hf_config
    topology = _rollout_worker_topology(worker)
    flattened_size = int(topology["dp_size"]) * int(topology["tp_size"])
    if int(topology.get("ep_size", 1)) > 1:
        flattened_size = 1
    intermediate_size = int(getattr(hf_config, "moe_intermediate_size", 0))
    hidden_size = int(getattr(hf_config, "hidden_size", 0))
    refreshed = 0
    discovered = 0
    for module in model.modules():
        w13_weight = getattr(module, "w13_weight", None)
        w2_weight = getattr(module, "w2_weight", None)
        if w13_weight is None or w2_weight is None:
            continue
        discovered += 1
        if bool(getattr(module, "hyper_local_expert_leaf", False)):
            before = (tuple(w13_weight.shape), tuple(w2_weight.shape))
            ensure_layout = getattr(module, "ensure_physical_weight_layout", None)
            if not callable(ensure_layout):
                raise RuntimeError("Hyper local FusedMoE cannot restore its physical layout")
            ensure_layout()
            after = (tuple(module.w13_weight.shape), tuple(module.w2_weight.shape))
            refreshed += int(before != after)
            continue
        if not is_deepseek and not _is_native_qwen3_moe_worker(worker):
            raise RuntimeError(
                "Hyper Qwen3-MoE contains a routed expert that is not the common local leaf"
            )
        if intermediate_size % flattened_size != 0:
            raise ValueError(
                "Native MoE FusedMoE intermediate size must divide the flattened "
                f"runtime size: intermediate={intermediate_size}, size={flattened_size}"
            )
        local_intermediate_size = intermediate_size // flattened_size
        checkpoint_w13_tail = (2 * local_intermediate_size, hidden_size)
        runtime_w13_tail = (hidden_size, 2 * local_intermediate_size)
        w13_tail = tuple(int(size) for size in w13_weight.shape[1:])
        w2_tail = tuple(int(size) for size in w2_weight.shape[1:])
        if w13_tail == runtime_w13_tail and w2_tail == (local_intermediate_size, hidden_size):
            continue
        if w13_tail != checkpoint_w13_tail or w2_tail != (hidden_size, local_intermediate_size):
            raise RuntimeError(
                "Native MoE FusedMoE has an unsupported pre-refresh layout: "
                f"w13={tuple(w13_weight.shape)}, w2={tuple(w2_weight.shape)}"
            )
        process_weights = getattr(
            getattr(module, "quant_method", None),
            "process_weights_after_loading",
            None,
        )
        if not callable(process_weights):
            raise RuntimeError("Native MoE FusedMoE quant method cannot refresh weights")
        process_weights(module)
        refreshed += 1
    if discovered <= 0:
        raise RuntimeError("FusedMoE refresh found no routed expert layers")
    if refreshed not in (0, discovered):
        raise RuntimeError(
            "FusedMoE workers mixed checkpoint and runtime layouts: "
            f"refreshed={refreshed}, discovered={discovered}"
        )
    return refreshed


def prepare_direct_reshard_layout(worker: Any) -> dict[str, Any]:
    """Restore executable FusedMoE storage before layout planning."""
    refreshed = _refresh_fused_moe_weights(worker)
    result = {
        "prepared": True,
        "moe_refresh_count": refreshed,
    }
    result.update(_rollout_worker_topology(worker))
    return result


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
    pending_tp_rank = getattr(worker, "_hyper_pending_content_tp_rank", None)
    if pending_tp_rank is not None:
        refreshed_moe_layers = (
            _refresh_fused_moe_weights(worker)
            if hasattr(worker, "model_config")
            else 0
        )
        refreshed_mla_layers = (
            _refresh_native_deepseek_v3_derived_weights(worker)
            if hasattr(worker, "model_config")
            else 0
        )
        fragments_by_version = getattr(
            worker,
            "_hyper_pending_content_fragments",
            {},
        )
        fragments = fragments_by_version.pop(int(pending_version), None)
        if not fragments:
            raise RuntimeError(
                "finish_weight_update requires direct content fragments before commit"
            )
        worker._hyper_loaded_content_identity = aggregate_direct_content_identity(
            fragments
        )
        worker._hyper_loaded_content_tp_rank = int(pending_tp_rank)
        worker._hyper_loaded_moe_refresh_count = refreshed_moe_layers
        worker._hyper_loaded_mla_refresh_count = refreshed_mla_layers
    else:
        worker._hyper_loaded_content_identity = None
        worker._hyper_loaded_content_tp_rank = None
    worker._weight_update_active = False  # pylint: disable=W0212
    worker._is_checkpoint_format = True  # pylint: disable=W0212
    worker._hyper_loaded_policy_version = pending_version
    worker._hyper_pending_policy_version = None
    worker._hyper_pending_content_tp_rank = None


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
        if not _uses_custom_weight_update_lifecycle(worker):
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
        worker._hyper_pending_policy_version = None
        worker._hyper_pending_content_fragments = {}
        worker._hyper_pending_content_tp_rank = None
        worker._is_checkpoint_format = True  # pylint: disable=W0212
        worker._weight_update_active = True  # pylint: disable=W0212

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
        if "weights" in memory_tags:
            prepared = engine_core.model_executor.collective_rpc(
                "prepare_direct_reshard_layout"
            )
            if not prepared or not all(
                isinstance(result, Mapping) and bool(result.get("prepared"))
                for result in prepared
            ):
                raise RuntimeError(
                    "vLLM workers did not restore executable weight layouts "
                    f"during atomic wake: {prepared}"
                )
        return None

    EngineCore.wake_up = wake_up
    _patch_state.engine_core_wake = True


def install_vllm_weight_sync_hooks(*, private_lifecycle: bool = True) -> None:
    """Install stable worker RPCs and optionally pinned private lifecycle patches."""
    from vllm.v1.worker.worker_base import WorkerBase  # pylint: disable=C0415
    if not hasattr(WorkerBase, "get_policy_weight_fingerprint"):
        setattr(
            WorkerBase,
            "get_policy_weight_fingerprint",
            get_policy_weight_fingerprint,
        )
    if not hasattr(WorkerBase, "get_policy_version"):
        setattr(WorkerBase, "get_policy_version", get_policy_version)
    if not hasattr(WorkerBase, "get_weight_sync_memory_stats"):
        setattr(
            WorkerBase,
            "get_weight_sync_memory_stats",
            get_weight_sync_memory_stats,
        )
    if not hasattr(WorkerBase, "get_all_parameter_manifest"):
        setattr(
            WorkerBase,
            "get_all_parameter_manifest",
            get_all_parameter_manifest,
        )
    if not hasattr(WorkerBase, "write_parameter_manifest"):
        setattr(WorkerBase, "write_parameter_manifest", write_parameter_manifest)
    if not hasattr(WorkerBase, "verify_policy_weight_identity"):
        setattr(
            WorkerBase,
            "verify_policy_weight_identity",
            verify_policy_weight_identity,
        )
    if not hasattr(WorkerBase, "verify_direct_content_identity"):
        setattr(
            WorkerBase,
            "verify_direct_content_identity",
            verify_direct_content_identity,
        )
    for name, method in (
        ("abort_weight_update", abort_weight_update),
        ("prepare_direct_reshard_layout", prepare_direct_reshard_layout),
        ("get_direct_reshard_layout", get_direct_reshard_layout),
        ("init_direct_reshard_group", init_direct_reshard_group),
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
    "get_direct_reshard_layout",
    "get_all_parameter_manifest",
    "get_policy_weight_fingerprint",
    "get_policy_version",
    "get_weight_sync_memory_stats",
    "init_direct_reshard_group",
    "install_vllm_weight_sync_hooks",
    "prepare_direct_reshard_layout",
    "verify_policy_weight_identity",
    "verify_direct_content_identity",
    "write_parameter_manifest",
    "receive_direct_reshard",
    "receive_ipc_direct_reshard",
]
