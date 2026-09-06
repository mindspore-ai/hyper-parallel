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
"""Validate Hyper-RL configuration and adapt it to Hyper-Parallel."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from rl.agentic.core.types import InteractionMode
from rl.agentic.envs.environment import ENVIRONMENTS, load_agentic_module
from rl.algorithm.loss import RLAlgorithm
from rl.consistency import consistency_profile, validate_consistency_model_identity
from rl.roles.model import (
    ModelRegistration,
    VLLMModelRegistration,
    normalize_model_implementation,
    resolve_vllm_model,
)
from rl.roles.rollout import ROLLOUT_ENGINES

from hyper_parallel import get_platform
from hyper_parallel.platform.platform import PlatformType
from hyper_parallel.auto_models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.auto_models.components.checkpoint.config import CheckpointingConfig
from hyper_parallel.auto_models.components.distributed.config import (
    FSDP2Config,
    FSDP2MixedPrecisionConfig,
)
from hyper_parallel.auto_models.components.optim.lr_scheduler import MultiLRScheduler
from hyper_parallel.auto_models.components.optim.optimizer import AdamW
from hyper_parallel.auto_models.trainer.config import (
    AcceleratorConfig,
    ActivationCheckpointConfig,
    MixedPrecisionConfig,
    Target,
    TrainerConfig,
    TrainingConfig,
)
platform = get_platform()
_HCCL_MIN_PORT = 1024
_HCCL_MAX_PORT = 65520
_EXPECTED_TOP_LEVEL = frozenset(
    (
        "model",
        "data",
        "rollout",
        "agentic",
        "algorithm",
        "evaluation",
        "train",
        "logging",
        "consistency",
    )
)


def required_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one required mapping-valued configuration section."""
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def optional_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one optional mapping-valued configuration section."""
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def _path_value(section: Mapping[str, Any], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration field '{name}' must be a non-empty path string")
    return value


def uses_colocated_vllm(config: Mapping[str, Any]) -> bool:
    """Return whether rollout shares each trainer rank's NPU."""
    rollout = config.get("rollout", {})
    if not isinstance(rollout, Mapping) or rollout.get("engine") != "vllm":
        return False
    vllm = rollout.get("vllm", {})
    return isinstance(vllm, Mapping) and vllm.get("deployment", "disjoint") == "colocated"


def _validate_model_and_data_paths(
    model: Mapping[str, Any],
    data: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> None:
    """Validate model, tokenizer, and dataset paths used by the run."""
    model_name = model.get("name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model.name must be a non-empty string")
    checks = (
        (Path(_path_value(model, "weights_path")), "Model weights directory", True),
        (Path(_path_value(model, "tokenizer_path")), "Tokenizer directory", True),
        (Path(_path_value(data, "train_path")), "Training parquet file", False),
    )
    for path, description, is_directory in checks:
        exists = path.is_dir() if is_directory else path.is_file()
        if not exists:
            raise ValueError(f"{description} does not exist: {path}")
    if bool(evaluation.get("enabled", True)):
        test_path = Path(_path_value(data, "test_path"))
        if not test_path.is_file():
            raise ValueError(f"Evaluation parquet file does not exist: {test_path}")


def _validate_training_sizes(
    train: Mapping[str, Any],
    rollout: Mapping[str, Any],
    data: Mapping[str, Any],
    algorithm: RLAlgorithm,
) -> None:
    """Validate rollout and optimization batch sizes."""
    for field in (
        "max_steps",
        "prompt_batch_size",
        "micro_batch_size",
        "response_mini_batch_size",
    ):
        value = int(train.get(field, 1 if field == "prompt_batch_size" else 0))
        if value <= 0:
            raise ValueError(f"train.{field} must be positive, got {value}")
    num_responses = int(rollout.get("num_return_sequences", 0))
    minimum_responses = 2 if algorithm.requirements.data.grouped_responses else 1
    if num_responses < minimum_responses:
        raise ValueError(
            "rollout.num_return_sequences must be at least "
            f"{minimum_responses} for algorithm '{algorithm.name}'"
        )
    mini_batch_size = int(train["response_mini_batch_size"])
    local_responses = int(train.get("prompt_batch_size", 1)) * num_responses
    if mini_batch_size > local_responses:
        raise ValueError(
            "train.response_mini_batch_size cannot exceed the local rollout response count: "
            f"mini_batch={mini_batch_size}, responses={local_responses}"
        )
    if int(rollout.get("max_new_tokens", 0)) <= 0:
        raise ValueError("rollout.max_new_tokens must be positive")
    if not isinstance(rollout.get("ignore_eos", False), bool):
        raise ValueError("rollout.ignore_eos must be a boolean")
    if int(data.get("max_prompt_length", 0)) <= 0:
        raise ValueError("data.max_prompt_length must be positive")
    if int(train.get("policy_update_epochs", 0)) <= 0:
        raise ValueError("train.policy_update_epochs must be positive")
    gate = train.get("learning_gate", {})
    if not isinstance(gate, Mapping):
        raise ValueError("train.learning_gate must be a mapping")
    if float(gate.get("min_gradient_norm", 0.0)) < 0:
        raise ValueError("train.learning_gate.min_gradient_norm must be non-negative")


def _validate_evaluation(evaluation: Mapping[str, Any]) -> None:
    """Validate optional evaluation limits."""
    if not bool(evaluation.get("enabled", True)):
        return
    if not isinstance(evaluation.get("ignore_eos", False), bool):
        raise ValueError("evaluation.ignore_eos must be a boolean")
    for field in ("batch_size", "max_new_tokens"):
        if int(evaluation.get(field, 0)) <= 0:
            raise ValueError(f"evaluation.{field} must be positive")
    max_samples = evaluation.get("max_samples")
    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("evaluation.max_samples must be positive or null")
    for field in ("log_samples", "progress_steps"):
        if int(evaluation.get(field, 0)) < 0:
            raise ValueError(f"evaluation.{field} must be non-negative")


def _parallel_size(vllm: Mapping[str, Any], field: str) -> int:
    """Return one strictly positive vLLM parallel size."""
    value = vllm.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"rollout.vllm.{field} must be a positive integer")
    return value


def _trainer_topology(accelerator: Mapping[str, Any]) -> dict[str, int]:
    """Return the validated Trainer topology used by Hyper-RL orchestration."""
    topology = {
        "dp_replicate": int(accelerator.get("dp_replicate", 1)),
        "dp_shard": int(accelerator.get("dp_shard", 0)),
        "tp": int(accelerator.get("tp", 1)),
        "cp": int(accelerator.get("cp", 1)),
        "pp": int(accelerator.get("pp", 1)),
    }
    non_positive = {name: size for name, size in topology.items() if size <= 0}
    if non_positive:
        raise ValueError(
            "Trainer parallel sizes must be positive integers, "
            f"got {non_positive}"
        )
    unsupported = {
        name: size
        for name, size in topology.items()
        if (name == "dp_replicate" and size != 1)
        or (name == "tp" and size not in (1, 2))
        or (name in ("cp", "pp") and size != 1)
    }
    if unsupported:
        raise ValueError(
            "Hyper-RL Trainer currently supports dp_replicate=1, TP1/TP2, "
            f"and CP=PP=1; invalid topology={unsupported}"
        )
    return topology


def _trainer_world_size(accelerator: Mapping[str, Any]) -> int:
    """Return the physical process count for one supported Trainer topology."""
    topology = _trainer_topology(accelerator)
    return math.prod(topology.values())


def _trainer_data_parallel_size(accelerator: Mapping[str, Any]) -> int:
    """Return the logical Trainer data-parallel degree excluding model axes."""
    topology = _trainer_topology(accelerator)
    return topology["dp_replicate"] * topology["dp_shard"]


def _validate_vllm_port(vllm: Mapping[str, Any]) -> int:
    """Return the explicit port shared by all rollout clients."""
    port = vllm.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 < port <= 65535:
        raise ValueError(
            "Shared rollout requires rollout.vllm.port to be an explicit integer between 1 and 65535"
        )
    return port


def _reject_removed_vllm_topology(vllm: Mapping[str, Any]) -> None:
    """Reject the removed user-facing rollout topology switch."""
    if "topology" in vllm:
        raise ValueError(
            "rollout.vllm.topology was removed; configure deployment, "
            "data_parallel_size, and tensor_parallel_size instead"
        )


def _validate_vllm_basics(vllm: Mapping[str, Any]) -> tuple[str, int, int]:
    """Validate settings shared by colocated and disjoint vLLM."""
    _reject_removed_vllm_topology(vllm)
    deployment = str(vllm.get("deployment", "disjoint"))
    if deployment not in ("disjoint", "colocated"):
        raise ValueError(
            "rollout.vllm.deployment must be 'disjoint' or 'colocated', "
            f"got {deployment!r}"
        )
    rollout_dp = _parallel_size(vllm, "data_parallel_size")
    rollout_tp = _parallel_size(vllm, "tensor_parallel_size")
    if str(vllm.get("dtype", "bfloat16")) not in ("bfloat16", "bf16"):
        raise ValueError("The Qwen vLLM rollout path requires bfloat16")
    if str(vllm.get("host", "127.0.0.1")) not in ("127.0.0.1", "localhost"):
        raise ValueError("The external vLLM server must bind to loopback")
    _validate_vllm_port(vllm)
    utilization = float(vllm.get("gpu_memory_utilization", 0.9))
    if not 0 < utilization < 1:
        raise ValueError("rollout.vllm.gpu_memory_utilization must be between 0 and 1")
    return deployment, rollout_dp, rollout_tp


def _validate_colocated_vllm(
    vllm: Mapping[str, Any],
    accelerator: Mapping[str, Any],
    rollout_tp: int,
) -> None:
    """Validate colocated rollout topology and residency requirements."""
    trainer_world_size = _trainer_world_size(accelerator)
    if trainer_world_size <= 1:
        raise ValueError(
            "Colocated rollout requires a multi-rank Trainer topology"
        )
    if rollout_tp > trainer_world_size or trainer_world_size % rollout_tp != 0:
        raise ValueError(
            "Colocated rollout requires the Trainer world size divisible by "
            "rollout.vllm.tensor_parallel_size: "
            f"trainer_world_size={trainer_world_size}, rollout_tp={rollout_tp}"
        )
    if not bool(accelerator.get("cpu_offload", False)):
        raise ValueError("Colocated rollout requires train.accelerator.cpu_offload=true")
    if not bool(accelerator.get("reshard_after_forward", True)):
        raise ValueError(
            "Colocated rollout requires train.accelerator.reshard_after_forward=true"
        )
    if "visible_devices" in vllm:
        raise ValueError(
            "Colocated rollout derives its physical NPUs from the Trainer; "
            "remove rollout.vllm.visible_devices"
        )
    _validate_vllm_port(vllm)
    rollout_dp = _parallel_size(vllm, "data_parallel_size")
    if rollout_dp * rollout_tp != trainer_world_size:
        raise ValueError(
            "Colocated rollout devices must match the Trainer world: "
            f"data_parallel_size={rollout_dp}, tensor_parallel_size={rollout_tp}, "
            f"trainer_world_size={trainer_world_size}"
        )


def _training_device_id() -> str:
    """Resolve the physical device used by this trainer rank."""
    training_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not training_devices:
        return str(local_rank)
    trainer_ids = [device.strip() for device in training_devices.split(",")]
    if local_rank >= len(trainer_ids):
        raise ValueError(
            f"LOCAL_RANK={local_rank} exceeds "
            f"ASCEND_RT_VISIBLE_DEVICES={training_devices!r}"
        )
    return trainer_ids[local_rank]


def _validate_disjoint_vllm(
    vllm: Mapping[str, Any],
    accelerator: Mapping[str, Any],
    rollout_tp: int,
) -> None:
    """Validate disjoint trainer and rollout device ownership."""
    trainer_count = _trainer_world_size(accelerator)
    rollout_dp = _parallel_size(vllm, "data_parallel_size")
    _validate_vllm_port(vllm)
    visible_devices = vllm.get("visible_devices")
    if visible_devices is None:
        raise ValueError("rollout.vllm.visible_devices must select the external server NPUs")
    device_ids = [device.strip() for device in str(visible_devices).split(",")]
    expected_devices = rollout_dp * rollout_tp
    if not all(device_ids) or len(device_ids) != expected_devices:
        raise ValueError(
            "Disjoint rollout requires data_parallel_size * tensor_parallel_size devices: "
            f"expected {expected_devices} rollout devices, "
            f"got {device_ids}"
        )
    if len(set(device_ids)) != len(device_ids):
        raise ValueError(f"Disjoint rollout devices must be unique, got {device_ids}")
    training_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if training_devices is None:
        trainer_device_ids = [_training_device_id()]
    else:
        trainer_device_ids = [
            device.strip() for device in training_devices.split(",")[:trainer_count]
        ]
    overlap = sorted(set(trainer_device_ids) & set(device_ids))
    if overlap:
        raise ValueError(
            "The external vLLM server must use NPUs disjoint from the trainer: "
            f"training_devices={trainer_device_ids}, rollout_devices={device_ids}, "
            f"overlap={overlap}"
        )


def _validate_vllm_limits(vllm: Mapping[str, Any]) -> None:
    """Validate optional vLLM concurrency and capacity limits."""
    normalize_model_implementation(vllm.get("model_implementation", "native"))
    if "request_concurrency" in vllm:
        raise ValueError(
            "rollout.vllm.request_concurrency was replaced by automatic child admission "
            "derived from max_num_seqs"
        )
    if "api_server_count" in vllm:
        raise ValueError(
            "rollout.vllm.api_server_count is controlled by vLLM upstream and must be removed"
        )
    if "max_num_seqs" not in vllm:
        raise ValueError(
            "rollout.vllm.max_num_seqs is required for automatic child admission"
        )
    for field in (
        "kv_cache_memory_bytes",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
    ):
        if field in vllm and int(vllm[field]) <= 0:
            raise ValueError(f"rollout.vllm.{field} must be positive")
    if (
        vllm.get("max_num_seqs") is not None
        and vllm.get("max_num_batched_tokens") is not None
        and int(vllm["max_num_seqs"]) > int(vllm["max_num_batched_tokens"])
    ):
        raise ValueError(
            "rollout.vllm.max_num_seqs cannot exceed max_num_batched_tokens"
        )
    server_hccl_base_port = vllm.get("server_hccl_if_base_port")
    server_hccl_port_range = vllm.get("server_hccl_npu_socket_port_range")
    if server_hccl_base_port is None and server_hccl_port_range is None:
        return
    if server_hccl_base_port is None or server_hccl_port_range is None:
        raise ValueError(
            "rollout.vllm.server_hccl_if_base_port and "
            "server_hccl_npu_socket_port_range must be configured together"
        )
    if (
        not isinstance(server_hccl_base_port, int)
        or isinstance(server_hccl_base_port, bool)
        or not _HCCL_MIN_PORT <= server_hccl_base_port <= _HCCL_MAX_PORT
    ):
        raise ValueError(
            "rollout.vllm.server_hccl_if_base_port must be an integer in "
            f"[{_HCCL_MIN_PORT}, {_HCCL_MAX_PORT}]"
        )
    try:
        range_start_text, range_end_text = str(server_hccl_port_range).split("-", maxsplit=1)
        range_start = int(range_start_text)
        range_end = int(range_end_text)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "rollout.vllm.server_hccl_npu_socket_port_range must use START-END"
        ) from error
    if not (
        _HCCL_MIN_PORT
        <= range_start
        <= server_hccl_base_port
        <= range_end
        <= _HCCL_MAX_PORT
    ):
        raise ValueError(
            "rollout.vllm server HCCL socket ports must be in "
            f"[{_HCCL_MIN_PORT}, {_HCCL_MAX_PORT}] and contain the base port"
        )


def _weight_sync_parallel_sizes(accelerator: Mapping[str, Any]) -> dict[str, int]:
    """Validate the pure-FSDP topology required for weight synchronization."""
    parallel_sizes = {
        field: int(accelerator.get(field, 1))
        for field in ("dp_replicate", "tp", "cp", "pp")
    }
    non_tp_sizes = {
        field: size for field, size in parallel_sizes.items() if field != "tp"
    }
    if any(size != 1 for size in non_tp_sizes.values()):
        raise ValueError(
            "Weight synchronization currently requires pure FSDP training with "
            "dp_replicate=1 and train CP/PP=1, got "
            f"{parallel_sizes}"
        )
    return parallel_sizes


def _trainer_tp2_weight_sync_supported(
    trainer_tp: int,
    rollout_model: VLLMModelRegistration,
    effective_strategy: str,
    fallback_strategy: str,
) -> bool:
    """Return whether the configured trainer TP2 sync path is supported."""
    fallback_supported = fallback_strategy == "none" or (
        effective_strategy == "direct_reshard"
        and fallback_strategy == "full_gather"
    )
    return (
        trainer_tp == 2
        and rollout_model.family == "qwen3"
        and effective_strategy in ("full_gather", "direct_reshard")
        and fallback_supported
    )


def _validate_vllm_weight_sync(
    vllm: Mapping[str, Any],
    deployment: str,
    rollout_model: VLLMModelRegistration,
    accelerator: Mapping[str, Any],
) -> None:
    """Validate full-weight DP sync and TP-aware direct-reshard recovery."""
    weight_sync = optional_mapping(vllm, "weight_sync")
    strategy = str(weight_sync.get("strategy", "full_gather"))
    if strategy not in ("direct_reshard", "full_gather"):
        raise ValueError(
            "rollout.vllm.weight_sync.strategy must be 'direct_reshard' or "
            "'full_gather', "
            f"got {strategy!r}"
        )
    fallback_strategy = str(
        weight_sync.get(
            "fallback_strategy",
            "full_gather" if strategy == "direct_reshard" else "none",
        )
    )
    if fallback_strategy not in ("full_gather", "none"):
        raise ValueError(
            "rollout.vllm.weight_sync.fallback_strategy must be 'full_gather' "
            f"or 'none', got {fallback_strategy!r}"
        )
    bucket_size_mb = int(weight_sync.get("bucket_size_mb", 128))
    if bucket_size_mb <= 0:
        raise ValueError("rollout.vllm.weight_sync.bucket_size_mb must be positive")
    if deployment not in ("colocated", "disjoint"):
        raise ValueError(f"Unsupported direct-reshard deployment: {deployment!r}")
    rollout_tp = int(vllm.get("tensor_parallel_size", 1))
    effective_strategy = "full_gather" if rollout_tp == 1 else strategy
    if effective_strategy == "direct_reshard" and rollout_model.family != "qwen3":
        raise ValueError(
            "Direct reshard currently supports Qwen3 rollout models only; "
            f"got family={rollout_model.family!r}"
        )
    parallel_sizes = _weight_sync_parallel_sizes(accelerator)
    trainer_tp = parallel_sizes["tp"]
    if trainer_tp == 1:
        return
    trainer_tp2_supported = _trainer_tp2_weight_sync_supported(
        trainer_tp,
        rollout_model,
        effective_strategy,
        fallback_strategy,
    )
    if not trainer_tp2_supported:
        raise ValueError(
            "Trainer TP2 weight synchronization currently requires Qwen3 with "
            "full-gather or direct-reshard weight sync; "
            f"got deployment={deployment!r}, rollout_tp={rollout_tp}, "
            f"family={rollout_model.family!r}, is_hyper={rollout_model.is_hyper!r}, "
            f"strategy={effective_strategy!r}, fallback={fallback_strategy!r}"
        )


def _validate_model_implementation(
    vllm: Mapping[str, Any],
    model_registration: ModelRegistration,
) -> VLLMModelRegistration:
    """Validate native versus Hyper rollout support for one checkpoint family."""
    return resolve_vllm_model(
        model_registration,
        vllm.get("model_implementation", "native"),
    )


def _validate_vllm(
    rollout: Mapping[str, Any],
    accelerator: Mapping[str, Any],
    model_registration: ModelRegistration,
) -> None:
    """Validate the configured vLLM-Ascend deployment."""
    if platform.platform_type != PlatformType.PYTORCH or platform.device_type() != "npu":
        raise ValueError("The vLLM-Ascend rollout backend requires the Torch NPU platform")
    vllm = optional_mapping(rollout, "vllm")
    deployment, _, rollout_tp = _validate_vllm_basics(vllm)
    if deployment == "colocated":
        _validate_colocated_vllm(vllm, accelerator, rollout_tp)
    else:
        _validate_disjoint_vllm(vllm, accelerator, rollout_tp)
    _validate_vllm_limits(vllm)
    rollout_model = _validate_model_implementation(vllm, model_registration)
    _validate_vllm_weight_sync(vllm, deployment, rollout_model, accelerator)
    if rollout_model.is_hyper and rollout_model.family != "qwen3" and rollout_tp != 1:
        raise ValueError(
            "Hyper-vLLM tensor parallelism currently supports Qwen3 only; "
            f"family={rollout_model.family!r}, tensor_parallel_size={rollout_tp}"
        )


def _validate_agentic(agentic: Mapping[str, Any]) -> None:
    """Validate the selected agent environment and turn limits."""
    module_path = agentic.get("module_path")
    if module_path is not None:
        load_agentic_module(module_path)
    environment_name = agentic.get("environment")
    if environment_name not in ENVIRONMENTS.names:
        raise ValueError(
            f"Unknown agentic.environment '{environment_name}'; "
            f"available={ENVIRONMENTS.names}"
        )
    if int(agentic.get("max_turns", 0)) <= 0:
        raise ValueError("agentic.max_turns must be positive")
    if int(agentic.get("max_observation_tokens", 0)) < 0:
        raise ValueError("agentic.max_observation_tokens must be non-negative")
    mode = InteractionMode.parse(
        agentic.get(
            "interaction_mode",
            "single_turn" if int(agentic["max_turns"]) == 1 else "multi_turn",
        )
    )
    if mode is InteractionMode.SINGLE_TURN and int(agentic["max_turns"]) != 1:
        raise ValueError("single_turn interaction requires agentic.max_turns=1")
    max_episode_tokens = agentic.get("max_episode_tokens")
    if max_episode_tokens is not None and int(max_episode_tokens) <= 0:
        raise ValueError("agentic.max_episode_tokens must be positive when configured")
    if not isinstance(agentic.get("apply_chat_template", False), bool):
        raise ValueError("agentic.apply_chat_template must be a boolean")


def validate_rollout_and_agentic(
    rollout: Mapping[str, Any],
    agentic: Mapping[str, Any],
    accelerator: Mapping[str, Any],
    model_registration: ModelRegistration,
) -> None:
    """Validate rollout deployment and environment settings."""
    engine_name = rollout.get("engine")
    if engine_name not in ROLLOUT_ENGINES.names:
        raise ValueError(
            f"Unknown rollout.engine '{engine_name}'; available={ROLLOUT_ENGINES.names}"
    )
    if engine_name == "vllm":
        _validate_vllm(rollout, accelerator, model_registration)
    seed = rollout.get("seed")
    if seed is not None and int(seed) < 0:
        raise ValueError("rollout.seed must be non-negative or null")
    _validate_agentic(agentic)


def _validate_topology(accelerator: Mapping[str, Any]) -> None:
    """Validate the currently accepted Trainer FSDP and TP topology."""
    _trainer_topology(accelerator)


def _validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Validate checkpoint options against available role support."""
    save_final = bool(checkpoint.get("save_final", True))
    verify_reload = bool(checkpoint.get("verify_reload", False))
    save_steps = int(checkpoint.get("save_steps", 0))
    if verify_reload and not save_final:
        raise ValueError("checkpoint.verify_reload requires checkpoint.save_final=true")
    if save_steps < 0:
        raise ValueError("checkpoint.save_steps must be non-negative")
    _path_value(checkpoint, "output_dir")


def _validate_logging(config: Mapping[str, Any]) -> None:
    backends = config.get("backends", ())
    if not isinstance(backends, list) or not backends:
        raise ValueError("logging.backends must be a non-empty list")
    unsupported = set(backends) - {"console", "wandb"}
    if unsupported:
        raise ValueError(f"Unsupported logging backends: {sorted(unsupported)}")
    wandb = required_mapping(config, "wandb")
    if wandb.get("mode", "auto") not in {"auto", "online", "offline", "disabled"}:
        raise ValueError(f"Unsupported W&B mode: {wandb.get('mode')}")


def validate_config(config: Mapping[str, Any], algorithm: RLAlgorithm) -> None:
    """Validate Hyper-RL configuration before distributed startup."""
    unknown = set(config) - _EXPECTED_TOP_LEVEL
    if unknown:
        raise ValueError(f"Unsupported top-level configuration keys: {sorted(unknown)}")
    consistency_profile(config)
    if algorithm.requirements.roles.critic:
        raise NotImplementedError(
            "The initial HyperAutoModel RL runtime supports critic-free algorithms only; "
            f"algorithm '{algorithm.name}' requires a Critic"
        )
    model = required_mapping(config, "model")
    data = required_mapping(config, "data")
    rollout = required_mapping(config, "rollout")
    agentic = required_mapping(config, "agentic")
    evaluation = required_mapping(config, "evaluation")
    train = required_mapping(config, "train")
    accelerator = required_mapping(train, "accelerator")
    model_registration = build_model_registration(config)
    validate_consistency_model_identity(config, model_registration)
    _validate_model_and_data_paths(model, data, evaluation)
    _validate_training_sizes(train, rollout, data, algorithm)
    _validate_evaluation(evaluation)
    validate_rollout_and_agentic(rollout, agentic, accelerator, model_registration)
    _validate_topology(accelerator)
    _validate_checkpoint(required_mapping(train, "checkpoint"))
    _validate_logging(required_mapping(config, "logging"))


def _load_automatic_limit_text_config(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load and validate the model metadata used for automatic vLLM limits."""
    model_path = Path(str(model["weights_path"])) / "config.json"
    try:
        with model_path.open(encoding="utf-8") as config_file:
            model_config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Automatic max_num_seqs requires a readable model config: {model_path}"
        ) from error
    text_config = model_config.get("text_config", model_config)
    layer_types = text_config.get("layer_types") or []
    if text_config.get("model_type") != "qwen3" or any(
        layer_type != "full_attention" for layer_type in layer_types
    ):
        raise ValueError(
            "Automatic max_num_seqs currently supports dense Qwen3 "
            "full-attention models only"
        )
    return text_config


def _positive_integer(section: Mapping[str, Any], field: str, prefix: str) -> int:
    """Read one strictly positive, non-Boolean integer configuration value."""
    value = section.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{prefix}.{field} must be a positive integer")
    return value


def _validate_automatic_limit_inputs(
    data: Mapping[str, Any],
    rollout: Mapping[str, Any],
    agentic: Mapping[str, Any],
    train: Mapping[str, Any],
    accelerator: Mapping[str, Any],
    vllm: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Validate automatic-limit inputs and return DP, TP, and observation limits."""
    required_vllm_fields = (
        "kv_cache_memory_bytes",
        "max_model_len",
        "max_num_batched_tokens",
        "block_size",
    )
    missing = [field for field in required_vllm_fields if vllm.get(field) is None]
    if missing:
        raise ValueError(
            f"Automatic max_num_seqs requires explicit rollout.vllm fields: {missing}"
        )
    positive_fields = (
        (data, "max_prompt_length", "data"),
        (rollout, "max_new_tokens", "rollout"),
        (rollout, "num_return_sequences", "rollout"),
        (agentic, "max_turns", "agentic"),
        (train, "prompt_batch_size", "train"),
        (accelerator, "dp_shard", "train.accelerator"),
        (vllm, "kv_cache_memory_bytes", "rollout.vllm"),
        (vllm, "max_model_len", "rollout.vllm"),
        (vllm, "max_num_batched_tokens", "rollout.vllm"),
        (vllm, "block_size", "rollout.vllm"),
    )
    for section, field, prefix in positive_fields:
        _positive_integer(section, field, prefix)
    data_parallel_size = _positive_integer(
        vllm, "data_parallel_size", "rollout.vllm"
    )
    tensor_parallel_size = _positive_integer(
        vllm, "tensor_parallel_size", "rollout.vllm"
    )
    max_observation_tokens = agentic.get("max_observation_tokens", 0)
    if (
        not isinstance(max_observation_tokens, int)
        or isinstance(max_observation_tokens, bool)
        or max_observation_tokens < 0
    ):
        raise ValueError("agentic.max_observation_tokens must be a non-negative integer")
    return data_parallel_size, tensor_parallel_size, max_observation_tokens


def _automatic_context_tokens(
    data: Mapping[str, Any],
    rollout: Mapping[str, Any],
    agentic: Mapping[str, Any],
    vllm: Mapping[str, Any],
    max_observation_tokens: int,
) -> int:
    """Calculate and validate the maximum per-sequence context length."""
    max_turns = int(agentic["max_turns"])
    context_tokens = (
        int(data["max_prompt_length"])
        + max_turns * int(rollout["max_new_tokens"])
        + (max_turns - 1) * max_observation_tokens
    )
    max_model_len = int(vllm["max_model_len"])
    if context_tokens > max_model_len:
        raise ValueError(
            "Automatic max_num_seqs requires workload context within max_model_len: "
            f"context_tokens={context_tokens}, max_model_len={max_model_len}"
        )
    return context_tokens


def _automatic_kv_capacity(
    text_config: Mapping[str, Any],
    vllm: Mapping[str, Any],
    tensor_parallel_size: int,
    context_tokens: int,
    dtype_bytes: int,
) -> int:
    """Calculate how many maximum-length sequences fit in the KV cache."""
    num_kv_heads = int(text_config["num_key_value_heads"])
    if num_kv_heads % tensor_parallel_size != 0:
        raise ValueError(
            "Automatic max_num_seqs requires num_key_value_heads divisible by "
            f"tensor_parallel_size: {num_kv_heads} % {tensor_parallel_size} != 0"
        )
    kv_heads_per_rank = num_kv_heads // tensor_parallel_size
    head_dim = int(
        text_config.get(
            "head_dim",
            int(text_config["hidden_size"])
            // int(text_config["num_attention_heads"]),
        )
    )
    block_size = int(vllm["block_size"])
    block_bytes = (
        int(text_config["num_hidden_layers"])
        * block_size
        * 2
        * kv_heads_per_rank
        * head_dim
        * dtype_bytes
    )
    pool_blocks = int(vllm["kv_cache_memory_bytes"]) // block_bytes
    blocks_per_sequence = math.ceil(context_tokens / block_size)
    kv_capacity = pool_blocks // blocks_per_sequence
    if kv_capacity <= 0:
        raise ValueError(
            "Automatic max_num_seqs found insufficient KV cache for one "
            "maximum-length sequence"
        )
    return kv_capacity


def resolve_vllm_automatic_limits(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve workload- and KV-bounded vLLM limits before validation.

    Args:
        config: Fully merged Hyper-RL configuration.

    Returns:
        A detached configuration with ``max_num_seqs`` resolved when it is null.

    Raises:
        ValueError: If automatic capacity lacks an explicit resource bound or the
            model uses an unsupported hybrid KV layout.
    """
    resolved = deepcopy(dict(config))
    rollout = resolved.get("rollout")
    if not isinstance(rollout, dict) or rollout.get("engine") != "vllm":
        return resolved
    vllm = rollout.get("vllm")
    if isinstance(vllm, Mapping):
        _reject_removed_vllm_topology(vllm)
    if (
        not isinstance(vllm, dict)
        or "max_num_seqs" not in vllm
        or vllm["max_num_seqs"] is not None
    ):
        return resolved
    model = required_mapping(resolved, "model")
    data = required_mapping(resolved, "data")
    agentic = required_mapping(resolved, "agentic")
    train = required_mapping(resolved, "train")
    accelerator = required_mapping(train, "accelerator")
    text_config = _load_automatic_limit_text_config(model)
    data_parallel_size, tensor_parallel_size, max_observation_tokens = (
        _validate_automatic_limit_inputs(
            data, rollout, agentic, train, accelerator, vllm
        )
    )
    dtype = str(vllm.get("dtype", "bfloat16"))
    dtype_bytes = {"bfloat16": 2, "bf16": 2}.get(dtype)
    if dtype_bytes is None:
        raise ValueError(f"Automatic max_num_seqs does not support dtype {dtype!r}")
    context_tokens = _automatic_context_tokens(
        data, rollout, agentic, vllm, max_observation_tokens
    )
    kv_capacity = _automatic_kv_capacity(
        text_config,
        vllm,
        tensor_parallel_size,
        context_tokens,
        dtype_bytes,
    )
    trainer_dp_size = _trainer_data_parallel_size(accelerator)
    global_children = (
        trainer_dp_size
        * int(train.get("prompt_batch_size", 1))
        * int(rollout["num_return_sequences"])
    )
    workload_capacity = math.ceil(global_children / data_parallel_size)
    max_num_batched_tokens = int(vllm["max_num_batched_tokens"])
    vllm["max_num_seqs"] = min(
        workload_capacity,
        kv_capacity,
        max_num_batched_tokens,
    )
    return resolved


def build_model_registration(config: Mapping[str, Any]) -> ModelRegistration:
    """Resolve the configured model shared by training and rollout."""
    model = required_mapping(config, "model")
    name = model.get("registry_name")
    if not isinstance(name, str) or not name:
        raise ValueError("model.registry_name must be a non-empty string")
    config_path = Path(str(model["weights_path"])) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Model config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        hf_config = json.load(config_file)
    architectures = hf_config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError(
            f"Model config must define exactly one architecture, got {architectures!r}"
        )
    text_config = hf_config.get("text_config", hf_config)
    if not isinstance(text_config, Mapping):
        raise ValueError("Model text_config must be a mapping when present")
    return ModelRegistration(
        name=name,
        hyper_model_name=str(model["name"]),
        weights_path=str(model["weights_path"]),
        tokenizer_path=str(model["tokenizer_path"]),
        hf_architecture=str(architectures[0]),
        model_type=str(hf_config.get("model_type", "")),
        text_model_type=str(text_config.get("model_type", hf_config.get("model_type", ""))),
        tie_word_embeddings=bool(text_config.get("tie_word_embeddings", False)),
    )


def _normalize_dtype_name(name: Any) -> Optional[str]:
    """Normalize a configured mixed-precision dtype for the master FSDP2 API."""
    normalized = None if name is None else str(name).strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
        "fp16": "float16",
        "float16": "float16",
    }
    canonical_name = aliases.get(normalized)
    if normalized is not None and canonical_name is None:
        raise ValueError(f"Unsupported mixed-precision dtype: {name!r}")
    return canonical_name


def _build_optimizer_target(optimizer_config: Mapping[str, Any]) -> Target:
    """Build the HyperAutoModel AdamW target."""
    optimizer_kwargs = {
        "adamw_lr": float(optimizer_config.get("lr", 1.0e-6)),
        "adamw_weight_decay": float(optimizer_config.get("weight_decay", 0.0)),
        "adamw_betas": tuple(optimizer_config.get("betas", (0.9, 0.999))),
        "adamw_eps": float(optimizer_config.get("eps", 1.0e-8)),
    }
    if optimizer_config.get("foreach") is not None:
        optimizer_kwargs["foreach"] = bool(optimizer_config["foreach"])
    return Target(
        AdamW,
        target_path="hyper_parallel.auto_models.components.optim.optimizer.AdamW",
        adamw_config=optimizer_kwargs,
        no_decay_params=["bias", "norm", "ln_"],
    )


def _activation_checkpoint_mode(accelerator_config: Mapping[str, Any]) -> str:
    """Normalize the activation-checkpoint mode."""
    configured_value = accelerator_config.get("activation_checkpoint", "off")
    if isinstance(configured_value, bool):
        mode = "full" if configured_value else "off"
    else:
        mode = str(configured_value).lower()
    if mode not in ("off", "full", "selective"):
        raise ValueError(
            "train.accelerator.activation_checkpoint must be off, full, selective, "
            f"or a Boolean, got {configured_value!r}"
        )
    return mode


def _build_model_target(
    model_config: Mapping[str, Any],
    param_dtype_name: str,
    mixed_precision_enabled: bool,
) -> Target:
    """Build the pretrained causal-language-model target."""
    return Target(
        HyperAutoModelForCausalLM.from_pretrained,
        target_path=(
            "hyper_parallel.auto_models._transformers."
            "HyperAutoModelForCausalLM.from_pretrained"
        ),
        pretrained_model_name_or_path=str(model_config["weights_path"]),
        torch_dtype=param_dtype_name if mixed_precision_enabled else "float32",
        attn_implementation=str(model_config.get("attn_implementation", "sdpa")),
        force_hf=True,
        local_files_only=True,
        trust_remote_code=True,
    )


def _build_lr_scheduler_target(optimizer_config: Mapping[str, Any]) -> Target:
    """Build the configured learning-rate scheduler target."""
    return Target(
        MultiLRScheduler,
        target_path=(
            "hyper_parallel.auto_models.components.optim.lr_scheduler.MultiLRScheduler"
        ),
        lr_decay_style=str(optimizer_config.get("lr_decay_style", "constant")),
        lr_config={
            "lr_warmup_ratio": float(optimizer_config.get("lr_warmup_ratio", 0.0)),
            "min_lr": float(optimizer_config.get("lr_min", 0.0)),
        },
    )


def _build_training_config(
    train_config: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
    trainer_dp_size: int,
    prompt_batch_size: int,
    backend: str,
) -> TrainingConfig:
    """Build core training-loop settings."""
    return TrainingConfig(
        train_iters=int(train_config["max_steps"]),
        global_batch_size=trainer_dp_size * prompt_batch_size,
        micro_batch_size=prompt_batch_size,
        backend=backend,
        max_grad_norm=float(optimizer_config.get("max_grad_norm", 1.0)),
        init_device=str(train_config.get("init_device", "meta")),
        loss_aggregation="token_weighted",
        seed=int(train_config.get("seed", 1234)),
    )


def _build_accelerator_config(
    accelerator_config: Mapping[str, Any],
) -> AcceleratorConfig:
    """Build the supported parallel-dimension configuration."""
    return AcceleratorConfig(
        tp_size=int(accelerator_config.get("tp", 1)),
        cp_size=int(accelerator_config.get("cp", 1)),
        ep_size=1,
        pp_size=int(accelerator_config.get("pp", 1)),
        sequence_parallel=False,
        loss_parallel=False,
    )


def _build_fsdp_config(
    accelerator_config: Mapping[str, Any],
    mixed_precision_config: Mapping[str, Any],
    dp_shard: int,
) -> FSDP2Config:
    """Build FSDP, mixed-precision, and CPU-offload settings."""
    mixed_precision_enabled = bool(mixed_precision_config.get("enabled", True))
    return FSDP2Config(
        dp_shard_size=dp_shard,
        mix_precision=FSDP2MixedPrecisionConfig(
            param_dtype=(
                _normalize_dtype_name(
                    mixed_precision_config.get("param_dtype", "bfloat16")
                )
                if mixed_precision_enabled
                else None
            ),
            reduce_dtype=(
                _normalize_dtype_name(
                    mixed_precision_config.get("reduce_dtype", "float32")
                )
                if mixed_precision_enabled
                else None
            ),
            output_dtype=(
                _normalize_dtype_name(mixed_precision_config.get("output_dtype"))
                if mixed_precision_enabled
                else None
            ),
        ),
        enable_offload=bool(accelerator_config.get("cpu_offload", False)),
        reshard_after_forward=bool(
            accelerator_config.get("reshard_after_forward", True)
        ),
        reshard_after_backward=False,
        requires_grad_sync=True,
        comm_fusion=bool(accelerator_config.get("comm_fusion", True)),
    )


def _build_checkpoint_config(
    checkpoint_config: Mapping[str, Any],
) -> CheckpointingConfig:
    """Build checkpoint save settings."""
    return CheckpointingConfig(
        save_ckpt=(
            bool(checkpoint_config.get("save_final", True))
            or int(checkpoint_config.get("save_steps", 0)) > 0
        ),
        checkpoint_dir=str(checkpoint_config["output_dir"]),
        save_consolidated="none",
    )


def build_runtime_config(config: Mapping[str, Any]) -> TrainerConfig:
    """Translate Hyper-RL YAML into the HyperAutoModel runtime configuration."""
    model_config = required_mapping(config, "model")
    train_config = required_mapping(config, "train")
    accelerator_config = required_mapping(train_config, "accelerator")
    optimizer_config = required_mapping(train_config, "optimizer")
    mixed_precision_config = required_mapping(train_config, "mixed_precision")
    checkpoint_config = required_mapping(train_config, "checkpoint")
    if model_config.get("config_overrides") not in (None, {}):
        raise ValueError("model.config_overrides is not supported by HyperAutoModel")

    prompt_batch_size = int(train_config.get("prompt_batch_size", 1))
    topology = _trainer_topology(accelerator_config)
    dp_shard = topology["dp_shard"]
    trainer_dp_size = topology["dp_replicate"] * topology["dp_shard"]
    cpu_offload = bool(accelerator_config.get("cpu_offload", False))
    param_dtype_name = str(mixed_precision_config.get("param_dtype", "bfloat16"))
    mixed_precision_enabled = bool(mixed_precision_config.get("enabled", True))
    backend = str(train_config.get("comm_backend") or "hccl")
    if cpu_offload and ":" not in backend:
        backend = f"cpu:gloo,{platform.device_type()}:{backend}"

    return TrainerConfig(
        model=_build_model_target(
            model_config, param_dtype_name, mixed_precision_enabled
        ),
        optimizer=_build_optimizer_target(optimizer_config),
        lr_scheduler=_build_lr_scheduler_target(optimizer_config),
        training=_build_training_config(
            train_config,
            optimizer_config,
            trainer_dp_size,
            prompt_batch_size,
            backend,
        ),
        accelerator=_build_accelerator_config(accelerator_config),
        fsdp_config=_build_fsdp_config(
            accelerator_config, mixed_precision_config, dp_shard
        ),
        mixed_precision=MixedPrecisionConfig(enabled=mixed_precision_enabled),
        activation_checkpoint=ActivationCheckpointConfig(
            mode=_activation_checkpoint_mode(accelerator_config)
        ),
        checkpoint=_build_checkpoint_config(checkpoint_config),
    )


__all__ = [
    "build_model_registration",
    "build_runtime_config",
    "optional_mapping",
    "required_mapping",
    "resolve_vllm_automatic_limits",
    "uses_colocated_vllm",
    "validate_config",
    "validate_rollout_and_agentic",
]
