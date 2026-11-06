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
"""RL configuration validate."""
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional
import torch
from rl.agentic.core.types import InteractionMode
from rl.agentic.envs.environment import ENVIRONMENTS, load_agentic_module
from rl.algorithm.loss import RLAlgorithm
from rl.distillation.config import parse_distillation_config
from rl.data.prompts import validate_chat_template_kwargs, validate_row_adapter_config
from rl.workers.engine.model import ModelRegistration, resolve_model, resolve_vllm_model
from rl.reward_model.scoring import load_reward_function
from rl.rollout.registry import ROLLOUT_ENGINES
from rl.weight_sync.config import resolve_weight_sync_config
from rl.config.schema import AsyncTrainingConfig, ColocateAsyncConfig, _EXPECTED_TOP_LEVEL, _HCCL_MAX_PORT, _HCCL_MIN_PORT, _async_integer, _normalize_dtype_name, _optimizer_fp32_main_params, _positive_integer, _require_async, _trainer_topology, _trainer_world_size, build_model_registration, optional_mapping, required_mapping, router_replay_mode, uses_colocated_vllm


def _path_value(section: Mapping[str, Any], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration field '{name}' must be a non-empty path string")
    return value


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
    agentic: Optional[Mapping[str, Any]] = None,
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
    if agentic is not None and agentic.get("runner") in {"codex", "deepseek"}:
        local_responses *= int(agentic.get("max_turns", 1))
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
    gate_max_step = gate.get("max_step")
    valid_gate_step = isinstance(gate_max_step, int) and not isinstance(gate_max_step, bool)
    if gate_max_step is not None and (not valid_gate_step or gate_max_step <= 0):
        raise ValueError("train.learning_gate.max_step must be a positive integer or null")


def _validate_evaluation(evaluation: Mapping[str, Any]) -> None:
    """Validate optional evaluation limits."""
    if not isinstance(evaluation.get("before_train", False), bool):
        raise ValueError("evaluation.before_train must be a boolean")
    interval = evaluation.get("every_n_steps")
    if interval is not None and (isinstance(interval, bool) or not isinstance(interval, int) or interval < 0):
        raise ValueError("evaluation.every_n_steps must be a non-negative integer or null")
    if not isinstance(evaluation.get("at_end", False), bool):
        raise ValueError("evaluation.at_end must be a boolean")
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
    sleep_level = vllm.get("sleep_level", 1)
    if not isinstance(sleep_level, int) or isinstance(sleep_level, bool) or sleep_level not in (1, 2):
        raise ValueError("rollout.vllm.sleep_level must be an integer equal to 1 or 2")
    if sleep_level == 2 and deployment != "colocated":
        raise ValueError("rollout.vllm.sleep_level=2 requires deployment='colocated'")
    rollout_dp = _positive_integer(vllm, "data_parallel_size", "rollout.vllm")
    rollout_tp = _positive_integer(vllm, "tensor_parallel_size", "rollout.vllm")
    if str(vllm.get("dtype", "bfloat16")) not in ("bfloat16", "bf16"):
        raise ValueError("The vLLM rollout path requires bfloat16")
    if str(vllm.get("host", "127.0.0.1")) not in ("127.0.0.1", "localhost"):
        raise ValueError("The external vLLM server must bind to loopback")
    _ = _validate_vllm_port(vllm)
    utilization = float(vllm.get("gpu_memory_utilization", 0.9))
    if not 0 < utilization < 1:
        raise ValueError("rollout.vllm.gpu_memory_utilization must be between 0 and 1")
    return deployment, rollout_dp, rollout_tp


def _validate_colocated_vllm(
    vllm: Mapping[str, Any],
    accelerator: Mapping[str, Any],
    trainer_world_size: int,
    rollout_dp: int,
    rollout_tp: int,
) -> None:
    """Validate colocated rollout topology and residency requirements."""
    if trainer_world_size <= 1:
        raise ValueError(
            "Colocated rollout requires a multi-rank Trainer topology"
        )
    if not bool(accelerator.get("reshard_after_forward", True)):
        raise ValueError(
            "Colocated rollout requires train.accelerator.reshard_after_forward=true"
        )
    if "visible_devices" in vllm:
        raise ValueError(
            "Colocated rollout derives its physical NPUs from the Trainer; "
            "remove rollout.vllm.visible_devices"
        )
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
    trainer_count: int,
    rollout_dp: int,
    rollout_tp: int,
) -> None:
    """Validate disjoint trainer and rollout device ownership."""
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


def _validate_vllm_runtime_options(vllm: Mapping[str, Any]) -> None:
    """Validate seeds, queue mode and runtime boolean switches."""
    if "seed" in vllm:
        seed = vllm["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("rollout.vllm.seed must be a non-negative integer")
    if "task_queue_enable" in vllm:
        queue_mode = vllm["task_queue_enable"]
        if isinstance(queue_mode, bool) or not isinstance(queue_mode, int) or queue_mode not in (1, 2):
            raise ValueError("rollout.vllm.task_queue_enable must be 1 or 2")
    for name, default in (("trust_remote_code", True), ("disable_log_stats", False)):
        if not isinstance(vllm.get(name, default), bool):
            raise ValueError(f"rollout.vllm.{name} must be a boolean")


def _validate_vllm_limits(vllm: Mapping[str, Any]) -> None:
    """Validate optional vLLM concurrency and capacity limits."""
    _validate_vllm_runtime_options(vllm)
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
    _validate_vllm_hccl_ports(vllm)


def _validate_dense_parallelism(
    vllm: Mapping[str, Any],
    accelerator: Mapping[str, Any],
) -> None:
    """Reject expert parallelism for the supported Qwen3 dense model."""
    _validate_trainer_ep(accelerator)
    for name in ("enable_expert_parallel", "enable_eplb"):
        if vllm.get(name, False):
            raise ValueError(f"rollout.vllm.{name} is not supported for Qwen3 dense")


def _validate_trainer_ep(accelerator: Mapping[str, Any]) -> None:
    """Accept only inactive expert-parallel dimensions for dense Qwen3."""
    for name in ("ep", "edp_shard"):
        size = accelerator.get(name, 1)
        if not isinstance(size, int) or isinstance(size, bool) or size != 1:
            raise ValueError(f"Qwen3 dense requires train.accelerator.{name}=1")


def _validate_model_scope(config: Mapping[str, Any], model: ModelRegistration) -> None:
    """Validate checkpoint-specific RL boundaries before constructing roles."""
    _validate_router_replay(config, model)
    accelerator = required_mapping(required_mapping(config, "train"), "accelerator")
    if model.family == "qwen3":
        _validate_trainer_ep(accelerator)
        return
    if model.family == "qwen3_5":
        _validate_qwen3_5_scope(config, accelerator)
        return
    rollout = required_mapping(config, "rollout")
    vllm = optional_mapping(rollout, "vllm")
    if (rollout.get("engine") != "vllm" or vllm.get("deployment") != "colocated"
            or vllm.get("model_implementation", "native") != "native"):
        raise ValueError("Qwen3-MoE requires colocated native vLLM rollout")
    if required_mapping(config, "algorithm").get("name") != "grpo":
        raise ValueError("Qwen3-MoE currently supports GRPO only")
    for name in ("enable_expert_parallel", "enable_eplb"):
        if not isinstance(vllm.get(name, False), bool):
            raise ValueError(f"rollout.vllm.{name} must be a boolean")
    if vllm.get("enable_eplb", False):
        raise ValueError("Qwen3-MoE does not support enable_eplb")
    _validate_moe_parallelism(accelerator, model)


def _validate_router_replay(config: Mapping[str, Any], model: ModelRegistration) -> None:
    """Fail before model allocation for unsupported replay backends and runners."""
    enabled = router_replay_mode(config) == "R3"
    vllm = optional_mapping(required_mapping(config, "rollout"), "vllm")
    if ("enable_return_routed_experts" in vllm
            and vllm["enable_return_routed_experts"] is not enabled):
        raise ValueError("rollout.vllm.enable_return_routed_experts must match train.router_replay.mode")
    if not enabled:
        return
    if parse_distillation_config(config.get("distillation")).enabled:
        raise ValueError("R3 does not support enabled distillation")
    if model.family != "qwen3_moe":
        raise ValueError("R3 currently requires a Qwen3-MoE router adapter")
    agentic = required_mapping(config, "agentic")
    if agentic.get("runner", "internal") not in ("internal", "codex"):
        raise ValueError("R3 currently supports only internal and Codex runners")
    if vllm.get("data_parallel_size", 1) != 1:
        raise ValueError(
            "R3 requires rollout.vllm.data_parallel_size=1: router capture in fixed "
            "vLLM 0.23.0/vLLM-Ascend 0.23.0.post1 does not support DP padding"
        )
    if vllm.get("enforce_eager", True) is not True:
        raise ValueError("R3 currently requires rollout.vllm.enforce_eager=true")
    if vllm.get("logprobs_mode") != "raw_logprobs":
        raise ValueError("R3 requires rollout.vllm.logprobs_mode=raw_logprobs")


def _validate_qwen3_5_scope(config: Mapping[str, Any], accelerator: Mapping[str, Any]) -> None:
    """Limit Qwen3.5-family text policies to the supported FSDP/native rollout path."""
    rollout = required_mapping(config, "rollout")
    vllm = optional_mapping(rollout, "vllm")
    if (rollout.get("engine") != "vllm" or vllm.get("deployment") != "colocated"
            or vllm.get("model_implementation", "native") != "native"):
        raise ValueError("Qwen3.5/Qwen3.8 requires colocated native vLLM rollout")
    if required_mapping(config, "algorithm").get("name") != "grpo":
        raise ValueError("Qwen3.5/Qwen3.8 currently supports GRPO only")
    for name in ("tp", "cp", "pp", "dp_replicate", "ep", "edp_shard"):
        size = accelerator.get(name, 1)
        if not isinstance(size, int) or isinstance(size, bool) or size != 1:
            raise ValueError(f"Qwen3.5/Qwen3.8 requires train.accelerator.{name}=1")
    if vllm.get("enforce_eager", True) is not True:
        _validate_qwen3_5_graphs(vllm)
    for name in ("enable_expert_parallel", "enable_eplb", "batch_invariant"):
        if vllm.get(name, False) is not False:
            raise ValueError(f"Qwen3.5/Qwen3.8 requires rollout.vllm.{name}=false")
    if vllm.get("max_num_seqs") is None:
        raise ValueError("Qwen3.5/Qwen3.8 requires an explicit rollout.vllm.max_num_seqs")
    resolve_weight_sync_config(
        optional_mapping(vllm, "weight_sync"), deployment="colocated", model_family="qwen3_5",
        rollout_tp=int(vllm.get("tensor_parallel_size", 1)),
    )


def _validate_qwen3_5_graphs(vllm: Mapping[str, Any]) -> None:
    """Bound graph publication to explicit decode captures without cross-DP dispatch."""
    compilation = vllm.get("compilation_config", {})
    if (not isinstance(compilation, Mapping) or compilation.get("mode") != 3
            or compilation.get("cudagraph_mode") != "FULL_DECODE_ONLY"
            or int(vllm.get("data_parallel_size", 1)) != 1):
        raise ValueError("Qwen3.5/Qwen3.8 requires enforce_eager=true or mode=3 FULL_DECODE_ONLY with rollout DP=1")
    sizes = compilation.get("cudagraph_capture_sizes")
    max_seqs = vllm.get("max_num_seqs")
    if (not isinstance(max_seqs, int) or isinstance(max_seqs, bool) or max_seqs < 1
            or not isinstance(sizes, list) or not sizes
            or any(isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= max_seqs for size in sizes)
            or sizes != sorted(set(sizes)) or sizes[0] != 1 or sizes[-1] != max_seqs
            or compilation.get("max_cudagraph_capture_size") != max_seqs):
        raise ValueError("Qwen3.5/Qwen3.8 decode capture sizes must cover 1 through max_num_seqs explicitly")


def _validate_moe_parallelism(accelerator: Mapping[str, Any], model: ModelRegistration) -> None:
    """Check expert meshes against the training world and checkpoint experts."""
    sizes = {name: accelerator.get(name, 1) for name in ("ep", "edp_shard")}
    for name, size in sizes.items():
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"train.accelerator.{name} must be a positive integer")
    if sizes["ep"] == 1 and sizes["edp_shard"] != 1:
        raise ValueError("train.accelerator.edp_shard requires ep>1")
    world_size = _trainer_world_size(accelerator)
    if world_size % (sizes["ep"] * sizes["edp_shard"]):
        raise ValueError("train.accelerator.ep * edp_shard must divide training world_size")
    with (Path(model.weights_path) / "config.json").open(encoding="utf-8") as config_file:
        num_experts = _positive_integer(json.load(config_file), "num_experts", "model config")
    if num_experts % sizes["ep"]:
        raise ValueError("train.accelerator.ep must divide num_experts")


def _validate_vllm(
    rollout: Mapping[str, Any],
    accelerator: Mapping[str, Any],
    model_registration: ModelRegistration,
) -> None:
    """Validate the configured vLLM-Ascend deployment."""
    if (torch.accelerator.current_accelerator() or torch.device("cpu")).type != "npu":
        raise ValueError("The vLLM-Ascend rollout backend requires the Torch NPU backend")
    vllm = optional_mapping(rollout, "vllm")
    deployment, rollout_dp, rollout_tp = _validate_vllm_basics(vllm)
    trainer_world_size = _trainer_world_size(accelerator)
    if deployment == "colocated":
        if not accelerator.get("cpu_offload", False) and (
            model_registration.family != "qwen3" or int(accelerator.get("tp", 1)) != 1
        ):
            raise ValueError("Colocated device-resident training currently requires Qwen3 dense with TP1")
        _validate_colocated_vllm(vllm, accelerator, trainer_world_size, rollout_dp, rollout_tp)
    else:
        _validate_disjoint_vllm(vllm, trainer_world_size, rollout_dp, rollout_tp)
    _validate_vllm_limits(vllm)
    rollout_model = resolve_vllm_model(
        model_registration,
        vllm.get("model_implementation", "native"),
    )
    if model_registration.family == "qwen3":
        _validate_dense_parallelism(vllm, accelerator)
    resolve_weight_sync_config(
        optional_mapping(vllm, "weight_sync"),
        deployment=deployment,
        model_family=rollout_model.family,
        rollout_tp=rollout_tp,
    )


def _validate_agentic(agentic: Mapping[str, Any]) -> None:
    """Validate the selected agent environment and turn limits."""
    runner = agentic.get("runner", "internal")
    if runner not in {"internal", "codex", "deepseek"}:
        raise ValueError("agentic.runner must be 'internal', 'codex', or 'deepseek'")
    module_path = agentic.get("module_path")
    if module_path is not None:
        load_agentic_module(module_path)
    environment_name = agentic.get("environment")
    if runner == "internal" and environment_name not in ENVIRONMENTS.names:
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
    if runner == "codex":
        _validate_codex_agentic(agentic)
    if runner == "deepseek":
        _validate_deepseek_agentic(agentic)


def _validate_gateway_config(config: Mapping[str, Any], prefix: str) -> None:
    """Validate settings shared by external Agentic gateways."""
    port = int(config.get("gateway_port", 0))
    if not 0 < port < 65536:
        raise ValueError(f"{prefix}.gateway_port must be in [1, 65535]")
    for name in ("timeout_seconds", "request_timeout"):
        if not math.isfinite(float(config.get(name, 0.0))) or float(config.get(name, 0.0)) <= 0.0:
            raise ValueError(f"{prefix}.{name} must be positive")
    for name, default in (("max_request_bytes", 8 * 1024 * 1024), ("max_response_bytes", 32 * 1024 * 1024)):
        _positive_integer({name: config.get(name, default)}, name, prefix)


def _validate_codex_context(codex: Mapping[str, Any]) -> None:
    """Validate optional instructions and the task tool allowlist."""
    instructions = codex.get("model_instructions")
    if not isinstance(codex.get("preserve_environment_context", False), bool):
        raise ValueError("agentic.codex.preserve_environment_context must be a boolean")
    if instructions is not None and (not isinstance(instructions, str) or not instructions.strip()):
        raise ValueError("agentic.codex.model_instructions must be a non-empty string when configured")
    allowed_tools = codex.get("allowed_tools")
    if allowed_tools is not None and (
            not isinstance(allowed_tools, list)
            or any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in allowed_tools)
            or len(set(allowed_tools)) != len(allowed_tools)):
        raise ValueError("agentic.codex.allowed_tools must be null or a list of unique non-empty tool names")


def _validate_codex_agentic(agentic: Mapping[str, Any]) -> None:
    """Validate the pinned Codex CLI and MCP execution contract."""
    codex = agentic.get("codex")
    if not isinstance(codex, Mapping):
        raise ValueError("agentic.codex must be a mapping for the Codex runner")
    for name in ("version", "executable", "session_root"):
        value = codex.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"agentic.codex.{name} must be a non-empty string")
    _validate_codex_context(codex)
    reward = codex.get("reward_callable")
    if not isinstance(reward, str) or reward.count(":") != 1 or not all(reward.split(":")):
        raise ValueError("agentic.codex.reward_callable must use module:function syntax")
    if not isinstance(codex.get("admin_auth", False), bool):
        raise ValueError("agentic.codex.admin_auth must be a boolean")
    if codex["version"] != "0.152.1":
        raise ValueError("The Codex runner protocol is validated only for codex-cli 0.152.1")
    _validate_gateway_config(codex, "agentic.codex")
    _positive_integer({"max_concurrent_episodes_per_rank": codex.get("max_concurrent_episodes_per_rank", 8)},
                      "max_concurrent_episodes_per_rank", "agentic.codex")
    _positive_integer({"max_inflight_requests": codex.get("max_inflight_requests", 1)},
                      "max_inflight_requests", "agentic.codex")
    if codex.get("sandbox", "danger-full-access") not in {"danger-full-access", "workspace-write"}:
        raise ValueError("agentic.codex.sandbox must be danger-full-access or workspace-write")
    overlap = codex.get("overlap_preparation", False)
    if not isinstance(overlap, bool):
        raise ValueError("agentic.codex.overlap_preparation must be a boolean")
    fraction = codex.get("preparation_cpu_fraction", 0.5)
    if (isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not 0 < fraction <= 1):
        raise ValueError("agentic.codex.preparation_cpu_fraction must be in (0, 1]")
    if codex.get("preparation_max_cpus") is not None:
        _positive_integer(codex, "preparation_max_cpus", "agentic.codex")
    if overlap and (codex.get("program_callable") or codex.get("execution_callable")):
        raise ValueError("Codex overlap preparation currently requires the native local CLI program")
    _validate_codex_mcp_servers(codex)


def _validate_deepseek_agentic(agentic: Mapping[str, Any]) -> None:
    """Validate the pinned DeepSeek Harness execution contract."""
    deepseek = agentic.get("deepseek")
    if not isinstance(deepseek, Mapping):
        raise ValueError("agentic.deepseek must be a mapping for the DeepSeek runner")
    for name in ("version", "provider", "model", "session_root", "reward_callable"):
        value = deepseek.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"agentic.deepseek.{name} must be a non-empty string")
    if deepseek["version"] != "0.1.1rc1":
        raise ValueError(
            "The DeepSeek runner protocol is validated only for deepseek-harness-sdk 0.1.1rc1"
        )
    if deepseek.get("reasoning_effort", "off") not in {"off", "high", "max"}:
        raise ValueError("agentic.deepseek.reasoning_effort must be 'off', 'high', or 'max'")
    runtime_bin = deepseek.get("runtime_bin")
    if runtime_bin is not None and (not isinstance(runtime_bin, str) or not runtime_bin):
        raise ValueError("agentic.deepseek.runtime_bin must be a non-empty string when configured")
    _validate_gateway_config(deepseek, "agentic.deepseek")
    _positive_integer({"max_concurrent_episodes_per_rank": deepseek.get("max_concurrent_episodes_per_rank", 8)},
                      "max_concurrent_episodes_per_rank", "agentic.deepseek")
    if float(deepseek.get("shutdown_timeout_seconds", 5.0)) <= 0.0:
        raise ValueError("agentic.deepseek.shutdown_timeout_seconds must be positive")
    required_tool_calls = deepseek.get("required_tool_calls")
    if required_tool_calls is not None and int(required_tool_calls) < 0:
        raise ValueError("agentic.deepseek.required_tool_calls must be non-negative")
    required_tool_name = deepseek.get("required_tool_name")
    if required_tool_name is not None and (not isinstance(required_tool_name, str) or not required_tool_name):
        raise ValueError("agentic.deepseek.required_tool_name must be a non-empty string")


def _validate_external_agentic_rollout(
    runner: str,
    engine_name: Any,
    rollout: Mapping[str, Any],
    agentic: Mapping[str, Any],
    *,
    model_family: Optional[str] = None,
) -> None:
    """Validate requirements shared by Codex and DeepSeek vLLM adapters."""
    display_name = {"codex": "Codex", "deepseek": "DeepSeek"}[runner]
    if engine_name != "vllm":
        raise ValueError(f"The {display_name} runner currently requires rollout.engine=vllm")
    vllm = required_mapping(rollout, "vllm")
    runner_config = required_mapping(agentic, runner)
    if str(vllm.get("logprobs_mode", "")) != "raw_logprobs":
        raise ValueError(f"The {display_name} runner requires rollout.vllm.logprobs_mode=raw_logprobs")
    if not bool(vllm.get("enable_auto_tool_choice", False)):
        raise ValueError(f"The {display_name} runner requires rollout.vllm.enable_auto_tool_choice=true")
    parser = vllm.get("tool_call_parser")
    if not isinstance(parser, str) or not parser:
        raise ValueError(f"The {display_name} runner requires rollout.vllm.tool_call_parser")
    expected_parser = "qwen3_coder" if model_family == "qwen3_5" else "hermes"
    if parser != expected_parser:
        raise ValueError(
            f"{display_name} tool evidence requires rollout.vllm.tool_call_parser={expected_parser}"
        )
    if int(vllm.get("port", 0)) == int(runner_config["gateway_port"]):
        raise ValueError(f"{display_name} gateway_port must differ from rollout.vllm.port")
    if runner == "codex" and runner_config.get("model_context_window") is not None:
        context_window = _positive_integer(runner_config, "model_context_window", "agentic.codex")
        model_limit = _positive_integer(vllm, "max_model_len", "rollout.vllm")
        output_limit = _positive_integer(rollout, "max_new_tokens", "rollout")
        # The pinned CLI starts compaction near 90% of its configured window.
        # Leave room for one response and fixed overhead before vLLM's hard limit.
        if context_window > model_limit or 9 * context_window // 10 + output_limit + 512 > model_limit:
            raise ValueError(
                "Repository Codex context reserve exceeds rollout.vllm.max_model_len: "
                "require model_context_window <= max_model_len and "
                "floor(0.9 * model_context_window) + rollout.max_new_tokens + 512 <= max_model_len"
            )


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
    runner = str(agentic.get("runner", "internal"))
    if runner in {"codex", "deepseek"}:
        _validate_external_agentic_rollout(
            runner, engine_name, rollout, agentic, model_family=model_registration.family,
        )


def _validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Validate checkpoint options against available role support."""
    save_final = bool(checkpoint.get("save_final", True))
    verify_reload = bool(checkpoint.get("verify_reload", False))
    save_steps = int(checkpoint.get("save_steps", 0))
    if verify_reload and not save_final:
        raise ValueError("checkpoint.verify_reload requires checkpoint.save_final=true")
    if save_steps < 0:
        raise ValueError("checkpoint.save_steps must be non-negative")
    _ = _path_value(checkpoint, "output_dir")


def _validate_logging(config: Mapping[str, Any]) -> None:
    """Validate configured metric backends."""
    backends = config.get("backends", ())
    if not isinstance(backends, list) or not backends:
        raise ValueError("logging.backends must be a non-empty list")
    unsupported = set(backends) - {"console", "wandb"}
    if unsupported:
        raise ValueError(f"Unsupported logging backends: {sorted(unsupported)}")
    wandb = required_mapping(config, "wandb")
    if wandb.get("mode", "auto") not in {"auto", "online", "offline", "disabled"}:
        raise ValueError(f"Unsupported W&B mode: {wandb.get('mode')}")


def _validate_critic(
    train: Mapping[str, Any],
    rollout: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    """Validate PPO Critic overrides without changing GRPO configuration."""
    critic = optional_mapping(train, "critic")
    allowed = {
        "weights_path",
        "optimizer",
        "micro_batch_size",
        "response_mini_batch_size",
        "update_epochs",
    }
    unsupported = set(critic) - allowed
    if unsupported:
        raise ValueError(f"Unsupported train.critic fields: {sorted(unsupported)}")
    for field in ("micro_batch_size", "response_mini_batch_size", "update_epochs"):
        if field in critic and (
            isinstance(critic[field], bool) or int(critic[field]) <= 0
        ):
            raise ValueError(f"train.critic.{field} must be positive")
    response_count = int(train["prompt_batch_size"]) * int(
        rollout["num_return_sequences"]
    )
    if int(critic.get("response_mini_batch_size", train["response_mini_batch_size"])) > response_count:
        raise ValueError("Critic mini-batch cannot exceed local response count")
    _ = optional_mapping(critic, "optimizer")
    if "weights_path" in critic:
        weights_path = _path_value(critic, "weights_path")
        registration = resolve_model({**model, "weights_path": weights_path})
        if registration.family != "qwen3":
            raise ValueError("PPO Critic requires a dense Qwen3 checkpoint")


def _validate_distillation(config: Mapping[str, Any]) -> None:
    """Validate distillation sampling and task-reward contracts."""
    distillation = parse_distillation_config(config.get('distillation'))
    if distillation.enabled:
        if 'reward_model' in config and not distillation.use_task_rewards:
            raise ValueError('Pure OPD cannot request task reward-model scoring')
        rollout_settings = required_mapping(config, 'rollout')
        if (float(rollout_settings.get('temperature', 1.0)) != 1.0
                or float(rollout_settings.get('top_p', 1.0)) != 1.0
                or int(rollout_settings.get('top_k', 0)) not in (0, -1)):
            raise ValueError('Initial PG OPD requires temperature=1, top_p=1 and disabled top_k')
        if required_mapping(rollout_settings, 'vllm').get('logprobs_mode') != 'raw_logprobs':
            raise ValueError('PG OPD requires rollout.vllm.logprobs_mode=raw_logprobs')
        if (required_mapping(config, 'agentic').get('runner', 'internal') != 'internal'
                and not distillation.use_task_rewards):
            raise ValueError('Pure OPD currently requires the internal runner; external agents retain episode GRPO')


def _validate_log_prob_source(config: Mapping[str, Any], train: Mapping[str, Any],
                              rollout: Mapping[str, Any], agentic: Mapping[str, Any]) -> None:
    """Require sampled probabilities whenever the configured consumer needs them."""
    if train.get("old_log_prob_source", "rollout") not in ("rollout", "actor"):
        raise ValueError("train.old_log_prob_source must be rollout or actor")
    collect_log_probs = rollout.get("collect_log_probs", True)
    if not isinstance(collect_log_probs, bool):
        raise ValueError("rollout.collect_log_probs must be a boolean")
    if not collect_log_probs:
        if train.get("old_log_prob_source", "rollout") != "actor":
            raise ValueError("rollout.collect_log_probs=false requires train.old_log_prob_source=actor")
        if agentic.get("runner", "internal") != "internal":
            raise ValueError("Disabling rollout log-probabilities requires the internal runner")
        if (parse_distillation_config(config.get("distillation")).enabled):
            raise ValueError("Consistency and distillation require rollout log-probabilities")


def _validate_training_residency(config: Mapping[str, Any], algorithm: RLAlgorithm) -> None:
    """Validate training residency and gradient synchronization switches."""
    train = required_mapping(config, "train")
    accelerator = required_mapping(train, "accelerator")
    for name in ("cpu_offload", "reference_cpu_offload", "offload_during_rollout"):
        if name in accelerator and not isinstance(accelerator[name], bool):
            raise ValueError(f"train.accelerator.{name} must be a boolean")
    zero_copy = accelerator.get("comm_fusion_zero_copy")
    if zero_copy is not None and not isinstance(zero_copy, bool):
        raise ValueError("train.accelerator.comm_fusion_zero_copy must be a boolean or null")
    if not isinstance(train.get("gradient_sync_each_micro_batch", False), bool):
        raise ValueError("train.gradient_sync_each_micro_batch must be a boolean")
    if uses_colocated_vllm(config) and not accelerator.get("cpu_offload", False):
        if algorithm.name != "grpo":
            raise ValueError("Colocated device-resident training currently requires GRPO with consistency off")


def _validate_training_optimizer(train: Mapping[str, Any], rollout: Mapping[str, Any],
                                 model: Mapping[str, Any], algorithm: RLAlgorithm) -> None:
    """Keep Actor and Critic optimizer precision policies consistent."""
    optimizer = required_mapping(train, "optimizer")
    no_decay = optimizer.get("no_decay_params", ["bias", "norm", "ln_"])
    if not isinstance(no_decay, (list, tuple)) or any(not isinstance(name, str) for name in no_decay):
        raise ValueError("train.optimizer.no_decay_params must be a list of parameter-name strings")
    main_params = _optimizer_fp32_main_params(optimizer)
    mixed_precision = required_mapping(train, "mixed_precision")
    if main_params and (not mixed_precision.get("enabled", True) or
                        _normalize_dtype_name(mixed_precision.get("reduce_dtype", "float32")) != "float32"):
        raise ValueError("FP32 main parameters require enabled mixed precision and float32 gradient reduction")
    if algorithm.requirements.roles.critic:
        _validate_critic(train, rollout, model)
        critic_optimizer = {**optimizer, **optional_mapping(optional_mapping(train, "critic"), "optimizer")}
        if _optimizer_fp32_main_params(critic_optimizer) != main_params:
            raise ValueError("Actor and Critic must use the same fp32_main_params policy")


def _validate_rollout_offload(config: Mapping[str, Any], model_registration: ModelRegistration,
                              algorithm: RLAlgorithm) -> None:
    """Restrict phase offload to model and rollout combinations that support it."""
    accelerator = required_mapping(required_mapping(config, "train"), "accelerator")
    if accelerator.get("offload_during_rollout", False) and (
        not uses_colocated_vllm(config) or model_registration.family != "qwen3"
        or int(accelerator.get("tp", 1)) != 1 or algorithm.name != "grpo"

        or parse_distillation_config(config.get("distillation")).enabled or "reward_model" in config
    ):
        raise ValueError(
            "offload_during_rollout requires colocated Qwen3 dense TP1 GRPO with consistency off, "
            "no distillation or reward model"
        )


def _validate_dynamic_batching(config: Mapping[str, Any], model_registration: ModelRegistration,
                               algorithm: RLAlgorithm) -> None:
    """Validate dynamic token budgets and optional Actor DP balancing."""
    train = required_mapping(config, "train")
    accelerator = required_mapping(train, "accelerator")
    if not isinstance(train.get("balance_dp_tokens", False), bool):
        raise ValueError("train.balance_dp_tokens must be a boolean")
    if train.get("balance_dp_tokens", False) and train.get("max_tokens_per_micro_batch") is None:
        raise ValueError("train.balance_dp_tokens requires train.max_tokens_per_micro_batch")
    token_budgets = ("max_tokens_per_micro_batch", "old_log_prob_max_tokens_per_micro_batch",
                     "reference_log_prob_max_tokens_per_micro_batch")
    for field in token_budgets:
        if train.get(field) is not None:
            _positive_integer(train, field, "train")
    if any(train.get(field) is not None for field in token_budgets):
        if (model_registration.family != "qwen3" or int(accelerator.get("tp", 1)) != 1
                or algorithm.name != "grpo" or parse_distillation_config(config.get("distillation")).enabled):
            raise ValueError(
                "Dynamic Actor batching requires Qwen3 dense TP1 GRPO with consistency off, no distillation"
            )


def _validate_accelerator_options(
    train: Mapping[str, Any], accelerator: Mapping[str, Any],
) -> None:
    """Check supported parallel execution flags and explicit HP replacement rules."""
    for field, supported in (("sequence_parallel", False), ("loss_parallel", False),
                             ("reshard_after_backward", False), ("requires_grad_sync", True)):
        value = accelerator.get(field, supported)
        if not isinstance(value, bool) or value is not supported:
            raise ValueError(f"train.accelerator.{field} currently requires {supported}")
    if "plan_overrides" not in train or not isinstance(train["plan_overrides"], list):
        raise ValueError("train.plan_overrides must explicitly declare a list of HP rules")


def validate_config(config: Mapping[str, Any], algorithm: RLAlgorithm) -> None:
    """Validate Hyper-RL configuration before distributed startup."""
    if "consistency" in config:
        raise ValueError("consistency was removed; remove the field and use native-vLLM")
    unknown = set(config) - _EXPECTED_TOP_LEVEL
    if unknown:
        raise ValueError(f"Unsupported top-level configuration keys: {sorted(unknown)}")
    _validate_distillation(config)
    model = required_mapping(config, "model")
    data = required_mapping(config, "data")
    validate_row_adapter_config(data)
    validate_chat_template_kwargs(data.get("chat_template_kwargs"))
    rollout = required_mapping(config, "rollout")
    agentic = required_mapping(config, "agentic")
    if agentic.get("runner") in {"codex", "deepseek"} and algorithm.name != "grpo":
        name = {"codex": "Codex", "deepseek": "DeepSeek"}[agentic["runner"]]
        raise ValueError(f"Segmented {name} agent trajectories currently require GRPO")
    evaluation = required_mapping(config, "evaluation")
    train = required_mapping(config, "train")
    _validate_log_prob_source(config, train, rollout, agentic)
    if train.get("model_init_cpu_threads") is not None:
        _positive_integer(train, "model_init_cpu_threads", "train")
    accelerator = required_mapping(train, "accelerator")
    _validate_training_residency(config, algorithm)
    _validate_accelerator_options(train, accelerator)
    _validate_training_optimizer(train, rollout, model, algorithm)
    model_registration = build_model_registration(config)
    _validate_rollout_offload(config, model_registration, algorithm)
    _validate_model_scope(config, model_registration)
    _validate_token_log_probs(config, model_registration, algorithm)
    _validate_dynamic_batching(config, model_registration, algorithm)
    reference_offload = accelerator.get("reference_cpu_offload", accelerator.get("cpu_offload", False))
    if reference_offload != accelerator.get("cpu_offload", False) and (
        model_registration.family != "qwen3" or int(accelerator.get("tp", 1)) != 1
        or algorithm.name != "grpo"
    ):
        raise ValueError("Independent Reference offload requires Qwen3 dense TP1 GRPO with consistency off")
    _validate_model_and_data_paths(model, data, evaluation)
    _validate_training_sizes(train, rollout, data, algorithm, agentic)
    _validate_evaluation(evaluation)
    validate_rollout_and_agentic(rollout, agentic, accelerator, model_registration)
    if rollout.get("engine") != "vllm":
        _ = _trainer_topology(accelerator)
    checkpoint = required_mapping(train, "checkpoint")
    _validate_checkpoint(checkpoint)
    if rollout.get("engine") == "vllm" and optional_mapping(rollout, "vllm").get("sleep_level", 1) == 2:
        save_steps = int(checkpoint.get("save_steps", 0))
        if 0 < save_steps < int(train["max_steps"]):
            raise ValueError(
                "rollout.vllm.sleep_level=2 does not support intermediate checkpoints: "
                "resuming rollout requires a complete weight publication"
            )
        if "reward_model" in config and bool(evaluation.get("enabled", True)):
            raise ValueError(
                "rollout.vllm.sleep_level=2 does not support reward_model with evaluation.enabled=true: "
                "resuming rollout after reward scoring requires a complete weight publication"
            )
    _validate_logging(required_mapping(config, "logging"))
    if "reward_model" in config:
        _validate_reward_model(config, model_registration)


def _validate_reward_model_topology(config: Mapping[str, Any], reward: Mapping[str, Any]) -> None:
    """Check reward model device coverage and network port separation."""
    rollout_vllm = required_mapping(required_mapping(config, "rollout"), "vllm")
    for field in ("tensor_parallel_size", "data_parallel_size", "port", "server_hccl_if_base_port"):
        value = reward.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"reward_model.{field} must be a positive integer")
    tp, dp = reward["tensor_parallel_size"], reward["data_parallel_size"]
    rollout_world = int(rollout_vllm["tensor_parallel_size"]) * int(rollout_vllm["data_parallel_size"])
    if tp not in (1, 2) or tp * dp != rollout_world:
        raise ValueError("RM TP1/TP2 x DP must cover the complete rollout device set")
    if not 1024 <= reward["port"] <= 65535 or reward["port"] == int(rollout_vllm["port"]):
        raise ValueError("RM port must be valid and distinct from rollout")
    base_port = reward["server_hccl_if_base_port"]
    if not 1024 <= base_port <= 65436:
        raise ValueError("RM HCCL base port must leave room for its range")
    try:
        start, end = map(int, str(reward["server_hccl_npu_socket_port_range"]).split("-"))
    except (KeyError, ValueError) as error:
        raise ValueError("RM HCCL socket range must use start-end syntax") from error
    if not 1024 <= start <= base_port <= end <= 65535:
        raise ValueError("RM HCCL socket range must contain its base port")
    actor_base = rollout_vllm.get("server_hccl_if_base_port")
    if actor_base is not None and start <= int(actor_base) <= end:
        raise ValueError("RM and Actor rollout HCCL ports overlap")


def _validate_reward_model_runtime(reward: Mapping[str, Any]) -> None:
    """Check reward model dtype and finite runtime limits."""
    if reward.get("dtype", "bfloat16") != "bfloat16":
        raise ValueError("Colocated RM currently requires bfloat16")
    for field in ("startup_timeout", "request_timeout", "gpu_memory_utilization"):
        value = float(reward.get(field, 0.35 if field == "gpu_memory_utilization" else 600))
        if not math.isfinite(value) or value <= 0 or (field == "gpu_memory_utilization" and value > 1):
            raise ValueError(f"reward_model.{field} must be finite and positive")
    for field in ("max_num_seqs", "max_model_len", "kv_cache_memory_bytes", "max_new_tokens", "max_concurrency"):
        if field in reward and (not isinstance(reward[field], int) or isinstance(reward[field], bool)
                                or reward[field] <= 0):
            raise ValueError(f"reward_model.{field} must be positive")
    if int(reward.get("max_retries", 2)) < 0:
        raise ValueError("reward_model.max_retries must be non-negative")


def _validate_reward_model(config: Mapping[str, Any], model_registration: ModelRegistration) -> None:
    """Require a complete single-node, colocated RM before any runtime starts."""
    reward = required_mapping(config, "reward_model")
    allowed = {
        "scorer", "model_path", "scoring", "deployment", "tensor_parallel_size", "data_parallel_size",
        "port", "dtype", "gpu_memory_utilization", "startup_timeout", "request_timeout", "max_num_seqs",
        "max_model_len", "kv_cache_memory_bytes", "max_new_tokens", "server_hccl_if_base_port",
        "server_hccl_npu_socket_port_range", "log_path", "max_retries", "served_model_name",
        "max_concurrency",
    }
    if set(reward) - allowed:
        raise ValueError(f"Unsupported reward_model fields: {sorted(set(reward) - allowed)}")
    agentic = required_mapping(config, "agentic")
    if agentic.get("runner", "internal") != "internal" or agentic.get("environment") != "gsm8k_tools":
        raise ValueError("Colocated reward_model currently requires the internal GSM8K environment")
    if required_mapping(config, "algorithm").get("name") not in {"grpo", "gspo"}:
        raise ValueError("Colocated reward_model currently supports GRPO or GSPO")
    if model_registration.family != "qwen3":
        raise ValueError("Colocated reward_model currently requires Qwen3 dense")
    if reward.get("deployment") != "colocated" or not uses_colocated_vllm(config):
        raise ValueError("reward_model requires colocated vLLM rollout")
    train = required_mapping(config, "train")
    if not required_mapping(train, "accelerator").get("cpu_offload", False):
        raise ValueError("Colocated reward_model requires train.accelerator.cpu_offload=true")
    scorer = reward.get("scorer")
    try:
        load_reward_function(scorer)
    except (ImportError, AttributeError, TypeError) as error:
        raise ValueError(f"Invalid reward_model.scorer: {error}") from error
    if reward.get("scoring") not in {"generative", "discriminative"}:
        raise ValueError("reward_model.scoring must be generative or discriminative")
    path = Path(_path_value(reward, "model_path"))
    identity_path = path / "config.json"
    if not identity_path.is_file():
        raise ValueError(f"Reward model configuration not found: {identity_path}")
    if reward["scoring"] == "discriminative":
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if not any("SequenceClassification" in name for name in identity.get("architectures", [])):
            raise ValueError("Discriminative RM requires a trained sequence-classification checkpoint")
    _validate_reward_model_topology(config, reward)
    _validate_reward_model_runtime(reward)


def _validate_token_log_probs(config: Mapping[str, Any], model: ModelRegistration, algorithm: RLAlgorithm) -> None:
    """Limit the opt-in probability adapter to validated full-vocabulary RL roles."""
    train = config["train"]
    mode = train.get("token_log_probs_mode", "auto")
    if mode not in ("auto", "dense", "selected_chunked"):
        raise ValueError("train.token_log_probs_mode must be auto, dense or selected_chunked")
    chunk_size = train.get("token_log_probs_chunk_size", 512)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 0 < chunk_size <= 512:
        raise ValueError("train.token_log_probs_chunk_size must be an integer between 1 and 512")
    if mode == "selected_chunked" and (
        model.family not in ("qwen3", "qwen3_5") or int(train["accelerator"].get("tp", 1)) != 1
        or algorithm.name != "grpo"
        or parse_distillation_config(config.get("distillation")).enabled
    ):
        raise ValueError("selected_chunked requires Qwen3/Qwen3.5 text TP1 GRPO with consistency off, no distillation")


def _validate_vllm_hccl_ports(vllm: Mapping[str, Any]) -> None:
    """Validate the paired server HCCL port configuration."""
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


def _validate_codex_mcp_servers(codex: Mapping[str, Any]) -> None:
    """Validate tool-server environment mappings and inherited variable names."""
    mcp_servers = codex.get("mcp_servers", [])
    if not isinstance(mcp_servers, list) or not all(isinstance(item, Mapping) for item in mcp_servers):
        raise ValueError("agentic.codex.mcp_servers must be a list of mappings")
    for server in mcp_servers:
        environment = server.get("env", {})
        if not isinstance(environment, Mapping) or not all(
            isinstance(key, str) and key and isinstance(value, str) for key, value in environment.items()
        ):
            raise ValueError("agentic.codex.mcp_servers env must map non-empty names to strings")
        inherited_names = server.get("inherit_env", [])
        if not isinstance(inherited_names, list) or not all(
            isinstance(item, str) and item for item in inherited_names
        ):
            raise ValueError("agentic.codex.mcp_servers inherit_env must be a list of non-empty names")


def validate_async_config(config: Mapping[str, Any]) -> AsyncTrainingConfig:
    """Admit only validated GRPO/topology/task combinations before creating models."""
    settings = AsyncTrainingConfig.from_mapping(config.get("async_training", {}))
    data, rollout, agent = (config.get(key, {}) for key in ("data", "rollout", "agentic"))
    train = config.get("train", {})
    vllm, accelerator = rollout.get("vllm", {}), train.get("accelerator", {})
    runner, turns = agent.get("runner", "internal"), agent.get("max_turns", 1)
    candidates, dp_size = rollout.get("num_return_sequences", 0), accelerator.get("dp_shard", 1)
    _require_async(data.get("num_workers", 0) == 0 and not data.get("pin_memory", False),
             "Async CPU source requires num_workers=0 and pin_memory=false")
    _require_async(config.get("algorithm", {}).get("name") == "grpo", "Asynchronous training supports GRPO only")
    _require_async(rollout.get("engine") == "vllm" and vllm.get("deployment") == "disjoint",
             "Asynchronous training requires disjoint vLLM inference devices")
    _require_async(vllm.get("logprobs_mode", "raw_logprobs") == "raw_logprobs", "Async GRPO requires raw_logprobs")
    _require_async(tuple(rollout.get(key, default) for key, default in
                         (("temperature", 1.), ("top_p", 1.), ("top_k", 0)))
             == (1., 1., 0), "Async recipe requires temperature=1, top_p=1, top_k=0")
    _require_async(_async_integer(candidates, 2), "GRPO requires at least two candidates")
    _require_async(runner in ("internal", "codex") and _async_integer(turns), "Invalid async runner or max_turns")
    _require_async(runner != "internal" or turns == 1 and agent.get("interaction_mode", "single_turn") == "single_turn",
             "Async internal sampling requires single_turn")
    _require_async(runner != "codex" or isinstance(agent.get("codex"), Mapping), "Async Codex requires codex settings")
    _require_async(runner != "codex" or not config.get("evaluation", {}).get("enabled", False),
             "Async Codex evaluation must run separately from the producer gateway")
    _require_async("reward_model" not in config and not config.get("distillation", {}).get("enabled", False),
             "Model rewards and distillation are not validated for asynchronous training")
    _require_async(config.get("model", {}).get("name") == "qwen3", "Async recipe supports Qwen3 dense only")
    _require_async(all(accelerator.get(key, 1) == 1 for key in ("cp", "pp", "dp_replicate", "ep", "edp_shard")),
             "Unsupported async parallel topology")
    _require_async(_async_integer(dp_size), "dp_shard must be an explicit positive size")
    _require_async(settings.max_buffer_groups >= dp_size * settings.groups_per_dp_rank,
                   "Buffer must hold a global batch")
    mini, micro = train["response_mini_batch_size"], train["micro_batch_size"]
    _require_async(_async_integer(mini) and _async_integer(micro),
                   "Training mini/micro batch sizes must be positive integers")
    _require_async(not mini % micro and (runner == "codex" or not settings.groups_per_dp_rank * candidates % mini),
             "Local candidate rows must divide into equal mini/micro batches")
    return settings


def _validate_colocate_topology(config: Mapping[str, Any]) -> int:
    """Require one FSDP world sharing its devices with a single rollout TP replica."""
    rollout, train = required_mapping(config, "rollout"), required_mapping(config, "train")
    vllm, accelerator = required_mapping(rollout, "vllm"), required_mapping(train, "accelerator")
    dp_size = accelerator.get("dp_shard")
    _require_async(_async_integer(dp_size, 2), "Colocate async requires at least two FSDP ranks")
    _require_async(all(_async_integer(accelerator.get(key, 1)) and accelerator.get(key, 1) == 1
                       for key in ("tp", "cp", "pp", "dp_replicate", "ep", "edp_shard")),
                   "Colocate async supports FSDP with training TP/CP/PP/EP equal to one")
    _require_async(config.get("model", {}).get("name") == "qwen3", "Colocate async requires Qwen3 dense")
    _require_async(config.get("algorithm", {}).get("name") == "grpo", "Colocate async supports GRPO only")
    _require_async(rollout.get("engine") == "vllm" and vllm.get("deployment") == "colocated",
                   "Colocate async requires colocated vLLM")
    _require_async(_async_integer(vllm.get("data_parallel_size")) and vllm.get("data_parallel_size") == 1
                   and _async_integer(vllm.get("tensor_parallel_size"))
                   and vllm.get("tensor_parallel_size") == dp_size,
                   "Colocate async requires rollout DP=1 and TP=training world size")
    _require_async(accelerator.get("cpu_offload") is True
                   and accelerator.get("reshard_after_forward", True) is True,
                   "Colocate async requires CPU offload and reshard_after_forward")
    _require_async("visible_devices" not in vllm, "Colocated rollout inherits training devices")
    if "WORLD_SIZE" in os.environ:
        _require_async(int(os.environ["WORLD_SIZE"]) == dp_size
                       and int(os.environ.get("LOCAL_WORLD_SIZE", str(dp_size))) == dp_size,
                       "Colocate async requires a single-node torchrun world matching dp_shard")
    return dp_size


def _validate_colocate_generation(config: Mapping[str, Any]) -> None:
    """Require the supported partial-generation, residency and probability contracts."""
    rollout, agent = required_mapping(config, "rollout"), required_mapping(config, "agentic")
    vllm = required_mapping(rollout, "vllm")
    _require_async(_async_integer(vllm.get("sleep_level", 1)) and vllm.get("sleep_level", 1) == 1
                   and vllm.get("enforce_eager") is True,
                   "Colocate async requires sleep_level=1 and eager execution")
    _require_async(vllm.get("weight_sync", {}).get("strategy", "full_gather") == "full_gather",
                   "Colocate async requires full_gather IPC publication")
    runner, turns = agent.get("runner", "internal"), agent.get("max_turns", 1)
    _require_async(runner in ("internal", "codex") and _async_integer(turns),
                   "Colocate async requires internal or Codex sampling and positive max_turns")
    _require_async(runner != "internal" or turns == 1 and agent.get("interaction_mode", "single_turn") == "single_turn",
                   "Colocate async internal sampling requires single_turn")
    if runner == "codex":
        required_mapping(agent, "codex")
    _require_async(not vllm.get("batch_invariant", False), "Colocate async requires consistency off")
    _require_async(vllm.get("logprobs_mode", "raw_logprobs") == "raw_logprobs"
                   and tuple(rollout.get(key, default) for key, default in
                             (("temperature", 1.), ("top_p", 1.), ("top_k", 0))) == (1., 1., 0),
                   "Colocate async requires raw probabilities and unfiltered temperature-one sampling")


def validate_colocate_async_config(config: Mapping[str, Any]) -> ColocateAsyncConfig:
    """Admit single-node Qwen3 dense GRPO with internal or existing Codex sampling."""
    _require_async("async_training" not in config, "Choose either async_training or colocate_async")
    settings = ColocateAsyncConfig.from_mapping(required_mapping(config, "colocate_async"))
    dp_size = _validate_colocate_topology(config)
    _validate_colocate_generation(config)
    rollout, train = required_mapping(config, "rollout"), required_mapping(config, "train")
    data = required_mapping(config, "data")
    _require_async(data.get("num_workers", 0) == 0 and not data.get("pin_memory", False),
                   "Colocate async CPU source requires num_workers=0 and pin_memory=false")
    _require_async(router_replay_mode(config) == "disabled" and "reward_model" not in config
                   and not config.get("distillation", {}).get("enabled", False),
                   "Colocate async does not support replay, model rewards or distillation")
    candidates = rollout.get("num_return_sequences")
    mini, micro = train.get("response_mini_batch_size"), train.get("micro_batch_size")
    _require_async(_async_integer(candidates, 2) and _async_integer(mini) and _async_integer(micro),
                   "Candidate and mini/micro batch sizes must be positive integers; candidates >= 2")
    _require_async(not mini % micro and (config["agentic"].get("runner") == "codex"
                                       or not settings.groups_per_dp_rank * candidates % mini),
                   "Local candidate rows must divide into equal mini/micro batches")
    settings.sampling_config(dp_size)
    return settings
