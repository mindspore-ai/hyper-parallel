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
import os
from pathlib import Path
from typing import Any, Mapping
from hyper_parallel import get_platform
from hyper_parallel.platform.platform import PlatformType
from hyper_parallel.trainer.config import (
    AcceleratorConfig,
    CheckpointConfig,
    DataConfig,
    HyperTrainerConfig,
    MixedPrecisionConfig,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
)
from rl.agentic.registry import ENVIRONMENTS
from rl.algorithm.loss import RLAlgorithm
from rl.roles.model import ModelRegistration
from rl.roles.rollout import ROLLOUT_ENGINES
from rl.roles.weight_sync.transfer import normalize_model_implementation
platform = get_platform()
_EXPECTED_TOP_LEVEL = frozenset(
    ("model", "data", "rollout", "agentic", "algorithm", "evaluation", "train", "logging")
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
    for field in ("batch_size", "max_new_tokens"):
        if int(evaluation.get(field, 0)) <= 0:
            raise ValueError(f"evaluation.{field} must be positive")
    max_samples = evaluation.get("max_samples")
    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("evaluation.max_samples must be positive or null")
    for field in ("log_samples", "progress_steps"):
        if int(evaluation.get(field, 0)) < 0:
            raise ValueError(f"evaluation.{field} must be non-negative")
def _validate_vllm_basics(vllm: Mapping[str, Any]) -> tuple[str, int]:
    """Validate settings shared by colocated and disjoint vLLM."""
    deployment = str(vllm.get("deployment", "disjoint"))
    if deployment not in ("disjoint", "colocated"):
        raise ValueError(
            "rollout.vllm.deployment must be 'disjoint' or 'colocated', "
            f"got {deployment!r}"
        )
    rollout_tp = int(vllm.get("tensor_parallel_size", 1))
    if rollout_tp <= 0:
        raise ValueError("rollout.vllm.tensor_parallel_size must be positive")
    if str(vllm.get("dtype", "bfloat16")) not in ("bfloat16", "bf16"):
        raise ValueError("The Hyper Qwen3.5 vLLM adapter requires bfloat16")
    if str(vllm.get("host", "127.0.0.1")) not in ("127.0.0.1", "localhost"):
        raise ValueError("The external vLLM server must bind to loopback")
    utilization = float(vllm.get("gpu_memory_utilization", 0.9))
    if not 0 < utilization < 1:
        raise ValueError("rollout.vllm.gpu_memory_utilization must be between 0 and 1")
    return deployment, rollout_tp


def _validate_colocated_vllm(
    vllm: Mapping[str, Any],
    accelerator: Mapping[str, Any],
    rollout_tp: int,
) -> None:
    """Validate colocated rollout topology and residency requirements."""
    if rollout_tp != 1:
        raise ValueError("The initial colocated rollout path supports tensor_parallel_size=1")
    dp_shard = int(accelerator.get("dp_shard", 1))
    if dp_shard <= 1:
        raise ValueError(
            "Colocated rollout requires multi-rank FSDP with "
            "train.accelerator.dp_shard > 1"
        )
    if not bool(accelerator.get("cpu_offload", False)):
        raise ValueError("Colocated rollout requires train.accelerator.cpu_offload=true")
    if not bool(accelerator.get("reshard_after_forward", True)):
        raise ValueError(
            "Colocated rollout requires train.accelerator.reshard_after_forward=true"
        )
    if vllm.get("visible_devices") is not None:
        raise ValueError(
            "Colocated rollout derives one physical NPU from each trainer rank; "
            "remove rollout.vllm.visible_devices"
        )
    base_port = vllm.get("port")
    if base_port is None:
        raise ValueError("Colocated rollout requires an explicit rollout.vllm.port base")
    if int(base_port) <= 0 or int(base_port) + dp_shard - 1 > 65535:
        raise ValueError("rollout.vllm.port range exceeds valid TCP ports")


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
    if int(accelerator.get("dp_shard", 1)) != 1:
        raise ValueError(
            "The external vLLM HCCL refitter currently supports "
            "train.accelerator.dp_shard=1"
        )
    visible_devices = vllm.get("visible_devices")
    if visible_devices is None:
        raise ValueError("rollout.vllm.visible_devices must select the external server NPUs")
    device_ids = [device.strip() for device in str(visible_devices).split(",")]
    if not all(device_ids) or len(device_ids) < rollout_tp:
        raise ValueError(
            "rollout.vllm.visible_devices must contain at least one device ID per TP rank"
        )
    training_device = _training_device_id()
    if training_device in device_ids:
        raise ValueError(
            "The external vLLM server must use NPUs disjoint from the trainer: "
            f"training_device={training_device}, rollout_devices={device_ids}"
        )


def _validate_vllm_limits(vllm: Mapping[str, Any]) -> None:
    """Validate optional vLLM concurrency and capacity limits."""
    normalize_model_implementation(vllm.get("model_implementation", "hyper"))
    if int(vllm.get("request_concurrency", 1)) <= 0:
        raise ValueError("rollout.vllm.request_concurrency must be positive")
    for field in (
        "kv_cache_memory_bytes",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
    ):
        if field in vllm and int(vllm[field]) <= 0:
            raise ValueError(f"rollout.vllm.{field} must be positive")


def _validate_vllm(
    rollout: Mapping[str, Any],
    accelerator: Mapping[str, Any],
) -> None:
    """Validate the configured vLLM-Ascend deployment."""
    if platform.platform_type != PlatformType.PYTORCH or platform.device_type() != "npu":
        raise ValueError("The vLLM-Ascend rollout backend requires the Torch NPU platform")
    vllm = optional_mapping(rollout, "vllm")
    deployment, rollout_tp = _validate_vllm_basics(vllm)
    if deployment == "colocated":
        _validate_colocated_vllm(vllm, accelerator, rollout_tp)
    else:
        _validate_disjoint_vllm(vllm, accelerator, rollout_tp)
    _validate_vllm_limits(vllm)


def _validate_agentic(agentic: Mapping[str, Any]) -> None:
    """Validate the selected agent environment and turn limits."""
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


def validate_rollout_and_agentic(
    rollout: Mapping[str, Any],
    agentic: Mapping[str, Any],
    accelerator: Mapping[str, Any],
) -> None:
    """Validate rollout deployment and environment settings."""
    engine_name = rollout.get("engine")
    if engine_name not in ROLLOUT_ENGINES.names:
        raise ValueError(
            f"Unknown rollout.engine '{engine_name}'; available={ROLLOUT_ENGINES.names}"
        )
    if engine_name == "vllm":
        _validate_vllm(rollout, accelerator)
    seed = rollout.get("seed")
    if seed is not None and int(seed) < 0:
        raise ValueError("rollout.seed must be non-negative or null")
    _validate_agentic(agentic)
def _validate_topology(accelerator: Mapping[str, Any]) -> None:
    """Validate the currently supported pure-FSDP topology."""
    topology = {
        "dp_replicate": int(accelerator.get("dp_replicate", 1)),
        "dp_shard": int(accelerator.get("dp_shard", 0)),
        "tp": int(accelerator.get("tp", 1)),
        "cp": int(accelerator.get("cp", 1)),
        "pp": int(accelerator.get("pp", 1)),
    }
    if topology["dp_shard"] <= 0:
        raise ValueError(
            "train.accelerator.dp_shard must be positive, "
            f"got {topology['dp_shard']}"
        )
    unsupported = {
        key: value for key, value in topology.items() if key != "dp_shard" and value != 1
    }
    if unsupported:
        raise ValueError(f"Hyper-RL demo supports pure FSDP only; invalid topology={unsupported}")
def _validate_checkpoint(checkpoint: Mapping[str, Any], algorithm: RLAlgorithm) -> None:
    """Validate checkpoint options against available role support."""
    save_final = bool(checkpoint.get("save_final", True))
    verify_reload = bool(checkpoint.get("verify_reload", False))
    save_steps = int(checkpoint.get("save_steps", 0))
    if verify_reload and not save_final:
        raise ValueError("checkpoint.verify_reload requires checkpoint.save_final=true")
    if save_steps < 0:
        raise ValueError("checkpoint.save_steps must be non-negative")
    if algorithm.requirements.roles.critic and (
        save_final or save_steps > 0 or checkpoint.get("load_path") is not None
    ):
        raise NotImplementedError(
            "Critic checkpoint save/resume is not implemented yet; set "
            "checkpoint.save_final=false, save_steps=0, and load_path=null for "
            f"algorithm '{algorithm.name}'"
        )
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
    model = required_mapping(config, "model")
    data = required_mapping(config, "data")
    rollout = required_mapping(config, "rollout")
    agentic = required_mapping(config, "agentic")
    evaluation = required_mapping(config, "evaluation")
    train = required_mapping(config, "train")
    accelerator = required_mapping(train, "accelerator")
    _validate_model_and_data_paths(model, data, evaluation)
    _validate_training_sizes(train, rollout, data, algorithm)
    _validate_evaluation(evaluation)
    validate_rollout_and_agentic(rollout, agentic, accelerator)
    _validate_topology(accelerator)
    _validate_checkpoint(required_mapping(train, "checkpoint"), algorithm)
    _validate_logging(required_mapping(config, "logging"))
def build_model_registration(config: Mapping[str, Any]) -> ModelRegistration:
    """Resolve the configured model shared by training and rollout."""
    model = required_mapping(config, "model")
    name = model.get("registry_name")
    if not isinstance(name, str) or not name:
        raise ValueError("model.registry_name must be a non-empty string")
    return ModelRegistration(
        name=name,
        hyper_model_name=str(model["name"]),
        weights_path=str(model["weights_path"]),
        tokenizer_path=str(model["tokenizer_path"]),
    )
def build_base_config(config: Mapping[str, Any]) -> HyperTrainerConfig:
    """Translate Hyper-RL YAML into Hyper-Parallel's trainer schema."""
    model_config = required_mapping(config, "model")
    data_config = required_mapping(config, "data")
    train_config = required_mapping(config, "train")
    accelerator_config = required_mapping(train_config, "accelerator")
    optimizer_config = required_mapping(train_config, "optimizer")
    mixed_precision_config = required_mapping(train_config, "mixed_precision")
    checkpoint_config = required_mapping(train_config, "checkpoint")
    rollout_config = required_mapping(config, "rollout")
    agentic_config = required_mapping(config, "agentic")
    max_steps = int(train_config["max_steps"])
    prompt_batch_size = int(train_config.get("prompt_batch_size", 1))
    save_steps = int(checkpoint_config.get("save_steps", 0))
    if bool(checkpoint_config.get("save_final", True)) and save_steps == 0:
        save_steps = max_steps
    dp_shard = int(accelerator_config["dp_shard"])
    cpu_offload = bool(accelerator_config.get("cpu_offload", False))
    comm_backend = train_config.get("comm_backend")
    if cpu_offload and comm_backend in (None, "hccl"):
        comm_backend = "cpu:gloo,npu:hccl"
    max_turns = int(agentic_config["max_turns"])
    per_turn_tokens = int(rollout_config["max_new_tokens"]) + int(
        agentic_config.get("max_observation_tokens", 0)
    )
    return HyperTrainerConfig(
        model=ModelConfig(
            name=str(model_config["name"]),
            weights_path=str(model_config["weights_path"]),
            tokenizer_path=str(model_config["tokenizer_path"]),
            config_overrides=model_config.get("config_overrides"),
        ),
        data=DataConfig(
            type="dummy",
            train_path=str(data_config["train_path"]),
            max_seq_len=int(data_config["max_prompt_length"]) + max_turns * per_turn_tokens,
            num_workers=int(data_config.get("num_workers", 0)),
            prefetch_factor=data_config.get("prefetch_factor"),
            pin_memory=bool(data_config.get("pin_memory", True)),
            shuffle=bool(data_config.get("shuffle", True)),
        ),
        train=TrainConfig(
            max_steps=max_steps,
            num_train_epochs=1,
            global_batch_size=dp_shard * prompt_batch_size,
            micro_batch_size=prompt_batch_size,
            seed=int(train_config.get("seed", 1234)),
            backend="torch",
            init_device=str(train_config.get("init_device", "meta")),
            comm_backend=comm_backend,
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            accelerator=AcceleratorConfig(
                dp_replicate=int(accelerator_config.get("dp_replicate", 1)),
                dp_shard=dp_shard,
                tp=int(accelerator_config.get("tp", 1)),
                cp=int(accelerator_config.get("cp", 1)),
                pp=int(accelerator_config.get("pp", 1)),
                ep=1,
                etp=1,
                reshard_after_forward=bool(
                    accelerator_config.get("reshard_after_forward", True)
                ),
                comm_fusion=bool(accelerator_config.get("comm_fusion", True)),
                cpu_offload=cpu_offload,
            ),
            mixed_precision=MixedPrecisionConfig(
                enabled=bool(mixed_precision_config.get("enabled", True)),
                param_dtype=str(mixed_precision_config.get("param_dtype", "bfloat16")),
                reduce_dtype=str(mixed_precision_config.get("reduce_dtype", "float32")),
                output_dtype=mixed_precision_config.get("output_dtype"),
            ),
            optimizer=OptimizerConfig(
                lr=float(optimizer_config.get("lr", 1.0e-6)),
                lr_min=float(optimizer_config.get("lr_min", 0.0)),
                lr_decay_style=str(optimizer_config.get("lr_decay_style", "constant")),
                lr_warmup_ratio=float(optimizer_config.get("lr_warmup_ratio", 0.0)),
                loss_aggregation="token_weighted",
                weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
                max_grad_norm=float(optimizer_config.get("max_grad_norm", 1.0)),
                eps=float(optimizer_config.get("eps", 1.0e-8)),
                betas=tuple(optimizer_config.get("betas", (0.9, 0.999))),
                foreach=optimizer_config.get("foreach"),
            ),
            checkpoint=CheckpointConfig(
                output_dir=str(checkpoint_config["output_dir"]),
                save_steps=save_steps,
                save_hf_weights=False,
                load_path=checkpoint_config.get("load_path"),
                save_async=False,
            ),
        ),
    )
__all__ = [
    "build_base_config",
    "build_model_registration",
    "optional_mapping",
    "required_mapping",
    "uses_colocated_vllm",
    "validate_config",
    "validate_rollout_and_agentic",
]
