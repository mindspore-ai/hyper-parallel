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

import json
import os
from pathlib import Path
from typing import Any, Mapping

from rl.agentic.registry import ENVIRONMENTS
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
from hyper_models._transformers import HyperAutoModelForCausalLM
from hyper_models.components.checkpoint.config import CheckpointingConfig
from hyper_models.components.distributed.config import (
    CPUOffloadPolicy,
    FSDP2Config,
    MixedPrecisionPolicy,
)
from hyper_models.components.optim.lr_scheduler import MultiLRScheduler
from hyper_models.components.optim.optimizer import AdamW
from hyper_models.trainer.config import (
    AcceleratorConfig,
    ActivationCheckpointConfig,
    MixedPrecisionConfig,
    Target,
    TrainerConfig,
    TrainingConfig,
)
platform = get_platform()
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
        raise ValueError("The Qwen vLLM rollout path requires bfloat16")
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
    normalize_model_implementation(vllm.get("model_implementation", "native"))
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
    deployment, rollout_tp = _validate_vllm_basics(vllm)
    if deployment == "colocated":
        _validate_colocated_vllm(vllm, accelerator, rollout_tp)
    else:
        _validate_disjoint_vllm(vllm, accelerator, rollout_tp)
    _validate_vllm_limits(vllm)
    rollout_model = _validate_model_implementation(vllm, model_registration)
    if rollout_model.is_hyper and rollout_tp != 1:
        raise ValueError("Hyper-vLLM currently requires tensor_parallel_size=1")


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
            "The initial HyperModels RL runtime supports critic-free algorithms only; "
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


def _resolve_dtype(name: Any) -> Any:
    """Resolve one configured dtype through the active platform abstraction."""
    normalized = None if name is None else str(name).strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
        "fp16": "float16",
        "float16": "float16",
    }
    if normalized is None:
        return None
    attribute = aliases.get(normalized)
    if attribute is None or not hasattr(platform.tensor_dtype, attribute):
        raise ValueError(f"Unsupported mixed-precision dtype: {name!r}")
    return getattr(platform.tensor_dtype, attribute)


def build_runtime_config(config: Mapping[str, Any]) -> TrainerConfig:
    """Translate Hyper-RL YAML into the HyperModels runtime configuration."""
    model_config = required_mapping(config, "model")
    train_config = required_mapping(config, "train")
    accelerator_config = required_mapping(train_config, "accelerator")
    optimizer_config = required_mapping(train_config, "optimizer")
    mixed_precision_config = required_mapping(train_config, "mixed_precision")
    checkpoint_config = required_mapping(train_config, "checkpoint")
    config_overrides = model_config.get("config_overrides")
    if config_overrides not in (None, {}):
        raise ValueError("model.config_overrides is not supported by HyperAutoModel")

    max_steps = int(train_config["max_steps"])
    prompt_batch_size = int(train_config.get("prompt_batch_size", 1))
    dp_shard = int(accelerator_config["dp_shard"])
    cpu_offload = bool(accelerator_config.get("cpu_offload", False))
    param_dtype_name = str(mixed_precision_config.get("param_dtype", "bfloat16"))
    reduce_dtype_name = str(mixed_precision_config.get("reduce_dtype", "float32"))
    output_dtype_name = mixed_precision_config.get("output_dtype")
    enabled = bool(mixed_precision_config.get("enabled", True))
    backend = str(train_config.get("comm_backend") or "hccl")
    if cpu_offload and ":" not in backend:
        backend = f"cpu:gloo,{platform.device_type()}:{backend}"

    optimizer_kwargs = {
        "adamw_lr": float(optimizer_config.get("lr", 1.0e-6)),
        "adamw_weight_decay": float(optimizer_config.get("weight_decay", 0.0)),
        "adamw_betas": tuple(optimizer_config.get("betas", (0.9, 0.999))),
        "adamw_eps": float(optimizer_config.get("eps", 1.0e-8)),
    }
    if optimizer_config.get("foreach") is not None:
        optimizer_kwargs["foreach"] = bool(optimizer_config["foreach"])

    activation_checkpoint_value = accelerator_config.get(
        "activation_checkpoint", "off"
    )
    if isinstance(activation_checkpoint_value, bool):
        activation_checkpoint = "full" if activation_checkpoint_value else "off"
    else:
        activation_checkpoint = str(activation_checkpoint_value).lower()
    if activation_checkpoint not in ("off", "full", "selective"):
        raise ValueError(
            "train.accelerator.activation_checkpoint must be off, full, selective, "
            f"or a Boolean, got {activation_checkpoint_value!r}"
        )
    return TrainerConfig(
        model=Target(
            HyperAutoModelForCausalLM.from_pretrained,
            target_path=(
                "hyper_models._transformers.HyperAutoModelForCausalLM.from_pretrained"
            ),
            pretrained_model_name_or_path=str(model_config["weights_path"]),
            torch_dtype=param_dtype_name if enabled else "float32",
            attn_implementation=str(model_config.get("attn_implementation", "sdpa")),
            force_hf=True,
            local_files_only=True,
            trust_remote_code=True,
        ),
        optimizer=Target(
            AdamW,
            target_path="hyper_models.components.optim.optimizer.AdamW",
            adamw_config=optimizer_kwargs,
            no_decay_params=["bias", "norm", "ln_"],
        ),
        lr_scheduler=Target(
            MultiLRScheduler,
            target_path=(
                "hyper_models.components.optim.lr_scheduler.MultiLRScheduler"
            ),
            lr_decay_style=str(optimizer_config.get("lr_decay_style", "constant")),
            lr_config={
                "lr_warmup_ratio": float(optimizer_config.get("lr_warmup_ratio", 0.0)),
                "min_lr": float(optimizer_config.get("lr_min", 0.0)),
            },
        ),
        training=TrainingConfig(
            train_iters=max_steps,
            num_train_epochs=1,
            global_batch_size=dp_shard * prompt_batch_size,
            micro_batch_size=prompt_batch_size,
            backend=backend,
            max_grad_norm=float(optimizer_config.get("max_grad_norm", 1.0)),
            init_device=str(train_config.get("init_device", "meta")),
            loss_aggregation="token_weighted",
            seed=int(train_config.get("seed", 1234)),
        ),
        accelerator=AcceleratorConfig(
            tp_size=int(accelerator_config.get("tp", 1)),
            cp_size=int(accelerator_config.get("cp", 1)),
            ep_size=1,
            pp_size=int(accelerator_config.get("pp", 1)),
            sequence_parallel=False,
            loss_parallel=False,
        ),
        fsdp_config=FSDP2Config(
            dp_shard_size=dp_shard,
            mp_policy=(
                MixedPrecisionPolicy(
                    param_dtype=_resolve_dtype(param_dtype_name),
                    reduce_dtype=_resolve_dtype(reduce_dtype_name),
                    output_dtype=_resolve_dtype(output_dtype_name),
                )
                if enabled
                else None
            ),
            offload_policy=(
                CPUOffloadPolicy(
                    pin_memory=bool(accelerator_config.get("pin_memory", False))
                )
                if cpu_offload
                else None
            ),
            reshard_after_forward=bool(
                accelerator_config.get("reshard_after_forward", True)
            ),
            defer_fsdp_grad_sync=True,
            comm_fusion=bool(accelerator_config.get("comm_fusion", True)),
        ),
        mixed_precision=MixedPrecisionConfig(enabled=enabled),
        activation_checkpoint=ActivationCheckpointConfig(mode=activation_checkpoint),
        checkpoint=CheckpointingConfig(
            enabled=(
                bool(checkpoint_config.get("save_final", True))
                or int(checkpoint_config.get("save_steps", 0)) > 0
                or checkpoint_config.get("load_path") is not None
            ),
            checkpoint_dir=str(checkpoint_config["output_dir"]),
            save_consolidated="none",
            restore_from=checkpoint_config.get("load_path"),
        ),
    )
__all__ = [
    "build_model_registration",
    "build_runtime_config",
    "optional_mapping",
    "required_mapping",
    "uses_colocated_vllm",
    "validate_config",
    "validate_rollout_and_agentic",
]
