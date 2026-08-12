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
"""Synchronous requirements-driven Hyper-Parallel RL trainer."""

import functools
import json
import logging
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from transformers import AutoTokenizer

from rl.agentic import ENVIRONMENTS
from rl.algorithm import build_algorithm
from rl.contracts import ExperienceBatch, Message, PromptRecord
from rl.dataset import (
    ExperienceBuilder,
    PromptDataset,
    build_padded_evaluation_batches,
    collate_prompt_samples,
)
from rl.roles import (
    ActorManager,
    ActorModel,
    CriticManager,
    CriticModel,
    CriticUpdateMetrics,
    UpdateMetrics,
    attach_value_head,
    register_configured_model,
)
from rl.roles.rollout import PolicySnapshot, ROLLOUT_ENGINES, build_rollout_engine
from rl.roles.rollout.vllm_policy import normalize_model_implementation
from rl.roles.rollout.worker import RolloutManager
from rl.utils.monitoring import TrainingTracker, sanitize_config
from hyper_parallel import HSDPModule, destroy_process_group, get_platform, hsdp_sync_stream
from hyper_parallel.core.distributed_checkpoint import load as dcp_load
from hyper_parallel.platform.platform import PlatformType
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.callbacks.base import CheckpointCallback
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
from hyper_parallel.trainer.utils.discovery import discover_model_spec

platform = get_platform()
logger = logging.getLogger(__name__)

_EXPECTED_TOP_LEVEL = frozenset(
    ("model", "data", "rollout", "agentic", "algorithm", "evaluation", "train", "logging")
)
_GIB = 1024 ** 3


def _uses_colocated_vllm(config: Mapping[str, Any]) -> bool:
    """Return whether the selected rollout deploys one server per trainer NPU."""
    rollout = config.get("rollout", {})
    if not isinstance(rollout, Mapping) or rollout.get("engine") != "vllm":
        return False
    vllm_config = rollout.get("vllm", {})
    return isinstance(vllm_config, Mapping) and vllm_config.get("deployment", "disjoint") == "colocated"


def _iter_state_tensors(value: Any):
    """Yield tensors recursively from one optimizer state value."""
    if platform.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_state_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_state_tensors(item)


def _required_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one required mapping-valued config section."""
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def _optional_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one optional mapping-valued config section."""
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def _path_value(section: Mapping[str, Any], name: str) -> str:
    """Resolve and validate one required path-like config field."""
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration field '{name}' must be a non-empty path string")
    return value


def _validate_model_and_data_paths(
    model: Mapping[str, Any],
    data: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> None:
    """Validate model identity and all enabled local data paths."""
    model_name = model.get("name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model.name must be a non-empty string")
    path_checks = (
        (Path(_path_value(model, "weights_path")), "Model weights directory", True),
        (Path(_path_value(model, "tokenizer_path")), "Tokenizer directory", True),
        (Path(_path_value(data, "train_path")), "Training parquet file", False),
    )
    for path, description, must_be_directory in path_checks:
        exists = path.is_dir() if must_be_directory else path.is_file()
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
    algorithm: Any,
) -> None:
    """Validate update, mini-batch, response, and token counts."""
    positive_train_fields = (
        "max_steps",
        "prompt_batch_size",
        "micro_batch_size",
        "response_mini_batch_size",
    )
    for field in positive_train_fields:
        default_value = 1 if field == "prompt_batch_size" else 0
        value = int(train.get(field, default_value))
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
    local_response_count = int(train.get("prompt_batch_size", 1)) * num_responses
    if mini_batch_size > local_response_count:
        raise ValueError(
            "train.response_mini_batch_size cannot exceed the local rollout response count: "
            f"mini_batch={mini_batch_size}, responses={local_response_count}"
        )
    if int(rollout.get("max_new_tokens", 0)) <= 0:
        raise ValueError("rollout.max_new_tokens must be positive")
    if int(data.get("max_prompt_length", 0)) <= 0:
        raise ValueError("data.max_prompt_length must be positive")
    if int(train.get("policy_update_epochs", 0)) <= 0:
        raise ValueError("train.policy_update_epochs must be positive")
    learning_gate = train.get("learning_gate", {})
    if not isinstance(learning_gate, Mapping):
        raise ValueError("train.learning_gate must be a mapping")
    if float(learning_gate.get("min_gradient_norm", 0.0)) < 0:
        raise ValueError("train.learning_gate.min_gradient_norm must be non-negative")


def _validate_evaluation(evaluation: Mapping[str, Any]) -> None:
    """Validate optional evaluation limits and progress settings."""
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


def _validate_rollout_and_agentic(
    rollout: Mapping[str, Any],
    agentic: Mapping[str, Any],
    accelerator: Mapping[str, Any],
) -> None:
    """Validate selected rollout and environment implementations."""
    engine_name = rollout.get("engine")
    if engine_name not in ROLLOUT_ENGINES.names:
        raise ValueError(
            f"Unknown rollout.engine '{engine_name}'; available={ROLLOUT_ENGINES.names}"
        )
    if engine_name == "vllm":
        if platform.platform_type != PlatformType.PYTORCH or platform.device_type() != "npu":
            raise ValueError("The vLLM-Ascend rollout backend requires the Torch NPU platform")
        vllm_config = _optional_mapping(rollout, "vllm")
        deployment = str(vllm_config.get("deployment", "disjoint"))
        if deployment not in ("disjoint", "colocated"):
            raise ValueError(
                "rollout.vllm.deployment must be 'disjoint' or 'colocated', "
                f"got {deployment!r}"
            )
        rollout_tp = int(vllm_config.get("tensor_parallel_size", 1))
        if rollout_tp <= 0:
            raise ValueError("rollout.vllm.tensor_parallel_size must be positive")
        if str(vllm_config.get("dtype", "bfloat16")) not in ("bfloat16", "bf16"):
            raise ValueError("The Hyper Qwen3.5 vLLM adapter requires bfloat16")
        host = str(vllm_config.get("host", "127.0.0.1"))
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("The external vLLM server must bind to loopback")
        gpu_memory_utilization = float(vllm_config.get("gpu_memory_utilization", 0.9))
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError("rollout.vllm.gpu_memory_utilization must be between 0 and 1")
        if deployment == "colocated":
            if rollout_tp != 1:
                raise ValueError("The initial colocated rollout path supports tensor_parallel_size=1")
            if int(accelerator.get("dp_shard", 1)) <= 1:
                raise ValueError("Colocated rollout requires multi-rank FSDP with train.accelerator.dp_shard > 1")
            if not bool(accelerator.get("cpu_offload", False)):
                raise ValueError("Colocated rollout requires train.accelerator.cpu_offload=true")
            if not bool(accelerator.get("reshard_after_forward", True)):
                raise ValueError("Colocated rollout requires train.accelerator.reshard_after_forward=true")
            if vllm_config.get("visible_devices") is not None:
                raise ValueError(
                    "Colocated rollout derives one physical NPU from each trainer rank; "
                    "remove rollout.vllm.visible_devices"
                )
            base_port = vllm_config.get("port")
            if base_port is None:
                raise ValueError("Colocated rollout requires an explicit rollout.vllm.port base")
            final_port = int(base_port) + int(accelerator.get("dp_shard", 1)) - 1
            if int(base_port) <= 0 or final_port > 65535:
                raise ValueError("rollout.vllm.port range exceeds valid TCP ports")
        else:
            if int(accelerator.get("dp_shard", 1)) != 1:
                raise ValueError(
                    "The external vLLM HCCL refitter currently supports train.accelerator.dp_shard=1"
                )
            visible_devices = vllm_config.get("visible_devices")
            if visible_devices is None:
                raise ValueError("rollout.vllm.visible_devices must select the external server NPUs")
            device_ids = [device.strip() for device in str(visible_devices).split(",")]
            if not all(device_ids) or len(device_ids) < rollout_tp:
                raise ValueError(
                    "rollout.vllm.visible_devices must contain at least one device ID per TP rank"
                )
            training_visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if training_visible_devices:
                training_device_ids = [
                    device.strip() for device in training_visible_devices.split(",")
                ]
                if local_rank >= len(training_device_ids):
                    raise ValueError(
                        f"LOCAL_RANK={local_rank} exceeds ASCEND_RT_VISIBLE_DEVICES={training_visible_devices!r}"
                    )
                training_device_id = training_device_ids[local_rank]
            else:
                training_device_id = str(local_rank)
            if training_device_id in device_ids:
                raise ValueError(
                    "The external vLLM server must use NPUs disjoint from the trainer: "
                    f"training_device={training_device_id}, rollout_devices={device_ids}"
                )
        normalize_model_implementation(vllm_config.get("model_implementation", "hyper"))
        request_concurrency = int(vllm_config.get("request_concurrency", 1))
        if request_concurrency <= 0:
            raise ValueError("rollout.vllm.request_concurrency must be positive")
        for field in ("kv_cache_memory_bytes", "max_model_len", "max_num_seqs", "max_num_batched_tokens"):
            if field in vllm_config and int(vllm_config[field]) <= 0:
                raise ValueError(f"rollout.vllm.{field} must be positive")
    rollout_seed = rollout.get("seed")
    if rollout_seed is not None and int(rollout_seed) < 0:
        raise ValueError("rollout.seed must be non-negative or null")
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


def _validate_topology(accelerator: Mapping[str, Any]) -> None:
    """Validate the minimal demo's pure-FSDP topology."""
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
        key: value
        for key, value in topology.items()
        if key != "dp_shard" and value != 1
    }
    if unsupported:
        raise ValueError(f"Hyper-RL demo supports pure FSDP only; invalid topology={unsupported}")


def _validate_checkpoint(checkpoint: Mapping[str, Any], algorithm: Any) -> None:
    """Validate checkpoint policy and create its configured output directory."""
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
    Path(_path_value(checkpoint, "output_dir")).mkdir(parents=True, exist_ok=True)


def _validate_logging(logging_config: Mapping[str, Any]) -> None:
    """Validate enabled experiment tracking backends and W&B mode."""
    backends = logging_config.get("backends", ())
    if not isinstance(backends, list) or not backends:
        raise ValueError("logging.backends must be a non-empty list")
    unsupported_backends = set(backends) - {"console", "wandb"}
    if unsupported_backends:
        raise ValueError(f"Unsupported logging backends: {sorted(unsupported_backends)}")
    wandb_config = _required_mapping(logging_config, "wandb")
    if wandb_config.get("mode", "auto") not in {
        "auto",
        "online",
        "offline",
        "disabled",
    }:
        raise ValueError(f"Unsupported W&B mode: {wandb_config.get('mode')}")


class SyncTrainer(BaseTrainer):
    """Minimal RL trainer extending Hyper-Parallel's model/FSDP skeleton.

    Args:
        resolved_config: Fully merged Hyper-RL YAML configuration.
    """

    def __init__(self, resolved_config: Mapping[str, Any]) -> None:
        """Validate configuration and build the complete distributed runtime."""
        self.resolved_config = deepcopy(dict(resolved_config))
        self.algorithm = build_algorithm(_required_mapping(self.resolved_config, "algorithm"))
        self._validate_config(self.resolved_config)
        self.model_registration = register_configured_model(
            _required_mapping(self.resolved_config, "model")
        )
        self._defer_checkpoint_errors = True
        base_config = self._build_base_config(self.resolved_config)
        discover_model_spec(base_config.model.name)
        super().__init__(base_config)
        self._runtime_started = False
        self._tracker: Optional[TrainingTracker] = None
        try:
            self._setup()
            self._runtime_started = True
            self._validate_runtime_topology()
            self._build_runtime()
        except Exception:
            self._cleanup_distributed()
            raise

    @staticmethod
    def _validate_config(config: Mapping[str, Any]) -> None:
        """Validate Hyper-RL-only configuration before distributed startup."""
        unknown = set(config) - _EXPECTED_TOP_LEVEL
        if unknown:
            raise ValueError(f"Unsupported top-level configuration keys: {sorted(unknown)}")
        model = _required_mapping(config, "model")
        data = _required_mapping(config, "data")
        rollout = _required_mapping(config, "rollout")
        agentic = _required_mapping(config, "agentic")
        evaluation = _required_mapping(config, "evaluation")
        train = _required_mapping(config, "train")
        accelerator = _required_mapping(train, "accelerator")
        checkpoint = _required_mapping(train, "checkpoint")
        logging_config = _required_mapping(config, "logging")
        algorithm = build_algorithm(_required_mapping(config, "algorithm"))
        _validate_model_and_data_paths(model, data, evaluation)
        _validate_training_sizes(train, rollout, data, algorithm)
        _validate_evaluation(evaluation)
        _validate_rollout_and_agentic(rollout, agentic, accelerator)
        _validate_topology(accelerator)
        _validate_checkpoint(checkpoint, algorithm)
        _validate_logging(logging_config)

    @staticmethod
    def _build_base_config(config: Mapping[str, Any]) -> HyperTrainerConfig:
        """Translate Hyper-RL YAML into the existing Hyper-Parallel trainer schema."""
        model_config = _required_mapping(config, "model")
        data_config = _required_mapping(config, "data")
        train_config = _required_mapping(config, "train")
        accelerator_config = _required_mapping(train_config, "accelerator")
        optimizer_config = _required_mapping(train_config, "optimizer")
        mixed_precision_config = _required_mapping(train_config, "mixed_precision")
        checkpoint_config = _required_mapping(train_config, "checkpoint")
        rollout_config = _required_mapping(config, "rollout")
        agentic_config = _required_mapping(config, "agentic")

        max_steps = int(train_config["max_steps"])
        prompt_batch_size = int(train_config.get("prompt_batch_size", 1))
        save_final = bool(checkpoint_config.get("save_final", True))
        configured_save_steps = int(checkpoint_config.get("save_steps", 0))
        effective_save_steps = configured_save_steps
        if save_final and effective_save_steps == 0:
            effective_save_steps = max_steps
        dp_shard = int(accelerator_config["dp_shard"])
        cpu_offload = bool(accelerator_config.get("cpu_offload", False))
        comm_backend = train_config.get("comm_backend")
        if cpu_offload and comm_backend in (None, "hccl"):
            comm_backend = "cpu:gloo,npu:hccl"
        model = ModelConfig(
            name=str(model_config["name"]),
            weights_path=str(model_config["weights_path"]),
            tokenizer_path=str(model_config["tokenizer_path"]),
            config_overrides=model_config.get("config_overrides"),
        )
        max_turns = int(agentic_config["max_turns"])
        per_turn_tokens = int(rollout_config["max_new_tokens"]) + int(
            agentic_config.get("max_observation_tokens", 0)
        )
        data = DataConfig(
            type="dummy",
            train_path=str(data_config["train_path"]),
            max_seq_len=int(data_config["max_prompt_length"]) + max_turns * per_turn_tokens,
            num_workers=int(data_config.get("num_workers", 0)),
            prefetch_factor=data_config.get("prefetch_factor"),
            pin_memory=bool(data_config.get("pin_memory", True)),
            shuffle=bool(data_config.get("shuffle", True)),
        )
        accelerator = AcceleratorConfig(
            dp_replicate=int(accelerator_config.get("dp_replicate", 1)),
            dp_shard=dp_shard,
            tp=int(accelerator_config.get("tp", 1)),
            cp=int(accelerator_config.get("cp", 1)),
            pp=int(accelerator_config.get("pp", 1)),
            ep=1,
            etp=1,
            reshard_after_forward=bool(accelerator_config.get("reshard_after_forward", True)),
            comm_fusion=bool(accelerator_config.get("comm_fusion", True)),
            cpu_offload=cpu_offload,
        )
        mixed_precision = MixedPrecisionConfig(
            enabled=bool(mixed_precision_config.get("enabled", True)),
            param_dtype=str(mixed_precision_config.get("param_dtype", "bfloat16")),
            reduce_dtype=str(mixed_precision_config.get("reduce_dtype", "float32")),
            output_dtype=mixed_precision_config.get("output_dtype"),
        )
        optimizer = OptimizerConfig(
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
        )
        checkpoint = CheckpointConfig(
            output_dir=str(checkpoint_config["output_dir"]),
            save_steps=effective_save_steps,
            save_hf_weights=False,
            load_path=checkpoint_config.get("load_path"),
            save_async=False,
        )
        train = TrainConfig(
            max_steps=max_steps,
            num_train_epochs=1,
            global_batch_size=dp_shard * prompt_batch_size,
            micro_batch_size=prompt_batch_size,
            seed=int(train_config.get("seed", 1234)),
            backend="torch",
            init_device=str(train_config.get("init_device", "meta")),
            comm_backend=comm_backend,
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            accelerator=accelerator,
            mixed_precision=mixed_precision,
            optimizer=optimizer,
            checkpoint=checkpoint,
        )
        return HyperTrainerConfig(model=model, data=data, train=train)

    def _validate_runtime_topology(self) -> None:
        """Validate torchrun world size against the requested FSDP shard count."""
        world_size = platform.get_world_size()
        dp_shard = int(self.args.train.accelerator.dp_shard)
        if world_size != dp_shard:
            raise ValueError(
                f"torchrun world size must equal train.accelerator.dp_shard: "
                f"world_size={world_size}, dp_shard={dp_shard}"
            )
        if _uses_colocated_vllm(self.resolved_config):
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
            if local_world_size != world_size:
                raise ValueError(
                    "The initial colocated rollout DP path is single-node only: "
                    f"world_size={world_size}, local_world_size={local_world_size}"
                )
            visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
            if visible_devices is not None:
                device_ids = [device.strip() for device in visible_devices.split(",")]
                if len(device_ids) < local_world_size or len(set(device_ids[:local_world_size])) != local_world_size:
                    raise ValueError(
                        "Colocated rollout requires one unique visible physical NPU per local trainer rank: "
                        f"ASCEND_RT_VISIBLE_DEVICES={visible_devices!r}"
                    )

    def _build_runtime(self) -> None:
        """Build tokenizer, data, requirement-selected roles, optimizers, and tracking."""
        self._build_tokenizer_and_data()
        self._build_models_and_optimizers()
        self._build_rollout_runtime()
        self._build_learning_roles()
        self.checkpoint_callback = CheckpointCallback(self)
        self._build_tracker()

    def _build_tokenizer_and_data(self) -> None:
        """Build the shared tokenizer, train split, and optional evaluation split."""
        model_config = _required_mapping(self.resolved_config, "model")
        data_config = _required_mapping(self.resolved_config, "data")
        evaluation_config = _required_mapping(self.resolved_config, "evaluation")
        self._evaluation_enabled = bool(evaluation_config.get("enabled", True))
        tokenizer_path = str(model_config["tokenizer_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for response truncation")
        self.tokenizer.padding_side = "left"

        dataset_kwargs = {
            "tokenizer": self.tokenizer,
            "max_prompt_length": int(data_config["max_prompt_length"]),
            "prompt_column": str(data_config.get("prompt_column", "question")),
            "answer_column": str(data_config.get("answer_column", "answer")),
        }
        self.train_dataset = PromptDataset(
            parquet_path=str(data_config["train_path"]),
            max_samples=(
                None
                if data_config.get("max_train_samples") is None
                else int(data_config["max_train_samples"])
            ),
            **dataset_kwargs,
        )
        self.test_dataset: Optional[PromptDataset] = None
        if self._evaluation_enabled:
            self.test_dataset = PromptDataset(
                parquet_path=str(data_config["test_path"]),
                **dataset_kwargs,
            )
        self.collate_fn = functools.partial(
            collate_prompt_samples,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )
        self._build_dataloader()

    def _build_models_and_optimizers(self) -> None:
        """Build requirement-selected models and their independent optimizers."""
        actor_module = self._build_one_parallel_model(frozen=False)
        reference_module = None
        if self.algorithm.requirements.roles.reference:
            reference_module = self._build_one_parallel_model(frozen=True)
        critic_model = None
        if self.algorithm.requirements.roles.critic:
            critic_model = self._build_one_critic_model()
        self.model = ActorModel(actor_module)
        self.reference_model = (
            None if reference_module is None else ActorModel(reference_module)
        )
        if self.reference_model is not None:
            self.reference_model.eval()
        self.critic_model = critic_model
        self.optimizer = None
        self.lr_scheduler = None
        self._build_optimizer()
        self._build_lr_scheduler()
        actor_optimizer = self.optimizer
        actor_lr_scheduler = self.lr_scheduler
        self.critic_optimizer = None
        self.critic_lr_scheduler = None
        if self.critic_model is not None:
            self.model = self.critic_model
            self._build_optimizer()
            self._build_lr_scheduler()
            self.critic_optimizer = self.optimizer
            self.critic_lr_scheduler = self.lr_scheduler
            self.model = ActorModel(actor_module)
            self.optimizer = actor_optimizer
            self.lr_scheduler = actor_lr_scheduler
        self._build_training_context()

    def _build_rollout_runtime(self) -> None:
        """Build the selected generation engine and train/evaluation rollout managers."""
        rollout_config = _required_mapping(self.resolved_config, "rollout")
        agentic_config = _required_mapping(self.resolved_config, "agentic")
        evaluation_config = _required_mapping(self.resolved_config, "evaluation")
        self.rollout_engine = build_rollout_engine(
            rollout_config,
            self.model_registration,
            actor=self.model,
        )
        self.rollout_manager = RolloutManager(
            engine=self.rollout_engine,
            tokenizer=self.tokenizer,
            environment_name=str(agentic_config["environment"]),
            num_return_sequences=int(rollout_config["num_return_sequences"]),
            max_turns=int(agentic_config["max_turns"]),
            max_observation_tokens=int(agentic_config["max_observation_tokens"]),
            max_new_tokens=int(rollout_config["max_new_tokens"]),
            temperature=float(rollout_config.get("temperature", 1.0)),
            top_p=float(rollout_config.get("top_p", 1.0)),
            top_k=int(rollout_config.get("top_k", 0)),
            pad_token_id=int(self.tokenizer.pad_token_id),
            eos_token_id=int(self.tokenizer.eos_token_id),
            do_sample=True,
            collect_old_log_probs=self.algorithm.requirements.data.rollout_log_probs,
            seed=(None if rollout_config.get("seed") is None else int(rollout_config["seed"])),
        )
        self._evaluation_batch_size = int(evaluation_config.get("batch_size", 1))
        self._evaluation_max_samples = evaluation_config.get("max_samples")
        if self._evaluation_max_samples is not None:
            self._evaluation_max_samples = int(self._evaluation_max_samples)
        self._evaluation_log_samples = int(evaluation_config.get("log_samples", 0))
        self._evaluation_progress_steps = int(evaluation_config.get("progress_steps", 0))
        self._last_evaluation_step = -1
        self.evaluation_rollout_manager: Optional[RolloutManager] = None
        if self._evaluation_enabled:
            self.evaluation_rollout_manager = RolloutManager(
                engine=self.rollout_engine,
                tokenizer=self.tokenizer,
                environment_name=str(agentic_config["environment"]),
                num_return_sequences=1,
                max_turns=int(agentic_config["max_turns"]),
                max_observation_tokens=int(agentic_config["max_observation_tokens"]),
                max_new_tokens=int(evaluation_config["max_new_tokens"]),
                temperature=float(evaluation_config.get("temperature", 1.0)),
                top_p=float(evaluation_config.get("top_p", 1.0)),
                top_k=int(evaluation_config.get("top_k", 0)),
                pad_token_id=int(self.tokenizer.pad_token_id),
                eos_token_id=int(self.tokenizer.eos_token_id),
                do_sample=bool(evaluation_config.get("do_sample", False)),
            )

    def _build_learning_roles(self) -> None:
        """Build actor, experience, and optional Critic coordination roles."""
        train_config = _required_mapping(self.resolved_config, "train")
        optimizer_config = _required_mapping(train_config, "optimizer")
        self.actor_manager = ActorManager(
            actor=self.model,
            algorithm=self.algorithm,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=self.device,
            dp_group_info=self._dp_group_info,
            dp_size=self.parallel_dims.dp_size,
            micro_batch_size=int(train_config["micro_batch_size"]),
            response_mini_batch_size=int(train_config["response_mini_batch_size"]),
            policy_update_epochs=int(train_config["policy_update_epochs"]),
            max_grad_norm=float(optimizer_config.get("max_grad_norm", 1.0)),
        )
        self.experience_builder = ExperienceBuilder(
            algorithm=self.algorithm,
            reference=self.reference_model,
            critic=self.critic_model,
            micro_batch_size=int(train_config["micro_batch_size"]),
        )
        self.critic_manager: Optional[CriticManager] = None
        if self.critic_model is not None:
            if self.critic_optimizer is None:
                raise RuntimeError("Critic optimizer was not initialized")
            self.critic_manager = CriticManager(
                critic=self.critic_model,
                algorithm=self.algorithm,
                optimizer=self.critic_optimizer,
                lr_scheduler=self.critic_lr_scheduler,
                device=self.device,
                dp_group_info=self._dp_group_info,
                dp_size=self.parallel_dims.dp_size,
                micro_batch_size=int(train_config["micro_batch_size"]),
                response_mini_batch_size=int(train_config["response_mini_batch_size"]),
                update_epochs=int(
                    train_config.get("critic_update_epochs", train_config["policy_update_epochs"])
                ),
                max_grad_norm=float(optimizer_config.get("max_grad_norm", 1.0)),
            )

    def _build_one_parallel_model(self, frozen: bool) -> platform.Module:
        """Build, optionally freeze, FSDP-wrap, materialize, and load one model."""
        self._build_model()
        if frozen:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self._build_parallelized_model()
        if frozen:
            self.model.eval()
        return self.model

    def _build_one_critic_model(self) -> CriticModel:
        """Build an independent backbone and attach its registered value capability."""
        self._build_model()
        attach_value_head(self.model, str(self.model_registration.hyper_model_name))
        self._build_parallelized_model()
        return CriticModel(self.model)

    def _build_tracker(self) -> None:
        """Initialize console/W&B tracking on global rank zero only."""
        logging_config = _required_mapping(self.resolved_config, "logging")
        wandb_config = _required_mapping(logging_config, "wandb")
        self._log_steps = int(logging_config.get("log_steps", 1))
        self._log_samples = int(logging_config.get("log_samples", 0))
        if self._log_steps <= 0:
            raise ValueError(f"logging.log_steps must be positive, got {self._log_steps}")
        if self._log_samples < 0:
            raise ValueError(f"logging.log_samples must be non-negative, got {self._log_samples}")
        self._tracker = TrainingTracker(
            rank=platform.get_rank(),
            world_size=platform.get_world_size(),
            backends=tuple(logging_config["backends"]),
            project_name=str(logging_config.get("project_name", "hyper-rl")),
            experiment_name=str(logging_config.get("experiment_name", "hyper_rl_run")),
            resolved_config=self.resolved_config,
            wandb_mode=str(wandb_config.get("mode", "auto")),
            wandb_entity=wandb_config.get("entity"),
            wandb_directory=str(wandb_config.get("directory", "outputs/wandb")),
        )

    def _next_batch(self, data_iterator: Any) -> tuple[Mapping[str, Any], Any]:
        """Fetch one batch, advancing the distributed sampler epoch on exhaustion."""
        try:
            return next(data_iterator), data_iterator
        except StopIteration:
            self.state.epoch += 1
            self.sampler.set_epoch(self.state.epoch)
            data_iterator = iter(self.train_dataloader)
            return next(data_iterator), data_iterator

    @staticmethod
    def _build_prompt_records(
        batch: Mapping[str, Any],
        input_ids: Any,
        attention_mask: Any,
    ) -> tuple[PromptRecord, ...]:
        """Attach exact initial observation tokens to environment inputs."""
        records = []
        for index in range(input_ids.shape[0]):
            valid_ids = input_ids[index][attention_mask[index].bool()].detach()
            records.append(
                PromptRecord(
                    prompt_id=str(int(batch["sample_indices"][index])),
                    messages=(Message("user", batch["prompts"][index]),),
                    ground_truth=batch["ground_truths"][index],
                    metadata={"input_ids": valid_ids},
                )
            )
        return tuple(records)

    def _checkpoint_will_save(self, step: int) -> bool:
        """Return whether periodic or final checkpoint policy saves this step."""
        checkpoint_config = _required_mapping(
            _required_mapping(self.resolved_config, "train"),
            "checkpoint",
        )
        save_steps = int(checkpoint_config.get("save_steps", 0))
        periodic_save = save_steps > 0 and step % save_steps == 0
        final_save = (
            bool(checkpoint_config.get("save_final", True))
            and step == self.state.max_steps
        )
        return periodic_save or final_save

    def _run_validation(
        self,
        step: int,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Evaluate greedy rule-reward accuracy on the evaluation split."""
        test_dataset, rollout_manager, rank, world_size, batches = (
            self._prepare_validation(step)
        )
        local_record = self._collect_validation_record(
            step,
            test_dataset,
            rollout_manager,
            rank,
            batches,
        )
        gathered: list[Optional[dict[str, Any]]] = [None] * world_size
        platform.all_gather_object(gathered, local_record)
        self._last_evaluation_step = step
        if rank != 0:
            return {}, []
        records = [record for record in gathered if record is not None]
        return self._summarize_validation(step, records)

    def _prepare_validation(
        self,
        step: int,
    ) -> tuple[PromptDataset, RolloutManager, int, int, list[list[tuple[int, bool]]]]:
        """Resolve validation dependencies, distribute batches, and log its start."""
        if self.test_dataset is None or self.evaluation_rollout_manager is None:
            raise RuntimeError("Validation runtime was not initialized")
        test_dataset = self.test_dataset
        rollout_manager = self.evaluation_rollout_manager
        rank = platform.get_rank()
        world_size = platform.get_world_size()
        batches = build_padded_evaluation_batches(
            dataset_size=len(test_dataset),
            num_replicas=world_size,
            rank=rank,
            batch_size=self._evaluation_batch_size,
            max_samples=self._evaluation_max_samples,
        )
        requested_samples = (
            len(test_dataset)
            if self._evaluation_max_samples is None
            else min(len(test_dataset), self._evaluation_max_samples)
        )
        if rank == 0:
            logger.info(
                "step=%d validation started: samples=%d, batches_per_rank=%d, "
                "batch_size_per_rank=%d",
                step,
                requested_samples,
                len(batches),
                self._evaluation_batch_size,
            )
        return test_dataset, rollout_manager, rank, world_size, batches

    def _evaluate_validation_batch(
        self,
        step: int,
        rank: int,
        entries: list[tuple[int, bool]],
        test_dataset: PromptDataset,
        rollout_manager: RolloutManager,
        sample_limit: int,
    ) -> dict[str, Any]:
        """Generate and score one fixed-size per-rank validation batch."""
        samples = [test_dataset[sample_index] for sample_index, _ in entries]
        batch = self.collate_fn(samples)
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        rollout = rollout_manager.generate(
            prompt_records=self._build_prompt_records(batch, input_ids, attention_mask),
            policy_version=step,
        )
        record: dict[str, Any] = {
            "correct": 0.0,
            "total": 0,
            "generated_tokens": 0,
            "response_length": 0,
            "generation_seconds": rollout.generation_seconds,
            "samples": [],
        }
        for local_index, ((_, is_valid), response) in enumerate(
            zip(entries, rollout.responses)
        ):
            if not is_valid:
                continue
            reward = float(rollout.rewards[local_index].item())
            response_length = int(rollout.action_mask[local_index].sum().item())
            record["correct"] += reward
            record["total"] += 1
            record["generated_tokens"] += response_length
            record["response_length"] += response_length
            if len(record["samples"]) < sample_limit:
                record["samples"].append(
                    {
                        "step": step,
                        "rank": rank,
                        "prompt": batch["prompts"][local_index],
                        "response": response,
                        "ground_truth": batch["ground_truths"][local_index],
                        "extracted_answer": rollout.trajectories[
                            local_index
                        ].metadata.get("extracted_answer"),
                        "reward": reward,
                    }
                )
        return record

    def _collect_validation_record(
        self,
        step: int,
        test_dataset: PromptDataset,
        rollout_manager: RolloutManager,
        rank: int,
        batches: list[list[tuple[int, bool]]],
    ) -> dict[str, Any]:
        """Accumulate local validation metrics across synchronized batches."""
        record: dict[str, Any] = {
            "correct": 0.0,
            "total": 0,
            "generated_tokens": 0,
            "response_length": 0,
            "generation_seconds": 0.0,
            "samples": [],
        }
        for batch_index, entries in enumerate(batches, start=1):
            sample_limit = self._evaluation_log_samples - len(record["samples"])
            batch_record = self._evaluate_validation_batch(
                step,
                rank,
                entries,
                test_dataset,
                rollout_manager,
                sample_limit,
            )
            for key in (
                "correct",
                "total",
                "generated_tokens",
                "response_length",
                "generation_seconds",
            ):
                record[key] += batch_record[key]
            record["samples"].extend(batch_record["samples"])
            self._log_validation_progress(step, rank, batch_index, len(batches))
        return record

    def _log_validation_progress(
        self,
        step: int,
        rank: int,
        batch_index: int,
        batch_count: int,
    ) -> None:
        """Log configured validation progress from global rank zero."""
        should_log = self._evaluation_progress_steps > 0 and (
            batch_index % self._evaluation_progress_steps == 0
            or batch_index == batch_count
        )
        if rank == 0 and should_log:
            logger.info(
                "step=%d validation progress: %d/%d batches per rank",
                step,
                batch_index,
                batch_count,
            )

    def _summarize_validation(
        self,
        step: int,
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Reduce gathered validation records into rank-zero metrics and samples."""
        correct = sum(float(record["correct"]) for record in records)
        total = sum(int(record["total"]) for record in records)
        generated_tokens = sum(int(record["generated_tokens"]) for record in records)
        response_length = sum(int(record["response_length"]) for record in records)
        generation_seconds = max(
            float(record["generation_seconds"]) for record in records
        )
        accuracy = correct / max(total, 1)
        metrics = {
            "validation/accuracy": accuracy,
            "validation/correct": correct,
            "validation/total": float(total),
            "validation/response_length_mean": response_length / max(total, 1),
            "validation/generated_tokens": float(generated_tokens),
            "validation/generation_seconds": generation_seconds,
            "validation/tokens_per_second": (
                generated_tokens / max(generation_seconds, 1.0e-9)
            ),
        }
        logger.info(
            "step=%d validation completed: accuracy=%.6f (%d/%d)",
            step,
            accuracy,
            int(correct),
            total,
        )
        return metrics, self._round_robin_samples(
            records,
            limit=self._evaluation_log_samples,
        )

    def _rollout_statistics(
        self,
        rollout: ExperienceBatch,
        batch: Mapping[str, Any],
        step: int,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Gather per-rank rollout statistics and bounded samples on rank zero."""
        response_lengths = rollout.action_mask.sum(dim=-1).detach().cpu().tolist()
        reward_values = rollout.rewards.detach().cpu().tolist()
        rank = platform.get_rank()
        batch_rows = {
            str(int(sample_index)): row
            for row, sample_index in enumerate(batch["sample_indices"])
        }
        local_samples = []
        for index, response in enumerate(rollout.responses[:self._log_samples]):
            trajectory = rollout.trajectories[index]
            batch_row = batch_rows[trajectory.prompt_id]
            local_samples.append(
                {
                    "step": step,
                    "rank": rank,
                    "prompt": batch["prompts"][batch_row],
                    "response": response,
                    "ground_truth": batch["ground_truths"][batch_row],
                    "extracted_answer": trajectory.metadata.get("extracted_answer"),
                    "reward": float(reward_values[index]),
                }
            )
        group_rewards: dict[str, list[float]] = {}
        for trajectory, reward in zip(rollout.trajectories, reward_values):
            group_id = trajectory.group_id or trajectory.prompt_id
            group_rewards.setdefault(group_id, []).append(float(reward))
        local = {
            "reward_sum": float(sum(reward_values)),
            "reward_count": len(reward_values),
            "reward_min": float(min(reward_values)),
            "reward_max": float(max(reward_values)),
            "zero_std_groups": sum(
                int(max(rewards) == min(rewards))
                for rewards in group_rewards.values()
            ),
            "length_sum": int(sum(response_lengths)),
            "length_max": int(max(response_lengths, default=0)),
            "generated_tokens": int(rollout.action_mask.sum().item()),
            "generation_seconds": float(rollout.generation_seconds),
            "samples": local_samples,
        }
        gathered: list[Optional[dict[str, Any]]] = [None] * platform.get_world_size()
        platform.all_gather_object(gathered, local)
        if rank != 0:
            return {}, []
        records = [record for record in gathered if record is not None]
        reward_sum = sum(record["reward_sum"] for record in records)
        reward_count = sum(record["reward_count"] for record in records)
        generated_tokens = sum(record["generated_tokens"] for record in records)
        generation_seconds = max(record["generation_seconds"] for record in records)
        length_sum = sum(record["length_sum"] for record in records)
        reward_mean = reward_sum / max(reward_count, 1)
        metrics = {
            "reward/mean": reward_mean,
            "reward/min": min(record["reward_min"] for record in records),
            "reward/max": max(record["reward_max"] for record in records),
            "reward/accuracy": reward_mean,
            "reward/zero_std_groups": float(sum(record["zero_std_groups"] for record in records)),
            "rollout/response_length_mean": length_sum / max(reward_count, 1),
            "rollout/response_length_max": float(max(record["length_max"] for record in records)),
            "rollout/generated_tokens": float(generated_tokens),
            "rollout/generation_seconds": generation_seconds,
            "rollout/tokens_per_second": generated_tokens / max(generation_seconds, 1.0e-9),
        }
        return metrics, self._round_robin_samples(records)

    def _round_robin_samples(
        self,
        records: list[dict[str, Any]],
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Select a globally bounded sample table while representing multiple ranks."""
        effective_limit = self._log_samples if limit is None else limit
        selected = []
        sample_index = 0
        while len(selected) < effective_limit:
            added = False
            for record in records:
                samples = record["samples"]
                if sample_index < len(samples):
                    selected.append(samples[sample_index])
                    added = True
                    if len(selected) >= effective_limit:
                        break
            if not added:
                break
            sample_index += 1
        return selected

    @staticmethod
    def _system_memory_metrics() -> dict[str, float]:
        """Read peak device memory metrics when the backend exposes them."""
        handle = platform.get_device_handle(platform.device_type())
        allocated_fn = getattr(handle, "max_memory_allocated", None)
        reserved_fn = getattr(handle, "max_memory_reserved", None)
        allocated = float(allocated_fn()) / _GIB if allocated_fn is not None else 0.0
        reserved = float(reserved_fn()) / _GIB if reserved_fn is not None else 0.0
        return {
            "system/world_size": float(platform.get_world_size()),
            "system/max_memory_allocated_gb": allocated,
            "system/max_memory_reserved_gb": reserved,
        }

    def _build_metrics(
        self,
        update: UpdateMetrics,
        rollout_metrics: Mapping[str, float],
        critic_update: Optional[CriticUpdateMetrics] = None,
    ) -> dict[str, float]:
        """Combine globally reduced actor, rollout, and system diagnostics."""
        metrics = {
            "train/global_step": float(self.state.global_step),
            "train/learning_rate": update.learning_rate,
            "train/gradient_norm": update.gradient_norm,
            "train/total_loss": update.total_loss,
            "train/policy_loss": update.policy_loss,
            "train/kl_loss": update.kl_loss,
            "train/old_policy_kl": update.old_policy_kl,
            "train/old_current_log_ratio_abs": update.old_current_log_ratio_abs,
            "train/clip_fraction": update.clip_fraction,
            "train/optimizer_steps": float(update.optimizer_steps),
        }
        metrics.update(rollout_metrics)
        if critic_update is not None:
            metrics.update(
                {
                    "critic/value_loss": critic_update.value_loss,
                    "critic/gradient_norm": critic_update.gradient_norm,
                    "critic/learning_rate": critic_update.learning_rate,
                    "critic/optimizer_steps": float(critic_update.optimizer_steps),
                }
            )
        metrics.update(self._system_memory_metrics())
        policy_fingerprint = getattr(self.rollout_engine, "policy_fingerprint", None)
        if policy_fingerprint is not None:
            metrics["policy/version"] = float(self.rollout_engine.policy_version)
            metrics["policy/fingerprint_changed"] = float(
                bool(self.rollout_engine.policy_fingerprint_changed)
            )
        return metrics

    def _enforce_learning_gate(self, metrics: Mapping[str, float], step: int) -> None:
        """Fail a numerical acceptance run when learning invariants are absent."""
        train_config = _required_mapping(self.resolved_config, "train")
        gate = train_config.get("learning_gate", {})
        if not bool(gate.get("enabled", False)):
            return

        def validate() -> None:
            """Validate globally reduced metrics only on rank zero."""
            if platform.get_rank() != 0:
                return
            failures = []
            gradient_norm = float(metrics["train/gradient_norm"])
            minimum_gradient = float(gate.get("min_gradient_norm", 0.0))
            if not gradient_norm > minimum_gradient:
                failures.append(
                    f"Learning gate requires gradient_norm > {minimum_gradient}, got {gradient_norm}"
                )
            reward_minimum = float(metrics.get("reward/min", 0.0))
            reward_maximum = float(metrics.get("reward/max", 0.0))
            if bool(gate.get("require_mixed_rewards", False)) and not reward_maximum > reward_minimum:
                failures.append(
                    "Learning gate requires nonzero global reward variance, "
                    f"got reward/min={reward_minimum} and reward/max={reward_maximum}"
                )
            if bool(gate.get("require_fingerprint_change", False)) and not bool(
                metrics.get("policy/fingerprint_changed", 0.0)
            ):
                failures.append("Learning gate requires a changed replicated norm probe")
            if int(metrics.get("policy/version", -1)) != step:
                failures.append(
                    f"Learning gate expected policy version {step}, got {metrics.get('policy/version')}"
                )
            if failures:
                raise RuntimeError("; ".join(failures))

        self._run_rank_synchronized(f"learning gate step {step}", validate)

    def _checkpoint_dir(self, step: int) -> Path:
        """Return the deterministic DCP directory for one completed step."""
        return Path(self.args.train.checkpoint.output_dir) / f"step_{step}"

    def dispatch_save_event(self, checkpoint_dir: str) -> None:
        """Handle BaseTrainer's checkpoint event without its unused callbacks."""
        logger.info("rank=%d checkpoint event: saved %s", platform.get_rank(), checkpoint_dir)

    def dispatch_load_event(self, checkpoint_dir: str) -> None:
        """Handle BaseTrainer's resume event without its unused callbacks."""
        logger.info("rank=%d checkpoint event: loaded %s", platform.get_rank(), checkpoint_dir)

    def _write_checkpoint_config(self, checkpoint_dir: Path) -> None:
        """Persist resolved non-secret Hyper-RL configuration beside DCP state."""
        def write_config() -> None:
            """Write metadata only from rank zero."""
            if platform.get_rank() != 0:
                return
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            config_path = checkpoint_dir / "resolved_config.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    sanitize_config(self.resolved_config),
                    handle,
                    sort_keys=False,
                    allow_unicode=True,
                )
            manifest_path = checkpoint_dir / "checkpoint_complete.json"
            temporary_manifest = checkpoint_dir / f".{manifest_path.name}.{os.getpid()}.tmp"
            with temporary_manifest.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "step": self.state.global_step,
                        "world_size": platform.get_world_size(),
                    },
                    handle,
                )
            os.replace(temporary_manifest, manifest_path)

        self._run_rank_synchronized("checkpoint config write", write_config)

    def _verify_checkpoint_reload(self, checkpoint_dir: Path) -> None:
        """Immediately load the final actor DCP state on every rank."""
        def reload_checkpoint() -> None:
            """Reload one rank's model shard from the completed checkpoint."""
            if not checkpoint_dir.is_dir():
                raise RuntimeError(f"Final checkpoint directory was not created: {checkpoint_dir}")
            model_state = self.model.state_dict()
            dcp_load(model_state, checkpoint_id=str(checkpoint_dir), use_collectives=False)
            self.model.load_state_dict(model_state)

        self._run_rank_synchronized("checkpoint reload verification", reload_checkpoint)
        logger.info("rank=%d verified checkpoint reload from %s", platform.get_rank(), checkpoint_dir)

    def _validate_checkpoint_for_resume(self) -> None:
        """Reject incomplete or topology-incompatible checkpoints before loading."""
        load_path = self.checkpoint_callback.load_path
        if not load_path:
            return

        def validate_files() -> None:
            """Validate shared and rank-local completion artifacts."""
            checkpoint_dir = Path(load_path)
            manifest_path = checkpoint_dir / "checkpoint_complete.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"Checkpoint completion manifest is missing: {manifest_path}")
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            world_size = platform.get_world_size()
            if int(manifest.get("world_size", -1)) != world_size:
                raise RuntimeError(
                    "Checkpoint world size does not match the active job: "
                    f"checkpoint={manifest.get('world_size')}, active={world_size}"
                )
            rank = platform.get_rank()
            required_paths = [
                checkpoint_dir / "extra_state.json",
                checkpoint_dir / f"rng_rank{rank}.pt",
            ]
            if self.optimizer is not None:
                required_paths.append(checkpoint_dir / f"optimizer_rank{rank}.pt")
            if self.lr_scheduler is not None:
                required_paths.append(checkpoint_dir / "scheduler.pt")
            if hasattr(self.train_dataloader, "state_dict"):
                required_paths.append(checkpoint_dir / f"dataloader_rank{rank}.pt")
            missing = [str(path) for path in required_paths if not path.is_file()]
            if missing:
                raise RuntimeError(f"Checkpoint is incomplete; missing artifacts={missing}")

        self._run_rank_synchronized("checkpoint resume preflight", validate_files)

    def _ensure_checkpoint_saved(self, step: int) -> None:
        """Require every rank to finish all checkpoint artifacts for one step."""
        self._run_rank_synchronized(
            "checkpoint save",
            lambda: self.checkpoint_callback.ensure_saved(step),
        )

    def _invalidate_checkpoint_manifest(self, step: int) -> None:
        """Remove a prior completion marker before overwriting checkpoint artifacts."""
        def invalidate() -> None:
            """Invalidate an existing checkpoint only from rank zero."""
            if platform.get_rank() == 0:
                (self._checkpoint_dir(step) / "checkpoint_complete.json").unlink(missing_ok=True)

        self._run_rank_synchronized("checkpoint manifest invalidation", invalidate)

    def _maybe_record_periodic_checkpoint(self) -> None:
        """Attach resolved configuration to a checkpoint saved on this step."""
        save_steps = int(_required_mapping(
            _required_mapping(self.resolved_config, "train"),
            "checkpoint",
        ).get("save_steps", 0))
        if save_steps > 0 and self.state.global_step % save_steps == 0:
            self._write_checkpoint_config(self._checkpoint_dir(self.state.global_step))

    def _finalize_checkpoint(self) -> None:
        """Save, annotate, and optionally reload the final actor checkpoint."""
        checkpoint_config = _required_mapping(
            _required_mapping(self.resolved_config, "train"),
            "checkpoint",
        )
        if not bool(checkpoint_config.get("save_final", True)):
            return
        if self._evaluation_enabled and self._last_evaluation_step != self.state.global_step:
            validation_metrics, validation_samples = self._run_validation(
                self.state.global_step
            )
            self._tracker.log(
                validation_metrics,
                step=self.state.global_step,
                sample_tables={"validation/samples": validation_samples},
            )
        if _uses_colocated_vllm(self.resolved_config):
            if self.rollout_engine.phase == "rollout":
                self.rollout_engine.prepare_for_training()
                self._release_training_state_for_rollout()
            elif self.rollout_engine.phase != "training":
                raise RuntimeError(
                    "Final checkpoint requires colocated vLLM in training residency, "
                    f"got phase={self.rollout_engine.phase!r}"
                )
        self._invalidate_checkpoint_manifest(self.state.global_step)
        self.checkpoint_callback.save_now(self.state)
        self._ensure_checkpoint_saved(self.state.global_step)
        checkpoint_dir = self._checkpoint_dir(self.state.global_step)
        self._write_checkpoint_config(checkpoint_dir)
        if bool(checkpoint_config.get("verify_reload", False)):
            self._verify_checkpoint_reload(checkpoint_dir)

    def _cleanup_distributed(self) -> None:
        """Close tracking and destroy the initialized process group."""
        if self._tracker is not None:
            try:
                self._tracker.finish()
            except Exception as exc:  # pylint: disable=W0718
                logger.warning("Tracking cleanup failed: %s", exc)
            finally:
                self._tracker = None
        rollout_engine = getattr(self, "rollout_engine", None)
        close_rollout = getattr(rollout_engine, "close", None)
        if callable(close_rollout):
            try:
                close_rollout()
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                logger.warning("Rollout cleanup failed: %s", exc)
        if not self._runtime_started:
            return
        try:
            destroy_process_group()
        except (RuntimeError, ValueError) as exc:
            logger.warning("Distributed cleanup failed: %s", exc)
        self._runtime_started = False

    @staticmethod
    def _reshard_model(model: Optional[Any]) -> None:
        """Explicitly release every nested full FSDP parameter allocation."""
        if model is None:
            return
        module = getattr(model, "module", model)
        seen = set()
        for _, candidate in platform.get_cells_and_names(module):
            if isinstance(candidate, HSDPModule) and id(candidate) not in seen:
                candidate.reshard()
                seen.add(id(candidate))

    @staticmethod
    def _validate_optimizer_cpu_residency(optimizer: Optional[Any], role: str) -> None:
        """Fail if a colocated optimizer retains tensor state on the NPU."""
        if optimizer is None:
            return
        device_states = [
            str(tensor.device)
            for state in optimizer.state.values()
            for tensor in _iter_state_tensors(state)
            if not str(tensor.device).startswith("cpu")
        ]
        if device_states:
            raise RuntimeError(
                f"Colocated {role} optimizer state must be CPU resident, got devices={sorted(set(device_states))}"
            )

    def _release_training_state_for_rollout(self) -> None:
        """Reshard FSDP state and release allocator cache before waking vLLM."""
        if not _uses_colocated_vllm(self.resolved_config):
            return

        def release_training_state() -> None:
            """Release rank-local FSDP and optimizer residency."""
            hsdp_sync_stream()
            self._reshard_model(self.model)
            self._reshard_model(self.reference_model)
            self._reshard_model(self.critic_model)
            self._validate_optimizer_cpu_residency(self.optimizer, "actor")
            self._validate_optimizer_cpu_residency(self.critic_optimizer, "critic")
            platform.get_current_stream().synchronize()

        self._run_rank_synchronized("training-state release", release_training_state)

        def release_allocator_cache() -> None:
            """Release rank-local allocator cache after all models are sharded."""
            device_handle = platform.get_device_handle(platform.device_type())
            empty_cache = getattr(device_handle, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
            platform.get_current_stream().synchronize()

        self._run_rank_synchronized("allocator-cache release", release_allocator_cache)

    @staticmethod
    def _run_rank_synchronized(operation: str, callback: Any) -> None:
        """Run local work and make its failure visible on every training rank."""
        local_error = None
        try:
            callback()
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        world_size = platform.get_world_size()
        if world_size <= 1:
            if local_error is not None:
                raise local_error
            return
        errors: list[Optional[str]] = [None] * world_size
        platform.all_gather_object(errors, None if local_error is None else str(local_error))
        if any(error is not None for error in errors):
            raise RuntimeError(f"{operation} failed on at least one rank: {errors}")

    def train(self) -> None:
        """Run exactly ``train.max_steps`` synchronous registered-algorithm updates."""
        completed = False
        try:
            self._validate_checkpoint_for_resume()
            self.checkpoint_callback.on_train_begin(self.state)
            self._run_rank_synchronized(
                "checkpoint resume",
                self.checkpoint_callback.raise_if_load_failed,
            )
            if self.state.global_step > self.rollout_engine.policy_version:
                self.rollout_engine.prepare_for_training()
                self.rollout_engine.update_weights(
                    PolicySnapshot(
                        version=self.state.global_step,
                        model_name=self.model_registration.name,
                        payload=self.model,
                        metadata={"reason": "checkpoint_resume"},
                    )
                )
                self._release_training_state_for_rollout()
                self.rollout_engine.prepare_for_rollout()
            else:
                self._release_training_state_for_rollout()
            if hasattr(self, "sampler"):
                self.sampler.set_epoch(self.state.epoch)
            data_iterator = iter(self.train_dataloader)
            while self.state.global_step < self.state.max_steps:
                batch, data_iterator = self._next_batch(data_iterator)
                next_step = self.state.global_step + 1
                sample_index = int(batch["sample_indices"][0])
                logger.info(
                    "rank=%d step=%d sample_index=%d",
                    platform.get_rank(),
                    next_step,
                    sample_index,
                )
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                rollout = self.rollout_manager.generate(
                    prompt_records=self._build_prompt_records(
                        batch,
                        input_ids,
                        attention_mask,
                    ),
                    policy_version=self.state.global_step,
                )
                self.rollout_engine.prepare_for_training()
                if rollout.old_log_probs is None:
                    raise RuntimeError("Training rollout did not produce old_log_probs")
                experience = self.experience_builder.build(rollout)
                update = self.actor_manager.update(experience)
                critic_update = (
                    None
                    if self.critic_manager is None
                    else self.critic_manager.update(experience)
                )
                self.rollout_engine.update_weights(
                    PolicySnapshot(
                        version=next_step,
                        model_name=self.model_registration.name,
                        payload=self.model,
                        metadata={"optimizer_steps": update.optimizer_steps},
                    )
                )
                self._release_training_state_for_rollout()
                self.rollout_engine.prepare_for_rollout()
                self.state.global_step = next_step
                rollout_metrics, samples = self._rollout_statistics(rollout, batch, next_step)
                metrics = self._build_metrics(update, rollout_metrics, critic_update)
                self._enforce_learning_gate(metrics, next_step)
                checkpoint_will_save = self._checkpoint_will_save(self.state.global_step)
                validation_samples: list[dict[str, Any]] = []
                if checkpoint_will_save and self._evaluation_enabled:
                    validation_metrics, validation_samples = self._run_validation(
                        self.state.global_step
                    )
                    metrics.update(validation_metrics)
                should_log = (
                    self.state.global_step % self._log_steps == 0
                    or (checkpoint_will_save and self._evaluation_enabled)
                )
                if should_log:
                    self._tracker.log(
                        metrics,
                        step=self.state.global_step,
                        samples=samples,
                        sample_tables={"validation/samples": validation_samples},
                    )
                if checkpoint_will_save and _uses_colocated_vllm(self.resolved_config):
                    self.rollout_engine.prepare_for_training()
                    self._release_training_state_for_rollout()
                if (
                    self.checkpoint_callback.save_steps > 0
                    and self.state.global_step % self.checkpoint_callback.save_steps == 0
                ):
                    self._invalidate_checkpoint_manifest(self.state.global_step)
                self.checkpoint_callback.on_step_end(
                    self.state,
                    loss=update.total_loss,
                    grad_norm=update.gradient_norm,
                )
                if (
                    self.checkpoint_callback.save_steps > 0
                    and self.state.global_step % self.checkpoint_callback.save_steps == 0
                ):
                    self._ensure_checkpoint_saved(self.state.global_step)
                self._maybe_record_periodic_checkpoint()
                if (
                    checkpoint_will_save
                    and _uses_colocated_vllm(self.resolved_config)
                    and self.state.global_step < self.state.max_steps
                ):
                    self._release_training_state_for_rollout()
                    self.rollout_engine.prepare_for_rollout()
            self._finalize_checkpoint()
            completed = True
        finally:
            if completed:
                platform.barrier()
            self._cleanup_distributed()
