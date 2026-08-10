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
import logging
import os
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
from rl.roles.rollout.worker import RolloutManager
from rl.utils.monitoring import TrainingTracker, sanitize_config
from hyper_parallel import destroy_process_group, get_platform
from hyper_parallel.core.distributed_checkpoint import load as dcp_load
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
    positive_train_fields = ("max_steps", "micro_batch_size", "response_mini_batch_size")
    for field in positive_train_fields:
        value = int(train.get(field, 0))
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
    if mini_batch_size > num_responses:
        raise ValueError(
            "train.response_mini_batch_size cannot exceed rollout.num_return_sequences: "
            f"mini_batch={mini_batch_size}, responses={num_responses}"
        )
    if int(rollout.get("max_new_tokens", 0)) <= 0:
        raise ValueError("rollout.max_new_tokens must be positive")
    if int(data.get("max_prompt_length", 0)) <= 0:
        raise ValueError("data.max_prompt_length must be positive")
    if int(train.get("policy_update_epochs", 0)) <= 0:
        raise ValueError("train.policy_update_epochs must be positive")


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
) -> None:
    """Validate selected rollout and environment implementations."""
    engine_name = rollout.get("engine")
    if engine_name not in ROLLOUT_ENGINES.names:
        raise ValueError(
            f"Unknown rollout.engine '{engine_name}'; available={ROLLOUT_ENGINES.names}"
        )
    if engine_name == "vllm":
        raise ValueError(
            "The vLLM adapter is registered but not enabled in the synchronous trainer: "
            "versioned actor-to-vLLM weight refit is not implemented; use rollout.engine=hyper"
        )
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
        _validate_rollout_and_agentic(rollout, agentic)
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
        save_final = bool(checkpoint_config.get("save_final", True))
        configured_save_steps = int(checkpoint_config.get("save_steps", 0))
        effective_save_steps = configured_save_steps
        if save_final and effective_save_steps == 0:
            effective_save_steps = max_steps
        dp_shard = int(accelerator_config["dp_shard"])
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
            global_batch_size=dp_shard,
            micro_batch_size=1,
            seed=int(train_config.get("seed", 1234)),
            backend="torch",
            init_device=str(train_config.get("init_device", "meta")),
            comm_backend=train_config.get("comm_backend"),
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
        rank = platform.get_rank()
        local_samples = []
        for index, response in enumerate(rollout.responses[:self._log_samples]):
            local_samples.append(
                {
                    "step": step,
                    "rank": rank,
                    "prompt": batch["prompts"][0],
                    "response": response,
                    "ground_truth": batch["ground_truths"][0],
                    "extracted_answer": rollout.trajectories[index].metadata.get(
                        "extracted_answer"
                    ),
                    "reward": float(rollout.rewards[index].item()),
                }
            )
        local = {
            "reward_sum": float(rollout.rewards.sum().item()),
            "reward_count": int(rollout.rewards.numel()),
            "zero_std_groups": int(float(rollout.rewards.std(unbiased=True).item()) == 0.0),
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
            "reward/min": float(reward_sum == reward_count),
            "reward/max": float(reward_sum > 0),
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
        return metrics

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
        if platform.get_rank() == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            config_path = checkpoint_dir / "resolved_config.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    sanitize_config(self.resolved_config),
                    handle,
                    sort_keys=False,
                    allow_unicode=True,
                )
        platform.barrier()

    def _verify_checkpoint_reload(self, checkpoint_dir: Path) -> None:
        """Immediately load the final actor DCP state on every rank."""
        if not checkpoint_dir.is_dir():
            raise RuntimeError(f"Final checkpoint directory was not created: {checkpoint_dir}")
        model_state = self.model.state_dict()
        dcp_load(model_state, checkpoint_id=str(checkpoint_dir), use_collectives=False)
        self.model.load_state_dict(model_state)
        platform.barrier()
        logger.info("rank=%d verified checkpoint reload from %s", platform.get_rank(), checkpoint_dir)

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
        self.checkpoint_callback.on_train_end(self.state)
        checkpoint_dir = self._checkpoint_dir(self.state.global_step)
        self._write_checkpoint_config(checkpoint_dir)
        if bool(checkpoint_config.get("verify_reload", False)):
            self._verify_checkpoint_reload(checkpoint_dir)

    def _cleanup_distributed(self) -> None:
        """Close tracking and destroy the initialized process group."""
        if self._tracker is not None:
            self._tracker.finish()
            self._tracker = None
        if not self._runtime_started:
            return
        try:
            destroy_process_group()
        except (RuntimeError, ValueError) as exc:
            logger.warning("Distributed cleanup failed: %s", exc)
        self._runtime_started = False

    def train(self) -> None:
        """Run exactly ``train.max_steps`` synchronous registered-algorithm updates."""
        self.checkpoint_callback.on_train_begin(self.state)
        if self.state.global_step > self.rollout_engine.policy_version:
            self.rollout_engine.update_weights(
                PolicySnapshot(
                    version=self.state.global_step,
                    model_name=self.model_registration.name,
                    payload=self.model,
                    metadata={"reason": "checkpoint_resume"},
                )
            )
        if hasattr(self, "sampler"):
            self.sampler.set_epoch(self.state.epoch)
        data_iterator = iter(self.train_dataloader)
        completed = False
        try:
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
                self.state.global_step = next_step
                rollout_metrics, samples = self._rollout_statistics(rollout, batch, next_step)
                metrics = self._build_metrics(update, rollout_metrics, critic_update)
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
                self.checkpoint_callback.on_step_end(
                    self.state,
                    loss=update.total_loss,
                    grad_norm=update.gradient_norm,
                )
                self._maybe_record_periodic_checkpoint()
            self._finalize_checkpoint()
            completed = True
        finally:
            if completed:
                platform.barrier()
            self._cleanup_distributed()
