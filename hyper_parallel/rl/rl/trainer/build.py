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
"""The sole construction boundary for all RL training modes."""
import functools
import os
import json
import logging
from pathlib import Path
import random
from dataclasses import replace
from typing import Any, Mapping, Optional
import torch
import torch.distributed as dist
from rl.workers.targets import TargetPreparer
from rl.workers.engine.residency import TrainingResidency
from rl.agentic.codex import CodexRuntime
from rl.agentic.ds_harness import DeepSeekRuntime
from rl.algorithm.distillation import compose_training_algorithm
from rl.algorithm.loss import build_algorithm
from rl.distillation.config import parse_distillation_config
from rl.distillation.teacher import TeacherManager
from rl.trainer.checkpoint import CheckpointResources, RLCheckpointManager
from rl.config.schema import build_model_registration, required_mapping, router_replay_mode, uses_colocated_vllm
from rl.config.resolve import build_runtime_config, resolve_router_replay_config, resolve_unified_execution_config
from rl.config.limits import resolve_vllm_automatic_limits
from rl.config.validate import validate_config
from rl.data.experience import ExperiencePreparer
from rl.data.prompts import PromptDataset, build_train_dataloader, collate_prompt_samples
from rl.trainer.evaluation import Evaluator
from rl.reward_model import RewardModelClient, load_reward_function, scorer_fingerprint
from rl.rollout.topology import resolve_vllm_rollout_topology
from rl.workers.engine.model import load_tokenizer, build_role_model
from rl.workers.engine.optimizer import build_role_optimizer
from rl.workers.engine.offload import TrainingStateOffload
from rl.workers.actor import Actor
from rl.workers.critic import Critic
from rl.workers.adapters.router_adapter import install_router_replay
from rl.rollout.registry import build_rollout_engine
from rl.rollout.manager import CodexRolloutManager, DeepSeekRolloutManager, RolloutManager
from rl.utils.monitoring.tracker import TrainingTracker
from rl.utils.monitoring.config import sanitize_config
from rl.trainer.state import RLTrainerState
from rl.trainer.distributed import run_synchronized
from rl.trainer.lifecycle import RunLifecycle
from rl.weight_sync.publication import PolicyPublication
from rl.distillation.runtime import TeacherScorer
from rl.reward_model.runtime import RewardScorer
from rl.unified_execution.validation import capture_training_structure
from rl.unified_execution.runtimes.hyper_parallel.training import guard_training_numerics, install_training_attention
from rl.unified_execution.batch_invariant import initialize_batch_invariant, verify_batch_invariant
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo
from hyper_parallel.trainer.config import normalize_distributed_setup_overrides
from hyper_parallel.trainer.runtime.distributed import create_distributed_setup_from_config, initialize_distributed

def _configure_batch_invariant_communication(config: Mapping[str, Any]) -> None:
    """Align HCCL options before Trainer and rollout process groups initialize."""
    rollout = config.get("rollout")
    if not isinstance(rollout, Mapping) or rollout.get("engine") != "vllm":
        return
    vllm = rollout.get("vllm")
    if not isinstance(vllm, Mapping) or not bool(vllm.get("batch_invariant", False)):
        return
    # vLLM-Ascend applies these settings before its first TP group. HCCL caches
    # them at first initialization, so the Trainer must use the same values.
    os.environ["HCCL_DETERMINISTIC"] = "strict"
    os.environ["LCCL_DETERMINISTIC"] = "1"

def _resolve_eos_token_ids(model: Any, tokenizer: Any) -> tuple[int, ...]:
    """Return every model EOS ID with the tokenizer EOS as a fallback."""
    generation_config = getattr(model, "generation_config", None)
    configured_ids = getattr(generation_config, "eos_token_id", None)
    if configured_ids is None:
        candidates = []
    elif isinstance(configured_ids, (list, tuple)):
        candidates = [int(token_id) for token_id in configured_ids]
    else:
        candidates = [int(configured_ids)]
    tokenizer_eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos_token_id is not None:
        candidates.append(int(tokenizer_eos_token_id))
    normalized_ids = tuple(dict.fromkeys(candidates))
    if not normalized_ids:
        raise ValueError("Model or tokenizer must define at least one EOS token ID")
    return normalized_ids

def _validate_shared_vllm_metadata(
    metadata: list[Any], deployment: str, world_size: int, expected_devices: int,
) -> None:
    """Require all trainer ranks to agree on one local vLLM endpoint and NPU set."""
    for rank, rank_metadata in enumerate(metadata):
        rank_local_world_size, _, _, rank_devices = rank_metadata
        if int(rank_local_world_size) != world_size:
            raise ValueError(
                "The shared vLLM rollout path is single-node only: "
                f"rank={rank}, world_size={world_size}, local_world_size={rank_local_world_size}"
            )
        device_ids = [device.strip() for device in rank_devices.split(",")]
        if (
            len(device_ids) != expected_devices
            or not all(device_ids)
            or len(set(device_ids)) != expected_devices
        ):
            raise ValueError(
                f"Shared {deployment} rollout requires its full unique physical NPU set on every rank: "
                f"rank={rank}, devices={rank_devices!r}, expected={expected_devices}"
            )
    endpoints = [(host, port) for _, host, port, _ in metadata]
    if len(set(endpoints)) != 1:
        raise ValueError(
            f"Shared {deployment} rollout requires the same endpoint on every rank: endpoints={endpoints!r}"
        )
    device_mappings = [rank_devices for _, _, _, rank_devices in metadata]
    if len(set(device_mappings)) != 1:
        raise ValueError(
            f"Shared {deployment} rollout requires the same full physical NPU set on every rank: "
            f"mappings={device_mappings!r}"
        )

class _TrainerBuilder:
    """Assemble selected resources once; loops retain only the resources they use."""

    _run_rank_synchronized = staticmethod(run_synchronized)

    def __init__(self, resolved_config: Mapping[str, Any]) -> None:
        """Validate configuration and build the complete distributed runtime."""
        self.resolved_config = resolve_vllm_automatic_limits(
            resolve_router_replay_config(resolve_unified_execution_config(resolved_config))
        )
        self.distillation_config = parse_distillation_config(self.resolved_config.get('distillation'))
        self.algorithm = compose_training_algorithm(
            build_algorithm(required_mapping(self.resolved_config, 'algorithm')), self.distillation_config,
        )
        self.teacher_manager = None
        validate_config(self.resolved_config, self.algorithm)
        _configure_batch_invariant_communication(self.resolved_config)
        self.model_registration = build_model_registration(self.resolved_config)
        self.runtime_config = build_runtime_config(self.resolved_config)
        self.state = RLTrainerState(max_steps=self.runtime_config.training.train_iters)
        self._runtime_started = False
        self._rollout_closed_for_final_checkpoint = False
        self._prefetch_iterator = None
        self._prefetched_batch = None
        self._tracker: Optional[TrainingTracker] = None
        self._router_replay = None
        try:
            self._setup_runtime()
            self._validate_runtime_topology()
            self._build_runtime()
        except BaseException as error:
            lifecycle = RunLifecycle(
                _tracker=self._tracker, rollout_manager=getattr(self, "rollout_manager", None),
                rollout_engine=getattr(self, "rollout_engine", None),
                reward_model_client=getattr(self, "reward_model_client", None),
                _router_replay=self._router_replay, _runtime_started=self._runtime_started, state=self.state,
            )
            lifecycle.fail(error, "initialization")
            raise

    def _setup_runtime(self) -> None:
        """Initialize one process group, device, mesh, and seed for every role."""
        initialize_distributed(self.runtime_config.training.backend)
        self._runtime_started = True
        self.distributed_setup = create_distributed_setup_from_config(
            self.runtime_config
        )
        normalize_distributed_setup_overrides(self.distributed_setup, self.runtime_config)
        # Size-one FSDP is also needed to honor requested CPU parameter/gradient offload.
        needs_fsdp = (self.runtime_config.accelerator.tp_size > 1 or
                      bool(getattr(self.runtime_config.fsdp_config, "enable_offload", False)))
        if needs_fsdp and self.distributed_setup.strategy_config is None:
            self.distributed_setup.strategy_config = self.runtime_config.fsdp_config
        self.parallel_dims = self.distributed_setup.mesh_context
        self.mesh = self.parallel_dims.device_mesh
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device((torch.accelerator.current_accelerator() or torch.device("cpu")).type, local_rank)
        self.device_handle = torch.get_device_module()
        self.device_handle.set_device(local_rank)
        dp_mesh = self.parallel_dims.dp_cp_mesh
        dp_group = None
        if dp_mesh is not None and self.parallel_dims.dp_size > 1:
            dp_group = dp_mesh.get_group()
        self._dp_group_info = GroupInfo(
            group_name="rl_dp",
            group=dp_group,
            rank_size=self.parallel_dims.dp_size,
        )
        seed = int(self.runtime_config.training.seed)
        random.seed(seed)
        torch.manual_seed(seed)
        unified = self.resolved_config.get("rollout", {}).get("vllm", {}).get("unified_execution")
        if unified is not None and "batch_invariant" in unified:
            self.batch_invariant_state = initialize_batch_invariant(unified["batch_invariant"])

    def _validate_runtime_topology(self) -> None:
        """Validate torchrun world size against the resolved Trainer mesh."""
        world_size = dist.get_world_size()
        expected_world_size = (
            int(self.parallel_dims.dp_size)
            * int(self.parallel_dims.cp_size)
            * int(self.parallel_dims.tp_size)
            * int(self.parallel_dims.pp_size)
        )
        if world_size != expected_world_size:
            raise ValueError(
                "torchrun world size must equal the resolved Trainer mesh size: "
                f"world_size={world_size}, expected_world_size={expected_world_size}"
            )
        rollout_config = required_mapping(self.resolved_config, "rollout")
        if rollout_config.get("engine") != "vllm":
            return
        vllm_config = required_mapping(rollout_config, "vllm")
        deployment = str(vllm_config.get("deployment", "disjoint"))
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE", str(world_size))
        if deployment == "colocated":
            visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
            device_mapping = visible_devices or ",".join(
                str(device) for device in range(int(local_world_size))
            )
        else:
            device_mapping = str(vllm_config.get("visible_devices", ""))
        local_metadata = (
            local_world_size,
            str(vllm_config.get("host", "127.0.0.1")),
            int(vllm_config["port"]),
            device_mapping,
        )
        metadata: list[Any] = [None] * world_size
        dist.all_gather_object(metadata, local_metadata)
        expected_devices = (
            world_size
            if deployment == "colocated"
            else int(vllm_config["data_parallel_size"])
            * int(vllm_config["tensor_parallel_size"])
        )
        _validate_shared_vllm_metadata(metadata, deployment, world_size, expected_devices)

    def _build_runtime(self) -> None:
        """Build tokenizer, data, requirement-selected roles, optimizers, and tracking."""
        self._build_tokenizer_and_data()
        self._build_models_and_optimizers()
        offload_manager = None
        if self.resolved_config.get("train", {}).get("accelerator", {}).get("offload_during_rollout", False):
            roles = [(self.actor.actor_model, self.optimizer)]
            if self.reference_actor is not None:
                roles.append((self.reference_actor.actor_model, None))
            offload_manager = TrainingStateOffload(roles)
        self.residency = TrainingResidency(
            actor=self.actor, reference_actor=self.reference_actor, critic=self.critic,
            device=self.device, run_synchronized=self._run_rank_synchronized,
            colocated=uses_colocated_vllm(self.resolved_config),
            cpu_offload=self.resolved_config["train"]["accelerator"].get("cpu_offload", True),
            offload_manager=offload_manager,
        )
        self._build_rollout_runtime()
        self._build_reward_model_runtime()
        self.reward_scorer = None
        if self.reward_model_client is not None:
            self.reward_scorer = RewardScorer(
                reward_model_client=self.reward_model_client, reward_model_scorer=self.reward_model_scorer,
                reward_model_scorer_id=self.reward_model_scorer_id, reward_model_config=self.reward_model_config,
                parallel_dims=self.parallel_dims, rollout_engine=self.rollout_engine, residency=self.residency,
            )
            if self.evaluator is not None:
                self.evaluator.score_batch = functools.partial(self.reward_scorer.score, evaluation=True)
        if self.distillation_config.enabled:
            self.teacher_manager = TeacherManager(self.distillation_config)
            self._run_rank_synchronized(
                'teacher startup validation',
                lambda: self.teacher_manager.validate(self.model_registration.tokenizer_path),
            )
        self.teacher_scorer = (None if self.teacher_manager is None else TeacherScorer(
            teacher_manager=self.teacher_manager, parallel_dims=self.parallel_dims,
        ))
        self.experience_preparer = ExperiencePreparer(self.algorithm, self._dp_group_info)
        train = self.resolved_config["train"]
        self.targets = TargetPreparer(
            actor=self.actor, reference_actor=self.reference_actor, critic=self.critic,
            algorithm=self.algorithm, experience_preparer=self.experience_preparer,
            run_synchronized=self._run_rank_synchronized,
            balance_dp_tokens=train.get("balance_dp_tokens", False),
            old_log_prob_source=train.get("old_log_prob_source", "rollout"),
            has_teacher=self.teacher_manager is not None,
        )
        checkpoint_config = required_mapping(
            required_mapping(self.resolved_config, "train"),
            "checkpoint",
        )
        self.checkpoints = RLCheckpointManager(
            CheckpointResources(
                actor=self.actor, critic=self.critic, state=self.state,
                train_dataloader=self.train_dataloader, device=self.device,
                device_handle=self.device_handle, tokenizer=self.tokenizer,
                model_registration=self.model_registration,
                teacher_fingerprint=None if self.teacher_manager is None else self.teacher_manager.fingerprint,
            ),
            checkpoint_config,
            self.resolved_config,
            self._run_rank_synchronized,
        )
        self._build_tracker()
        self.publication = PolicyPublication(
            actor=self.actor, model_registration=self.model_registration, rollout_engine=self.rollout_engine,
            residency=self.residency, reward_model_client=self.reward_model_client,
        )
        self.lifecycle = RunLifecycle(
            _tracker=self._tracker, rollout_manager=self.rollout_manager, rollout_engine=self.rollout_engine,
            reward_model_client=self.reward_model_client, _router_replay=self._router_replay,
            _runtime_started=self._runtime_started, state=self.state,
        )

    def _build_tokenizer_and_data(self) -> None:
        """Build the shared tokenizer, train split, and optional evaluation split."""
        model_config = required_mapping(self.resolved_config, "model")
        data_config = required_mapping(self.resolved_config, "data")
        evaluation_config = required_mapping(self.resolved_config, "evaluation")
        train_config = required_mapping(self.resolved_config, "train")
        self._evaluation_enabled = bool(evaluation_config.get("enabled", True))
        self.tokenizer = load_tokenizer(model_config)
        self.train_dataset = PromptDataset.from_config(data_config, self.tokenizer)
        self.test_dataset = (PromptDataset.from_config(data_config, self.tokenizer, split="test")
                             if self._evaluation_enabled else None)
        self.collate_fn = functools.partial(
            collate_prompt_samples,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )
        self.train_dataloader, self.sampler = build_train_dataloader(
            self.train_dataset,
            self.collate_fn,
            batch_size=int(train_config.get("prompt_batch_size", 1)),
            rank=int(self.parallel_dims.dp_rank),
            world_size=int(self.parallel_dims.dp_size),
            seed=int(self.runtime_config.training.seed),
            shuffle=bool(data_config.get("shuffle", True)),
            num_workers=int(data_config.get("num_workers", 0)),
            pin_memory=bool(data_config.get("pin_memory", True)),
            prefetch_factor=(
                None
                if data_config.get("prefetch_factor") is None
                else int(data_config["prefetch_factor"])
            ),
        )

    def _build_reference_model(self, train_config: Mapping[str, Any]) -> Any:
        """Build the frozen Reference with its resolved offload and token settings."""
        if not self.algorithm.requirements.roles.reference:
            return None
        reference_runtime = self.runtime_config
        reference_setup = self.distributed_setup
        if "reference" in train_config or "reference_cpu_offload" in train_config.get("accelerator", {}):
            reference_runtime = build_runtime_config(self.resolved_config, reference=True)
            reference_setup = replace(self.distributed_setup, strategy_config=reference_runtime.fsdp_config)
            normalize_distributed_setup_overrides(reference_setup, reference_runtime)
        self._record_role_config("reference", reference_runtime, reference_setup)
        return build_role_model(
            reference_runtime,
            reference_setup,
            frozen=True,
            cpu_threads=train_config.get("model_init_cpu_threads"),
            token_log_probs_mode=train_config.get("token_log_probs_mode", "auto"),
            token_log_probs_chunk_size=train_config.get("token_log_probs_chunk_size", 512),
        )

    @staticmethod
    def _install_unified_training(model, unified) -> None:
        """Attach shared attention and numerical-policy guards to a built role."""
        if unified["attention_backend"] == "flash_attention_npu_v4":
            install_training_attention(model)
        if "batch_invariant" in unified:
            guard_training_numerics(model, unified["batch_invariant"])

    def _build_models_and_optimizers(self) -> None:
        """Build requirement-selected models and their independent optimizers."""
        train_config = required_mapping(self.resolved_config, "train")
        optimizer_config = required_mapping(train_config, "optimizer")
        actor_model = build_role_model(
            self.runtime_config,
            self.distributed_setup,
            frozen=False,
            cpu_threads=train_config.get("model_init_cpu_threads"),
            token_log_probs_mode=train_config.get("token_log_probs_mode", "auto"),
            token_log_probs_chunk_size=train_config.get("token_log_probs_chunk_size", 512),
        )
        unified = self.resolved_config.get("rollout", {}).get("vllm", {}).get("unified_execution")
        if unified is not None:
            unified["training_structure"] = capture_training_structure(actor_model, unified)
            self._install_unified_training(actor_model, unified)
        self._record_role_config("actor", self.runtime_config, self.distributed_setup)
        reference_model = self._build_reference_model(train_config)
        if reference_model is not None and unified is not None:
            self._install_unified_training(reference_model, unified)
        critic_model = None
        critic_config = train_config.get("critic", {})
        if self.algorithm.requirements.roles.critic:
            critic_runtime = build_runtime_config(self.resolved_config, critic=True)
            critic_setup = replace(self.distributed_setup, strategy_config=critic_runtime.fsdp_config)
            normalize_distributed_setup_overrides(critic_setup, critic_runtime)
            self._record_role_config("critic", critic_runtime, critic_setup)
            critic_model = build_role_model(
                critic_runtime, critic_setup, frozen=False,
                cpu_threads=train_config.get("model_init_cpu_threads"),
            )
            if unified is not None and "batch_invariant" in unified:
                guard_training_numerics(critic_model, unified["batch_invariant"])
        if unified is not None and "batch_invariant" in unified:
            self.batch_invariant_state = verify_batch_invariant(unified["batch_invariant"])
        actor_optimizer, actor_lr_scheduler = build_role_optimizer(
            self.runtime_config,
            actor_model,
        )
        critic_optimizer = None
        critic_lr_scheduler = None
        if critic_model is not None:
            critic_optimizer, critic_lr_scheduler = build_role_optimizer(critic_runtime, critic_model)
        actor_kwargs = {
            "algorithm": self.algorithm,
            "micro_batch_size": int(train_config["micro_batch_size"]),
        }
        replay_kwargs = {}
        if router_replay_mode(self.resolved_config) == "R3":
            self._router_replay = install_router_replay(actor_model)
            replay_kwargs["router_replay"] = self._router_replay
        self.actor = Actor(
            actor_model=actor_model,
            optimizer=actor_optimizer,
            lr_scheduler=actor_lr_scheduler,
            device=self.device,
            dp_group_info=self._dp_group_info,
            dp_size=self.parallel_dims.dp_size,
            response_mini_batch_size=int(train_config["response_mini_batch_size"]),
            update_epochs=int(train_config["policy_update_epochs"]),
            max_grad_norm=float(optimizer_config.get("max_grad_norm", 1.0)),
            gradient_sync_each_micro_batch=train_config.get("gradient_sync_each_micro_batch", False),
            max_tokens_per_micro_batch=train_config.get("max_tokens_per_micro_batch"),
            balance_dp_tokens=train_config.get("balance_dp_tokens", False),
            log_prob_max_tokens_per_micro_batch=train_config.get("old_log_prob_max_tokens_per_micro_batch"),
            distillation=self.distillation_config if self.distillation_config.enabled else None,
            **actor_kwargs,
            **replay_kwargs,
        )
        self.reference_actor = (
            None
            if reference_model is None
            else Actor(
                actor_model=reference_model, dp_group_info=self._dp_group_info,
                dp_size=self.parallel_dims.dp_size,
                log_prob_max_tokens_per_micro_batch=train_config.get("reference_log_prob_max_tokens_per_micro_batch"),
                **actor_kwargs,
            )
        )
        self.critic = None
        if critic_model is not None:
            self.critic = Critic(
                critic_model=critic_model,
                algorithm=self.algorithm,
                optimizer=critic_optimizer,
                lr_scheduler=critic_lr_scheduler,
                device=self.device,
                dp_group_info=self._dp_group_info,
                dp_size=self.parallel_dims.dp_size,
                micro_batch_size=int(critic_config.get("micro_batch_size", train_config["micro_batch_size"])),
                response_mini_batch_size=int(critic_config.get(
                    "response_mini_batch_size", train_config["response_mini_batch_size"])),
                update_epochs=int(critic_config.get("update_epochs", train_config.get(
                    "critic_update_epochs", train_config["policy_update_epochs"]))),
                max_grad_norm=critic_runtime.training.max_grad_norm,
            )
        self.model = self.actor.actor_model
        self.optimizer = self.actor.optimizer
        self.lr_scheduler = self.actor.lr_scheduler

    def _configure_rollout_tensor_parallel(self) -> None:
        """Give a TP rollout engine its request-ownership process group."""
        if self.parallel_dims.tp_size <= 1:
            return
        configure = getattr(self.rollout_engine, "configure_trainer_tensor_parallel", None)
        if not callable(configure):
            raise ValueError("Trainer TP rollout requires request ownership support")
        tp_mesh = self.parallel_dims.device_mesh["tp"]
        configure(
            group=tp_mesh.get_group(),
            tp_rank=int(self.parallel_dims.tp_rank),
            tp_size=int(self.parallel_dims.tp_size),
            request_rank=int(self.parallel_dims.dp_rank),
            request_size=int(self.parallel_dims.dp_size),
        )

    def _new_rollout_manager(
        self,
        generation: Mapping[str, Any],
        internal: Mapping[str, Any],
    ) -> Any:
        """Build the manager for the already selected internal or external runner."""
        common = {
            "pad_token_id": internal["pad_token_id"],
            "eos_token_id": internal["eos_token_id"],
            **generation,
        }
        if self.codex_runtime is not None:
            return CodexRolloutManager(runtime=self.codex_runtime, **common)
        if self.deepseek_runtime is not None:
            return DeepSeekRolloutManager(runtime=self.deepseek_runtime, **common)
        return RolloutManager(**generation, **internal)

    def _build_rollout_runtime(self) -> None:
        """Build the selected generation engine and train/evaluation rollout managers."""
        rollout_config = required_mapping(self.resolved_config, "rollout")
        agentic_config = required_mapping(self.resolved_config, "agentic")
        evaluation_config = required_mapping(self.resolved_config, "evaluation")
        self.rollout_engine = build_rollout_engine(
            rollout_config,
            self.model_registration,
        )
        self._configure_rollout_tensor_parallel()
        eos_token_ids = _resolve_eos_token_ids(self.model, self.tokenizer)
        runner_name = str(agentic_config.get("runner", "internal"))
        manager_kwargs = {
            "pad_token_id": int(self.tokenizer.pad_token_id),
            "eos_token_id": eos_token_ids[0],
        }
        if runner_name == "internal":
            manager_kwargs.update({
                "engine": self.rollout_engine,
                "tokenizer": self.tokenizer,
                "environment_name": str(agentic_config["environment"]),
                "max_turns": int(agentic_config["max_turns"]),
                "max_observation_tokens": int(agentic_config["max_observation_tokens"]),
                "max_episode_tokens": (
                    None
                    if agentic_config.get("max_episode_tokens") is None
                    else int(agentic_config["max_episode_tokens"])
                ),
                "environment_settings": {
                    **agentic_config,
                    **({"defer_reward_model": True} if "reward_model" in self.resolved_config else {}),
                },
                "interaction_mode": agentic_config.get("interaction_mode"),
                "eos_token_ids": eos_token_ids,
            })
        generation_kwargs = {
            "num_return_sequences": int(rollout_config["num_return_sequences"]),
            "max_new_tokens": int(rollout_config["max_new_tokens"]),
            "temperature": float(rollout_config.get("temperature", 1.0)),
            "top_p": float(rollout_config.get("top_p", 1.0)),
            "top_k": int(rollout_config.get("top_k", 0)),
            "collect_old_log_probs": (
                self.algorithm.requirements.data.rollout_log_probs
                and bool(rollout_config.get("collect_log_probs", True))
            ),
            "seed": None if rollout_config.get("seed") is None else int(rollout_config["seed"]),
            "ignore_eos": bool(rollout_config.get("ignore_eos", False)),
            "do_sample": True,
        }
        if router_replay_mode(self.resolved_config) == "R3":
            generation_kwargs["collect_router_trace"] = True
        self.codex_runtime: Optional[CodexRuntime] = None
        self.deepseek_runtime: Optional[DeepSeekRuntime] = None
        if runner_name == "codex":
            codex_config = {
                **required_mapping(agentic_config, "codex"),
                "max_turns": int(agentic_config["max_turns"]),
                "max_episode_tokens": agentic_config.get("max_episode_tokens"),
            }
            self.codex_runtime = CodexRuntime(self.rollout_engine, codex_config)
        elif runner_name == "deepseek":
            deepseek_config = {
                **required_mapping(agentic_config, "deepseek"),
                "max_turns": int(agentic_config["max_turns"]),
                "max_episode_tokens": agentic_config.get("max_episode_tokens"),
            }
            self.deepseek_runtime = DeepSeekRuntime(self.rollout_engine, deepseek_config)
        self.rollout_manager = self._new_rollout_manager(generation_kwargs, manager_kwargs)
        self.evaluator: Optional[Evaluator] = None
        if self._evaluation_enabled:
            evaluation_kwargs = {
                "num_return_sequences": 1,
                "max_new_tokens": int(evaluation_config["max_new_tokens"]),
                "temperature": float(evaluation_config.get("temperature", 1.0)),
                "top_p": float(evaluation_config.get("top_p", 1.0)),
                "top_k": int(evaluation_config.get("top_k", 0)),
                "do_sample": bool(evaluation_config.get("do_sample", False)),
                "ignore_eos": bool(evaluation_config.get("ignore_eos", False)),
                "collect_old_log_probs": False,
                "seed": generation_kwargs["seed"] if runner_name != "internal" else None,
            }
            evaluation_rollout_manager = self._new_rollout_manager(
                evaluation_kwargs, manager_kwargs
            )
            max_samples = evaluation_config.get("max_samples")
            self.evaluator = Evaluator(
                dataset=self.test_dataset,
                collate_fn=self.collate_fn,
                rollout_manager=evaluation_rollout_manager,
                device=self.device,
                batch_size=int(evaluation_config.get("batch_size", 1)),
                max_samples=None if max_samples is None else int(max_samples),
                log_samples=int(evaluation_config.get("log_samples", 0)),
                progress_steps=int(evaluation_config.get("progress_steps", 0)),
                data_parallel_rank=int(self.parallel_dims.dp_rank),
                data_parallel_size=int(self.parallel_dims.dp_size),
                is_request_owner=int(self.parallel_dims.tp_rank) == 0,
                score_batch=None,
            )

    def _build_reward_model_runtime(self) -> None:
        """Create an idle colocated RM handle only for explicit model scoring."""
        self.reward_model_client = None
        reward_config = self.resolved_config.get("reward_model")
        if reward_config is None:
            return
        self.reward_model_config = dict(reward_config)
        self.reward_model_scorer = load_reward_function(reward_config["scorer"])
        self.reward_model_scorer_id = scorer_fingerprint(reward_config)
        rollout_config = required_mapping(required_mapping(self.resolved_config, "rollout"), "vllm")
        devices = resolve_vllm_rollout_topology(rollout_config, os.environ).visible_devices
        self.reward_model_client = RewardModelClient(
            reward_config, devices, owner=dist.get_rank() == 0,
        )

    def _record_role_config(self, role: str, runtime: Any, setup: Any) -> None:
        """Retain effective HP fields and condition-filtered rules in checkpoint identity."""
        effective = self.resolved_config.setdefault("effective_training", {})
        effective[role] = {
            "source": f"train.{role}.plan_overrides" if role != "actor" and
                "plan_overrides" in self.resolved_config["train"].get(role, {}) else "train.plan_overrides",
            "runtime": runtime.to_dict(),
            "active_module_matches": [list(rule.match) for rule in (setup.module_replacements or ())],
            "active_method_matches": [rule.match for rule in (setup.method_replacements or ())],
        }

    def _build_tracker(self) -> None:
        """Initialize console/W&B tracking on global rank zero only."""
        if dist.get_rank() == 0:
            output = Path(self.resolved_config["train"]["checkpoint"]["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(sanitize_config(self.resolved_config), ensure_ascii=False, indent=2)
            (output / "effective_config.json").write_text(payload + "\n", encoding="utf-8")
            logging.getLogger(__name__).info("Effective RL configuration: %s", payload)
        logging_config = required_mapping(self.resolved_config, "logging")
        wandb_config = required_mapping(logging_config, "wandb")
        self._log_steps = int(logging_config.get("log_steps", 1))
        self._log_samples = int(logging_config.get("log_samples", 0))
        if self._log_steps <= 0:
            raise ValueError(f"logging.log_steps must be positive, got {self._log_steps}")
        if self._log_samples < 0:
            raise ValueError(f"logging.log_samples must be non-negative, got {self._log_samples}")
        self._tracker = TrainingTracker(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            backends=tuple(logging_config["backends"]),
            project_name=str(logging_config.get("project_name", "hyper-rl")),
            experiment_name=str(logging_config.get("experiment_name", "hyper_rl_run")),
            resolved_config=self.resolved_config,
            wandb_mode=str(wandb_config.get("mode", "auto")),
            wandb_entity=wandb_config.get("entity"),
            wandb_directory=str(wandb_config.get("directory", "outputs/wandb")),
        )


def build_resources(config: Mapping[str, Any]) -> _TrainerBuilder:
    """Build resources with failure cleanup; callers select explicit dependencies."""
    return _TrainerBuilder(config)


def build_trainer(config: Mapping[str, Any]) -> Any:
    """Select the concrete training loop from the declared execution mode."""
    # Concrete loops import this builder; defer their imports to break the cycle.
    if "async_training" in config and "colocate_async" in config:
        raise ValueError("Choose exactly one asynchronous execution mode")
    if "async_training" in config:
        from rl.trainer.fully_async import FullyAsyncTrainer  # pylint: disable=C0415
        return FullyAsyncTrainer(config)
    if "colocate_async" in config:
        from rl.trainer.colocate_async import ColocateAsyncTrainer  # pylint: disable=C0415
        return ColocateAsyncTrainer(config)
    from rl.trainer.sync import SyncTrainer  # pylint: disable=C0415
    return SyncTrainer(build_resources(config))
