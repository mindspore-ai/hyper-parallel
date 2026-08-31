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
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from rl.algorithm.loss import build_algorithm
from rl.config import (
    build_model_registration,
    build_runtime_config,
    required_mapping,
    resolve_vllm_automatic_limits,
    uses_colocated_vllm,
    validate_config,
)
from rl.consistency import (
    CONSISTENCY_PROFILE_OFF,
    configure_consistency_profile,
    consistency_runtime_state,
    install_trainer_consistency_profile,
    measure_post_update_old_policy_mismatch,
    validate_consistency_forward_inputs,
    validate_pre_update_consistency,
)
from rl.dataset.batch_builder import ExperiencePreparer
from rl.dataset.contracts import ExperienceBatch
from rl.dataset.data_source import (
    PromptDataset,
    build_prompt_records,
    collate_prompt_samples,
)
from rl.evaluation import Evaluator
from rl.roles.policy.actor import Actor
from rl.roles.policy.critic import Critic
from rl.roles.model import (
    build_role_model,
    build_role_optimizer,
    iter_hsdp_roots,
)
from rl.roles.rollout.registry import build_rollout_engine
from rl.roles.rollout.worker import RolloutManager
from rl.roles.weight_sync.checkpoint import RLCheckpointManager
from rl.roles.weight_sync.sync import PolicySnapshot
from rl.utils.monitoring.metrics import (
    build_training_metrics,
    enforce_learning_gate,
    summarize_rollout,
    summarize_training_diagnostics,
)
from rl.utils.monitoring.tracker import TrainingTracker

from hyper_parallel.auto_models.components.distributed.infrastructure import (
    create_distributed_setup_from_config,
    destroy_process_group,
    initialize_distributed,
)
from hyper_parallel import get_platform, hsdp_sync_stream
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo

platform = get_platform()
logger = logging.getLogger(__name__)


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


def _rollout_tensor_artifact(tensor: Any, *, dtype_name: Optional[str] = None) -> dict[str, Any]:
    """Serialize one small rollout tensor with an exact byte digest."""
    value = tensor.detach().to("cpu").contiguous()
    if dtype_name is not None:
        value = platform.tensor_type_cast(value, dtype_name)
    array = platform.tensor_to_numpy(value)
    payload = array.tobytes()
    return {
        "dtype": str(value.dtype).rsplit(".", maxsplit=1)[-1],
        "shape": [int(size) for size in value.shape],
        "sha256": sha256(payload).hexdigest(),
        "values": array.tolist(),
    }


def _write_rollout_artifact(rollout: ExperienceBatch, policy_version: int) -> None:
    """Persist and optionally compare token, mask, and raw-logprob contracts."""
    output_dir = os.environ.get("HYPER_RL_ROLLOUT_ARTIFACT_DIR")
    if not output_dir:
        return
    strategy = os.environ.get("HYPER_RL_ROLLOUT_ARTIFACT_STRATEGY")
    if not strategy:
        raise ValueError("HYPER_RL_ROLLOUT_ARTIFACT_STRATEGY must be set")
    oracle_run_id = os.environ.get("HYPER_RL_WEIGHT_ORACLE_RUN_ID")
    if not oracle_run_id:
        raise RuntimeError("HYPER_RL_WEIGHT_ORACLE_RUN_ID must identify the verification run")
    rank = int(platform.get_rank())
    artifact = {
        "strategy": strategy,
        "oracle_run_id": oracle_run_id,
        "policy_version": int(policy_version),
        "trainer_rank": rank,
        "worker_policy_version": rollout.worker_policy_version,
        "worker_policy_fingerprint": rollout.worker_policy_fingerprint,
        "sequences": _rollout_tensor_artifact(rollout.sequences),
        "attention_mask": _rollout_tensor_artifact(
            rollout.attention_mask,
            dtype_name="int32",
        ),
        "action_mask": _rollout_tensor_artifact(
            rollout.action_mask,
            dtype_name="int32",
        ),
        "raw_logprobs": _rollout_tensor_artifact(
            rollout.old_log_probs,
            dtype_name="float32",
        ),
    }
    filename = f"{strategy}-policy{int(policy_version)}-rank{rank}.json"
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    oracle_dir = os.environ.get("HYPER_RL_ROLLOUT_ARTIFACT_ORACLE_DIR")
    if oracle_dir:
        oracle_path = Path(oracle_dir) / f"full_gather-policy{int(policy_version)}-rank{rank}.json"
        if not oracle_path.is_file():
            raise RuntimeError(f"Full-gather rollout artifact is missing: {oracle_path}")
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        if oracle.get("oracle_run_id") != oracle_run_id:
            raise RuntimeError(
                "Full-gather rollout oracle run mismatch: "
                f"expected={oracle_run_id!r}, actual={oracle.get('oracle_run_id')!r}"
            )
        compared_fields = ("sequences", "attention_mask", "action_mask", "raw_logprobs")
        mismatches = [
            field
            for field in compared_fields
            if {
                key: artifact[field][key]
                for key in ("dtype", "shape", "sha256")
            }
            != {
                key: oracle.get(field, {}).get(key)
                for key in ("dtype", "shape", "sha256")
            }
        ]
        for field in ("policy_version", "worker_policy_version", "worker_policy_fingerprint"):
            if artifact[field] != oracle.get(field):
                mismatches.append(field)
        if mismatches:
            artifact["oracle_match"] = False
            artifact["oracle_mismatches"] = mismatches
            artifact["oracle_path"] = str(oracle_path)
            mismatch_target = directory / f"mismatch-{filename}"
            mismatch_temporary = (
                directory / f".{mismatch_target.name}.{os.getpid()}.tmp"
            )
            mismatch_temporary.write_text(
                json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            mismatch_temporary.replace(mismatch_target)
            raise RuntimeError(
                "Direct rollout artifact differs from full-gather oracle: "
                f"policy_version={policy_version}, rank={rank}, fields={mismatches}"
            )
        artifact["oracle_match"] = True
    target = directory / filename
    temporary = directory / f".{filename}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


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


@dataclass
class RLTrainerState:
    """Mutable state owned by the synchronous RL orchestration loop."""

    max_steps: int
    global_step: int = 0
    epoch: int = 0
    consumed_samples: int = 0
    consumed_tokens: int = 0


class _DistributedPromptSampler:
    """Deterministically shard prompt indices across data-parallel ranks."""

    def __init__(
        self,
        dataset_size: int,
        *,
        rank: int,
        world_size: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        """Store deterministic rank-local sampling parameters."""
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        """Yield this rank's deterministic, equally sized index shard."""
        indices = list(range(self.dataset_size))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        usable_size = self.dataset_size - self.dataset_size % self.world_size
        return iter(indices[:usable_size][self.rank::self.world_size])

    def __len__(self) -> int:
        """Return the number of complete samples owned by this rank."""
        return self.dataset_size // self.world_size

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle order for one data epoch."""
        self.epoch = int(epoch)


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


class SyncTrainer:
    """Synchronous RL orchestrator composed from HyperAutoModel role runtimes.

    Args:
        resolved_config: Fully merged Hyper-RL YAML configuration.
    """

    def __init__(self, resolved_config: Mapping[str, Any]) -> None:
        """Validate configuration and build the complete distributed runtime."""
        self.resolved_config = resolve_vllm_automatic_limits(resolved_config)
        self._consistency_profile = configure_consistency_profile(self.resolved_config)
        self.algorithm = build_algorithm(
            required_mapping(self.resolved_config, "algorithm")
        )
        validate_config(self.resolved_config, self.algorithm)
        _configure_batch_invariant_communication(self.resolved_config)
        install_trainer_consistency_profile(self.resolved_config)
        logger.info(
            "training-inference consistency runtime: enabled=%s recipe=%s state=%s",
            self._consistency_profile != CONSISTENCY_PROFILE_OFF,
            self._consistency_profile,
            consistency_runtime_state(),
        )
        self.model_registration = build_model_registration(self.resolved_config)
        self.runtime_config = build_runtime_config(self.resolved_config)
        self.state = RLTrainerState(max_steps=self.runtime_config.training.train_iters)
        self._runtime_started = False
        self._tracker: Optional[TrainingTracker] = None
        try:
            self._setup_runtime()
            self._validate_runtime_topology()
            self._build_runtime()
        except Exception:
            self._cleanup_distributed()
            raise

    def _setup_runtime(self) -> None:
        """Initialize one process group, device, mesh, and seed for every role."""
        initialize_distributed(self.runtime_config.training.backend)
        self._runtime_started = True
        self.distributed_setup = create_distributed_setup_from_config(
            self.runtime_config
        )
        self.parallel_dims = self.distributed_setup.mesh_context
        self.mesh = self.parallel_dims.device_mesh
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = platform.device(local_rank)
        self.device_handle = platform.get_device_handle(platform.device_type())
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
        platform.manual_seed(seed)

    def train(self) -> None:
        """Run synchronous rollout, learning, publication, and checkpointing."""
        completed = False
        try:
            self.checkpoints.validate_resume()
            self.checkpoints.begin(self.state)
            if self.state.global_step > self.rollout_engine.policy_version:
                self._release_training_state_for_rollout()
                self.rollout_engine.prepare_for_training()
                self.rollout_engine.update_weights(
                    PolicySnapshot(
                        version=self.state.global_step,
                        model_name=self.model_registration.name,
                        payload=self.actor.actor_model,
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
                self._train_step(batch)
            save_final = bool(
                required_mapping(
                    required_mapping(self.resolved_config, "train"),
                    "checkpoint",
                ).get("save_final", True)
            )
            if (
                save_final
                and self.evaluator is not None
                and self.evaluator.last_step != self.state.global_step
            ):
                validation_metrics, validation_samples = self.evaluator.run(
                    self.state.global_step
                )
                self._tracker.log(
                    validation_metrics,
                    step=self.state.global_step,
                    sample_tables={"validation/samples": validation_samples},
                )
            if save_final and uses_colocated_vllm(self.resolved_config):
                if self.rollout_engine.phase == "rollout":
                    self.rollout_engine.prepare_for_training()
                    self._release_training_state_for_rollout()
                elif self.rollout_engine.phase != "training":
                    raise RuntimeError(
                        "Final checkpoint requires colocated vLLM in training residency, "
                        f"got phase={self.rollout_engine.phase!r}"
                    )
            self.checkpoints.finalize(self.state)
            completed = True
        finally:
            if completed:
                platform.barrier()
            self._cleanup_distributed()

    def _prepare_experience(
        self,
        rollout: ExperienceBatch,
        collect_diagnostics: bool,
        timings: dict[str, float],
    ) -> tuple[ExperienceBatch, dict[str, float]]:
        """Run required role inference and build immutable training targets."""
        requirements = self.algorithm.requirements.data
        actor_log_probs = None
        consistency_gate_enabled = self._consistency_profile != CONSISTENCY_PROFILE_OFF
        if consistency_gate_enabled:
            validate_consistency_forward_inputs(
                rollout,
                group=self._dp_group_info.group,
                group_size=self.parallel_dims.dp_size,
                operation="pre-update",
            )
        if collect_diagnostics or consistency_gate_enabled:
            stage_started = time.perf_counter()
            actor_log_probs = self._run_rank_synchronized(
                "pre-update Actor log-probabilities",
                lambda: self.actor.compute_log_probs(rollout),
            )
            if actor_log_probs is None:
                raise RuntimeError(
                    "Pre-update Actor log-probability computation failed without a synchronized error"
                )
            timings["old_log_prob"] = time.perf_counter() - stage_started
        consistency_metrics = {}
        if consistency_gate_enabled:
            consistency_metrics = validate_pre_update_consistency(
                rollout,
                actor_log_probs,
                expected_policy_version=self.state.global_step,
                expected_policy_fingerprint=self.rollout_engine.policy_fingerprint,
                group=self._dp_group_info.group,
                group_size=self.parallel_dims.dp_size,
            )
        reference_log_probs = None
        if requirements.reference_log_probs:
            if self.reference_actor is None:
                raise RuntimeError(
                    f"Algorithm '{self.algorithm.name}' requires a reference model"
                )
            stage_started = time.perf_counter()
            reference_log_probs = self.reference_actor.compute_log_probs(rollout)
            timings["ref"] = time.perf_counter() - stage_started
        values = None
        if requirements.values:
            if self.critic is None:
                raise RuntimeError(f"Algorithm '{self.algorithm.name}' requires a Critic")
            stage_started = time.perf_counter()
            values = self.critic.compute_values(rollout)
            timings["values"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        experience = self.experience_preparer.prepare(
            rollout,
            reference_log_probs=reference_log_probs,
            values=values,
        )
        timings["adv"] = time.perf_counter() - stage_started
        diagnostic_metrics = (
            {}
            if actor_log_probs is None
            else summarize_training_diagnostics(experience, actor_log_probs)
        )
        diagnostic_metrics.update(consistency_metrics)
        return experience, diagnostic_metrics

    def _publish_policy(self, next_step: int, actor_update: Any) -> None:
        """Transfer the updated Actor and restore rollout residency."""
        hsdp_sync_stream()
        self._reshard_model(self.actor.actor_model)
        # Refit wakes the colocated vLLM weights before copying the updated
        # parameters.  Clear rank-local FSDP and allocator residency first so
        # vLLM does not have to restore its weights on top of cached training
        # allocations left by the preceding optimizer step.
        self._release_training_state_for_rollout()
        self.rollout_engine.update_weights(
            PolicySnapshot(
                version=next_step,
                model_name=self.model_registration.name,
                payload=self.actor.actor_model,
                metadata={"optimizer_steps": actor_update.optimizer_steps},
            )
        )
        self.rollout_engine.prepare_for_rollout()

    def _train_step(self, batch: Mapping[str, Any]) -> None:
        """Run one explicit RL2-style role pipeline."""
        step_started = time.perf_counter()
        timings: dict[str, float] = {}
        next_step = self.state.global_step + 1
        logger.info(
            "rank=%d step=%d sample_index=%d",
            platform.get_rank(),
            next_step,
            int(batch["sample_indices"][0]),
        )
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        stage_started = time.perf_counter()
        prompt_records = self._run_rank_synchronized(
            "prompt record construction",
            lambda: build_prompt_records(batch, input_ids, attention_mask),
        )
        if prompt_records is None:
            raise RuntimeError("Prompt record construction failed without a synchronized error")
        #start vllm serve load model weight to NPU
        rollout = self.rollout_manager.generate(
            prompt_records=prompt_records,
            policy_version=self.state.global_step,
        )
        self._run_rank_synchronized(
            "rollout artifact publication",
            lambda: _write_rollout_artifact(rollout, self.state.global_step),
        )
        timings["gen"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        self.rollout_engine.prepare_for_training()
        timings["prepare_training"] = time.perf_counter() - stage_started
        if rollout.old_log_probs is None:
            raise RuntimeError("Training rollout did not produce old_log_probs")
        collect_diagnostics = next_step % self._log_steps == 0 or (
            self.evaluator is not None and self.checkpoints.will_save(next_step)
        )
        experience, diagnostic_metrics = self._prepare_experience(
            rollout,
            collect_diagnostics,
            timings,
        )
        stage_started = time.perf_counter()
        actor_update = self.actor.update(experience)
        timings["update_actor"] = time.perf_counter() - stage_started
        if self._consistency_profile != CONSISTENCY_PROFILE_OFF:
            stage_started = time.perf_counter()
            post_update_log_probs = self._run_rank_synchronized(
                "post-update Actor log-probabilities",
                lambda: self.actor.compute_log_probs(experience),
            )
            if post_update_log_probs is None:
                raise RuntimeError(
                    "Post-update Actor log-probability computation failed without a synchronized error"
                )
            diagnostic_metrics.update(
                measure_post_update_old_policy_mismatch(
                    experience,
                    post_update_log_probs,
                    group=self._dp_group_info.group,
                    group_size=self.parallel_dims.dp_size,
                )
            )
            timings["post_update_old_log_prob"] = time.perf_counter() - stage_started
        critic_update = None
        if self.critic is not None:
            stage_started = time.perf_counter()
            critic_update = self.critic.update(experience)
            timings["update_critic"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        #synic weight
        self._publish_policy(next_step, actor_update)
        timings["weight_sync"] = time.perf_counter() - stage_started
        timings["step"] = time.perf_counter() - step_started
        if collect_diagnostics:
            diagnostic_metrics.update(
                {f"timing_s/{name}": value for name, value in timings.items()}
            )
            diagnostic_metrics["perf/time_per_step"] = timings["step"]
            total_tokens = diagnostic_metrics.get("training/total_tokens", 0.0)
            diagnostic_metrics["perf/total_num_tokens"] = total_tokens
            diagnostic_metrics["perf/tokens_per_second_per_device"] = total_tokens / max(
                timings["step"] * platform.get_world_size(),
                1.0e-9,
            )
        self._complete_step(
            step=next_step,
            batch=batch,
            rollout=rollout,
            actor_update=actor_update,
            critic_update=critic_update,
            diagnostic_metrics=diagnostic_metrics,
        )

    def _complete_step(
        self,
        *,
        step: int,
        batch: Mapping[str, Any],
        rollout: ExperienceBatch,
        actor_update: Any,
        critic_update: Optional[Any],
        diagnostic_metrics: Mapping[str, float],
    ) -> None:
        """Record, evaluate, and checkpoint one successfully published policy."""
        self.state.global_step = step
        rollout_metrics, samples = summarize_rollout(
            rollout,
            batch,
            step=step,
            sample_limit=self._log_samples,
        )
        metrics = build_training_metrics(
            step=step,
            actor_update=actor_update,
            rollout_metrics=rollout_metrics,
            critic_update=critic_update,
            policy=self.rollout_engine,
            diagnostic_metrics=diagnostic_metrics,
        )
        train_config = required_mapping(self.resolved_config, "train")
        enforce_learning_gate(
            metrics,
            step=step,
            config=train_config.get("learning_gate", {}),
            run_synchronized=self._run_rank_synchronized,
        )
        checkpoint_will_save = self.checkpoints.will_save(step)
        validation_samples: list[dict[str, Any]] = []
        if checkpoint_will_save and self.evaluator is not None:
            validation_metrics, validation_samples = self.evaluator.run(step)
            metrics.update(validation_metrics)
        if step % self._log_steps == 0 or (
            checkpoint_will_save and self.evaluator is not None
        ):
            self._tracker.log(
                metrics,
                step=step,
                samples=samples,
                sample_tables={"validation/samples": validation_samples},
            )
        if checkpoint_will_save and uses_colocated_vllm(self.resolved_config):
            self.rollout_engine.prepare_for_training()
            self._release_training_state_for_rollout()
        self.checkpoints.complete_step(
            self.state,
            loss=actor_update.total_loss,
            grad_norm=actor_update.gradient_norm,
        )
        if (
            checkpoint_will_save
            and uses_colocated_vllm(self.resolved_config)
            and step < self.state.max_steps
        ):
            self._release_training_state_for_rollout()
            self.rollout_engine.prepare_for_rollout()

    def _validate_runtime_topology(self) -> None:
        """Validate torchrun world size against the resolved Trainer mesh."""
        world_size = platform.get_world_size()
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
        platform.all_gather_object(metadata, local_metadata)
        for rank, rank_metadata in enumerate(metadata):
            rank_local_world_size = rank_metadata[0]
            rank_devices = rank_metadata[3]
            if int(rank_local_world_size) != world_size:
                raise ValueError(
                    "The shared vLLM rollout path is single-node only: "
                    f"rank={rank}, world_size={world_size}, local_world_size={rank_local_world_size}"
                )
            device_ids = [device.strip() for device in rank_devices.split(",")]
            expected_devices = (
                world_size
                if deployment == "colocated"
                else int(vllm_config["data_parallel_size"])
                * int(vllm_config["tensor_parallel_size"])
            )
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
        if any(endpoint != endpoints[0] for endpoint in endpoints):
            raise ValueError(
                f"Shared {deployment} rollout requires the same endpoint on every rank: endpoints={endpoints!r}"
            )
        device_mappings = [rank_devices for _, _, _, rank_devices in metadata]
        if any(mapping != device_mappings[0] for mapping in device_mappings):
            raise ValueError(
                f"Shared {deployment} rollout requires the same full physical NPU set on every rank: "
                f"mappings={device_mappings!r}"
            )

    def _build_runtime(self) -> None:
        """Build tokenizer, data, requirement-selected roles, optimizers, and tracking."""
        self._build_tokenizer_and_data()
        self._build_models_and_optimizers()
        self._build_rollout_runtime()
        self.experience_preparer = ExperiencePreparer(self.algorithm)
        checkpoint_config = required_mapping(
            required_mapping(self.resolved_config, "train"),
            "checkpoint",
        )
        self.checkpoints = RLCheckpointManager(
            self,
            checkpoint_config,
            self.resolved_config,
            self._run_rank_synchronized,
        )
        self._build_tracker()

    def _build_tokenizer_and_data(self) -> None:
        """Build the shared tokenizer, train split, and optional evaluation split."""
        model_config = required_mapping(self.resolved_config, "model")
        data_config = required_mapping(self.resolved_config, "data")
        evaluation_config = required_mapping(self.resolved_config, "evaluation")
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
            "prompt_column": (
                None
                if data_config.get("prompt_column") is None
                else str(data_config["prompt_column"])
            ),
            "answer_column": (
                None
                if data_config.get("answer_column") is None
                else str(data_config["answer_column"])
            ),
            "prompt_instruction": (
                None
                if data_config.get("prompt_instruction") is None
                else str(data_config["prompt_instruction"])
            ),
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

    def _build_dataloader(self) -> None:
        """Build the stateful prompt loader over the RL data-parallel domain."""
        data_config = required_mapping(self.resolved_config, "data")
        train_config = required_mapping(self.resolved_config, "train")
        dp_size = int(self.parallel_dims.dp_size)
        dp_rank = int(self.parallel_dims.dp_rank)
        self.sampler = _DistributedPromptSampler(
            len(self.train_dataset),
            rank=dp_rank,
            world_size=dp_size,
            seed=int(self.runtime_config.training.seed),
            shuffle=bool(data_config.get("shuffle", True)),
        )
        num_workers = int(data_config.get("num_workers", 0))
        loader_kwargs = {
            "batch_size": int(train_config.get("prompt_batch_size", 1)),
            "sampler": self.sampler,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
            "pin_memory": bool(data_config.get("pin_memory", True)),
            "drop_last": True,
        }
        prefetch_factor = data_config.get("prefetch_factor")
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        self.train_dataloader = StatefulDataLoader(
            self.train_dataset,
            **loader_kwargs,
        )

    def _build_models_and_optimizers(self) -> None:
        """Build requirement-selected models and their independent optimizers."""
        train_config = required_mapping(self.resolved_config, "train")
        optimizer_config = required_mapping(train_config, "optimizer")
        actor_model = self._build_one_parallel_model(frozen=False)
        reference_model = None
        if self.algorithm.requirements.roles.reference:
            reference_model = self._build_one_parallel_model(frozen=True)
        critic_model = None
        if self.algorithm.requirements.roles.critic:
            raise NotImplementedError(
                "Critic construction is not available in the initial HyperAutoModel "
                "migration; use a critic-free algorithm such as GRPO"
            )
        actor_optimizer, actor_lr_scheduler = self._build_optimizer_for(actor_model)
        critic_optimizer = None
        critic_lr_scheduler = None
        if critic_model is not None:
            critic_optimizer, critic_lr_scheduler = self._build_optimizer_for(critic_model)
        actor_kwargs = {
            "algorithm": self.algorithm,
            "micro_batch_size": int(train_config["micro_batch_size"]),
        }
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
            **actor_kwargs,
        )
        self.reference_actor = (
            None
            if reference_model is None
            else Actor(actor_model=reference_model, **actor_kwargs)
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
                micro_batch_size=int(train_config["micro_batch_size"]),
                response_mini_batch_size=int(train_config["response_mini_batch_size"]),
                update_epochs=int(
                    train_config.get(
                        "critic_update_epochs",
                        train_config["policy_update_epochs"],
                    )
                ),
                max_grad_norm=float(optimizer_config.get("max_grad_norm", 1.0)),
            )
        self.model = self.actor.actor_model
        self.optimizer = self.actor.optimizer
        self.lr_scheduler = self.actor.lr_scheduler

    def _build_optimizer_for(self, model: Any) -> tuple[Any, Any]:
        """Build independent HyperAutoModel optimizer state for one role model."""
        return build_role_optimizer(self.runtime_config, model)

    def _build_rollout_runtime(self) -> None:
        """Build the selected generation engine and train/evaluation rollout managers."""
        rollout_config = required_mapping(self.resolved_config, "rollout")
        agentic_config = required_mapping(self.resolved_config, "agentic")
        evaluation_config = required_mapping(self.resolved_config, "evaluation")
        self.rollout_engine = build_rollout_engine(
            rollout_config,
            self.model_registration,
        )
        if self.parallel_dims.tp_size > 1:
            configure_tp = getattr(
                self.rollout_engine,
                "configure_trainer_tensor_parallel",
                None,
            )
            if not callable(configure_tp):
                raise ValueError(
                    "Trainer TP rollout requires request ownership support"
                )
            tp_mesh = self.parallel_dims.device_mesh["tp"]
            configure_tp(
                group=tp_mesh.get_group(),
                tp_rank=int(self.parallel_dims.tp_rank),
                tp_size=int(self.parallel_dims.tp_size),
                request_rank=int(self.parallel_dims.dp_rank),
                request_size=int(self.parallel_dims.dp_size),
            )
        eos_token_ids = _resolve_eos_token_ids(self.model, self.tokenizer)
        manager_kwargs = {
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
            "environment_settings": dict(agentic_config),
            "interaction_mode": agentic_config.get("interaction_mode"),
            "pad_token_id": int(self.tokenizer.pad_token_id),
            "eos_token_id": eos_token_ids[0],
            "eos_token_ids": eos_token_ids,
        }
        self.rollout_manager = RolloutManager(
            num_return_sequences=int(rollout_config["num_return_sequences"]),
            max_new_tokens=int(rollout_config["max_new_tokens"]),
            temperature=float(rollout_config.get("temperature", 1.0)),
            top_p=float(rollout_config.get("top_p", 1.0)),
            top_k=int(rollout_config.get("top_k", 0)),
            do_sample=True,
            collect_old_log_probs=self.algorithm.requirements.data.rollout_log_probs,
            seed=(None if rollout_config.get("seed") is None else int(rollout_config["seed"])),
            ignore_eos=bool(rollout_config.get("ignore_eos", False)),
            **manager_kwargs,
        )
        self.evaluator: Optional[Evaluator] = None
        if self._evaluation_enabled:
            evaluation_rollout_manager = RolloutManager(
                num_return_sequences=1,
                max_new_tokens=int(evaluation_config["max_new_tokens"]),
                temperature=float(evaluation_config.get("temperature", 1.0)),
                top_p=float(evaluation_config.get("top_p", 1.0)),
                top_k=int(evaluation_config.get("top_k", 0)),
                do_sample=bool(evaluation_config.get("do_sample", False)),
                ignore_eos=bool(evaluation_config.get("ignore_eos", False)),
                **manager_kwargs,
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
            )

    def _build_one_parallel_model(self, frozen: bool) -> platform.Module:
        """Build, freeze when requested, parallelize, materialize, and load one model."""
        return build_role_model(
            self.runtime_config,
            self.distributed_setup,
            frozen=frozen,
        )

    def _build_tracker(self) -> None:
        """Initialize console/W&B tracking on global rank zero only."""
        logging_config = required_mapping(self.resolved_config, "logging")
        wandb_config = required_mapping(logging_config, "wandb")
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
        for hsdp_root in iter_hsdp_roots(model):
            hsdp_root.reshard()

    @staticmethod
    def _validate_optimizer_cpu_residency(optimizer: Optional[Any], role: str) -> None:
        """Fail if a colocated optimizer retains tensor state on the NPU."""
        if optimizer is None:
            return
        optimizers = getattr(optimizer, "chained_optimizers", (optimizer,))
        device_states = [
            str(tensor.device)
            for component in optimizers
            for state in component.state.values()
            for tensor in _iter_state_tensors(state)
            if not str(tensor.device).startswith("cpu")
        ]
        if device_states:
            raise RuntimeError(
                f"Colocated {role} optimizer state must be CPU resident, got devices={sorted(set(device_states))}"
            )

    def _release_training_state_for_rollout(self) -> None:
        """Reshard FSDP state and release allocator cache before waking vLLM."""
        if not uses_colocated_vllm(self.resolved_config):
            return
        def release_training_state() -> None:
            """Release rank-local FSDP and optimizer residency."""
            hsdp_sync_stream()
            self._reshard_model(self.actor.actor_model)
            self._reshard_model(
                None
                if self.reference_actor is None
                else self.reference_actor.actor_model
            )
            self._reshard_model(
                None if self.critic is None else self.critic.critic_model
            )
            self._validate_optimizer_cpu_residency(self.optimizer, "actor")
            self._validate_optimizer_cpu_residency(
                None if self.critic is None else self.critic.optimizer,
                "critic",
            )
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
    def _run_rank_synchronized(operation: str, callback: Any) -> Any:
        """Run local work and make its failure visible on every training rank."""
        result = None
        local_error = None
        try:
            result = callback()
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        world_size = platform.get_world_size()
        if world_size <= 1:
            if local_error is not None:
                raise local_error
            return result
        errors: list[Optional[str]] = [None] * world_size
        local_traceback = (
            None
            if local_error is None
            else "".join(
                traceback.format_exception(
                    type(local_error), local_error, local_error.__traceback__
                )
            )
        )
        platform.all_gather_object(errors, local_traceback)
        if any(error is not None for error in errors):
            raise RuntimeError(f"{operation} failed on at least one rank: {errors}")
        return result
