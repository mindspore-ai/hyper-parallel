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
"""CPU unit tests for the main synchronous Trainer orchestration contracts."""
from __future__ import annotations

# Tests intentionally exercise stable internal stage boundaries.
# Local test doubles are not public APIs; the suite intentionally uses Torch CPU tensors.
# pylint: disable=forbidden-backend-import,missing-public-docstring,protected-access,unnecessary-lambda

import gc
import unittest
import weakref
from dataclasses import replace
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

import rl.utils.process_cleanup as cleanup_module
import rl.trainer as trainer_module
import rl.roles.model_setup as model_setup_module
import rl.utils.monitoring.metrics as metrics_module
from rl.algorithm import build_algorithm
from rl.dataset.contracts import ExperienceBatch
from rl.dataset.data_source import PromptDataset
from rl.distillation.config import DistillationConfig
from rl.roles import Actor
from rl.trainer import SyncTrainer
from rl.utils.monitoring.metrics import ActorUpdateMetrics

from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_state import HSDPStateV2
from hyper_parallel.core.fully_shard.hsdp_utils import ParamModuleInfo
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, FSDPMeshInfo, MixedPrecisionPolicy


def _rollout() -> ExperienceBatch:
    """Return one valid rollout row for orchestration tests."""
    return ExperienceBatch(
        trajectories=(),
        sequences=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.bool),
        action_mask=torch.tensor([[False, True]]),
        rewards=torch.ones(1),
        old_log_probs=torch.zeros((1, 1)),
        responses=("answer",),
        generation_seconds=0.1,
    )


@pytest.mark.parametrize("failed_stage", [None, "_setup_runtime", "_build_runtime"])
def test_trainer_initializes_batch_invariant_settings_before_distributed_runtime(
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str | None,
) -> None:
    """Configuration and deterministic communication precede runtime setup."""
    calls: list[str] = []
    resolved = {"algorithm": {"name": "grpo", "loss_aggregation": "token-mean"}}
    monkeypatch.setattr(trainer_module, "resolve_vllm_automatic_limits", lambda config: config)
    monkeypatch.setattr(trainer_module, "resolve_router_replay_config", lambda config: config)
    monkeypatch.setattr(
        trainer_module,
        "configure_consistency_profile",
        lambda _config: calls.append("consistency-config") or "off",
    )
    monkeypatch.setattr(
        trainer_module,
        "build_algorithm",
        lambda config: calls.append("algorithm") or build_algorithm(config),
    )
    monkeypatch.setattr(trainer_module, "validate_config", lambda *_args: calls.append("validate"))
    monkeypatch.setattr(
        trainer_module,
        "_configure_batch_invariant_communication",
        lambda _config: calls.append("batch-invariant"),
    )
    monkeypatch.setattr(
        trainer_module,
        "install_trainer_consistency_profile",
        lambda _config: calls.append("consistency-install"),
    )
    monkeypatch.setattr(
        trainer_module,
        "build_model_registration",
        lambda _config: calls.append("model-registration") or SimpleNamespace(name="qwen"),
    )
    monkeypatch.setattr(
        trainer_module,
        "build_runtime_config",
        lambda _config: calls.append("runtime-config")
        or SimpleNamespace(training=SimpleNamespace(train_iters=1)),
    )
    monkeypatch.setattr(SyncTrainer, "_setup_runtime", lambda self: calls.append("distributed"))
    monkeypatch.setattr(SyncTrainer, "_validate_runtime_topology", lambda self: calls.append("topology"))
    monkeypatch.setattr(SyncTrainer, "_build_runtime", lambda self: calls.append("roles"))

    if failed_stage is not None:
        failure = RuntimeError("runtime initialization failed")
        monkeypatch.setattr(SyncTrainer, failed_stage, Mock(side_effect=failure))
        cleanup = Mock(wraps=SyncTrainer._cleanup)
        monkeypatch.setattr(SyncTrainer, "_cleanup", lambda trainer: cleanup(trainer))

        with pytest.raises(RuntimeError, match="runtime initialization failed") as error:
            SyncTrainer(resolved)

        assert error.value is failure
        cleanup.assert_called_once()
        failed_trainer = cleanup.call_args.args[0]
        assert failed_trainer._tracker is None
        assert not failed_trainer._runtime_started
        return

    trainer = SyncTrainer(resolved)

    assert trainer.state.max_steps == 1
    assert calls.index("batch-invariant") < calls.index("distributed")
    assert calls == [
        "consistency-config",
        "algorithm",
        "validate",
        "batch-invariant",
        "consistency-install",
        "model-registration",
        "runtime-config",
        "distributed",
        "topology",
        "roles",
    ]


def test_trainer_builds_independent_trainable_and_reference_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GRPO construction exposes one trainable Actor and one frozen Reference."""
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    trainer.resolved_config = {
        "train": {
            "micro_batch_size": 1,
            "response_mini_batch_size": 2,
            "policy_update_epochs": 1,
            "optimizer": {"max_grad_norm": 1.0},
        }
    }
    trainer.device = torch.device("cpu")
    trainer._dp_group_info = None
    trainer.parallel_dims = SimpleNamespace(dp_size=1)
    trainer.runtime_config = object()
    trainer.distributed_setup = object()
    actor_model = torch.nn.Linear(2, 2)
    reference_model = torch.nn.Linear(2, 2)
    scheduler = SimpleNamespace(step=lambda: None)
    monkeypatch.setattr(
        trainer_module,
        "build_role_model",
        lambda unused_runtime, unused_setup, *, frozen, cpu_threads=None, **_options:
        reference_model if frozen else actor_model,
    )
    monkeypatch.setattr(
        trainer_module,
        "build_role_optimizer",
        lambda unused_runtime, model: (torch.optim.SGD(model.parameters(), lr=0.1), scheduler),
    )

    trainer._build_models_and_optimizers()

    assert isinstance(trainer.actor, Actor)
    assert isinstance(trainer.reference_actor, Actor)
    assert trainer.actor.actor_model is actor_model
    assert trainer.reference_actor.actor_model is reference_model
    assert trainer.actor.actor_model is not trainer.reference_actor.actor_model
    assert trainer.actor.optimizer is not None
    assert trainer.reference_actor.optimizer is None
    assert all(not parameter.requires_grad for parameter in trainer.reference_actor.parameters())


def test_trainer_step_orchestrates_required_role_outputs_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One step orders consistency, required roles, updates, publication, and completion."""
    calls: list[str] = []
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer._consistency_profile = "qwen3"
    trainer.algorithm = SimpleNamespace(
        name="acceptance",
        requirements=SimpleNamespace(
            data=SimpleNamespace(reference_log_probs=True, values=True)
        ),
    )
    trainer.state = SimpleNamespace(global_step=0, max_steps=1)
    trainer.device = torch.device("cpu")
    trainer._log_steps = 2
    trainer.evaluator = None
    trainer.checkpoints = SimpleNamespace(will_save=lambda unused_step: False)
    trainer._run_rank_synchronized = lambda _name, callback: callback()
    trainer.parallel_dims = SimpleNamespace(dp_size=1)
    trainer._dp_group_info = SimpleNamespace(group=None, rank_size=1)
    trainer._release_training_state_for_rollout = lambda: calls.append("release")
    trainer._reshard_model = lambda _model: calls.append("reshard")
    trainer._move_training_state = lambda *, offload: calls.append("offload" if offload else "restore")
    trainer.model_registration = SimpleNamespace(name="qwen")
    rollout = _rollout()
    actor_log_probs = rollout.old_log_probs.clone()
    reference_log_probs = torch.full_like(actor_log_probs, -0.5)
    values = torch.full_like(actor_log_probs, 0.25)
    trainer.rollout_manager = SimpleNamespace(
        generate=lambda **_kwargs: calls.append("rollout") or rollout
    )
    trainer.rollout_engine = SimpleNamespace(
        prepare_for_training=lambda: calls.append("prepare-training"),
        update_weights=lambda _snapshot: calls.append("update-weights"),
        prepare_for_rollout=lambda: calls.append("prepare-rollout"),
    )

    class ActorRole:
        """Record actor inference and update calls for orchestration assertions."""
        actor_model = object()

        @staticmethod
        def compute_log_probs(unused_experience: ExperienceBatch) -> torch.Tensor:
            """Return scripted policy probabilities."""
            del unused_experience
            calls.append("actor-logprobs")
            return actor_log_probs

        @staticmethod
        def update(experience: ExperienceBatch) -> Any:
            """Record one policy or value update."""
            assert experience is not None
            calls.append("actor-update")
            return SimpleNamespace(optimizer_steps=1)

    class ReferenceRole:
        @staticmethod
        def compute_log_probs(unused_experience: ExperienceBatch) -> torch.Tensor:
            """Return scripted policy probabilities."""
            del unused_experience
            calls.append("reference")
            return reference_log_probs

    class CriticRole:
        """Record bootstrapped value inference and the separate Critic update."""
        @staticmethod
        def compute_values(unused_experience: ExperienceBatch, *, include_last: bool = False) -> torch.Tensor:
            """Return scripted value estimates."""
            del unused_experience
            calls.append("critic-values")
            return torch.cat((values, values[:, -1:]), dim=1) if include_last else values

        @staticmethod
        def update(unused_experience: ExperienceBatch) -> Any:
            """Record one policy or value update."""
            del unused_experience
            calls.append("critic-update")
            return SimpleNamespace()

    class Preparer:
        """Validate role outputs before producing the experience used for training."""
        @staticmethod
        def prepare(
            experience: ExperienceBatch,
            *,
            reference_log_probs: torch.Tensor,
            values: Any,
            bootstrap_values: Any,
        ) -> ExperienceBatch:
            """Check target inputs and return the prepared training experience."""
            assert reference_log_probs is not None
            assert values is not None
            assert bootstrap_values.shape == values.shape[:1]
            calls.append("targets")
            return experience

    def complete_step(self: SyncTrainer, **kwargs: Any) -> None:
        """Record checkpoint completion in the fixture."""
        assert kwargs["step"] == 1
        assert kwargs["critic_update"] is not None
        assert kwargs["diagnostic_metrics"] == {
            "training/tokens": 1.0,
            "training/exact": 1.0,
            "training/post_update": 1.0,
        }
        calls.append("complete")
        self.state.global_step = 1

    trainer.actor = ActorRole()
    trainer.reference_actor = ReferenceRole()
    trainer.critic = CriticRole()
    trainer.experience_preparer = Preparer()
    trainer._complete_step = MethodType(complete_step, trainer)
    _install_step_diagnostics(monkeypatch, calls)
    batch = {
        "sample_indices": [0],
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.ones((1, 1), dtype=torch.bool),
        "prompts": ["prompt"],
        "ground_truths": ["answer"],
    }

    trainer._train_step(batch)

    assert calls == [
        "rollout",
        "prepare-training",
        "restore",
        "forward-contract",
        "actor-logprobs",
        "exact-gate",
        "reference",
        "critic-values",
        "targets",
        "diagnostics",
        "release",
        "actor-update",
        "actor-logprobs",
        "post-update-check",
        "critic-update",
        "reshard",
        "release",
        "update-weights",
        "release",
        "offload",
        "prepare-rollout",
        "complete",
    ]
    assert trainer.state.global_step == 1


def test_trainer_publication_releases_training_state_before_rollout_wake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colocated publication releases training allocations before waking rollout."""
    calls: list[str] = []
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.actor = SimpleNamespace(actor_model=object())
    trainer.model_registration = SimpleNamespace(name="qwen")
    trainer._reshard_model = lambda _model: calls.append("reshard")
    trainer._release_training_state_for_rollout = lambda: calls.append("release")
    trainer.rollout_engine = SimpleNamespace(
        update_weights=lambda _snapshot: calls.append("update"),
        prepare_for_rollout=lambda: calls.append("wake"),
    )
    monkeypatch.setattr(trainer_module, "hsdp_sync_stream", lambda: calls.append("stream"))

    trainer._move_training_state = lambda *, offload: calls.append("offload" if offload else "restore")
    trainer._publish_policy(2)

    assert calls == ["stream", "reshard", "release", "update", "release", "offload", "wake"]


def test_post_publication_release_failure_does_not_wake_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed cache release must not expose the pending policy or allocate rollout KV memory."""
    trainer = object.__new__(SyncTrainer)
    trainer.actor = SimpleNamespace(actor_model=object())
    trainer.model_registration = SimpleNamespace(name="qwen")
    trainer._reshard_model = Mock()
    trainer._release_training_state_for_rollout = Mock(side_effect=[None, RuntimeError("cache release failed")])
    trainer.rollout_engine = SimpleNamespace(update_weights=Mock(), prepare_for_rollout=Mock())
    monkeypatch.setattr(trainer_module, "hsdp_sync_stream", lambda: None)

    with pytest.raises(RuntimeError, match="cache release failed"):
        trainer._publish_policy(2)

    trainer.rollout_engine.update_weights.assert_called_once()
    trainer.rollout_engine.prepare_for_rollout.assert_not_called()


def test_trainer_completes_metrics_and_evaluation_after_published_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful publication advances step before metrics, evaluation, and logging."""
    calls: list[str] = []
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.state = SimpleNamespace(global_step=0, max_steps=2)
    trainer._log_samples = 1
    trainer._log_steps = 1
    trainer.resolved_config = {"train": {"learning_gate": {"enabled": False}}}
    trainer.rollout_engine = SimpleNamespace()
    trainer._run_rank_synchronized = lambda _name, callback: callback()
    trainer.checkpoints = SimpleNamespace(
        will_save=lambda unused_step: True,
        complete_step=lambda *_args, **_kwargs: None,
    )
    trainer.evaluator = SimpleNamespace(
        run=lambda step: calls.append(f"evaluate:{step}") or ({"validation/accuracy": 1.0}, [])
    )
    trainer._tracker = SimpleNamespace(
        log=lambda unused_metrics, step, **_kwargs: calls.append(f"log:{step}")
    )
    actor_update = ActorUpdateMetrics(0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.01, 2, 1)

    def summarize(*_args: Any, **_kwargs: Any) -> tuple[dict[str, float], list[Any]]:
        """Return scripted rollout metrics."""
        assert trainer.state.global_step == 1
        calls.append("summarize")
        return {"reward/mean": 1.0}, []

    monkeypatch.setattr(trainer_module, "summarize_rollout", summarize)
    monkeypatch.setattr(
        trainer_module,
        "build_training_metrics",
        lambda **_kwargs: calls.append("metrics") or {"train/gradient_norm": 1.0},
    )
    monkeypatch.setattr(
        trainer_module,
        "enforce_learning_gate",
        lambda *_args, **_kwargs: calls.append("gate"),
    )
    monkeypatch.setattr(trainer_module, "uses_colocated_vllm", lambda _config: False)

    SyncTrainer._complete_step(
        trainer,
        step=1,
        batch={},
        rollout=_rollout(),
        actor_update=actor_update,
        critic_update=None,
        diagnostic_metrics={},
    )

    assert trainer.state.global_step == 1
    assert calls == ["summarize", "metrics", "gate", "evaluate:1", "log:1"]


@pytest.mark.parametrize("omit_sampling_probabilities", [False, True])
def test_trainer_runs_one_complete_synchronous_training_step(
    monkeypatch: pytest.MonkeyPatch,
    omit_sampling_probabilities: bool,
) -> None:
    """Production train and step methods complete one strongly synchronous publication."""
    calls: list[str] = []
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer._consistency_profile = "off"
    trainer._old_log_prob_source = "actor" if omit_sampling_probabilities else "rollout"
    trainer.algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    trainer.state = SimpleNamespace(global_step=0, max_steps=1, epoch=0)
    trainer.resolved_config = {
        "train": {
            "checkpoint": {"save_final": False},
            "learning_gate": {"enabled": False},
        }
    }
    trainer.device = torch.device("cpu")
    trainer._log_steps = 1
    trainer._log_samples = 0
    trainer.evaluator = None
    trainer._runtime_started = False
    trainer._dp_group_info = SimpleNamespace(group=None, rank_size=1)
    trainer.parallel_dims = SimpleNamespace(dp_size=1)
    trainer.model_registration = SimpleNamespace(name="qwen")
    batch = {
        "sample_indices": [0],
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.ones((1, 1), dtype=torch.bool),
        "prompts": ["prompt"],
        "ground_truths": ["answer"],
    }
    trainer.train_dataloader = [batch]
    rollout = _rollout()
    recomputed_log_probs = rollout.old_log_probs.clone()
    if omit_sampling_probabilities:
        rollout = replace(rollout, old_log_probs=None, rollout_log_probs=None)
    trainer.rollout_manager = SimpleNamespace(
        generate=lambda **_kwargs: calls.append("rollout") or rollout
    )

    class Engine:
        """Record inference and lifecycle calls without starting a backend."""
        policy_version = 0
        weight_sync_configured_strategy = "full_gather"
        weight_sync_last_strategy = "full_gather"

        @staticmethod
        def prepare_for_training() -> None:
            """Record the switch to training residency."""
            calls.append("prepare-training")

        @staticmethod
        def update_weights(snapshot: Any) -> None:
            """Record a policy publication."""
            calls.append(f"publish:{snapshot.version}")

        @staticmethod
        def prepare_for_rollout() -> None:
            """Record the switch to generation residency."""
            calls.append("prepare-rollout")

        @staticmethod
        def close() -> None:
            """Release the fixture-owned service."""
            calls.append("engine-close")

    class ActorRole:
        """Record actor inference and update calls for orchestration assertions."""
        actor_model = torch.nn.Identity()

        @staticmethod
        def compute_log_probs(experience: ExperienceBatch) -> torch.Tensor:
            """Return scripted policy probabilities."""
            calls.append("actor-logprobs")
            assert experience.sequences is rollout.sequences
            return recomputed_log_probs.clone()

        @staticmethod
        def update(experience: ExperienceBatch) -> ActorUpdateMetrics:
            """Record the optimizer update without executing distributed training."""
            if omit_sampling_probabilities:
                assert experience.rollout_log_probs is None
                assert rollout.old_log_probs is None
                torch.testing.assert_close(experience.old_log_probs, recomputed_log_probs)
            else:
                assert experience is rollout
            calls.append("actor-update")
            return ActorUpdateMetrics(0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.01, 1, 1)

    class ReferenceRole:
        @staticmethod
        def compute_log_probs(experience: ExperienceBatch) -> torch.Tensor:
            """Return scripted policy probabilities."""
            calls.append("reference")
            return torch.zeros_like(experience.sequences[:, 1:], dtype=torch.float32)

    class Preparer:
        @staticmethod
        def prepare(experience: ExperienceBatch, **kwargs: Any) -> ExperienceBatch:
            """Prepare the fixture state for the operation."""
            assert kwargs["reference_log_probs"] is not None
            calls.append("targets")
            return experience

    checkpoints = _recording_checkpoints(calls)
    class Tracker:
        """Tracker."""
        @staticmethod
        def log(unused_metrics: Any, *, step: int, **_kwargs: Any) -> None:
            """Record the metrics emitted by the trainer."""
            del unused_metrics
            calls.append(f"log:{step}")

        @staticmethod
        def finish() -> None:
            """Record tracking shutdown."""
            calls.append("tracker-finish")

    trainer.rollout_engine = Engine()
    trainer.actor = ActorRole()
    trainer.reference_actor = ReferenceRole()
    trainer.critic = None
    trainer.experience_preparer = Preparer()
    trainer.checkpoints = checkpoints
    trainer._tracker = Tracker()
    trainer._release_training_state_for_rollout = lambda: calls.append("release")
    monkeypatch.setattr(trainer_module, "hsdp_sync_stream", lambda: calls.append("stream"))
    monkeypatch.setattr(trainer_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(trainer_module.dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(metrics_module, "_system_memory_metrics", lambda: {})
    monkeypatch.setattr(metrics_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(metrics_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        metrics_module.dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )

    trainer.train()

    assert trainer.state.global_step == 1
    assert calls.index("rollout") < calls.index("prepare-training")
    assert calls.index("prepare-training") < calls.index("reference")
    assert calls.index("reference") < calls.index("targets")
    assert calls.index("targets") < calls.index("actor-update")
    assert calls.index("actor-update") < calls.index("publish:1")
    assert calls.index("publish:1") < calls.index("prepare-rollout")
    assert calls.index("prepare-rollout") < calls.index("checkpoint-step")
    assert calls[-4:] == ["checkpoint-finalize", "barrier", "tracker-finish", "engine-close"]


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_trainer_sets_up_runtime_with_mocked_distributed_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    cpu_offload: bool,
) -> None:
    """Production runtime setup builds device, mesh, DP group, and deterministic seeds."""
    calls: list[Any] = []
    dp_mesh = SimpleNamespace(get_group=lambda: "dp-group")
    mesh_context = SimpleNamespace(
        device_mesh="device-mesh",
        dp_cp_mesh=dp_mesh,
        dp_size=2,
    )
    setup = SimpleNamespace(mesh_context=mesh_context, strategy_config=None)
    handle = SimpleNamespace(set_device=lambda rank: calls.append(("set-device", rank)))
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.runtime_config = SimpleNamespace(
        training=SimpleNamespace(backend="hccl", seed=1234),
        accelerator=SimpleNamespace(tp_size=tp_size),
        fsdp_config=SimpleNamespace(enable_offload=cpu_offload),
    )
    trainer._runtime_started = False
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(
        trainer_module,
        "initialize_distributed",
        lambda backend: calls.append(("initialize", backend)),
    )
    monkeypatch.setattr(
        trainer_module,
        "create_distributed_setup_from_config",
        lambda config: calls.append(("mesh", config)) or setup,
    )
    monkeypatch.setattr(trainer_module.torch.accelerator, 'current_accelerator', lambda: torch.device("cpu"))
    monkeypatch.setattr(trainer_module.torch, 'get_device_module', lambda _kind=None: handle)
    monkeypatch.setattr(trainer_module.torch, "manual_seed", lambda seed: calls.append(("seed", seed)))

    trainer._setup_runtime()

    assert trainer._runtime_started
    assert trainer.distributed_setup is setup
    assert setup.strategy_config is (trainer.runtime_config.fsdp_config if tp_size > 1 or cpu_offload else None)
    assert trainer.parallel_dims is mesh_context
    assert trainer.mesh == "device-mesh"
    assert trainer.device == torch.device("cpu", 1)
    assert trainer._dp_group_info.group == "dp-group"
    assert trainer._dp_group_info.rank_size == 2
    assert calls == [
        ("initialize", "hccl"),
        ("mesh", trainer.runtime_config),
        ("set-device", 1),
        ("seed", 1234),
    ]


def test_trainer_builds_tokenizer_data_and_dataloader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production data construction wires tokenizer, splits, sampler, collate, and loader."""
    calls: list[Any] = []

    class Tokenizer:
        """Provide tokenizer metadata needed for runtime initialization."""
        eos_token_id = 2
        eos_token = "<eos>"
        pad_token_id = None
        padding_side = "right"

        @property
        def pad_token(self) -> str:
            """Expose the EOS token as the fake tokenizer padding token."""
            return self.eos_token

        @pad_token.setter
        def pad_token(self, value: str) -> None:
            """Update the fixture padding token."""
            assert value == self.eos_token
            self.pad_token_id = self.eos_token_id

    class Dataset(PromptDataset):
        """Record dataset construction without invoking real data loading."""

        @staticmethod
        def __len__() -> int:
            return 8

    loader = object()
    sampler = object()

    def build_dataloader(dataset: Any, collate_fn: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Record the source and dataloader settings."""
        calls.append(("loader", dataset, collate_fn, kwargs))
        return loader, sampler

    tokenizer = Tokenizer()
    monkeypatch.setattr(
        model_setup_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: calls.append(("tokenizer", args, kwargs)) or tokenizer,
    )
    monkeypatch.setattr(Dataset, "__init__", lambda _self, **kwargs: calls.append(("dataset", kwargs)))
    monkeypatch.setattr(trainer_module, "PromptDataset", Dataset)
    monkeypatch.setattr(trainer_module, "build_train_dataloader", build_dataloader)
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.resolved_config = {
        "model": {"tokenizer_path": "/model"},
        "data": {
            "train_path": "/train.parquet",
            "test_path": "/test.parquet",
            "max_prompt_length": 32,
            "prompt_column": "prompt",
            "answer_column": "answer",
            "prompt_instruction": "instruction",
            "max_train_samples": 6,
            "shuffle": True,
            "num_workers": 2,
            "prefetch_factor": 3,
            "pin_memory": False,
        },
        "evaluation": {"enabled": True},
        "train": {"prompt_batch_size": 2},
    }
    trainer.parallel_dims = SimpleNamespace(dp_size=2, dp_rank=1)
    trainer.runtime_config = SimpleNamespace(training=SimpleNamespace(seed=1234))

    trainer._build_tokenizer_and_data()

    assert tokenizer.pad_token_id == 2
    assert tokenizer.padding_side == "left"
    dataset_calls = [call for call in calls if call[0] == "dataset"]
    assert len(dataset_calls) == 2
    assert dataset_calls[0][1]["parquet_path"] == "/train.parquet"
    assert dataset_calls[0][1]["max_samples"] == 6
    assert dataset_calls[1][1]["parquet_path"] == "/test.parquet"
    loader_call = next(call for call in calls if call[0] == "loader")
    assert loader_call[3]["batch_size"] == 2
    assert loader_call[3]["rank"] == 1
    assert loader_call[3]["world_size"] == 2
    assert loader_call[3]["seed"] == 1234
    assert loader_call[3]["shuffle"] is True
    assert loader_call[3]["num_workers"] == 2
    assert loader_call[3]["prefetch_factor"] == 3
    assert loader_call[3]["pin_memory"] is False
    assert trainer.train_dataloader is loader
    assert trainer.sampler is sampler


@pytest.mark.parametrize("runner", ["internal", "codex", "deepseek"])
def test_trainer_builds_rollout_evaluator_and_tp_request_owner(
    monkeypatch: pytest.MonkeyPatch,
    runner: str,
) -> None:
    """Production rollout construction wires TP ownership and separate evaluation settings."""
    calls: list[Any] = []

    class Engine:
        def configure_trainer_tensor_parallel(self, **kwargs: Any) -> None:
            """Record the supplied trainer topology."""
            calls.append(("tp", kwargs))

    engine = Engine()
    managers: list[Any] = []

    def manager(**kwargs: Any) -> Any:
        """Construct the task-manager fixture."""
        result = SimpleNamespace(kwargs=kwargs)
        managers.append(result)
        return result

    monkeypatch.setattr(
        trainer_module,
        "build_rollout_engine",
        lambda rollout, model: calls.append(("engine", rollout, model)) or engine,
    )
    monkeypatch.setattr(trainer_module, "RolloutManager", manager)
    monkeypatch.setattr(trainer_module, "CodexRolloutManager", manager)
    monkeypatch.setattr(trainer_module, "DeepSeekRolloutManager", manager)
    runtime = SimpleNamespace(engine=engine)
    monkeypatch.setattr(trainer_module, "CodexRuntime", lambda *_args: runtime)
    monkeypatch.setattr(trainer_module, "DeepSeekRuntime", lambda *_args: runtime)
    monkeypatch.setattr(
        trainer_module,
        "Evaluator",
        lambda **kwargs: calls.append(("evaluator", kwargs)) or SimpleNamespace(**kwargs),
    )
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    trainer.model_registration = SimpleNamespace(name="qwen")
    trainer.model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[2, 3])
    )
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    trainer.parallel_dims = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        dp_rank=0,
        dp_size=1,
        device_mesh={"tp": SimpleNamespace(get_group=lambda: "tp-group")},
    )
    _configure_rollout_test(trainer, runner)
    trainer._evaluation_enabled = True
    trainer.test_dataset = [object()]
    trainer.collate_fn = object()
    trainer.device = torch.device("cpu")

    trainer._build_rollout_runtime()

    tp_call = next(call for call in calls if call[0] == "tp")
    assert tp_call[1] == {
        "group": "tp-group",
        "tp_rank": 1,
        "tp_size": 2,
        "request_rank": 0,
        "request_size": 1,
    }
    assert len(managers) == 2
    if runner == "internal":
        assert managers[0].kwargs["engine"] is engine
        assert managers[1].kwargs["engine"] is engine
        assert managers[0].kwargs["eos_token_ids"] == (2, 3)
    else:
        assert managers[0].kwargs["runtime"] is runtime
        assert managers[1].kwargs["runtime"] is runtime
    assert managers[0].kwargs["num_return_sequences"] == 4
    assert managers[0].kwargs["do_sample"] is True
    assert managers[0].kwargs["collect_old_log_probs"] is True
    assert managers[0].kwargs["seed"] == 10
    assert managers[0].kwargs["max_new_tokens"] == 16
    assert managers[1].kwargs["num_return_sequences"] == 1
    assert managers[1].kwargs["do_sample"] is False
    assert managers[1].kwargs["collect_old_log_probs"] is False
    assert managers[1].kwargs["seed"] == (None if runner == "internal" else 10)
    assert managers[1].kwargs["max_new_tokens"] == 8
    assert trainer.evaluator.rollout_manager is managers[1]


def test_trainer_builds_runtime_and_tracker_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production runtime and tracker methods retain component order and rank-zero config."""
    calls: list[Any] = []
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer._dp_group_info = None
    trainer.algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    trainer.resolved_config = {
        "train": {
            "checkpoint": {"output_dir": "/tmp/checkpoints"},
        },
        "logging": {
            "backends": ["console", "wandb"],
            "project_name": "project",
            "experiment_name": "experiment",
            "log_steps": 2,
            "log_samples": 3,
            "wandb": {
                "mode": "offline",
                "entity": "team",
                "directory": "/tmp/wandb",
            },
        },
    }
    trainer._run_rank_synchronized = lambda _name, callback: callback()
    trainer._build_tokenizer_and_data = lambda: calls.append("data")
    trainer._build_models_and_optimizers = lambda: calls.append("models")
    trainer._build_rollout_runtime = lambda: calls.append("rollout")
    tracker_kwargs: dict[str, Any] = {}
    monkeypatch.setattr(
        trainer_module,
        "RLCheckpointManager",
        lambda *args: calls.append(("checkpoint", args)) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        trainer_module,
        "TrainingTracker",
        lambda **kwargs: tracker_kwargs.update(kwargs) or calls.append("tracker") or SimpleNamespace(),
    )
    monkeypatch.setattr(trainer_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_module.dist, "get_world_size", lambda: 4)

    trainer._build_runtime()

    assert calls[:3] == ["data", "models", "rollout"]
    assert calls[3][0] == "checkpoint"
    assert calls[4] == "tracker"
    assert trainer.experience_preparer.algorithm is trainer.algorithm
    assert trainer._log_steps == 2
    assert trainer._log_samples == 3
    assert tracker_kwargs["rank"] == 0
    assert tracker_kwargs["world_size"] == 4
    assert tracker_kwargs["backends"] == ("console", "wandb")
    assert tracker_kwargs["wandb_mode"] == "offline"


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
def test_trainer_validates_supported_shared_runtime_topology(
    monkeypatch: pytest.MonkeyPatch,
    deployment: str,
) -> None:
    """Production topology validation accepts one consistent shared DP2xTP1 deployment."""
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.parallel_dims = SimpleNamespace(
        dp_size=2,
        cp_size=1,
        tp_size=1,
        pp_size=1,
    )
    vllm: dict[str, Any] = {
        "deployment": deployment,
        "data_parallel_size": 2,
        "tensor_parallel_size": 1,
        "port": 8100,
    }
    if deployment == "disjoint":
        vllm["visible_devices"] = "4,5"
    trainer.resolved_config = {"rollout": {"engine": "vllm", "vllm": vllm}}
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(trainer_module.dist, "get_world_size", lambda: 2)

    def gather(output: list[Any], value: Any) -> None:
        """Supply the peer-rank diagnostic records."""
        output[:] = [value, value]

    monkeypatch.setattr(trainer_module.dist, "all_gather_object", gather)

    trainer._validate_runtime_topology()


def test_trainer_releases_colocated_training_state_for_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production residency release reshards roles, validates CPU state, and clears cache."""
    calls: list[str] = []
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.resolved_config = {
        "rollout": {"engine": "vllm", "vllm": {"deployment": "colocated"}}
    }
    trainer.actor = SimpleNamespace(actor_model="actor")
    trainer.reference_actor = SimpleNamespace(actor_model="reference")
    trainer.critic = SimpleNamespace(
        critic_model="critic",
        optimizer=SimpleNamespace(state={}),
    )
    trainer.optimizer = SimpleNamespace(state={})
    trainer._reshard_model = lambda model: calls.append(f"reshard:{model}")
    trainer._run_rank_synchronized = lambda name, callback: calls.append(name) or callback()
    stream = SimpleNamespace(synchronize=lambda: calls.append("stream"))
    handle = SimpleNamespace(empty_cache=lambda: calls.append("empty-cache"))
    monkeypatch.setattr(trainer_module, "hsdp_sync_stream", lambda: calls.append("hsdp-sync"))
    handle.current_stream = lambda: stream
    monkeypatch.setattr(trainer_module.torch.accelerator, 'current_accelerator', lambda: torch.device("cpu"))
    monkeypatch.setattr(trainer_module.torch, 'get_device_module', lambda _kind=None: handle)

    trainer._release_training_state_for_rollout()

    assert calls == [
        "training-state release",
        "hsdp-sync",
        "reshard:actor",
        "reshard:reference",
        "reshard:critic",
        "stream",
        "allocator-cache release",
        "empty-cache",
        "stream",
    ]


class _CPUOffloadRole(torch.nn.Module, HSDPModule):
    """Expose real HSDP storage lifecycle with CPU-only local computation."""

    def __init__(self, *, requires_grad: bool) -> None:
        """Create one meta parameter before the test materializes its local shard."""
        torch.nn.Module.__init__(self)
        HSDPModule.__init__(self)
        self.weight = torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16, device="meta"),
                                         requires_grad=requires_grad)
        self.forward_calls = 0

    def forward(self) -> torch.Tensor:
        """Compute a real differentiable scalar from the materialized local parameter."""
        self.forward_calls += 1
        return self.weight.to_local().square().sum()


def _cpu_offload_role(
    monkeypatch: pytest.MonkeyPatch, *, requires_grad: bool, materialized: bool = True,
) -> tuple[_CPUOffloadRole, HSDPParamV2, HSDPStateV2]:
    """Build actual storage/state helpers without process groups or a pin-memory backend."""
    monkeypatch.setattr("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", lambda: 0)
    mesh_info = object.__new__(FSDPMeshInfo)
    mesh_info.mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("fsdp",), _init_backend=False)
    mesh_info.shard_mesh_dim = 0
    mesh_info.replicate_mesh_dim = None
    mesh_info.shard_mesh_rank = 0
    mesh_info.shard_mesh_size = 2
    mesh_info.shard_process_group = None
    role = _CPUOffloadRole(requires_grad=requires_grad)
    precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    parameter = HSDPParamV2(
        role.weight, ParamModuleInfo(role, "weight", [], []), mesh_info,
        mp_policy=precision, device=torch.device("cpu"), offload_policy=CPUOffloadPolicy(),
    )
    if materialized:
        parameter.sharded_param._local_tensor = torch.tensor([3.0, 5.0], dtype=torch.bfloat16,
                                                            requires_grad=requires_grad)
    state = object.__new__(HSDPStateV2)
    state.is_shard = True
    state.module_name = "role"
    state._reset_sharded_params = False
    state.hsdp_params = [parameter]
    state.offload_policy = CPUOffloadPolicy()
    state.mp_policy = precision
    role.hsdp_scheduler = SimpleNamespace(hsdp_state=state)
    # Clone emulates the differentiable pinned-host allocation on a CPU-only runner.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor.clone())
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda tensor: False)
    return role, parameter, state


def test_initial_rollout_release_resets_real_actor_reference_storage_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh shards reset before rollout without changing optimizer identity or autograd."""
    actor, actor_parameter, actor_state = _cpu_offload_role(monkeypatch, requires_grad=True)
    reference, reference_parameter, reference_state = _cpu_offload_role(monkeypatch, requires_grad=False)
    original_actor = actor.weight
    original_reference = reference.weight
    previous_local = weakref.ref(actor.weight.to_local())
    previous_storage = StorageWeakRef(actor.weight.to_local().untyped_storage())
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.resolved_config = {"rollout": {"engine": "vllm", "vllm": {"deployment": "colocated"}}}
    trainer.actor = SimpleNamespace(actor_model=actor)
    trainer.reference_actor = SimpleNamespace(actor_model=reference)
    trainer.critic = None
    trainer.optimizer = optimizer
    trainer._run_rank_synchronized = lambda name, callback: callback()
    stream = SimpleNamespace(synchronize=lambda: None)
    handle = SimpleNamespace(empty_cache=lambda: None, current_stream=lambda: stream)
    monkeypatch.setattr(trainer_module, "hsdp_sync_stream", lambda: None)
    monkeypatch.setattr(trainer_module.torch, "get_device_module", lambda: handle)

    trainer._release_training_state_for_rollout()

    assert actor.forward_calls == reference.forward_calls == 0
    assert actor_state._reset_sharded_params and reference_state._reset_sharded_params
    assert actor.weight is original_actor and reference.weight is original_reference
    assert optimizer.param_groups[0]["params"][0] is original_actor
    assert actor.weight.device.type == reference.weight.device.type == "cpu"
    assert actor.weight.requires_grad and not reference.weight.requires_grad
    for parameter, requires_grad in ((actor_parameter, True), (reference_parameter, False)):
        assert not parameter._sharded_param_data.is_meta
        assert parameter._sharded_param_data.grad_fn is None
        local_tensor = parameter.sharded_param._local_tensor
        assert not parameter._sharded_param_data.requires_grad, (
            f"Expected non-trainable communication storage, got {parameter._sharded_param_data.requires_grad}"
        )
        assert local_tensor.requires_grad == requires_grad, (
            f"Expected local requires_grad={requires_grad}, got {local_tensor.requires_grad}"
        )
        assert local_tensor.is_leaf, f"Expected a local leaf, got grad_fn={local_tensor.grad_fn}"
        assert local_tensor.grad_fn is None, f"Expected no local conversion graph, got {local_tensor.grad_fn}"
        assert parameter._sharded_param_data.data_ptr() == local_tensor.data_ptr(), (
            f"Expected shared shard storage: communication={parameter._sharded_param_data.data_ptr()}, "
            f"local={local_tensor.data_ptr()}"
        )
        assert parameter.reduce_dtype is torch.float32
        torch.testing.assert_close(parameter.sharded_param.to_local(), torch.tensor([3.0, 5.0], dtype=torch.bfloat16))
    gc.collect()
    assert previous_local() is None
    assert previous_storage.expired(), (
        f"Expected the previous local storage to expire, got {previous_storage.expired()}"
    )
    pointers = [parameter._sharded_param_data.data_ptr() for parameter in (actor_parameter, reference_parameter)]

    trainer._release_training_state_for_rollout()

    assert pointers == [
        parameter._sharded_param_data.data_ptr() for parameter in (actor_parameter, reference_parameter)
    ]
    assert optimizer.param_groups[0]["params"][0] is original_actor
    actor().backward()
    torch.testing.assert_close(actor.weight.to_local().grad, torch.tensor([6.0, 10.0], dtype=torch.bfloat16))
    assert reference.weight.to_local().grad is None
    trainer._release_training_state_for_rollout()
    torch.testing.assert_close(actor.weight.to_local().grad, torch.tensor([6.0, 10.0], dtype=torch.bfloat16))

    loss = actor()
    local_version = actor.weight.to_local()._version
    with torch.no_grad():
        actor_parameter._sharded_param_data.add_(1)
    assert actor.weight.to_local()._version > local_version, (
        f"Expected a shared version update: before={local_version}, after={actor.weight.to_local()._version}"
    )
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize("requires_grad", [True, False])
def test_rollout_reshard_frees_full_parameter_storage(
    monkeypatch: pytest.MonkeyPatch, requires_grad: bool,
) -> None:
    """Rollout releases full storage for both trainable and frozen offloaded roles."""
    model, parameter, state = _cpu_offload_role(monkeypatch, requires_grad=requires_grad)
    state.lazy_init()
    sharded_parameter = model.weight
    shard_pointer = parameter._sharded_param_data.data_ptr()
    # Materialize a gathered result without starting distributed communication.
    parameter.init_unsharded_param_buffers(
        all_gather_input_numels=[2], all_gather_input_dtypes=[torch.bfloat16],
        world_size=2, device=torch.device("cpu"),
    )
    parameter.unsharded_param_buffers[0].copy_(torch.tensor([3.0, 5.0, 7.0, 11.0], dtype=torch.bfloat16))
    parameter.init_unsharded_param()
    parameter.to_unsharded()
    state.is_shard = False
    full_parameter = model.weight
    full_storage = full_parameter.untyped_storage()
    assert full_storage.nbytes() > 0, f"Expected allocated full storage, got {full_storage.nbytes()} bytes"

    SyncTrainer._reshard_model(model)

    assert full_storage.nbytes() == 0, f"Expected released full storage, got {full_storage.nbytes()} bytes"
    assert model.weight is sharded_parameter, f"Expected the original shard, got {type(model.weight)}"
    assert parameter._sharded_param_data.data_ptr() == shard_pointer, (
        f"Expected unchanged shard storage: before={shard_pointer}, after={parameter._sharded_param_data.data_ptr()}"
    )
    assert model.weight.requires_grad == requires_grad, (
        f"Expected shard requires_grad={requires_grad}, got {model.weight.requires_grad}"
    )
    torch.testing.assert_close(model.weight.to_local(), torch.tensor([3.0, 5.0], dtype=torch.bfloat16))


def test_initial_rollout_release_rejects_unmaterialized_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unfinished meta build fails the existing HSDP initialization validator."""
    model, _, _ = _cpu_offload_role(monkeypatch, requires_grad=True, materialized=False)

    with pytest.raises(RuntimeError, match="materialized from meta"):
        SyncTrainer._reshard_model(model)


def test_initial_rollout_release_retains_cpu_offload_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already initialized offload state must reject a non-CPU shard before rollout."""
    model, parameter, state = _cpu_offload_role(monkeypatch, requires_grad=True)
    state._reset_sharded_params = True
    parameter.sharded_param = SimpleNamespace(device=torch.device("cuda"))

    with pytest.raises(RuntimeError, match="materialized on CPU"):
        SyncTrainer._reshard_model(model)


@pytest.mark.parametrize("reference_offload", [None, False, True])
@pytest.mark.parametrize("token_budget", [None, 3000])
def test_trainer_builds_required_models_optimizers_and_role_runtime(
    monkeypatch: pytest.MonkeyPatch,
    reference_offload: bool | None,
    token_budget: int | None,
) -> None:
    """GRPO construction creates independent trainable/reference models and Actor roles."""
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.algorithm = build_algorithm(
        {"name": "grpo", "loss_aggregation": "token-mean"}
    )
    trainer.resolved_config = {
        "train": {
            "optimizer": {"max_grad_norm": 2.0},
            "micro_batch_size": 1,
            "response_mini_batch_size": 2,
            "policy_update_epochs": 3,
            "max_tokens_per_micro_batch": token_budget,
            "old_log_prob_max_tokens_per_micro_batch": 4096,
            "reference_log_prob_max_tokens_per_micro_batch": 8192,
        }
    }
    trainer.device = torch.device("cpu")
    trainer._dp_group_info = "dp-group"
    trainer.parallel_dims = SimpleNamespace(dp_size=2)
    trainer.runtime_config = SimpleNamespace(fsdp_config=SimpleNamespace(enable_offload=False))
    trainer.distributed_setup = DistributedSetup(strategy_config=trainer.runtime_config.fsdp_config)
    reference_runtime = SimpleNamespace(fsdp_config=SimpleNamespace(enable_offload=reference_offload))
    if reference_offload is not None:
        trainer.resolved_config["train"]["accelerator"] = {"reference_cpu_offload": reference_offload}
    runtime_builder = Mock(return_value=reference_runtime)
    monkeypatch.setattr(trainer_module, "build_runtime_config", runtime_builder)
    setups = []
    trainable_model = torch.nn.Linear(2, 2)
    reference_model = torch.nn.Linear(2, 2)
    models = iter((trainable_model, reference_model))
    frozen_flags = []

    def build_model(unused_runtime: Any, unused_setup: Any, *, frozen: bool, cpu_threads: Any = None,
                    **options: Any) -> Any:
        """Record model construction requests and return the test model."""
        del unused_runtime, cpu_threads
        if options:
            assert options == {"token_log_probs_mode": "auto", "token_log_probs_chunk_size": 512}
        setups.append(unused_setup)
        frozen_flags.append(frozen)
        return next(models)

    monkeypatch.setattr(trainer_module, "build_role_model", build_model)
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.01}])
    scheduler = SimpleNamespace()
    monkeypatch.setattr(
        trainer_module,
        "build_role_optimizer",
        lambda unused_runtime, _model: (optimizer, scheduler),
    )
    actor_calls = []

    def actor(**kwargs: Any) -> Any:
        """Construct the policy fixture."""
        actor_calls.append(kwargs)
        return SimpleNamespace(
            actor_model=kwargs["actor_model"],
            optimizer=kwargs.get("optimizer"),
            lr_scheduler=kwargs.get("lr_scheduler"),
        )

    monkeypatch.setattr(trainer_module, "Actor", actor)

    trainer._build_models_and_optimizers()

    assert frozen_flags == [False, True]
    assert setups[0] is trainer.distributed_setup
    assert trainer.distributed_setup.strategy_config.enable_offload is False
    assert setups[1].mesh_context is trainer.distributed_setup.mesh_context
    if reference_offload is not None:
        runtime_builder.assert_called_once_with(trainer.resolved_config, reference=True)
        assert setups[1].strategy_config is reference_runtime.fsdp_config
        assert setups[1] is not setups[0]
    else:
        runtime_builder.assert_not_called()
        assert setups[1] is setups[0]
    assert actor_calls[0] == {
        "actor_model": trainable_model,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "device": torch.device("cpu"),
        "dp_group_info": "dp-group",
        "dp_size": 2,
        "response_mini_batch_size": 2,
        "update_epochs": 3,
        "max_grad_norm": 2.0,
        "gradient_sync_each_micro_batch": False,
        "max_tokens_per_micro_batch": token_budget,
        "balance_dp_tokens": False,
        "log_prob_max_tokens_per_micro_batch": 4096,
        "distillation": None,
        "algorithm": trainer.algorithm,
        "micro_batch_size": 1,
    }
    assert actor_calls[1] == {
        "actor_model": reference_model,
        "dp_group_info": "dp-group",
        "dp_size": 2,
        "log_prob_max_tokens_per_micro_batch": 8192,
        "algorithm": trainer.algorithm,
        "micro_batch_size": 1,
    }
    assert trainer.actor.actor_model is trainable_model
    assert trainer.reference_actor.actor_model is reference_model
    assert trainer.critic is None
    assert trainer.model is trainable_model
    assert trainer.optimizer is optimizer
    assert trainer.lr_scheduler is scheduler


def test_trainer_runtime_helpers_cover_epoch_cleanup_and_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader rollover, recursive CPU state, reshard, cleanup, and rank sync compose."""
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.state = SimpleNamespace(epoch=0)
    epochs = []
    trainer.sampler = SimpleNamespace(set_epoch=lambda epoch: epochs.append(epoch))
    trainer.train_dataloader = [{"sample": 2}]

    batch, iterator = trainer._next_batch(iter(()))

    assert batch == {"sample": 2}
    assert next(iter([iterator])) is iterator
    assert trainer.state.epoch == 1
    assert epochs == [1]

    roots = [
        SimpleNamespace(
            reshard=lambda: epochs.append("reshard"),
            hsdp_scheduler=SimpleNamespace(hsdp_state=SimpleNamespace(lazy_init=lambda: epochs.append("lazy-init"))),
        ) for _ in range(2)
    ]
    monkeypatch.setattr(trainer_module, "iter_hsdp_roots", lambda _model: roots)
    trainer._reshard_model(object())
    trainer._validate_optimizer_cpu_residency(
        SimpleNamespace(
            state={
                "parameter": {
                    "step": torch.tensor(1),
                    "moments": [torch.tensor([1.0]), (torch.tensor([2.0]),)],
                }
            }
        ),
        "actor",
    )
    assert epochs[-4:] == ["lazy-init", "reshard", "lazy-init", "reshard"]

    monkeypatch.setattr(trainer_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        trainer_module.dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value, value]),
    )
    assert trainer._run_rank_synchronized("success", lambda: "result") == "result"

    cleanup = []
    trainer._tracker = SimpleNamespace(finish=lambda: cleanup.append("tracker"))
    trainer.rollout_engine = SimpleNamespace(close=lambda: cleanup.append("rollout"))
    trainer._runtime_started = True
    monkeypatch.setattr(
        cleanup_module,
        "destroy_process_group",
        lambda: cleanup.append("distributed"),
    )

    trainer._cleanup()

    assert cleanup == ["tracker", "rollout", "distributed"]
    assert trainer._tracker is None
    assert not trainer._runtime_started


def test_trainer_complete_step_coordinates_colocated_checkpoint_residency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saving step evaluates, enters training residency, commits, and restores rollout."""
    calls = []
    trainer = object.__new__(SyncTrainer)
    trainer.distillation_config = DistillationConfig()
    trainer.state = SimpleNamespace(global_step=0, max_steps=2)
    trainer._log_samples = 0
    trainer._log_steps = 10
    trainer.resolved_config = {"train": {"learning_gate": {"enabled": False}}}
    trainer._run_rank_synchronized = lambda _name, callback: callback()
    trainer.checkpoints = SimpleNamespace(
        will_save=lambda unused_step: True,
        complete_step=lambda *_args, **_kwargs: calls.append("checkpoint"),
    )
    trainer.evaluator = SimpleNamespace(
        run=lambda step: calls.append(f"evaluate:{step}")
        or ({"validation/accuracy": 1.0}, [{"sample": step}])
    )
    trainer._tracker = SimpleNamespace(
        log=lambda *_args, **_kwargs: calls.append("log")
    )
    trainer.rollout_engine = SimpleNamespace(
        prepare_for_training=lambda: calls.append("training"),
        prepare_for_rollout=lambda: calls.append("rollout"),
    )
    trainer._release_training_state_for_rollout = lambda: calls.append("release")
    trainer._move_training_state = lambda *, offload: calls.append("offload" if offload else "restore")
    monkeypatch.setattr(trainer_module, "uses_colocated_vllm", lambda _config: True)
    monkeypatch.setattr(trainer_module, "summarize_rollout", lambda *_args, **_kwargs: ({}, []))
    monkeypatch.setattr(trainer_module, "build_training_metrics", lambda **_kwargs: {})
    monkeypatch.setattr(trainer_module, "enforce_learning_gate", lambda *_args, **_kwargs: None)
    update = SimpleNamespace(total_loss=0.1, gradient_norm=1.0)

    SyncTrainer._complete_step(
        trainer,
        step=1,
        batch={},
        rollout=_rollout(),
        actor_update=update,
        critic_update=None,
        diagnostic_metrics={},
    )

    assert trainer.state.global_step == 1
    assert calls == [
        "evaluate:1",
        "log",
        "training",
        "restore",
        "release",
        "checkpoint",
        "release",
        "offload",
        "rollout",
    ]


def _install_step_diagnostics(monkeypatch, calls):
    """Record forward-contract and consistency calls for step-order assertions."""
    monkeypatch.setattr(trainer_module, "hsdp_sync_stream", lambda: None)
    monkeypatch.setattr(trainer_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        trainer_module,
        "validate_consistency_forward_inputs",
        lambda *_args, **_kwargs: calls.append("forward-contract"),
    )
    monkeypatch.setattr(
        trainer_module,
        "validate_pre_update_consistency",
        lambda *_args, **_kwargs: calls.append("exact-gate")
        or {"training/exact": 1.0},
    )
    monkeypatch.setattr(
        trainer_module,
        "summarize_training_diagnostics",
        lambda *_args: calls.append("diagnostics") or {"training/tokens": 1.0},
    )
    monkeypatch.setattr(
        trainer_module,
        "measure_post_update_old_policy_mismatch",
        lambda *_args, **_kwargs: calls.append("post-update-check")
        or {"training/post_update": 1.0},
    )


def _recording_checkpoints(calls):
    """Create the checkpoint double used by the complete synchronous step test."""
    class Checkpoints:
        """Record save and restore operations without filesystem side effects."""
        @staticmethod
        def validate_resume() -> None:
            """Represent the scripted validate resume boundary."""
            calls.append("validate-resume")

        @staticmethod
        def begin(unused_state: Any) -> None:
            """Represent the scripted begin boundary."""
            del unused_state
            calls.append("checkpoint-begin")

        @staticmethod
        def will_save(unused_step: int) -> bool:
            """Represent the scripted will save boundary."""
            del unused_step
            return False

        @staticmethod
        def complete_step(unused_state: Any, **_kwargs: Any) -> None:
            """Record checkpoint completion in the fixture."""
            del unused_state
            calls.append("checkpoint-step")

        @staticmethod
        def finalize(unused_state: Any) -> None:
            """Represent the scripted finalize boundary."""
            del unused_state
            calls.append("checkpoint-finalize")
    return Checkpoints()


def _configure_rollout_test(trainer, runner):
    """Configure independent training and evaluation rollout settings."""
    trainer.resolved_config = {
        "train": {},
        "rollout": {
            "engine": "vllm",
            "num_return_sequences": 4,
            "max_new_tokens": 16,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 5,
            "seed": 10,
        },
        "agentic": {
            "runner": runner,
            "codex": {},
            "deepseek": {},
            "environment": "env",
            "max_turns": 2,
            "max_observation_tokens": 8,
            "max_episode_tokens": 32,
            "interaction_mode": "multi_turn",
        },
        "evaluation": {
            "enabled": True,
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "do_sample": False,
            "batch_size": 2,
            "max_samples": 4,
            "log_samples": 1,
            "progress_steps": 1,
        },
    }


def test_balance_prepares_groups_before_anchor_forward_without_rebuilding_targets() -> None:
    """Complete-group targets are calculated once and carried through redistribution to both anchors."""
    trainer = object.__new__(SyncTrainer)
    trainer._balance_dp_tokens = True
    trainer._old_log_prob_source = "actor"
    trainer._consistency_profile = "off"
    trainer.algorithm = SimpleNamespace(name="grpo", requirements=SimpleNamespace(
        data=SimpleNamespace(reference_log_probs=True, values=False)))
    trainer._run_rank_synchronized = lambda _label, fn: fn()
    raw = replace(_rollout(), old_log_probs=None)
    targeted = replace(raw, advantages=torch.ones((1, 1)))
    moved = replace(targeted, sequences=torch.tensor([[3, 4]]))
    old, reference = torch.full((1, 1), -0.2), torch.full((1, 1), -0.3)
    trainer.actor = Mock(balance_rollout=Mock(return_value=moved), compute_log_probs=Mock(return_value=old))
    trainer.reference_actor = Mock(compute_log_probs=Mock(return_value=reference))
    trainer.experience_preparer = Mock(prepare_targets=Mock(return_value=targeted))
    trainer.experience_preparer.attach_role_outputs.side_effect = lambda batch, **kw: replace(
        batch, reference_log_probs=kw["reference_log_probs"])
    actual, _ = trainer._prepare_experience(raw, False, {})
    trainer.experience_preparer.prepare_targets.assert_called_once_with(raw)
    trainer.actor.balance_rollout.assert_called_once_with(targeted)
    trainer.actor.compute_log_probs.assert_called_once_with(moved)
    trainer.reference_actor.compute_log_probs.assert_called_once_with(moved)
    assert actual.advantages is targeted.advantages
    assert raw.old_log_probs is None and raw.advantages is None
    torch.testing.assert_close(actual.old_log_probs, old)
    torch.testing.assert_close(actual.reference_log_probs, reference)


class TestTrainingContracts(unittest.TestCase):
    """Policy anchors and validation scheduling remain independent of optional diagnostics."""

    def test_probability_source_and_diagnostics_are_independent(self) -> None:
        """Diagnostics alone must not change anchors; actor selection forces inference."""
        for source, diagnostics, sampled_present in (("rollout", False, True), ("rollout", True, True),
                                                     ("actor", False, True), ("actor", True, False)):
            with self.subTest(source=source, diagnostics=diagnostics, sampled_present=sampled_present):
                sampled = torch.full((1, 1), -2.0) if sampled_present else None
                rollout = replace(_rollout(), old_log_probs=sampled,
                                  rollout_log_probs=sampled.clone() if sampled_present else None)
                recomputed = torch.full((1, 1), -1.5, requires_grad=True)
                trainer = SyncTrainer.__new__(SyncTrainer)
                trainer._old_log_prob_source = source
                trainer._consistency_profile = "off"
                trainer.algorithm = SimpleNamespace(
                    name="grpo",
                    requirements=SimpleNamespace(data=SimpleNamespace(reference_log_probs=False, values=False)),
                )
                trainer.actor = Mock()
                trainer.actor.compute_log_probs.return_value = recomputed
                trainer._run_rank_synchronized = lambda _operation, callback: callback()
                trainer.experience_preparer = Mock()
                trainer.experience_preparer.prepare.side_effect = lambda batch, **_kwargs: batch
                timings = {}
                with patch("rl.trainer.summarize_training_diagnostics", return_value={}) as summary:
                    prepared, _ = trainer._prepare_experience(rollout, diagnostics, timings)
                self.assertEqual(summary.call_count, int(diagnostics))
                expected = recomputed.detach() if source == "actor" else sampled
                torch.testing.assert_close(prepared.old_log_probs, expected)
                if sampled_present:
                    torch.testing.assert_close(prepared.rollout_log_probs, sampled)
                    torch.testing.assert_close(rollout.old_log_probs, sampled)
                else:
                    self.assertIsNone(prepared.rollout_log_probs)
                    self.assertIsNone(rollout.old_log_probs)
                self.assertFalse(prepared.old_log_probs.requires_grad)
                self.assertEqual(trainer.actor.compute_log_probs.call_count, int(diagnostics or source == "actor"))
                self.assertEqual("old_log_prob" in timings, diagnostics or source == "actor")

    def test_evaluation_schedule_without_checkpoint_and_without_final_duplicate(self) -> None:
        """Evaluate the initial and published final policy once with saving disabled."""
        for last_step in (5, 6):
            with self.subTest(last_step=last_step):
                trainer = SyncTrainer.__new__(SyncTrainer)
                trainer.resolved_config = {
                    "train": {"checkpoint": {"save_final": False}},
                    "evaluation": {"before_train": True, "every_n_steps": 5, "at_end": True},
                }
                trainer.state = SimpleNamespace(global_step=0, max_steps=last_step)
                trainer.checkpoints = Mock()
                trainer.checkpoints.will_save.return_value = False
                trainer.rollout_engine = SimpleNamespace(policy_version=0)
                trainer._release_training_state_for_rollout = Mock()
                trainer.rollout_manager = SimpleNamespace(preparation_enabled=False)
                trainer.train_dataloader = [object()] * last_step
                trainer.evaluator = Mock(last_step=-1)
                calls = []

                def evaluate(step: int, *, trainer=trainer, record_call=calls.append) -> tuple[dict[str, float], list]:
                    """Record the exact published policy version."""
                    record_call(step)
                    self.assertEqual(trainer.state.global_step, step)
                    trainer.evaluator.last_step = step
                    return {"validation/accuracy": 0.5}, []

                def train_step(batch: object, *, trainer=trainer) -> None:
                    """Advance a simulated successful publication."""
                    del batch
                    trainer.state.global_step += 1
                    trainer.rollout_engine.policy_version = trainer.state.global_step
                    if trainer._evaluation_due(trainer.state.global_step):
                        trainer.evaluator.run(trainer.state.global_step)

                trainer.evaluator.run.side_effect = evaluate
                trainer._tracker = Mock()
                trainer._next_batch = lambda iterator: (next(iterator), iterator)
                trainer._train_step = train_step
                self.assertTrue(trainer._train_and_checkpoint())
                self.assertEqual(calls, [0, 5] if last_step == 5 else [0, 5, 6])
                trainer.checkpoints.will_save.assert_not_called()

    def test_periodic_evaluation_preserves_legacy_schedule_when_unspecified(self) -> None:
        """An explicit interval overrides saving; omitted intervals retain existing behavior."""
        trainer = SyncTrainer.__new__(SyncTrainer)
        trainer.evaluator = Mock()
        trainer.checkpoints = Mock()
        trainer.checkpoints.will_save.return_value = True
        for interval, expected in ((None, True), (0, False), (5, True), (3, False)):
            with self.subTest(interval=interval):
                trainer.resolved_config = {"evaluation": {"every_n_steps": interval}}
                self.assertEqual(trainer._evaluation_due(5), expected)
                self.assertFalse(trainer._evaluation_due(0))
        trainer.evaluator = None
        self.assertFalse(trainer._evaluation_due(5))


class TestCodexPreparation(unittest.TestCase):
    """Lookahead preserves order and never crosses evaluation, save or epoch boundaries."""

    def test_preparation_and_boundaries(self) -> None:
        """Only a normal step consumes the next batch, using CPU inputs and its target policy."""
        for boundary in ("normal", "evaluation", "checkpoint", "last_step", "disabled", "empty"):
            with self.subTest(boundary=boundary):
                trainer = object.__new__(SyncTrainer)
                trainer.state = SimpleNamespace(global_step=0, max_steps=3, epoch=4)
                trainer.rollout_manager = SimpleNamespace(preparation_enabled=boundary != "disabled", prepare=Mock())
                trainer._evaluation_due = lambda step, boundary=boundary: boundary == "evaluation"
                trainer.checkpoints = SimpleNamespace(will_save=lambda step, boundary=boundary: boundary == "checkpoint")
                trainer._run_rank_synchronized = lambda _name, operation: operation()
                trainer._prefetched_batch = None
                batch = {"sample_indices": [1], "input_ids": torch.tensor([[1, 2]]),
                         "attention_mask": torch.ones(1, 2, dtype=torch.bool),
                         "prompts": ["question"], "ground_truths": ["42"]}
                trainer._prefetch_iterator = iter([] if boundary == "empty" else [batch])
                trainer._prepare_next_rollout(3 if boundary == "last_step" else 1)
                self.assertEqual(trainer.state.epoch, 4)
                if boundary == "normal":
                    self.assertIs(trainer._prefetched_batch, batch)
                    prompts, version = trainer.rollout_manager.prepare.call_args.args
                    self.assertEqual((prompts[0].prompt_id, version), ("1", 1))
                    self.assertEqual(prompts[0].metadata["input_ids"].device.type, "cpu")
                    self.assertIsNone(next(trainer._prefetch_iterator, None))
                else:
                    trainer.rollout_manager.prepare.assert_not_called()
                    self.assertIsNone(trainer._prefetched_batch)
                    self.assertIs(next(trainer._prefetch_iterator, None), None if boundary == "empty" else batch)
