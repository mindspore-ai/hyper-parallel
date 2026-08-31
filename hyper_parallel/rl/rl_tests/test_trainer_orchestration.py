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
"""CPU tests for RL2-style role orchestration in the synchronous trainer."""
# These tests intentionally inspect orchestration state and invoke internal hooks.
# pylint: disable=protected-access

import os
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

import rl.agentic.core.runner as runner_backend
from rl.algorithm import build_algorithm
from rl.agentic.core.runner import AgentRunner
from rl.consistency import (
    CONSISTENCY_PROFILE_OFF,
    QWEN3_ASCEND_CONSISTENCY_V1,
    measure_post_update_old_policy_mismatch,
    validate_pre_update_consistency,
)
from rl.dataset.contracts import ExperienceBatch
import rl.trainer as trainer_backend
import rl.utils.monitoring.metrics as metrics_backend
from rl.roles import Actor
from rl.trainer import SyncTrainer, _configure_batch_invariant_communication
from rl.utils.monitoring.metrics import ActorUpdateMetrics, build_training_metrics


class _StopAfterRolePipeline(RuntimeError):
    """Stop the trainer after the role pipeline reaches weight publication."""


def test_training_metrics_expose_effective_weight_sync_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication metrics distinguish direct success from fallback usage."""
    monkeypatch.setattr(metrics_backend, "_system_memory_metrics", lambda: {})
    policy = SimpleNamespace(
        policy_version=2,
        policy_fingerprint="digest-v2",
        policy_fingerprint_changed=True,
        weight_sync_configured_strategy="direct_reshard",
        weight_sync_last_strategy="direct_reshard",
        weight_sync_fallback_count=0,
        weight_sync_direct_success_count=2,
    )
    actor_update = ActorUpdateMetrics(
        total_loss=0.0,
        policy_loss=0.0,
        kl_loss=0.0,
        old_policy_kl=0.0,
        old_current_log_ratio_abs=0.0,
        clip_fraction=0.0,
        gradient_norm=1.0,
        learning_rate=1.0e-5,
        valid_tokens=8,
        optimizer_steps=1,
    )

    metrics = build_training_metrics(
        step=2,
        actor_update=actor_update,
        rollout_metrics={},
        policy=policy,
    )

    assert metrics["weight_sync/configured_direct_reshard"] == 1.0
    assert metrics["weight_sync/last_direct_reshard"] == 1.0
    assert metrics["weight_sync/last_full_gather"] == 0.0
    assert metrics["weight_sync/fallback_count"] == 0.0
    assert metrics["weight_sync/direct_success_count"] == 2.0


def test_batch_invariant_rollout_aligns_hccl_before_distributed_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trainer and vLLM workers must cache identical deterministic HCCL options."""
    monkeypatch.setenv("HCCL_DETERMINISTIC", "false")
    monkeypatch.setenv("LCCL_DETERMINISTIC", "0")

    _configure_batch_invariant_communication(
        {
            "rollout": {
                "engine": "vllm",
                "vllm": {"batch_invariant": True},
            }
        }
    )

    assert os.environ["HCCL_DETERMINISTIC"] == "strict"
    assert os.environ["LCCL_DETERMINISTIC"] == "1"


def _shared_topology_trainer(deployment: str = "colocated") -> SyncTrainer:
    """Build the minimal Trainer state required for topology validation."""
    trainer = object.__new__(SyncTrainer)
    trainer.runtime_config = SimpleNamespace(
        fsdp_config=SimpleNamespace(dp_shard_size=2)
    )
    trainer.parallel_dims = SimpleNamespace(
        dp_size=2,
        cp_size=1,
        tp_size=1,
        pp_size=1,
    )
    trainer.resolved_config = {
        "rollout": {
            "engine": "vllm",
            "vllm": {
                "deployment": deployment,
                "data_parallel_size": 2,
                "tensor_parallel_size": 1,
                "port": 8100,
            },
        }
    }
    if deployment == "disjoint":
        trainer.resolved_config["rollout"]["vllm"]["visible_devices"] = "2,3"
    return trainer


def test_runtime_topology_accepts_fsdp2_tp2_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime validates the full mesh instead of equating world size with FSDP."""
    trainer = _shared_topology_trainer()
    trainer.parallel_dims.tp_size = 2
    trainer.resolved_config = {"rollout": {"engine": "mock"}}
    monkeypatch.setattr(trainer_backend.platform, "get_world_size", lambda: 4)

    trainer._validate_runtime_topology()  # pylint: disable=protected-access


def test_runtime_topology_rejects_world_size_outside_resolved_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launcher count inconsistent with DP x TP fails before model construction."""
    trainer = _shared_topology_trainer()
    trainer.parallel_dims.tp_size = 2
    trainer.resolved_config = {"rollout": {"engine": "mock"}}
    monkeypatch.setattr(trainer_backend.platform, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match="resolved Trainer mesh size"):
        trainer._validate_runtime_topology()  # pylint: disable=protected-access


def test_internal_dp_runtime_rejects_extra_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared server must own exactly the devices represented by Trainer ranks."""
    trainer = _shared_topology_trainer()
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setattr(trainer_backend.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        trainer_backend.platform,
        "all_gather_object",
        lambda values, value: values.__setitem__(slice(None), [value, value]),
    )

    with pytest.raises(ValueError, match="full unique physical NPU set"):
        trainer._validate_runtime_topology()  # pylint: disable=protected-access


def test_internal_dp_runtime_rejects_inconsistent_device_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rank must publish the same physical mapping to the shared server."""
    trainer = _shared_topology_trainer()
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(trainer_backend.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        trainer_backend.platform,
        "all_gather_object",
        lambda values, _value: values.__setitem__(
            slice(None),
            [
                ("2", "127.0.0.1", 8100, "0,1"),
                ("2", "127.0.0.1", 8100, "2,3"),
            ],
        ),
    )

    with pytest.raises(ValueError, match="same full physical NPU set"):
        trainer._validate_runtime_topology()  # pylint: disable=protected-access


def test_internal_dp_runtime_reports_one_rank_invalid_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All ranks validate gathered metadata instead of diverging before a collective."""
    trainer = _shared_topology_trainer()
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(trainer_backend.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        trainer_backend.platform,
        "all_gather_object",
        lambda values, _value: values.__setitem__(
            slice(None),
            [
                ("2", "127.0.0.1", 8100, "0,1"),
                ("2", "127.0.0.1", 8100, "2,2"),
            ],
        ),
    )

    with pytest.raises(ValueError, match="rank=1"):
        trainer._validate_runtime_topology()  # pylint: disable=protected-access


def test_disjoint_runtime_rejects_inconsistent_shared_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every Trainer rank must connect to the same external endpoint."""
    trainer = _shared_topology_trainer("disjoint")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(trainer_backend.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        trainer_backend.platform,
        "all_gather_object",
        lambda values, _value: values.__setitem__(
            slice(None),
            [
                ("2", "127.0.0.1", 8100, "2,3"),
                ("2", "127.0.0.1", 8200, "2,3"),
            ],
        ),
    )

    with pytest.raises(ValueError, match="same endpoint"):
        trainer._validate_runtime_topology()  # pylint: disable=protected-access


def test_disjoint_runtime_rejects_inconsistent_full_rollout_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every Trainer rank must publish the complete external rollout device set."""
    trainer = _shared_topology_trainer("disjoint")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(trainer_backend.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        trainer_backend.platform,
        "all_gather_object",
        lambda values, _value: values.__setitem__(
            slice(None),
            [
                ("2", "127.0.0.1", 8100, "2,3"),
                ("2", "127.0.0.1", 8100, "4,5"),
            ],
        ),
    )

    with pytest.raises(ValueError, match="same full physical NPU set"):
        trainer._validate_runtime_topology()  # pylint: disable=protected-access


def test_response_mask_includes_first_eos_and_excludes_later_tokens() -> None:
    """Natural EOS remains trainable while all following positions are excluded."""
    runner = object.__new__(AgentRunner)
    runner.settings = SimpleNamespace(eos_token_ids=(2,), ignore_eos=False)
    response_ids = torch.tensor([[10, 2, 0, 0], [20, 21, 22, 0]])

    implicit = runner._response_mask(response_ids, None)  # pylint: disable=protected-access
    explicit = runner._response_mask(  # pylint: disable=protected-access
        response_ids,
        torch.tensor([[True, True, False, False], [True, True, False, False]]),
    )

    assert implicit.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert explicit.tolist() == [[True, True, False, False], [True, True, False, False]]


def test_rollout_setup_failure_closes_partially_built_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synchronized setup failure must close environments created on this rank."""
    runner = object.__new__(AgentRunner)
    runner.environment_name = "test"
    runner.num_samples = 2
    runner.max_observation_tokens = None

    def synchronize_error(error: Optional[Exception], operation: str) -> None:
        """Emulate a single-rank generation engine error boundary."""
        del operation
        if error is not None:
            raise error

    runner.engine = SimpleNamespace(policy_version=0, synchronize_error=synchronize_error)
    closed = []

    class Environment:
        """Record cleanup of the environment created before setup fails."""

        async def close(self) -> None:
            """Record one asynchronous close."""
            closed.append(True)

    calls = 0

    def build_environment(name: str, prompt: object) -> Environment:
        """Create one environment, then fail the next replica setup."""
        del name, prompt
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("environment setup failed")
        return Environment()

    monkeypatch.setattr(runner_backend.ENVIRONMENTS, "build", build_environment)

    with pytest.raises(RuntimeError, match="environment setup failed"):
        runner.rollout([SimpleNamespace(prompt_id="prompt-0")], policy_version=0)

    assert closed == [True]


def test_model_build_exposes_actor_model_to_checkpoint_runtime() -> None:
    """Checkpoint state points at the trainable model rather than a role wrapper."""
    trainer = object.__new__(SyncTrainer)
    trainer.algorithm = build_algorithm(
        {"name": "grpo", "loss_aggregation": "token-mean"}
    )
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
    actor_model = torch.nn.Linear(2, 2)
    reference_model = torch.nn.Linear(2, 2)
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    trainer._build_one_parallel_model = lambda frozen: (
        reference_model if frozen else actor_model
    )

    scheduler = SimpleNamespace(step=lambda: None)
    trainer._build_optimizer_for = lambda model: (
        torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler,
    )

    trainer._build_models_and_optimizers()

    assert isinstance(trainer.actor, Actor)
    assert trainer.model is trainer.actor.actor_model
    assert trainer.optimizer is trainer.actor.optimizer
    assert trainer.lr_scheduler is trainer.actor.lr_scheduler
    assert trainer.actor.actor_model is actor_model
    assert isinstance(trainer.reference_actor, Actor)
    assert trainer.reference_actor.actor_model is reference_model
    assert trainer.reference_actor.optimizer is None
    assert all(
        not parameter.requires_grad
        for parameter in trainer.reference_actor.parameters()
    )


def test_partial_distributed_setup_remains_cleanup_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mesh failure after process-group init must not leak distributed state."""
    trainer = object.__new__(SyncTrainer)
    trainer.runtime_config = SimpleNamespace(
        training=SimpleNamespace(backend="hccl")
    )
    trainer._runtime_started = False
    trainer._tracker = None
    destroyed = []
    monkeypatch.setattr(trainer_backend, "initialize_distributed", lambda _backend: None)
    monkeypatch.setattr(
        trainer_backend,
        "create_distributed_setup_from_config",
        lambda _config: (_ for _ in ()).throw(RuntimeError("mesh failed")),
    )
    monkeypatch.setattr(
        trainer_backend,
        "destroy_process_group",
        lambda: destroyed.append(True),
    )

    with pytest.raises(RuntimeError, match="mesh failed"):
        trainer._setup_runtime()  # pylint: disable=protected-access
    trainer._cleanup_distributed()  # pylint: disable=protected-access

    assert destroyed == [True]
    assert trainer._runtime_started is False  # pylint: disable=protected-access


def _rollout() -> ExperienceBatch:
    """Build one valid rollout batch for trainer orchestration tests."""
    return ExperienceBatch(
        trajectories=(),
        sequences=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.bool),
        action_mask=torch.tensor([[False, True]]),
        rewards=torch.ones(1),
        old_log_probs=torch.zeros((1, 1)),
        responses=("",),
        generation_seconds=0.0,
    )


def test_publish_policy_releases_training_memory_before_vllm_weight_wake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colocated refit must clear cached training allocations before weight wake."""
    calls: list[str] = []
    monkeypatch.setattr(trainer_backend, "hsdp_sync_stream", lambda: None)
    trainer = object.__new__(SyncTrainer)
    trainer.actor = SimpleNamespace(actor_model=object())
    trainer.model_registration = SimpleNamespace(name="test")
    trainer._reshard_model = lambda model: calls.append("reshard")
    trainer._release_training_state_for_rollout = lambda: calls.append("release")
    trainer.rollout_engine = SimpleNamespace(
        update_weights=lambda _snapshot: calls.append("update_weights"),
        prepare_for_rollout=lambda: calls.append("prepare_for_rollout"),
    )

    trainer._publish_policy(2, SimpleNamespace(optimizer_steps=2))

    assert calls == [
        "reshard",
        "release",
        "update_weights",
        "prepare_for_rollout",
    ]


def _trainer(algorithm_name: str, calls: list[str]) -> SyncTrainer:
    """Construct the smallest real train-loop shell around fake external roles."""
    trainer = object.__new__(SyncTrainer)
    trainer._consistency_profile = CONSISTENCY_PROFILE_OFF
    trainer.algorithm = build_algorithm(
        {"name": algorithm_name, "loss_aggregation": "token-mean"}
    )
    trainer.state = SimpleNamespace(global_step=0, max_steps=1, epoch=0)
    trainer.checkpoints = SimpleNamespace(
        validate_resume=lambda: None,
        begin=lambda _state: None,
        will_save=lambda _step: False,
    )
    trainer._run_rank_synchronized = lambda _name, callback: callback()
    trainer._release_training_state_for_rollout = lambda: None
    trainer._cleanup_distributed = lambda: None
    trainer.train_dataloader = [object()]
    trainer._next_batch = lambda iterator: (
        {
            "sample_indices": torch.tensor([0]),
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.bool),
            "prompts": ["prompt"],
            "ground_truths": ["2"],
        },
        iterator,
    )
    trainer.device = torch.device("cpu")
    trainer._log_steps = 1
    trainer.evaluator = None
    trainer.rollout_manager = SimpleNamespace(generate=lambda **_kwargs: _rollout())
    trainer.model_registration = SimpleNamespace(name="test")

    class _Engine:
        """Record rollout residency and publication calls."""
        policy_version = 0

        @staticmethod
        def prepare_for_training() -> None:
            """Record transition to training residency."""
            calls.append("prepare_for_training")

        @staticmethod
        def update_weights(snapshot: Any) -> None:
            """Record publication of the trainable Actor model."""
            assert snapshot.payload is trainer.actor.actor_model
            calls.append("update_weights")
            raise _StopAfterRolePipeline

    class _Actor:
        """Record reference diagnostics and policy updates."""
        actor_model = torch.nn.Identity()

        @staticmethod
        def compute_log_probs(experience: Any) -> Any:
            """Return deterministic diagnostics for the fake Actor."""
            assert experience is not None
            calls.append("actor_log_probs")
            return torch.zeros((1, 1))

        @staticmethod
        def update(experience: Any) -> Any:
            """Record one fake policy update."""
            assert experience is not None
            calls.append("actor_update")
            return SimpleNamespace(optimizer_steps=1)

    class _ReferenceActor:
        """Record frozen-policy inference."""
        @staticmethod
        def compute_log_probs(experience: Any) -> Any:
            """Return deterministic reference log probabilities."""
            assert experience is not None
            calls.append("reference_log_probs")
            return torch.zeros((1, 1))

    class _Critic:
        """Record value inference and Critic updates."""
        @staticmethod
        def compute_values(experience: Any) -> Any:
            """Return deterministic values for the fake Critic."""
            assert experience is not None
            calls.append("critic_values")
            return torch.zeros((1, 1))

        @staticmethod
        def update(experience: Any) -> Any:
            """Record one fake Critic update."""
            assert experience is not None
            calls.append("critic_update")
            return SimpleNamespace()

    class _Preparer:
        """Record target preparation and required role outputs."""
        @staticmethod
        def prepare(
            experience: Any,
            *,
            reference_log_probs: Any = None,
            values: Any = None,
        ) -> Any:
            """Validate that algorithm-required role outputs are present."""
            assert experience is not None
            if trainer.algorithm.requirements.data.reference_log_probs:
                assert reference_log_probs is not None
            if trainer.algorithm.requirements.data.values:
                assert values is not None
            calls.append("prepare_experience")
            return experience

    trainer.rollout_engine = _Engine()
    trainer.actor = _Actor()
    trainer.model = trainer.actor
    trainer.reference_actor = _ReferenceActor()
    trainer.critic = _Critic() if algorithm_name == "ppo" else None
    trainer.experience_preparer = _Preparer()
    return trainer


@pytest.mark.parametrize(
    ("algorithm_name", "expected"),
    [
        (
            "grpo",
            [
                "prepare_for_training",
                "actor_log_probs",
                "reference_log_probs",
                "prepare_experience",
                "actor_update",
                "update_weights",
            ],
        ),
        (
            "ppo",
            [
                "prepare_for_training",
                "actor_log_probs",
                "reference_log_probs",
                "critic_values",
                "prepare_experience",
                "actor_update",
                "critic_update",
                "update_weights",
            ],
        ),
    ],
)
def test_train_explicitly_orchestrates_required_role_outputs(
    algorithm_name: str,
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trainer calls only required roles and preserves RL2 ordering."""
    calls: list[str] = []
    trainer = _trainer(algorithm_name, calls)
    monkeypatch.setattr(trainer_backend.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_backend.platform, "get_world_size", lambda: 1)

    with pytest.raises(_StopAfterRolePipeline):
        trainer.train()

    assert calls == expected


def test_train_rejects_missing_required_reference_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing reference role must not silently fall back to the train actor."""
    calls: list[str] = []
    trainer = _trainer("grpo", calls)
    trainer.reference_actor = None
    monkeypatch.setattr(trainer_backend.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_backend.platform, "get_world_size", lambda: 1)

    with pytest.raises(RuntimeError, match="requires a reference model"):
        trainer.train()

    assert calls == ["prepare_for_training", "actor_log_probs"]


def test_pre_update_consistency_accepts_identical_worker_owned_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict gate reports exact FP32 rollout values before optimization."""
    rollout = _rollout()
    rollout = ExperienceBatch(
        **{
            **rollout.__dict__,
            "worker_policy_version": 0,
            "worker_policy_fingerprint": "digest-v0",
        }
    )
    monkeypatch.setattr(trainer_backend.platform, "get_rank", lambda: 0)

    metrics = validate_pre_update_consistency(
        rollout,
        rollout.old_log_probs.clone(),
        expected_policy_version=0,
        expected_policy_fingerprint="digest-v0",
        group=None,
        group_size=1,
    )

    assert metrics["training/pre_update_exact_tokens"] == 1.0
    assert metrics["training/pre_update_mismatch_count"] == 0.0
    assert metrics["training/pre_update_max_abs_diff"] == 0.0


def test_pre_update_consistency_rejects_equal_values_with_different_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signed zero proves the gate compares bit patterns rather than float equality."""
    rollout = _rollout()
    rollout = ExperienceBatch(
        **{
            **rollout.__dict__,
            "old_log_probs": torch.tensor([[0.0]], dtype=torch.float32),
            "worker_policy_version": 0,
            "worker_policy_fingerprint": "digest-v0",
        }
    )
    monkeypatch.setattr(trainer_backend.platform, "get_rank", lambda: 0)

    with pytest.raises(RuntimeError, match="actor_bits.*rollout_bits"):
        validate_pre_update_consistency(
            rollout,
            torch.tensor([[-0.0]], dtype=torch.float32),
            expected_policy_version=0,
            expected_policy_fingerprint="digest-v0",
            group=None,
            group_size=1,
        )


def test_post_update_negative_control_counts_exact_bit_changes() -> None:
    """The negative control counts changed response-token bits after optimization."""
    rollout = _rollout()
    metrics = measure_post_update_old_policy_mismatch(
        rollout,
        rollout.old_log_probs + 0.25,
        group=None,
        group_size=1,
    )

    assert metrics["training/post_update_old_policy_tokens"] == 1.0
    assert metrics["training/post_update_old_policy_mismatch_count"] == 1.0
    assert metrics["training/post_update_negative_control_valid"] == 1.0


def test_profiled_prepare_experience_gates_before_reference_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict mismatch stops the role pipeline before reference inference or update."""
    calls: list[str] = []
    trainer = _trainer("grpo", calls)
    trainer._consistency_profile = QWEN3_ASCEND_CONSISTENCY_V1
    trainer._dp_group_info = SimpleNamespace(group=None)
    trainer.parallel_dims = SimpleNamespace(dp_size=1)
    trainer.rollout_engine.policy_fingerprint = "digest-v0"
    rollout = _rollout()
    rollout = ExperienceBatch(
        **{
            **rollout.__dict__,
            "old_log_probs": torch.ones((1, 1)),
            "worker_policy_version": 0,
            "worker_policy_fingerprint": "digest-v0",
        }
    )
    monkeypatch.setattr(trainer_backend.platform, "get_rank", lambda: 0)

    with pytest.raises(RuntimeError, match="bit-exact gate failed"):
        trainer._prepare_experience(  # pylint: disable=protected-access
            rollout,
            collect_diagnostics=False,
            timings={},
        )

    assert calls == ["actor_log_probs"]


def test_profiled_prepare_experience_preflights_before_actor_forward() -> None:
    """A rank-local packed-input error must synchronize before any FSDP forward."""
    calls: list[str] = []
    trainer = _trainer("grpo", calls)
    trainer._consistency_profile = QWEN3_ASCEND_CONSISTENCY_V1
    trainer._dp_group_info = SimpleNamespace(group=None)
    trainer.parallel_dims = SimpleNamespace(dp_size=1)
    rollout = _rollout()
    rollout = ExperienceBatch(
        **{
            **rollout.__dict__,
            "attention_mask": torch.tensor([[False, True]]),
            "worker_policy_version": 0,
            "worker_policy_fingerprint": "digest-v0",
        }
    )

    with pytest.raises(RuntimeError, match="pre-update consistency forward preflight"):
        trainer._prepare_experience(  # pylint: disable=protected-access
            rollout,
            collect_diagnostics=False,
            timings={},
        )

    assert not calls


def test_optimizer_cpu_residency_accepts_chained_optimizer() -> None:
    """Colocated residency validation inspects states owned by sub-optimizers."""
    optimizer = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(state={}), SimpleNamespace(state={})]
    )

    SyncTrainer._validate_optimizer_cpu_residency(optimizer, "actor")
