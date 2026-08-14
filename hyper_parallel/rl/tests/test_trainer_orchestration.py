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

from types import SimpleNamespace

import pytest
import torch

from rl.algorithm import build_algorithm
from rl.dataset.contracts import ExperienceBatch
import rl.trainer as trainer_backend
from rl.roles import Actor
from rl.trainer import SyncTrainer


class _StopAfterRolePipeline(RuntimeError):
    """Stop the trainer after the role pipeline reaches weight publication."""


def test_model_build_exposes_actor_through_base_trainer_aliases() -> None:
    """Checkpoint and rollout aliases share the trainable Actor runtime."""
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
    trainer._build_one_parallel_model = (
        lambda frozen: reference_model if frozen else actor_model
    )

    def build_optimizer() -> None:
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)

    scheduler = SimpleNamespace(step=lambda: None)
    trainer._build_optimizer = build_optimizer
    trainer._build_lr_scheduler = lambda: setattr(trainer, "lr_scheduler", scheduler)
    trainer._build_training_context = lambda: None

    trainer._build_models_and_optimizers()

    assert isinstance(trainer.actor, Actor)
    assert trainer.model is trainer.actor
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


def _trainer(algorithm_name: str, calls: list[str]) -> SyncTrainer:
    """Construct the smallest real train-loop shell around fake external roles."""
    trainer = object.__new__(SyncTrainer)
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
            calls.append("prepare_for_training")

        @staticmethod
        def update_weights(snapshot) -> None:
            assert snapshot.payload is trainer.actor.actor_model
            calls.append("update_weights")
            raise _StopAfterRolePipeline

    class _Actor:
        """Record reference diagnostics and policy updates."""
        actor_model = object()

        @staticmethod
        def compute_log_probs(experience):
            assert experience is not None
            calls.append("actor_log_probs")
            return torch.zeros((1, 1))

        @staticmethod
        def update(experience):
            assert experience is not None
            calls.append("actor_update")
            return SimpleNamespace(optimizer_steps=1)

    class _ReferenceActor:
        """Record frozen-policy inference."""
        @staticmethod
        def compute_log_probs(experience):
            assert experience is not None
            calls.append("reference_log_probs")
            return torch.zeros((1, 1))

    class _Critic:
        """Record value inference and Critic updates."""
        @staticmethod
        def compute_values(experience):
            assert experience is not None
            calls.append("critic_values")
            return torch.zeros((1, 1))

        @staticmethod
        def update(experience):
            assert experience is not None
            calls.append("critic_update")
            return SimpleNamespace()

    class _Preparer:
        """Record target preparation and required role outputs."""
        @staticmethod
        def prepare(experience, *, reference_log_probs=None, values=None):
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
