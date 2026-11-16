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
"""CPU unit tests for distributed Hyper-RL monitoring metrics."""
# Local test doubles are not public APIs; the suite intentionally uses Torch CPU tensors.
# pylint: disable=missing-public-docstring

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import rl.utils.monitoring.metrics as metrics_module
from rl.data.contracts import ExperienceBatch, Trajectory
from rl.utils.monitoring.metrics import (
    ActorMetricAccumulator,
    ActorMicroBatchMetrics,
    CriticUpdateMetrics,
    build_training_metrics,
    enforce_learning_gate,
    summarize_rollout,
    summarize_training_diagnostics,
)


def test_actor_metrics_aggregate_by_global_valid_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Role sums use global action tokens while gradient norms use optimizer steps."""
    accumulator = ActorMetricAccumulator.create(
        torch.tensor(0.0),
        dp_group_info=SimpleNamespace(group="dp"),
        dp_size=2,
    )
    accumulator.add_micro_batch(
        ActorMicroBatchMetrics(
            total_loss_sum=torch.tensor(2.0),
            policy_loss_sum=torch.tensor(1.0),
            kl_loss_sum=torch.tensor(0.5),
            old_policy_kl_sum=torch.tensor(0.25),
            log_ratio_abs_sum=torch.tensor(0.75),
            clipped_token_count=torch.tensor(1.0),
        )
    )
    accumulator.add_optimizer_step(global_tokens=4, gradient_norm=2.0)
    accumulator.add_optimizer_step(global_tokens=4, gradient_norm=4.0)

    def all_reduce(tensor: torch.Tensor, group: Any) -> None:
        assert group == "dp"
        tensor.mul_(2)

    monkeypatch.setattr(metrics_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(metrics_module, "_system_memory_metrics", lambda: {})
    actor_update = accumulator.finalize(learning_rate=0.01)
    critic_update = CriticUpdateMetrics(0.25, 1.5, 0.02, 6, 2)
    policy = SimpleNamespace(
        policy_version=3,
        weight_sync_configured_strategy="full_gather",
        weight_sync_last_strategy="full_gather",
    )

    metrics = build_training_metrics(
        step=3,
        actor_update=actor_update,
        rollout_metrics={"reward/mean": 0.5},
        critic_update=critic_update,
        policy=policy,
    )

    assert actor_update.total_loss == 0.5
    assert actor_update.policy_loss == 0.25
    assert actor_update.gradient_norm == 3.0
    assert actor_update.valid_tokens == 8
    assert actor_update.optimizer_steps == 2
    assert metrics["critic/value_loss"] == 0.25
    assert metrics["policy/version"] == 3.0
    assert metrics["weight_sync/configured_full_gather"] == 1.0
    assert metrics["weight_sync/last_full_gather"] == 1.0
    dynamic_metrics = build_training_metrics(
        step=3, actor_update=replace(actor_update, micro_batches=3, input_tokens=18, padded_input_tokens=24),
        rollout_metrics={},
    )
    assert dynamic_metrics["train/micro_batches_per_rank"] == 3.0
    assert dynamic_metrics["train/input_tokens"] == 18.0
    assert dynamic_metrics["train/padded_input_tokens"] == 24.0
    assert dynamic_metrics["train/padding_fraction"] == 0.25
    assert "train/padding_fraction" not in metrics


def _trajectory(
    trajectory_id: str,
    group_id: str,
    action_mask: torch.Tensor,
    reward: float,
    truncated: bool,
) -> Trajectory:
    """Build one compact trajectory for rollout metric aggregation."""
    tokens = torch.arange(action_mask.numel())
    return Trajectory(
        trajectory_id=trajectory_id,
        prompt_id=trajectory_id,
        group_id=group_id,
        policy_version=0,
        turns=(),
        token_ids=tokens,
        attention_mask=torch.ones_like(tokens, dtype=torch.bool),
        action_mask=action_mask,
        rollout_log_probs=torch.zeros(action_mask.numel() - 1),
        reward=reward,
        reward_components={"task": reward},
        done=not truncated,
        truncated=truncated,
        terminal_reason="completed" if not truncated else "max_turns",
        metadata={"extracted_answer": str(int(reward))},
    )





def test_learning_gate_accepts_valid_learning_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete set of positive learning signals passes one synchronized gate."""
    operations: list[str] = []
    metrics = {
        "train/gradient_norm": 1.5,
        "reward/min": 0.0,
        "reward/max": 1.0,
        "policy/version": 4.0,
    }
    config = {
        "enabled": True,
        "min_gradient_norm": 0.1,
        "require_mixed_rewards": True,
    }
    monkeypatch.setattr(metrics_module.dist, "get_rank", lambda: 0)

    def run_synchronized(operation: str, callback: Any) -> None:
        operations.append(operation)
        callback()

    enforce_learning_gate(
        metrics,
        step=4,
        config=config,
        run_synchronized=run_synchronized,
    )

    assert operations == ["learning gate step 4"]
    assert metrics["policy/version"] == 4.0
