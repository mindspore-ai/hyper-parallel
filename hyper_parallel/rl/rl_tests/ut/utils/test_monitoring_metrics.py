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
"""CPU unit tests for distributed HyperParallel-RL monitoring metrics."""
# Local test doubles are not public APIs; the suite intentionally uses Torch CPU tensors.
# pylint: disable=forbidden-backend-import,missing-public-docstring

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import rl.utils.monitoring.metrics as metrics_module
from rl.consistency import measure_post_update_old_policy_mismatch
from rl.dataset.contracts import ExperienceBatch, Trajectory
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
        dp_group_info="dp",
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

    monkeypatch.setattr(metrics_module.platform, "all_reduce", all_reduce)
    monkeypatch.setattr(metrics_module, "_system_memory_metrics", lambda: {})
    actor_update = accumulator.finalize(learning_rate=0.01)
    critic_update = CriticUpdateMetrics(0.25, 1.5, 0.02, 6, 2)
    policy = SimpleNamespace(
        policy_version=3,
        policy_fingerprint="digest-v3",
        policy_fingerprint_changed=True,
        weight_sync_configured_strategy="direct_reshard",
        weight_sync_last_strategy="direct_reshard",
        weight_sync_fallback_count=0,
        weight_sync_direct_success_count=3,
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
    assert metrics["weight_sync/configured_direct_reshard"] == 1.0
    assert metrics["weight_sync/last_direct_reshard"] == 1.0


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


def test_rollout_and_training_diagnostics_use_global_masked_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global rollout and probability diagnostics exclude every non-action token."""
    action_mask = torch.tensor(
        [[False, True, True, False], [False, True, False, False]]
    )
    attention_mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    rollout = ExperienceBatch(
        trajectories=(
            _trajectory("0", "group", action_mask[0], 1.0, False),
            _trajectory("1", "group", action_mask[1], 0.0, True),
        ),
        sequences=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]]),
        attention_mask=attention_mask,
        action_mask=action_mask,
        rewards=torch.tensor([1.0, 0.0]),
        old_log_probs=torch.tensor([[0.0, 0.0, 99.0], [0.0, 99.0, 99.0]]),
        responses=("one", "zero"),
        generation_seconds=2.0,
        advantages=torch.tensor([[1.0, 2.0, 99.0], [3.0, 99.0, 99.0]]),
        values=torch.tensor([[0.5, 1.5, 99.0], [2.5, 99.0, 99.0]]),
        returns=torch.tensor([[1.0, 2.0, 99.0], [3.0, 99.0, 99.0]]),
    )
    actor_log_probs = torch.tensor(
        [[0.1, 0.2, -999.0], [0.3, -999.0, -999.0]]
    )
    batch = {
        "sample_indices": [0, 1],
        "prompts": ["p0", "p1"],
        "ground_truths": ["1", "0"],
    }

    def all_gather(output: list[Any], value: dict[str, Any]) -> None:
        output[0] = value
        if "reward_sum" not in value:
            output[1] = value.copy()
            return
        second = value.copy()
        second.update(
            {
                "reward_sum": 2.0,
                "reward_square_sum": 2.0,
                "reward_count": 2,
                "reward_min": 1.0,
                "reward_max": 1.0,
                "zero_std_groups": 1,
                "length_sum": 4,
                "length_square_sum": 8,
                "length_min": 2,
                "length_max": 2,
                "truncated_count": 0,
                "generated_tokens": 4,
                "generation_seconds": 4.0,
                "samples": [{**value["samples"][0], "rank": 1}],
            }
        )
        output[1] = second

    monkeypatch.setattr(metrics_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(metrics_module.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(metrics_module.platform, "all_gather_object", all_gather)
    rollout_metrics, samples = summarize_rollout(rollout, batch, step=1, sample_limit=2)
    diagnostic_metrics = summarize_training_diagnostics(rollout, actor_log_probs)
    mismatch_metrics = measure_post_update_old_policy_mismatch(
        rollout,
        actor_log_probs,
        group=None,
        group_size=1,
    )

    assert rollout_metrics["reward/mean"] == 0.75
    assert rollout_metrics["reward/std"] == pytest.approx(math.sqrt(0.1875))
    assert rollout_metrics["rollout/response_length_mean"] == 1.75
    assert rollout_metrics["rollout/truncated_ratio"] == 0.25
    assert rollout_metrics["rollout/generated_tokens"] == 7.0
    assert rollout_metrics["rollout/generation_seconds"] == 4.0
    assert rollout_metrics["rollout/tokens_per_second"] == 1.75
    assert [sample["rank"] for sample in samples] == [0, 1]
    assert diagnostic_metrics["training/log_probs_valid_tokens"] == 6.0
    assert diagnostic_metrics["training/total_tokens"] == 14.0
    assert diagnostic_metrics["training/rollout_log_probs_diff_mean"] == pytest.approx(0.2)
    assert diagnostic_metrics["train/advantage_mean"] == 2.0
    assert mismatch_metrics["training/post_update_old_policy_mismatch_count"] == 3.0


def test_learning_gate_accepts_valid_learning_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete set of positive learning signals passes one synchronized gate."""
    operations: list[str] = []
    metrics = {
        "train/gradient_norm": 1.5,
        "reward/min": 0.0,
        "reward/max": 1.0,
        "policy/fingerprint_changed": 1.0,
        "policy/version": 4.0,
    }
    config = {
        "enabled": True,
        "min_gradient_norm": 0.1,
        "require_mixed_rewards": True,
        "require_fingerprint_change": True,
    }
    monkeypatch.setattr(metrics_module.platform, "get_rank", lambda: 0)

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
