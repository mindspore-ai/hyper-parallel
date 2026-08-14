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
"""CPU tests for model-free experience target preparation."""

import pytest
import torch

from rl.algorithm import build_algorithm
from rl.dataset.contracts import ExperienceBatch
from rl.dataset import ExperiencePreparer


def _algorithm(name: str):
    """Build a registered algorithm with the minimal valid config."""
    return build_algorithm({"name": name, "loss_aggregation": "token-mean"})


def _rollout(
    rewards: list[float] | None = None,
    old_log_probs: torch.Tensor | None = None,
    include_old_log_probs: bool = True,
) -> ExperienceBatch:
    """Return a minimal rollout batch with two trainable response tokens."""
    sequences = torch.tensor(
        [
            [1, 2, 7],
            [1, 3, 7],
            [1, 4, 7],
            [1, 5, 7],
        ],
        dtype=torch.long,
    )
    action_mask = torch.tensor(
        [
            [False, True, True],
            [False, True, True],
            [False, True, True],
            [False, True, True],
        ],
        dtype=torch.bool,
    )
    if rewards is None:
        rewards = [0.0, 1.0, 0.0, 1.0]
    if not include_old_log_probs:
        old_log_probs = None
    elif old_log_probs is None:
        old_log_probs = torch.full((4, 2), -0.2)
    return ExperienceBatch(
        trajectories=(),
        sequences=sequences,
        attention_mask=torch.ones_like(sequences, dtype=torch.bool),
        action_mask=action_mask,
        rewards=torch.tensor(rewards, dtype=torch.float32),
        old_log_probs=old_log_probs,
        responses=("", "", "", ""),
        generation_seconds=0.0,
    )


def test_grpo_preparer_adds_reference_log_probs_and_advantages_only() -> None:
    """GRPO combines role output with advantages without invoking a model."""
    reference_log_probs = torch.full(
        (4, 2), -0.5, requires_grad=True
    )

    prepared = ExperiencePreparer(_algorithm("grpo")).prepare(
        _rollout(),
        reference_log_probs=reference_log_probs,
    )

    assert prepared.reference_log_probs.shape == (4, 2)
    assert prepared.advantages.shape == (4, 2)
    assert prepared.values is None
    assert prepared.returns is None
    assert prepared.reference_log_probs.requires_grad is False
    assert prepared.advantages.requires_grad is False


def test_ppo_preparer_requires_critic() -> None:
    """PPO target construction rejects missing critic values."""
    with pytest.raises(ValueError, match="critic values"):
        ExperiencePreparer(_algorithm("ppo")).prepare(
            _rollout(),
            reference_log_probs=torch.full((4, 2), -0.5),
        )


def test_ppo_preparer_adds_reference_values_returns_and_advantages() -> None:
    """PPO combines reference and critic outputs into actor and critic targets."""
    reference_log_probs = torch.full((4, 2), -0.5, requires_grad=True)
    values = torch.full((4, 2), 0.25, requires_grad=True)

    prepared = ExperiencePreparer(_algorithm("ppo")).prepare(
        _rollout(rewards=[1.0, 3.0, 0.0, 2.0]),
        reference_log_probs=reference_log_probs,
        values=values,
    )
    assert prepared.reference_log_probs.shape == (4, 2)
    assert prepared.values.shape == (4, 2)
    assert prepared.returns.shape == (4, 2)
    assert prepared.advantages.shape == (4, 2)
    assert prepared.reference_log_probs.requires_grad is False
    assert prepared.values.requires_grad is False
    assert prepared.advantages.requires_grad is False
    assert prepared.returns.requires_grad is False


def test_preparer_rejects_missing_reference_log_probs() -> None:
    """A reference-dependent algorithm rejects incomplete role outputs."""
    with pytest.raises(ValueError, match="reference log-probabilities"):
        ExperiencePreparer(_algorithm("grpo")).prepare(_rollout())


def test_preparer_rejects_missing_rollout_log_probs() -> None:
    """Algorithms still receive a clear error when rollout old log-probs are absent."""
    rollout = _rollout(include_old_log_probs=False)
    preparer = ExperiencePreparer(_algorithm("grpo"))

    with pytest.raises(ValueError, match="rollout log-probabilities"):
        preparer.prepare(
            rollout,
            reference_log_probs=torch.full((4, 2), -0.5),
        )
