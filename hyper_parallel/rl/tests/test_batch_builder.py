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
"""CPU tests for requirements-driven batch preparation."""

import torch

from rl.algorithm import build_algorithm
from rl.contracts import ExperienceBatch
from rl.dataset import ExperienceBuilder


class _LogProbModel(torch.nn.Module):
    """Return deterministic frozen-policy log-probabilities."""

    def sequence_log_probs(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a constant log-probability for each next-token position."""
        del attention_mask
        return torch.full(
            (sequences.shape[0], sequences.shape[1] - 1), -0.5
        )


def _rollout() -> ExperienceBatch:
    sequences = torch.tensor([[1, 2], [1, 3]])
    action_mask = torch.tensor([[False, True], [False, True]])
    return ExperienceBatch(
        trajectories=(),
        sequences=sequences,
        attention_mask=torch.ones_like(sequences, dtype=torch.bool),
        action_mask=action_mask,
        rewards=torch.tensor([0.0, 1.0]),
        old_log_probs=torch.zeros((2, 1)),
        responses=("", ""),
        generation_seconds=0.0,
    )


def test_grpo_builder_populates_recipe_fields_without_values() -> None:
    algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    prepared = ExperienceBuilder(algorithm, reference=_LogProbModel()).build(_rollout())
    assert prepared.advantages.shape == (2, 1)
    assert prepared.reference_log_probs.tolist() == [[-0.5], [-0.5]]
    assert prepared.values is None
    assert prepared.returns is None


def test_builder_requires_declared_critic_capability() -> None:
    algorithm = build_algorithm({"name": "ppo", "loss_aggregation": "token-mean"})
    try:
        ExperienceBuilder(algorithm, reference=_LogProbModel())
    except ValueError as error:
        assert "Critic" in str(error)
    else:
        raise AssertionError("PPO must not build without a Critic")
