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
"""CPU tests for the complete PPO Recipe and its internal GAE composition."""

import torch

from rl.algorithm import PPOAlgorithm, build_algorithm


def test_ppo_is_a_complete_registered_recipe_with_critic_requirements() -> None:
    algorithm = build_algorithm({"name": "ppo", "loss_aggregation": "token-mean"})
    assert isinstance(algorithm, PPOAlgorithm)
    assert algorithm.requirements.roles.critic is True
    assert algorithm.requirements.data.values is True
    assert algorithm.requirements.data.returns is True


def test_ppo_gae_skips_non_action_observation_positions() -> None:
    """Exclude non-action observation tokens from PPO GAE targets."""
    algorithm = build_algorithm(
        {
            "name": "ppo",
            "loss_aggregation": "token-mean",
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "normalize_advantages": False,
        }
    )
    mask = torch.tensor([[True, False, True]])
    values = torch.tensor([[0.2, 99.0, 0.4]])
    targets = algorithm.build_targets(
        rewards=torch.tensor([1.0]),
        action_mask=mask,
        values=values,
    )
    assert torch.allclose(targets.advantages, torch.tensor([[0.8, 0.0, 0.6]]))
    assert torch.allclose(targets.returns, torch.tensor([[1.0, 0.0, 1.0]]))


def test_ppo_value_loss_uses_clipped_value_prediction() -> None:
    """Use the larger of unclipped and clipped PPO value errors."""
    algorithm = build_algorithm(
        {
            "name": "ppo",
            "loss_aggregation": "token-mean",
            "value_clip_ratio": 0.2,
        }
    )
    output = algorithm.compute_critic_loss(
        current_values=torch.tensor([[2.0]]),
        old_values=torch.tensor([[0.0]]),
        returns=torch.tensor([[1.0]]),
        action_mask=torch.tensor([[True]]),
    )
    # max((2-1)^2, (0.2-1)^2) / 2
    assert torch.allclose(output.loss_sum, torch.tensor(0.5))


def test_advantage_normalization_does_not_change_critic_returns() -> None:
    """Normalize actor advantages without modifying Critic return targets."""
    algorithm = build_algorithm(
        {
            "name": "ppo",
            "loss_aggregation": "token-mean",
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "normalize_advantages": True,
        }
    )
    targets = algorithm.build_targets(
        rewards=torch.tensor([1.0, 3.0]),
        action_mask=torch.tensor([[True], [True]]),
        values=torch.tensor([[0.25], [0.5]]),
    )
    assert torch.allclose(targets.returns, torch.tensor([[1.0], [3.0]]))
    assert torch.allclose(targets.advantages.mean(), torch.tensor(0.0))
