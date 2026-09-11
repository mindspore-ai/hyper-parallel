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
"""CPU unit tests for HyperParallel-RL objectives, KL, and value loss."""
# HyperParallel-RL CPU tests intentionally exercise the verified Torch runtime.
# pylint: disable=forbidden-backend-import

import math

import torch

from rl.algorithm import build_algorithm
from rl.algorithm.loss import ClippedPolicyObjective


def test_clipped_policy_objective_matches_manual_ratio_clipping() -> None:
    """The clipped and dual-clipped token objectives match a hand calculation."""
    ratios = torch.tensor([1.0, 4.0, 1.5, 0.5])
    old_log_probs = torch.zeros(4)
    current_log_probs = ratios.log()
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])

    output = ClippedPolicyObjective(
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        dual_clip=3.0,
    ).compute(current_log_probs, old_log_probs, advantages)

    torch.testing.assert_close(output.loss, torch.tensor([-1.0, 3.0, -1.2, 0.8]))
    torch.testing.assert_close(output.clipped, torch.tensor([0.0, 1.0, 1.0, 1.0]))


def test_actor_loss_and_kl_match_manual_masked_sums() -> None:
    """Actor loss exposes policy, KL, clipping, and valid-token sums."""
    algorithm = build_algorithm(
        {
            "name": "grpo",
            "loss_aggregation": "token-mean",
            "kl_coef": 0.5,
        }
    )
    current_log_probs = torch.zeros((1, 3))
    old_log_probs = torch.zeros((1, 3))
    reference_log_probs = torch.tensor([[0.0, math.log(2.0), math.log(2.0)]])
    advantages = torch.tensor([[1.0, 2.0, 3.0]])
    action_mask = torch.tensor([[True, False, True]])

    output = algorithm.compute_actor_loss(
        current_log_probs,
        old_log_probs,
        reference_log_probs,
        advantages,
        action_mask,
    )

    raw_kl = 1.0 - math.log(2.0)
    torch.testing.assert_close(output.policy_loss_sum, torch.tensor(-4.0))
    torch.testing.assert_close(output.regularization_loss_sum, torch.tensor(raw_kl))
    torch.testing.assert_close(output.total_loss_sum, torch.tensor(-4.0 + 0.5 * raw_kl))
    torch.testing.assert_close(output.old_policy_kl_sum, torch.tensor(0.0))
    torch.testing.assert_close(output.valid_token_count, torch.tensor(2.0))
    torch.testing.assert_close(output.clipped_token_count, torch.tensor(0.0))


def test_ppo_value_loss_matches_manual_clipped_regression() -> None:
    """PPO value clipping uses the larger current or clipped squared error."""
    algorithm = build_algorithm(
        {
            "name": "ppo",
            "loss_aggregation": "token-mean",
            "value_clip_ratio": 0.2,
        }
    )
    current_values = torch.tensor([[1.5, 0.0, -1.0]])
    old_values = torch.tensor([[1.0, 0.0, -0.5]])
    returns = torch.tensor([[2.0, 3.0, -0.25]])
    action_mask = torch.tensor([[True, False, True]])

    output = algorithm.compute_critic_loss(
        current_values,
        old_values,
        returns,
        action_mask,
    )

    expected_first = 0.5 * max((1.5 - 2.0) ** 2, (1.2 - 2.0) ** 2)
    expected_last = 0.5 * max((-1.0 + 0.25) ** 2, (-0.7 + 0.25) ** 2)
    torch.testing.assert_close(output.loss_sum, torch.tensor(expected_first + expected_last))
    torch.testing.assert_close(output.valid_token_count, torch.tensor(2.0))
