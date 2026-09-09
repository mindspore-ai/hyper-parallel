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
"""CPU unit tests for Hyper-RL advantage estimators and rule rewards."""
# Hyper-RL CPU tests intentionally exercise the verified Torch runtime.
# pylint: disable=forbidden-backend-import

import math

import pytest
import torch

from rl.algorithm import compute_rule_reward, extract_answer
from rl.algorithm.advantage import GAEAdvantageEstimator, GroupRelativeAdvantageEstimator


def test_grpo_advantages_match_manual_group_normalization() -> None:
    """Group rewards are normalized independently and expanded only to actions."""
    rewards = torch.tensor([1.0, 3.0, 2.0, 2.0])
    action_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [False, True, True],
            [True, True, True],
        ]
    )
    epsilon = 1.0e-6

    output = GroupRelativeAdvantageEstimator(epsilon=epsilon).estimate(
        rewards,
        action_mask,
        group_ids=("prompt-a", "prompt-a", "prompt-b", "prompt-b"),
    )

    normalized = 1.0 / (math.sqrt(2.0) + epsilon)
    expected = torch.tensor(
        [
            [-normalized, -normalized, 0.0],
            [normalized, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(output.advantages, expected)
    assert output.returns is None


def test_gae_and_returns_match_manual_backward_recursion() -> None:
    """GAE follows action positions and normalizes advantages after returns."""
    rewards = torch.tensor([2.0])
    action_mask = torch.tensor([[True, False, True]])
    values = torch.tensor([[0.5, 99.0, 1.0]])
    epsilon = 1.0e-6

    output = GAEAdvantageEstimator(
        gamma=0.9,
        gae_lambda=0.8,
        normalize=True,
        epsilon=epsilon,
    ).estimate(rewards, action_mask, values=values)

    raw_advantages = torch.tensor([1.12, 1.0])
    centered = raw_advantages - raw_advantages.mean()
    expected_selected = centered / (raw_advantages.std(unbiased=False) + epsilon)
    expected_advantages = torch.tensor(
        [[expected_selected[0], 0.0, expected_selected[1]]]
    )
    expected_returns = torch.tensor([[1.62, 0.0, 2.0]])
    torch.testing.assert_close(output.advantages, expected_advantages)
    torch.testing.assert_close(output.returns, expected_returns)


@pytest.mark.parametrize(
    ("solution", "ground_truth", "expected_answer"),
    [
        ("work\n#### 42", "42", "42"),
        ("work\n#### -12", "-12", "-12"),
        ("first #### 1\nfinal #### 3.14", "3.14", "3.14"),
        ("work\n#### 1,024", "1,024", "1024"),
    ],
)
def test_rule_reward_extracts_supported_numeric_answers(
    solution: str,
    ground_truth: str,
    expected_answer: str,
) -> None:
    """Supported numeric formats extract and score in scalar and list form."""
    assert extract_answer(solution) == expected_answer
    assert compute_rule_reward(solution, ground_truth) == 1.0
    assert compute_rule_reward([solution, "#### 0"], ground_truth) == [1.0, 0.0]
