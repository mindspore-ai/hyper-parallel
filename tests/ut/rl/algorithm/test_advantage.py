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
"""Boundary tests for grouped GRPO rewards and masked PPO targets."""

import unittest

import torch

from rl.algorithm.advantage import GAEAdvantageEstimator, GroupRelativeAdvantageEstimator
from rl.algorithm.reward import compute_rule_reward, extract_answer, get_reward


class TestAdvantageBoundaries(unittest.TestCase):
    """Use explicit small tensors to expose grouping and masking regressions."""

    def test_grpo_normalizes_interleaved_groups_independently(self) -> None:
        """Response ordering must not mix rewards from different prompts."""
        result = GroupRelativeAdvantageEstimator().estimate(
            torch.tensor([1.0, 100.0, 3.0, 100.0]),
            torch.tensor([[1, 0], [1, 1], [1, 1], [0, 1]]),
            group_ids=("a", "b", "a", "b"),
        )
        scale = 1.0 / (2.0 ** 0.5 + 1e-6)
        torch.testing.assert_close(
            result.advantages, torch.tensor([[-scale, 0.0], [0.0, 0.0], [scale, scale], [0.0, 0.0]]),
        )
        self.assertIsNone(result.returns)

    def test_grpo_rejects_singleton_or_misaligned_groups(self) -> None:
        """Insufficient responses cannot silently become NaN training targets."""
        for groups, error in ((("a",), "one value per reward"), (("a", "b"), "at least two responses")):
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, error):
                GroupRelativeAdvantageEstimator().estimate(torch.ones(2), torch.ones(2, 2), group_ids=groups)

    def test_gae_rejects_missing_and_misaligned_value_inputs(self) -> None:
        """PPO values and bootstrap state must match the batch contract."""
        estimator = GAEAdvantageEstimator()
        for kwargs, error in (
            ({}, "requires old critic"),
            ({"values": torch.zeros(1, 3)}, "next-token"),
            ({"values": torch.zeros(1, 2), "bootstrap_values": torch.zeros(2)}, "one value per sequence"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, error):
                estimator.estimate(torch.ones(1), torch.ones(1, 2), **kwargs)

    def test_gae_empty_and_single_action_masks_stay_finite(self) -> None:
        """Empty rows stay zero and a singleton action avoids invalid whitening."""
        result = GAEAdvantageEstimator().estimate(
            torch.tensor([100.0, 3.0]), torch.tensor([[False, False], [False, True]]),
            values=torch.tensor([[10.0, 20.0], [99.0, 1.0]]),
        )
        torch.testing.assert_close(result.advantages, torch.tensor([[0.0, 0.0], [0.0, 2.0]]))
        torch.testing.assert_close(result.returns, torch.tensor([[0.0, 0.0], [0.0, 3.0]]))

    def test_strict_reward_uses_final_answer_and_bounded_tail(self) -> None:
        """Reward extraction rejects absent/tail-expired answers and preserves exact matching."""
        self.assertEqual(extract_answer("#### 1\ncorrection: #### -1,234.5"), "-1234.5")
        self.assertIsNone(extract_answer("#### 2" + "x" * 300))
        self.assertIsNone(extract_answer("The answer is 2"))
        self.assertEqual(compute_rule_reward(["#### 2", "#### 2.0", "missing"], "2"), [1.0, 0.0, 0.0])
        self.assertEqual(get_reward("gsm8k")("#### 1,234", "$1,234"), 1.0)
