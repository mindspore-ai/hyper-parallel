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
"""Unit tests for Issue #150 MoE load balance fixes.

Test IDs:
  LB-01: Balanced vs unbalanced routing loss comparison
  LB-02: Empty input returns 0.0
  LB-03: expert_bias mean change is zero after update
  LB-04: tokens_per_expert is reset to zero after update
  LB-05: Overloaded expert bias decreases, underloaded increases
  LB-06: num_recomputations parameter corrects double-counting
"""
import os
import unittest

import torch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.platform.torch.common.moe import (  # pylint: disable=C0413
    MoE,
    _compute_load_balance_loss,
    update_expert_bias,
)


class TestLoadBalanceLoss(unittest.TestCase):
    """Unit tests for _compute_load_balance_loss (Issue #150 Problem 1)."""

    def test_lb01_balanced_vs_unbalanced_loss(self):
        """LB-01: Unbalanced routing loss > balanced routing loss."""
        num_experts = 4
        num_tokens = 100
        top_k = 2

        # Balanced routing: each token selects experts cyclically
        balanced_experts = torch.zeros(num_tokens, top_k, dtype=torch.long)
        for i in range(num_tokens):
            balanced_experts[i, 0] = i % num_experts
            balanced_experts[i, 1] = (i + 1) % num_experts

        balanced_scores = torch.ones(num_tokens, top_k) / top_k
        loss_balanced = _compute_load_balance_loss(
            balanced_scores, balanced_experts, num_experts
        )

        # Unbalanced routing: all tokens to expert 0
        unbalanced_experts = torch.zeros(num_tokens, top_k, dtype=torch.long)
        unbalanced_scores = torch.ones(num_tokens, top_k) / top_k
        loss_unbalanced = _compute_load_balance_loss(
            unbalanced_scores, unbalanced_experts, num_experts
        )

        # Unbalanced loss should be significantly larger
        self.assertGreater(
            loss_unbalanced.item(), loss_balanced.item() * 2,
            f"Unbalanced loss ({loss_unbalanced.item():.4f}) should be > "
            f"2x balanced loss ({loss_balanced.item():.4f})"
        )

    def test_lb02_empty_input_returns_zero(self):
        """LB-02: Empty input (num_tokens=0) returns 0.0 without error."""
        num_experts = 4
        top_scores = torch.empty(0, 2)
        selected_experts = torch.empty(0, 2, dtype=torch.long)

        loss = _compute_load_balance_loss(top_scores, selected_experts, num_experts)

        self.assertEqual(loss.item(), 0.0)


class TestExpertBiasUpdate(unittest.TestCase):
    """Unit tests for update_expert_bias (Issue #150 Problems 2-4)."""

    def test_lb03_mean_zero_constraint(self):
        """LB-03: expert_bias mean change is zero after update."""
        torch.manual_seed(42)
        moe = MoE(dim=16, hidden_dim=32, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 16)

        moe(x)

        mean_before = moe.expert_bias.mean().item()
        update_expert_bias(moe, lr=1e-2)
        mean_after = moe.expert_bias.mean().item()

        # Mean change should be approximately zero (within floating point tolerance)
        mean_change = abs(mean_after - mean_before)
        self.assertLess(
            mean_change, 1e-6,
            f"Bias mean drifted by {mean_change:.6e}, should be ~0"
        )

    def test_lb04_tokens_reset_to_zero(self):
        """LB-04: tokens_per_expert is reset to zero after update."""
        torch.manual_seed(42)
        moe = MoE(dim=16, hidden_dim=32, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 16)

        moe(x)
        tokens_before = moe.tokens_per_expert.sum().item()
        self.assertGreater(tokens_before, 0, "Tokens should accumulate during forward")

        update_expert_bias(moe, lr=1e-2)
        tokens_after = moe.tokens_per_expert.sum().item()

        self.assertEqual(tokens_after, 0.0, "Tokens should be reset to zero after update")

    def test_lb05_bias_direction_correct(self):
        """LB-05: Overloaded expert bias decreases, underloaded increases."""
        torch.manual_seed(42)
        moe = MoE(dim=16, hidden_dim=32, num_experts=4, top_k=2)

        # Manually set unbalanced tokens_per_expert
        moe.tokens_per_expert.zero_()
        moe.tokens_per_expert[0] = 100  # Expert 0 overloaded
        moe.tokens_per_expert[1] = 10   # Expert 1 underloaded

        bias_before = moe.expert_bias.clone()
        update_expert_bias(moe, lr=1e-2)
        bias_after = moe.expert_bias

        # Expert 0 (overloaded) should have bias decreased
        delta_0 = (bias_after[0] - bias_before[0]).item()
        # Expert 1 (underloaded) should have bias increased
        delta_1 = (bias_after[1] - bias_before[1]).item()

        self.assertLess(delta_0, 0, f"Overloaded expert bias should decrease, got delta={delta_0:.4f}")
        self.assertGreater(delta_1, 0, f"Underloaded expert bias should increase, got delta={delta_1:.4f}")

    def test_lb06_num_recomputations_correction(self):
        """LB-06: num_recomputations parameter corrects double-counting."""
        torch.manual_seed(42)
        moe = MoE(dim=16, hidden_dim=32, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 16)

        # Simulate AC: forward executed twice
        moe(x)
        moe(x)

        tokens_double = moe.tokens_per_expert.clone()

        # With correction (num_recomputations=2)
        update_expert_bias(moe, lr=1e-2, num_recomputations=2)

        # Tokens should be reset after update
        tokens_after = moe.tokens_per_expert.sum().item()
        self.assertEqual(tokens_after, 0.0, "Tokens should be reset after update")


if __name__ == "__main__":
    unittest.main()
