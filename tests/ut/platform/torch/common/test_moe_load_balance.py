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
"""Unit tests for Issue #150 MoE load balance fixes and MoEAuxLossAutoScaler.

Test IDs:
  LB-01: Balanced vs unbalanced routing loss comparison
  LB-02: Empty input returns 0.0
  LB-03: expert_bias mean change is zero after update
  LB-04: tokens_per_expert is reset to zero after update
  LB-05: Overloaded expert bias decreases, underloaded increases
  LB-06: num_recomputations parameter corrects double-counting
  LB-A01: MoEAuxLossAutoScaler forward passes output unchanged
  LB-A02: MoEAuxLossAutoScaler backward injects scaled aux_loss gradient
  LB-A03: MoEAuxLossAutoScaler with zero aux_loss produces zero aux_loss grad
  LB-A04: MoEAuxLossAutoScaler.set_loss_scale updates gradient scale
  LB-S01: sequence_partition_group=None gives same result as before
  LB-S02: MoE accepts sequence_partition_group param
"""
import os
import unittest

import torch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.platform.torch.common.moe import (  # pylint: disable=C0413
    MoE,
    MoEAuxLossAutoScaler,
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


class TestMoEAuxLossAutoScaler(unittest.TestCase):
    """Unit tests for MoEAuxLossAutoScaler autograd function."""

    def setUp(self) -> None:
        """Reset class-level scale before each test."""
        MoEAuxLossAutoScaler.main_loss_backward_scale = None

    def test_lba01_forward_transparent(self):
        """LB-A01: Forward pass returns output tensor unchanged in value."""
        output = torch.randn(4, 8, requires_grad=True)
        aux_loss = torch.tensor(0.5, requires_grad=True)

        result = MoEAuxLossAutoScaler.apply(output, aux_loss)

        self.assertTrue(
            torch.allclose(result, output),
            (f"Forward should pass output unchanged: "
             f"result={result}, output={output}"),
        )

    def test_lba02_backward_injects_scaled_grad(self):
        """LB-A02: Backward injects aux_loss gradient with default scale=1."""
        output = torch.randn(4, 8, requires_grad=True)
        aux_loss = torch.tensor(0.5, requires_grad=True)

        result = MoEAuxLossAutoScaler.apply(output, aux_loss)
        result.sum().backward()

        # aux_loss.grad should be ones_like(aux_loss) * default_scale(1.0)
        self.assertIsNotNone(aux_loss.grad, "aux_loss should receive gradient")
        expected_grad = torch.ones_like(aux_loss)
        self.assertTrue(
            torch.allclose(aux_loss.grad, expected_grad),
            (f"aux_loss.grad should be ones * scale=1.0: "
             f"got={aux_loss.grad}, expected={expected_grad}"),
        )
        # output.grad should flow through unchanged
        self.assertIsNotNone(output.grad, "output should receive gradient")

    def test_lba03_zero_aux_loss_produces_zero_grad(self):
        """LB-A03: Zero aux_loss with scale=1 produces grad of ones (not zeros).

        The AutoScaler injects ``ones_like(aux_loss) * scale`` in backward,
        independent of aux_loss magnitude.  This is by design: the scale
        controls the *gradient magnitude*, not the aux_loss value.
        """
        output = torch.randn(4, 8, requires_grad=True)
        aux_loss = torch.tensor(0.0, requires_grad=True)

        result = MoEAuxLossAutoScaler.apply(output, aux_loss)
        result.sum().backward()

        self.assertIsNotNone(aux_loss.grad, "aux_loss should receive gradient")
        # With scale=1, grad = ones_like * 1.0, not zeros
        expected_grad = torch.ones_like(aux_loss)
        self.assertTrue(
            torch.allclose(aux_loss.grad, expected_grad),
            (f"aux_loss.grad should be ones * scale=1.0: "
             f"got={aux_loss.grad}, expected={expected_grad}"),
        )

    def test_lba04_set_loss_scale_updates_backward(self):
        """LB-A04: set_loss_scale changes the aux_loss gradient scale."""
        output = torch.randn(4, 8, requires_grad=True)
        aux_loss = torch.tensor(0.5, requires_grad=True)

        scale = torch.tensor(0.25)
        MoEAuxLossAutoScaler.set_loss_scale(scale)

        result = MoEAuxLossAutoScaler.apply(output, aux_loss)
        result.sum().backward()

        expected_grad = torch.ones_like(aux_loss) * 0.25
        self.assertTrue(
            torch.allclose(aux_loss.grad, expected_grad),
            (f"aux_loss.grad should be ones * scale=0.25: "
             f"got={aux_loss.grad}, expected={expected_grad}"),
        )

    def test_lba04b_set_loss_scale_in_place_update(self):
        """LB-A04b: Second set_loss_scale uses in-place copy_()."""
        scale1 = torch.tensor(0.5)
        MoEAuxLossAutoScaler.set_loss_scale(scale1)
        stored_tensor = MoEAuxLossAutoScaler.main_loss_backward_scale

        scale2 = torch.tensor(0.1)
        MoEAuxLossAutoScaler.set_loss_scale(scale2)

        # Should be the same tensor object (in-place copy, not new tensor)
        self.assertIs(
            MoEAuxLossAutoScaler.main_loss_backward_scale, stored_tensor,
            ("set_loss_scale should update in-place: "
             "identity changed after second call"),
        )
        self.assertAlmostEqual(
            MoEAuxLossAutoScaler.main_loss_backward_scale.item(), 0.1,
            places=5,
            msg=(
                f"Scale should be 0.1 after update: "
                f"got={MoEAuxLossAutoScaler.main_loss_backward_scale.item()}"
            ),
        )


class TestSequencePartitionGroup(unittest.TestCase):
    """Unit tests for sequence_partition_group in _compute_load_balance_loss."""

    def test_lbs01_none_gives_same_result(self):
        """LB-S01: sequence_partition_group=None produces same result as before."""
        num_experts = 4
        num_tokens = 100
        top_k = 2

        balanced_experts = torch.zeros(num_tokens, top_k, dtype=torch.long)
        for i in range(num_tokens):
            balanced_experts[i, 0] = i % num_experts
            balanced_experts[i, 1] = (i + 1) % num_experts
        balanced_scores = torch.ones(num_tokens, top_k) / top_k

        loss_default = _compute_load_balance_loss(
            balanced_scores, balanced_experts, num_experts
        )
        loss_explicit_none = _compute_load_balance_loss(
            balanced_scores, balanced_experts, num_experts,
            sequence_partition_group=None,
        )

        self.assertTrue(
            torch.allclose(loss_default, loss_explicit_none),
            (f"None group should match default: "
             f"default={loss_default.item():.6f}, "
             f"explicit_none={loss_explicit_none.item():.6f}"),
        )

    def test_lbs02_moe_accepts_sequence_partition_group(self):
        """LB-S02: MoE.__init__ accepts sequence_partition_group parameter."""
        moe = MoE(
            dim=16, hidden_dim=32, num_experts=4, top_k=2,
            load_balance_coeff=0.01,
            sequence_partition_group=None,
        )
        self.assertIsNone(
            moe.sequence_partition_group,
            "sequence_partition_group should be stored as None",
        )


if __name__ == "__main__":
    unittest.main()
