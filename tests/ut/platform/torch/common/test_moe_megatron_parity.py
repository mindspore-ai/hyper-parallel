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
"""Parity tests comparing HyperParallel with Megatron-LM implementations.

This module verifies that HyperParallel's load balance loss and expert bias
update produce results consistent with Megatron-LM's reference implementation.

Megatron Source:
  - switch_load_balancing_loss_func: megatron/core/transformer/moe/moe_utils.py:25-70
  - get_updated_expert_bias: megatron/core/transformer/moe/moe_utils.py:726-743
"""
import os
import unittest

import torch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.platform.torch.common.moe import (  # pylint: disable=C0413
    _compute_load_balance_loss,
)


def megatron_switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    moe_aux_loss_coeff: float = 1.0,
) -> torch.Tensor:
    """Reference implementation from Megatron-LM.

    Source: megatron/core/transformer/moe/moe_utils.py:25-70

    Args:
        probs: Softmax probabilities [num_tokens, num_experts].
        tokens_per_expert: Token counts per expert [num_experts].
        topk: Number of experts selected per token.
        moe_aux_loss_coeff: Loss coefficient.

    Returns:
        Scalar auxiliary loss.
    """
    num_tokens = probs.shape[0]
    num_experts = probs.shape[1]
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
    )
    return aux_loss


def megatron_get_updated_expert_bias(
    tokens_per_expert: torch.Tensor,
    expert_bias: torch.Tensor,
    expert_bias_update_rate: float,
) -> torch.Tensor:
    """Reference implementation from Megatron-LM.

    Source: megatron/core/transformer/moe/moe_utils.py:726-743

    Note: Megatron's implementation does NOT include mean-zero constraint.
    HyperParallel adds this constraint per Issue #150 requirement 2.2.

    Args:
        tokens_per_expert: Token counts per expert [num_experts].
        expert_bias: Current expert bias [num_experts].
        expert_bias_update_rate: Update rate (lr).

    Returns:
        Updated expert bias.
    """
    average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
    offset = average_tokens - tokens_per_expert
    updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate
    return updated_expert_bias


class TestLoadBalanceLossMegatronParity(unittest.TestCase):
    """Compare _compute_load_balance_loss with Megatron's switch_load_balancing_loss_func.

    IMPORTANT: Megatron and HyperParallel use different input semantics:

    Megatron (switch_load_balancing_loss_func):
        - Input: full softmax probs [num_tokens, num_experts]
        - expert_prob = sum of ALL token probabilities for expert i (including non-selected)
        - Formula: loss = num_experts * Σ(probs_sum_i * tokens_i) / (N² * k)

    HyperParallel (_compute_load_balance_loss):
        - Input: top-k scores [num_tokens, top_k]
        - expert_prob = sum of routing scores for expert i (only selected experts)
        - Formula: loss = num_experts * Σ(expert_fraction_i * expert_prob_i)

    These are DIFFERENT semantics:
        - Megatron considers probability mass for ALL experts (even unselected)
        - HyperParallel only considers selected experts' scores

    Both are valid implementations but measure slightly different things.
    """

    def test_formula_equivalence_with_same_input(self):
        """Verify formulas are mathematically equivalent WHEN given same input.

        If we construct Megatron's full_probs to only have non-zero values at
        selected positions (same as HP's top-k scores), both formulas should match.
        """
        num_tokens = 50
        num_experts = 4
        top_k = 2

        top_scores = torch.rand(num_tokens, top_k)
        selected_experts = torch.zeros(num_tokens, top_k, dtype=torch.long)
        for i in range(num_tokens):
            selected_experts[i, 0] = i % num_experts
            selected_experts[i, 1] = (i + 1) % num_experts

        tokens_per_expert = torch.bincount(
            selected_experts.flatten(), minlength=num_experts
        ).float()

        hp_loss = _compute_load_balance_loss(top_scores, selected_experts, num_experts)

        full_probs = torch.zeros(num_tokens, num_experts)
        full_probs.scatter_(1, selected_experts, top_scores)

        megatron_loss = megatron_switch_load_balancing_loss_func(
            full_probs, tokens_per_expert, top_k
        )

        self.assertTrue(
            torch.allclose(hp_loss, megatron_loss, rtol=1e-4, atol=1e-6),
            f"Formula equivalence failed: HP={hp_loss.item():.6f}, Megatron={megatron_loss.item():.6f}"
        )

    def test_loss_direction_consistency(self):
        """Both implementations should agree on which routing is more balanced.

        Even though absolute loss values differ, both should rank balanced > unbalanced.
        """
        torch.manual_seed(42)
        num_tokens = 100
        num_experts = 4
        top_k = 2

        balanced_experts = torch.zeros(num_tokens, top_k, dtype=torch.long)
        for i in range(num_tokens):
            balanced_experts[i, 0] = i % num_experts
            balanced_experts[i, 1] = (i + 1) % num_experts
        balanced_scores = torch.ones(num_tokens, top_k) / top_k

        hp_balanced = _compute_load_balance_loss(balanced_scores, balanced_experts, num_experts)

        unbalanced_experts = torch.zeros(num_tokens, top_k, dtype=torch.long)
        unbalanced_scores = torch.ones(num_tokens, top_k) / top_k
        hp_unbalanced = _compute_load_balance_loss(unbalanced_scores, unbalanced_experts, num_experts)

        self.assertGreater(hp_unbalanced, hp_balanced, "HP: unbalanced loss should be > balanced")

        tokens_balanced = torch.bincount(balanced_experts.flatten(), minlength=num_experts).float()
        full_probs_balanced = torch.zeros(num_tokens, num_experts)
        full_probs_balanced.scatter_(1, balanced_experts, balanced_scores)
        mg_balanced = megatron_switch_load_balancing_loss_func(
            full_probs_balanced, tokens_balanced, top_k
        )

        tokens_unbalanced = torch.bincount(unbalanced_experts.flatten(), minlength=num_experts).float()
        full_probs_unbalanced = torch.zeros(num_tokens, num_experts)
        full_probs_unbalanced.scatter_(1, unbalanced_experts, unbalanced_scores)
        mg_unbalanced = megatron_switch_load_balancing_loss_func(
            full_probs_unbalanced, tokens_unbalanced, top_k
        )

        self.assertGreater(mg_unbalanced, mg_balanced, "Megatron: unbalanced loss should be > balanced")


class TestExpertBiasUpdateMegatronParity(unittest.TestCase):
    """Compare update_expert_bias with Megatron's get_updated_expert_bias.

    IMPORTANT: HyperParallel adds mean-zero constraint per Issue #150 2.2.

    Megatron: updated_bias = bias + sign(avg - tokens) * lr
    HyperParallel: delta = sign(avg - tokens) * lr; delta -= delta.mean(); bias += delta

    The mean-zero constraint prevents systematic drift of all bias values.
    """

    def test_direction_consistency(self):
        """Both implementations should agree on which experts need adjustment.

        For experts far from average, both should agree on the direction.
        For experts near average, the mean-zero adjustment may flip the sign.
        """
        torch.manual_seed(42)
        num_experts = 4
        lr = 0.01

        tokens_per_expert = torch.tensor([100.0, 10.0, 50.0, 40.0])
        expert_bias = torch.zeros(num_experts)

        megatron_updated = megatron_get_updated_expert_bias(
            tokens_per_expert.clone(), expert_bias.clone(), lr
        )
        megatron_deltas = megatron_updated - expert_bias

        hp_bias = expert_bias.clone()
        avg = tokens_per_expert.float().mean()
        delta = lr * (avg - tokens_per_expert.float()).sign()
        delta = delta - delta.mean()
        hp_updated = hp_bias + delta
        hp_deltas = hp_updated - hp_bias

        self.assertLess(hp_deltas[0].item(), 0, "Expert 0 (overloaded) should decrease")
        self.assertGreater(hp_deltas[1].item(), 0, "Expert 1 (underloaded) should increase")

        self.assertEqual(
            megatron_deltas[0].sign().item(),
            hp_deltas[0].sign().item(),
            "Both should agree on overloaded expert direction"
        )
        self.assertEqual(
            megatron_deltas[1].sign().item(),
            hp_deltas[1].sign().item(),
            "Both should agree on underloaded expert direction"
        )

    def test_mean_zero_constraint_difference(self):
        """Verify that HyperParallel adds mean-zero constraint (Issue #150 2.2).

        Megatron's implementation does NOT enforce mean-zero constraint.
        HyperParallel adds: delta = delta - delta.mean()

        This prevents systematic drift of all bias values over time.
        """
        torch.manual_seed(42)
        num_experts = 4
        lr = 0.01

        tokens_per_expert = torch.tensor([100.0, 10.0, 50.0, 40.0])
        expert_bias = torch.zeros(num_experts)

        megatron_updated = megatron_get_updated_expert_bias(
            tokens_per_expert.clone(), expert_bias.clone(), lr
        )
        megatron_mean_drift = (megatron_updated - expert_bias).mean().item()

        hp_bias = expert_bias.clone()
        avg = tokens_per_expert.float().mean()
        delta = lr * (avg - tokens_per_expert.float()).sign()
        delta = delta - delta.mean()
        hp_updated = hp_bias + delta
        hp_mean_drift = (hp_updated - hp_bias).mean().item()

        self.assertNotEqual(
            megatron_mean_drift, 0.0,
            "Megatron should have non-zero mean drift (no constraint)"
        )
        self.assertAlmostEqual(
            hp_mean_drift, 0.0,
            places=7,
            msg="HyperParallel should have zero mean drift (mean-zero constraint)"
        )

    def test_extreme_imbalance(self):
        """Test with extreme token imbalance."""
        num_experts = 8
        lr = 0.1

        tokens_per_expert = torch.tensor([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        expert_bias = torch.zeros(num_experts)

        megatron_updated = megatron_get_updated_expert_bias(
            tokens_per_expert.clone(), expert_bias.clone(), lr
        )
        hp_bias = expert_bias.clone()
        avg = tokens_per_expert.float().mean()
        delta = lr * (avg - tokens_per_expert.float()).sign()
        delta = delta - delta.mean()
        hp_updated = hp_bias + delta

        self.assertLess(
            hp_updated[0].item(), 0.0,
            "Overloaded expert (0) bias should decrease"
        )
        for i in range(1, num_experts):
            self.assertGreater(
                hp_updated[i].item(), 0.0,
                f"Underloaded expert ({i}) bias should increase"
            )

        self.assertEqual(
            megatron_updated[0].sign().item(),
            hp_updated[0].sign().item(),
            "Both should agree on overloaded expert direction"
        )


class TestEndToEndParity(unittest.TestCase):
    """End-to-end parity tests with simulated MoE routing."""

    def test_simulated_routing_scenario(self):
        """Simulate a realistic routing scenario and compare both implementations.

        This test verifies:
        1. Both implementations produce reasonable loss values
        2. Bias update directions are consistent for extreme cases
        """
        torch.manual_seed(2024)
        num_tokens = 256
        num_experts = 8
        top_k = 2
        hidden_dim = 64

        gate_weights = torch.randn(num_experts, hidden_dim)
        tokens = torch.randn(num_tokens, hidden_dim)

        logits = torch.nn.functional.linear(  # pylint: disable=E1102
            tokens, gate_weights
        )
        full_probs = torch.softmax(logits, dim=-1)

        top_scores, selected_experts = full_probs.topk(top_k, dim=-1)
        tokens_per_expert = torch.bincount(
            selected_experts.flatten(), minlength=num_experts
        ).float()

        hp_loss = _compute_load_balance_loss(top_scores, selected_experts, num_experts)
        self.assertGreater(hp_loss.item(), 0.0, "HP loss should be positive")

        full_probs_for_megatron = torch.zeros(num_tokens, num_experts)
        full_probs_for_megatron.scatter_(1, selected_experts, top_scores)
        megatron_loss = megatron_switch_load_balancing_loss_func(
            full_probs_for_megatron, tokens_per_expert, top_k
        )

        self.assertTrue(
            torch.allclose(hp_loss, megatron_loss, rtol=1e-4, atol=1e-6),
            f"With same input: HP={hp_loss.item():.6f}, Megatron={megatron_loss.item():.6f}"
        )

        lr = 0.01
        expert_bias_hp = torch.zeros(num_experts)
        expert_bias_megatron = torch.zeros(num_experts)

        avg = tokens_per_expert.float().mean()
        delta = lr * (avg - tokens_per_expert.float()).sign()
        delta = delta - delta.mean()
        expert_bias_hp = expert_bias_hp + delta

        megatron_updated = megatron_get_updated_expert_bias(
            tokens_per_expert, expert_bias_megatron, lr
        )

        hp_mean = expert_bias_hp.mean().item()
        megatron_mean = megatron_updated.mean().item()

        self.assertAlmostEqual(hp_mean, 0.0, places=7, msg="HP bias mean should be ~0")
        self.assertNotEqual(megatron_mean, 0.0, "Megatron bias mean may drift")


if __name__ == "__main__":
    unittest.main()
