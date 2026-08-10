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
"""CPU unit tests for GRPO advantage, clipping, KL, and normalization."""

import math

import pytest
import torch

from rl.algorithm.grpo import (
    GRPOAlgorithm,
    GRPOConfig,
    compute_group_advantages,
    low_variance_kl,
    masked_mean,
)


def test_group_advantages_use_sample_standard_deviation() -> None:
    """Verify GRPO normalization uses PyTorch's sample standard deviation."""
    rewards = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    actual = compute_group_advantages(rewards)
    magnitude = 0.5 / (math.sqrt(1.0 / 3.0) + 1.0e-6)
    expected = torch.tensor([[-magnitude, magnitude, magnitude, -magnitude]])
    assert torch.allclose(actual, expected), f"Unexpected advantages: expected={expected}, got={actual}"


def test_zero_variance_group_produces_zero_advantages() -> None:
    """Verify standard GRPO retains equal-reward groups with zero advantages."""
    rewards = torch.ones(1, 8)
    actual = compute_group_advantages(rewards)
    expected = torch.zeros_like(rewards)
    assert torch.equal(actual, expected), f"Unexpected zero-variance advantages: expected={expected}, got={actual}"


def test_group_advantages_reject_single_response() -> None:
    """Verify a response group must contain enough values for sample std."""
    with pytest.raises(ValueError, match="at least two"):
        compute_group_advantages(torch.tensor([1.0]))


def test_algorithm_normalizes_each_trajectory_group_independently() -> None:
    """Verify batched prompts never share GRPO reward statistics."""
    rewards = torch.tensor([0.0, 1.0, 10.0, 12.0])
    algorithm = GRPOAlgorithm(GRPOConfig())
    actual = algorithm.compute_advantages(rewards, ("p0", "p0", "p1", "p1"))
    expected = torch.cat(
        (
            compute_group_advantages(rewards[:2]),
            compute_group_advantages(rewards[2:]),
        )
    )
    assert torch.allclose(actual, expected)


def test_masked_mean_ignores_invalid_tokens() -> None:
    """Verify response masks exclude padded or post-EOS tokens."""
    values = torch.tensor([[1.0, 3.0, 100.0]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    actual = masked_mean(values, mask)
    expected = torch.tensor(2.0)
    assert torch.equal(actual, expected), f"Unexpected masked mean: expected={expected}, got={actual}"


def test_low_variance_kl_is_zero_for_equal_policies() -> None:
    """Verify the k3 estimator vanishes for identical log probabilities."""
    log_probs = torch.tensor([[-1.0, -2.0]])
    actual = low_variance_kl(log_probs, log_probs)
    expected = torch.zeros_like(log_probs)
    assert torch.equal(actual, expected), f"Unexpected equal-policy KL: expected={expected}, got={actual}"


def test_low_variance_kl_is_non_negative_and_clamped() -> None:
    """Verify numerical clamps keep extreme k3 values finite and bounded."""
    current = torch.tensor([[-1000.0, 1000.0]])
    reference = torch.tensor([[1000.0, -1000.0]])
    actual = low_variance_kl(current, reference)
    assert torch.isfinite(actual).all(), f"KL must be finite, got={actual}"
    assert (actual >= 0).all(), f"KL must be non-negative, got={actual}"
    assert (actual <= 10).all(), f"KL must respect the upper clamp, got={actual}"


def test_dual_clipped_grpo_policy_loss() -> None:
    """Verify positive and negative advantages use the configured clipping rules."""
    current = torch.log(torch.tensor([[10.0, 10.0]]))
    old = torch.zeros_like(current)
    reference = current.detach().clone()
    advantages = torch.tensor([[1.0, -1.0]])
    mask = torch.ones_like(current, dtype=torch.bool)
    output = GRPOAlgorithm(GRPOConfig(kl_coef=0.0)).compute_actor_loss(
        current_log_probs=current,
        old_log_probs=old,
        reference_log_probs=reference,
        advantages=advantages,
        action_mask=mask,
    )
    expected_policy_sum = torch.tensor(1.8)
    assert torch.allclose(output.policy_loss_sum, expected_policy_sum), (
        f"Unexpected dual-clipped policy sum: expected={expected_policy_sum}, got={output.policy_loss_sum}"
    )
    expected_clipped = torch.tensor(2.0)
    assert torch.equal(output.clipped_token_count, expected_clipped), (
        f"Unexpected clipped-token count: expected={expected_clipped}, got={output.clipped_token_count}"
    )


def test_global_token_mean_scaling_matches_fsdp_gradient_average() -> None:
    """Verify per-rank scaling followed by FSDP averaging equals the global token mean."""
    dp_size = 2
    global_tokens = 5
    local_loss_sums = (torch.tensor(2.0), torch.tensor(9.0))
    scaled = [loss_sum / global_tokens * dp_size for loss_sum in local_loss_sums]
    fsdp_average = sum(scaled) / dp_size
    expected = sum(local_loss_sums) / global_tokens
    assert torch.equal(fsdp_average, expected), (
        f"FSDP-scaled loss mismatch: expected={expected}, got={fsdp_average}"
    )


def test_grpo_loss_backpropagates_nonzero_finite_gradients() -> None:
    """Verify a mixed-reward GRPO group drives the actor log probabilities."""
    current = torch.zeros((2, 2), requires_grad=True)
    old = torch.zeros_like(current)
    reference = torch.zeros_like(current)
    rewards = torch.tensor([0.0, 1.0])
    advantages = compute_group_advantages(rewards).unsqueeze(-1).expand_as(current)
    mask = torch.ones_like(current, dtype=torch.bool)

    output = GRPOAlgorithm(GRPOConfig(kl_coef=0.001)).compute_actor_loss(
        current_log_probs=current,
        old_log_probs=old,
        reference_log_probs=reference,
        advantages=advantages,
        action_mask=mask,
    )
    output.total_loss_sum.backward()

    assert current.grad is not None
    assert torch.isfinite(current.grad).all(), f"GRPO gradient must be finite, got={current.grad}"
    assert current.grad.abs().sum() > 0, f"Mixed rewards must produce a non-zero gradient, got={current.grad}"
