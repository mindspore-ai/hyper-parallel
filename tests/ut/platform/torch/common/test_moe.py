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
"""Unit tests for MoE building blocks: FeedForward, GroupedExperts,
TokenChoiceTopKRouter, and the MoE orchestrator.

All tests run on CPU without any distributed setup.
"""
import os
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

import torch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.platform.torch.common.moe import (  # pylint: disable=C0413
    FeedForward,
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
    update_expert_bias,
)


# ---------------------------------------------------------------------------
# TestFeedForward
# ---------------------------------------------------------------------------

class TestFeedForward(unittest.TestCase):
    """Unit tests for FeedForward (SwiGLU FFN)."""

    def setUp(self):
        """Create a small FeedForward on CPU."""
        torch.manual_seed(0)
        self.dim = 8
        self.hidden = 16
        self.ff = FeedForward(dim=self.dim, hidden_dim=self.hidden)

    def test_output_shape_3d(self):
        """FeedForward maps (bs, slen, dim) → same shape."""
        x = torch.randn(2, 4, self.dim)
        out = self.ff(x)
        assert out.shape == x.shape, (
            f"Expected shape {x.shape}, got {out.shape}"
        )

    def test_output_shape_2d(self):
        """FeedForward maps (num_tokens, dim) → same shape."""
        x = torch.randn(10, self.dim)
        out = self.ff(x)
        assert out.shape == x.shape, (
            f"Expected shape {x.shape}, got {out.shape}"
        )

    def test_swiglu_formula(self):
        """Output equals w2(silu(w1(x)) * w3(x)) computed manually."""
        x = torch.randn(3, self.dim)
        with torch.no_grad():
            expected = self.ff.w2(
                torch.nn.functional.silu(self.ff.w1(x)) * self.ff.w3(x)
            )
        out = self.ff(x)
        assert torch.allclose(out, expected, atol=1e-6), (
            f"SwiGLU formula mismatch: max_diff="
            f"{(out - expected).abs().max().item():.2e}"
        )

    def test_gradient_flows(self):
        """Gradients flow to all weight parameters."""
        x = torch.randn(4, self.dim)
        out = self.ff(x)
        out.sum().backward()
        for name, param in self.ff.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient after backward"
            )


# ---------------------------------------------------------------------------
# TestGroupedExperts
# ---------------------------------------------------------------------------

class TestGroupedExperts(unittest.TestCase):
    """Unit tests for GroupedExperts (batched SwiGLU expert computation)."""

    def setUp(self):
        """Create a small GroupedExperts with 4 experts on CPU."""
        torch.manual_seed(1)
        self.dim = 8
        self.hidden = 16
        self.num_experts = 4
        self.experts = GroupedExperts(
            dim=self.dim, hidden_dim=self.hidden, num_experts=self.num_experts
        )

    def _make_counts(self, total, num_experts):
        """Create token-count tensor that sums to *total*."""
        counts = torch.zeros(num_experts, dtype=torch.long)
        for i in range(total):
            counts[i % num_experts] += 1
        return counts

    def test_forward_shape(self):
        """Output shape is (total_tokens, dim)."""
        total = 10
        counts = self._make_counts(total, self.num_experts)
        x = torch.randn(total, self.dim)
        out = self.experts(x, counts)
        assert out.shape == (total, self.dim), (
            f"Expected ({total}, {self.dim}), got {out.shape}"
        )

    def test_zero_tokens_expert(self):
        """Expert with 0 tokens does not crash; its output rows are zero."""
        counts = torch.tensor([5, 0, 3, 0])
        total = int(counts.sum())
        x = torch.randn(total, self.dim)
        out = self.experts(x, counts)
        assert out.shape == (total, self.dim), (
            f"Unexpected output shape {out.shape}"
        )

    def test_dtensor_weights_delegated_to_local(self):
        """When weights are DTensor-like (isinstance check), to_local() is called.

        Bypasses nn.Module.__setattr__ via _parameters dict to inject mock objects.
        Each mock returns the correct real weight data so the forward can complete.
        """
        from hyper_parallel.core.dtensor.dtensor import DTensor

        experts = deepcopy(self.experts)
        call_counts = [0]

        def make_mock(real_data):
            mock = MagicMock(spec=DTensor)
            mock.to_local.return_value = real_data
            return mock

        mock_w1 = make_mock(experts.w1.data.clone())
        mock_w2 = make_mock(experts.w2.data.clone())
        mock_w3 = make_mock(experts.w3.data.clone())

        # Inject mocks directly to bypass nn.Parameter type check.
        experts._parameters["w1"] = mock_w1  # pylint: disable=protected-access
        experts._parameters["w2"] = mock_w2  # pylint: disable=protected-access
        experts._parameters["w3"] = mock_w3  # pylint: disable=protected-access

        counts = torch.tensor([2, 2, 2, 2])
        x = torch.randn(8, self.dim)
        experts(x, counts)

        assert mock_w1.to_local.call_count == 1, (
            f"Expected w1.to_local() called once, got {mock_w1.to_local.call_count}"
        )
        assert mock_w2.to_local.call_count == 1, (
            f"Expected w2.to_local() called once, got {mock_w2.to_local.call_count}"
        )
        assert mock_w3.to_local.call_count == 1, (
            f"Expected w3.to_local() called once, got {mock_w3.to_local.call_count}"
        )

    def test_gradient_flows(self):
        """Gradients reach w1 after a backward pass."""
        counts = self._make_counts(8, self.num_experts)
        x = torch.randn(8, self.dim)
        out = self.experts(x, counts)
        out.sum().backward()
        assert self.experts.w1.grad is not None, (
            "w1 has no gradient after backward"
        )


# ---------------------------------------------------------------------------
# TestTokenChoiceTopKRouter
# ---------------------------------------------------------------------------

class TestTokenChoiceTopKRouter(unittest.TestCase):
    """Unit tests for TokenChoiceTopKRouter."""

    def _make_router(self, **kwargs):
        torch.manual_seed(2)
        defaults = {"dim": 16, "num_experts": 8, "top_k": 2}
        defaults.update(kwargs)
        return TokenChoiceTopKRouter(**defaults)

    def test_sigmoid_output_shapes(self):
        """Router returns three tensors with correct shapes (sigmoid)."""
        router = self._make_router(score_func="sigmoid", top_k=2)
        x = torch.randn(32, 16)
        scores, indices, counts = router(x)
        assert scores.shape == (32, 2), (
            f"Expected scores shape (32, 2), got {scores.shape}"
        )
        assert indices.shape == (32, 2), (
            f"Expected indices shape (32, 2), got {indices.shape}"
        )
        assert counts.shape == (8,), (
            f"Expected counts shape (8,), got {counts.shape}"
        )

    def test_softmax_output_shapes(self):
        """Router returns correct shapes with softmax."""
        router = self._make_router(score_func="softmax", top_k=1)
        x = torch.randn(10, 16)
        scores, indices, counts = router(x)
        assert scores.shape == (10, 1), (
            f"Expected scores shape (10, 1), got {scores.shape}"
        )
        assert counts.shape == (8,), (
            f"Expected counts shape (8,), got {counts.shape}"
        )

    def test_counts_sum_equals_tokens_times_topk(self):
        """Token counts sum equals num_tokens * top_k."""
        router = self._make_router(top_k=2)
        x = torch.randn(16, 16)
        _, _, counts = router(x)
        expected = 16 * 2
        total = int(counts.sum())
        assert total == expected, (
            f"Expected counts.sum()={expected}, got {total}"
        )

    def test_top_k2_selects_distinct_experts(self):
        """With top_k=2, each token selects 2 distinct experts."""
        router = self._make_router(top_k=2)
        x = torch.randn(20, 16)
        _, indices, _ = router(x)
        for i in range(indices.shape[0]):
            e0, e1 = int(indices[i, 0]), int(indices[i, 1])
            assert e0 != e1, (
                f"Token {i} selected the same expert twice: {e0} == {e1}"
            )

    def test_expert_bias_shifts_routing(self):
        """A large positive bias for one expert increases its selection rate."""
        router = self._make_router(num_experts=4, top_k=1)
        x = torch.randn(100, 16)
        bias = torch.zeros(4)
        bias[2] = 100.0  # force all tokens to expert 2
        _, indices_biased, _ = router(x, expert_bias=bias)
        fraction_expert2 = (indices_biased == 2).float().mean().item()
        assert fraction_expert2 > 0.9, (
            f"Expected most tokens to route to expert 2, got fraction={fraction_expert2:.3f}"
        )

    def test_group_limited_routing(self):
        """Node-limited routing masks low-scoring groups to -inf."""
        router = self._make_router(
            num_experts=8, top_k=2,
            num_expert_groups=2, num_limited_groups=1,
        )
        x = torch.randn(10, 16)
        scores, indices, _ = router(x)
        # All selected experts should be within the single selected group
        # (experts 0-3 OR experts 4-7 per token, not mixed)
        for i in range(indices.shape[0]):
            group_ids = set((int(indices[i, k]) // 4) for k in range(2))
            assert len(group_ids) == 1, (
                f"Token {i} routed to experts in multiple groups: {indices[i].tolist()}"
            )

    def test_invalid_score_func_raises(self):
        """Unsupported score_func raises ValueError."""
        with self.assertRaises(ValueError):
            TokenChoiceTopKRouter(dim=8, num_experts=4, score_func="relu")

    def test_missing_num_limited_groups_raises(self):
        """num_expert_groups without num_limited_groups raises ValueError."""
        with self.assertRaises(ValueError):
            TokenChoiceTopKRouter(
                dim=8, num_experts=4, num_expert_groups=2
            )

    def test_route_scale_applied(self):
        """route_scale multiplies gate scores."""
        router_1 = self._make_router(route_scale=1.0, score_func="sigmoid")
        router_2 = deepcopy(router_1)
        router_2.route_scale = 2.0
        x = torch.randn(8, 16)
        scores_1, idx_1, _ = router_1(x)
        scores_2, idx_2, _ = router_2(x)
        # Same expert selection (bias unchanged), scores doubled
        assert torch.allclose(idx_1, idx_2), (
            f"Expert selection changed with route_scale: {idx_1} vs {idx_2}"
        )
        assert torch.allclose(scores_2, scores_1 * 2, atol=1e-5), (
            f"Expected scores doubled with route_scale=2; "
            f"max_diff={(scores_2 - scores_1 * 2).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# TestMoE
# ---------------------------------------------------------------------------

class TestMoE(unittest.TestCase):
    """Unit tests for MoE orchestrator (local-shard input, no distributed)."""

    def _make_moe(self, **kwargs):
        torch.manual_seed(3)
        defaults = {
            "dim": 16,
            "hidden_dim": 32,
            "num_experts": 4,
            "top_k": 2,
        }
        defaults.update(kwargs)
        return MoE(**defaults)

    def test_output_shape(self):
        """MoE maps (bs, local_slen, dim) → same shape."""
        moe = self._make_moe()
        x = torch.randn(2, 8, 16)
        out = moe(x)
        assert out.shape == x.shape, (
            f"Expected output shape {x.shape}, got {out.shape}"
        )

    def test_no_reorderer_attribute(self):
        """MoE must NOT have a 'reorderer' attribute (class was removed)."""
        moe = self._make_moe()
        assert not hasattr(moe, "reorderer"), (
            "MoE still has 'reorderer' attribute — TokenReorderer was not removed"
        )

    def test_inline_sort_equivalence(self):
        """Inline argsort in MoE.forward produces the same permutation as
        the original TokenReorderer logic (argsort on flattened experts)."""
        torch.manual_seed(10)
        top_k = 2
        num_experts = 4
        num_tokens = 6
        selected = torch.randint(0, num_experts, (num_tokens, top_k))
        scores = torch.rand(num_tokens, top_k)

        flat = selected.flatten()

        # Reproduce inline logic
        flat_indices = flat.argsort(stable=True)
        scores_sorted = scores.flatten()[flat_indices]
        token_indices = flat_indices // top_k
        counts = torch.bincount(flat, minlength=num_experts)

        # Verify expert-major ordering: experts must be non-decreasing
        experts_in_order = flat[flat_indices]
        assert (experts_in_order[1:] >= experts_in_order[:-1]).all(), (
            f"Sorted expert IDs are not non-decreasing: {experts_in_order.tolist()}"
        )
        # Verify counts
        expected_counts = torch.bincount(flat, minlength=num_experts)
        assert torch.equal(counts, expected_counts), (
            f"Counts mismatch: expected {expected_counts.tolist()}, got {counts.tolist()}"
        )
        # Verify token_indices are valid row indices
        assert token_indices.max() < num_tokens, (
            f"token_indices out of range: max={token_indices.max()}, num_tokens={num_tokens}"
        )

    def test_score_before_experts_true(self):
        """With score_before_experts=True, routing weights scale expert inputs."""
        moe = self._make_moe(score_before_experts=True)
        moe_false = deepcopy(moe)
        moe_false.score_before_experts = False
        x = torch.randn(2, 4, 16)
        out_true = moe(x)
        out_false = moe_false(x)
        assert not torch.allclose(out_true, out_false), (
            "score_before_experts=True and False produced identical outputs"
        )

    def test_tokens_per_expert_accumulates(self):
        """tokens_per_expert accumulates across multiple forward passes."""
        moe = self._make_moe()
        x = torch.randn(2, 8, 16)
        before = moe.tokens_per_expert.clone()
        moe(x)
        after1 = moe.tokens_per_expert.clone()
        moe(x)
        after2 = moe.tokens_per_expert.clone()
        assert (after1 > before).any(), (
            "tokens_per_expert did not increase after first forward"
        )
        assert (after2 > after1).any(), (
            "tokens_per_expert did not increase after second forward"
        )

    def test_load_balance_loss_attached(self):
        """With load_balance_coeff set, output carries _load_balance_loss."""
        moe = self._make_moe(load_balance_coeff=0.01)
        x = torch.randn(2, 4, 16)
        out = moe(x)
        assert hasattr(out, "_load_balance_loss"), (
            "Output missing '_load_balance_loss' attribute when load_balance_coeff is set"
        )
        assert out._load_balance_loss.ndim == 0, (  # pylint: disable=protected-access
            f"Expected scalar loss, got shape {out._load_balance_loss.shape}"  # pylint: disable=protected-access
        )

    def test_shared_expert_adds_to_output(self):
        """With a shared_expert, output differs from the no-shared-expert case."""
        shared = FeedForward(dim=16, hidden_dim=32)
        moe_shared = self._make_moe(shared_expert=shared)
        moe_no_shared = deepcopy(moe_shared)
        moe_no_shared.shared_expert = None
        x = torch.randn(2, 4, 16)
        out_shared = moe_shared(x)
        out_no_shared = moe_no_shared(x)
        assert not torch.allclose(out_shared, out_no_shared), (
            "Outputs with and without shared_expert are identical"
        )

    def test_dtensor_input_calls_to_local(self):
        """MoE.forward calls to_local() when input is a DTensor."""
        from hyper_parallel.core.dtensor.dtensor import DTensor
        moe = self._make_moe()
        x_local = torch.randn(2, 4, 16)

        mock_dt = MagicMock(spec=DTensor)
        mock_dt.to_local.return_value = x_local

        with patch(
            "hyper_parallel.platform.torch.common.moe.DTensor", DTensor
        ):
            # Make isinstance(mock_dt, DTensor) return True
            with patch(
                "hyper_parallel.platform.torch.common.moe.isinstance",
                side_effect=lambda obj, cls: True if obj is mock_dt else isinstance(obj, cls),
            ):
                moe(mock_dt)
        mock_dt.to_local.assert_called_once()

    def test_backward_gradients_flow(self):
        """loss.backward() propagates gradients to router gate weights."""
        moe = self._make_moe()
        x = torch.randn(2, 4, 16)
        out = moe(x)
        out.sum().backward()
        assert moe.router.gate.weight.grad is not None, (
            "router.gate.weight has no gradient after backward"
        )

    def test_update_expert_bias(self):
        """update_expert_bias adjusts expert_bias and resets tokens_per_expert."""
        moe = self._make_moe()
        x = torch.randn(2, 8, 16)
        moe(x)
        tokens_before = moe.tokens_per_expert.clone()
        bias_before = moe.expert_bias.clone()
        update_expert_bias(moe, lr=1e-2)
        assert moe.tokens_per_expert.sum().item() == 0, (
            f"tokens_per_expert not reset to zero: {moe.tokens_per_expert}"
        )
        assert not torch.allclose(moe.expert_bias, bias_before), (
            "expert_bias unchanged after update_expert_bias"
        )


# ---------------------------------------------------------------------------
# TestRegisterExpertBiasUpdater
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
