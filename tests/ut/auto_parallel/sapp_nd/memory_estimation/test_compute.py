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
"""Unit tests for compute.py: Expert compute FLOPs estimation.

Test IDs:
  CT-R01: router_compute_cost basic formula
  CT-R02: router_compute_cost scales with cp
  CT-R03: router_compute_cost not scaled by EP (replicated)
  CT-B01: expert_compute_cost_balanced basic formula with EP=4
  CT-B02: expert_compute_cost_balanced EP=1 (no EP)
  CT-B03: expert_compute_cost_balanced scales with cp
  CT-B04: expert_compute_cost_balanced reduces to 1/ep
  CT-I01: expert_compute_cost_imbalanced falls back to balanced when tokens=None
  CT-I02: expert_compute_cost_imbalanced with skewed tokens
  CT-I03: expert_compute_cost_imbalanced falls back when n_exp%ep!=0
  CT-I04: expert_compute_cost_imbalanced reduces to balanced under uniform
  CT-I05: expert_compute_cost_imbalanced max(etp,1) guards zero tokens
  CT-S01: shared_expert_compute_cost basic formula
  CT-S02: shared_expert_compute_cost uses etp when etp>1
  CT-S03: shared_expert_compute_cost falls back to tp when etp<=1
  CT-S04: etp=1 falls back to tp (etp=1 means "off", same as etp=0)
  CT-L01: expert_layer_compute returns 0 when n_exp=1 (dense)
  CT-L02: expert_layer_compute dispatches to balanced when tokens=None
  CT-L03: expert_layer_compute dispatches to imbalanced when tokens set
  CT-N01: n_ffMM scales balanced compute linearly
  CT-N02: n_ffMM scales imbalanced compute linearly
  CT-N03: n_ffMM scales shared expert compute linearly
  CT-N04: router_compute_cost not scaled by n_ffMM (single gate linear)
  CT-N05: n_ffMM defaults to 1 when attribute missing
"""
import os
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.compute import EvalExpertCompute


def _make_ccfg(
    n_exp=8,
    n_shared_exp=1,
    h=4096,
    hff=14336,
    hff_exp=14336,
    ep=4,
    tp=1,
    cp=1,
    etp=0,
    n_chosen_exp=2,
    s=1024,
    b=4,
    n_ffMM=3,  # pylint: disable=invalid-name
    tokens_per_expert=None,
):
    """Create a mock CostModelConfig for compute tests.

    Args:
        n_ffMM: Number of feedforward linear layers per expert
            (SwiGLU: gate+up+down = 3; standard MLP: 2).
    """
    ccfg = MagicMock()
    ccfg.n_exp = n_exp
    ccfg.n_shared_exp = n_shared_exp
    ccfg.h = h
    ccfg.hff = hff
    ccfg.hff_exp = hff_exp
    ccfg.ep = ep
    ccfg.t = tp
    ccfg.cp = cp
    ccfg.etp = etp
    ccfg.n_chosen_exp = n_chosen_exp
    ccfg.s = s
    ccfg.b = b
    ccfg.n_ffMM = n_ffMM
    ccfg.tokens_per_expert = tokens_per_expert
    return ccfg


class TestRouterCompute(unittest.TestCase):
    """CT-R: Router compute cost tests."""

    def test_router_basic(self):
        """CT-R01: router_compute_cost basic formula = 2*s*b*h*n_exp/cp."""
        ccfg = _make_ccfg(s=1024, b=4, h=4096, n_exp=8, cp=1)
        ctx = MagicMock()
        expected = 2 * 1024 * 4 * 4096 * 8
        self.assertAlmostEqual(
            EvalExpertCompute.router_compute_cost(ccfg, ctx), expected
        )

    def test_router_scales_with_cp(self):
        """CT-R02: router_compute_cost halves when cp=2."""
        ccfg1 = _make_ccfg(s=1024, b=4, h=4096, n_exp=8, cp=1)
        ccfg2 = _make_ccfg(s=1024, b=4, h=4096, n_exp=8, cp=2)
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.router_compute_cost(ccfg2, ctx),
            EvalExpertCompute.router_compute_cost(ccfg1, ctx) / 2,
        )

    def test_router_not_scaled_by_ep(self):
        """CT-R03: router_compute_cost does not change with EP (replicated)."""
        ccfg_ep1 = _make_ccfg(s=1024, b=4, h=4096, n_exp=8, ep=1)
        ccfg_ep4 = _make_ccfg(s=1024, b=4, h=4096, n_exp=8, ep=4)
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.router_compute_cost(ccfg_ep1, ctx),
            EvalExpertCompute.router_compute_cost(ccfg_ep4, ctx),
        )


class TestExpertComputeBalanced(unittest.TestCase):
    """CT-B: Balanced expert compute cost tests."""

    def test_balanced_basic(self):
        """CT-B01: expert_compute_cost_balanced with EP=4."""
        ccfg = _make_ccfg(s=1024, b=4, h=4096, hff_exp=14336, ep=4,
                          n_chosen_exp=2, cp=1)
        ctx = MagicMock()
        n_ff = ccfg.n_ffMM
        expected = 2 * n_ff * 1024 * 4 * 4096 * 14336 * 2 / 4
        self.assertAlmostEqual(
            EvalExpertCompute.expert_compute_cost_balanced(ccfg, ctx), expected
        )

    def test_balanced_ep1(self):
        """CT-B02: expert_compute_cost_balanced with EP=1 (no EP)."""
        ccfg = _make_ccfg(s=1024, b=4, h=4096, hff_exp=14336, ep=1,
                          n_chosen_exp=2, cp=1)
        ctx = MagicMock()
        n_ff = ccfg.n_ffMM
        expected = 2 * n_ff * 1024 * 4 * 4096 * 14336 * 2 / 1
        self.assertAlmostEqual(
            EvalExpertCompute.expert_compute_cost_balanced(ccfg, ctx), expected
        )

    def test_balanced_scales_with_cp(self):
        """CT-B03: expert_compute_cost_balanced halves when cp=2."""
        ccfg1 = _make_ccfg(s=1024, b=4, h=4096, hff_exp=14336, ep=4,
                           n_chosen_exp=2, cp=1)
        ccfg2 = _make_ccfg(s=1024, b=4, h=4096, hff_exp=14336, ep=4,
                           n_chosen_exp=2, cp=2)
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.expert_compute_cost_balanced(ccfg2, ctx),
            EvalExpertCompute.expert_compute_cost_balanced(ccfg1, ctx) / 2,
        )

    def test_balanced_reduces_by_ep(self):
        """CT-B04: expert_compute_cost_balanced reduces to 1/ep of EP=1 value."""
        ccfg_ep1 = _make_ccfg(s=1024, b=4, h=4096, hff_exp=14336, ep=1,
                              n_chosen_exp=2, cp=1)
        ccfg_ep4 = _make_ccfg(s=1024, b=4, h=4096, hff_exp=14336, ep=4,
                              n_chosen_exp=2, cp=1)
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.expert_compute_cost_balanced(ccfg_ep4, ctx),
            EvalExpertCompute.expert_compute_cost_balanced(ccfg_ep1, ctx) / 4,
        )


class TestExpertComputeImbalanced(unittest.TestCase):
    """CT-I: Imbalanced expert compute cost tests."""

    def test_imbalanced_fallback_no_tokens(self):
        """CT-I01: imbalanced falls back to balanced when tokens=None."""
        ccfg = _make_ccfg(ep=4, tokens_per_expert=None)
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.expert_compute_cost_imbalanced(ccfg, ctx),
            EvalExpertCompute.expert_compute_cost_balanced(ccfg, ctx),
        )

    def test_imbalanced_skewed_tokens(self):
        """CT-I02: imbalanced with skewed tokens uses max(rank_tokens)."""
        # 4 experts, ep=2, so 2 experts per rank
        # tokens_per_expert = [100, 200, 10, 10]
        # rank0: expert0+expert1 = 300, rank1: expert2+expert3 = 20
        # max_inbound = 300
        tokens = [100, 200, 10, 10]
        ccfg = _make_ccfg(n_exp=4, ep=2, hff_exp=14336, h=4096, cp=1,
                          tokens_per_expert=tokens)
        ctx = MagicMock()
        n_ff = ccfg.n_ffMM
        expected = 2 * n_ff * 300 * 4096 * 14336
        self.assertAlmostEqual(
            EvalExpertCompute.expert_compute_cost_imbalanced(ccfg, ctx), expected
        )

    def test_imbalanced_fallback_not_divisible(self):
        """CT-I03: imbalanced falls back when n_exp%ep!=0."""
        ccfg = _make_ccfg(n_exp=8, ep=3, tokens_per_expert=[1]*8)
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.expert_compute_cost_imbalanced(ccfg, ctx),
            EvalExpertCompute.expert_compute_cost_balanced(ccfg, ctx),
        )

    def test_imbalanced_fallback_short_tokens(self):
        """CT-I03b: imbalanced falls back when len(tokens) < n_exp."""
        ccfg = _make_ccfg(n_exp=8, ep=4, tokens_per_expert=[100, 200])
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.expert_compute_cost_imbalanced(ccfg, ctx),
            EvalExpertCompute.expert_compute_cost_balanced(ccfg, ctx),
        )

    def test_imbalanced_uniform_equals_balanced(self):
        """CT-I04: imbalanced reduces to balanced under uniform token distribution."""
        # Under uniform routing: each expert gets s*b*n_chosen/n_exp tokens.
        # Per-rank token count = (n_exp/ep) * (s*b*n_chosen/n_exp) = s*b*n_chosen/ep,
        # which equals the balanced formula's implicit per-rank count.
        n_exp, ep, s, b, h = 8, 4, 1024, 4, 4096
        hff_exp, n_chosen_exp = 14336, 2
        token_per_exp = s * b * n_chosen_exp // n_exp  # 1024
        tokens = [token_per_exp] * n_exp
        ccfg_bal = _make_ccfg(n_exp=n_exp, ep=ep, s=s, b=b, h=h,
                              hff_exp=hff_exp, n_chosen_exp=n_chosen_exp, cp=1,
                              tokens_per_expert=None)
        ccfg_imb = _make_ccfg(n_exp=n_exp, ep=ep, s=s, b=b, h=h,
                              hff_exp=hff_exp, n_chosen_exp=n_chosen_exp, cp=1,
                              tokens_per_expert=tokens)
        ctx = MagicMock()
        balanced = EvalExpertCompute.expert_compute_cost_balanced(ccfg_bal, ctx)
        imbalanced = EvalExpertCompute.expert_compute_cost_imbalanced(ccfg_imb, ctx)
        self.assertAlmostEqual(imbalanced, balanced, places=4)

    def test_imbalanced_max_etp_guard(self):
        """CT-I05: imbalanced uses max(etp,1) to guard zero tokens."""
        # All tokens are zero for one rank, max should still be >= 1
        tokens = [0, 0, 100, 100]
        ccfg = _make_ccfg(n_exp=4, ep=2, h=4096, hff_exp=14336, cp=1,
                          tokens_per_expert=tokens)
        ctx = MagicMock()
        result = EvalExpertCompute.expert_compute_cost_imbalanced(ccfg, ctx)
        self.assertGreater(result, 0)


class TestSharedExpertCompute(unittest.TestCase):
    """CT-S: Shared expert compute cost tests."""

    def test_shared_basic(self):
        """CT-S01: shared_expert_compute_cost basic formula."""
        ccfg = _make_ccfg(s=1024, b=4, h=4096, hff=14336, n_shared_exp=1,
                          tp=1, etp=0, cp=1)
        ctx = MagicMock()
        n_ff = ccfg.n_ffMM
        expected = 2 * n_ff * 1024 * 4 * 4096 * 14336 * 1 / 1
        self.assertAlmostEqual(
            EvalExpertCompute.shared_expert_compute_cost(ccfg, ctx), expected
        )

    def test_shared_uses_etp(self):
        """CT-S02: shared_expert_compute_cost uses etp when etp>1."""
        ccfg = _make_ccfg(s=1024, b=4, h=4096, hff=14336, n_shared_exp=1,
                          tp=1, etp=2, cp=1)
        ctx = MagicMock()
        n_ff = ccfg.n_ffMM
        expected = 2 * n_ff * 1024 * 4 * 4096 * 14336 * 1 / 2
        self.assertAlmostEqual(
            EvalExpertCompute.shared_expert_compute_cost(ccfg, ctx), expected
        )

    def test_shared_fallback_tp(self):
        """CT-S03: shared_expert_compute_cost uses tp when etp<=1."""
        ccfg = _make_ccfg(s=1024, b=4, h=4096, hff=14336, n_shared_exp=1,
                          tp=2, etp=0, cp=1)
        ctx = MagicMock()
        n_ff = ccfg.n_ffMM
        expected = 2 * n_ff * 1024 * 4 * 4096 * 14336 * 1 / 2
        self.assertAlmostEqual(
            EvalExpertCompute.shared_expert_compute_cost(ccfg, ctx), expected
        )

    def test_etp1_falls_back_to_tp(self):
        """CT-S04: etp=1 falls back to tp (etp=1 means "off", same as etp=0).

        In MindFormers, expert_model_parallel=1 means "disabled".
        With etp>1 check, etp=1 falls back to tp, matching this intent.
        """
        ccfg_etp0 = _make_ccfg(s=1024, b=4, h=4096, hff=14336, n_shared_exp=1,
                               tp=2, etp=0, cp=1)
        ccfg_etp1 = _make_ccfg(s=1024, b=4, h=4096, hff=14336, n_shared_exp=1,
                               tp=2, etp=1, cp=1)
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.shared_expert_compute_cost(ccfg_etp1, ctx),
            EvalExpertCompute.shared_expert_compute_cost(ccfg_etp0, ctx),
        )


class TestExpertLayerCompute(unittest.TestCase):
    """CT-L: Expert layer compute dispatcher tests."""

    def test_layer_returns_zero_dense(self):
        """CT-L01: expert_layer_compute returns 0 when n_exp=1 (dense)."""
        ccfg = _make_ccfg(n_exp=1)
        ctx = MagicMock()
        self.assertEqual(EvalExpertCompute.expert_layer_compute(ccfg, ctx), 0)

    def test_layer_dispatches_balanced(self):
        """CT-L02: expert_layer_compute dispatches to balanced when tokens=None."""
        ccfg = _make_ccfg(n_exp=8, ep=4, tokens_per_expert=None)
        ctx = MagicMock()
        result = EvalExpertCompute.expert_layer_compute(ccfg, ctx)
        expected = (
            EvalExpertCompute.router_compute_cost(ccfg, ctx)
            + EvalExpertCompute.expert_compute_cost_balanced(ccfg, ctx)
            + EvalExpertCompute.shared_expert_compute_cost(ccfg, ctx)
        )
        self.assertAlmostEqual(result, expected)

    def test_layer_dispatches_imbalanced(self):
        """CT-L03: expert_layer_compute dispatches to imbalanced when tokens set."""
        tokens = [100, 200, 10, 10]
        ccfg = _make_ccfg(n_exp=4, ep=2, tokens_per_expert=tokens)
        ctx = MagicMock()
        result = EvalExpertCompute.expert_layer_compute(ccfg, ctx)
        expected = (
            EvalExpertCompute.router_compute_cost(ccfg, ctx)
            + EvalExpertCompute.expert_compute_cost_imbalanced(ccfg, ctx)
            + EvalExpertCompute.shared_expert_compute_cost(ccfg, ctx)
        )
        self.assertAlmostEqual(result, expected)


class TestNffmmFactor(unittest.TestCase):
    """CT-N: n_ffMM (num feedforward linear layers) factor tests.

    SwiGLU has 3 linear layers (gate+up+down); standard MLP has 2.
    The factor must scale routed/shared expert FLOPs linearly but NOT
    affect router FLOPs (router is a single gate linear layer).
    """

    def test_nffmm_scales_balanced(self):
        """CT-N01: n_ffMM=3 produces 3x the FLOPs of n_ffMM=1 (balanced)."""
        ccfg1 = _make_ccfg(n_ffMM=1, ep=4)
        ccfg3 = _make_ccfg(n_ffMM=3, ep=4)
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.expert_compute_cost_balanced(ccfg3, ctx),
            EvalExpertCompute.expert_compute_cost_balanced(ccfg1, ctx) * 3,
        )

    def test_nffmm_scales_imbalanced(self):
        """CT-N02: n_ffMM=3 produces 3x the FLOPs of n_ffMM=1 (imbalanced)."""
        tokens = [100, 200, 10, 10]
        ccfg1 = _make_ccfg(n_ffMM=1, n_exp=4, ep=2,
                           tokens_per_expert=tokens)
        ccfg3 = _make_ccfg(n_ffMM=3, n_exp=4, ep=2,
                           tokens_per_expert=tokens)
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.expert_compute_cost_imbalanced(ccfg3, ctx),
            EvalExpertCompute.expert_compute_cost_imbalanced(ccfg1, ctx) * 3,
        )

    def test_nffmm_scales_shared(self):
        """CT-N03: n_ffMM=3 produces 3x the FLOPs of n_ffMM=1 (shared)."""
        ccfg1 = _make_ccfg(n_ffMM=1, n_shared_exp=1)
        ccfg3 = _make_ccfg(n_ffMM=3, n_shared_exp=1)
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.shared_expert_compute_cost(ccfg3, ctx),
            EvalExpertCompute.shared_expert_compute_cost(ccfg1, ctx) * 3,
        )

    def test_router_not_scaled_by_nffmm(self):
        """CT-N04: router_compute_cost not scaled by n_ffMM (single gate)."""
        ccfg1 = _make_ccfg(n_ffMM=1, n_exp=8)
        ccfg3 = _make_ccfg(n_ffMM=3, n_exp=8)
        ctx = MagicMock()
        self.assertAlmostEqual(
            EvalExpertCompute.router_compute_cost(ccfg1, ctx),
            EvalExpertCompute.router_compute_cost(ccfg3, ctx),
        )

    def test_nffmm_defaults_to_1(self):
        """CT-N05: n_ffMM defaults to 1 when attribute is missing.

        Ensures formulas work when ccfg is a minimal mock without n_ffMM
        (e.g. in tests that don't set arch_hooks).
        """
        from unittest.mock import Mock
        # Use plain Mock (not MagicMock) with a spec that excludes n_ffMM
        # so getattr falls back to the default of 1.
        ccfg = Mock(spec=[
            "n_exp", "n_shared_exp", "h", "hff", "hff_exp",
            "ep", "t", "cp", "etp", "n_chosen_exp", "s", "b",
            "tokens_per_expert",
        ])
        ccfg.n_exp = 8
        ccfg.n_shared_exp = 1
        ccfg.h = 4096
        ccfg.hff = 14336
        ccfg.hff_exp = 14336
        ccfg.ep = 4
        ccfg.t = 1
        ccfg.cp = 1
        ccfg.etp = 0
        ccfg.n_chosen_exp = 2
        ccfg.s = 1024
        ccfg.b = 4
        ccfg.tokens_per_expert = None
        ctx = MagicMock()
        # Should not raise; uses getattr fallback of 1
        result = EvalExpertCompute.expert_compute_cost_balanced(ccfg, ctx)
        self.assertGreater(result, 0)
        # Verify n_ffMM is not an attribute of the spec'd mock
        self.assertFalse(hasattr(ccfg, "n_ffMM"))


if __name__ == "__main__":
    unittest.main()
