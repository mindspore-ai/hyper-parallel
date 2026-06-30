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
"""Unit tests for body.py: MoE/dense layer param splitting and memory estimation.

Test IDs:
  BD-N01: num_params_layer returns 3-tuple with dense FFN (n_exp=1)
  BD-N02: num_params_layer returns 3-tuple with MoE FFN (n_exp>1)
  BD-N03: num_params_layer with routed/shared expert breakdown
  BD-P01: stat_p_layer dense FFN memory
  BD-P02: stat_p_layer MoE with EP sharding on routed, partial on shared
  BD-P03: stat_p_layer MoE non-exp params use non_exp_partial sharding
  BD-O01: stat_os_layer dense FFN optimizer state
  BD-O02: stat_os_layer MoE optimizer state with EP/partial sharding
  BD-O03: stat_os_layer returns 0 when swap_os is True
  BD-G01: stat_grad_layer dense FFN gradient memory
  BD-G02: stat_grad_layer MoE gradient with EP/partial sharding
  BD-G03: stat_grad_layer shared expert uses shard_grad_exp_partial
"""
import os
import unittest
from unittest.mock import MagicMock, PropertyMock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.body import EvalBody
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalFFn, EvalAttn, EvalNorm


def _make_ccfg(
    n_exp=1,
    n_shared_exp=0,
    h=4096,
    hff=14336,
    hff_exp=14336,
    ep=1,
    etp=1,
    bytes_p=2,
    bytes_os=12,
    bytes_grad=2,
    shard_p_os_non_exp_partial=1.0,
    shard_p_os_exp=1.0,
    shard_p_os_exp_partial=1.0,
    shard_grad_non_exp=1.0,
    shard_grad_exp=1.0,
    shard_grad_exp_partial=1.0,
):
    """Create a mock CostModelConfig for body tests."""
    ccfg = MagicMock()
    ccfg.n_exp = n_exp
    ccfg.n_shared_exp = n_shared_exp
    ccfg.h = h
    ccfg.hff = hff
    ccfg.hff_exp = hff_exp
    ccfg.ep = ep
    ccfg.etp = etp
    ccfg.n_ffMM = 1
    ccfg.n_ffBMM = 0
    ccfg.bytes_p = bytes_p
    ccfg.bytes_os = bytes_os
    ccfg.bytes_grad = bytes_grad
    ccfg.shard_p_os_non_exp_partial = shard_p_os_non_exp_partial
    ccfg.shard_p_os_exp = shard_p_os_exp
    ccfg.shard_p_os_exp_partial = shard_p_os_exp_partial
    ccfg.shard_grad_non_exp = shard_grad_non_exp
    ccfg.shard_grad_exp = shard_grad_exp
    ccfg.shard_grad_exp_partial = shard_grad_exp_partial
    return ccfg


def _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, ffn_p=200.0,
              routed_p=300.0, shared_p=100.0, swap_os=False):
    """Create a mock Context for body tests.

    The ctx.eval.num_p(ccfg, ctx) must return the 3-tuple that
    num_params_layer would produce. We wire it through the context
    attributes to simulate the real hook-manager flow.
    """
    ctx = MagicMock()
    ctx.swap_os = swap_os

    # Wire context formula pointers to real static methods
    ctx.attn_num_p = EvalAttn.num_params_attn if attn_p == "real" else (lambda c, x: attn_p)
    ctx.norm_num_p = EvalNorm.num_params_norm if norm_p == "real" else (lambda c, x: norm_p)
    ctx.ffn_num_p = EvalFFn.num_params_ffn if ffn_p == "real" else (lambda c, x: ffn_p)
    ctx.ffn_routed_num_p = EvalFFn.num_params_routed_expert if routed_p == "real" else (
        lambda c, x: routed_p
    )
    ctx.ffn_shared_num_p = EvalFFn.num_params_shared_expert if shared_p == "real" else (
        lambda c, x: shared_p
    )

    # ctx.eval.num_p returns the tuple from EvalBody.num_params_layer
    ctx.eval = MagicMock()
    ctx.eval.num_p = lambda c, x: EvalBody.num_params_layer(c, ctx)
    return ctx


class TestNumParamsLayer(unittest.TestCase):
    """Test EvalBody.num_params_layer 3-tuple output."""

    def test_dense_ffn_returns_non_exp_only(self):
        """BD-N01: Dense FFN (n_exp=1) returns (attn+norm+ffn, 0, 0)."""
        ccfg = _make_ccfg(n_exp=1)
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, ffn_p=200.0)
        result = EvalBody.num_params_layer(ccfg, ctx)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        non_exp, routed, shared = result
        self.assertAlmostEqual(non_exp, 350.0)  # 100 + 50 + 200
        self.assertEqual(routed, 0.0)
        self.assertEqual(shared, 0.0)

    def test_moe_ffn_returns_separate_routed_shared(self):
        """BD-N02: MoE FFN (n_exp>1) returns (attn+norm, routed, shared)."""
        ccfg = _make_ccfg(n_exp=8, n_shared_exp=1, h=4096, hff=14336, hff_exp=14336)
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, routed_p=300.0, shared_p=100.0)
        result = EvalBody.num_params_layer(ccfg, ctx)
        non_exp, routed, shared = result
        self.assertAlmostEqual(non_exp, 150.0)  # 100 + 50, no dense FFN
        self.assertAlmostEqual(routed, 300.0)
        self.assertAlmostEqual(shared, 100.0)

    def test_moe_no_shared_expert(self):
        """BD-N03: MoE without shared expert returns shared=0."""
        ccfg = _make_ccfg(n_exp=8, n_shared_exp=0)
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, routed_p=300.0, shared_p=0.0)
        result = EvalBody.num_params_layer(ccfg, ctx)
        non_exp, routed, shared = result
        self.assertAlmostEqual(non_exp, 150.0)
        self.assertAlmostEqual(routed, 300.0)
        self.assertAlmostEqual(shared, 0.0)

    def test_moe_none_routed_and_shared(self):
        """BD-N03b: MoE with None routed/shared pointers returns 0 for those."""
        ccfg = _make_ccfg(n_exp=8, n_shared_exp=1)
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, routed_p=300.0, shared_p=100.0)
        ctx.ffn_routed_num_p = None
        ctx.ffn_shared_num_p = None
        result = EvalBody.num_params_layer(ccfg, ctx)
        non_exp, routed, shared = result
        self.assertAlmostEqual(non_exp, 150.0)
        self.assertAlmostEqual(routed, 0.0)
        self.assertAlmostEqual(shared, 0.0)


class TestStatPLayer(unittest.TestCase):
    """Test EvalBody.stat_p_layer model parameter memory."""

    def test_dense_stat_p(self):
        """BD-P01: Dense FFN stat_p = (attn+norm+ffn) * bytes_p / shard_p_os_non_exp_partial."""
        ccfg = _make_ccfg(n_exp=1, bytes_p=2, shard_p_os_non_exp_partial=2.0)
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, ffn_p=200.0)
        result = EvalBody.stat_p_layer(ccfg, ctx)
        expected = 350.0 * 2 / 2.0
        self.assertAlmostEqual(result, expected, places=4)

    def test_moe_stat_p_with_sharding(self):
        """BD-P02: MoE stat_p splits non_exp/routed/shared with different sharding."""
        ccfg = _make_ccfg(
            n_exp=8,
            n_shared_exp=1,
            ep=4,
            bytes_p=2,
            shard_p_os_non_exp_partial=2.0,
            shard_p_os_exp=2.0,
            shard_p_os_exp_partial=4.0,
        )
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, routed_p=400.0, shared_p=200.0)
        result = EvalBody.stat_p_layer(ccfg, ctx)
        # non_exp: 150 * 2 / 2 = 150
        # routed: 400/4 * 2 / 2 = 100
        # shared: 200 * 2 / 4 = 100
        expected = 150.0 + 100.0 + 100.0
        self.assertAlmostEqual(result, expected, places=4)

    def test_moe_stat_p_no_sharding(self):
        """BD-P03: MoE stat_p with all shard factors=1 (no sharding)."""
        ccfg = _make_ccfg(
            n_exp=8,
            n_shared_exp=1,
            ep=1,
            bytes_p=2,
            shard_p_os_non_exp_partial=1.0,
            shard_p_os_exp=1.0,
            shard_p_os_exp_partial=1.0,
        )
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, routed_p=400.0, shared_p=200.0)
        result = EvalBody.stat_p_layer(ccfg, ctx)
        # non_exp: 150 * 2 / 1 = 300
        # routed: 400/1 * 2 / 1 = 800
        # shared: 200 * 2 / 1 = 400
        expected = 300.0 + 800.0 + 400.0
        self.assertAlmostEqual(result, expected, places=4)


class TestStatOsLayer(unittest.TestCase):
    """Test EvalBody.stat_os_layer optimizer state memory."""

    def test_dense_stat_os(self):
        """BD-O01: Dense FFN optimizer state = params * 2*bytes_os / shard."""
        ccfg = _make_ccfg(n_exp=1, bytes_os=12, shard_p_os_non_exp_partial=2.0)
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, ffn_p=200.0)
        result = EvalBody.stat_os_layer(ccfg, ctx)
        expected = 350.0 * 2 * 12 / 2.0
        self.assertAlmostEqual(result, expected, places=4)

    def test_moe_stat_os_with_sharding(self):
        """BD-O02: MoE optimizer state with EP/partial sharding."""
        ccfg = _make_ccfg(
            n_exp=8,
            n_shared_exp=1,
            ep=4,
            bytes_os=12,
            shard_p_os_non_exp_partial=2.0,
            shard_p_os_exp=2.0,
            shard_p_os_exp_partial=4.0,
        )
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, routed_p=400.0, shared_p=200.0)
        result = EvalBody.stat_os_layer(ccfg, ctx)
        # non_exp: 150 * 2*12 / 2 = 1800
        # routed: 400/4 * 2*12 / 2 = 1200
        # shared: 200 * 2*12 / 4 = 1200
        expected = 1800.0 + 1200.0 + 1200.0
        self.assertAlmostEqual(result, expected, places=4)

    def test_swap_os_returns_zero(self):
        """BD-O03: stat_os_layer returns 0 when swap_os is True."""
        ccfg = _make_ccfg(n_exp=1, bytes_os=12)
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, ffn_p=200.0, swap_os=True)
        result = EvalBody.stat_os_layer(ccfg, ctx)
        self.assertEqual(result, 0)


class TestStatGradLayer(unittest.TestCase):
    """Test EvalBody.stat_grad_layer gradient memory."""

    def test_dense_stat_grad(self):
        """BD-G01: Dense FFN gradient = params * bytes_grad / shard_grad_non_exp."""
        ccfg = _make_ccfg(n_exp=1, bytes_grad=2, shard_grad_non_exp=2.0)
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, ffn_p=200.0)
        result = EvalBody.stat_grad_layer(ccfg, ctx)
        expected = 350.0 * 2 / 2.0
        self.assertAlmostEqual(result, expected, places=4)

    def test_moe_stat_grad_with_sharding(self):
        """BD-G02: MoE gradient with EP/sharded gradients."""
        ccfg = _make_ccfg(
            n_exp=8,
            n_shared_exp=1,
            ep=4,
            bytes_grad=2,
            shard_grad_non_exp=2.0,
            shard_grad_exp=2.0,
            shard_grad_exp_partial=4.0,
        )
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, routed_p=400.0, shared_p=200.0)
        result = EvalBody.stat_grad_layer(ccfg, ctx)
        # non_exp: 150 * 2 / 2 = 150
        # routed: 400/4 * 2 / 2 = 100
        # shared: 200 * 2 / 4 = 100 (ZeRO partial sharding via shard_grad_exp_partial)
        expected = 150.0 + 100.0 + 100.0
        self.assertAlmostEqual(result, expected, places=4)

    def test_moe_stat_grad_uses_shard_grad_exp_partial_not_os(self):
        """BD-G03: shared expert gradient uses shard_grad_exp_partial, not shard_p_os_exp_partial.

        When has_grad_shard=False, shard_grad_exp_partial = t_exp (TP only)
        while shard_p_os_exp_partial may be larger (includes os_max_shard).
        Gradient sharding must NOT depend on optimizer state sharding.
        """
        ccfg = _make_ccfg(
            n_exp=8,
            n_shared_exp=1,
            ep=4,
            bytes_grad=2,
            shard_grad_non_exp=1.0,
            shard_grad_exp=1.0,
            shard_p_os_exp_partial=4.0,
            shard_grad_exp_partial=2.0,
        )
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, routed_p=400.0, shared_p=200.0)
        result = EvalBody.stat_grad_layer(ccfg, ctx)
        # non_exp: 150 * 2 / 1 = 300
        # routed: 400/4 * 2 / 1 = 200
        # shared: 200 * 2 / 2 = 200 (uses shard_grad_exp_partial=2, NOT 4)
        expected = 300.0 + 200.0 + 200.0
        self.assertAlmostEqual(result, expected, places=4)


class TestNumParamsRoutedExpert(unittest.TestCase):
    """Test EvalFFn.num_params_routed_expert with ETP correction."""

    def test_basic_no_etp(self):
        """BD-R01: routed expert params with etp=1 (no TP slicing)."""
        ccfg = _make_ccfg(n_exp=8, h=4096, hff_exp=2048, etp=1)
        result = EvalFFn.num_params_routed_expert(ccfg, None)
        # n_exp * max(n_ffMM, n_ffBMM) * (hff_exp * h + hff_exp) = 8 * 1 * (2048*4096 + 2048)
        expected = 8 * 1 * (2048 * 4096 + 2048)
        self.assertAlmostEqual(result, expected, places=0)

    def test_etp_correction(self):
        """BD-R02: routed expert params with etp>1 uses hff_exp/etp."""
        ccfg_no_etp = _make_ccfg(n_exp=256, h=7168, hff_exp=2048, etp=1)
        ccfg_etp4 = _make_ccfg(n_exp=256, h=7168, hff_exp=2048, etp=4)
        result_no_etp = EvalFFn.num_params_routed_expert(ccfg_no_etp, None)
        result_etp4 = EvalFFn.num_params_routed_expert(ccfg_etp4, None)
        # With etp=4, hff_sliced = 2048/4 = 512, params should be exactly 1/4
        self.assertAlmostEqual(result_etp4, result_no_etp / 4, places=0)

    def test_etp_zero_treated_as_one(self):
        """BD-R03: etp=0 is treated as 1 (max(etp,1) safeguard)."""
        ccfg = _make_ccfg(n_exp=8, h=4096, hff_exp=2048, etp=0)
        result = EvalFFn.num_params_routed_expert(ccfg, None)
        expected = 8 * 1 * (2048 * 4096 + 2048)
        self.assertAlmostEqual(result, expected, places=0)


class TestNumParamsSharedExpert(unittest.TestCase):
    """Test EvalFFn.num_params_shared_expert uses hff (not hff_exp)."""

    def test_shared_uses_hff_not_hff_exp(self):
        """BD-S01: shared expert uses ccfg.hff, NOT ccfg.hff_exp.

        DeepSeek-V3 has hff_exp=2048 (routed) but hff=18432 (shared).
        Using hff_exp for shared would severely underestimate.
        """
        ccfg = _make_ccfg(n_exp=256, n_shared_exp=1, h=7168, hff=18432, hff_exp=2048)
        result = EvalFFn.num_params_shared_expert(ccfg, None)
        # Should use hff=18432, not hff_exp=2048
        expected_with_hff = 1 * 1 * (18432 * 7168 + 18432)
        wrong_with_hff_exp = 1 * 1 * (2048 * 7168 + 2048)
        self.assertAlmostEqual(result, expected_with_hff, places=0)
        self.assertNotAlmostEqual(result, wrong_with_hff_exp, places=0)

    def test_shared_no_etp(self):
        """BD-S02: shared expert is NOT affected by etp (no TP slicing)."""
        ccfg = _make_ccfg(n_shared_exp=1, h=4096, hff=14336, etp=4)
        result = EvalFFn.num_params_shared_expert(ccfg, None)
        # etp should not affect shared expert — always uses full hff
        expected = 1 * 1 * (14336 * 4096 + 14336)
        self.assertAlmostEqual(result, expected, places=0)

    def test_backward_compat_no_shared(self):
        """BD-S03: when hff_exp=hff and no shared expert, routed+shared = old num_params_ffn."""
        ccfg = _make_ccfg(n_exp=8, n_shared_exp=0, h=4096, hff=14336, hff_exp=14336, etp=1)
        routed = EvalFFn.num_params_routed_expert(ccfg, None)
        shared = EvalFFn.num_params_shared_expert(ccfg, None)
        old = EvalFFn.num_params_ffn(ccfg, None)
        self.assertAlmostEqual(routed + shared, old, places=0)


class TestConfigOptimizerShard(unittest.TestCase):
    """Test config_optimizer_shard has_op guard on shard_p_os_exp.

    Verifies that when has_op=False, d_exp is NOT used as a sharding factor
    for expert optimizer state, preventing memory underestimation.
    """

    @staticmethod
    def _make_parser_ccfg(
        n_exp=8, d_exp=4, cp=1, t_exp=1, ep=2,
        has_op=True, has_grad_shard=True, os_max_shard=1,
    ):
        """Create a mock _CostModVar for parser-level shard tests."""
        from hyper_parallel.auto_parallel.sapp_nd.nd.common._cost_model_variables import _CostModVar
        ccfg = MagicMock(spec=_CostModVar)
        ccfg.n_exp = n_exp
        ccfg.d_exp = d_exp
        ccfg.cp = cp
        ccfg.t_exp = t_exp
        ccfg.ep = ep
        ccfg.has_op = has_op
        ccfg.has_grad_shard = has_grad_shard
        ccfg.os_max_shard = os_max_shard
        return ccfg

    def test_has_op_true_uses_d_exp(self):
        """BD-H01: has_op=True => shard_p_os_exp = d_exp * cp * t_exp."""
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import _CostModelParser
        ccfg = self._make_parser_ccfg(d_exp=4, cp=2, t_exp=1, has_op=True)
        _CostModelParser.config_optimizer_shard(None, ccfg)
        expected = 4 * 2 * 1  # d_exp * cp * t_exp
        self.assertEqual(ccfg.shard_p_os_exp, expected)

    def test_has_op_false_ignores_d_exp(self):
        """BD-H02: has_op=False => shard_p_os_exp = 1 * cp * t_exp (d_exp bypassed).

        Without the guard, d_exp=4 would produce shard_p_os_exp=8, causing
        expert param/OS/grad memory to be underestimated by 4x.
        """
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import _CostModelParser
        ccfg = self._make_parser_ccfg(d_exp=4, cp=2, t_exp=1, has_op=False)
        _CostModelParser.config_optimizer_shard(None, ccfg)
        expected = 1 * 2 * 1  # (d_exp if has_op else 1) * cp * t_exp
        self.assertEqual(ccfg.shard_p_os_exp, expected)

    def test_has_op_false_non_exp_symmetric(self):
        """BD-H03: has_op guard is symmetric between non-exp and expert paths.

        Non-exp: shard_p_os_non_exp = (d if has_op else 1) * cp * t
        Expert:  shard_p_os_exp     = (d_exp if has_op else 1) * cp * t_exp
        Both bypass the DP sharding factor when has_op=False.
        """
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import _CostModelParser
        ccfg = self._make_parser_ccfg(d_exp=4, cp=2, t_exp=1, has_op=False)
        ccfg.d = 4
        ccfg.t = 1
        _CostModelParser.config_optimizer_shard(None, ccfg)
        # Non-exp: (d if has_op else 1) * cp * t = 1 * 2 * 1 = 2
        # Expert:  (d_exp if has_op else 1) * cp * t_exp = 1 * 2 * 1 = 2
        self.assertEqual(ccfg.shard_p_os_non_exp, 2)
        self.assertEqual(ccfg.shard_p_os_exp, 2)


if __name__ == "__main__":
    unittest.main()
