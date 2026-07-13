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
"""Unit tests for comm.py: DP/TP/CP/EP communication volume estimation.

Test IDs:
  CM-D01: dp_comm_non_exp with ZeRO level 2
  CM-D02: dp_comm_non_exp with ZeRO level 3
  CM-D03: dp_comm_exp with 3-tuple from num_p and ZeRO level 2
  CM-D04: dp_comm_exp with 3-tuple from num_p and ZeRO level 3
  CM-D06: dp_comm_layer sums non_exp and exp
  CM-E01: ep_comm_layer_balanced basic formula with (ep-1)/ep correction
  CM-E02: ep_comm_layer_balanced scales with mb and n_chosen_exp
  CM-E03: ep_comm_layer_balanced returns 0 when EP=1
  CM-E04: ep_comm_layer dispatches to balanced when tokens_per_expert is None
  CM-E05: ep_comm_layer dispatches to imbalanced when tokens_per_expert is set
  CM-E06: ep_comm_layer_imbalanced with tokens_per_expert
  CM-E07: ep_comm_layer_imbalanced falls back to balanced when n_exp not divisible by ep
  CM-E08: ep_comm_layer_imbalanced falls back to balanced when tokens_per_expert empty
  CM-E09: ep_comm_layer_imbalanced reduces to balanced under uniform distribution
  CM-T01: tp_comm_exp MoE formula uses hff_exp for routed, hff for shared
  CM-T02: tp_comm_exp dense formula uses s*b*hff*mb
  CM-C01: Ring CP p=1 comm is 3x p>1 (rec_factor gate by int(ccfg.p == 1))
  CM-C02: Ulysses CP p=1 comm is 2x p>1 (rec_factor gate by int(ccfg.p == 1))
  CM-C03: When rec_coeff=0 (SEL_REC_LAYER + gather=False), p has no effect
  CM-C04: Ring CP exact formula at p=1 (rec_factor=1, coefficient=1.5)
  CM-C05: Ulysses CP exact formula at p=1 (rec_factor=1, coefficient=1.0)
  CM-C04b: Ring CP exact formula at p>1 (rec_factor=0, coefficient=0.5)
  CM-C05b: Ulysses CP exact formula at p>1 (rec_factor=0, coefficient=0.5)
"""
import os
import unittest
from unittest.mock import MagicMock

from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.comm import EvalLayerComm


def _make_ccfg(
    n_exp=8,
    n_shared_exp=1,
    h=4096,
    hff=14336,
    hff_exp=14336,
    ep=4,
    tp=1,
    cp=1,
    p=1,
    t_exp=1,
    comm_d_non_exp=2,
    comm_d_exp=2,
    comm_ep=1.0,
    comm_t=1.0,
    comm_cp=1.0,
    n_chosen_exp=2,
    s=1024,
    b=4,
    bytes_compute=2,
    n_gather=2,
    n_attMM=2,  # pylint: disable=invalid-name
    n_ffMM=1,  # pylint: disable=invalid-name
    n_ffBMM=0,  # pylint: disable=invalid-name
    rec_op=None,
    cp_algo="colossalai_cp",
    tokens_per_expert=None,
):
    """Create a mock CostModelConfig for comm tests."""
    ccfg = MagicMock()
    ccfg.n_exp = n_exp
    ccfg.n_shared_exp = n_shared_exp
    ccfg.h = h
    ccfg.hff = hff
    ccfg.hff_exp = hff_exp
    ccfg.ep = ep
    ccfg.t = tp
    ccfg.cp = cp
    ccfg.p = p
    ccfg.t_exp = t_exp
    ccfg.comm_d_non_exp = comm_d_non_exp
    ccfg.comm_d_exp = comm_d_exp
    ccfg.comm_ep = comm_ep
    ccfg.comm_t = comm_t
    ccfg.comm_cp = comm_cp
    ccfg.n_chosen_exp = n_chosen_exp
    ccfg.s = s
    ccfg.b = b
    ccfg.bytes_compute = bytes_compute
    ccfg.n_gather = n_gather
    ccfg.n_attMM = n_attMM
    ccfg.n_ffMM = n_ffMM
    ccfg.n_ffBMM = n_ffBMM
    ccfg.cp_algo = cp_algo
    ccfg.tokens_per_expert = tokens_per_expert

    # rec_op mock
    if rec_op is None:
        rec_op = MagicMock()
        rec_op.gather = False
    ccfg.rec_op = rec_op
    return ccfg


def _make_ctx(num_p_result=(150.0, 400.0, 200.0), current_node=None):
    """Create a mock Context for comm tests.

    Args:
        num_p_result: What ctx.eval.num_p(ccfg, ctx) returns.
            Tuple for MoE, scalar for dense.
        current_node: Mock LayerType for TP/CP rec_layer checks.
    """
    ctx = MagicMock()
    ctx.eval = MagicMock()
    ctx.eval.num_p = lambda c, x: num_p_result
    ctx.current_node = current_node
    return ctx


class TestDpCommNonExp(unittest.TestCase):
    """Test dp_comm_non_exp ZeRO level branching."""

    def test_zero_level2(self):
        """CM-D01: ZeRO level 2 non-exp comm = 2 * non_exp / (cp * t)."""
        ccfg = _make_ccfg(tp=2, cp=1, comm_d_non_exp=2)
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.dp_comm_non_exp(ccfg, ctx)
        # Level 2: non_exp/(cp*t) + non_exp/t = 150/(1*2) + 150/2 = 75 + 75 = 150
        expected = 150.0 / (1 * 2) + 150.0 / 2
        self.assertAlmostEqual(result, expected, places=4)

    def test_zero_level3(self):
        """CM-D02: ZeRO level 3 non-exp comm = non_exp / t."""
        ccfg = _make_ccfg(tp=2, cp=1, comm_d_non_exp=3)
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.dp_comm_non_exp(ccfg, ctx)
        expected = 150.0 / 2
        self.assertAlmostEqual(result, expected, places=4)


class TestDpCommExp(unittest.TestCase):
    """Test dp_comm_exp with 3-tuple and ZeRO level branching."""

    def test_zero_level2_with_tuple(self):
        """CM-D03: ZeRO level 2 exp comm with 3-tuple uses routed+shared."""
        ccfg = _make_ccfg(ep=4, tp=1, t_exp=1, cp=1, comm_d_exp=2)
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.dp_comm_exp(ccfg, ctx)
        exp_param_size = 400.0 + 200.0
        # Level 2: exp_param/(cp*t_exp*ep) + exp_param/max(ep, t_exp)
        expected = exp_param_size / (1 * 1 * 4) + exp_param_size / max(4, 1)
        self.assertAlmostEqual(result, expected, places=4)

    def test_zero_level3_with_tuple(self):
        """CM-D04: ZeRO level 3 exp comm = exp_param / (cp*t_exp*ep)."""
        ccfg = _make_ccfg(ep=4, tp=1, t_exp=1, cp=1, comm_d_exp=3)
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.dp_comm_exp(ccfg, ctx)
        exp_param_size = 400.0 + 200.0
        expected = exp_param_size / (1 * 1 * 4)
        self.assertAlmostEqual(result, expected, places=4)


class TestDpCommLayer(unittest.TestCase):
    """Test dp_comm_layer sums non_exp and exp."""

    def test_layer_sums_both(self):
        """CM-D06: dp_comm_layer = dp_comm_non_exp + dp_comm_exp."""
        ccfg = _make_ccfg(tp=2, ep=4, cp=1, comm_d_non_exp=2, comm_d_exp=2)
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.dp_comm_layer(ccfg, ctx)
        non_exp = EvalLayerComm.dp_comm_non_exp(ccfg, ctx)
        exp = EvalLayerComm.dp_comm_exp(ccfg, ctx)
        self.assertAlmostEqual(result, non_exp + exp, places=4)


class TestEpCommLayerBalanced(unittest.TestCase):
    """Test ep_comm_layer_balanced with (ep-1)/ep correction factor."""

    def test_basic_formula(self):
        """CM-E01: balanced EP comm = 2 * T_cross * h * bytes_compute, where T_cross = T_local*(ep-1)/ep."""
        ccfg = _make_ccfg(
            n_chosen_exp=2, s=1024, b=4, h=4096,
            cp=1, tp=1, ep=4, comm_ep=1.0, bytes_compute=2,
        )
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)
        t_local = mb * 2 * 1024 * 4 / (1 * 1)
        t_cross = t_local * (4 - 1) / 4
        expected = t_cross * 4096 * 2 * 2  # *2 for dispatch+combine
        self.assertAlmostEqual(result, expected, places=4)

    def test_scales_with_mb(self):
        """CM-E02: balanced EP comm scales linearly with mb and n_chosen_exp."""
        ccfg = _make_ccfg(
            n_chosen_exp=4, s=512, b=2, h=2048,
            cp=2, tp=2, ep=8, comm_ep=1.0, bytes_compute=2,
        )
        ctx = _make_ctx()
        mb = 3
        result = EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)
        t_local = mb * 4 * 512 * 2 / 2  # /cp only, /tp removed
        t_cross = t_local * (8 - 1) / 8
        expected = t_cross * 2048 * 2 * 2
        self.assertAlmostEqual(result, expected, places=4)

    def test_ep1_returns_zero(self):
        """CM-E03: balanced EP comm returns 0 when EP=1."""
        ccfg = _make_ccfg(ep=1, comm_ep=1.0)
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)
        self.assertEqual(result, 0)

    def test_comm_ep_zero_returns_zero(self):
        """CM-E03b: balanced EP comm returns 0 when comm_ep=0 (scalar multiplier)."""
        ccfg = _make_ccfg(ep=4, comm_ep=0)
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)
        self.assertEqual(result, 0)

    def test_comm_ep_scalar_multiplier(self):
        """CM-E03c: comm_ep is a scalar multiplier (consistent with comm_t/comm_cp)."""
        ccfg_half = _make_ccfg(
            n_chosen_exp=2, s=1024, b=4, h=4096,
            cp=1, tp=1, ep=4, comm_ep=0.5, bytes_compute=2,
        )
        ccfg_full = _make_ccfg(
            n_chosen_exp=2, s=1024, b=4, h=4096,
            cp=1, tp=1, ep=4, comm_ep=1.0, bytes_compute=2,
        )
        ctx = _make_ctx()
        mb = 1
        result_half = EvalLayerComm.ep_comm_layer_balanced(ccfg_half, ctx, mb)
        result_full = EvalLayerComm.ep_comm_layer_balanced(ccfg_full, ctx, mb)
        self.assertAlmostEqual(result_half, result_full * 0.5, places=4)

    def test_correction_factor(self):
        """CM-E01b: verify (ep-1)/ep correction vs old formula (which assumed all cross)."""
        ccfg = _make_ccfg(
            n_chosen_exp=8, s=4096, b=1, h=7168,
            cp=1, tp=1, ep=4, comm_ep=1.0, bytes_compute=2,
        )
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)
        # Old formula (no correction): 2 * mb * n_chosen_exp * s * b * h / (cp * t) * bytes_compute
        old_result = 2 * 1 * 8 * 4096 * 1 * 7168 / (1 * 1) * 2
        correction = (4 - 1) / 4  # = 0.75
        expected = old_result * correction
        self.assertAlmostEqual(result, expected, places=4)


class TestEpCommLayerDispatch(unittest.TestCase):
    """Test ep_comm_layer dispatches to balanced or imbalanced."""

    def test_dispatches_balanced_when_no_tokens_per_expert(self):
        """CM-E04: ep_comm_layer calls balanced when tokens_per_expert is None."""
        ccfg = _make_ccfg(ep=4, comm_ep=1.0, tokens_per_expert=None)
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.ep_comm_layer(ccfg, ctx, mb)
        balanced = EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)
        self.assertAlmostEqual(result, balanced, places=4)

    def test_dispatches_imbalanced_when_tokens_per_expert_set(self):
        """CM-E05: ep_comm_layer calls imbalanced when tokens_per_expert is set."""
        n_exp = 8
        ep = 4
        # Global uniform tokens
        tokens = [1024] * n_exp
        ccfg = _make_ccfg(n_exp=n_exp, ep=ep, comm_ep=1.0, tokens_per_expert=tokens)
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.ep_comm_layer(ccfg, ctx, mb)
        imbalanced = EvalLayerComm.ep_comm_layer_imbalanced(ccfg, ctx, mb)
        self.assertAlmostEqual(result, imbalanced, places=4)


class TestEpCommLayerImbalanced(unittest.TestCase):
    """Test ep_comm_layer_imbalanced with token distribution."""

    def test_basic_imbalanced(self):
        """CM-E06: imbalanced comm uses max(rank_tokens) with (ep-1)/ep normalization."""
        n_exp = 8
        ep = 4
        h = 4096
        bytes_compute = 2
        # Global token counts per expert (all EP ranks combined)
        tokens = [3000, 2500, 1000, 500, 500, 300, 200, 100]
        ccfg = _make_ccfg(
            n_exp=n_exp, ep=ep, h=h, bytes_compute=bytes_compute,
            comm_ep=1.0, tokens_per_expert=tokens,
        )
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.ep_comm_layer_imbalanced(ccfg, ctx, mb)
        # experts_per_rank = 2
        # Rank 0: experts 0,1 -> 3000+2500=5500 (max)
        # Rank 1: experts 2,3 -> 1000+500=1500
        # Rank 2: experts 4,5 -> 500+300=800
        # Rank 3: experts 6,7 -> 200+100=300
        max_inbound = 5500
        t_cross = max_inbound * mb * (ep - 1) / ep
        expected = t_cross * h * bytes_compute * 2 * 1.0  # dispatch+combine, comm_ep=1.0
        self.assertAlmostEqual(result, expected, places=4)

    def test_fallback_on_non_divisible(self):
        """CM-E07: imbalanced falls back to balanced when n_exp not divisible by ep."""
        ccfg = _make_ccfg(
            n_exp=3, ep=2, h=4096, s=1024, b=4, n_chosen_exp=2,
            cp=1, tp=1, comm_ep=1.0, tokens_per_expert=[100, 200, 300],
        )
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.ep_comm_layer_imbalanced(ccfg, ctx, mb)
        balanced = EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)
        self.assertAlmostEqual(result, balanced, places=4)

    def test_fallback_to_balanced_on_empty(self):
        """CM-E08: imbalanced falls back to balanced when tokens_per_expert is empty."""
        ccfg = _make_ccfg(ep=4, comm_ep=1.0, tokens_per_expert=[])
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.ep_comm_layer_imbalanced(ccfg, ctx, mb)
        balanced = EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)
        self.assertAlmostEqual(result, balanced, places=4)

    def test_uniform_equals_balanced(self):
        """CM-E09: imbalanced reduces to balanced under uniform token distribution."""
        n_exp = 8
        ep = 4
        n_chosen_exp = 2
        s = 1024
        b = 4
        h = 4096
        bytes_compute = 2
        # Global: each expert gets t_local * ep / n_exp tokens (all EP ranks combined)
        # where t_local = n_chosen_exp * s * b / (cp * t) is per-rank token count
        t_local = n_chosen_exp * s * b / (1 * 1)
        token_per_exp_global = t_local * ep / n_exp
        tokens = [token_per_exp_global] * n_exp
        ccfg_bal = _make_ccfg(
            n_exp=n_exp, ep=ep, h=h, s=s, b=b, n_chosen_exp=n_chosen_exp,
            cp=1, tp=1, bytes_compute=bytes_compute, comm_ep=1.0,
        )
        ccfg_imbal = _make_ccfg(
            n_exp=n_exp, ep=ep, h=h, s=s, b=b, n_chosen_exp=n_chosen_exp,
            cp=1, tp=1, bytes_compute=bytes_compute, comm_ep=1.0,
            tokens_per_expert=tokens,
        )
        ctx = _make_ctx()
        mb = 1
        vol_bal = EvalLayerComm.ep_comm_layer_balanced(ccfg_bal, ctx, mb)
        vol_imbal = EvalLayerComm.ep_comm_layer_imbalanced(ccfg_imbal, ctx, mb)
        self.assertAlmostEqual(vol_imbal, vol_bal, places=4)

    def test_ep1_returns_zero(self):
        """CM-E06b: imbalanced returns 0 when EP=1."""
        ccfg = _make_ccfg(ep=1, comm_ep=1.0, tokens_per_expert=[100])
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.ep_comm_layer_imbalanced(ccfg, ctx, mb)
        self.assertEqual(result, 0)

    def test_comm_ep_zero_returns_zero(self):
        """CM-E06d: imbalanced returns 0 when comm_ep=0 (scalar multiplier)."""
        ccfg = _make_ccfg(ep=4, comm_ep=0, tokens_per_expert=[100] * 8)
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.ep_comm_layer_imbalanced(ccfg, ctx, mb)
        self.assertEqual(result, 0)

    def test_imbalanced_greater_than_balanced(self):
        """CM-E06c: imbalanced comm > balanced comm for skewed distribution."""
        n_exp = 8
        ep = 4
        # Global skewed tokens: rank 0 (experts 0,1) gets much more than its fair share
        ccfg_bal = _make_ccfg(
            n_exp=n_exp, ep=ep, h=4096, s=1024, b=4, n_chosen_exp=2,
            cp=1, tp=1, bytes_compute=2, comm_ep=1.0,
        )
        ccfg_imbal = _make_ccfg(
            n_exp=n_exp, ep=ep, h=4096, s=1024, b=4, n_chosen_exp=2,
            cp=1, tp=1, bytes_compute=2, comm_ep=1.0,
            tokens_per_expert=[6000, 5000, 1000, 500, 500, 300, 200, 100],
        )
        ctx = _make_ctx()
        mb = 1
        vol_bal = EvalLayerComm.ep_comm_layer_balanced(ccfg_bal, ctx, mb)
        vol_imbal = EvalLayerComm.ep_comm_layer_imbalanced(ccfg_imbal, ctx, mb)
        self.assertGreater(vol_imbal, vol_bal)


class TestTpCommExp(unittest.TestCase):
    """Test tp_comm_exp MoE vs dense formula branching."""

    def test_moe_formula(self):
        """CM-T01: MoE TP comm uses hff_exp for routed, hff for shared."""
        ccfg = _make_ccfg(
            n_exp=8, n_shared_exp=1, ep=2,
            h=4096, hff=14336, hff_exp=2048, bytes_compute=2,
            n_ffMM=1, n_gather=2,
        )
        ctx = _make_ctx()
        mb = 1
        result = EvalLayerComm.tp_comm_exp(ccfg, ctx, mb)
        # Routed: n_exp/ep * hff_exp = 4 * 2048 = 8192
        # Shared: n_shared_exp * hff = 1 * 14336 = 14336
        routed_comm = 8 / 2 * 2048
        shared_comm = 1 * 14336
        inner = 0.25 * 2 * 4096 * 2 * 1 * (routed_comm + shared_comm)
        rec_layer = ctx.current_node == MagicMock()  # False
        rec_factor = int(not rec_layer) | False  # 1
        expected = rec_factor * 1.0 * inner / 1
        self.assertAlmostEqual(result, expected, places=0)

    def test_dense_formula(self):
        """CM-T02: Dense TP comm uses s*b*hff*mb base."""
        ccfg = _make_ccfg(
            n_exp=1, n_shared_exp=0,
            h=4096, hff=14336, s=1024, b=4,
            n_gather=2, comm_t=1.0, cp=1,
        )
        ctx = _make_ctx()
        mb = 2
        result = EvalLayerComm.tp_comm_exp(ccfg, ctx, mb)
        inner = 0.25 * 2 * 1024 * 4 * 14336 * 2
        expected = 1.0 * 1.0 * inner / 1.0
        self.assertAlmostEqual(result, expected, places=0)


class TestCpCommNonExpRecFactor(unittest.TestCase):
    """Test cp_comm_non_exp rec_factor gating by int(ccfg.p == 1) [HYPOTHESIS].

    The [HYPOTHESIS] in comm.py:135-137 assumes that when PP>1, the pipeline
    bubble fully hides CP communication, so rec_factor is zeroed out.
    Tests verify the gating behavior with the same mock style as DP/TP/EP tests.

    Test IDs:
      CM-C01: Ring CP p=1 comm is 3x p>1 (coefficient 1.5 vs 0.5)
      CM-C02: Ulysses CP p=1 comm is 2x p>1 (coefficient 1.0 vs 0.5)
      CM-C03: When rec_coeff=0 (SEL_REC_LAYER + gather=False), p makes no difference
      CM-C04: Ring CP exact formula verification at p=1
      CM-C05: Ulysses CP exact formula verification at p=1
    """

    def _make_ctx(self, current_node=None):
        """Create a mock Context for CP tests (no eval.num_p needed)."""
        ctx = MagicMock()
        ctx.current_node = current_node
        return ctx

    def test_ring_p1_vs_p2_ratio(self):
        """CM-C01: Ring CP comm with p=1 is 3x the comm with p>1.

        rec_factor = rec_coeff * int(p == 1).
        When p=1: rec_factor = 1, coefficient = 2*0.5*1 + 0.5 = 1.5.
        When p>1: rec_factor = 0, coefficient = 2*0.5*0 + 0.5 = 0.5.
        Ratio = 1.5 / 0.5 = 3.
        """
        ccfg_p1 = _make_ccfg(p=1, cp_algo="colossalai_cp")
        ccfg_p2 = _make_ccfg(p=2, cp_algo="colossalai_cp")
        ctx = self._make_ctx()

        comm_p1 = EvalLayerComm.cp_comm_non_exp(ccfg_p1, ctx)
        comm_p2 = EvalLayerComm.cp_comm_non_exp(ccfg_p2, ctx)

        self.assertGreater(comm_p1, 0)
        self.assertGreater(comm_p2, 0)
        self.assertAlmostEqual(comm_p1 / comm_p2, 3.0, places=4)

    def test_ulysses_p1_vs_p2_ratio(self):
        """CM-C02: Ulysses CP comm with p=1 is 2x the comm with p>1.

        When p=1: rec_factor = 1, coefficient = 0.5*1 + 0.5 = 1.0.
        When p>1: rec_factor = 0, coefficient = 0.5*0 + 0.5 = 0.5.
        Ratio = 1.0 / 0.5 = 2.
        """
        ccfg_p1 = _make_ccfg(p=1, cp_algo="ulysses_cp")
        ccfg_p2 = _make_ccfg(p=2, cp_algo="ulysses_cp")
        ctx = self._make_ctx()

        comm_p1 = EvalLayerComm.cp_comm_non_exp(ccfg_p1, ctx)
        comm_p2 = EvalLayerComm.cp_comm_non_exp(ccfg_p2, ctx)

        self.assertGreater(comm_p1, 0)
        self.assertGreater(comm_p2, 0)
        self.assertAlmostEqual(comm_p1 / comm_p2, 2.0, places=4)

    def test_rec_coeff_zero_makes_p_irrelevant(self):
        """CM-C03: When rec_coeff=0, p has no effect on cp_comm_non_exp.

        rec_coeff = int(not rec_layer) | rec_op.gather.
        With rec_layer=True (SEL_REC_LAYER) and gather=False:
        rec_coeff = int(not True) | False = 0 | 0 = 0.
        Then rec_factor = 0 * int(p == 1) = 0 regardless of p.
        """
        ctx = self._make_ctx(current_node=LayerType.SEL_REC_LAYER)

        ccfg_p1 = _make_ccfg(p=1, cp_algo="colossalai_cp")
        ccfg_p1.rec_op = Config({
            'attBMM': 1, 'headCast': 1, 'dropout': 1, 'softmax': 1,
            'normOp': 1, 'gather': 0, 'ffAct': 1,
        })
        ccfg_p2 = _make_ccfg(p=2, cp_algo="colossalai_cp")
        ccfg_p2.rec_op = Config({
            'attBMM': 1, 'headCast': 1, 'dropout': 1, 'softmax': 1,
            'normOp': 1, 'gather': 0, 'ffAct': 1,
        })

        comm_p1 = EvalLayerComm.cp_comm_non_exp(ccfg_p1, ctx)
        comm_p2 = EvalLayerComm.cp_comm_non_exp(ccfg_p2, ctx)

        self.assertAlmostEqual(comm_p1, comm_p2, places=4,
                               msg="When rec_coeff=0, p should not affect cp_comm_non_exp")

    def test_ring_exact_formula_p1(self):
        """CM-C04: Ring CP exact formula at p=1 (rec_factor=1).

        cp_comm = comm_cp * 2 * s * b * ((2*0.5*rec_factor + 0.5) * n_attMM * h) / t
        With rec_factor=1: inner_coeff = 2*0.5*1 + 0.5 = 1.5
        """
        ccfg = _make_ccfg(p=1, cp_algo="colossalai_cp")
        ctx = self._make_ctx()

        result = EvalLayerComm.cp_comm_non_exp(ccfg, ctx)
        expected = (
            ccfg.comm_cp * 2 * ccfg.s * ccfg.b
            * (1.5 * ccfg.n_attMM * ccfg.h)
            / ccfg.t
        )
        self.assertAlmostEqual(result, expected, places=4)

    def test_ulysses_exact_formula_p1(self):
        """CM-C05: Ulysses CP exact formula at p=1 (rec_factor=1).

        cp_comm = comm_cp * 2 * s * b * ((0.5*rec_factor + 0.5) * n_attMM * h) / t
        With rec_factor=1: inner_coeff = 0.5*1 + 0.5 = 1.0
        """
        ccfg = _make_ccfg(p=1, cp_algo="ulysses_cp")
        ctx = self._make_ctx()

        result = EvalLayerComm.cp_comm_non_exp(ccfg, ctx)
        expected = (
            ccfg.comm_cp * 2 * ccfg.s * ccfg.b
            * (1.0 * ccfg.n_attMM * ccfg.h)
            / ccfg.t
        )
        self.assertAlmostEqual(result, expected, places=4)

    def test_ring_exact_formula_p_gt1(self):
        """CM-C04b: Ring CP exact formula at p>1 (rec_factor=0).

        With rec_factor=0: inner_coeff = 2*0.5*0 + 0.5 = 0.5
        """
        ccfg = _make_ccfg(p=4, cp_algo="colossalai_cp")
        ctx = self._make_ctx()

        result = EvalLayerComm.cp_comm_non_exp(ccfg, ctx)
        expected = (
            ccfg.comm_cp * 2 * ccfg.s * ccfg.b
            * (0.5 * ccfg.n_attMM * ccfg.h)
            / ccfg.t
        )
        self.assertAlmostEqual(result, expected, places=4)

    def test_ulysses_exact_formula_p_gt1(self):
        """CM-C05b: Ulysses CP exact formula at p>1 (rec_factor=0).

        With rec_factor=0: inner_coeff = 0.5*0 + 0.5 = 0.5
        """
        ccfg = _make_ccfg(p=4, cp_algo="ulysses_cp")
        ctx = self._make_ctx()

        result = EvalLayerComm.cp_comm_non_exp(ccfg, ctx)
        expected = (
            ccfg.comm_cp * 2 * ccfg.s * ccfg.b
            * (0.5 * ccfg.n_attMM * ccfg.h)
            / ccfg.t
        )
        self.assertAlmostEqual(result, expected, places=4)


class TestNodeCommEvalRepr(unittest.TestCase):
    """Test NodeCommEval __repr__ with _qname for None safety."""

    def test_repr_with_none_ep(self):
        """CM-Q01: NodeCommEval.__repr__ does not crash when ep is None."""
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import NodeCommEval
        comm = NodeCommEval(dp=lambda: 0, tp=lambda: 0, cp=lambda: 0, ep=None)
        # Should not raise AttributeError
        repr_str = repr(comm)
        self.assertIn("None", repr_str)

    def test_repr_with_ep_balanced_none(self):
        """CM-Q02: NodeCommEval.__repr__ handles ep_balanced/ep_imbalanced=None."""
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import NodeCommEval
        comm = NodeCommEval(
            dp=lambda: 0, tp=lambda: 0, cp=lambda: 0, ep=lambda: 0,
            ep_balanced=None, ep_imbalanced=None,
        )
        repr_str = repr(comm)
        self.assertIn("dyn.comm", repr_str)

    def test_qname_with_callable(self):
        """CM-Q03: _qname returns __qualname__ for callables."""
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import _qname
        def my_fun() -> None:
            """Trivial callable used to verify _qname reads __qualname__."""
            return None
        self.assertEqual(_qname(my_fun), my_fun.__qualname__)

    def test_qname_with_none(self):
        """CM-Q04: _qname returns 'None' string for None."""
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import _qname
        self.assertEqual(_qname(None), "None")


if __name__ == "__main__":
    unittest.main()
