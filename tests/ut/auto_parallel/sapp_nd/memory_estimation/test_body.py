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


from hyper_parallel.auto_parallel._model_spec import KindActivations
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.body import EvalBody
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalFFn, EvalAttn, EvalNorm
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import (
    HYPER_SELECTIVE_REC_OP, derive_comm_flags, derive_expert_degrees, derive_optimizer_sharding,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType


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
    ccfg.bytes_p = bytes_p
    ccfg.bytes_os = bytes_os
    # AdamW's two states and no copy of the parameters, as derive gives them.
    ccfg.bytes_optim = 2 * bytes_os
    ccfg.bytes_grad = bytes_grad
    ccfg.shard_p_os_non_exp_partial = shard_p_os_non_exp_partial
    ccfg.shard_p_os_exp = shard_p_os_exp
    ccfg.shard_p_os_exp_partial = shard_p_os_exp_partial
    ccfg.shard_grad_non_exp = shard_grad_non_exp
    ccfg.shard_grad_exp = shard_grad_exp
    ccfg.shard_grad_exp_partial = shard_grad_exp_partial
    # No bias or norm stated: the parameter formulas count their own.
    for name in ("qkv_bias", "o_bias", "mlp_bias", "norm_bias", "layer_norms", "shared_expert_gate"):
        setattr(ccfg, name, None)
    return ccfg


def _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, ffn_p=200.0,
              routed_p=300.0, shared_p=100.0, swap_os=False, router_p=0.0):
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
    ctx.ffn_router_num_p = EvalFFn.num_params_router if router_p == "real" else (lambda c, x: router_p)

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

    def test_a_moe_layers_router_is_a_non_expert_part(self):
        """BD-N02b: a MoE layer's router, a weight per expert over the hidden width, is among its non-expert parts."""
        ccfg = _make_ccfg(n_exp=8, n_shared_exp=1, h=4096)
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, routed_p=300.0, shared_p=100.0, router_p="real")
        non_exp, routed, shared = EvalBody.num_params_layer(ccfg, ctx)
        self.assertEqual((non_exp, routed, shared), (150.0 + 4096 * 8, 300.0, 100.0))

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


class TestReducedGradLayer(unittest.TestCase):
    """Test EvalBody.reduced_grad_layer, the gradients FSDP reduce-scatters."""

    def test_fsdp_reduces_what_it_shards(self):
        """BD-G04: whole and sharded gradients of the parts a data-parallel rank shards.

        Experts under EP 4 kept whole on each rank (no expert shard) are not
        reduce-scattered; the other parts, sharded over 4, are.
        """
        ccfg = _make_ccfg(
            n_exp=8,
            n_shared_exp=1,
            ep=4,
            bytes_grad=2,
            shard_grad_non_exp=4.0,
            shard_grad_exp=1.0,
            shard_grad_exp_partial=4.0,
        )
        ccfg.t, ccfg.t_exp = 1, 1
        ctx = _make_ctx(ccfg, attn_p=100.0, norm_p=50.0, routed_p=400.0, shared_p=200.0)
        whole, sharded = EvalBody.reduced_grad_layer(ccfg, ctx)
        # non_exp: 150 * 2 whole, / 4 sharded; shared: 200 * 2 whole, / 4 sharded
        self.assertAlmostEqual(whole, (150.0 + 200.0) * 2, places=4)
        self.assertAlmostEqual(sharded, (150.0 + 200.0) * 2 / 4, places=4)


class TestNumParamsRoutedExpert(unittest.TestCase):
    """Test EvalFFn.num_params_routed_expert with ETP correction."""

    def test_basic_no_etp(self):
        """BD-R01: routed expert params with etp=1 (no TP slicing)."""
        ccfg = _make_ccfg(n_exp=8, h=4096, hff_exp=2048, etp=1)
        result = EvalFFn.num_params_routed_expert(ccfg, None)
        # n_exp * n_ffMM * (hff_exp * h + hff_exp) = 8 * 1 * (2048*4096 + 2048)
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
    """Test EvalFFn.num_params_shared_expert prices each shared expert at the routed width."""

    def test_shared_uses_hff_exp_not_hff(self):
        """BD-S01: each shared expert is hff_exp wide, whatever the dense layers' hff.

        DeepSeek-V3's dense layers are 18432 wide and its one shared expert
        2048, as wide as a routed one; Qwen2-57B-A14B states its 20480 wide
        shared expert as eight of 2560.
        """
        ccfg = _make_ccfg(n_exp=256, n_shared_exp=1, h=7168, hff=18432, hff_exp=2048)
        result = EvalFFn.num_params_shared_expert(ccfg, None)
        self.assertAlmostEqual(result, 1 * 1 * (2048 * 7168 + 2048), places=0)

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


class TestLayerActiv(unittest.TestCase):
    """Test EvalBody.layer_activ (no recompute / select recompute)."""

    def _make_ctx(self, qkv=10.0, score=20.0, proj=30.0, ffn=100.0,
                  moe=200.0, norm=5.0):
        ctx = MagicMock()
        ctx.attn_qkv_activ = lambda c, x: qkv
        ctx.attn_score_activ = lambda c, x: score
        ctx.attn_proj_activ = lambda c, x: proj
        ctx.ffn_activ = lambda c, x: ffn
        ctx.ffn_moe_activ = lambda c, x: moe
        ctx.norm_activ = lambda c, x: norm
        return ctx

    def test_dense_layer_activ(self):
        """BD-A01: Dense (n_exp=1) layer_activ = attn + ffn + norm."""
        ccfg = _make_ccfg(n_exp=1)
        ctx = self._make_ctx()
        result = EvalBody.layer_activ(ccfg, ctx)
        expected = (10 + 20 + 30) + 100 + 5
        self.assertAlmostEqual(result, expected, places=4)

    def test_moe_layer_activ(self):
        """BD-A02: MoE (n_exp>1) layer_activ = attn + moe + norm."""
        ccfg = _make_ccfg(n_exp=8)
        ctx = self._make_ctx()
        result = EvalBody.layer_activ(ccfg, ctx)
        expected = (10 + 20 + 30) + 200 + 5
        self.assertAlmostEqual(result, expected, places=4)


class TestCensusActiv(unittest.TestCase):
    """Test EvalBody.layer_activ on a layer whose kind has a census record."""

    RECORD = KindActivations(saved=100.0, saved_tp=300.0, working=150.0, working_tp=330.0, seq_length=4096)

    @staticmethod
    def _ccfg(record, sp=2, cp=1):
        """A MoE layer at TP 2, micro-batch 2 of 4096 tokens, priced with *record*."""
        ccfg = _make_ccfg(n_exp=8)
        ccfg.kind_activations = record
        ccfg.s, ccfg.b, ccfg.t, ccfg.sp, ccfg.cp = 4096, 2, 2, sp, cp
        return ccfg

    @staticmethod
    def _ctx(node=LayerType.NOT_REC_LAYER):
        """A layer of *node* at micro factor 3, between its passes."""
        ctx = MagicMock()
        ctx.current_node, ctx.micro_factor = node, 3
        ctx.working_set, ctx.working_on_saved = 0, False
        return ctx

    def test_a_layer_keeps_what_its_census_states(self):
        """
        Feature: EvalBody.census_activ, between a layer's passes.
        Description: The layer with sequence parallelism, without it, and
            at CP 2.
        Expectation: The census's bytes per token for the micro-batches in
            flight: TP splits one part, sequence parallelism the other, and
            CP the tokens.
        """
        tokens = 3 * 4096 * 2
        self.assertEqual(EvalBody.layer_activ(self._ccfg(self.RECORD), self._ctx()), tokens * (50 + 150))
        self.assertEqual(EvalBody.layer_activ(self._ccfg(self.RECORD, sp=1), self._ctx()), tokens * (100 + 150))
        self.assertEqual(EvalBody.layer_activ(self._ccfg(self.RECORD, cp=2), self._ctx()), tokens / 2 * (50 + 150))

    def test_gathered_keys_and_values_stay_whole(self):
        """
        Feature: EvalBody.census_activ under context parallelism.
        Description: The layer at CP 2 under colossalai CP, with 2 key heads
            64 wide.
        Expectation: A census counts a rank's share of the sequence;
            colossalai CP keeps the other half's keys and values too, split
            over TP.
        """
        ccfg = self._ccfg(self.RECORD, cp=2)
        ccfg.cp_algo, ccfg.n_linrec, ccfg.n_kv, ccfg.dh, ccfg.bytes_compute = "colossalai_cp", 0, 2, 64, 2
        kept = 100 / 2 + 300 / 2 + 2 * 2 * 64 * 2 / 2
        self.assertEqual(EvalBody.layer_activ(ccfg, self._ctx()), 3 * 4096 * 2 / 2 * kept)

    def test_a_selective_layer_keeps_what_hyperparallels_policy_saves(self):
        """
        Feature: EvalBody.census_activ for a selective layer.
        Description: A selective layer whose kind's record states what it
            keeps under HyperParallel's selective checkpointing, with that
            policy's switches.
        Expectation: The record's selective bytes, split as the rest.
        """
        ccfg = self._ccfg(KindActivations(100.0, 300.0, 150.0, 330.0, 4096, selective=20.0, selective_tp=60.0))
        ccfg.rec_op = Config(dict(HYPER_SELECTIVE_REC_OP))
        self.assertEqual(EvalBody.layer_activ(ccfg, self._ctx(LayerType.SEL_REC_LAYER)), 3 * 4096 * 2 * (10 + 30))

    # A record per op: the ops' bytes sum to what the plain layer keeps.
    BY_OP = KindActivations(100.0, 300.0, 150.0, 330.0, 4096,
                            ops={"attMM": 40.0, "normOp": 30.0, "other": 30.0},
                            ops_tp={"attMM": 100.0, "attBMM": 60.0, "normOp": 80.0, "ffAct": 60.0})

    @staticmethod
    def _switches(**recompute):
        """Every switch keeping its op, but those *recompute* sets."""
        return Config(dict(dict.fromkeys(HYPER_SELECTIVE_REC_OP, 1), **recompute))

    def test_a_selective_layer_keeps_the_ops_its_switches_keep(self):
        """
        Feature: EvalBody.census_kept, a census's records per op.
        Description: A selective layer of a kind whose record states what
            it keeps for each op, recomputing the norms and the activation.
        Expectation: What the record states for the ops the switches keep,
            split as the plain layer's, the ops no switch names and the
            census's other always kept.
        """
        ccfg = self._ccfg(self.BY_OP)
        ccfg.rec_op = self._switches(normOp=0, ffAct=0)
        self.assertEqual(EvalBody.layer_activ(ccfg, self._ctx(LayerType.SEL_REC_LAYER)), 3 * 4096 * 2 * (35 + 80))


class TestFullrecLayerActiv(unittest.TestCase):
    """Test EvalBody.fullrec_layer_activ (full recompute)."""

    def test_basic_formula(self):
        """BD-F01: fullrec_layer_activ = micro_factor * bytes_compute * s * b * h / shard_recompute_input."""
        ccfg = _make_ccfg(h=4096)
        ccfg.s = 1024
        ccfg.b = 4
        ccfg.bytes_compute = 2
        ccfg.shard_recompute_input = 1.0
        ctx = MagicMock()
        ctx.micro_factor = 3
        result = EvalBody.fullrec_layer_activ(ccfg, ctx)
        expected = 3 * 2 * 1024 * 4 * 4096 / 1.0
        self.assertAlmostEqual(result, expected, places=4)

    def test_shard_recompute_input_divides(self):
        """BD-F02: shard_recompute_input > 1 reduces activation memory."""
        ccfg = _make_ccfg(h=4096)
        ccfg.s = 1024
        ccfg.b = 4
        ccfg.bytes_compute = 2
        ccfg.shard_recompute_input = 2.0
        ctx = MagicMock()
        ctx.micro_factor = 3
        result = EvalBody.fullrec_layer_activ(ccfg, ctx)
        expected = 3 * 2 * 1024 * 4 * 4096 / 2.0
        self.assertAlmostEqual(result, expected, places=4)


class TestFullrecLayerActivGradclip(unittest.TestCase):
    """Test EvalBody.fullrec_layer_activ_gradclip (gradient clipping)."""

    def _make_ccfg_gradclip(self, has_clip=True, ep=1, **kwargs):
        ccfg = _make_ccfg(ep=ep, **kwargs)
        ccfg.has_clip = has_clip
        ccfg.bytes_compute = 2
        ccfg.s = 1024
        ccfg.b = 4
        ccfg.shard_recompute_input = 1.0
        return ccfg

    def _make_ctx_gradclip(self, non_exp=100.0, routed=0.0, shared=0.0,
                           dp_comm=0.0):
        ctx = MagicMock()
        ctx.micro_factor = 1
        ctx.eval = MagicMock()
        ctx.eval.num_p = lambda c, x: (non_exp, routed, shared)
        ctx.eval.dyn = MagicMock()
        ctx.eval.dyn.comm = MagicMock()
        ctx.eval.dyn.comm.dp = lambda c, x: dp_comm
        return ctx

    def test_no_clip_returns_forward(self):
        """BD-GC01: has_clip=False → returns forward_activation directly."""
        ccfg = self._make_ccfg_gradclip(has_clip=False)
        ctx = self._make_ctx_gradclip()
        result = EvalBody.fullrec_layer_activ_gradclip(ccfg, ctx)
        forward = EvalBody.fullrec_layer_activ(ccfg, ctx)
        self.assertAlmostEqual(result, forward, places=4)

    def test_clip_returns_grad_clip_when_smaller(self):
        """BD-GC02: grad_clip_mem < forward+dp → returns forward (clipping not enough)."""
        # Make grad_clip_mem very small so forward + dp > grad_clip_mem
        ccfg = self._make_ccfg_gradclip(has_clip=True, ep=1,
                                        bytes_os=1, shard_p_os_exp=1,
                                        shard_p_os_exp_partial=1,
                                        shard_p_os_non_exp_partial=1)
        ccfg.bytes_os = 1
        ctx = self._make_ctx_gradclip(non_exp=0.001, dp_comm=0)
        result = EvalBody.fullrec_layer_activ_gradclip(ccfg, ctx)
        # forward is large (1*2*1024*4*4096), grad_clip_mem tiny → returns forward
        forward = EvalBody.fullrec_layer_activ(ccfg, ctx)
        self.assertAlmostEqual(result, forward, places=0)

    def test_has_clip_zero_multiplier(self):
        """BD-GC01b: int(has_clip=False) = 0 makes grad_clip_mem = 0."""
        ccfg = self._make_ccfg_gradclip(has_clip=False)
        ctx = self._make_ctx_gradclip()
        result = EvalBody.fullrec_layer_activ_gradclip(ccfg, ctx)
        forward = EvalBody.fullrec_layer_activ(ccfg, ctx)
        # With has_clip=False, grad_clip_mem *= 0, so 0 < forward → returns forward
        self.assertAlmostEqual(result, forward, places=4)


class TestFullrecLayerCommGradclip(unittest.TestCase):
    """Test EvalBody.fullrec_layer_comm_gradclip."""

    def test_positive_activ_returns_dp_comm(self):
        """BD-CG01: activation > 0 → returns dp_comm."""
        ccfg = _make_ccfg(n_exp=1, bytes_os=12)
        ccfg.has_clip = True
        ccfg.bytes_compute = 2
        ccfg.s = 1024
        ccfg.b = 4
        ccfg.shard_recompute_input = 1.0
        ccfg.bytes_os = 12
        ctx = MagicMock()
        ctx.micro_factor = 1
        ctx.eval = MagicMock()
        ctx.eval.num_p = lambda c, x: (100.0, 0.0, 0.0)
        ctx.eval.dyn = MagicMock()
        ctx.eval.dyn.comm = MagicMock()
        ctx.eval.dyn.comm.dp = lambda c, x: 50.0
        result = EvalBody.fullrec_layer_comm_gradclip(ccfg, ctx)
        self.assertAlmostEqual(result, 50.0, places=4)

    def test_zero_activ_returns_zero(self):
        """BD-CG02: activation = 0 → returns 0."""
        ccfg = _make_ccfg(n_exp=1)
        ccfg.has_clip = False
        ccfg.bytes_compute = 0  # makes forward_activation = 0
        ccfg.s = 0
        ccfg.b = 0
        ccfg.shard_recompute_input = 1.0
        ctx = MagicMock()
        ctx.micro_factor = 1
        ctx.eval = MagicMock()
        ctx.eval.num_p = lambda c, x: (0.0, 0.0, 0.0)
        ctx.eval.dyn = MagicMock()
        ctx.eval.dyn.comm = MagicMock()
        ctx.eval.dyn.comm.dp = lambda c, x: 50.0
        result = EvalBody.fullrec_layer_comm_gradclip(ccfg, ctx)
        self.assertEqual(result, 0)


class TestActCpLayer(unittest.TestCase):
    """Test EvalBody.act_cp_layer (CP activation memory breakdown)."""

    def _make_ccfg_cp(self, a=32, t=1, cp=2, s=1024, b=4,
                       dc_kv=0, n_kv=32, dh=128, dhr=0,
                       cp_algo="colossalai_cp", device_per_node=8,
                       h=4096):
        """Create a mock CostModelConfig for CP activation tests."""
        ccfg = MagicMock()
        ccfg.a = a
        ccfg.t = t
        ccfg.cp = cp
        ccfg.s = s
        ccfg.b = b
        ccfg.dc_kv = dc_kv
        ccfg.n_kv = n_kv
        ccfg.dh = dh
        ccfg.dhr = dhr
        ccfg.h = h
        ccfg.cp_algo = cp_algo
        ccfg.device_per_node = device_per_node
        return ccfg

    def _make_ctx_cp(self, comm_buffer=0.0):
        ctx = MagicMock()
        # Mock EvalLayerComm.cp_comm_buffer
        return ctx

    def test_ring_cp_mha_basic(self):
        """BD-CP01: Ring CP with MHA attention produces valid breakdown."""
        ccfg = self._make_ccfg_cp(a=32, t=1, cp=2, cp_algo="colossalai_cp")
        ctx = self._make_ctx_cp()
        result = EvalBody.act_cp_layer(ccfg, ctx)
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.cp_types import CPMemoryBreakdown
        self.assertIsInstance(result, CPMemoryBreakdown)
        self.assertEqual(result.cp_degree, 2)
        self.assertEqual(result.seq_len, 1024)
        self.assertGreater(result.total_memory, 0)
        self.assertGreater(result.kv_cache_memory, 0)

    def test_ulysses_cp_mha_basic(self):
        """BD-CP02: Ulysses CP with MHA produces valid breakdown."""
        ccfg = self._make_ccfg_cp(a=32, t=1, cp=2, cp_algo="ulysses_cp")
        ctx = self._make_ctx_cp()
        result = EvalBody.act_cp_layer(ccfg, ctx)
        self.assertEqual(result.cp_degree, 2)
        self.assertGreater(result.total_memory, 0)

    def test_ring_cp_reduces_vs_no_cp(self):
        """BD-CP03: Ring CP s2_reduction > 0 (scores are divided by cp)."""
        ccfg = self._make_ccfg_cp(a=32, t=1, cp=4, cp_algo="colossalai_cp")
        ctx = self._make_ctx_cp()
        result = EvalBody.act_cp_layer(ccfg, ctx)
        self.assertGreater(result.s2_reduction, 0)
        self.assertGreater(result.kv_reduction, 0)

    def test_ulysses_reduces_per_rank_heads(self):
        """BD-CP04: Ulysses CP divides heads by cp, reducing s2 items."""
        ccfg = self._make_ccfg_cp(a=32, t=1, cp=4, cp_algo="ulysses_cp")
        ctx = self._make_ctx_cp()
        result = EvalBody.act_cp_layer(ccfg, ctx)
        self.assertGreater(result.s2_reduction, 0)

    def test_invalid_a_raises(self):
        """BD-CP05: a <= 0 raises ValueError."""
        ccfg = self._make_ccfg_cp(a=0)
        ctx = self._make_ctx_cp()
        with self.assertRaises(ValueError):
            EvalBody.act_cp_layer(ccfg, ctx)

    def test_invalid_cp_raises(self):
        """BD-CP06: cp <= 0 raises ValueError."""
        ccfg = self._make_ccfg_cp(cp=0)
        ctx = self._make_ctx_cp()
        with self.assertRaises(ValueError):
            EvalBody.act_cp_layer(ccfg, ctx)

    def test_mla_keeps_every_heads_keys_and_values(self):
        """BD-CP07: MLA (dc_kv > 0) keeps its heads' K and V, not the latent.

        Each head's key is 128 wide plus the 64-wide rotary part and its
        value 128 wide, so each of K and V is 32 * (2 * 128 + 64) / 2 wide.
        """
        ccfg = self._make_ccfg_cp(a=32, t=1, cp=2, dc_kv=512, n_kv=32, dh=128, dhr=64,
                                   cp_algo="colossalai_cp")
        ctx = self._make_ctx_cp()
        result = EvalBody.act_cp_layer(ccfg, ctx)
        self.assertEqual(result.kv_cache_memory, 2 * 2 * (1024 / 2) * 4 * 32 * (2 * 128 + 64) / 2)

    def test_gqa_kv_dim(self):
        """BD-CP08: GQA (n_kv < a) uses n_kv * dh / t for kv_dim."""
        ccfg = self._make_ccfg_cp(a=32, t=1, cp=2, dc_kv=0,
                                   n_kv=8, dh=128, cp_algo="colossalai_cp")
        ctx = self._make_ctx_cp()
        result = EvalBody.act_cp_layer(ccfg, ctx)
        # kv_dim = 8 * 128 / 1 = 1024
        self.assertGreater(result.kv_cache_memory, 0)


class TestConfigOptimizerShard(unittest.TestCase):
    """Test derive_optimizer_sharding's has_op guard on shard_p_os_exp.

    Verifies that when has_op=False, d_exp is NOT used as a sharding factor
    for expert optimizer state, preventing memory underestimation.
    """

    @staticmethod
    def _make_parser_ccfg(
        n_exp=8, d_exp=4, cp=1, t_exp=1, ep=2,
        has_op=True, has_grad_shard=True, os_max_shard=1, d=4, t=1,
        edp_group=None,
    ):
        """Create a mock _CostModVar for parser-level shard tests.

        A strategy's config carries the expert data-parallel group the runtime
        forms.  Left out here, it is the older reconstruction from ``d_exp``,
        which agrees with the group wherever EP divides DP.
        """
        from hyper_parallel.auto_parallel.sapp_nd.nd.common._cost_model_variables import _CostModVar
        ccfg = MagicMock(spec=_CostModVar)
        ccfg.d = d
        ccfg.t = t
        ccfg.n_exp = n_exp
        ccfg.d_exp = d_exp
        ccfg.cp = cp
        ccfg.t_exp = t_exp
        ccfg.ep = ep
        ccfg.edp_group = d_exp * cp * t_exp if edp_group is None else edp_group
        ccfg.has_op = has_op
        ccfg.has_grad_shard = has_grad_shard
        ccfg.os_max_shard = os_max_shard
        ccfg.expert_shard = None
        ccfg.expert_shard_group = False
        # The legacy schemas' rule, CP sharding parameter state on top of the
        # optimizer's ranks; HyperParallel's states True (param_cp).
        ccfg.shard_spans_cp = False
        return ccfg

    def test_has_op_true_uses_d_exp(self):
        """BD-H01: has_op=True => shard_p_os_exp = d_exp * cp * t_exp."""
        ccfg = self._make_parser_ccfg(d_exp=4, cp=2, t_exp=1, has_op=True)
        derive_optimizer_sharding(ccfg)
        expected = 4 * 2 * 1  # d_exp * cp * t_exp
        self.assertEqual(ccfg.shard_p_os_exp, expected)

    def test_has_op_false_ignores_d_exp(self):
        """BD-H02: has_op=False => shard_p_os_exp = 1 * cp * t_exp (d_exp bypassed).

        Without the guard, d_exp=4 would produce shard_p_os_exp=8, causing
        expert param/OS/grad memory to be underestimated by 4x.
        """
        ccfg = self._make_parser_ccfg(d_exp=4, cp=2, t_exp=1, has_op=False)
        derive_optimizer_sharding(ccfg)
        expected = 1 * 2 * 1  # (d_exp if has_op else 1) * cp * t_exp
        self.assertEqual(ccfg.shard_p_os_exp, expected)

    def test_has_op_false_non_exp_symmetric(self):
        """BD-H03: has_op guard is symmetric between non-exp and expert paths.

        Non-exp: shard_p_os_non_exp = (d if has_op else 1) * cp * t
        Expert:  shard_p_os_exp     = (d_exp if has_op else 1) * cp * t_exp
        Both bypass the DP sharding factor when has_op=False.
        """
        ccfg = self._make_parser_ccfg(d_exp=4, cp=2, t_exp=1, has_op=False)
        ccfg.d = 4
        ccfg.t = 1
        derive_optimizer_sharding(ccfg)
        # Non-exp: (d if has_op else 1) * cp * t = 1 * 2 * 1 = 2
        # Expert:  (d_exp if has_op else 1) * cp * t_exp = 1 * 2 * 1 = 2
        self.assertEqual(ccfg.shard_p_os_non_exp, 2)
        self.assertEqual(ccfg.shard_p_os_exp, 2)


class TestExpertDataParallelGroup(unittest.TestCase):
    """The ranks that hold the same experts, as the runtime's expert mesh spans them."""

    @staticmethod
    def _ccfg(d, t, cp, ep, n_exp=8, hff_exp=1408):
        """A config a strategy has just been laid over."""
        from hyper_parallel.auto_parallel.sapp_nd.nd.common._cost_model_variables import _CostModVar
        ccfg = MagicMock(spec=_CostModVar)
        ccfg.d, ccfg.t, ccfg.cp, ccfg.ep, ccfg.etp = d, t, cp, ep, 0
        ccfg.n_exp, ccfg.hff_exp = n_exp, hff_exp
        ccfg.t_exp, ccfg.d_exp, ccfg.edp_group = t, 0, 0
        return ccfg

    def test_context_parallelism_does_not_change_the_group(self):
        """CP takes its ranks from DP, so the group a strategy keeps is the same.

        On 64 ranks at EP 16, 4 ranks hold the same experts whatever CP takes,
        since the runtime spreads its experts over the stage's whole dp * cp *
        tp domain; the older reconstruction read 0 at CP 8.
        """
        got = {}
        for cp in (1, 2, 4, 8):
            ccfg = self._ccfg(d=64 // cp, t=1, cp=cp, ep=16)
            derive_expert_degrees(ccfg)
            got[cp] = ccfg.edp_group
        self.assertEqual(got, {1: 4, 2: 4, 4: 4, 8: 4})

    def test_a_stage_narrower_than_its_experts_is_refused(self):
        """A strategy spreading experts over more ranks than the stage holds."""
        ccfg = self._ccfg(d=2, t=1, cp=1, ep=16)
        with self.assertRaises(TypeError):
            derive_expert_degrees(ccfg)

    def test_context_parallel_ranks_reduce_an_experts_gradient(self):
        """An expert's gradient is reduced over CP's ranks too (X3).

        On 64 ranks with an optimizer shard: at CP 2, DP 32 and EP 32 leave
        one DP rank over EP, but the two CP ranks hold the same experts and
        reduce their gradients, so the expert's DP communication is priced,
        where it read 0; at CP 1 and EP 64 no other rank holds them; at TP 2
        without EP the TP ranks hold different slices and reduce nothing.
        """
        got = []
        for d, t, cp, ep in ((32, 1, 2, 32), (64, 1, 1, 64), (1, 2, 1, 1)):
            ccfg = self._ccfg(d=d, t=t, cp=cp, ep=ep)
            ccfg.has_op, ccfg.has_grad_shard = True, False
            derive_expert_degrees(ccfg)
            derive_comm_flags(ccfg)
            got.append(ccfg.comm_d_exp)
        self.assertEqual(got, [2, 0, 0])


if __name__ == "__main__":
    unittest.main()
