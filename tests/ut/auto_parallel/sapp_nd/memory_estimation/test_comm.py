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
"""Unit tests for comm-layer memory estimation."""
# pylint: disable=W0105,W0125
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
# pylint: disable=missing-class-docstring,missing-function-docstring
import os
import unittest
from unittest.mock import MagicMock

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.comm import EvalLayerComm
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"


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
    ccfg.comm_cp = comm_cp
    ccfg.tokens_per_expert = tokens_per_expert

    ccfg.fsdp = False
    ccfg.comm_fsdp = 0.0
    ccfg.comm_hsdp = 0.0
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


class TestDpCommNonExpFsdp(unittest.TestCase):

    def test_zero3_fsdp_only_grad_rs(self):
        ccfg = _make_ccfg(tp=2, cp=1, comm_d_non_exp=3)
        ccfg.fsdp = True
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.dp_comm_non_exp(ccfg, ctx)
        expected = 150.0 / (1 * 2)
        self.assertAlmostEqual(result, expected, places=4)

    def test_zero3_no_fsdp_includes_param_ag(self):
        ccfg = _make_ccfg(tp=2, cp=1, comm_d_non_exp=3)
        ccfg.fsdp = False
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.dp_comm_non_exp(ccfg, ctx)
        expected = 150.0 / (1 * 2) + 150.0 / 2 + 150.0 / 2
        self.assertAlmostEqual(result, expected, places=4)


class TestDpCommExpFsdp(unittest.TestCase):

    def test_zero3_fsdp_only_grad_rs(self):
        ccfg = _make_ccfg(ep=4, tp=1, t_exp=1, cp=1, comm_d_exp=3)
        ccfg.fsdp = True
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.dp_comm_exp(ccfg, ctx)
        exp_param_size = 400.0 + 200.0
        expected = exp_param_size / (1 * 1 * 4)
        self.assertAlmostEqual(result, expected, places=4)

    def test_zero3_no_fsdp_includes_param_ag(self):
        ccfg = _make_ccfg(ep=4, tp=1, t_exp=1, cp=1, comm_d_exp=3)
        ccfg.fsdp = False
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.dp_comm_exp(ccfg, ctx)
        exp_param_size = 400.0 + 200.0
        expected = (
            exp_param_size / (1 * 1 * 4)
            + exp_param_size / max(4, 1)
            + exp_param_size / max(4, 1)
        )
        self.assertAlmostEqual(result, expected, places=4)


class TestEpCommLayerBalancedNewReturns(unittest.TestCase):

    def test_n_exp_1_returns_zero(self):
        ccfg = _make_ccfg(n_exp=1, ep=4, comm_ep=1.0)
        ctx = _make_ctx()
        result = EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, 1)
        self.assertEqual(result, 0)

    def test_comm_ep_zero_before_n_exp_check(self):
        ccfg = _make_ccfg(n_exp=8, ep=4, comm_ep=0)
        ctx = _make_ctx()
        result = EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, 1)
        self.assertEqual(result, 0)


class TestEpCommLayerImbalancedNewReturns(unittest.TestCase):

    def test_n_exp_1_returns_zero(self):
        ccfg = _make_ccfg(n_exp=1, ep=4, comm_ep=1.0, tokens_per_expert=[100])
        ctx = _make_ctx()
        result = EvalLayerComm.ep_comm_layer_imbalanced(ccfg, ctx, 1)
        self.assertEqual(result, 0)


class TestEpCommLayerNewReturns(unittest.TestCase):

    def test_n_exp_1_returns_zero(self):
        ccfg = _make_ccfg(n_exp=1, ep=4, comm_ep=1.0, tokens_per_expert=None)
        ctx = _make_ctx()
        result = EvalLayerComm.ep_comm_layer(ccfg, ctx, 1)
        self.assertEqual(result, 0)

    def test_comm_ep_zero_returns_zero(self):
        ccfg = _make_ccfg(n_exp=8, ep=4, comm_ep=0, tokens_per_expert=None)
        ctx = _make_ctx()
        result = EvalLayerComm.ep_comm_layer(ccfg, ctx, 1)
        self.assertEqual(result, 0)


class TestFsdpCommLayer(unittest.TestCase):

    def test_pure_fsdp_non_exp_only(self):
        non_exp = 200.0
        ccfg = _make_ccfg(tp=2, cp=1, n_exp=1, ep=1, t_exp=1, bytes_compute=2)
        ccfg.fsdp = True
        ccfg.comm_fsdp = 1.0
        ccfg.d = 4
        ccfg.d_shard = 4
        ccfg.d_shard_or_d = 4
        ccfg.comm_hsdp = 0.0
        ctx = _make_ctx(num_p_result=(non_exp, 0.0, 0.0))
        result = EvalLayerComm.fsdp_comm_layer(ccfg, ctx)
        expected = 1.0 * non_exp / (4 * 1 * 2) * 2 * 2
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsdp_adds_inter_node_comm(self):
        non_exp, routed, shared = 200.0, 300.0, 100.0
        ccfg = _make_ccfg(tp=2, cp=1, n_exp=8, ep=4, t_exp=1, bytes_compute=2)
        ccfg.fsdp = True
        ccfg.comm_fsdp = 1.0
        ccfg.d = 8
        ccfg.d_shard = 4
        ccfg.d_shard_or_d = 4
        ccfg.comm_hsdp = 1.0
        ctx = _make_ctx(num_p_result=(non_exp, routed, shared))
        result = EvalLayerComm.fsdp_comm_layer(ccfg, ctx)
        exp = routed + shared
        d_shard = 4
        d_replicate = 8 // 4
        non_exp_comm = 1.0 * non_exp / (d_shard * 1 * 2) * 2 * 2
        exp_comm = 1.0 * exp / (d_shard * 4 * 1 * 1) * 2 * 2
        sharded_non_exp = non_exp / (d_shard * 1 * 2)
        sharded_exp = exp / (d_shard * 1 * 1)
        hsdp_comm = 1.0 / d_replicate * (sharded_non_exp + sharded_exp) * 2
        expected = non_exp_comm + hsdp_comm + exp_comm
        self.assertAlmostEqual(result, expected, places=4)


class TestFsdpBufferLayer(unittest.TestCase):

    def test_non_exp_buffer(self):
        non_exp = 200.0
        ccfg = _make_ccfg(tp=2, cp=1, n_exp=1, ep=1, t_exp=1)
        ccfg.fsdp = True
        ccfg.comm_fsdp = 1.0
        ccfg.fsdp_all_gather_buffer = 1.0
        ccfg.bytes_p = 4
        ccfg.d = 4
        ccfg.d_shard = 4
        ccfg.d_shard_or_d = 4
        ctx = _make_ctx(num_p_result=(non_exp, 0.0, 0.0))
        result = EvalLayerComm.fsdp_buffer_layer(ccfg, ctx)
        expected = 1.0 * 1.0 * non_exp * 4 / (1 * 2)
        self.assertAlmostEqual(result, expected, places=4)

    def test_with_expert(self):
        non_exp, routed, shared = 200.0, 300.0, 100.0
        ccfg = _make_ccfg(tp=2, cp=1, n_exp=8, ep=4, t_exp=1)
        ccfg.fsdp = True
        ccfg.comm_fsdp = 1.0
        ccfg.fsdp_all_gather_buffer = 1.0
        ccfg.bytes_p = 4
        ccfg.d = 4
        ccfg.d_shard = 4
        ccfg.d_shard_or_d = 4
        ctx = _make_ctx(num_p_result=(non_exp, routed, shared))
        result = EvalLayerComm.fsdp_buffer_layer(ccfg, ctx)
        non_exp_buf = 1.0 * 1.0 * non_exp * 4 / (1 * 2)
        exp_buf = 1.0 * 1.0 * (routed + shared) * 4 / (4 * 1 * 1)
        expected = non_exp_buf + exp_buf
        self.assertAlmostEqual(result, expected, places=4)


class TestFsdpBufferComm(unittest.TestCase):

    def test_returns_bytes_not_element_count(self):
        non_exp, routed, shared = 150.0, 400.0, 200.0
        ccfg = _make_ccfg(tp=2, cp=1, comm_t=0.0, bytes_compute=2)
        ccfg.fsdp = True
        ccfg.comm_fsdp = 1.0
        ccfg.fsdp_all_gather_buffer = 1.0
        ccfg.bytes_p = 4
        ccfg.d = 4
        ccfg.d_shard = 4
        ccfg.d_shard_or_d = 4
        ccfg.t_exp = 1
        ccfg.n_exp = 8
        ccfg.ep = 4
        ccfg.comm_hsdp = 0.0
        ctx = _make_ctx(num_p_result=(non_exp, routed, shared))
        result = EvalLayerComm.fsdp_buffer_comm(ccfg, ctx)
        non_exp_buf = 1.0 * 1.0 * non_exp * 2 / (1 * 2)
        exp_buf = 1.0 * 1.0 * (routed + shared) * 2 / (4 * 1 * 1)
        expected = non_exp_buf + exp_buf
        self.assertAlmostEqual(result, expected, places=4)

    def test_zero_when_no_fsdp(self):
        ccfg = _make_ccfg()
        ccfg.fsdp = False
        ccfg.comm_fsdp = 0.0
        ccfg.fsdp_all_gather_buffer = 0.0
        ccfg.bytes_compute = 2
        ctx = _make_ctx()
        result = EvalLayerComm.fsdp_buffer_comm(ccfg, ctx)
        self.assertEqual(result, 0.0)


class TestHsdpInterBufferComm(unittest.TestCase):

    def test_zero_for_pure_fsdp(self):
        ccfg = _make_ccfg(tp=2, cp=1, bytes_compute=2)
        ccfg.comm_hsdp = 0.0
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.hsdp_inter_buffer_comm(ccfg, ctx)
        self.assertEqual(result, 0.0)

    def test_hsdp_buffer_formula(self):
        non_exp, routed, shared = 150.0, 400.0, 200.0
        ccfg = _make_ccfg(tp=2, cp=1, n_exp=8, ep=4, t_exp=1, bytes_compute=2)
        ccfg.d = 8
        ccfg.d_shard = 4
        ccfg.d_shard_or_d = 4
        ccfg.comm_hsdp = 1.0
        ctx = _make_ctx(num_p_result=(non_exp, routed, shared))
        result = EvalLayerComm.hsdp_inter_buffer_comm(ccfg, ctx)
        sharded_non_exp = non_exp / (4 * 1 * 2)
        sharded_exp = (routed + shared) / (4 * 1 * 1)
        expected = 1.0 * (sharded_non_exp + sharded_exp) * 2
        self.assertAlmostEqual(result, expected, places=4)


class TestFsdpGradBufferComm(unittest.TestCase):

    def test_returns_bytes_grad_multiplier(self):
        non_exp, routed, shared = 150.0, 400.0, 200.0
        ccfg = _make_ccfg(tp=2, cp=1, comm_t=0.0, bytes_compute=2)
        ccfg.fsdp = True
        ccfg.comm_fsdp = 1.0
        ccfg.fsdp_all_gather_buffer = 1.0
        ccfg.bytes_grad = 4
        ccfg.d = 4
        ccfg.d_shard = 4
        ccfg.d_shard_or_d = 4
        ccfg.t_exp = 1
        ccfg.n_exp = 8
        ccfg.ep = 4
        ccfg.comm_hsdp = 0.0
        ctx = _make_ctx(num_p_result=(non_exp, routed, shared))
        result = EvalLayerComm.fsdp_grad_buffer_comm(ccfg, ctx)
        non_exp_buf = 1.0 * 1.0 * non_exp * 4 / (1 * 2)
        exp_buf = 1.0 * 1.0 * (routed + shared) * 4 / (4 * 1 * 1)
        expected = non_exp_buf + exp_buf
        self.assertAlmostEqual(result, expected, places=4)

    def test_zero_when_no_fsdp(self):
        ccfg = _make_ccfg()
        ccfg.fsdp = False
        ccfg.comm_fsdp = 0.0
        ccfg.fsdp_all_gather_buffer = 0.0
        ccfg.bytes_grad = 4
        ctx = _make_ctx()
        result = EvalLayerComm.fsdp_grad_buffer_comm(ccfg, ctx)
        self.assertEqual(result, 0.0)


def _make_tail_ccfg(
    h=16, v=64, t=2, cp=1, d=4, n_mtp=1, n_exp=1, ep=1, t_exp=1,
    bytes_p=2, bytes_compute=2, bytes_grad=2, bytes_os=12,
    fsdp=True, comm_fsdp=1.0, comm_hsdp=0.0, fsdp_all_gather_buffer=1.0,
    d_shard=0, is_shard_mtp_param=True, shard_p_os_non_exp_partial=None,
    shard_grad_non_exp=None, **kwargs,
):
    ccfg = MagicMock()
    ccfg.h = h
    ccfg.v = v
    ccfg.t = t
    ccfg.cp = cp
    ccfg.d = d
    ccfg.n_mtp = n_mtp
    ccfg.n_exp = n_exp
    ccfg.ep = ep
    ccfg.t_exp = t_exp
    ccfg.bytes_p = bytes_p
    ccfg.bytes_compute = bytes_compute
    ccfg.bytes_grad = bytes_grad
    ccfg.bytes_os = bytes_os
    ccfg.fsdp = fsdp
    ccfg.comm_fsdp = comm_fsdp
    ccfg.comm_hsdp = comm_hsdp
    ccfg.fsdp_all_gather_buffer = fsdp_all_gather_buffer
    ccfg.is_shard_mtp_param = is_shard_mtp_param
    if d_shard > 0:
        ccfg.d_shard = d_shard
        ccfg.d_shard_or_d = d_shard
    else:
        ccfg.d_shard = d if fsdp else 0
        ccfg.d_shard_or_d = d if fsdp else d
    if shard_p_os_non_exp_partial is not None:
        ccfg.shard_p_os_non_exp_partial = shard_p_os_non_exp_partial
    else:
        ccfg.shard_p_os_non_exp_partial = d * cp * t if fsdp else 1
    if shard_grad_non_exp is not None:
        ccfg.shard_grad_non_exp = shard_grad_non_exp
    else:
        ccfg.shard_grad_non_exp = d * cp * t if fsdp else 1
    ccfg.bytes_norm = 4
    ccfg.s = 8
    ccfg.b = 1
    ccfg.shard_output_activ = 1
    for k, val in kwargs.items():
        setattr(ccfg, k, val)
    return ccfg


def _make_tail_ctx(
    num_p_result=100.0,
    current_node=LayerType.OUTPUT_LAYER,
):
    ctx = MagicMock()
    ctx.eval = MagicMock()
    ctx.eval.num_p = lambda c, x: num_p_result
    ctx.current_node = current_node
    ctx.micro_factor = 1
    ctx.eval.dyn = MagicMock()
    ctx.eval.dyn.comm = MagicMock()
    ctx.eval.dyn.comm.fsdp = MagicMock(return_value=10.0)
    ctx.eval.dyn.comm.fsdp_grad = MagicMock(return_value=20.0)
    ctx.eval.stat = MagicMock()
    ctx.eval.stat.p = MagicMock(return_value=50.0)
    ctx.swap_os = False
    return ctx


class TestTailFsdpCommOutSingle(unittest.TestCase):

    def test_basic_formula(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTailSingle

        param_size = 100.0
        ccfg = _make_tail_ccfg(t=2, cp=1, comm_fsdp=1.0, fsdp_all_gather_buffer=1.0, bytes_compute=2)
        ctx = _make_tail_ctx(num_p_result=param_size)
        result = EvalTailSingle.fsdp_comm_out_single(ccfg, ctx)
        expected = 1.0 * 1.0 * param_size * 2 / (1 * 2)
        self.assertAlmostEqual(result, expected, places=4)

    def test_zero_when_no_fsdp(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTailSingle

        ccfg = _make_tail_ccfg(comm_fsdp=0.0, fsdp_all_gather_buffer=0.0)
        ctx = _make_tail_ctx()
        result = EvalTailSingle.fsdp_comm_out_single(ccfg, ctx)
        self.assertAlmostEqual(result, 0.0, places=4)


class TestTailFsdpGradCommOutSingle(unittest.TestCase):

    def test_basic_formula(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTailSingle

        param_size = 100.0
        ccfg = _make_tail_ccfg(t=2, cp=1, comm_fsdp=1.0, fsdp_all_gather_buffer=1.0, bytes_grad=4)
        ctx = _make_tail_ctx(num_p_result=param_size)
        result = EvalTailSingle.fsdp_grad_comm_out_single(ccfg, ctx)
        expected = 1.0 * 1.0 * param_size * 4 / (1 * 2)
        self.assertAlmostEqual(result, expected, places=4)


class TestTailHsdpCommOutSingle(unittest.TestCase):

    def test_zero_for_pure_fsdp(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTailSingle

        ccfg = _make_tail_ccfg(comm_hsdp=0.0)
        ctx = _make_tail_ctx()
        result = EvalTailSingle.hsdp_comm_out_single(ccfg, ctx)
        self.assertEqual(result, 0.0)

    def test_hsdp_formula(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTailSingle

        param_size = 200.0
        ccfg = _make_tail_ccfg(d=8, d_shard=4, t=2, cp=1, comm_hsdp=1.0, bytes_compute=2)
        ctx = _make_tail_ctx(num_p_result=param_size)
        result = EvalTailSingle.hsdp_comm_out_single(ccfg, ctx)
        sharded_size = param_size / (4 * 1 * 2)
        expected = 1.0 * sharded_size * 2
        self.assertAlmostEqual(result, expected, places=4)


class TestTailStatOutputSingleFsdp(unittest.TestCase):

    def test_fsdp_vs_no_fsdp_bytes(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTailSingle

        param_size = 100.0
        ccfg_fsdp = _make_tail_ccfg(fsdp=True, bytes_compute=2, bytes_p=4,
                                    shard_p_os_non_exp_partial=8)
        ccfg_nofsdp = _make_tail_ccfg(fsdp=False, bytes_compute=2, bytes_p=4,
                                      shard_p_os_non_exp_partial=1)
        ctx = _make_tail_ctx(num_p_result=param_size)
        r_fsdp = EvalTailSingle.stat_output_single_p(ccfg_fsdp, ctx)
        r_nofsdp = EvalTailSingle.stat_output_single_p(ccfg_nofsdp, ctx)
        self.assertAlmostEqual(r_fsdp, param_size * (2 / 8), places=4)
        self.assertAlmostEqual(r_nofsdp, param_size * (4 / 1), places=4)


class TestMtpFsdpCommMtp(unittest.TestCase):

    def test_zero_when_n_mtp_zero(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalMTP

        ccfg = _make_tail_ccfg(n_mtp=0)
        ctx = _make_tail_ctx()
        result = EvalMTP.fsdp_comm_mtp(ccfg, ctx)
        self.assertEqual(result, 0)

    def test_includes_mtp_param_and_head_tail(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalMTP

        n_mtp = 2
        h = 16
        t = 2
        cp = 1
        bytes_compute = 2
        comm_fsdp = 1.0
        fsdp_all_gather_buffer = 1.0
        ccfg = _make_tail_ccfg(
            n_mtp=n_mtp, h=h, t=t, cp=cp, bytes_compute=bytes_compute,
            comm_fsdp=comm_fsdp, fsdp_all_gather_buffer=fsdp_all_gather_buffer,
        )
        mtp_param = 2 * h * h + 4 * h
        ctx = _make_tail_ctx(num_p_result=100.0)
        ctx.eval.dyn.comm.fsdp = MagicMock(return_value=10.0)

        result = EvalMTP.fsdp_comm_mtp(ccfg, ctx)
        mtp_only = comm_fsdp * fsdp_all_gather_buffer * n_mtp * mtp_param * bytes_compute / (cp * t)
        self.assertGreater(result, 0)
        self.assertGreaterEqual(result, mtp_only - 1e-6)
        self.assertGreater(result, mtp_only + n_mtp * 10.0)


class TestMtpFsdpGradCommMtp(unittest.TestCase):

    def test_zero_when_n_mtp_zero(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalMTP

        ccfg = _make_tail_ccfg(n_mtp=0)
        ctx = _make_tail_ctx()
        result = EvalMTP.fsdp_grad_comm_mtp(ccfg, ctx)
        self.assertEqual(result, 0)

    def test_uses_bytes_grad(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalMTP

        n_mtp = 1
        h = 16
        t = 2
        cp = 1
        bytes_grad = 4
        comm_fsdp = 1.0
        fsdp_all_gather_buffer = 1.0
        ccfg = _make_tail_ccfg(
            n_mtp=n_mtp, h=h, t=t, cp=cp, bytes_grad=bytes_grad,
            comm_fsdp=comm_fsdp, fsdp_all_gather_buffer=fsdp_all_gather_buffer,
        )
        mtp_param = 2 * h * h + 4 * h
        ctx = _make_tail_ctx(num_p_result=100.0)
        ctx.eval.dyn.comm.fsdp_grad = MagicMock(return_value=5.0)

        result = EvalMTP.fsdp_grad_comm_mtp(ccfg, ctx)
        mtp_term = comm_fsdp * fsdp_all_gather_buffer * n_mtp * mtp_param * bytes_grad / (cp * t)
        self.assertGreater(result, 0)
        self.assertGreaterEqual(result, mtp_term - 1e-6)


class TestMtpHsdpCommMtp(unittest.TestCase):

    def test_zero_when_no_hsdp(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalMTP

        ccfg = _make_tail_ccfg(n_mtp=1, comm_hsdp=0.0)
        ctx = _make_tail_ctx()
        result = EvalMTP.hsdp_comm_mtp(ccfg, ctx)
        self.assertEqual(result, 0)

    def test_hsdp_formula(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalMTP

        n_mtp = 2
        h = 16
        d = 8
        d_shard = 4
        t = 2
        cp = 1
        bytes_compute = 2
        comm_hsdp = 1.0
        d_replicate = d // d_shard

        ccfg = _make_tail_ccfg(
            n_mtp=n_mtp, h=h, d=d, d_shard=d_shard, t=t, cp=cp,
            bytes_compute=bytes_compute, comm_hsdp=comm_hsdp,
        )
        mtp_param = 2 * h * h + 4 * h

        embed_param = 100.0
        output_param = 200.0
        ctx = _make_tail_ctx(num_p_result=embed_param)
        original_num_p = ctx.eval.num_p

        def _num_p(c, x):
            if ctx.current_node == LayerType.EMBEDDING_LAYER:
                return embed_param
            if ctx.current_node == LayerType.OUTPUT_LAYER:
                return output_param
            return original_num_p(c, x)

        ctx.eval.num_p = _num_p

        result = EvalMTP.hsdp_comm_mtp(ccfg, ctx)

        sharded_mtp = mtp_param / (d_shard * cp * t)
        contrib_mtp = comm_hsdp * n_mtp * sharded_mtp * bytes_compute / d_replicate
        sharded_embed = embed_param / (d_shard * cp * t)
        contrib_embed = comm_hsdp * n_mtp * sharded_embed * bytes_compute / d_replicate
        sharded_output = output_param / (d_shard * cp * t)
        contrib_output = comm_hsdp * n_mtp * sharded_output * bytes_compute / d_replicate

        expected = contrib_mtp + contrib_embed + contrib_output
        self.assertAlmostEqual(result, expected, places=4)


class TestEvalTailFsdpCommOutput(unittest.TestCase):

    def test_sums_components(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTail, EvalTailSingle, EvalMTP

        ccfg = _make_tail_ccfg(n_mtp=1, h=16, t=2, cp=1, comm_fsdp=1.0,
                               fsdp_all_gather_buffer=1.0, bytes_compute=2)
        ctx = _make_tail_ctx(num_p_result=100.0)
        ctx.eval.dyn.comm.fsdp = MagicMock(return_value=10.0)

        result = EvalTail.fsdp_comm_output(ccfg, ctx)
        out_single = EvalTailSingle.fsdp_comm_out_single(ccfg, ctx)
        mtp = EvalMTP.fsdp_comm_mtp(ccfg, ctx)
        self.assertAlmostEqual(result, out_single + mtp, places=4)


class TestEvalTailHsdpCommOutput(unittest.TestCase):

    def test_sums_components(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTail, EvalTailSingle, EvalMTP

        ccfg = _make_tail_ccfg(n_mtp=1, h=16, d=8, d_shard=4, t=2, cp=1,
                               comm_hsdp=1.0, bytes_compute=2)
        ctx = _make_tail_ctx(num_p_result=100.0)

        result = EvalTail.hsdp_comm_output(ccfg, ctx)
        out_single = EvalTailSingle.hsdp_comm_out_single(ccfg, ctx)
        mtp = EvalMTP.hsdp_comm_mtp(ccfg, ctx)
        self.assertAlmostEqual(result, out_single + mtp, places=4)


class TestEvalTailFsdpGradCommOutput(unittest.TestCase):

    def test_sums_components(self):
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTail, EvalTailSingle, EvalMTP

        ccfg = _make_tail_ccfg(n_mtp=1, h=16, t=2, cp=1, comm_fsdp=1.0,
                               fsdp_all_gather_buffer=1.0, bytes_grad=4)
        ctx = _make_tail_ctx(num_p_result=100.0)
        ctx.eval.dyn.comm.fsdp_grad = MagicMock(return_value=5.0)

        result = EvalTail.fsdp_grad_comm_output(ccfg, ctx)
        out_single = EvalTailSingle.fsdp_grad_comm_out_single(ccfg, ctx)
        mtp = EvalMTP.fsdp_grad_comm_mtp(ccfg, ctx)
        self.assertAlmostEqual(result, out_single + mtp, places=4)


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
        """CM-D02: ZeRO level 3 non-exp comm = grad_rs + param_ag(fwd+bwd)."""
        if False: """CM-D02: ZeRO level 3 non-exp comm = non_exp / t."""
        ccfg = _make_ccfg(tp=2, cp=1, comm_d_non_exp=3)
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.dp_comm_non_exp(ccfg, ctx)
        if False: expected = 150.0 / 2
        expected = 150.0 / (1 * 2) + 150.0 / 2 + 150.0 / 2
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
        """CM-D04: ZeRO level 3 exp comm = grad_rs + param_ag(fwd+bwd)."""
        if False: """CM-D04: ZeRO level 3 exp comm = exp_param / (cp*t_exp*ep)."""
        ccfg = _make_ccfg(ep=4, tp=1, t_exp=1, cp=1, comm_d_exp=3)
        ctx = _make_ctx(num_p_result=(150.0, 400.0, 200.0))
        result = EvalLayerComm.dp_comm_exp(ccfg, ctx)
        exp_param_size = 400.0 + 200.0
        if False: expected = exp_param_size / (1 * 1 * 4)
        expected = exp_param_size / (1 * 1 * 4) + exp_param_size / max(4, 1) + exp_param_size / max(4, 1)
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




class TestDpCommExpZeroReturn(unittest.TestCase):
    """Test dp_comm_exp early return when exp_param_size == 0."""

    def test_zero_exp_params_returns_zero(self):
        """CM-D05: dp_comm_exp returns 0 when routed+shared == 0 (dense-only)."""
        ccfg = _make_ccfg(comm_d_exp=2)
        ctx = _make_ctx(num_p_result=(150.0, 0.0, 0.0))
        result = EvalLayerComm.dp_comm_exp(ccfg, ctx)
        self.assertEqual(result, 0)


class TestTpCommNonExp(unittest.TestCase):
    """Test tp_comm_non_exp (TP communication for non-expert parameters)."""

    def test_dense_formula(self):
        """CM-T03: Dense (n_exp=1) TP non-exp comm = 0.25 * n_gather * s * b * h * mb."""
        ccfg = _make_ccfg(n_exp=1, n_gather=2, s=1024, b=4, h=4096,
                          comm_t=1.0, cp=1, bytes_compute=2)
        rec_op = MagicMock()
        rec_op.gather = False
        ccfg.rec_op = rec_op
        ctx = _make_ctx(current_node=None)
        mb = 1
        result = EvalLayerComm.tp_comm_non_exp(ccfg, ctx, mb)
        inner = 0.25 * 2 * 1024 * 4 * 4096 * 1
        expected = 1 * 1.0 * inner / 1
        self.assertAlmostEqual(result, expected, places=4)

    def test_moe_formula(self):
        """CM-T04: MoE (n_exp>1) TP non-exp comm uses h*h*n_attMM formula."""
        ccfg = _make_ccfg(n_exp=8, n_gather=2, h=4096, bytes_compute=2,
                          n_attMM=2, comm_t=1.0, cp=1)
        rec_op = MagicMock()
        rec_op.gather = False
        ccfg.rec_op = rec_op
        ctx = _make_ctx(current_node=None)
        mb = 1
        result = EvalLayerComm.tp_comm_non_exp(ccfg, ctx, mb)
        inner = 0.25 * 2 * 4096 * 4096 * 2 * 2
        expected = 1 * 1.0 * inner / 1
        self.assertAlmostEqual(result, expected, places=0)

    def test_cp_divides(self):
        """CM-T05: CP degree divides TP non-exp comm volume."""
        ccfg_cp1 = _make_ccfg(n_exp=1, n_gather=2, s=1024, b=4, h=4096,
                              comm_t=1.0, cp=1, bytes_compute=2)
        ccfg_cp2 = _make_ccfg(n_exp=1, n_gather=2, s=1024, b=4, h=4096,
                              comm_t=1.0, cp=2, bytes_compute=2)
        rec_op = MagicMock()
        rec_op.gather = False
        ccfg_cp1.rec_op = rec_op
        ccfg_cp2.rec_op = rec_op
        ctx = _make_ctx(current_node=None)
        mb = 1
        r1 = EvalLayerComm.tp_comm_non_exp(ccfg_cp1, ctx, mb)
        r2 = EvalLayerComm.tp_comm_non_exp(ccfg_cp2, ctx, mb)
        self.assertAlmostEqual(r1, r2 * 2, places=4)


class TestTpCommLayer(unittest.TestCase):
    """Test tp_comm_layer sums non_exp and exp."""

    def test_layer_sums_both(self):
        """CM-T06: tp_comm_layer = tp_comm_non_exp + tp_comm_exp."""
        ccfg = _make_ccfg(n_exp=8, n_gather=2, h=4096, bytes_compute=2,
                          n_attMM=2, n_ffMM=1, comm_t=1.0, cp=1)
        rec_op = MagicMock()
        rec_op.gather = False
        ccfg.rec_op = rec_op
        ctx = _make_ctx(current_node=None)
        mb = 1
        result = EvalLayerComm.tp_comm_layer(ccfg, ctx, mb)
        non_exp = EvalLayerComm.tp_comm_non_exp(ccfg, ctx, mb)
        exp = EvalLayerComm.tp_comm_exp(ccfg, ctx, mb)
        self.assertAlmostEqual(result, non_exp + exp, places=0)


class TestCpCommExp(unittest.TestCase):
    """Test cp_comm_exp (CP communication for expert parameters)."""

    def test_ring_returns_nonzero(self):
        """CM-CE01: Ring CP expert comm is non-zero."""
        ccfg = _make_ccfg(cp_algo="colossalai_cp", s=1024, b=4,
                          hff=14336, n_ffMM=1, comm_cp=1.0, tp=1)
        ctx = _make_ctx()
        result = EvalLayerComm.cp_comm_exp(ccfg, ctx)
        expected = 1.0 * 2 * 1024 * 4 * 1 * 14336 / 1
        self.assertAlmostEqual(result, expected, places=0)

    def test_ulysses_returns_nonzero(self):
        """CM-CE02: Ulysses CP expert comm uses same formula."""
        ccfg = _make_ccfg(cp_algo="ulysses_cp", s=1024, b=4,
                          hff=14336, n_ffMM=1, comm_cp=1.0, tp=1)
        ctx = _make_ctx()
        result = EvalLayerComm.cp_comm_exp(ccfg, ctx)
        expected = 1.0 * 2 * 1024 * 4 * 1 * 14336 / 1
        self.assertAlmostEqual(result, expected, places=0)

    def test_unknown_algo_returns_zero(self):
        """CM-CE03: Unknown cp_algo returns 0 for expert comm."""
        ccfg = _make_ccfg(cp_algo="unknown_algo", s=1024, b=4,
                          hff=14336, n_ffMM=1, comm_cp=1.0, tp=1)
        ctx = _make_ctx()
        result = EvalLayerComm.cp_comm_exp(ccfg, ctx)
        self.assertEqual(result, 0)


class TestCpCommLayer(unittest.TestCase):
    """Test cp_comm_layer sums non_exp and exp."""

    def test_layer_sums_both(self):
        """CM-CL01: cp_comm_layer = cp_comm_non_exp + cp_comm_exp."""
        ccfg = _make_ccfg(cp_algo="colossalai_cp", p=1, s=1024, b=4,
                          h=4096, hff=14336, n_attMM=2, n_ffMM=1,
                          comm_cp=1.0, tp=1)
        ctx = _make_ctx(current_node=None)
        result = EvalLayerComm.cp_comm_layer(ccfg, ctx)
        non_exp = EvalLayerComm.cp_comm_non_exp(ccfg, ctx)
        exp = EvalLayerComm.cp_comm_exp(ccfg, ctx)
        self.assertAlmostEqual(result, non_exp + exp, places=0)


class TestEpCommLayerEarlyReturn(unittest.TestCase):
    """Test ep_comm_layer early return when ep <= 1."""

    def test_ep1_returns_zero(self):
        """CM-E10: ep_comm_layer returns 0 when ep=1."""
        ccfg = _make_ccfg(ep=1, comm_ep=1.0, tokens_per_expert=None)
        ctx = _make_ctx()
        result = EvalLayerComm.ep_comm_layer(ccfg, ctx, 1)
        self.assertEqual(result, 0)

    def test_comm_ep_zero_returns_zero(self):
        """CM-E11: ep_comm_layer returns 0 when comm_ep=0."""
        ccfg = _make_ccfg(ep=4, comm_ep=0, tokens_per_expert=None)
        ctx = _make_ctx()
        result = EvalLayerComm.ep_comm_layer(ccfg, ctx, 1)
        self.assertEqual(result, 0)


class TestCpCommBuffer(unittest.TestCase):
    """Test cp_comm_buffer (CP communication buffer memory estimation)."""

    def _make_ccfg_buffer(self, cp=2, s=1024, b=4, t=1, a=32,
                          n_kv=32, dh=128, kv_lora_rank=0, h=4096,
                          device_per_node=8, cp_algo="colossalai_cp"):
        """Create a mock CostModelConfig for CP comm buffer tests."""
        ccfg = MagicMock()
        ccfg.cp = cp
        ccfg.s = s
        ccfg.b = b
        ccfg.t = t
        ccfg.a = a
        ccfg.n_kv = n_kv
        ccfg.dh = dh
        ccfg.kv_lora_rank = kv_lora_rank
        ccfg.h = h
        ccfg.device_per_node = device_per_node
        ccfg.cp_algo = cp_algo
        return ccfg

    def test_cp1_returns_zero(self):
        """CM-CB01: cp=1 returns 0.0 (no CP → no buffer)."""
        ccfg = self._make_ccfg_buffer(cp=1)
        ctx = MagicMock()
        result = EvalLayerComm.cp_comm_buffer(ccfg, ctx)
        self.assertEqual(result, 0.0)

    def test_ring_cp_intra_node(self):
        """CM-CB02: Ring CP with cp <= device_per_node (intra-node only)."""
        ccfg = self._make_ccfg_buffer(cp=4, device_per_node=8,
                                      cp_algo="colossalai_cp")
        ctx = MagicMock()
        result = EvalLayerComm.cp_comm_buffer(ccfg, ctx)
        # MHA: kv_dim = h/t = 4096/1 = 4096
        # chunk = (s/cp) * b * kv_dim * kv_bytes = (1024/4) * 4 * 4096 * 4
        kv_dim = 4096 / 1
        chunk = (1024 / 4) * 4 * kv_dim * 4
        intra_ranks = 4
        extra_chunks = intra_ranks - 1  # 3
        expected = extra_chunks * chunk
        self.assertAlmostEqual(result, expected, places=0)

    def test_ring_cp_cross_node(self):
        """CM-CB03: Ring CP with cp > device_per_node (cross-node)."""
        ccfg = self._make_ccfg_buffer(cp=16, device_per_node=8,
                                      cp_algo="colossalai_cp")
        ctx = MagicMock()
        result = EvalLayerComm.cp_comm_buffer(ccfg, ctx)
        kv_dim = 4096 / 1
        chunk = (1024 / 16) * 4 * kv_dim * 4
        intra_ranks = 8  # min(16, 8)
        extra_chunks = 2 * 8 - 1  # 15
        expected = extra_chunks * chunk
        self.assertAlmostEqual(result, expected, places=0)

    def test_ulysses_cp_intra_node(self):
        """CM-CB04: Ulysses CP with cp <= device_per_node."""
        ccfg = self._make_ccfg_buffer(cp=4, device_per_node=8,
                                      cp_algo="ulysses_cp")
        ctx = MagicMock()
        result = EvalLayerComm.cp_comm_buffer(ccfg, ctx)
        kv_dim = 4096 / 1
        chunk = 1024 * 4 * (kv_dim / 4) * 4
        intra_ranks = 4
        extra_chunks = intra_ranks - 1  # 3
        expected = extra_chunks * chunk
        self.assertAlmostEqual(result, expected, places=0)

    def test_ulysses_cp_cross_node(self):
        """CM-CB05: Ulysses CP with cp > device_per_node."""
        ccfg = self._make_ccfg_buffer(cp=16, device_per_node=8,
                                      cp_algo="ulysses_cp")
        ctx = MagicMock()
        result = EvalLayerComm.cp_comm_buffer(ccfg, ctx)
        kv_dim = 4096 / 1
        chunk = 1024 * 4 * (kv_dim / 16) * 4
        intra_ranks = 8
        extra_chunks = 2 * 8 - 1  # 15
        expected = extra_chunks * chunk
        self.assertAlmostEqual(result, expected, places=0)

    def test_mla_kv_dim(self):
        """CM-CB06: MLA uses kv_lora_rank for kv_dim (not h/t)."""
        ccfg_ring = self._make_ccfg_buffer(
            cp=2, device_per_node=8, cp_algo="colossalai_cp",
            kv_lora_rank=512, n_kv=32, dh=128)
        ccfg_mha = self._make_ccfg_buffer(
            cp=2, device_per_node=8, cp_algo="colossalai_cp",
            kv_lora_rank=0, n_kv=32, dh=128)
        ctx = MagicMock()
        r_mla = EvalLayerComm.cp_comm_buffer(ccfg_ring, ctx)
        r_mha = EvalLayerComm.cp_comm_buffer(ccfg_mha, ctx)
        # MLA kv_dim=512, MHA kv_dim=4096 → different buffer sizes
        self.assertNotAlmostEqual(r_mla, r_mha, places=0)


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


class TestNodeComputeEvalRepr(unittest.TestCase):
    """Test NodeComputeEval and NodeDynEval __repr__ with compute field."""

    def test_compute_repr_all_none(self):
        """CT-Q01: NodeComputeEval with all None returns compute=None."""
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import NodeComputeEval
        comp = NodeComputeEval()
        self.assertEqual(repr(comp), "compute=None")

    def test_compute_repr_with_router(self):
        """CT-Q02: NodeComputeEval with router shows router qualname."""
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import NodeComputeEval

        def router_flops():
            return 0

        comp = NodeComputeEval(router=router_flops)
        repr_str = repr(comp)
        self.assertIn("compute.router=", repr_str)
        self.assertIn("router_flops", repr_str)
        self.assertNotIn("expert_balanced", repr_str)

    def test_compute_repr_all_set(self):
        """CT-Q03: NodeComputeEval with all fields shows all names."""
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import NodeComputeEval

        def r():
            return 0

        def eb():
            return 0

        def ei():
            return 0

        def se():
            return 0

        comp = NodeComputeEval(router=r, expert_balanced=eb,
                               expert_imbalanced=ei, shared_expert=se)
        repr_str = repr(comp)
        self.assertIn("compute.router=", repr_str)
        self.assertIn("compute.expert_balanced=", repr_str)
        self.assertIn("compute.expert_imbalanced=", repr_str)
        self.assertIn("compute.shared_expert=", repr_str)

    def test_dyn_repr_with_compute(self):
        """CT-Q04: NodeDynEval with compute includes compute in repr."""
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import (
            NodeDynEval, NodeCommEval, NodeComputeEval,
        )

        def act():
            return 0

        comm = NodeCommEval(dp=lambda: 0, tp=lambda: 0, cp=lambda: 0, ep=lambda: 0)
        compute = NodeComputeEval(router=lambda: 0)
        dyn = NodeDynEval(activation=act, comm=comm, compute=compute)
        repr_str = repr(dyn)
        self.assertIn("dyn.activation=", repr_str)
        self.assertIn("compute.router=", repr_str)

    def test_dyn_repr_without_compute(self):
        """CT-Q05: NodeDynEval without compute omits compute from repr."""
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import (
            NodeDynEval, NodeCommEval,
        )

        def act():
            return 0

        comm = NodeCommEval(dp=lambda: 0, tp=lambda: 0, cp=lambda: 0, ep=lambda: 0)
        dyn = NodeDynEval(activation=act, comm=comm, compute=None)
        repr_str = repr(dyn)
        self.assertIn("dyn.activation=", repr_str)
        for field in ("compute.router=", "compute.expert_balanced=",
                      "compute.expert_imbalanced=", "compute.shared_expert="):
            self.assertNotIn(field, repr_str)


if __name__ == "__main__":
    unittest.main()
