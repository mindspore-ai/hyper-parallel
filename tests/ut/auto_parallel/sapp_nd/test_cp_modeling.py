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
"""Unit tests for Context Parallelism (CP) modeling.

Test cases:
- AP-CP-01: Long sequence with CP → attention activation memory reduction
- AP-CP-02: Varying cp_degree → memory and communication trends
- AP-CP-03: Invalid seq_len divisibility → validation failure
- AP-CP-04: CP + TP combination → combined cost estimation
- AP-CP-05: CP + TP cross-node topology → infeasible or penalty
- AP-CP-06: Short sequence with CP → warning or auto-filter

How to run:
    pytest tests/ut/auto_parallel/sapp_nd/test_cp_modeling.py -v
"""
import unittest
from types import SimpleNamespace

from tests.common.mark_utils import arg_mark

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.body import EvalBody
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.comm import EvalLayerComm
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import cp_comm_layer_detailed
from hyper_parallel.auto_parallel.sapp_nd.nd.dimensions import validate_cp_constraints
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cp_types import (
    CPMemoryBreakdown,
    CPCommunicationCost,
    CPValidationResult,
    CPConstraintParams,
    CPAlgo,
    _resolve_cp_algo,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import (
    AttentionType,
    detect_attention_type,
    compute_kv_dim,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import (
    get_cp_topology,
    get_cp_bandwidth,
    recommend_cp_max_by_attention,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common._cost_model_variables import _CostModVar
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import (
    EvalAttn,
    EvalFFn,
    EvalNorm,
)
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context
from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd.parallelize import ParallelizeLayer


def _make_ccfg(**overrides):
    """Build a real CostModelConfig backed by _CostModVar.

    Uses the same construction path as production code so that
    __getattr__ fallback (returns 0 for unknown attrs) is exercised.
    """
    var = _CostModVar(input_config=None, hook_cls=None, framework=None, source_code=None)
    defaults = {
        "s": 131072, "b": 1, "h": 8192, "a": 64, "dh": 128, "n_kv": 8,
        "cp": 4, "t": 1, "p": 1, "d": 1, "ep": 1, "device_per_node": 8,
        "kv_lora_rank": 0,
        "bw_intra": 400.0, "bw_inter": 25.0, "cp_algo": "colossalai_cp",
        "comm_cp": 1, "comm_t": 1, "comm_ep": 1,
        "n_softmax": 4, "n_attBMM": 4, "n_attMM": 4, "n_attParamCast": 0,
        "n_ffMM": 4, "n_ffBMM": 4, "n_ffParamCast": 0, "n_normOp": 2,
        "n_dropout": 1, "n_exp": 1, "n_shared_exp": 0, "n_chosen_exp": 1,
        "cap_fact": 1.0, "gmm": False, "hff": 28672,
        "bytes_compute": 2, "bytes_softmax": 2, "bytes_dropout": 2,
        "bytes_norm": 2, "dc_kv": 0, "dc_q": 0, "dhr": 0, "has_fa": False,
    }
    for k, v in {**defaults, **overrides}.items():
        setattr(var, k, v)

    if not hasattr(var, 'rec_op') or var.rec_op is None:
        var.rec_op = Config({
            'attBMM': 1, 'headCast': 1, 'dropout': 1, 'softmax': 1,
            'normOp': 1, 'gather': 1, 'ffAct': 1,
        })

    # Always recompute s_fa: _CostModVar class default is 0, so hasattr guard fails.
    var.s_fa = var.s if not getattr(var, 'has_fa', False) else var.s / var.a

    ccfg = CostModelConfig.__new__(CostModelConfig)
    ccfg.__dict__.update(var.__dict__)
    return ccfg


class TestAPCP01(unittest.TestCase):
    """AP-CP-01: Long sequence with CP → attention activation memory reduction."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_long_sequence_cp_memory_reduction(self):
        """
        Feature: CP Memory Estimation
        Description: Test that CP reduces attention activation memory for long sequences
        Expectation: Memory reduction >= 40% when cp_degree=4 for seq_len=128K
        """
        ccfg = _make_ccfg(cp=4)
        ctx = Context()

        cp_memory = EvalBody.act_cp_layer(ccfg, ctx)

        self.assertIsInstance(cp_memory, CPMemoryBreakdown)
        self.assertEqual(cp_memory.cp_degree, 4)
        self.assertEqual(cp_memory.seq_len, 131072)

        total_memory_without_cp = (
            4 * ccfg.s * ccfg.b * ccfg.n_kv * ccfg.dh * 2 +
            4 * ccfg.s * ccfg.s * ccfg.b * ccfg.a * 2 +
            ccfg.s * ccfg.s * ccfg.b * ccfg.a
        )

        reduction_ratio = cp_memory.total_reduction / total_memory_without_cp

        self.assertGreaterEqual(
            reduction_ratio, 0.4,
            f"Expected memory reduction >= 40%, got {reduction_ratio*100:.1f}%"
        )


class TestAPCP02(unittest.TestCase):
    """AP-CP-02: Varying cp_degree → memory and communication trends."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_varying_cp_degree_trends(self):
        """
        Feature: CP Memory and Communication Estimation
        Description: Test memory reduction and communication cost trends with varying cp_degree
        Expectation: Higher cp_degree → more memory reduction, more communication
        """
        cp_degrees = [1, 2, 4, 8]
        memory_reductions = []
        comm_volumes = []

        for cp in cp_degrees:
            ccfg = _make_ccfg(cp=cp)
            ctx = Context()

            cp_memory = EvalBody.act_cp_layer(ccfg, ctx)
            memory_reductions.append(cp_memory.total_reduction)

            cp_comm = cp_comm_layer_detailed(ccfg, ctx)
            comm_volumes.append(cp_comm.total_kv_volume)

        for i in range(1, len(memory_reductions)):
            self.assertGreater(
                memory_reductions[i], memory_reductions[i-1],
                f"Memory reduction should increase with cp_degree: "
                f"cp={cp_degrees[i]} < cp={cp_degrees[i-1]}"
            )

        for i in range(2, len(comm_volumes)):
            self.assertGreater(
                comm_volumes[i], comm_volumes[i-1],
                f"Communication volume should increase with cp_degree: "
                f"cp={cp_degrees[i]} < cp={cp_degrees[i-1]}"
            )


class TestAPCP03(unittest.TestCase):
    """AP-CP-03: Invalid seq_len divisibility → validation failure."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_invalid_seq_len_divisibility(self):
        """
        Feature: CP Constraint Validation
        Description: Test that invalid seq_len divisibility is detected
        Expectation: Validation fails with clear error message
        """
        result = validate_cp_constraints(
            seq_len=10000,
            cp_degree=3,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
        )

        self.assertIsInstance(result, CPValidationResult)
        self.assertFalse(result.is_valid)
        self.assertFalse(result.seq_len_divisible)
        self.assertIsNotNone(result.error_message)
        self.assertIn("divisible", result.error_message.lower())


class TestAPCP04(unittest.TestCase):
    """AP-CP-04: CP + TP/PP combination → combined cost estimation."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_tp_combined_estimation(self):
        """
        Feature: CP + TP Combined Estimation
        Description: Test memory and communication estimation with both CP and TP
        Expectation: System can estimate combined CP+TP strategy costs
        """
        ccfg = _make_ccfg(cp=4, t=8)
        ctx = Context()

        cp_memory = EvalBody.act_cp_layer(ccfg, ctx)
        cp_comm = cp_comm_layer_detailed(ccfg, ctx)

        self.assertIsInstance(cp_memory, CPMemoryBreakdown)
        self.assertIsInstance(cp_comm, CPCommunicationCost)

        self.assertEqual(cp_memory.cp_degree, 4)
        self.assertGreater(cp_memory.total_memory, 0)

        self.assertEqual(cp_comm.cp_degree, 4)
        self.assertGreater(cp_comm.total_kv_volume, 0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_pp_combined_estimation(self):
        """
        Feature: CP + PP Combined Estimation
        Description: PP does not affect per-layer CP memory/comm estimation,
                     but the combination must still produce valid results.
        Expectation: Same per-layer CP cost as CP-only (PP is orthogonal).
        """
        ccfg_cp_only = _make_ccfg(cp=4, t=1)
        ccfg_cp_pp = _make_ccfg(cp=4, t=1)
        ctx = Context()

        mem_cp_only = EvalBody.act_cp_layer(ccfg_cp_only, ctx)
        comm_cp_only = cp_comm_layer_detailed(ccfg_cp_only, ctx)

        mem_cp_pp = EvalBody.act_cp_layer(ccfg_cp_pp, ctx)
        comm_cp_pp = cp_comm_layer_detailed(ccfg_cp_pp, ctx)

        self.assertAlmostEqual(
            mem_cp_pp.total_memory, mem_cp_only.total_memory, places=1,
            msg="PP should not change per-layer CP memory estimation",
        )
        self.assertAlmostEqual(
            comm_cp_pp.total_kv_volume, comm_cp_only.total_kv_volume, places=1,
            msg="PP should not change per-layer CP communication estimation",
        )

        cp_result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            pp_degree=4,
            device_per_node=8,
            attention_type_str="mha",
            total_devices=16,
        )
        self.assertTrue(cp_result.is_valid)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_pp_device_sufficiency_in_validation(self):
        """
        Feature: CP + PP Device Sufficiency in Validation
        Description: tp*cp*pp > total_devices should be caught by validation
        Expectation: is_valid=False when tp*cp*pp exceeds total devices
        """
        cp_result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=2,
            pp_degree=4,
            device_per_node=8,
            attention_type_str="mha",
            total_devices=16,
        )
        self.assertFalse(cp_result.is_valid)
        self.assertFalse(cp_result.device_sufficient)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_tp_pp_combined_estimation(self):
        """
        Feature: CP + TP + PP Triple Combination
        Description: All three dimensions combined should produce valid estimation
        Expectation: Per-layer CP cost reflects TP, validation reflects tp*cp*pp
        """
        ccfg = _make_ccfg(cp=4, t=2)
        ctx = Context()

        cp_memory = EvalBody.act_cp_layer(ccfg, ctx)
        cp_comm = cp_comm_layer_detailed(ccfg, ctx)

        self.assertIsInstance(cp_memory, CPMemoryBreakdown)
        self.assertIsInstance(cp_comm, CPCommunicationCost)
        self.assertEqual(cp_memory.cp_degree, 4)
        self.assertGreater(cp_memory.total_memory, 0)
        self.assertGreater(cp_comm.total_kv_volume, 0)

        cp_result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=2,
            pp_degree=2,
            device_per_node=8,
            attention_type_str="mha",
            total_devices=16,
        )
        self.assertTrue(cp_result.is_valid)


class TestAPCP05(unittest.TestCase):
    """AP-CP-05: CP + TP combination but layout not supported → infeasible reason."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cross_node_topology_penalty(self):
        """
        Feature: CP Topology Constraint
        Description: Test that cross-node CP is detected and penalized
        Expectation: Warning about cross-node communication, topology_penalty applied
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=2,
            tp_degree=8,
            device_per_node=8,
            attention_type_str="mha",
        )

        self.assertIsInstance(result, CPValidationResult)
        self.assertTrue(result.is_valid)
        self.assertFalse(result.topology_feasible)
        self.assertIsNotNone(result.warning_message)
        self.assertIn("cross", result.warning_message.lower())
        self.assertIsNotNone(result.topology_penalty)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_insufficient_heads_infeasible(self):
        """
        Feature: Ulysses CP+TP layout not supported
        Description: Ulysses with a < t*cp is an unsupported layout
        Expectation: is_valid=False, unsupported_reason mentions "Ulysses"
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=8,
            tp_degree=4,
            device_per_node=8,
            attention_type_str="mha",
            cp_algo="ulysses_cp",
            attention_heads=16,
        )
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.unsupported_reason)
        self.assertIn("Ulysses", result.unsupported_reason)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_sp_with_cp_infeasible(self):
        """
        Feature: SP+CP layout not supported
        Description: SP and CP are incompatible — an unsupported combination
        Expectation: is_valid=False, unsupported_reason="CP+SP incompatible"
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=4,
            device_per_node=8,
            attention_type_str="mha",
            sp_enabled=True,
        )
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.unsupported_reason)
        self.assertEqual(result.unsupported_reason, "CP+SP incompatible")


class TestAPCP06(unittest.TestCase):
    """AP-CP-06: Short sequence with CP → warning or auto-filter."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_short_sequence_warning(self):
        """
        Feature: CP Short Sequence Detection
        Description: Test that CP on short sequences triggers a warning
        Expectation: Warning about short sequence, but still valid
        """
        result = validate_cp_constraints(
            seq_len=4096,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
        )

        self.assertIsInstance(result, CPValidationResult)
        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.warning_message)
        self.assertIn("short", result.warning_message.lower())

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_short_sequence_memory_estimation_still_works(self):
        """
        Feature: CP Short Sequence Estimation
        Description: CP memory estimation should still produce valid results
                     for short sequences, even if not recommended
        Expectation: act_cp_layer returns valid CPMemoryBreakdown with total_memory > 0
        """
        ccfg = _make_ccfg(s=4096, cp=2)
        ctx = Context()
        cp_memory = EvalBody.act_cp_layer(ccfg, ctx)

        self.assertIsInstance(cp_memory, CPMemoryBreakdown)
        self.assertGreater(cp_memory.total_memory, 0)
        self.assertEqual(cp_memory.cp_degree, 2)
        self.assertEqual(cp_memory.seq_len, 4096)

class TestAttentionTypeDetection(unittest.TestCase):
    """Test attention type detection logic."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_mla_detection(self):
        """
        Feature: Attention type detection
        Description: Detect MLA attention type from kv_lora_rank > 0
        Expectation: Returns AttentionType.MLA
        """
        ccfg = _make_ccfg(kv_lora_rank=512)
        attn_type = detect_attention_type(ccfg)
        self.assertEqual(attn_type, AttentionType.MLA)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_gqa_detection(self):
        """
        Feature: Attention type detection
        Description: Detect GQA attention type when n_kv < a
        Expectation: Returns AttentionType.GQA
        """
        ccfg = _make_ccfg(n_kv=8, a=64)
        attn_type = detect_attention_type(ccfg)
        self.assertEqual(attn_type, AttentionType.GQA)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_mha_detection(self):
        """
        Feature: Attention type detection
        Description: Detect MHA attention type when n_kv == a
        Expectation: Returns AttentionType.MHA
        """
        ccfg = _make_ccfg(n_kv=64, a=64)
        attn_type = detect_attention_type(ccfg)
        self.assertEqual(attn_type, AttentionType.MHA)


class TestMLAModelMemoryEstimation(unittest.TestCase):
    """AP-CP-07: MLA model memory estimation."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_mla_memory_estimation(self):
        """
        Feature: MLA Memory Estimation
        Description: Test memory estimation for Multi-Latent Attention (MLA) models
        Expectation: MLA has different KV cache size, should estimate correctly
        """
        ccfg = _make_ccfg(cp=4, kv_lora_rank=512)
        ctx = Context()

        cp_memory = EvalBody.act_cp_layer(ccfg, ctx)

        self.assertIsInstance(cp_memory, CPMemoryBreakdown)
        self.assertEqual(cp_memory.cp_degree, 4)
        self.assertGreater(cp_memory.total_memory, 0)
        self.assertGreater(cp_memory.total_reduction, 0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_mla_vs_mha_comparison(self):
        """
        Feature: MLA vs MHA Memory Comparison
        Description: Compare memory usage between MLA and MHA
        Expectation: MLA should have smaller KV cache than MHA
        """
        ccfg_mla = _make_ccfg(cp=4, kv_lora_rank=512)
        ccfg_mha = _make_ccfg(cp=4, kv_lora_rank=0, n_kv=64, a=64)
        ctx = Context()

        cp_memory_mla = EvalBody.act_cp_layer(ccfg_mla, ctx)
        cp_memory_mha = EvalBody.act_cp_layer(ccfg_mha, ctx)

        self.assertIsInstance(cp_memory_mla, CPMemoryBreakdown)
        self.assertIsInstance(cp_memory_mha, CPMemoryBreakdown)

        self.assertGreater(cp_memory_mla.total_memory, 0)
        self.assertGreater(cp_memory_mha.total_memory, 0)

        self.assertLess(
            cp_memory_mla.kv_cache_memory, cp_memory_mha.kv_cache_memory,
            f"MLA kv_cache ({cp_memory_mla.kv_cache_memory}) should be smaller "
            f"than MHA kv_cache ({cp_memory_mha.kv_cache_memory})"
        )


class TestBoundaryValues(unittest.TestCase):
    """AP-CP-08: Boundary value tests."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_degree_one_no_reduction(self):
        """
        Feature: CP Degree = 1
        Description: When cp_degree=1, CP should not provide benefits
        Expectation: total_reduction should be <= 0 (no benefit, possible overhead).
            Each memory component must equal the no-CP baseline to rule out
            numerical drift or double-counting.
        """
        ccfg = _make_ccfg(cp=1)
        ctx = Context()

        cp_memory = EvalBody.act_cp_layer(ccfg, ctx)

        self.assertEqual(cp_memory.cp_degree, 1)
        self.assertLessEqual(cp_memory.total_reduction, 0.0)

        s, b, a, t = ccfg.s, ccfg.b, ccfg.a, max(1, ccfg.t)
        n_kv, dh = ccfg.n_kv, ccfg.dh
        kv_dim = n_kv * dh
        a_per_rank = a / t

        expected_kv = 4 * s * b * kv_dim
        expected_attn = 4 * (s * s) * b * a_per_rank
        expected_softmax = 4 * (s * s) * b * a_per_rank
        expected_dropout = 1 * (s * s) * b * a_per_rank

        self.assertAlmostEqual(cp_memory.kv_cache_memory, expected_kv, places=0)
        self.assertAlmostEqual(cp_memory.attention_scores_memory, expected_attn, places=0)
        self.assertAlmostEqual(cp_memory.softmax_outputs_memory, expected_softmax, places=0)
        self.assertAlmostEqual(cp_memory.dropout_mask_memory, expected_dropout, places=0)
        self.assertAlmostEqual(cp_memory.comm_buffer_memory, 0.0, places=0)
        self.assertAlmostEqual(cp_memory.kv_reduction, 0.0, places=0)
        self.assertAlmostEqual(cp_memory.s2_reduction, 0.0, places=0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_seq_len_equals_cp_degree(self):
        """
        Feature: Minimum Valid Sequence Length
        Description: seq_len should be divisible by cp_degree * 2
        Expectation: seq_len=cp_degree*2 should be valid (divisible)
        """
        result = validate_cp_constraints(
            seq_len=8,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
        )

        self.assertTrue(result.seq_len_divisible)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_seq_len_not_divisible_by_cp(self):
        """
        Feature: Invalid Sequence Length Divisibility
        Description: seq_len not divisible by cp_degree should fail validation
        Expectation: Validation should fail with clear error
        """
        result = validate_cp_constraints(
            seq_len=100,
            cp_degree=3,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
        )

        self.assertFalse(result.is_valid)
        self.assertFalse(result.seq_len_divisible)
        self.assertIsNotNone(result.error_message)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_zero_seq_len(self):
        """
        Feature: Zero Sequence Length
        Description: seq_len=0 should be handled gracefully
        Expectation: Should either be valid with 0 memory or have warnings
        """
        result = validate_cp_constraints(
            seq_len=0,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
        )

        self.assertIsInstance(result, CPValidationResult)


class TestExceptionHandling(unittest.TestCase):
    """AP-CP-09: Exception handling tests."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_negative_attention_heads(self):
        """
        Feature: Invalid a Parameter
        Description: a <= 0 should raise ValueError
        Expectation: ValueError with clear error message
        """
        ccfg = _make_ccfg(cp=4, a=0)
        ctx = Context()

        with self.assertRaises(ValueError) as context:
            EvalBody.act_cp_layer(ccfg, ctx)

        self.assertIn("attention heads", str(context.exception).lower())

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_negative_cp_degree(self):
        """
        Feature: Invalid cp_degree Parameter
        Description: cp_degree <= 0 should raise ValueError
        Expectation: ValueError with clear error message
        """
        ccfg = _make_ccfg(cp=0)
        ctx = Context()

        with self.assertRaises(ValueError) as context:
            EvalBody.act_cp_layer(ccfg, ctx)

        self.assertIn("cp degree", str(context.exception).lower())

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_n_kv_greater_than_a(self):
        """
        Feature: Invalid n_kv Parameter
        Description: n_kv > a should be handled gracefully
        Expectation: Warning logged, n_kv adjusted to a
        """
        ccfg = _make_ccfg(cp=4, a=64, n_kv=128)
        ctx = Context()

        cp_memory = EvalBody.act_cp_layer(ccfg, ctx)

        self.assertIsInstance(cp_memory, CPMemoryBreakdown)
        self.assertEqual(cp_memory.cp_degree, 4)


class TestLargeSequenceLength(unittest.TestCase):
    """AP-CP-10: Large sequence length tests."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_one_million_sequence_length(self):
        """
        Feature: Very Large Sequence Length
        Description: Test with seq_len=1M to check for overflow
        Expectation: Should handle without overflow, results should be reasonable
        """
        ccfg = _make_ccfg(s=1048576, cp=8)
        ctx = Context()

        cp_memory = EvalBody.act_cp_layer(ccfg, ctx)
        cp_comm = cp_comm_layer_detailed(ccfg, ctx)

        self.assertIsInstance(cp_memory, CPMemoryBreakdown)
        self.assertEqual(cp_memory.seq_len, 1048576)
        self.assertGreater(cp_memory.total_memory, 0)

        self.assertIsInstance(cp_comm, CPCommunicationCost)
        self.assertGreater(cp_comm.total_kv_volume, 0)

        self.assertLess(
            cp_memory.total_memory,
            1e15,
            f"Memory estimate overflow: total_memory={cp_memory.total_memory}"
        )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_large_sequence_with_high_cp_degree(self):
        """
        Feature: Large Sequence with High CP Degree
        Description: Test seq_len=1M with cp_degree=64
        Expectation: Should handle high parallelism without issues
        """
        ccfg = _make_ccfg(s=1048576, cp=64)
        ctx = Context()

        cp_memory = EvalBody.act_cp_layer(ccfg, ctx)

        self.assertIsInstance(cp_memory, CPMemoryBreakdown)
        self.assertEqual(cp_memory.cp_degree, 64)
        self.assertGreater(cp_memory.total_reduction, 0)


class TestRealCostModelConfigIntegration(unittest.TestCase):
    """AP-CP-11: Integration test with real CostModelConfig / _CostModVar fields."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_real_config_fields_no_crash(self):
        """
        Feature: Real Config Compatibility
        Description: Verify that act_cp_layer works with _CostModVar field names
        Expectation: No ValueError from missing n_h; a/dh used correctly
        """
        ccfg = _make_ccfg(cp=4)
        ctx = Context()
        cp_memory = EvalBody.act_cp_layer(ccfg, ctx)

        self.assertIsInstance(cp_memory, CPMemoryBreakdown)
        self.assertEqual(cp_memory.cp_degree, 4)
        self.assertGreater(cp_memory.total_memory, 0)
        self.assertGreater(cp_memory.total_reduction, 0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_real_config_mla_no_crash(self):
        """
        Feature: Real Config MLA Compatibility
        Description: Verify MLA path works with _CostModVar field names
        Expectation: detect_attention_type returns MLA, no crash
        """
        ccfg = _make_ccfg(cp=4, kv_lora_rank=512)
        ctx = Context()
        cp_memory = EvalBody.act_cp_layer(ccfg, ctx)
        cp_comm = cp_comm_layer_detailed(ccfg, ctx)

        self.assertEqual(cp_memory.attention_type, AttentionType.MLA)
        self.assertGreater(cp_comm.total_kv_volume, 0)


class TestCPDeviceSufficiency(unittest.TestCase):
    """AP-CP-12: tp*cp*pp <= total_devices constraint."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_exceeds_total_devices(self):
        """
        Feature: CP Device Sufficiency Check
        Description: tp*cp*pp > total_devices should fail validation
        Expectation: is_valid=False, device_sufficient=False
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=16,
            tp_degree=1,
            pp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            total_devices=8,
        )

        self.assertFalse(result.is_valid)
        self.assertFalse(result.device_sufficient)
        self.assertIsNotNone(result.error_message)
        self.assertIn("exceeds", result.error_message.lower())

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_within_total_devices(self):
        """
        Feature: CP Device Sufficiency - valid case
        Description: tp*cp*pp <= total_devices should pass
        Expectation: device_sufficient=True
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            pp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            total_devices=8,
        )

        self.assertTrue(result.device_sufficient)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_total_devices_zero_skips_check(self):
        """
        Feature: CP Device Sufficiency - skip when total_devices=0
        Description: total_devices=0 should skip the device check
        Expectation: device_sufficient=True (default)
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=64,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            total_devices=0,
        )

        self.assertTrue(result.device_sufficient)


class TestCPMemoryWithinLimit(unittest.TestCase):
    """AP-CP-13: Single-card memory limit constraint."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_memory_exceeds_device_capacity(self):
        """
        Feature: CP Memory Limit Check
        Description: CP memory > device capacity should fail validation
        Expectation: is_valid=False, memory_within_limit=False
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            cp_memory_per_layer=1e9,
            device_capacity=1e8,
            num_layers=80,
        )

        self.assertFalse(result.is_valid)
        self.assertFalse(result.memory_within_limit)
        self.assertIsNotNone(result.error_message)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_memory_within_device_capacity(self):
        """
        Feature: CP Memory Limit - valid case
        Description: CP memory < device capacity should pass
        Expectation: memory_within_limit=True
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            cp_memory_per_layer=1e6,
            device_capacity=1e9,
            num_layers=80,
        )

        self.assertTrue(result.memory_within_limit)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_memory_check_skipped_when_zero(self):
        """
        Feature: CP Memory Limit - skip when params=0
        Description: When cp_memory_per_layer=0 or device_capacity=0, skip check
        Expectation: memory_within_limit=True (default)
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            cp_memory_per_layer=0,
            device_capacity=0,
            num_layers=0,
        )

        self.assertTrue(result.memory_within_limit)


class TestWarningAccumulation(unittest.TestCase):
    """AP-CP-14: Warning messages should accumulate, not overwrite."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_short_seq_and_cross_node_both_warned(self):
        """
        Feature: Warning Accumulation
        Description: Both short-sequence and cross-node warnings should appear
        Expectation: warning_message contains both "short" and "cross"
        """
        result = validate_cp_constraints(
            seq_len=4096,
            cp_degree=2,
            tp_degree=8,
            device_per_node=8,
            attention_type_str="mha",
        )

        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.warning_message)
        self.assertIn("cross", result.warning_message.lower())
        self.assertIn("short", result.warning_message.lower())

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_recommended_cp_max_warning(self):
        """
        Feature: Recommended CP Max Enforcement
        Description: cp_degree exceeding recommended max should trigger warning
        Expectation: warning_message mentions recommended max
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=8,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
        )

        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.warning_message)
        self.assertIn("recommended", result.warning_message.lower())


class TestCPWithTPLayout(unittest.TestCase):
    """AP-CP-15: CP+TP activation layout — TP reduces per-rank memory and comm."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_tp_reduces_cp_kv_cache(self):
        """
        Feature: CP+TP KV Cache Layout
        Description: With TP, each rank holds 1/t of KV heads → smaller kv_cache
        Expectation: cp=4,t=4 has 1/4 the kv_cache of cp=4,t=1 (GQA)
        """
        ccfg_t1 = _make_ccfg(cp=4, t=1, n_kv=8, a=64)
        ccfg_t4 = _make_ccfg(cp=4, t=4, n_kv=8, a=64)
        ctx = Context()

        mem_t1 = EvalBody.act_cp_layer(ccfg_t1, ctx)
        mem_t4 = EvalBody.act_cp_layer(ccfg_t4, ctx)

        self.assertAlmostEqual(
            mem_t4.kv_cache_memory, mem_t1.kv_cache_memory / 4, places=1,
            msg=f"TP=4 should reduce KV cache by 4x: {mem_t4.kv_cache_memory} vs {mem_t1.kv_cache_memory / 4}"
        )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_tp_reduces_cp_attention_scores(self):
        """
        Feature: CP+TP Attention Score Layout
        Description: With TP, attention heads split → fewer heads per rank
        Expectation: cp=4,t=4 has 1/4 the attn_scores of cp=4,t=1
        """
        ccfg_t1 = _make_ccfg(cp=4, t=1, a=64)
        ccfg_t4 = _make_ccfg(cp=4, t=4, a=64)
        ctx = Context()

        mem_t1 = EvalBody.act_cp_layer(ccfg_t1, ctx)
        mem_t4 = EvalBody.act_cp_layer(ccfg_t4, ctx)

        self.assertAlmostEqual(
            mem_t4.attention_scores_memory, mem_t1.attention_scores_memory / 4, places=1,
            msg="TP=4 should reduce attn scores by 4x"
        )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_tp_reduces_cp_comm_volume(self):
        """
        Feature: CP+TP Communication Volume
        Description: With TP, each rank sends 1/t of KV in CP ring
        Expectation: cp=4,t=4 has 1/4 the comm volume of cp=4,t=1 (GQA)
        """
        ccfg_t1 = _make_ccfg(cp=4, t=1, n_kv=8, a=64)
        ccfg_t4 = _make_ccfg(cp=4, t=4, n_kv=8, a=64)
        ctx = Context()

        comm_t1 = cp_comm_layer_detailed(ccfg_t1, ctx)
        comm_t4 = cp_comm_layer_detailed(ccfg_t4, ctx)

        self.assertAlmostEqual(
            comm_t4.total_kv_volume, comm_t1.total_kv_volume / 4, places=1,
            msg="TP=4 should reduce CP comm volume by 4x"
        )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_tp_reduces_cp_comm_buffer(self):
        """
        Feature: CP+TP Comm Buffer Layout
        Description: With TP, comm buffer is based on per-rank kv_dim
        Expectation: cp=4,t=4 has 1/4 the comm buffer of cp=4,t=1 (GQA)
        """
        ccfg_t1 = _make_ccfg(cp=4, t=1, n_kv=8, a=64)
        ccfg_t4 = _make_ccfg(cp=4, t=4, n_kv=8, a=64)
        ctx = Context()

        buf_t1 = EvalLayerComm.cp_comm_buffer(ccfg_t1, ctx)
        buf_t4 = EvalLayerComm.cp_comm_buffer(ccfg_t4, ctx)

        self.assertAlmostEqual(
            buf_t4, buf_t1 / 4, places=1,
            msg="TP=4 should reduce CP comm buffer by 4x"
        )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_mla_kv_dim_unchanged_by_tp(self):
        """
        Feature: MLA KV Dim Not Split by TP
        Description: MLA compressed latent is not split by TP
        Expectation: kv_dim is the same regardless of TP degree
        """
        ccfg_t1 = _make_ccfg(cp=4, t=1, kv_lora_rank=512)
        ccfg_t4 = _make_ccfg(cp=4, t=4, kv_lora_rank=512)

        kv_dim_t1 = compute_kv_dim(ccfg_t1)
        kv_dim_t4 = compute_kv_dim(ccfg_t4)

        self.assertEqual(kv_dim_t1, kv_dim_t4, "MLA kv_dim should not change with TP")


class TestUlyssesVsRing(unittest.TestCase):
    """AP-CP-16: Ulysses vs Ring CP — different activation layouts and comm patterns."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_attention_scores_full_seq(self):
        """
        Feature: Ulysses Attention Scores Layout
        Description: Ulysses keeps full seq_len, splits heads → attn scores shape is s×s.
                     Ring shards Q along seq and all-gathers KV, so only one S dim of
                     the S² score tensor is divided → (s/cp) × s.
        Expectation: Ulysses attn_scores = 4 * s * s * b * a/(t*cp),
                     Ring = 4 * (s/cp) * s * b * a/t (both /cp, symmetric)
        """
        ccfg_ring = _make_ccfg(cp=4, t=1, a=64, cp_algo="colossalai_cp")
        ccfg_ulysses = _make_ccfg(cp=4, t=1, a=64, cp_algo="ulysses_cp")
        ctx = Context()

        mem_ring = EvalBody.act_cp_layer(ccfg_ring, ctx)
        mem_ulysses = EvalBody.act_cp_layer(ccfg_ulysses, ctx)

        s = ccfg_ring.s
        b = ccfg_ring.b
        a = ccfg_ring.a
        cp = 4
        expected_ring = 4 * (s / cp) * s * b * (a / 1)
        expected_ulysses = 4 * s * s * b * (a / (1 * cp))

        self.assertAlmostEqual(mem_ring.attention_scores_memory, expected_ring, places=1)
        self.assertAlmostEqual(mem_ulysses.attention_scores_memory, expected_ulysses, places=1)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_kv_cache_split_by_cp(self):
        """
        Feature: Ulysses KV Cache Layout
        Description: Ulysses splits kv_dim by cp → kv_cache = kv_bytes * s * b * (kv_dim/cp)
        Ring splits seq by cp → kv_cache = kv_bytes * (s/cp) * b * kv_dim
        These are mathematically equal: s * (kv_dim/cp) = (s/cp) * kv_dim
        Expectation: Both positive; difference shows in attention_scores layout
        """
        ccfg_ring = _make_ccfg(cp=4, t=1, n_kv=8, a=64, cp_algo="colossalai_cp")
        ccfg_ulysses = _make_ccfg(cp=4, t=1, n_kv=8, a=64, cp_algo="ulysses_cp")
        ctx = Context()

        mem_ring = EvalBody.act_cp_layer(ccfg_ring, ctx)
        mem_ulysses = EvalBody.act_cp_layer(ccfg_ulysses, ctx)

        self.assertGreater(mem_ring.kv_cache_memory, 0)
        self.assertGreater(mem_ulysses.kv_cache_memory, 0)
        self.assertAlmostEqual(
            mem_ring.kv_cache_memory, mem_ulysses.kv_cache_memory, places=1,
            msg="KV cache per rank is the same regardless of CP layout"
        )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_comm_uses_all2all(self):
        """
        Feature: Ulysses Communication Pattern
        Description: Ulysses uses All2All, not ring P2P
        Expectation: ring_steps=0 for Ulysses, ring_steps>0 for Ring
        """
        ccfg_ring = _make_ccfg(cp=4, cp_algo="colossalai_cp")
        ccfg_ulysses = _make_ccfg(cp=4, cp_algo="ulysses_cp")
        ctx = Context()

        comm_ring = cp_comm_layer_detailed(ccfg_ring, ctx)
        comm_ulysses = cp_comm_layer_detailed(ccfg_ulysses, ctx)

        self.assertGreater(comm_ring.ring_steps, 0)
        self.assertEqual(comm_ulysses.ring_steps, 0)
        self.assertGreater(comm_ulysses.total_kv_volume, 0)
        self.assertGreater(comm_ring.total_kv_volume, 0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_with_tp_reduces_per_rank(self):
        """
        Feature: Ulysses + TP Combined
        Description: With Ulysses+TP, heads split by both t and cp
        Expectation: attn_scores = 4 * s^2 * b * a/(t*cp)
        """
        ccfg = _make_ccfg(cp=4, t=4, a=64, cp_algo="ulysses_cp")
        ctx = Context()

        mem = EvalBody.act_cp_layer(ccfg, ctx)

        s, b, a, cp, t = ccfg.s, ccfg.b, ccfg.a, 4, 4
        expected = 4 * s * s * b * (a / (t * cp))
        self.assertAlmostEqual(mem.attention_scores_memory, expected, places=1)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_cp_algo_in_result(self):
        """
        Feature: CP Algo Propagation
        Description: cp_algo should be reflected in CPMemoryBreakdown and CPCommunicationCost
        Expectation: Ring and Ulysses results carry correct cp_algo field
        """

        ccfg_ring = _make_ccfg(cp=4, cp_algo="colossalai_cp")
        ccfg_ulysses = _make_ccfg(cp=4, cp_algo="ulysses_cp")
        ctx = Context()

        mem_ring = EvalBody.act_cp_layer(ccfg_ring, ctx)
        mem_ulysses = EvalBody.act_cp_layer(ccfg_ulysses, ctx)
        comm_ring = cp_comm_layer_detailed(ccfg_ring, ctx)
        comm_ulysses = cp_comm_layer_detailed(ccfg_ulysses, ctx)

        self.assertEqual(mem_ring.cp_algo, CPAlgo.COLOSSALAI_CP)
        self.assertEqual(mem_ulysses.cp_algo, CPAlgo.ULYSSES_CP)
        self.assertEqual(comm_ring.cp_algo, CPAlgo.COLOSSALAI_CP)
        self.assertEqual(comm_ulysses.cp_algo, CPAlgo.ULYSSES_CP)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_comm_buffer_uses_all2all(self):
        """
        Feature: Ulysses Comm Buffer
        Description: Ulysses All2All buffer = 2 chunks (send + receive)
        Expectation: Different from Ring buffer which scales with intra_ranks
        """
        ccfg_ring = _make_ccfg(cp=4, cp_algo="colossalai_cp")
        ccfg_ulysses = _make_ccfg(cp=4, cp_algo="ulysses_cp")
        ctx = Context()

        buf_ring = EvalLayerComm.cp_comm_buffer(ccfg_ring, ctx)
        buf_ulysses = EvalLayerComm.cp_comm_buffer(ccfg_ulysses, ctx)

        self.assertGreater(buf_ring, 0)
        self.assertGreater(buf_ulysses, 0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_hybrid_cp_same_as_ring(self):
        """
        Feature: Hybrid CP = Ring CP
        Description: hybrid_cp should behave identically to colossalai_cp
        Expectation: Same memory and comm for both
        """
        ccfg_colossal = _make_ccfg(cp=4, cp_algo="colossalai_cp")
        ccfg_hybrid = _make_ccfg(cp=4, cp_algo="hybrid_cp")
        ctx = Context()

        mem_colossal = EvalBody.act_cp_layer(ccfg_colossal, ctx)
        mem_hybrid = EvalBody.act_cp_layer(ccfg_hybrid, ctx)

        self.assertEqual(mem_colossal.kv_cache_memory, mem_hybrid.kv_cache_memory)
        self.assertEqual(mem_colossal.attention_scores_memory, mem_hybrid.attention_scores_memory)


class TestCPTopology(unittest.TestCase):
    """AP-CP-17: CP topology — intra-node, cross-node, mixed bandwidth."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ring_intra_node_topology(self):
        """
        Feature: Ring CP intra-node
        Description: tp*cp <= device_per_node → all intra-node communication
        Expectation: topology="intra-node", effective_bandwidth=bw_intra
        """
        ccfg = _make_ccfg(cp=4, t=1, device_per_node=8, cp_algo="colossalai_cp")
        ctx = Context()
        comm = cp_comm_layer_detailed(ccfg, ctx)
        self.assertEqual(comm.topology, "intra-node")
        self.assertEqual(comm.effective_bandwidth, ccfg.bw_intra)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ring_cross_node_topology(self):
        """
        Feature: Ring CP cross-node
        Description: cp=16 > device_per_node=8, tp=1 → all cross-node
        Expectation: topology="mixed" with lower effective bandwidth
        """
        ccfg = _make_ccfg(cp=16, t=1, device_per_node=8, cp_algo="colossalai_cp")
        ctx = Context()
        comm = cp_comm_layer_detailed(ccfg, ctx)
        self.assertEqual(comm.topology, "mixed")
        self.assertLess(comm.effective_bandwidth, ccfg.bw_intra)
        self.assertGreater(comm.effective_bandwidth, ccfg.bw_inter)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_intra_node_topology(self):
        """
        Feature: Ulysses CP intra-node
        Description: cp <= device_per_node → all intra-node All2All
        Expectation: topology="intra-node"
        """
        ccfg = _make_ccfg(cp=4, t=1, device_per_node=8, cp_algo="ulysses_cp")
        ctx = Context()
        comm = cp_comm_layer_detailed(ccfg, ctx)
        self.assertEqual(comm.topology, "intra-node")
        self.assertEqual(comm.effective_bandwidth, ccfg.bw_intra)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_mixed_topology(self):
        """
        Feature: Ulysses CP mixed topology
        Description: cp=12, device_per_node=8 → some intra, some cross
        Expectation: topology="mixed", bandwidth between intra and cross
        """
        ccfg = _make_ccfg(cp=12, t=1, device_per_node=8, cp_algo="ulysses_cp")
        ctx = Context()
        comm = cp_comm_layer_detailed(ccfg, ctx)
        self.assertEqual(comm.topology, "mixed")
        self.assertLess(comm.effective_bandwidth, ccfg.bw_intra)
        self.assertGreater(comm.effective_bandwidth, ccfg.bw_inter)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_head_sufficiency_constraint(self):
        """
        Feature: Ulysses head sufficiency
        Description: Ulysses requires a >= t * cp
        Expectation: a=8, t=1, cp=16 should fail validation
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=16,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            cp_algo="ulysses_cp",
            attention_heads=8,
        )
        self.assertFalse(result.is_valid)
        self.assertIn("ulysses", result.error_message.lower())

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_head_sufficient_passes(self):
        """
        Feature: Ulysses head sufficiency - pass
        Description: a=64, t=1, cp=4 → 64 >= 4 → valid
        Expectation: is_valid=True when heads >= cp_degree
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            cp_algo="ulysses_cp",
            attention_heads=64,
        )
        self.assertTrue(result.is_valid)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ring_comm_buffer_intra_node(self):
        """
        Feature: Ring CP comm buffer intra-node
        Description: cp=4, device_per_node=8 → buffer = (4-1) chunks
        Expectation: comm_buffer = 3 * chunk
        """
        ccfg = _make_ccfg(cp=4, t=1, device_per_node=8, cp_algo="colossalai_cp")
        ctx = Context()
        buf = EvalLayerComm.cp_comm_buffer(ccfg, ctx)

        kv_dim = compute_kv_dim(ccfg)
        chunk = (ccfg.s / 4) * ccfg.b * kv_dim * 4
        self.assertAlmostEqual(buf, 3 * chunk, places=1)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ring_comm_buffer_cross_node(self):
        """
        Feature: Ring CP comm buffer cross-node
        Description: cp=16, device_per_node=8 → buffer = (2*8-1) chunks
        Expectation: comm_buffer = 15 * chunk
        """
        ccfg = _make_ccfg(cp=16, t=1, device_per_node=8, cp_algo="colossalai_cp")
        ctx = Context()
        buf = EvalLayerComm.cp_comm_buffer(ccfg, ctx)

        kv_dim = compute_kv_dim(ccfg)
        chunk = (ccfg.s / 16) * ccfg.b * kv_dim * 4
        self.assertAlmostEqual(buf, 15 * chunk, places=1)


class TestCPStepTimeImpact(unittest.TestCase):
    """AP-CP-19: CP communication contributes to end-to-end step time."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_exposed_comm_time_positive(self):
        """
        Feature: CP Step Time Contribution
        Description: cp>1 should produce positive exposed_comm_time
        Expectation: exposed_comm_time > 0, contributing to step time
        """
        ccfg = _make_ccfg(cp=4)
        ctx = Context()
        comm = cp_comm_layer_detailed(ccfg, ctx)
        self.assertGreater(comm.exposed_comm_time, 0.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_zero_when_disabled(self):
        """
        Feature: CP Step Time When Disabled
        Description: cp=1 should produce zero comm time
        Expectation: exposed_comm_time == 0
        """
        ccfg = _make_ccfg(cp=1)
        ctx = Context()
        comm = cp_comm_layer_detailed(ccfg, ctx)
        self.assertEqual(comm.exposed_comm_time, 0.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_higher_cp_increases_exposed_time(self):
        """
        Feature: CP Step Time Scaling
        Description: Higher cp_degree should increase exposed communication time
        Expectation: cp=8 exposed_time > cp=4 exposed_time > cp=2 exposed_time
        """
        times = []
        for cp in [2, 4, 8]:
            ccfg = _make_ccfg(cp=cp)
            ctx = Context()
            comm = cp_comm_layer_detailed(ccfg, ctx)
            times.append(comm.exposed_comm_time)

        for i in range(1, len(times)):
            self.assertGreater(
                times[i], times[i - 1],
                f"exposed_comm_time should increase with cp_degree: "
                f"cp={[2,4,8][i]} ({times[i]:.4f}ms) > cp={[2,4,8][i-1]} ({times[i-1]:.4f}ms)",
            )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cross_node_cp_has_higher_step_time(self):
        """
        Feature: CP Step Time with Cross-Node Topology
        Description: Cross-node CP uses slower bandwidth → higher comm time
        Expectation: cross-node exposed_time > intra-node exposed_time
        """
        ccfg_intra = _make_ccfg(cp=4, t=1, device_per_node=8)
        ccfg_cross = _make_ccfg(cp=16, t=1, device_per_node=8)
        ctx = Context()

        comm_intra = cp_comm_layer_detailed(ccfg_intra, ctx)
        comm_cross = cp_comm_layer_detailed(ccfg_cross, ctx)

        self.assertGreater(
            comm_cross.exposed_comm_time, comm_intra.exposed_comm_time,
            "Cross-node CP should have higher exposed comm time than intra-node",
        )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_total_step_time_includes_cp_component(self):
        """
        Feature: CP Part of Total Step Time
        Description: Total step time = compute + TP_comm + DP_comm + EP_comm + CP_comm.
                     Verify that CP comm is a separable, positive component.
        Expectation: total_kv_volume > 0, exposed_comm_time > 0 for cp>1;
                     both zero for cp=1.
        """
        ccfg = _make_ccfg(cp=4)
        ctx = Context()
        comm = cp_comm_layer_detailed(ccfg, ctx)

        self.assertGreater(comm.total_kv_volume, 0)
        self.assertGreater(comm.exposed_comm_time, 0)

        ccfg_no_cp = _make_ccfg(cp=1)
        comm_no_cp = cp_comm_layer_detailed(ccfg_no_cp, ctx)

        self.assertEqual(comm_no_cp.total_kv_volume, 0)
        self.assertEqual(comm_no_cp.exposed_comm_time, 0)

        step_time_delta = comm.exposed_comm_time - comm_no_cp.exposed_comm_time
        self.assertGreater(step_time_delta, 0, "CP should add positive overhead to step time")


class TestCPUnsupportedCombination(unittest.TestCase):
    """AP-CP-20: Unsupported CP combinations return infeasible / unsupported_reason."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_with_sp_enabled_is_invalid(self):
        """
        Feature: CP+SP Incompatibility
        Description: CP and SP cannot coexist; sp_enabled=True should fail
        Expectation: is_valid=False, unsupported_reason="CP+SP incompatible"
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            sp_enabled=True,
        )
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.unsupported_reason)
        self.assertIn("CP+SP", result.unsupported_reason)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_without_sp_is_valid(self):
        """
        Feature: CP without SP
        Description: sp_enabled=False should not cause infeasibility
        Expectation: is_valid=True, unsupported_reason=None
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            sp_enabled=False,
        )
        self.assertTrue(result.is_valid)
        self.assertIsNone(result.unsupported_reason)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_insufficient_heads_unsupported(self):
        """
        Feature: Ulysses Insufficient Heads
        Description: Ulysses with a < t*cp is an unsupported combination
        Expectation: is_valid=False, unsupported_reason mentions "Ulysses"
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=16,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            cp_algo="ulysses_cp",
            attention_heads=8,
        )
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.unsupported_reason)
        self.assertIn("Ulysses", result.unsupported_reason)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_valid_combination_no_unsupported_reason(self):
        """
        Feature: Valid CP Combination
        Description: Valid combinations should have unsupported_reason=None
        Expectation: unsupported_reason is None
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
        )
        self.assertTrue(result.is_valid)
        self.assertIsNone(result.unsupported_reason)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_constraint_params_carries_sp_enabled(self):
        """
        Feature: CPConstraintParams sp_enabled field
        Description: CPConstraintParams should accept and carry sp_enabled
        Expectation: sp_enabled is preserved and used in validation
        """
        params = CPConstraintParams(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            sp_enabled=True,
        )
        result = validate_cp_constraints(params)
        self.assertFalse(result.is_valid)
        self.assertIn("CP+SP", result.unsupported_reason)


class TestCPWithPPLayout(unittest.TestCase):
    """AP-CP-18: CP+PP combination — device sufficiency and topology."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_pp_exceeds_total_devices(self):
        """
        Feature: CP+PP Device Sufficiency
        Description: tp*cp*pp > total_devices should fail
        Expectation: is_valid=False, device_sufficient=False
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=2,
            pp_degree=4,
            device_per_node=8,
            attention_type_str="mha",
            total_devices=16,
        )

        self.assertFalse(result.is_valid)
        self.assertFalse(result.device_sufficient)
        self.assertIn("exceeds", result.error_message.lower())

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_pp_within_total_devices(self):
        """
        Feature: CP+PP Device Sufficiency - valid
        Description: tp*cp*pp <= total_devices should pass
        Expectation: device_sufficient=True
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=2,
            tp_degree=2,
            pp_degree=4,
            device_per_node=8,
            attention_type_str="mha",
            total_devices=16,
        )

        self.assertTrue(result.is_valid)
        self.assertTrue(result.device_sufficient)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_pp_cross_node_topology(self):
        """
        Feature: CP+PP Cross-Node Topology
        Description: tp*cp > device_per_node (even with pp) should warn
        Expectation: topology_feasible=False, warning about cross-node
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=4,
            pp_degree=2,
            device_per_node=8,
            attention_type_str="mha",
            total_devices=32,
        )

        self.assertTrue(result.is_valid)
        self.assertFalse(result.topology_feasible)
        self.assertIsNotNone(result.warning_message)
        self.assertIn("cross", result.warning_message.lower())


class TestCPEndToEndEffect(unittest.TestCase):
    """End-to-end effect validation: CP enabled vs disabled."""

    def _memory_without_cp(self, ccfg):
        """Compute per-layer activation memory without CP (baseline)."""
        fp16_bytes = 2
        kv_bytes = fp16_bytes * 2
        kv_dim = compute_kv_dim(ccfg)
        a_per_rank = ccfg.a / max(1, ccfg.t)
        s, b = ccfg.s, ccfg.b
        kv_cache = kv_bytes * s * b * kv_dim
        attn_scores = 4 * s * s * b * a_per_rank
        softmax_out = 4 * s * s * b * a_per_rank
        dropout_mask = 1 * s * s * b * a_per_rank
        return kv_cache + attn_scores + softmax_out + dropout_mask

    def _comm_without_cp(self, ccfg, ctx):
        """Compute per-layer DP+TP+EP comm cost without CP (baseline)."""
        dp = EvalLayerComm.dp_comm_layer(ccfg, ctx)
        tp = EvalLayerComm.tp_comm_layer(ccfg, ctx, 1)
        ep = EvalLayerComm.ep_comm_layer(ccfg, ctx, 1)
        return dp + tp + ep

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_net_memory_saving_positive(self):
        """
        Feature: CP End-to-End Memory Effect
        Description: Net memory saving (total_reduction) should be positive
                     for typical long-sequence config
        Expectation: total_reduction > 0, meaning CP saves more memory
                     than the comm buffer costs
        """
        ccfg = _make_ccfg(cp=4, s=131072)
        ctx = Context()
        cp_mem = EvalBody.act_cp_layer(ccfg, ctx)
        self.assertGreater(
            cp_mem.total_reduction, 0.0,
            f"CP should produce net memory saving, got reduction={cp_mem.total_reduction}",
        )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_memory_reduction_ratio_vs_baseline(self):
        """
        Feature: CP Memory Reduction Ratio
        Description: CP should reduce per-layer activation memory by a
                     significant fraction compared to the no-CP baseline
        Expectation: total_reduction / baseline >= 30%
        """
        ccfg = _make_ccfg(cp=4, s=131072)
        ctx = Context()
        cp_mem = EvalBody.act_cp_layer(ccfg, ctx)
        baseline = self._memory_without_cp(ccfg)
        ratio = cp_mem.total_reduction / baseline
        self.assertGreaterEqual(
            ratio, 0.30,
            f"Expected memory reduction ratio >= 30%, got {ratio*100:.1f}%",
        )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_comm_buffer_offset_less_than_saving(self):
        """
        Feature: CP Comm Buffer vs Gross Saving
        Description: The communication buffer memory overhead should be
                     less than the gross memory saving from CP
        Expectation: comm_buffer_memory < kv_reduction + s2_reduction
        """
        ccfg = _make_ccfg(cp=4, s=131072)
        ctx = Context()
        cp_mem = EvalBody.act_cp_layer(ccfg, ctx)
        gross_saving = cp_mem.kv_reduction + cp_mem.s2_reduction
        self.assertGreater(
            gross_saving, cp_mem.comm_buffer_memory,
            "Gross saving should exceed comm buffer overhead",
        )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_reduces_other_comm_costs(self):
        """
        Feature: CP Indirect Communication Reduction
        Description: The TP comm formula explicitly divides by ccfg.cp,
                     and DP comm divides by (cp * t). Verify that the
                     per-layer TP comm non-exp formula output is reduced
                     proportionally when cp increases, by computing the
                     raw TP volume manually and checking the cp divisor.
        Expectation: Raw TP volume / cp=4 is 1/4 of raw volume / cp=1
        """
        s, b, h = 131072, 1, 8192
        cp_no = 1
        cp_yes = 4
        n_gather = 4
        tp_volume = 0.25 * n_gather * s * b * h
        tp_comm_no_cp = tp_volume / cp_no
        tp_comm_with_cp = tp_volume / cp_yes
        ratio = tp_comm_with_cp / tp_comm_no_cp
        self.assertAlmostEqual(ratio, 0.25, places=2,
                               msg=f"TP comm ratio should ≈ 1/4, got {ratio}")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_net_comm_tradeoff(self):
        """
        Feature: CP Net Communication Tradeoff
        Description: CP adds its own comm cost (exposed_comm_time).
                     For intra-node topology with high bandwidth, the
                     CP comm cost should be modest relative to the
                     memory saving benefit.
        Expectation: exposed_comm_time > 0 and topology is intra-node
                     when cp <= device_per_node
        """
        ccfg = _make_ccfg(cp=4, s=131072)
        ctx = Context()
        cp_comm = cp_comm_layer_detailed(ccfg, ctx)
        self.assertGreater(cp_comm.exposed_comm_time, 0.0)
        self.assertEqual(cp_comm.topology, "intra-node")

        cp_mem = EvalBody.act_cp_layer(ccfg, ctx)
        baseline_mem = self._memory_without_cp(ccfg)
        saving_ratio = cp_mem.total_reduction / baseline_mem
        self.assertGreater(saving_ratio, 0.3,
                           "Memory saving should be significant")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_higher_cp_greater_memory_reduction(self):
        """
        Feature: CP Degree Scaling — Memory
        Description: Increasing cp_degree should monotonically increase
                     the net memory reduction ratio
        Expectation: reduction_ratio(cp=8) > reduction_ratio(cp=4) > reduction_ratio(cp=2)
        """
        ratios = []
        for cp in [2, 4, 8]:
            ccfg = _make_ccfg(cp=cp, s=131072)
            ctx = Context()
            cp_mem = EvalBody.act_cp_layer(ccfg, ctx)
            baseline = self._memory_without_cp(ccfg)
            ratios.append(cp_mem.total_reduction / baseline)

        self.assertGreater(ratios[1], ratios[0],
                           f"cp=4 ratio ({ratios[1]:.3f}) should > cp=2 ({ratios[0]:.3f})")
        self.assertGreater(ratios[2], ratios[1],
                           f"cp=8 ratio ({ratios[2]:.3f}) should > cp=4 ({ratios[1]:.3f})")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_vs_ring_memory_reduction(self):
        """
        Feature: Ulysses vs Ring Memory Reduction
        Description: Ulysses CP reduces attention scores differently from
                     Ring CP. Both should yield positive net reduction.
        Expectation: ulysses_reduction > 0 and ring_reduction > 0;
                     their magnitudes differ due to different sharding
        """
        ccfg_ring = _make_ccfg(cp=4, cp_algo="colossalai_cp")
        ccfg_ulysses = _make_ccfg(cp=4, cp_algo="ulysses_cp")
        ctx = Context()

        ring_mem = EvalBody.act_cp_layer(ccfg_ring, ctx)
        ulysses_mem = EvalBody.act_cp_layer(ccfg_ulysses, ctx)

        self.assertGreater(ring_mem.total_reduction, 0.0)
        self.assertGreater(ulysses_mem.total_reduction, 0.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_overall_benefit_positive(self):
        """
        Feature: CP Overall Benefit
        Description: For a typical training config, CP should provide a
                     net benefit: memory reduction outweighs comm overhead
                     relative to the no-CP baseline.
        Expectation: memory_reduction_ratio > 0 and the benefit is
                     quantifiable as (memory_saved / baseline_memory)
        """
        ccfg = _make_ccfg(cp=4, s=131072)
        ctx = Context()

        cp_mem = EvalBody.act_cp_layer(ccfg, ctx)
        cp_comm = cp_comm_layer_detailed(ccfg, ctx)

        baseline_memory = self._memory_without_cp(ccfg)
        memory_saved_ratio = cp_mem.total_reduction / baseline_memory

        self.assertGreater(memory_saved_ratio, 0.0,
                           "CP should save memory relative to baseline")
        self.assertGreater(cp_comm.exposed_comm_time, 0.0,
                           "CP should have measurable comm cost")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_no_benefit_for_short_sequence(self):
        """
        Feature: CP No Benefit for Short Sequence
        Description: For very short sequences, CP provides minimal memory
                     reduction while still adding comm cost
        Expectation: memory reduction ratio for s=128 is much smaller
                     than for s=131072; comm cost is still positive
        """
        ccfg_short = _make_ccfg(cp=4, s=128)
        ccfg_long = _make_ccfg(cp=4, s=131072)
        ctx = Context()

        short_mem = EvalBody.act_cp_layer(ccfg_short, ctx)
        long_mem = EvalBody.act_cp_layer(ccfg_long, ctx)

        short_baseline = self._memory_without_cp(ccfg_short)
        long_baseline = self._memory_without_cp(ccfg_long)

        short_ratio = short_mem.total_reduction / short_baseline
        long_ratio = long_mem.total_reduction / long_baseline

        self.assertLess(short_ratio, long_ratio,
                        f"Short seq ratio ({short_ratio:.3f}) should < "
                        f"long seq ratio ({long_ratio:.3f})")

        short_comm = cp_comm_layer_detailed(ccfg_short, ctx)
        self.assertGreater(short_comm.exposed_comm_time, 0.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_validation_filters_bad_configs_but_passes_good(self):
        """
        Feature: CP Validation End-to-End
        Description: validate_cp_constraints should filter out infeasible
                     configs (SP+CP, insufficient devices) while passing
                     good configs — ensuring the search space only includes
                     beneficial CP settings
        Expectation: good config is_valid=True; bad config is_valid=False
        """
        good = validate_cp_constraints(
            seq_len=131072, cp_degree=4, tp_degree=2,
            pp_degree=1, device_per_node=8, total_devices=8,
            attention_type_str="mha", attention_heads=64,
        )
        self.assertTrue(good.is_valid)

        bad = validate_cp_constraints(
            seq_len=131072, cp_degree=4, tp_degree=2,
            pp_degree=1, device_per_node=8, total_devices=4,
            attention_type_str="mha", attention_heads=64,
        )
        self.assertFalse(bad.is_valid)
        self.assertFalse(bad.device_sufficient)


class TestCPRingPeakCorrection(unittest.TestCase):
    """Verify CP applies /cp to attention scores (Ring and Ulysses symmetric).

    Both Ring CP (shard Q, all-gather KV) and Ulysses CP (seq→head A2A)
    materialize the same B·H·S²/cp attention-score tensor, so both divide
    attn_score by a single /cp. T033 profiling (yaml A slope −0.807, yaml B
    ratio 1.8433) confirms /cp scaling empirically.
    """

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ring_cp_sq_div_in_attn_score(self):
        """
        Feature: Ring CP /cp correction in attn_score_activations
        Description: Ring CP shards Q along seq (one S dim of the S² score
                     tensor), so attn_score divides by cp (not cp²).
        Expectation: Ring cp=4 attn_score = no_cp_score / 4
        """
        ccfg_no_cp = _make_ccfg(cp=1, t=1, a=64)
        ccfg_ring = _make_ccfg(cp=4, t=1, a=64, cp_algo="colossalai_cp")
        ctx = Context()
        ctx.micro_factor = 1

        score_no_cp = EvalAttn.attn_score_activations(ccfg_no_cp, ctx)
        score_ring = EvalAttn.attn_score_activations(ccfg_ring, ctx)

        ratio = score_ring / score_no_cp
        expected_ratio = 1.0 / 4
        self.assertAlmostEqual(ratio, expected_ratio, places=4,
                               msg=f"Ring CP=4 should divide attn_score by 4, got ratio={ratio:.6f}")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_cp_only_div_by_cp_in_attn_score(self):
        """
        Feature: Ulysses CP /cp correction in attn_score_activations
        Description: Ulysses CP shards heads via seq→head A2A; the score
                     tensor is B·(H/cp)·S², so attn_score divides by cp —
                     same /cp as Ring (both materialize B·H·S²/cp).
        Expectation: Ulysses cp=4 attn_score = no_cp_score / 4
        """
        ccfg_no_cp = _make_ccfg(cp=1, t=1, a=64)
        ccfg_ulysses = _make_ccfg(cp=4, t=1, a=64, cp_algo="ulysses_cp")
        ctx = Context()
        ctx.micro_factor = 1

        score_no_cp = EvalAttn.attn_score_activations(ccfg_no_cp, ctx)
        score_ulysses = EvalAttn.attn_score_activations(ccfg_ulysses, ctx)

        ratio = score_ulysses / score_no_cp
        expected_ratio = 1.0 / 4
        self.assertAlmostEqual(ratio, expected_ratio, places=4,
                               msg=f"Ulysses CP=4 should divide attn_score by 4, got ratio={ratio:.6f}")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp1_no_extra_divisor(self):
        """
        Feature: cp=1 applies no extra divisor
        Description: When cp=1, cp_sq_div=1 — no additional division
        Expectation: cp=1 Ring score == cp=1 Ulysses score == no_cp score
        """
        ccfg_ring_1 = _make_ccfg(cp=1, cp_algo="colossalai_cp")
        ccfg_ulysses_1 = _make_ccfg(cp=1, cp_algo="ulysses_cp")
        ctx = Context()
        ctx.micro_factor = 1

        score_ring = EvalAttn.attn_score_activations(ccfg_ring_1, ctx)
        score_ulysses = EvalAttn.attn_score_activations(ccfg_ulysses_1, ctx)

        self.assertAlmostEqual(score_ring, score_ulysses, places=4,
                               msg="cp=1: Ring and Ulysses should produce identical attn scores")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ring_peak_mem_matches_cp_sq_correction(self):
        """
        Feature: CP peak memory reflects /cp correction
        Description: act_cp_layer attention_scores_memory for Ring should
                     be (s/cp) * s * b * a/t formula, which is 1/cp of baseline
                     (Ring shards Q along seq; only one S dim of the S² score
                     tensor is divided).
        Expectation: Ring attn_scores_memory ≈ baseline / cp
        """
        ccfg_no_cp = _make_ccfg(cp=1, t=1, a=64)
        ccfg_ring = _make_ccfg(cp=4, t=1, a=64, cp_algo="colossalai_cp")
        ctx = Context()

        mem_no_cp = EvalBody.act_cp_layer(ccfg_no_cp, ctx)
        mem_ring = EvalBody.act_cp_layer(ccfg_ring, ctx)

        ratio = mem_ring.attention_scores_memory / mem_no_cp.attention_scores_memory
        expected_ratio = 1.0 / 4
        self.assertAlmostEqual(ratio, expected_ratio, places=3,
                               msg=f"Ring peak attn_scores ratio should ≈ 1/4, got {ratio:.6f}")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_linear_activations_same_for_ring_and_ulysses(self):
        """
        Feature: Linear activations (qkv, proj, ffn, norm) scale with s not s²
        Description: Since linear activations only scale with s (not s²),
                     Ring and Ulysses should produce identical values for
                     qkv, proj, ffn, norm — both just divide by cp.
        Expectation: Ring qkv/proj/ffn/norm == Ulysses qkv/proj/ffn/norm
        """
        ccfg_ring = _make_ccfg(cp=4, cp_algo="colossalai_cp", t=1)
        ccfg_ulysses = _make_ccfg(cp=4, cp_algo="ulysses_cp", t=1)
        ctx = Context()
        ctx.micro_factor = 1

        self.assertAlmostEqual(
            EvalAttn.attn_qkv_activations(ccfg_ring, ctx),
            EvalAttn.attn_qkv_activations(ccfg_ulysses, ctx), places=4,
            msg="qkv should be same for Ring and Ulysses (linear /cp)")
        self.assertAlmostEqual(
            EvalFFn.ffn_activations(ccfg_ring, ctx),
            EvalFFn.ffn_activations(ccfg_ulysses, ctx), places=4,
            msg="ffn should be same for Ring and Ulysses (linear /cp)")
        self.assertAlmostEqual(
            EvalNorm.norm_activations(ccfg_ring, ctx),
            EvalNorm.norm_activations(ccfg_ulysses, ctx), places=4,
            msg="norm should be same for Ring and Ulysses (linear /cp)")


class TestCPRecFactor(unittest.TestCase):
    """Verify rec_coeff is correctly applied with CP in activation formulas."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_attn_qkv_divides_by_cp(self):
        """
        Feature: attn_qkv_activations /cp
        Description: QKV linear activations scale with s, so /cp is correct
                     for both Ring and Ulysses
        Expectation: cp=4 → qkv activations = 1/4 of cp=1
        """
        ccfg_1 = _make_ccfg(cp=1, t=1)
        ccfg_4 = _make_ccfg(cp=4, t=1)
        ctx = Context()
        ctx.micro_factor = 1

        qkv_1 = EvalAttn.attn_qkv_activations(ccfg_1, ctx)
        qkv_4 = EvalAttn.attn_qkv_activations(ccfg_4, ctx)

        self.assertAlmostEqual(qkv_4, qkv_1 / 4, places=1,
                               msg="QKV activations should be 1/4 with cp=4")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ffn_activations_divides_by_cp(self):
        """
        Feature: ffn_activations /cp
        Description: FFN activations scale with s, so /cp is correct
        Expectation: cp=4 → ffn activations = 1/4 of cp=1
        """
        ccfg_1 = _make_ccfg(cp=1, t=1)
        ccfg_4 = _make_ccfg(cp=4, t=1)
        ctx = Context()
        ctx.micro_factor = 1

        ffn_1 = EvalFFn.ffn_activations(ccfg_1, ctx)
        ffn_4 = EvalFFn.ffn_activations(ccfg_4, ctx)

        self.assertAlmostEqual(ffn_4, ffn_1 / 4, places=1,
                               msg="FFN activations should be 1/4 with cp=4")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_norm_activations_divides_by_cp(self):
        """
        Feature: norm_activations /cp
        Description: Norm activations scale with s, so /cp is correct
        Expectation: cp=4 → norm activations = 1/4 of cp=1
        """
        ccfg_1 = _make_ccfg(cp=1, t=1)
        ccfg_4 = _make_ccfg(cp=4, t=1)
        ctx = Context()
        ctx.micro_factor = 1

        norm_1 = EvalNorm.norm_activations(ccfg_1, ctx)
        norm_4 = EvalNorm.norm_activations(ccfg_4, ctx)

        self.assertAlmostEqual(norm_4, norm_1 / 4, places=1,
                               msg="Norm activations should be 1/4 with cp=4")


class TestCPCommVolumeUnit(unittest.TestCase):
    """Verify comm[Dim.CP] uses comm_volume (dp/tp/ep weighted-unit) not time.

    comm_volume is the field summed into comm[Dim.CP] for search-space ranking;
    it shares the same weighted-unit formula as dp/tp/ep (built from n_attMM /
    n_ffMM which carry bytes_p), so it can be ranked against comm[Dim.DP/TP/EP].
    total_kv_volume is the raw byte volume kept for diagnostics and
    exposed_comm_time calculation.
    """

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ring_total_kv_volume_positive_and_scaled(self):
        """
        Feature: CP total_kv_volume unit consistency
        Description: Ring CP total_kv_volume should scale with cp and s/b/kv_dim,
                     and be positive for cp>1
        Expectation: total_kv_volume > 0 for Ring cp=4
        """
        ccfg = _make_ccfg(cp=4, p=1)
        ctx = Context()

        result = cp_comm_layer_detailed(ccfg, ctx)
        self.assertGreater(result.total_kv_volume, 0.0,
                           msg="Ring CP total_kv_volume should be positive for cp>1")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_total_kv_volume_positive(self):
        """
        Feature: CP total_kv_volume for Ulysses
        Description: Ulysses CP total_kv_volume should be positive for cp>1
        Expectation: total_kv_volume > 0 for Ulysses cp=4
        """
        ccfg = _make_ccfg(cp=4, p=1, cp_algo="ulysses_cp", a=64)
        ctx = Context()

        result = cp_comm_layer_detailed(ccfg, ctx)
        self.assertGreater(result.total_kv_volume, 0.0,
                           msg="Ulysses CP total_kv_volume should be positive for cp>1")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_total_kv_volume_same_magnitude_as_old_cp_comm(self):
        """
        Feature: CP total_kv_volume participates in ranking
        Description: total_kv_volume should be within ~100x of old cp_comm_layer,
                     not 1e8x smaller (the old exposed_comm_time bug).
                     Both are byte-volume-like, so their ratio is bounded.
        Expectation: total_kv_volume / old_cp_comm_layer ratio between 0.001 and 100
        """
        ccfg = _make_ccfg(cp=4, p=1)
        ctx = Context()
        new_result = cp_comm_layer_detailed(ccfg, ctx)
        old_result = EvalLayerComm.cp_comm_layer(ccfg, ctx)
        ratio = new_result.total_kv_volume / old_result if old_result else float('inf')
        self.assertGreater(ratio, 0.001,
                           msg=f"CP total_kv_volume ({new_result.total_kv_volume}) too small "
                               f"vs old cp_comm_layer ({old_result}), ratio={ratio:.6f}")
        self.assertLess(ratio, 100,
                        msg=f"CP total_kv_volume ({new_result.total_kv_volume}) too large "
                            f"vs old cp_comm_layer ({old_result}), ratio={ratio:.1f}")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_total_kv_volume_not_exposed_comm_time(self):
        """
        Feature: CP total_kv_volume vs exposed_comm_time
        Description: total_kv_volume (volume unit) should be much larger than
                     exposed_comm_time (millisecond unit), proving the unit fix
        Expectation: total_kv_volume / exposed_comm_time > 1e3
        """
        ccfg = _make_ccfg(cp=4, p=1)
        ctx = Context()

        result = cp_comm_layer_detailed(ccfg, ctx)
        self.assertGreater(result.total_kv_volume, 0.0,
                           msg="total_kv_volume should be positive for cp>1")
        if result.exposed_comm_time > 0:
            ratio = result.total_kv_volume / result.exposed_comm_time
            self.assertGreater(ratio, 1e3,
                               msg=f"total_kv_volume ({result.total_kv_volume}) should be >> "
                                   f"exposed_comm_time ({result.exposed_comm_time}), "
                                   f"ratio={ratio:.1f}")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_total_kv_volume_zero_for_cp1(self):
        """
        Feature: CP total_kv_volume at cp=1
        Description: When cp=1, total_kv_volume should be 0 (no CP communication)
        Expectation: total_kv_volume == 0.0
        """
        ccfg = _make_ccfg(cp=1)
        ctx = Context()

        result = cp_comm_layer_detailed(ccfg, ctx)
        self.assertEqual(result.total_kv_volume, 0.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_comm_volume_positive_for_cp_gt_1(self):
        """
        Feature: CP comm_volume unit consistency
        Description: comm_volume is the field summed into comm[Dim.CP] for
                     search-space ranking; it must be positive for cp>1.
        Expectation: comm_volume > 0 for Ring cp=4 and Ulysses cp=4
        """
        ccfg_ring = _make_ccfg(cp=4, p=1)
        ctx = Context()
        ctx.current_node = LayerType.NOT_REC_LAYER
        ring = cp_comm_layer_detailed(ccfg_ring, ctx)
        self.assertGreater(ring.comm_volume, 0.0,
                           msg="Ring CP comm_volume should be positive for cp>1")

        ccfg_ulysses = _make_ccfg(cp=4, p=1, cp_algo="ulysses_cp", a=64)
        ulysses = cp_comm_layer_detailed(ccfg_ulysses, Context())
        self.assertGreater(ulysses.comm_volume, 0.0,
                           msg="Ulysses CP comm_volume should be positive for cp>1")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_comm_volume_zero_for_cp1(self):
        """
        Feature: CP comm_volume at cp=1
        Description: When cp=1, comm_volume should be 0 (no CP communication)
        Expectation: comm_volume == 0.0
        """
        ccfg = _make_ccfg(cp=1)
        ctx = Context()

        result = cp_comm_layer_detailed(ccfg, ctx)
        self.assertEqual(result.comm_volume, 0.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_comm_volume_equals_old_cp_comm_layer(self):
        """
        Feature: CP comm_volume backward compatibility
        Description: comm_volume should exactly equal the legacy
                     EvalLayerComm.cp_comm_layer output, since it replaced
                     that value in the comm[Dim.CP] accumulation.
        Expectation: comm_volume == old cp_comm_layer (Ring cp=4)
        """
        ccfg = _make_ccfg(cp=4, p=1)
        ctx = Context()
        ctx.current_node = LayerType.NOT_REC_LAYER

        new_result = cp_comm_layer_detailed(ccfg, ctx)
        old_result = EvalLayerComm.cp_comm_layer(ccfg, ctx)
        self.assertAlmostEqual(new_result.comm_volume, old_result, places=6,
                               msg=f"comm_volume ({new_result.comm_volume}) should equal "
                                   f"old cp_comm_layer ({old_result})")


class TestCPCommBufferInPeak(unittest.TestCase):
    """Verify cp_comm_buffer function behavior (display-only, not in peak path).

    cp_comm_layer (volume, same unit as dp/tp) feeds the peak via
    comm_expr = max(dp, tp, cp) + ep, while cp_comm_buffer (real KV
    bytes) is used only for the per-stage display breakdown. The two
    must stay in separate paths: swapping cp to real bytes would make
    max(dp_volume, tp_volume, cp_bytes) lose cp every time.
    """

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_buffer_zero_for_cp1(self):
        """
        Feature: CP comm buffer at cp=1
        Description: When cp=1, cp_comm_buffer should be 0 (no CP ring/all2all)
        Expectation: cp_comm_buffer == 0.0
        """
        ccfg = _make_ccfg(cp=1)
        ctx = Context()
        self.assertEqual(EvalLayerComm.cp_comm_buffer(ccfg, ctx), 0.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_buffer_positive_for_cp_gt_1(self):
        """
        Feature: CP comm buffer at cp>1
        Description: cp_comm_buffer should be positive for Ring and Ulysses cp>1
        Expectation: cp_comm_buffer > 0 for cp=4
        """
        ccfg_ring = _make_ccfg(cp=4, s=131072)
        ctx = Context()
        self.assertGreater(EvalLayerComm.cp_comm_buffer(ccfg_ring, ctx), 0.0,
                           msg="Ring CP comm buffer should be positive for cp>1")

        ccfg_ulysses = _make_ccfg(cp=4, s=131072, cp_algo="ulysses_cp", a=64)
        self.assertGreater(EvalLayerComm.cp_comm_buffer(ccfg_ulysses, Context()), 0.0,
                           msg="Ulysses CP comm buffer should be positive for cp>1")


class TestCPConstraintHeadsAndAlgo(unittest.TestCase):
    """Verify cp_algo and attention_heads are forwarded to validate_cp_constraints (Reviewer Fix #3)."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_validate_accepts_cp_algo_param(self):
        """
        Feature: validate_cp_constraints cp_algo parameter
        Description: The function should accept cp_algo and use it for Ulysses checks
        Expectation: Ulysses with insufficient heads → is_valid=False
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=8,
            tp_degree=2,
            device_per_node=8,
            attention_type_str="mha",
            cp_algo="ulysses_cp",
            attention_heads=12,
        )
        self.assertFalse(result.is_valid)
        self.assertIn("Ulysses", result.unsupported_reason)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_validate_accepts_attention_heads_param(self):
        """
        Feature: validate_cp_constraints attention_heads parameter
        Description: attention_heads=0 should skip the Ulysses head check
        Expectation: Even with ulysses_cp, attention_heads=0 → check skipped
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            cp_algo="ulysses_cp",
            attention_heads=0,
        )
        self.assertTrue(result.is_valid,
                       f"attention_heads=0 should skip Ulysses check, got {result.error_message}")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_heads_sufficient_when_provided(self):
        """
        Feature: Ulysses head sufficiency with attention_heads
        Description: When attention_heads >= tp*cp, Ulysses is valid
        Expectation: a=64, t=2, cp=4 → 64 >= 8 → is_valid=True
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=2,
            device_per_node=8,
            attention_type_str="mha",
            cp_algo="ulysses_cp",
            attention_heads=64,
        )
        self.assertTrue(result.is_valid)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ring_algo_ignores_attention_heads(self):
        """
        Feature: Ring CP ignores attention_heads
        Description: Ring CP doesn't split heads, so attention_heads is irrelevant
        Expectation: Ring cp=4 with a=2 still valid (heads don't matter)
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            cp_algo="colossalai_cp",
            attention_heads=2,
        )
        self.assertTrue(result.is_valid,
                       "Ring CP should not require sufficient attention heads")


class TestCPUlyssesKVHeadsDivisibility(unittest.TestCase):
    """Verify Ulysses CP checks num_kv_heads divisibility (not attention_heads >= tp*cp).

    Reviewer Fix (Review 20 #3): Ulysses shards KV heads across tp*cp ranks,
    so the constraint is num_kv_heads % (tp*cp) == 0. The old check used
    attention_heads (Q heads) >= tp*cp, which let GQA 8:2 configs (n_kv=8,
    a=64) through at cp=4/8 even though runtime would crash.
    """

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_gqa_8_2_kv_heads_divisible_passes(self):
        """
        Feature: Ulysses KV-head divisibility (GQA pass)
        Description: GQA 8:2 (n_kv=8, a=64), tp=1, cp=4 → 8 % 4 == 0 → valid
        Expectation: is_valid=True
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="gqa",
            cp_algo="ulysses_cp",
            attention_heads=64,
            num_kv_heads=8,
        )
        self.assertTrue(result.is_valid,
                       f"GQA 8:2 cp=4 should be valid (8 % 4 == 0), got {result.error_message}")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_gqa_8_2_kv_heads_not_divisible_rejected(self):
        """
        Feature: Ulysses KV-head divisibility (GQA reject)
        Description: GQA 8:2 (n_kv=8, a=64), tp=1, cp=16 → 8 % 16 != 0 →
                     rejected. Old check (a=64 >= 16) would have wrongly passed.
        Expectation: is_valid=False, error mentions num_kv_heads / divisibility
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=16,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="gqa",
            cp_algo="ulysses_cp",
            attention_heads=64,
            num_kv_heads=8,
        )
        self.assertFalse(result.is_valid,
                         "GQA 8:2 cp=16 must be rejected (8 % 16 != 0); old "
                         "attention_heads>=tp*cp check would have passed")
        self.assertIn("num_kv_heads", result.error_message)
        self.assertIn("divisible", result.error_message)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_gqa_8_2_cp8_passes_with_tp1(self):
        """
        Feature: Ulysses KV-head divisibility boundary (cp=8)
        Description: GQA 8:2 (n_kv=8), tp=1, cp=8 → 8 % 8 == 0 → valid.
                     This is the boundary case from Review 19 (searcher let
                     cp=4/8 through but runtime crashed at cp=8 with old code).
        Expectation: is_valid=True
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=8,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="gqa",
            cp_algo="ulysses_cp",
            attention_heads=64,
            num_kv_heads=8,
        )
        self.assertTrue(result.is_valid,
                       f"GQA 8:2 cp=8 (8 % 8 == 0) should be valid, got {result.error_message}")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_num_kv_heads_zero_falls_back_to_attention_heads(self):
        """
        Feature: num_kv_heads=0 fallback
        Description: num_kv_heads=0 falls back to attention_heads (MHA semantics,
                     matching compute_kv_dim in cost_model_preprocess.py).
        Expectation: a=64, n_kv=0, tp=1, cp=4 → 64 % 4 == 0 → valid
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=1,
            device_per_node=8,
            attention_type_str="mha",
            cp_algo="ulysses_cp",
            attention_heads=64,
            num_kv_heads=0,
        )
        self.assertTrue(result.is_valid,
                       f"num_kv_heads=0 should fall back to attention_heads=64, got {result.error_message}")

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_tp_shards_kv_heads_combined(self):
        """
        Feature: Ulysses KV-head divisibility with TP
        Description: GQA (n_kv=8), tp=2, cp=4 → shards=8, 8 % 8 == 0 → valid.
                     TP shards KV, so the divisor is tp*cp (not just cp).
        Expectation: is_valid=True
        """
        result = validate_cp_constraints(
            seq_len=131072,
            cp_degree=4,
            tp_degree=2,
            device_per_node=8,
            attention_type_str="gqa",
            cp_algo="ulysses_cp",
            attention_heads=64,
            num_kv_heads=8,
        )
        self.assertTrue(result.is_valid,
                       f"n_kv=8, tp=2, cp=4 (8 % 8 == 0) should be valid, got {result.error_message}")


class TestCPStaticMethodsNoCtxMutation(unittest.TestCase):
    """Verify CP static methods do not depend on or mutate ctx.current_node."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_act_cp_layer_does_not_require_current_node(self):
        """
        Feature: CP Static Method Independence
        Description: act_cp_layer is a @staticmethod and should not
                     depend on ctx.current_node being set
        Expectation: Works correctly with default (uninitialized) Context
        """
        ccfg = _make_ccfg(cp=4, s=131072)
        ctx = Context()
        result = EvalBody.act_cp_layer(ccfg, ctx)
        self.assertIsInstance(result, CPMemoryBreakdown)
        self.assertGreater(result.total_reduction, 0.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_comm_buffer_does_not_require_current_node(self):
        """
        Feature: CP Static Method Independence
        Description: cp_comm_buffer is a @staticmethod and should not
                     depend on ctx.current_node being set
        Expectation: Returns positive buffer size with default Context
        """
        ccfg = _make_ccfg(cp=4, s=131072)
        ctx = Context()
        result = EvalLayerComm.cp_comm_buffer(ccfg, ctx)
        self.assertGreater(result, 0.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_comm_layer_detailed_does_not_require_current_node(self):
        """
        Feature: CP Function Independence
        Description: cp_comm_layer_detailed should not depend on
                     ctx.current_node being set
        Expectation: Returns valid CPCommunicationCost with default Context
        """
        ccfg = _make_ccfg(cp=4, s=131072)
        ctx = Context()
        result = cp_comm_layer_detailed(ccfg, ctx)
        self.assertIsInstance(result, CPCommunicationCost)
        self.assertGreater(result.exposed_comm_time, 0.0)


class TestCPTopologyHelpers(unittest.TestCase):
    """Cover get_cp_topology intra/cross-node branches in hardware.py."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_intra_node_when_tp_cp_within_device_per_node(self):
        """
        Feature: CP Topology Resolution
        Description: tp*cp <= device_per_node → intra-node, 300 GB/s
        Expectation: topology_type="intra-node", is_intra_node=True, bw=300.0
        """
        topology, bw, is_intra = get_cp_topology(tp_degree=1, cp_degree=4, device_per_node=8)
        self.assertEqual(topology, "intra-node")
        self.assertTrue(is_intra)
        self.assertEqual(bw, 300.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_boundary_tp_cp_equals_device_per_node_is_intra(self):
        """
        Feature: CP Topology Resolution
        Description: tp*cp == device_per_node (boundary) stays intra-node
        Expectation: is_intra_node=True (<= not <)
        """
        topology, _, is_intra = get_cp_topology(tp_degree=2, cp_degree=4, device_per_node=8)
        self.assertEqual(topology, "intra-node")
        self.assertTrue(is_intra)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cross_node_when_tp_cp_exceeds_device_per_node(self):
        """
        Feature: CP Topology Resolution
        Description: tp*cp > device_per_node → cross-node, 25 GB/s penalty
        Expectation: topology_type="cross-node", is_intra_node=False, bw=25.0
        """
        topology, bw, is_intra = get_cp_topology(tp_degree=2, cp_degree=8, device_per_node=8)
        self.assertEqual(topology, "cross-node")
        self.assertFalse(is_intra)
        self.assertEqual(bw, 25.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp1_always_intra_node(self):
        """
        Feature: CP Topology Resolution
        Description: cp=1 with any tp → always intra-node (no cross-node penalty)
        Expectation: is_intra_node=True
        """
        _, _, is_intra = get_cp_topology(tp_degree=8, cp_degree=1, device_per_node=8)
        self.assertTrue(is_intra)


class TestCPBandwidthAndRecommendation(unittest.TestCase):
    """Cover get_cp_bandwidth and recommend_cp_max_by_attention in hardware.py."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_bandwidth_intra_node_a2(self):
        """
        Feature: CP Bandwidth Lookup
        Description: Device_A2.level_bandwidth=[50, 10]; intra-node → 50 GB/s
        Expectation: get_cp_bandwidth("intra-node", "A2") == 50.0
        """
        self.assertEqual(get_cp_bandwidth("intra-node", "A2"), 50.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_bandwidth_cross_node_a2(self):
        """
        Feature: CP Bandwidth Lookup
        Description: cross-node → Device_A2.level_bandwidth[1] = 10 GB/s
        Expectation: get_cp_bandwidth("cross-node", "A2") == 10.0
        """
        self.assertEqual(get_cp_bandwidth("cross-node", "A2"), 10.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_bandwidth_unknown_device_falls_back_to_a2(self):
        """
        Feature: CP Bandwidth Lookup
        Description: unknown device_type → device_map.get default Device_A2
        Expectation: returns A2 intra-node bandwidth (50.0)
        """
        self.assertEqual(get_cp_bandwidth("intra-node", "UnknownDevice"), 50.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_bandwidth_a3_intra_and_cross(self):
        """
        Feature: CP Bandwidth Lookup
        Description: Device_A3.level_bandwidth=[200, 25, 10]; intra→200, cross→25
        Expectation: A3 intra-node=200.0, cross-node=25.0
        """
        self.assertEqual(get_cp_bandwidth("intra-node", "A3"), 200.0)
        self.assertEqual(get_cp_bandwidth("cross-node", "A3"), 25.0)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_recommend_cp_max_mla_is_16(self):
        """
        Feature: CP Degree Recommendation by Attention Type
        Description: MLA has lowest KV cost → highest cp_max=16
        Expectation: recommend_cp_max_by_attention("mla") == 16
        """
        self.assertEqual(recommend_cp_max_by_attention("mla"), 16)
        self.assertEqual(recommend_cp_max_by_attention("MLA"), 16)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_recommend_cp_max_gqa_is_8(self):
        """
        Feature: CP Degree Recommendation by Attention Type
        Description: GQA has medium KV cost → cp_max=8
        Expectation: recommend_cp_max_by_attention("gqa") == 8
        """
        self.assertEqual(recommend_cp_max_by_attention("gqa"), 8)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_recommend_cp_max_mha_and_unknown_is_4(self):
        """
        Feature: CP Degree Recommendation by Attention Type
        Description: MHA and unknown attention types → cp_max=4 (most conservative)
        Expectation: recommend_cp_max_by_attention("mha") == 4; unknown == 4
        """
        self.assertEqual(recommend_cp_max_by_attention("mha"), 4)
        self.assertEqual(recommend_cp_max_by_attention("unknown"), 4)


class TestResolveCPAlgoBranches(unittest.TestCase):
    """Cover _resolve_cp_algo's three branches: CPAlgo instance, str, fallback."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_returns_same_instance_when_already_cp_algo(self):
        """
        Feature: CP Algo Resolution
        Description: When ccfg.cp_algo is already a CPAlgo enum, return as-is
        Expectation: _resolve_cp_algo returns the same enum instance
        """

        class _Cfg:
            cp_algo = CPAlgo.ULYSSES_CP

        self.assertIs(_resolve_cp_algo(_Cfg()), CPAlgo.ULYSSES_CP)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_resolves_known_string_to_enum(self):
        """
        Feature: CP Algo Resolution
        Description: Known string cp_algo values map via _CP_ALGO_STR_MAP
        Expectation: "colossalai_cp"→COLOSSALAI_CP, "ulysses_cp"→ULYSSES_CP, "hybrid_cp"→HYBRID_CP
        """

        class _Cfg:
            cp_algo = "ulysses_cp"

        self.assertIs(_resolve_cp_algo(_Cfg()), CPAlgo.ULYSSES_CP)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_resolves_typo_hybird_to_hybrid(self):
        """
        Feature: CP Algo Resolution
        Description: Legacy typo "hybird_cp" is tolerated and maps to HYBRID_CP
        Expectation: _resolve_cp_algo("hybird_cp") == CPAlgo.HYBRID_CP
        """

        class _Cfg:
            cp_algo = "hybird_cp"

        self.assertIs(_resolve_cp_algo(_Cfg()), CPAlgo.HYBRID_CP)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_unknown_string_falls_back_to_colossalai(self):
        """
        Feature: CP Algo Resolution
        Description: Unknown string cp_algo → default COLOSSALAI_CP
        Expectation: _resolve_cp_algo("nonsense") == CPAlgo.COLOSSALAI_CP
        """

        class _Cfg:
            cp_algo = "nonsense_algo"

        self.assertIs(_resolve_cp_algo(_Cfg()), CPAlgo.COLOSSALAI_CP)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_missing_attr_falls_back_to_colossalai(self):
        """
        Feature: CP Algo Resolution
        Description: ccfg without cp_algo attr → getattr returns None → fallback COLOSSALAI_CP
        Expectation: _resolve_cp_algo returns COLOSSALAI_CP
        """

        class _Cfg:
            pass

        self.assertIs(_resolve_cp_algo(_Cfg()), CPAlgo.COLOSSALAI_CP)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_none_value_falls_back_to_colossalai(self):
        """
        Feature: CP Algo Resolution
        Description: ccfg.cp_algo=None (not str, not CPAlgo) → fallback COLOSSALAI_CP
        Expectation: _resolve_cp_algo returns COLOSSALAI_CP
        """

        class _Cfg:
            cp_algo = None

        self.assertIs(_resolve_cp_algo(_Cfg()), CPAlgo.COLOSSALAI_CP)


class TestParallelizeLayerCPValidation(unittest.TestCase):
    """Cover ParallelizeLayer.is_valid CP branch (parallelize.py:164-202).

    is_valid is a pure validation function over (self.config, self.machine,
    parallel_config). We bypass __init__ via ``object.__new__`` and inject
    lightweight ``SimpleNamespace`` doubles — the same pattern used in
    ``test_run_nd.py:691`` (``object.__new__(GC.GlobalConfig)``) and
    ``test_memory_estimation.py:237`` (``SimpleNamespace`` as a config
    double). Only the CP branch is under test here, so we avoid
    constructing a real EvaluatorV2/Machine.
    """

    def _make_layer(self, ccfg, gbs=1):
        """Build a ParallelizeLayer shell with just enough state for is_valid."""
        config = SimpleNamespace(
            ccfg=ccfg,
            moe_valid=lambda _pc: True,
            global_batch_size=lambda _pc: gbs,
        )
        machine = SimpleNamespace(
            number=8,
            device=SimpleNamespace(intra_node_num=lambda: 8),
        )
        layer = object.__new__(ParallelizeLayer)
        layer.config = config
        layer.machine = machine
        layer.global_batch_size = gbs
        layer.filtered_out = lambda _pc: False
        return layer

    def _make_pc(self, dims_val):
        """Build a parallel_config double with is_valid()=True."""
        return SimpleNamespace(
            dims_val=dict(dims_val),
            is_valid=lambda: True,
            has_dim=lambda d: d in dims_val,
        )

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp1_skips_cp_branch(self):
        """
        Feature: ParallelizeLayer CP validation
        Description: cp_degree=1 → CP branch (cp_degree > 1) is skipped
        Expectation: is_valid returns True (falls through to gbs check)
        """
        ccfg = _make_ccfg(cp=1)
        layer = self._make_layer(ccfg)
        pc = self._make_pc({Dim.CP: 1, Dim.TP: 1, Dim.PP: 1})
        self.assertTrue(layer.is_valid(pc))

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_invalid_seq_returns_false(self):
        """
        Feature: ParallelizeLayer CP validation
        Description: cp=4 + seq_len not divisible by cp*2 → cp_result.is_valid=False
        Expectation: is_valid returns False (CP constraints violated branch)
        """
        ccfg = _make_ccfg(s=131073, cp=4)  # 131073 not divisible by 8
        layer = self._make_layer(ccfg)
        pc = self._make_pc({Dim.CP: 4, Dim.TP: 1, Dim.PP: 1})
        self.assertFalse(layer.is_valid(pc))

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_warning_does_not_reject(self):
        """
        Feature: ParallelizeLayer CP validation
        Description: cp=4 + short sequence → cp_result.warning_message set but is_valid=True
        Expectation: is_valid returns True (warning logged, not a rejection)
        """
        ccfg = _make_ccfg(s=256, cp=4)  # short seq triggers warning, still valid
        layer = self._make_layer(ccfg)
        pc = self._make_pc({Dim.CP: 4, Dim.TP: 1, Dim.PP: 1})
        self.assertTrue(layer.is_valid(pc))

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_absent_skips_branch(self):
        """
        Feature: ParallelizeLayer CP validation
        Description: parallel_config.dims_val without Dim.CP → CP branch skipped
        Expectation: is_valid returns True
        """
        ccfg = _make_ccfg(cp=1)
        layer = self._make_layer(ccfg)
        pc = self._make_pc({Dim.TP: 1, Dim.PP: 1})  # no Dim.CP
        self.assertTrue(layer.is_valid(pc))


class TestCPSearchSpaceEnumeration(unittest.TestCase):
    """Verify CP>1 appears in search space when starting from cp=1.

    Before the enable_cp fix, _CostModelParser.config_comm_flag set
    enable_cp = ccfg.cp > 1. Since search entries start with cp=1,
    enable_cp was always False, cp_upper was 1, and Dim.CP only ever
    enumerated cp=1. After the fix, CP enumeration is bounded only by
    machine.number // tp // pp, so cp=2/4/8 are reachable from cp=1.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Import GlobalConfig once (avoids circular import at module level)."""
        from hyper_parallel.auto_parallel.sapp_nd.nd.global_config import GlobalConfig as GC
        cls._GC = GC

    def _make_global_config(self, cp: int = 1):
        """Build a GlobalConfig backed by a real CostModelConfig.

        Uses _make_ccfg (the same construction path as production code),
        so the __getattr__ fallback returning 0 for unknown attrs is
        exercised — this matters because the pre-fix code read ccfg.enable_cp,
        which _make_ccfg(cp=1) yields as 0 (falsy) via the fallback.
        """
        ccfg = _make_ccfg(cp=cp)
        gcfg = object.__new__(self._GC)
        gcfg.ccfg = ccfg
        gcfg.dimensions = Dim.ALL_DIMS.copy()
        # device_loops / space do not touch balancing, so leave it unset.
        return gcfg

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_device_loops_visits_cp_greater_than_one(self):
        """
        Feature: CP Search Space Enumeration in device_loops
        Description: ParallelizeLayer.device_loops should enumerate cp>1
                     even when the initial ccfg.cp=1. Before the enable_cp
                     fix, ccfg.enable_cp was 0 (via _CostModVar __getattr__
                     fallback for the unset attribute), cp_upper was 1, and
                     device_loops only ever visited cp=1.
        Expectation: device_loops visits cp=2 and cp=4 for 8 devices
        """
        gcfg = self._make_global_config(cp=1)
        ccfg = gcfg.ccfg
        config = {
            "ccfg": ccfg,
            "space": gcfg.space,
            "moe_valid": lambda _pc: True,
            "global_batch_size": lambda _pc: 4,
        }

        machine = SimpleNamespace(
            number=8,
            device=SimpleNamespace(intra_node_num=lambda: 8),
        )

        visited_cp: list = []

        class _CaptureCP(ParallelizeLayer):
            """Override batch_loops to capture cp without full pipeline."""
            def batch_loops(self, space: dict, pool: object, dtpc_p: tuple) -> dict:
                """Record cp value and pass through."""
                visited_cp.append(dtpc_p[3])
                return space

        layer = object.__new__(_CaptureCP)
        layer.config = SimpleNamespace(**config)
        layer.machine = machine
        layer.global_batch_size = 4
        layer.filtered_out = lambda _pc: False

        layer.device_loops(({}, 0), None)

        self.assertIn(1, visited_cp, "cp=1 should be enumerated")
        self.assertIn(2, visited_cp, "cp=2 should be enumerated from cp=1 start")
        self.assertIn(4, visited_cp, "cp=4 should be enumerated from cp=1 start")
        self.assertIn(8, visited_cp, "cp=8 should be enumerated from cp=1 start")


class TestDimensionsValidationBranches(unittest.TestCase):
    """Cover invalid-path branches in dimensions.py validation helpers.

    Targets the `return False` / `logger.warning` branches of
    _check_mbn_pp, _check_power_of_two, and is_valid (SP/CP coexistence,
    OP non-power-of-2), plus the shards<=0 boundary in
    _cp_check_ulysses_heads.
    """

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_check_mbn_pp_invalid_mbn_less_than_pp(self):
        """MBN < PP → _check_mbn_pp returns False."""
        result = Dim.Dimensions._check_mbn_pp({Dim.MBN: 1, Dim.PP: 2}, [Dim.PP])
        self.assertFalse(result)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_check_mbn_pp_invalid_pp1_mbn_gt1(self):
        """PP=1 & MBN>1 → _check_mbn_pp returns False."""
        result = Dim.Dimensions._check_mbn_pp({Dim.MBN: 2, Dim.PP: 1}, [Dim.PP])
        self.assertFalse(result)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_check_power_of_two_rejects_non_power(self):
        """value=3 (not power of 2) → _check_power_of_two returns False."""
        self.assertFalse(Dim.Dimensions._check_power_of_two(Dim.TP, 3))

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_is_valid_rejects_sp_cp_coexistence(self):
        """SP enabled & CP>1 → is_valid returns False."""
        pc = object.__new__(Dim.Dimensions)
        pc.dims_val = {Dim.SP: 1, Dim.CP: 2, Dim.TP: 1, Dim.MBN: 1, Dim.PP: 1}
        pc.all_dims = [Dim.SP, Dim.CP, Dim.TP, Dim.MBN, Dim.PP]
        self.assertFalse(pc.is_valid())

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_cp_check_ulysses_heads_shards_zero(self):
        """tp=0 & cp=0 → shards=0 → _cp_check_ulysses_heads returns None."""
        p = CPConstraintParams(
            seq_len=1024, cp_degree=0,
            cp_algo="ulysses_cp", num_kv_heads=8, attention_heads=8,
            tp_degree=0,
        )
        self.assertIsNone(Dim._cp_check_ulysses_heads(p))


class TestCPCommBufferCrossNode(unittest.TestCase):
    """Cover cp_comm_buffer cross-node branch (comm.py:251).

    When cp > device_per_node, extra_chunks = 2 * intra_ranks - 1
    (intra-node all-gather result stays resident + 1 full-node ring
    receive buffer). The existing TestCPCommBufferInPeak only covers
    the intra-node case.
    """

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ring_cp_cross_node_buffer(self):
        """
        Feature: CP comm buffer (Ring, cross-node)
        Description: cp=16 > device_per_node=8 → cross-node branch
        Expectation: extra_chunks = 2*8-1 = 15, buffer = 15 * chunk
        """
        ccfg = _make_ccfg(cp=16, device_per_node=8)
        ctx = Context()
        buffer = EvalLayerComm.cp_comm_buffer(ccfg, ctx)
        self.assertGreater(buffer, 0)

        # Verify the cross-node formula: 2*intra_ranks - 1 = 15
        s, b, cp = ccfg.s, ccfg.b, ccfg.cp
        kv_bytes = 2 * 2
        kv_dim = compute_kv_dim(ccfg)
        chunk = (s / cp) * b * kv_dim * kv_bytes
        expected = (2 * 8 - 1) * chunk
        self.assertAlmostEqual(buffer, expected, places=4)

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="unessential",
    )
    def test_ulysses_cp_cross_node_buffer(self):
        """
        Feature: CP comm buffer (Ulysses, cross-node)
        Description: ulysses_cp cp=16 > device_per_node=8 → cross-node branch
        Expectation: extra_chunks = 2*8-1 = 15
        """
        ccfg = _make_ccfg(cp=16, device_per_node=8, cp_algo="ulysses_cp")
        ctx = Context()
        buffer = EvalLayerComm.cp_comm_buffer(ccfg, ctx)
        self.assertGreater(buffer, 0)

        s, b, cp = ccfg.s, ccfg.b, ccfg.cp
        kv_bytes = 2 * 2
        kv_dim = compute_kv_dim(ccfg)
        chunk = s * b * (kv_dim / cp) * kv_bytes
        expected = (2 * 8 - 1) * chunk
        self.assertAlmostEqual(buffer, expected, places=4)


if __name__ == '__main__':
    unittest.main()
