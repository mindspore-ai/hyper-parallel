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
"""Unit tests for FLOP-mode communication time estimation."""
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import Type
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import (
    _flop_mode_comp_comm,
    _flop_mode_dp_comm,
    _flop_mode_fsdp_comm,
    _flop_mode_pp_total_comm,
    _flop_mode_tp_comm,
    compute_hsdp_flop_total,
)


def _make_cfg(d=8, t=2, cp=1, p=2, m=4, **kwargs):
    defaults = {
        "d": d,
        "t": t,
        "cp": cp,
        "p": p,
        "m": m,
        "fsdp": True,
        "d_shard_or_d": kwargs.get("d_shard_or_d", 4),
        "n_gather": 1,
        "b": 1,
        "s": 8,
        "h": 16,
        "ep": kwargs.get("ep", 1),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_device(intra_node_num=8):
    device = MagicMock(spec=Type)
    device.intra_node_num.return_value = intra_node_num
    device.name = "TestDev"
    device.level_bandwidth = [50.0]
    device.level_bound_number = [8]
    hw_map = {
        "tp_bw_efficiency": 0.6,
        "tp_hsdp_sync": 0.0,
    }
    device.hw = lambda key: hw_map.get(key, 0.0)
    return device


class TestFlopModeTpComm(unittest.TestCase):

    def test_returns_positive_score(self):
        cfg = _make_cfg(d=8, t=2, cp=1, d_shard_or_d=4)
        device = _make_device(intra_node_num=8)
        score = _flop_mode_tp_comm(cfg, 4, device_type=device, mb=4)
        self.assertIsInstance(score, float)

    def test_d_rep_1_returns_zero(self):
        cfg = _make_cfg(d=4, t=2, cp=1, d_shard_or_d=4)
        device = _make_device(intra_node_num=8)
        score = _flop_mode_tp_comm(cfg, 4, device_type=device, mb=1)
        self.assertEqual(score, 0.0)


class TestFlopModeFsdpComm(unittest.TestCase):

    def test_hsdp_returns_positive_score(self):
        cfg = _make_cfg(d=8, t=2, cp=1, d_shard_or_d=4)
        device = _make_device(intra_node_num=8)
        score = _flop_mode_fsdp_comm(cfg, 60, 4, device, pp=2, mb=4)
        self.assertIsInstance(score, float)

    def test_fsdp_per_layer_model(self):
        cfg = _make_cfg(d=4, t=2, cp=1, d_shard_or_d=4)
        device = _make_device(intra_node_num=8)
        score = _flop_mode_fsdp_comm(cfg, 60, 4, device, pp=1, mb=1)
        self.assertIsInstance(score, float)
        self.assertNotEqual(score, 0.0)


class TestComputeHsdpFlopTotal(unittest.TestCase):

    def test_returns_sum_of_components(self):
        cfg = _make_cfg(d=8, t=2, cp=1, p=2, m=4, d_shard_or_d=4)
        device = _make_device(intra_node_num=8)
        total = compute_hsdp_flop_total(cfg, 4, device, fsdp_layer_count=60, mb=4, pp=2)
        comp = _flop_mode_comp_comm(cfg, 4, device_type=device, mb=4)
        tp = _flop_mode_tp_comm(cfg, 4, device_type=device, mb=4)
        dp = _flop_mode_dp_comm(1.0, cfg, 4, device, mb=4)
        shard = _flop_mode_fsdp_comm(cfg, 60, 4, device, pp=2, mb=4)
        pp = _flop_mode_pp_total_comm(cfg, 4, device, mb=4)
        expected = comp + tp + shard + dp + pp
        self.assertAlmostEqual(total, expected, places=5)

    def test_total_is_positive(self):
        cfg = _make_cfg(d=8, t=2, cp=1, p=2, m=4, d_shard_or_d=4)
        device = _make_device(intra_node_num=8)
        total = compute_hsdp_flop_total(cfg, 4, device, fsdp_layer_count=60, mb=4, pp=2)
        self.assertGreater(total, 0)


from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import device_map as _device_map
from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as _Dim
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import (
    _get_flop_coeffs,
    _compute_hsdp_features,
    _apply_flop_mode,
    estimate_comm_score,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.debug import PerfParts
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import (
    estimate_stage,
    estimate_pipeline,
    estimate_p2p,
    estimate_p2p_comm,
    _count_p2p_messages,
    _estimate_non_hsdp_perf,
    _estimate_hsdp_perf,
    _compute_stage_perfs,
    estimate_op_bulk_comp,
    op_table,
    apply_regression_coefficients,
    estimate_performance,
    RatioType,
    P2PCommType,
    RecType,
)
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard


class TestGetFlopCoeffs(unittest.TestCase):

    def test_v4_fallback_shard_fsdp(self):
        result = _get_flop_coeffs(None, "shard", "fsdp")
        self.assertIsInstance(result, dict)
        self.assertIn("INTERCEPT", result)

    def test_device_with_flop_coeffs(self):
        device = MagicMock(spec=Type)
        device.flop_coeffs = {"shard": {"fsdp": {"INTERCEPT": 42.0, "TP": 1.0}}}
        result = _get_flop_coeffs(device, "shard", "fsdp")
        self.assertEqual(result, {"INTERCEPT": 42.0, "TP": 1.0})


class TestComputeHsdpFeatures(unittest.TestCase):

    def test_basic_features(self):
        cfg = _make_cfg(d=8, t=2, cp=1, d_shard_or_d=4)
        device = _make_device(intra_node_num=8)
        f = _compute_hsdp_features(cfg, 4, device, mb=4)
        self.assertEqual(f["tp"], 2)
        self.assertEqual(f["cp"], 1)
        self.assertEqual(f["d"], 8)
        self.assertEqual(f["d_shard"], 4)
        self.assertEqual(f["d_replicate"], 2)
        self.assertEqual(f["sg"], 4 * 1 * 2)
        self.assertAlmostEqual(f["inv_sg"], 1.0 / 8)
        self.assertAlmostEqual(f["inv_tp"], 1.0 / 2)
        self.assertAlmostEqual(f["ag_vol"], 1.0 - 1.0 / 4)
        self.assertEqual(f["m"], 4)
        self.assertEqual(f["dev_per_node"], 8)

    def test_cross_flag_true(self):
        cfg = _make_cfg(d=16, t=4, cp=1, d_shard_or_d=4)
        device = _make_device(intra_node_num=8)
        f = _compute_hsdp_features(cfg, 4, device, mb=1)
        self.assertEqual(f["cross"], 1.0)


class TestEstimateCommScore(unittest.TestCase):

    def test_basic_dp_volume(self):
        cfg = _make_cfg(d=4, t=2, cp=1, d_shard_or_d=4)
        a2 = _device_map.get("A2")
        result = estimate_comm_score(cfg, 1e8, _Dim.DP, device=a2)
        self.assertGreater(result, 0)

    def test_zero_volume_returns_zero(self):
        cfg = _make_cfg(d=4, t=2, cp=1, d_shard_or_d=4)
        a2 = _device_map.get("A2")
        result = estimate_comm_score(cfg, 0.0, _Dim.DP, device=a2)
        self.assertEqual(result, 0)


class TestApplyFlopMode(unittest.TestCase):

    def test_hsdp_replaces_tp_comm(self):
        cfg = _make_cfg(d=8, t=2, cp=1, d_shard_or_d=4)
        device = _make_device(intra_node_num=8)
        comm = {_Dim.DP: 1.0, _Dim.TP: 100.0, _Dim.EP: 50.0, _Dim.CP: 0.0, _Dim.FSDP: 0.0}
        param = {"cfg": cfg, "device_type": device}
        _apply_flop_mode(comm, param, fsdp_layer_count=60)
        self.assertIsInstance(comm[_Dim.TP], float)
        self.assertNotEqual(comm[_Dim.TP], 100.0)

    def test_fsdp_comm_set_when_fsdp_enabled(self):
        cfg = _make_cfg(d=8, t=2, cp=1, d_shard_or_d=4, fsdp=True)
        device = _make_device(intra_node_num=8)
        comm = {_Dim.DP: 1.0, _Dim.TP: 100.0, _Dim.EP: 50.0, _Dim.CP: 0.0, _Dim.FSDP: 0.0}
        param = {"cfg": cfg, "device_type": device}
        _apply_flop_mode(comm, param, fsdp_layer_count=60)
        self.assertNotEqual(comm[_Dim.FSDP], 0.0)


class TestHSDPEstimateCommScore(unittest.TestCase):

    def test_estimate_comm_score_with_cp_greater_than_1(self):
        device = Type(name="Test2L", bounds=[8, None], bandwidths=[50, 10])
        ccfg_cp1 = SimpleNamespace(d=8, t=2, cp=1, p=1, d_shard=4, d_shard_or_d=4)
        ccfg_cp2 = SimpleNamespace(d=8, t=2, cp=2, p=1, d_shard=4, d_shard_or_d=4)
        result_cp1 = estimate_comm_score(
            ccfg_cp1, 100.0, _Dim.CP, overlap=0.0, device=device
        )
        result_cp2 = estimate_comm_score(
            ccfg_cp2, 100.0, _Dim.CP, overlap=0.0, device=device
        )
        self.assertNotEqual(result_cp1, result_cp2)


def _make_cfg_boost(d=8, t=2, cp=1, p=2, m=4, **kwargs):
    defaults = {
        "d": d,
        "t": t,
        "cp": cp,
        "p": p,
        "m": m,
        "fsdp": True,
        "d_shard_or_d": kwargs.get("d_shard_or_d", d),
        "n_gather": 1,
        "b": 1,
        "s": 8,
        "h": 16,
        "hff": 64,
        "n_exp": 1,
        "ep": 1,
        "vp": 1,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_device_boost(intra_node_num=8, levels=2, bw_list=None,
                       eff_list=None, lat_list=None,
                       bound_list=None, p2p_bw_list=None, p2p_eff_list=None,
                       p2p_ratio=0.002):
    device = MagicMock(spec=Type)
    device.intra_node_num.return_value = intra_node_num
    device.name = "TestDev"
    device.levels = levels
    device.level_bandwidth = bw_list or [50.0, 10.0]
    device.level_efficiency = eff_list or [0.005, 0.01]
    device.level_latency = lat_list or [1e-5, 2e-5]
    device.level_bound_number = bound_list or [8, 64]
    device.p2p_bandwidth = p2p_bw_list or [300, 25]
    device.p2p_efficiency = p2p_eff_list or [0.7, 0.9]
    device.comm_scale_factor = 1.0
    device.p2p_ratio = p2p_ratio
    hw_map = {
        "tp_bw_efficiency": 0.6,
        "tp_hsdp_sync": 0.0,
    }
    device.hw = lambda key: hw_map.get(key, 0.0)
    device.level_assign = MagicMock(return_value={
        _Dim.DP: [1, 8],
        _Dim.TP: [2, 1],
        _Dim.CP: [1, 1],
        _Dim.PP: [1, 1],
    })
    return device


class TestEstimateNonHsdpPerf(unittest.TestCase):

    def test_calls_estimate_perf_and_p2p(self):
        cfg = SimpleNamespace(p=1, vp=1, m=4)
        ccfg = SimpleNamespace(ptype=P2PCommType.NONE)
        result = _estimate_non_hsdp_perf(cfg, ccfg, [10.0], None, "npu", None)
        self.assertGreater(result, 0)


class TestOpTableCPSharding(unittest.TestCase):

    def _make_cfg(self, **overrides):
        defaults = {
            "n_kv": 0, "a": 8, "b": 1, "s": 128, "h": 16,
            "hff": 64, "cp": 2, "bytes_p": 2, "t": 2, "sp": 1,
            "dc_kv": 0, "n_exp": 1, "ep": 1,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_n_gather_is_zero(self):
        cfg = self._make_cfg()
        table = op_table(cfg)
        self.assertEqual(table["n_gather"], 0)

    def test_att_bmm_divided_by_cp(self):
        cfg_cp1 = self._make_cfg(cp=1)
        cfg_cp2 = self._make_cfg(cp=2)
        t1 = op_table(cfg_cp1)
        t2 = op_table(cfg_cp2)
        ratio = t2["n_attBMM"] / t1["n_attBMM"]
        self.assertAlmostEqual(ratio, 0.25, places=5)


class TestEstimateOpBulkCompScaling(unittest.TestCase):

    def _make_cfg_for_comp(self, d=8, t=2, d_shard_or_d=8, **kwargs):
        defaults = {
            "d": d, "t": t, "cp": 1, "p": 1, "m": 4,
            "fsdp": True, "d_shard_or_d": d_shard_or_d,
            "n_gather": 1, "b": 1, "s": 8, "h": 16, "hff": 64,
            "n_exp": 1, "ep": 1, "vp": 1,
            "n_kv": 0, "a": 8, "bytes_p": 2, "sp": 1,
            "dc_kv": 0, "v": 64, "n_mtp": 1,
            "n_lay": 2,
            "layer_custom_config": [(2, None)],
            "has_op": True, "has_grad_shard": True,
            "n_headCast": 1, "n_ffAct": 1,
            "n_attMM": 1, "n_ffMM": 1, "n_attBMM": 1, "n_ffBMM": 1,
            "n_softmax": 1, "n_normOp": 1, "n_dropout": 1,
            "rec_op": SimpleNamespace(
                attMM=1, ffMM=1, attBMM=1, ffBMM=1,
                softmax=1, headCast=1, gather=1, ffAct=1,
                normOp=1, dropout=1,
            ),
            "hff_exp": 64, "n_chosen_exp": 2, "n_shared_exp": 1,
            "cap_fact": 1, "d_exp": d, "t_exp": t,
            "os_max_shard": d * t,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def _make_stages(self):
        return [[[LayerType.NOT_REC_LAYER]]]

    def test_fsdp_branch_d_replicate_gt1(self):
        cfg = self._make_cfg_for_comp(d=8, t=2, d_shard_or_d=4)
        flops = estimate_op_bulk_comp(
            cfg, None, self._make_stages(), device_type=Hard.Device_V4
        )
        self.assertEqual(len(flops), 1)
        self.assertGreater(flops[0], 0)

    def test_non_fsdp_branch_comp_tp_scale(self):
        cfg = self._make_cfg_for_comp(d=8, t=2, d_shard_or_d=8)
        flops = estimate_op_bulk_comp(
            cfg, None, self._make_stages(), device_type=Hard.Device_V4
        )
        self.assertEqual(len(flops), 1)
        self.assertGreater(flops[0], 0)


class TestEstimateStageDebuggerBranches(unittest.TestCase):

    def _make_debugger(self, enabled=True):
        debugger = MagicMock()
        debugger.is_enabled.return_value = enabled
        debugger.info = {
            PerfParts.DP_COMM: [1.0, 2.0],
            PerfParts.MP_COMM: [0.5, 1.0],
            PerfParts.EP_COMM: [0.0, 0.0],
            PerfParts.CP_COMM: [0.0, 0.0],
            PerfParts.FW_COMPUTE: [10.0, 20.0],
        }
        return debugger

    def test_fsdp_comm_in_debugger(self):
        cfg = _make_cfg_boost()
        ccfg = SimpleNamespace(rtype=RatioType.DYNAMIC, static_ratio=None, dynamic_ratio=None)
        debugger = self._make_debugger()
        debugger.info[PerfParts.FSDP_COMM] = [1.0, 2.0]
        estimate_stage(cfg, ccfg, [10.0, 20.0], [1.0, 2.0], [2.0, 4.0], [0.5, 1.0], debugger=debugger)
        self.assertIn(PerfParts.FSDP_COMM, debugger.info)


class TestEstimatePipelineDebuggerBranches(unittest.TestCase):

    def _make_debugger(self, enabled=True):
        debugger = MagicMock()
        debugger.is_enabled.return_value = enabled
        debugger.info = {
            PerfParts.DP_COMM: [3.0, 5.0],
            PerfParts.MP_COMM: [1.0, 2.0],
            PerfParts.EP_COMM: [0.0, 0.0],
            PerfParts.CP_COMM: [0.0, 0.0],
            PerfParts.FSDP_COMM: [0.5, 1.0],
            PerfParts.FW_COMPUTE: [10.0, 20.0],
            PerfParts.BW_COMPUTE: [20.0, 40.0],
            PerfParts.RECOMPUTE: [2.0, 4.0],
        }
        return debugger

    def test_fsdp_comm_in_pipeline_debugger(self):
        cfg = SimpleNamespace(p=2, vp=1, m=4)
        debugger = self._make_debugger()
        result = estimate_pipeline(cfg, [3.0, 5.0], debugger=debugger)
        self.assertIn(PerfParts.FSDP_COMM, debugger.info)
        self.assertGreater(result, 0)


class TestCountP2PMessages(unittest.TestCase):

    def test_vp1_p2(self):
        cfg = SimpleNamespace(p=2, m=4, vp=1)
        self.assertEqual(_count_p2p_messages(cfg), 4 * 4)


class TestEstimateP2pCommDeviceRatio(unittest.TestCase):

    def test_with_device_type_ratio(self):
        device = _make_device_boost(p2p_ratio=0.005)
        cfg = SimpleNamespace(p=2, m=4, vp=1, sp=1)
        result = estimate_p2p_comm(cfg, 10.0, device_type=device)
        self.assertGreater(result, 0)
        expected_ratio = 0.005
        nb = _count_p2p_messages(cfg)
        expected = expected_ratio * nb / cfg.p * 10.0 / cfg.sp
        self.assertAlmostEqual(result, expected, places=6)


class TestEstimateP2pWithDeviceType(unittest.TestCase):

    def test_manual_with_device_type(self):
        device = _make_device_boost(p2p_ratio=0.001)
        cfg = SimpleNamespace(p=2, m=4, vp=1, sp=1)
        ccfg = SimpleNamespace(ptype=P2PCommType.MANUAL)
        result = estimate_p2p(cfg, ccfg, [10.0], device_type=device)
        self.assertGreater(result, 0)


class TestEstimateHsdpPerf(unittest.TestCase):

    def _make_cfg_hsdp(self, d=8, t=2, d_shard=4, **kwargs):
        defaults = {
            "d": d, "t": t, "cp": 1, "p": 1, "m": 4,
            "fsdp": True, "d_shard": d_shard,
            "d_shard_or_d": d_shard,
            "d_replicate": d // d_shard,
            "n_gather": 1, "b": 1, "s": 8, "h": 16, "hff": 64,
            "n_exp": 1, "ep": 1, "vp": 1,
            "n_lay": 2,
            "sp": 1,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_basic_hsdp_perf(self):
        cfg = self._make_cfg_hsdp()
        device_type = Hard.Device_V4
        result = _estimate_hsdp_perf(cfg, 4, device_type, None)
        self.assertGreater(result, 0)

    def test_hsdp_perf_with_debugger(self):
        cfg = self._make_cfg_hsdp()
        device_type = Hard.Device_V4
        debugger = MagicMock()
        debugger.is_enabled.return_value = True
        debugger.info = {}
        result = _estimate_hsdp_perf(cfg, 4, device_type, debugger)
        self.assertGreater(result, 0)
        self.assertIn(PerfParts.FW_COMPUTE, debugger.info)
        self.assertIn(PerfParts.FSDP_COMM, debugger.info)


class TestComputeStagePerfs(unittest.TestCase):

    @patch("hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate.estimate_comm", return_value=[1.0])
    @patch("hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate.estimate_comp", return_value=[10.0])
    def test_basic_compute_stage_perfs(self, mock_comp, mock_comm):
        cfg = SimpleNamespace(p=1, d_shard_or_d=8, d=8)
        ccfg = SimpleNamespace(retype=RecType.NONE, rtype=RatioType.COMPUTE_ONLY,
                               static_ratio=None, dynamic_ratio=None)
        stages = [[[LayerType.NOT_REC_LAYER]]]
        result = _compute_stage_perfs(cfg, ccfg, stages, None, Hard.Device_V4)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0], 0)


class TestApplyRegressionCoefficientsGuard(unittest.TestCase):

    def test_non_perfparts_key_skipped(self):
        debugger = MagicMock()
        debugger.is_enabled.return_value = True
        debugger.info = {
            PerfParts.FW_COMPUTE: 10.0,
            "COMM_RATIO": 1.0,
            PerfParts.TOTAL: 100.0,
        }
        coeffs = {"COMPUTE": 1.5, "FW_COMPUTE": 1.0, "BW_COMPUTE": 1.0,
                  "RECOMPUTE": 1.0, "DP_COMM": 1.0, "MP_COMM": 1.0,
                  "EP_COMM": 1.0, "CP_COMM": 1.0, "FSDP_COMM": 1.0,
                  "PP_COMM": 1.0, "BUBBLE": 1.0}
        result = apply_regression_coefficients(coeffs, debugger, 100.0)
        self.assertIsNotNone(result)


class TestEstimatePerformanceBranching(unittest.TestCase):

    def _make_full_cfg(self, d=8, t=2, d_shard_or_d=8, **kwargs):
        defaults = {
            "d": d, "t": t, "cp": 1, "p": 1, "m": 4,
            "fsdp": True, "d_shard_or_d": d_shard_or_d,
            "n_gather": 1, "b": 1, "s": 8, "h": 16, "hff": 64,
            "n_exp": 1, "ep": 1, "vp": 1,
            "n_kv": 0, "a": 8, "bytes_p": 2, "sp": 1,
            "dc_kv": 0, "v": 64, "n_mtp": 1,
            "n_lay": 2,
            "layer_custom_config": [(2, None)],
            "has_op": True, "has_grad_shard": True,
            "n_headCast": 1, "n_ffAct": 1,
            "n_attMM": 1, "n_ffMM": 1, "n_attBMM": 1, "n_ffBMM": 1,
            "n_softmax": 1, "n_normOp": 1, "n_dropout": 1,
            "rec_op": SimpleNamespace(
                attMM=1, ffMM=1, attBMM=1, ffBMM=1,
                softmax=1, headCast=1, gather=1, ffAct=1,
                normOp=1, dropout=1,
            ),
            "hff_exp": 64, "n_chosen_exp": 2, "n_shared_exp": 1,
            "cap_fact": 1, "d_exp": d, "t_exp": t,
            "os_max_shard": d * t,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    @patch("hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate.check_and_apply_custom_hook")
    @patch("hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate._resolve_estimate_args")
    @patch("hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate._compute_stage_perfs", return_value=[10.0])
    def test_non_hsdp_branch(self, mock_stage_perfs, mock_resolve, mock_hook):
        cfg = self._make_full_cfg(d=8, d_shard_or_d=8)
        ccfg = SimpleNamespace(
            rtype=RatioType.DYNAMIC, ptype=P2PCommType.NONE,
            retype=RecType.NONE, static_ratio=None, dynamic_ratio=None,
        )
        mock_resolve.return_value = (cfg, [[[LayerType.NOT_REC_LAYER]]], None, ccfg, None, Hard.Device_V4, None)
        result = estimate_performance(cfg, None, ccfg=ccfg, device_type=Hard.Device_V4)
        self.assertGreater(result, 0)

    @patch("hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate.check_and_apply_custom_hook")
    @patch("hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate._resolve_estimate_args")
    @patch("hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate._compute_stage_perfs", return_value=[10.0])
    def test_hsdp_branch(self, mock_stage_perfs, mock_resolve, mock_hook):
        cfg = self._make_full_cfg(d=8, d_shard_or_d=4)
        ccfg = SimpleNamespace(
            rtype=RatioType.DYNAMIC, ptype=P2PCommType.NONE,
            retype=RecType.NONE, static_ratio=None, dynamic_ratio=None,
        )
        mock_resolve.return_value = (cfg, [[[LayerType.NOT_REC_LAYER]]], None, ccfg, None, Hard.Device_V4, None)
        result = estimate_performance(cfg, None, ccfg=ccfg, device_type=Hard.Device_V4)
        self.assertGreater(result, 0)


if __name__ == "__main__":
    unittest.main()