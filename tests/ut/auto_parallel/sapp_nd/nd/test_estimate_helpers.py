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
"""Unit tests for estimate.py helper functions and HSDP FLOP branch.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/test_estimate_helpers.py
"""
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from hyper_parallel.auto_parallel.sapp_nd.nd.debug import PerfParts
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory  # noqa: F401  # pre-seed to break circular import
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig  # noqa: F401  # pre-seed to break circular import
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation import estimate as _estimate_mod


class TestCountP2PMessages(unittest.TestCase):
    """Tests for _count_p2p_messages helper."""

    def test_pp1_returns_zero(self):
        cfg = SimpleNamespace(p=1, m=4, vp=1)
        self.assertEqual(_estimate_mod._count_p2p_messages(cfg), 0)

    def test_pp2_vp1(self):
        cfg = SimpleNamespace(p=2, m=4, vp=1)
        self.assertEqual(_estimate_mod._count_p2p_messages(cfg), 4 * 4)

    def test_pp4_vp1(self):
        cfg = SimpleNamespace(p=4, m=4, vp=1)
        result = _estimate_mod._count_p2p_messages(cfg)
        expected = 4 * 4 * 4 + 4 * 4 * 4 - 14 * 4
        self.assertEqual(result, expected)

    def test_pp2_vp2(self):
        cfg = SimpleNamespace(p=2, m=4, vp=2)
        result = _estimate_mod._count_p2p_messages(cfg)
        expected = 8 * 4 * 2 - 4 * 4
        self.assertEqual(result, expected)

    def test_pp4_vp2(self):
        cfg = SimpleNamespace(p=4, m=4, vp=2)
        result = _estimate_mod._count_p2p_messages(cfg)
        expected = 16 * 4 * 2 + 12
        self.assertEqual(result, expected)

    def test_pp8_vp2_falls_through(self):
        cfg = SimpleNamespace(p=8, m=4, vp=2)
        result = _estimate_mod._count_p2p_messages(cfg)
        expected = 4 * 8 * 4 * 2 + 4 * 8 * 8 - 13 * 8
        self.assertEqual(result, expected)


class TestApplyRegressionCoefficients(unittest.TestCase):
    """Tests for FLOP-mode regression coefficient application."""

    def _make_debugger(self):
        debugger = MagicMock()
        debugger.info = {}
        for part in PerfParts:
            debugger.info[part] = 1.0
        debugger.info["COMM_RATIO"] = 1.0
        debugger.info["MB_COUNT"] = 1
        debugger.is_enabled.return_value = True
        return debugger

    def _make_full_coeffs(self, **overrides):
        coeffs = {"COMPUTE": 1.0}
        for part in PerfParts:
            if part not in (PerfParts.TOTAL, PerfParts.MEMORY):
                coeffs[part.name] = 1.0
        coeffs.update(overrides)
        return coeffs

    def test_skips_total_and_memory_from_loop(self):
        debugger = self._make_debugger()
        coeffs = self._make_full_coeffs(COMPUTE=2.0, DP_COMM=3.0)
        old_perf = 100.0
        result = _estimate_mod.apply_regression_coefficients(coeffs, debugger, old_perf)
        self.assertIsInstance(result, float)
        self.assertIn(PerfParts.TOTAL, debugger.info)

    def test_skips_non_perfparts_keys(self):
        debugger = self._make_debugger()
        debugger.info["COMM_RATIO"] = 5.0
        debugger.info["MB_COUNT"] = 3
        coeffs = self._make_full_coeffs(COMPUTE=2.0, DP_COMM=3.0)
        old_perf = 100.0
        _estimate_mod.apply_regression_coefficients(coeffs, debugger, old_perf)
        self.assertEqual(debugger.info["COMM_RATIO"], 5.0)
        self.assertEqual(debugger.info["MB_COUNT"], 3)

    def test_compute_parts_use_compute_ratio(self):
        debugger = self._make_debugger()
        coeffs = self._make_full_coeffs(COMPUTE=2.0)
        old_perf = 100.0
        _estimate_mod.apply_regression_coefficients(coeffs, debugger, old_perf)
        self.assertEqual(debugger.info[PerfParts.FW_COMPUTE], 2.0)
        self.assertEqual(debugger.info[PerfParts.BW_COMPUTE], 2.0)
        self.assertEqual(debugger.info[PerfParts.RECOMPUTE], 2.0)

    def test_comm_parts_use_named_ratio(self):
        debugger = self._make_debugger()
        coeffs = self._make_full_coeffs(COMPUTE=1.0, DP_COMM=5.0)
        old_perf = 100.0
        _estimate_mod.apply_regression_coefficients(coeffs, debugger, old_perf)
        self.assertEqual(debugger.info[PerfParts.DP_COMM], 5.0)

    def test_zero_raw_stays_zero(self):
        debugger = self._make_debugger()
        debugger.info[PerfParts.DP_COMM] = 0.0
        coeffs = self._make_full_coeffs(COMPUTE=1.0, DP_COMM=5.0)
        old_perf = 100.0
        _estimate_mod.apply_regression_coefficients(coeffs, debugger, old_perf)
        self.assertEqual(debugger.info[PerfParts.DP_COMM], 0.0)


if __name__ == "__main__":
    unittest.main()
