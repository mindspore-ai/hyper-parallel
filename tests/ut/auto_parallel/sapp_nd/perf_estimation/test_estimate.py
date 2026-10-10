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
"""Tests for the performance estimate's entry point.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/perf_estimation/test_estimate.py -v
"""
import os
import unittest
from typing import Any, Dict

import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import estimate_performance

DEEPSEEK_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nd", "deepseek.yaml"
)


def _plain_values(ccfg: CostModelConfig) -> Dict[str, Any]:
    """The config's plain fields, the ones layer hooks overwrite."""
    return {name: value for name, value in vars(ccfg).items()
            if isinstance(value, (bool, int, float, str))}


class TestEstimatePerformance(unittest.TestCase):
    """The estimate leaves the caller's config as it found it."""

    def test_estimating_twice_gives_the_same_score(self):
        """
        Feature: estimate_performance.
        Description: DeepSeek's dense and MoE layers are two groups, and the
            estimate applies their hooks to its config in place.  Estimate
            the same config twice.
        Expectation: The caller's config comes back unchanged, and the
            second score equals the first.
        """
        ccfg = CostModelConfig(DEEPSEEK_YAML)
        before = _plain_values(ccfg)
        first = estimate_performance(ccfg, device_type=Hard.Device_A2)
        self.assertEqual(_plain_values(ccfg), before)
        self.assertEqual(estimate_performance(ccfg, device_type=Hard.Device_A2), first)


if __name__ == "__main__":
    unittest.main()
