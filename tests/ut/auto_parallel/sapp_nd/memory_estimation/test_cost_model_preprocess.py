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
"""Unit tests for cost_model_preprocess.py: strategy_num_devices.

Tests the strategy_num_devices formula directly to avoid
circular import caused by cost_model_preprocess -> generate_partitions
-> memory_estimation -> _backbone -> cost_model_preprocess chain.

EP borrows devices from DP, so strategy_num_devices = d * t * cp * p
(without ep). See Thi's review: N/O/P run on 16 cards with fixed DP=8
while EP sweeps 2/4/8; *ep would claim 32/64/128 cards.

Test IDs:
  CM-N01: strategy_num_devices basic formula d*t*cp*p
  CM-N02: strategy_num_devices ignores ep (EP borrows from DP)
"""
import os
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"


def _get_cost_model_config():
    """Import CostModelConfig avoiding circular import.

    Break the cycle by pre-loading memory_estimation submodules before
    cost_model_preprocess tries to import them.
    """
    import hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context  # noqa: F401
    import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils  # noqa: F401
    import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size  # noqa: F401
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
    return CostModelConfig


class TestStrategyNumDevices(unittest.TestCase):
    """CM-N: strategy_num_devices tests."""

    @classmethod
    def setUpClass(cls):
        cls.CostModelConfig = _get_cost_model_config()

    def _make_ccfg(self, d=2, t=2, cp=1, p=4, ep=8):
        ccfg = self.CostModelConfig.__new__(self.CostModelConfig)
        ccfg.d = d
        ccfg.t = t
        ccfg.cp = cp
        ccfg.p = p
        ccfg.ep = ep
        return ccfg

    def test_basic_formula(self):
        """CM-N01: strategy_num_devices = d * t * cp * p."""
        ccfg = self._make_ccfg(d=4, t=2, cp=1, p=4, ep=8)
        result = ccfg.strategy_num_devices()
        self.assertEqual(result, 4 * 2 * 1 * 4)

    def test_ignores_ep(self):
        """CM-N02: strategy_num_devices ignores ep (EP borrows from DP)."""
        ccfg_ep1 = self._make_ccfg(d=8, t=1, cp=1, p=2, ep=1)
        ccfg_ep4 = self._make_ccfg(d=8, t=1, cp=1, p=2, ep=4)
        self.assertEqual(
            ccfg_ep1.strategy_num_devices(), ccfg_ep4.strategy_num_devices()
        )


if __name__ == "__main__":
    unittest.main()
