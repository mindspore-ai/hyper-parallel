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
"""Unit tests for hardware.py: EP device assignment and level_assign.

Test IDs:
  HW-L01: level_assign includes EP in device_number product
  HW-L02: level_assign distributes EP across hierarchy levels
  HW-L03: level_assign with EP=1 (no EP) matches old behavior
  HW-L04: level_assign with EP=8 on A2 (8 intra, single node)
  HW-L05: Dim.EP is a valid Dimension key in assignment
"""
import os
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import Type, Device_A2


class TestLevelAssign(unittest.TestCase):
    """HW-L: level_assign EP integration tests."""

    def test_device_number_includes_ep(self):
        """
        Feature: TestLevelAssign.
        Description: device_number = dp*tp*cp*pp*ep includes EP.
        Expectation: level_assign result contains Dim.EP key.
        """
        hw = Type("test", [8, None], [50, 10])
        # dp=2, tp=2, cp=1, pp=2, ep=2 → 16 devices
        result = hw.level_assign(dp=2, tp=2, cp=1, pp=2, ep=2)
        # Just verify the assignment dict has EP key
        self.assertIn(Dim.EP, result)

    def test_ep_distributed_across_levels(self):
        """
        Feature: TestLevelAssign.
        Description: EP=4 assigned across levels on A2.
        Expectation: Product of EP across levels equals original ep value.
        """
        result = Device_A2.level_assign(dp=1, tp=2, cp=1, pp=1, ep=4)
        # A2 has bounds=[8, None], 8 devices per node
        # level_assign distributes TP first, then EP from remaining
        ep_assignment = result[Dim.EP]
        # Product of EP across levels equals original ep (before level_assign)
        product = 1
        for v in ep_assignment:
            product *= v
        self.assertEqual(product, 4)

    def test_ep1_matches_no_ep(self):
        """
        Feature: TestLevelAssign.
        Description: EP=1 produces same device count as EP omitted.
        Expectation: EP assignment is [1, 1] across levels.
        """
        hw = Type("test", [8, None], [50, 10])
        result_ep1 = hw.level_assign(dp=2, tp=2, cp=1, pp=1, ep=1)
        # With ep=1, device_number = 2*2*1*1*1 = 4
        # EP assignment should be [1, 1] across levels
        ep_vals = result_ep1[Dim.EP]
        self.assertEqual(ep_vals, [1, 1])

    def test_ep8_on_a2(self):
        """
        Feature: TestLevelAssign.
        Description: EP=8 with TP=1 on A2 fills intra-node.
        Expectation: EP assignment first level equals 8.
        """
        result = Device_A2.level_assign(dp=1, tp=1, cp=1, pp=1, ep=8)
        ep_assignment = result[Dim.EP]
        self.assertEqual(ep_assignment[0], 8)

    def test_dim_ep_key(self):
        """
        Feature: TestLevelAssign.
        Description: Dim.EP is present in assignment dict.
        Expectation: Dim.EP key exists and value is a list.
        """
        hw = Type("test", [4, None], [50, 10])
        result = hw.level_assign(dp=1, tp=1, cp=1, pp=1, ep=2)
        self.assertIn(Dim.EP, result)
        self.assertIsInstance(result[Dim.EP], list)


if __name__ == "__main__":
    unittest.main()
