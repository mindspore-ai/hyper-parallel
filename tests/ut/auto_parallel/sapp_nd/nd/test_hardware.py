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
"""Unit tests for hardware module."""
# pylint: disable=missing-class-docstring,missing-function-docstring,W0105
"""Unit tests for hardware.py: EP device assignment and level_assign.

Test IDs:
  HW-L01: level_assign includes EP in device_number product
  HW-L02: level_assign distributes EP across hierarchy levels
  HW-L03: level_assign with EP=1 (no EP) matches old behavior
  HW-L04: level_assign with EP=8 on A2 (8 intra, single node)
  HW-L05: Dim.EP is a valid Dimension key in assignment
"""
# pylint: disable=missing-class-docstring,missing-function-docstring
import os
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import Type, Device_A2, Device_V4


class TestDeviceInstances(unittest.TestCase):

    def test_device_v4_attributes(self):
        self.assertEqual(Device_V4.level_efficiency, [0.005, 0.01])
        self.assertEqual(Device_V4.level_latency, [0.00001, 0.00002])
        self.assertEqual(Device_V4.comm_scale_factor, 1.0)
        self.assertEqual(Device_V4.p2p_bandwidth, [300, 25])
        self.assertEqual(Device_V4.p2p_efficiency, [0.7, 0.9])
        self.assertEqual(Device_V4.cp_overlap_ratio, 0.5)
        self.assertEqual(Device_V4.p2p_ratio, 0.002)
        self.assertIsInstance(Device_V4.flop_coeffs, dict)
        self.assertIn("shard", Device_V4.flop_coeffs)

    def test_device_v4_flop_coeffs_fsdp_shard(self):
        fsdp_shard = Device_V4.flop_coeffs["shard"]["fsdp"]
        for key in ("INTERCEPT", "INV_SG", "CROSS_INV_TP", "D_SHARD", "TP"):
            self.assertIn(key, fsdp_shard)

    def test_device_map_contains_v4(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import device_map
        self.assertIn("V4", device_map)
        self.assertIs(device_map["V4"], Device_V4)


class TestLevelAssignHSDP(unittest.TestCase):

    def test_hsdp_no_shard(self):
        hw = Type("test", [8, None], [50, 10])
        result = hw.level_assign(dp=4, tp=1, cp=1, pp=1, ep=1, d_shard=1)
        self.assertEqual(result[Dim.HSDP], [0, 0])

    def test_hsdp_with_d_shard_level0(self):
        hw = Type("test", [8, None], [50, 10])
        result = hw.level_assign(dp=4, tp=1, cp=1, pp=1, ep=1, d_shard=2)
        self.assertEqual(result[Dim.HSDP][0], 1)

    def test_hsdp_with_d_shard_level1_d_replicate_gt1(self):
        hw = Type("test", [8, None], [50, 10])
        result = hw.level_assign(dp=4, tp=1, cp=1, pp=1, ep=1, d_shard=2)
        self.assertEqual(result[Dim.HSDP][1], 2)


class TestGetCpTopology(unittest.TestCase):

    def test_intra_node(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import get_cp_topology
        topo, bw, is_intra = get_cp_topology(tp_degree=2, cp_degree=2, device_per_node=8)
        self.assertEqual(topo, "intra-node")
        self.assertTrue(is_intra)
        self.assertEqual(bw, 300.0)

    def test_cross_node(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import get_cp_topology
        topo, bw, is_intra = get_cp_topology(tp_degree=4, cp_degree=4, device_per_node=8)
        self.assertEqual(topo, "cross-node")
        self.assertFalse(is_intra)
        self.assertEqual(bw, 25.0)


class TestGetCpBandwidth(unittest.TestCase):

    def test_intra_node_a2(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import get_cp_bandwidth
        bw = get_cp_bandwidth("intra-node", "A2")
        self.assertEqual(bw, 50)

    def test_cross_node_a3(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import get_cp_bandwidth
        bw = get_cp_bandwidth("cross-node", "A3")
        self.assertEqual(bw, 25)


class TestRecommendCpMax(unittest.TestCase):

    def test_mla(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import recommend_cp_max_by_attention
        self.assertEqual(recommend_cp_max_by_attention("mla"), 16)

    def test_unknown_defaults_to_4(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import recommend_cp_max_by_attention
        self.assertEqual(recommend_cp_max_by_attention("unknown"), 4)


class TestDeviceV4FlopCoeffsDetail(unittest.TestCase):

    def test_shard_hsdp_keys(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import _FLOP_COEFFS_V4
        hsdp_shard = _FLOP_COEFFS_V4["shard"]["hsdp"]
        for key in ("INTERCEPT", "TP", "SG", "CROSS_AG_VOL", "CROSS_INV_TP",
                     "INV_SG_D_REP", "AG_VOL_D_REP"):
            self.assertIn(key, hsdp_shard)

    def test_flop_coeffs_are_numeric(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import _FLOP_COEFFS_V4
        for section in _FLOP_COEFFS_V4.values():
            for sub in section.values():
                for v in sub.values():
                    self.assertIsInstance(v, (int, float),
                                          msg=f"Value {v} is not numeric")


class TestMachineNewDevices(unittest.TestCase):

    def test_machine_str_device_v4(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import Machine
        m = Machine(8, "V4")
        self.assertIs(m.device, Device_V4)

    def test_machine_unknown_str_device_raises(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import Machine
        with self.assertRaises(ValueError):
            Machine(8, "UNKNOWN_DEV")


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
