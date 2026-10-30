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
"""CPU contracts for planner count formats and optional execution dependencies."""
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.expert_parallel.moonep import placement


class TestMoonEPPlacement(unittest.TestCase):
    """Reject invalid metadata before any optional device kernel is called."""

    def test_ungated_cost_accounts_for_two_projections(self) -> None:
        """A marginal gated saving must not enable borrowing for a cheaper ungated FFN."""
        geometry = (200000, 0, 512, 128, 20, 0)
        self.assertTrue(placement.placement_is_profitable(*geometry, projection_flops=6))
        self.assertFalse(placement.placement_is_profitable(*geometry, projection_flops=4))

    def test_native_and_legacy_counts_share_the_int32_planner(self) -> None:
        """Normalize equivalent wire types without changing the ownership decision."""
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                counts = torch.tensor([[512, 0], [0, 0]], dtype=dtype)
                copies = torch.full((2, 1), -1, dtype=torch.int32)
                with (patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}),
                      patch.object(placement, "allocate_experts", return_value=(None, None, None, copies)) as allocate):
                    result = placement.build_placement(
                        torch.empty(512, 4), torch.zeros(512, 1, dtype=torch.int32), counts, 256, 0,
                    )
                self.assertIsNone(result)
                actual_counts, actual_capacity = allocate.call_args.args
                self.assertEqual(actual_capacity, 256)
                self.assertEqual(actual_counts.dtype, torch.int32)
                torch.testing.assert_close(actual_counts, counts.to(torch.int32))

    def test_wide_counts_reject_overflow_before_allocator(self) -> None:
        """A large INT64 wire count must never wrap to a negative planner value."""
        counts = torch.tensor([[2**31, 0], [0, 0]], dtype=torch.int64)
        with (patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}),
              patch.object(placement, "allocate_experts") as allocate,
              self.assertRaisesRegex(ValueError, "INT32 planner limit")):
            placement.build_placement(torch.empty(0, 4), torch.empty(0, 1, dtype=torch.int32), counts, 1, 0)
        allocate.assert_not_called()

    def test_invalid_topologies_fail_before_allocator(self) -> None:
        """An invalid rank or expert extent cannot launch an asymmetric plan."""
        for counts, rank in ((torch.zeros(2, 3, dtype=torch.int32), 0),
                             (torch.zeros(2, 4, dtype=torch.int32), 2),
                             (torch.zeros(4, dtype=torch.int32), 0)):
            with self.subTest(shape=counts.shape, rank=rank):
                with (patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}),
                      patch.object(placement, "allocate_experts") as allocate,
                      self.assertRaises(ValueError)):
                    placement.build_placement(
                        torch.empty(0, 4), torch.empty(0, 1, dtype=torch.int32), counts, 1, rank,
                    )
                allocate.assert_not_called()
