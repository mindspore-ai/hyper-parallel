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
"""CPU checks for whole-expert MoonEP migration plans."""

import unittest

import torch

from hyper_parallel.distributed.expert_parallel.moonep.planning import (
    compile_plan, select_migration_allocation,
)


class TestMigrationPlanning(unittest.TestCase):
    """Check bounded expert swaps and uneven route compilation."""

    def test_swap_balances_load_without_extra_expert_slots(self) -> None:
        """A hot home rank transfers one full expert and receives a cold one."""
        counts = torch.tensor([[5, 4, 0, 0], [0, 0, 1, 2]], dtype=torch.int64)
        copied, allocation = select_migration_allocation(counts)

        torch.testing.assert_close(copied, counts)
        torch.testing.assert_close(allocation.sum(1), counts.sum(0))
        self.assertEqual((allocation > 0).sum(1).tolist(), [1, 1, 1, 1])
        self.assertEqual((allocation > 0).sum(0).tolist(), [2, 2])
        self.assertEqual(allocation.sum(0).tolist(), [6, 6])
        plan = compile_plan(copied, allocation, torch.tensor([0] * 5 + [1] * 4), 0, allow_uneven=True)
        self.assertEqual(sum(plan.token_send), 9)
        self.assertEqual(plan.metrics[0], 1)
        self.assertEqual(plan.migration_virtual_ids.tolist(), [2, 0, 3, 1])
        self.assertEqual(plan.dispatch_order.numel(), 0)

    def test_unhelpful_swap_is_skipped_but_uneven_plan_is_valid(self) -> None:
        """An indivisible hot expert does not cause an endless swap cycle."""
        counts = torch.tensor([[8, 1, 0, 0], [0, 0, 1, 2]], dtype=torch.int64)
        copied, allocation = select_migration_allocation(counts)

        self.assertEqual(allocation.argmax(1).tolist(), [0, 0, 1, 1])
        plan = compile_plan(copied, allocation, torch.tensor([0] * 8 + [1]), 0, allow_uneven=True)
        self.assertEqual(plan.metrics[0], 1)
        self.assertEqual(sum(plan.token_recv), 9)

    def test_invalid_counts_are_rejected(self) -> None:
        """Malformed or negative route tables fail before planning."""
        for counts in (torch.ones((2, 3), dtype=torch.int64),
                       torch.ones((2, 4), dtype=torch.float32),
                       torch.tensor([[1, -1], [0, 1]], dtype=torch.int64)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                select_migration_allocation(counts)


if __name__ == "__main__":
    unittest.main()
