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
"""CPU tests for the MoonEP to MegaMoE placement adapter."""

import unittest
from dataclasses import replace

import torch

from hyper_parallel.distributed.expert_parallel.moonep.megamoe_adapter import (
    external_route, permute_migration_input, restore_migration_output, virtual_route_counts,
)
from hyper_parallel.distributed.expert_parallel.moonep.planning import (
    compile_plan, select_allocation, select_migration_allocation,
)


class TestMegaMoeAdapter(unittest.TestCase):
    """Check that split expert placement preserves every source and receive row."""

    def test_split_expert_route_and_empty_slot_boundaries(self) -> None:
        """One source expert split across ranks retains exact virtual expert order."""
        counts = torch.tensor([[4, 3, 0, 0], [1, 0, 0, 0]], dtype=torch.int64)
        counts, allocation, _ = select_allocation(counts, capacity=2)
        plans = (
            compile_plan(counts, allocation, torch.tensor([0, 0, 0, 0, 1, 1, 1]), rank=0),
            compile_plan(counts, allocation, torch.tensor([0]), rank=1),
        )
        expected = torch.tensor([[1, 3, 0, 0, 3, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.int32)
        for rank, plan in enumerate(plans):
            with self.subTest(rank=rank):
                torch.testing.assert_close(virtual_route_counts(plan, rank, capacity=4), expected)
                route = external_route(plan, rank, capacity=4, rows_device=torch.device("cpu"))
                self.assertEqual(route.counts_by_source.dtype, torch.int32)
                self.assertEqual(route.group_boundaries.tolist(), [1, 4, 4, 4] if rank == 0 else [4, 4, 4, 4])
                self.assertEqual(int(expected[rank].sum()), sum(plan.token_send))
                self.assertEqual(int(expected[:, rank * 4:(rank + 1) * 4].sum()), sum(plan.token_recv))

    def test_missing_source_counts_is_rejected(self) -> None:
        """External routing requires the same global histogram as MoonEP's plan."""
        counts = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
        counts, allocation, _ = select_allocation(counts, capacity=1)
        plan = compile_plan(counts, allocation, torch.tensor([0]), rank=0)
        plan = replace(plan, source_counts=None)
        with self.assertRaisesRegex(ValueError, "source counts"):
            virtual_route_counts(plan, rank=0, capacity=2)

    def test_invalid_rank_and_capacity_are_rejected(self) -> None:
        """A malformed topology fails before creating native route descriptors."""
        counts = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
        counts, allocation, _ = select_allocation(counts, capacity=1)
        plan = compile_plan(counts, allocation, torch.tensor([0]), rank=0)
        for rank, capacity in ((2, 2), (0, 0)):
            with self.subTest(rank=rank, capacity=capacity), self.assertRaisesRegex(ValueError, "valid rank"):
                virtual_route_counts(plan, rank=rank, capacity=capacity)

    def test_empty_migration_route_keeps_autograd_connection(self) -> None:
        """A source with no tokens needs no NPU permutation or unpermutation."""
        counts = torch.zeros((2, 4), dtype=torch.int64)
        counts, allocation = select_migration_allocation(counts)
        plan = compile_plan(counts, allocation, torch.empty(0, dtype=torch.int64), rank=0, allow_uneven=True)
        hidden = torch.empty((0, 3), requires_grad=True)
        ids = torch.empty((0, 2), dtype=torch.int64)
        scores = torch.empty((0, 2), requires_grad=True)
        rows, inverse = permute_migration_input(hidden, ids, plan)
        output = restore_migration_output(rows, inverse, scores)
        self.assertEqual(tuple(output.shape), (0, 3))
        output.sum().backward()
        self.assertIsNotNone(scores.grad)


if __name__ == "__main__":
    unittest.main()
