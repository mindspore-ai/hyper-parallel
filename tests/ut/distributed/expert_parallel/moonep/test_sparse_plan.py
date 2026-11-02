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
"""CPU coverage of sparse MoonEP route conservation and inverse placement."""

import unittest

import torch

from hyper_parallel.distributed.expert_parallel.moonep.fused.plan import compile_local_route, compile_sparse_route
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestSparsePlan(unittest.TestCase):
    """Validate individual selection ownership, including empty ranks and tails."""

    def test_split_expert_offsets(self):
        """Splitting e0 preserves source order while packing only active experts."""
        routes = compile_sparse_route(torch.tensor([[4, 3, 0, 0], [1, 0, 0, 0]]),
                                      torch.tensor([[1, 4], [3, 0], [0, 0], [0, 0]]))
        self.assertEqual(routes[0].active_experts, (0, 1))
        self.assertEqual(routes[1].active_experts, (0,))
        self.assertEqual(routes[0].boundaries, (1, 4))
        self.assertEqual(routes[1].boundaries, (4,))
        self.assertEqual(routes[0].packed_edges().tolist(),
                         [[0, 0, 0, 0, 0, 1], [0, 1, 0, 1, 0, 3], [0, 0, 1, 4, 1, 3]])
        self.assertEqual(routes[1].packed_edges().tolist(), [[1, 1, 0, 0, 3, 1]])
        self.assertEqual([edge.rows for edge in routes[0].communication_blocks(2)], [2, 1])

    def test_empty_routes_have_no_tasks(self):
        """Zero rows create neither artificial slots nor zero-length messages."""
        routes = compile_sparse_route(torch.zeros(2, 4, dtype=torch.int64),
                                      torch.zeros(4, 2, dtype=torch.int64))
        for route in routes:
            self.assertEqual(route.active_experts, ())
            self.assertEqual(route.communication_blocks(128), ())
            self.assertEqual(route.packed_edges().shape, (0, 6))

    def test_random_routes_are_bijective(self):
        """An independent row-by-row oracle checks conservation and exact order."""
        generator = torch.Generator().manual_seed(2718)
        for ranks in (1, 2, 3, 16):
            for trial in range(12):
                experts = 2 * ranks
                counts = torch.randint(0, 5, (ranks, experts), generator=generator)
                allocation = torch.zeros(experts, ranks, dtype=torch.int64)
                expected = [[] for _ in range(ranks)]
                source_rows = [[] for _ in range(ranks)]
                for expert in range(experts):
                    rows = [(src, expert, row) for src in range(ranks) for row in range(int(counts[src, expert]))]
                    destinations = torch.randint(ranks, (len(rows),), generator=generator).sort().values.tolist()
                    for row, dst in zip(rows, destinations):
                        allocation[expert, dst] += 1
                        expected[dst].append(row)
                    for src in range(ranks):
                        source_rows[src].extend(row for row in rows if row[0] == src)
                routes = compile_sparse_route(counts, allocation)
                for rank, route in enumerate(routes):
                    self.assertEqual(compile_local_route(counts, allocation, rank), route)
                actual = [[None] * route.received_rows for route in routes]
                seen = [set() for _ in range(ranks)]
                for route in routes:
                    for edge in route.outgoing:
                        for offset in range(edge.rows):
                            row = edge.source_row + offset
                            self.assertNotIn(row, seen[edge.source])
                            seen[edge.source].add(row)
                            target_row = edge.target_row + offset
                            self.assertIsNone(actual[edge.target][target_row])
                            actual[edge.target][target_row] = source_rows[edge.source][row]
                with self.subTest(ranks=ranks, trial=trial):
                    self.assertEqual(actual, expected)
                    self.assertEqual([len(rows) for rows in seen], [len(rows) for rows in source_rows])

    def test_invalid_tables_fail_before_prefix_arithmetic(self):
        """Lost selections, nonintegral tables and int64 overflow fail explicitly."""
        for counts, allocation in ((torch.tensor([[1]]), torch.tensor([[0]])),
                                   (torch.tensor([[-1]]), torch.tensor([[-1]])),
                                   (torch.tensor([[1.0]]), torch.tensor([[1]])),
                                   (torch.tensor([[2**63 - 1, 1]]), torch.tensor([[2**63 - 1], [1]]))):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                compile_sparse_route(counts, allocation)
            with self.subTest(local_counts=counts), self.assertRaises(ValueError):
                compile_local_route(counts, allocation, 0)

    def test_local_rank_validation(self):
        """A local compiler must reject malformed or out-of-group PE indices."""
        counts = torch.zeros((2, 4), dtype=torch.int64)
        allocation = torch.zeros((4, 2), dtype=torch.int64)
        for rank in (-1, 2, True, 0.5):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                compile_local_route(counts, allocation, rank)
        for rank in range(2):
            route = compile_local_route(counts, allocation, rank)
            self.assertEqual(route.rank, rank)
            self.assertEqual(route.incoming, ())
            self.assertEqual(route.outgoing, ())

    def test_bad_block_size_rejected_even_for_empty_rank(self):
        """An empty route cannot hide a malformed transport configuration."""
        route = compile_sparse_route(torch.zeros(1, 1, dtype=torch.int64),
                                     torch.zeros(1, 1, dtype=torch.int64))[0]
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                route.communication_blocks(size)
