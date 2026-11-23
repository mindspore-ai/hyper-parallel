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
"""Independent pair-coverage and routing oracles for shared CP task plans."""

from itertools import accumulate
import unittest

from hyper_parallel.distributed.context_parallel.query_balance import query_task_plan, query_route
from hyper_parallel.distributed.context_parallel.ring_layout import ring_task_plan


def _rows(tasks):
    return [(task.document, position) for task in tasks for position in range(task.start, task.end)]


class TestQueryBalance(unittest.TestCase):
    """Verify complete attention graphs, including repeated packed document crossings."""

    def test_every_pair_exactly_once(self):
        for lengths in ((48,), (32, 16), (1, 3, 5, 23), (10, 182, 64), (1,) * 16):
            for size in (1, 2, 4, 8):
                for causal in (False, True):
                    for mode in ("none", "zigzag", "greedy", "hybrid"):
                        with self.subTest(lengths=lengths, size=size, causal=causal, mode=mode):
                            block = min(4, sum(lengths) // size) if mode in ("greedy", "hybrid") else None
                            threshold = 8 if mode == "hybrid" else None
                            pairs = []
                            for rank in range(size):
                                plan = ring_task_plan(lengths, size, rank, mode, causal, block, threshold)
                                query_rows = _rows(plan.assignments[rank])
                                for owner, blocks in enumerate(plan.blocks):
                                    key_rows = _rows(plan.assignments[owner])
                                    for part in blocks:
                                        qcount = part.query_end - part.query_start
                                        kcount = part.key_end - part.key_start
                                        for qi in range(qcount):
                                            for ki in range(kcount):
                                                if not part.causal or ki <= qi + kcount - qcount:
                                                    pairs.append((query_rows[part.query_start + qi],
                                                                  key_rows[part.key_start + ki]))
                            expected = [((doc, qi), (doc, ki)) for doc, length in enumerate(lengths)
                                        for qi in range(length) for ki in range(length)
                                        if not causal or ki <= qi]
                            self.assertEqual(sorted(pairs), sorted(expected))

    def test_route_delivers_original_positions_and_inverts(self):
        lengths, size = (1, 3, 5, 23), 8
        boundaries, local = (0, *accumulate(lengths)), sum(lengths) // size
        for mode in ("none", "zigzag", "greedy", "hybrid"):
            plan = query_task_plan(lengths, size, mode, block_size=2 if mode in ("greedy", "hybrid") else None,
                                   zigzag_min_length=8 if mode == "hybrid" else None)
            routes = [query_route(plan, lengths, rank) for rank in range(size)]
            workers = [[None] * route.query_count for route in routes]
            for owner, route in enumerate(routes):
                for source, target in zip(route.local_sources, route.local_targets):
                    workers[owner][target] = owner * local + source
                cursor = 0
                for worker, count in enumerate(route.input_splits):
                    self.assertEqual(count, routes[worker].output_splits[owner])
                    receive_start = sum(routes[worker].output_splits[:owner])
                    for source, target in zip(route.send_rows[cursor:cursor+count],
                                              routes[worker].receive_rows[receive_start:receive_start+count]):
                        workers[worker][target] = owner * local + source
                    cursor += count
            for worker, tasks in enumerate(plan):
                self.assertEqual(workers[worker], [boundaries[doc] + pos for doc, pos in _rows(tasks)])
            self.assertEqual(sorted(row for rows in workers for row in rows), list(range(sum(lengths))))

    def test_fast_path_and_capacity(self):
        for size in (2, 4, 8):
            for mode in ("zigzag", "hybrid"):
                plan = ring_task_plan((128 * size,), size, 0, mode, True,
                                      zigzag_min_length=128 if mode == "hybrid" else None)
                self.assertTrue(plan.fast_zigzag)
                self.assertTrue(all(len(blocks) == 1 for blocks in plan.blocks))
        for mode in ("greedy", "hybrid"):
            plans = query_task_plan((10, 182, 64), 8, mode, block_size=8,
                                    zigzag_min_length=128 if mode == "hybrid" else None)
            self.assertTrue(all(sum(task.length for task in tasks) <= 64 for tasks in plans))

    def test_disjoint_queries_and_bounded_packed_keys(self):
        for lengths in ((256,), (1, 7, 67, 181), (65536,)):
            for mode in ("none", "zigzag", "greedy", "hybrid"):
                for rank in range(8):
                    plan = ring_task_plan(lengths, 8, rank, mode, True,
                                          block_size=min(256, sum(lengths) // 8)
                                          if mode in ("greedy", "hybrid") else None,
                                          zigzag_min_length=128 if mode == "hybrid" else None)
                    for blocks, batches in zip(plan.blocks, plan.batches):
                        queries = [row for block in blocks for row in range(block.query_start, block.query_end)]
                        self.assertEqual(len(queries), len(set(queries)))
                        self.assertEqual(tuple(part for batch in batches for part in batch), blocks)
                        self.assertTrue(all(sum(part.key_end - part.key_start for part in batch)
                                            <= 4 * plan.capacity for batch in batches))
                        self.assertTrue(all(len({part.causal for part in batch}) == 1 for batch in batches))

    def test_invalid_metadata(self):
        for lengths, size, options in (((0, 8), 2, {}), ((7,), 2, {}), ((8,), 0, {}),
                                       ((8,), 2, {"mode": "hybrid"}), ((8,), 2, {"block_size": 2}),
                                       ((8,), 2, {"mode": "greedy", "block_size": 5})):
            with self.subTest(lengths=lengths, options=options), self.assertRaises(ValueError):
                query_task_plan(lengths, size, **options)
