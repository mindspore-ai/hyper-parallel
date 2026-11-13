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
"""CPU coverage and inverse routing for interval-based Indexer dispatch."""

from dataclasses import replace
from itertools import accumulate
import unittest

import torch

from hyper_parallel.distributed.context_parallel import indexer_balance


class TestIndexerBalance(unittest.TestCase):
    """Keep original query ownership and peer ordering through compact intervals."""

    def tearDown(self) -> None:
        """Do not retain routing or device-index cache entries across cases."""
        indexer_balance.indexer_route.cache_clear()
        indexer_balance.indexer_task_plan.cache_clear()
        indexer_balance._device_rows.cache_clear()

    def test_dispatch_and_return_cover_each_real_query_once(self):
        """Compare every peer payload to original document coordinates, including empty peers."""
        for lengths, size in (((3, 5, 24), 4), ((1,) * 16, 4), ((32,), 1)):
            local = sum(lengths) // size
            boundaries = (0, *accumulate(lengths))
            owners = [torch.arange(rank * local, (rank + 1) * local) for rank in range(size)]
            for ratio in (1, 2):
                for mode in ("zigzag", "greedy", "hybrid"):
                    options = {"block_size": 2 if mode != "zigzag" else None,
                               "zigzag_min_length": 8 if mode == "hybrid" else None}
                    tasks = indexer_balance.indexer_task_plan(lengths, size, ratio, mode, **options)
                    routes = [indexer_balance.indexer_route(lengths, size, rank, ratio, mode, **options)
                              for rank in range(size)]
                    rows = [indexer_balance._device_rows(route, torch.device("cpu")) for route in routes]
                    expected = [torch.tensor([boundaries[task.document] + offset for task in assigned
                                              for offset in range(task.start, task.end)], dtype=torch.long)
                                for assigned in tasks]
                    for rank, route in enumerate(routes):
                        sent, received, sources, targets = rows[rank]
                        worker = torch.full((route.query_count,), -1, dtype=torch.long)
                        worker[targets] = owners[rank][sources]
                        pieces = []
                        for peer in range(size):
                            peer_sends = owners[peer][rows[peer][0]].split(routes[peer].input_splits)
                            self.assertEqual(routes[peer].input_splits[rank], route.output_splits[peer])
                            pieces.append(peer_sends[rank])
                        worker[received] = torch.cat(pieces)
                        torch.testing.assert_close(worker, expected[rank], rtol=0, atol=0)
                        restored = torch.full((local,), -1, dtype=torch.long)
                        restored[sources] = expected[rank][targets]
                        returned = [expected[peer][rows[peer][1]].split(routes[peer].output_splits)[rank]
                                    for peer in range(size)]
                        restored[sent] = torch.cat(returned)
                        torch.testing.assert_close(restored, owners[rank], rtol=0, atol=0)

    def test_cached_hash_keeps_full_route_equality(self):
        """Cache identity includes segment geometry even when the payload permutation matches."""
        route = indexer_balance.indexer_route((3, 29), 4, 0, 2, "greedy", 2)
        copied = replace(route)
        self.assertEqual(route, copied)
        self.assertEqual(hash(route), hash(copied))
        changed = replace(route, segments=())
        self.assertNotEqual(route, changed)
        # Equal hashes must not allow a different route to reuse cached indices.
        object.__setattr__(changed, "_cached_hash", hash(route))
        self.assertEqual(len({route: 1, copied: 2, changed: 3}), 2)
