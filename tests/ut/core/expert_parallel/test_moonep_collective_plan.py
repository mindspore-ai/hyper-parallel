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

"""CPU planner invariant coverage without distributed initialization."""

import json
import random
import unittest
from pathlib import Path

from hyper_parallel.distributed.expert_parallel.moonep_collective_plan import (
    build_collective_plan,
)


class TestCollectivePlan(unittest.TestCase):
    """Check conservation, determinism and exact execution balance."""

    def test_conservation_and_balance(self):
        generator = random.Random(42)
        for size in (1, 2, 4, 8):
            for trial in range(40):
                counts = [[generator.randrange(12) for _ in range(size * 2)] for _ in range(size)]
                if trial == 0:
                    counts = [[0] * (size * 2) for _ in range(size)]
                if trial == 1:
                    counts = [[101] + [0] * (size * 2 - 1) for _ in range(size)]
                with self.subTest(size=size, trial=trial):
                    plan = build_collective_plan(counts)
                    self.assertEqual(plan, build_collective_plan(counts))
                    seen = set()
                    loads = [0] * size
                    for segment in plan.segments:
                        for offset in range(segment.offset, segment.offset + segment.count):
                            key = (segment.source, segment.expert, offset)
                            self.assertNotIn(key, seen)
                            seen.add(key)
                            self.assertLess(offset, counts[segment.source][segment.expert])
                        loads[segment.destination] += segment.count
                    self.assertEqual(len(seen), sum(map(sum, counts)))
                    self.assertEqual(tuple(loads), plan.execution_counts)
                    self.assertLessEqual(max(loads) - min(loads), 1)
                    for source in range(size):
                        send, _ = plan.splits(source)
                        self.assertEqual(sum(send), sum(counts[source]))
                        for destination in range(size):
                            self.assertEqual(send[destination], plan.splits(destination)[1][source])

    def test_native_host_planner_golden_allocations(self):
        """Match native host allocation policy without importing its package."""
        fixture = Path(__file__).with_name("fixtures") / "moonep_host_allocations.json"
        cases = json.loads(fixture.read_text())["cases"]
        for index, case in enumerate(cases):
            with self.subTest(index=index):
                plan = build_collective_plan(case["counts"])
                allocation = [[0] * len(plan.counts) for _ in plan.counts[0]]
                for segment in plan.segments:
                    allocation[segment.expert][segment.destination] += segment.count
                self.assertEqual(allocation, case["allocation"])

    def test_invalid_counts(self):
        for counts in ([], [[]], [[1], [2]], [[1, 0], [1]], [[-1]], [[1.0]]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                build_collective_plan(counts)
