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
"""Tests for deterministic sample-level dynamic-packing plans."""

import unittest

from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    SampleKey,
    SampleMetadata,
    WorkloadCost,
)


def _candidate(
        source_index: int,
        pack_tokens: int,
        *,
        cost: WorkloadCost = WorkloadCost(),
        source_rank: int = 0,
) -> BufferedSampleMetadata:
    return BufferedSampleMetadata(
        key=SampleKey(source_rank, source_index),
        metadata=SampleMetadata(pack_tokens=pack_tokens, cost=cost, sample_id=source_index),
    )


class TestDynamicPackingPlanner(unittest.TestCase):
    """Verify packing capacity, DP balance, overflow, and reproducibility."""

    def test_balances_individual_samples_and_fills_each_rank_bin(self) -> None:
        """Complementary samples are routed independently into full DP bins."""
        planner = DynamicPackingPlanner(data_parallel_size=2, seq_len=10, local_batch_size=1)
        candidates = tuple(_candidate(index, tokens) for index, tokens in enumerate((7, 7, 3, 3)))

        plan = planner.plan(candidates, step=0)

        self.assertIsNotNone(plan)
        self.assertEqual(set(plan.selected_keys), {candidate.key for candidate in candidates})
        for constructor in plan.constructors:
            packing_bin = constructor.bins[0]
            self.assertEqual(packing_bin.pack_tokens, 10)
            self.assertEqual(len(packing_bin.samples), 2)
            self.assertTrue(all(
                sample.target_data_rank == constructor.target_data_rank
                for sample in packing_bin.samples
            ))

    def test_defers_a_sample_that_cannot_fit_without_exceeding_capacity(self) -> None:
        """A non-fitting candidate remains unselected instead of overflowing a bin."""
        planner = DynamicPackingPlanner(data_parallel_size=1, seq_len=10, local_batch_size=2)
        candidates = tuple(_candidate(index, tokens) for index, tokens in enumerate((6, 6, 5, 4, 4)))

        plan = planner.plan(candidates, step=3)

        self.assertIsNotNone(plan)
        self.assertEqual([packing_bin.pack_tokens for packing_bin in plan.constructors[0].bins], [10, 10])
        self.assertNotIn(SampleKey(0, 2), plan.selected_keys)
        self.assertEqual(
            set(plan.selected_keys),
            {SampleKey(0, 0), SampleKey(0, 1), SampleKey(0, 3), SampleKey(0, 4)},
        )
        self.assertTrue(all(not packing_bin.oversized for packing_bin in plan.constructors[0].bins))

    def test_candidate_input_order_does_not_change_the_plan(self) -> None:
        """Stable sample keys make planning independent of gather order."""
        planner = DynamicPackingPlanner(data_parallel_size=2, seq_len=12, local_batch_size=2)
        candidates = tuple(
            _candidate(
                index,
                tokens,
                cost=WorkloadCost(encoder=encoder_cost, llm=tokens),
                source_rank=index % 2,
            )
            for index, (tokens, encoder_cost) in enumerate(((8, 2), (7, 9), (6, 1), (5, 4), (4, 3), (3, 8)))
        )

        forward = planner.plan(candidates, step=9)
        reversed_plan = planner.plan(tuple(reversed(candidates)), step=9)

        self.assertEqual(forward, reversed_plan)
        self.assertEqual(forward.plan_id, reversed_plan.plan_id)

    def test_oversized_sample_errors_by_default(self) -> None:
        """Overflow requires an explicit singleton policy."""
        planner = DynamicPackingPlanner(data_parallel_size=1, seq_len=10, local_batch_size=1)

        with self.assertRaisesRegex(ValueError, "requires 12 tokens, exceeding seq_len=10"):
            planner.plan((_candidate(0, 12),), step=0)

    def test_single_policy_places_oversized_sample_alone(self) -> None:
        """Explicit overflow never co-packs another sample with the oversized one."""
        planner = DynamicPackingPlanner(
            data_parallel_size=1,
            seq_len=10,
            local_batch_size=1,
            oversized_policy="single",
        )

        plan = planner.plan((_candidate(0, 12), _candidate(1, 2)), step=0)

        self.assertIsNotNone(plan)
        packing_bin = plan.constructors[0].bins[0]
        self.assertTrue(packing_bin.oversized)
        self.assertEqual(packing_bin.pack_tokens, 12)
        self.assertEqual(packing_bin.samples[0].key, SampleKey(0, 0))
        self.assertEqual(len(packing_bin.samples), 1)

    def test_requires_at_least_one_sample_for_every_distributed_bin(self) -> None:
        """A partial distributed yield is reported as unavailable."""
        planner = DynamicPackingPlanner(data_parallel_size=2, seq_len=10, local_batch_size=2)

        self.assertIsNone(planner.plan(tuple(_candidate(index, 1) for index in range(3)), step=0))


if __name__ == "__main__":
    unittest.main()
