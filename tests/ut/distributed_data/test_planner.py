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
"""Unit tests for whole-step local-batch planning."""

from __future__ import annotations

import unittest

from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import LocalBatchMeta, WorkloadCost


def _metadata(local_batch_id: int | str, encoder_cost: float) -> LocalBatchMeta:
    return LocalBatchMeta(
        local_batch_id=local_batch_id,
        cost_hint=WorkloadCost(encoder=encoder_cost),
    )


class TestDistributedBatchPlanner(unittest.TestCase):
    """Validate deterministic local-batch placement."""

    def test_plans_one_local_batch_for_every_execution_slot(self) -> None:
        """Every DP-rank accumulation slot should receive exactly one local batch."""
        planner = DistributedBatchPlanner(data_parallel_size=2, micro_batch_num=2)
        candidates = [_metadata(index, float(4 - index)) for index in range(4)]

        plan = planner.plan(candidates, step=0, local_batch_offset_start=0)

        planned_ids = set()
        for micro_batch_index in range(2):
            for data_rank in range(2):
                planned_ids.add(plan.local_batch_for(data_rank, micro_batch_index).meta.local_batch_id)
        self.assertEqual(planned_ids, {0, 1, 2, 3})
        self.assertEqual((plan.local_batch_offset_start, plan.local_batch_offset_end), (0, 2))

    def test_groups_similar_costs_in_the_same_global_microbatch(self) -> None:
        """Whole-step sorting should reduce synchronized DP straggler gaps."""
        planner = DistributedBatchPlanner(data_parallel_size=2, micro_batch_num=2)
        candidates = [_metadata("heavy-0", 8.0), _metadata("heavy-1", 7.0),
                      _metadata("light-0", 1.0), _metadata("light-1", 1.0)]

        plan = planner.plan(candidates, step=0, local_batch_offset_start=0)

        first_costs = [plan.local_batch_for(rank, 0).cost.encoder for rank in range(2)]
        second_costs = [plan.local_batch_for(rank, 1).cost.encoder for rank in range(2)]
        self.assertEqual(first_costs, [8.0, 7.0])
        self.assertEqual(second_costs, [1.0, 1.0])

    def test_same_metadata_produces_same_plan_id_and_placements(self) -> None:
        """Planning must be byte-stable for checkpoint replay."""
        planner = DistributedBatchPlanner(data_parallel_size=2, micro_batch_num=2)
        candidates = [_metadata(index, float(index + 1)) for index in range(4)]

        first = planner.plan(candidates, step=3, local_batch_offset_start=6)
        second = planner.plan(candidates, step=3, local_batch_offset_start=6)

        self.assertEqual(first, second)
        self.assertEqual(first.plan_id, second.plan_id)

    def test_rejects_incomplete_global_candidate_set(self) -> None:
        """The planner must not silently construct a partial optimizer step."""
        planner = DistributedBatchPlanner(data_parallel_size=2, micro_batch_num=2)

        with self.assertRaisesRegex(ValueError, "requires 4 candidates"):
            planner.plan([_metadata(0, 1.0)], step=0, local_batch_offset_start=0)

    def test_rejects_duplicate_local_batch_ids(self) -> None:
        """Local-batch identifiers must be unique within a planning window."""
        planner = DistributedBatchPlanner(data_parallel_size=1, micro_batch_num=2)

        with self.assertRaisesRegex(ValueError, "unique local_batch_id"):
            planner.plan(
                [_metadata("duplicate", 1.0), _metadata("duplicate", 2.0)],
                step=0,
                local_batch_offset_start=0,
            )

    def test_mixed_integer_and_string_ids_are_deterministic(self) -> None:
        """Planner tie-breaking should support integer and string IDs."""
        planner = DistributedBatchPlanner(data_parallel_size=1, micro_batch_num=2)
        candidates = [LocalBatchMeta(local_batch_id="1"), LocalBatchMeta(local_batch_id=1)]

        plan = planner.plan(candidates, step=0, local_batch_offset_start=0)

        self.assertEqual(
            plan,
            planner.plan(candidates, step=0, local_batch_offset_start=0),
        )
