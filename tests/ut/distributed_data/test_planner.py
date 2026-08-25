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
"""Unit tests for global-step batch planning."""

import unittest

from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import SampleMeta, WorkloadCost


def _metadata(sample_id: str, encoder_cost: float) -> SampleMeta:
    return SampleMeta(
        sample_id=sample_id,
        cost_hint=WorkloadCost(encoder=encoder_cost),
    )


class TestDistributedBatchPlanner(unittest.TestCase):
    """Validate deterministic whole-step planning and balancing."""

    def test_balances_all_microbatches_in_one_optimizer_step(self) -> None:
        """Heavy samples should be spread across both microbatches."""
        planner = DistributedBatchPlanner(data_parallel_size=1, raw_sample_size=2, micro_batch_num=2)
        candidates = [_metadata("0", 8.0), _metadata("1", 7.0), _metadata("2", 1.0), _metadata("3", 1.0)]

        plan = planner.plan(candidates, step=0, cursor_start=0)

        costs = []
        for micro_batch_index in range(2):
            samples = plan.samples_for(0, micro_batch_index)
            costs.append(sum(sample.cost.encoder for sample in samples))
        self.assertEqual(sorted(costs), [8.0, 9.0])
        self.assertEqual({sample.meta.sample_id for sample in plan.samples}, {"0", "1", "2", "3"})

    def test_same_metadata_produces_same_replay_id_and_placements(self) -> None:
        """Planning must be byte-stable for checkpoint replay."""
        planner = DistributedBatchPlanner(data_parallel_size=2, raw_sample_size=1, micro_batch_num=2)
        candidates = [_metadata(str(index), float(index + 1)) for index in range(4)]

        first = planner.plan(candidates, step=3, cursor_start=6)
        second = planner.plan(candidates, step=3, cursor_start=6)

        self.assertEqual(first, second)
        self.assertFalse(hasattr(first, "version"))
        self.assertEqual(first.replay_id, second.replay_id)
        for micro_batch_index in range(2):
            for data_rank in range(2):
                self.assertEqual(len(first.samples_for(data_rank, micro_batch_index)), 1)

    def test_rejects_incomplete_global_candidate_set(self) -> None:
        """A planner must never silently construct a partial optimizer step."""
        planner = DistributedBatchPlanner(data_parallel_size=2, raw_sample_size=1, micro_batch_num=2)
        with self.assertRaisesRegex(ValueError, "requires 4 candidates"):
            planner.plan([_metadata("0", 1.0)], step=0, cursor_start=0)

    def test_rejects_duplicate_sample_ids(self) -> None:
        """The dataset-wide sample identifier must be unique in a planning window."""
        planner = DistributedBatchPlanner(data_parallel_size=1, raw_sample_size=2, micro_batch_num=1)
        candidates = [_metadata("0", 1.0), _metadata("0", 2.0)]

        with self.assertRaisesRegex(ValueError, "unique sample_id"):
            planner.plan(candidates, step=0, cursor_start=0)

    def test_online_plan_balances_only_one_global_microbatch(self) -> None:
        """Online planning should preserve the optimizer microbatch position without later metadata."""
        planner = DistributedBatchPlanner(data_parallel_size=2, raw_sample_size=2, micro_batch_num=3)
        candidates = [_metadata(str(index), float(index + 1)) for index in range(4)]

        plan = planner.plan_microbatch(
            candidates,
            step=5,
            cursor_start=2,
            micro_batch_index=1,
        )

        self.assertEqual(plan.micro_batch_start, 1)
        self.assertEqual(plan.micro_batch_num, 1)
        self.assertEqual((plan.cursor_start, plan.cursor_end), (2, 4))
        for data_rank in range(2):
            self.assertEqual(len(plan.samples_for(data_rank, 1)), 2)
        with self.assertRaisesRegex(ValueError, "micro_batch_index must be in"):
            plan.samples_for(0, 0)

    def test_mixed_integer_and_string_sample_ids_are_deterministic(self) -> None:
        """Planner tie-breaking should support both map-style key types."""
        planner = DistributedBatchPlanner(data_parallel_size=1, raw_sample_size=2, micro_batch_num=1)
        candidates = [
            SampleMeta(sample_id="1"),
            SampleMeta(sample_id=1),
        ]

        plan = planner.plan(candidates, step=0, cursor_start=0)

        self.assertEqual(plan, planner.plan(candidates, step=0, cursor_start=0))
