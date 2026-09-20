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
"""Tests for balanced placement of externally selected steps."""

import unittest
from typing import Optional
from unittest.mock import patch

from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    SampleKey,
    SampleMetadata,
    WorkloadCost,
)
from tests.common.mark_utils import arg_mark


def _candidate(
        dataset_index: int,
        pack_tokens: int,
        *,
        cost: WorkloadCost = WorkloadCost(),
        reader_rank: int = 0,
        global_sample_position: Optional[int] = None,
) -> BufferedSampleMetadata:
    return BufferedSampleMetadata(
        key=SampleKey(reader_rank, dataset_index),
        metadata=SampleMetadata(pack_tokens=pack_tokens, cost=cost, sample_id=dataset_index),
        global_sample_position=(
            dataset_index if global_sample_position is None else global_sample_position
        ),
    )


def _reference_bins(candidates, bin_sizes) -> tuple[tuple[SampleKey, ...], ...]:
    """Return the original grouping already selected by the producer."""
    ordered = tuple(sorted(candidates, key=lambda item: item.global_sample_position))
    bins = []
    offset = 0
    for size in bin_sizes:
        bins.append(tuple(item.key for item in ordered[offset:offset + size]))
        offset += size
    return tuple(bins)


class TestDynamicPackingPlanner(unittest.TestCase):
    """Verify exact-set sample balancing, fallback, overflow, and stability."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_repeated_dataset_indices_remain_distinct_occurrences(self) -> None:
        """Feature: Sample occurrence identity.
        Description: Plan two occurrences of the same Dataset index.
        Expectation: Both SampleKey values survive as independently selected samples.
        """
        samples = tuple(
            BufferedSampleMetadata(SampleKey(0, 7, position), SampleMetadata(2), position)
            for position in range(2)
        )
        planner = DynamicPackingPlanner(
            cost_model=lambda metadata: metadata.cost,
            data_parallel_size=2, seq_len=8, local_batch_size=1,
        )
        plan = planner.plan(samples, reference_bins=_reference_bins(samples, (1, 1)), step=0)
        self.assertEqual(set(plan.selected_keys), {sample.key for sample in samples})
        self.assertEqual(len(plan.selected_keys), 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_cost_estimation_preserves_input_metadata_and_fallback(self) -> None:
        """Feature: Cost estimation without a selection wrapper.
        Description: Estimate costs on a step whose greedy packing reaches a dead end.
        Expectation: Reference grouping survives and Reader-owned metadata is unchanged.
        """
        samples = tuple(
            _candidate(index, tokens, cost=WorkloadCost(encoder=index + 1))
            for index, tokens in enumerate((6, 2, 2, 5, 3, 2))
        )
        reference_bins = _reference_bins(samples, (3, 3))
        for validate in (False, True):
            with self.subTest(validate=validate):
                planner = DynamicPackingPlanner(
                    data_parallel_size=1, seq_len=10, local_batch_size=2,
                    cost_model=lambda _metadata: WorkloadCost(llm=1), validate=validate,
                )
                plan = planner.plan(samples, reference_bins=reference_bins, step=0)
                self.assertEqual(
                    tuple(packing_bin.sample_keys for packing_bin in plan.local_batches[0]), reference_bins,
                )
                self.assertEqual(plan.rank_costs, (WorkloadCost(llm=6),))
                self.assertEqual(planner.last_sample_costs, {sample.key: WorkloadCost(llm=1) for sample in samples})
                self.assertEqual(
                    [sample.metadata.cost for sample in samples],
                    [WorkloadCost(encoder=index + 1) for index in range(len(samples))],
                )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_minimum_gain_uses_original_rank_grouping(self) -> None:
        """Feature: Native minimum balancing gain.
        Description: Require more improvement than the balanced placement provides.
        Expectation: Original bins and their costs are retained in rank order.
        """
        samples = tuple(
            _candidate(index, 5, cost=WorkloadCost(llm=cost), reader_rank=index // 2)
            for index, cost in enumerate((9, 9, 1, 1))
        )
        reference_bins = _reference_bins(samples, (2, 2))
        planner = DynamicPackingPlanner(
            cost_model=lambda metadata: metadata.cost,
            data_parallel_size=2, seq_len=10, local_batch_size=1, min_balance_gain=0.6,
        )
        plan = planner.plan(samples, reference_bins=reference_bins, step=0)
        self.assertEqual(tuple(batch[0].sample_keys for batch in plan.local_batches), reference_bins)
        self.assertEqual([cost.llm for cost in plan.rank_costs], [18, 2])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_capacity_checks_remain_active_for_trusted_inputs(self) -> None:
        """Feature: Mandatory packing capacity.
        Description: Supply two individually valid samples that overflow their only bin.
        Expectation: Capacity enforcement remains active with metadata auditing disabled.
        """
        samples = (_candidate(0, 3), _candidate(1, 3))
        for validate in (False, True):
            with self.subTest(validate=validate):
                planner = DynamicPackingPlanner(
                    cost_model=lambda metadata: metadata.cost,
                    data_parallel_size=1, seq_len=5, local_batch_size=1, validate=validate,
                )
                with self.assertRaisesRegex(ValueError, "cannot admit sample"):
                    planner.plan(samples, reference_bins=_reference_bins(samples, (2,)), step=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_enabled_cost_first_balancing_keeps_nonimproving_reference(self) -> None:
        """Feature: Cost-first LPT planning.
        Description: Plan skewed and equal-cost selected sample sets.
        Expectation: Makespan improves when possible, otherwise reference ownership is retained.
        """
        planner = DynamicPackingPlanner(
            data_parallel_size=2, seq_len=10, local_batch_size=1,
            cost_model=lambda metadata: metadata.cost,
        )
        for costs, expected_costs in (((9, 9, 1, 1), [10, 10]), ((1, 1, 1, 1), [2, 2])):
            with self.subTest(costs=costs):
                samples = tuple(
                    _candidate(index, 5, cost=WorkloadCost(llm=cost), reader_rank=index // 2)
                    for index, cost in enumerate(costs)
                )
                reference_bins = _reference_bins(samples, (2, 2))
                plan = planner.plan(samples, reference_bins=reference_bins, step=0)
                self.assertEqual([cost.llm for cost in plan.rank_costs], expected_costs)
                self.assertEqual(set(plan.selected_keys), {sample.key for sample in samples})
                self.assertEqual(planner.last_sample_costs, {sample.key: sample.metadata.cost for sample in samples})
                if costs[0] == 1:
                    self.assertEqual(
                        tuple(batch[0].sample_keys for batch in plan.local_batches), reference_bins,
                    )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_balances_individual_samples_and_conserves_selection(self) -> None:
        """Feature: Sample-level load balancing.
        Description: Place complementary selected samples across data-parallel bins.
        Expectation: Every selected sample is conserved and every bin is full.
        """
        planner = DynamicPackingPlanner(
            cost_model=lambda metadata: metadata.cost,
            data_parallel_size=2, seq_len=10, local_batch_size=1,
        )
        candidates = tuple(_candidate(index, tokens) for index, tokens in enumerate((7, 3, 7, 3)))
        reference_bins = _reference_bins(candidates, (2, 2))

        plan = planner.plan(candidates, reference_bins=reference_bins, step=0)

        self.assertEqual(set(plan.selected_keys), {candidate.key for candidate in candidates})
        for local_batch in plan.local_batches:
            packing_bin = local_batch[0]
            self.assertEqual(packing_bin.pack_tokens, 10)
            self.assertEqual(len(packing_bin.sample_keys), 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_workload_cost_changes_placement_not_step_membership(self) -> None:
        """Feature: Workload-aware sample placement.
        Description: Change sample costs while preserving packing footprints.
        Expectation: Cost changes placement without changing step membership.
        """
        planner = DynamicPackingPlanner(
            cost_model=lambda metadata: metadata.cost,
            data_parallel_size=2, seq_len=10, local_batch_size=1,
        )
        first_costs = tuple(
            _candidate(index, 5, cost=WorkloadCost(llm=9 if index < 2 else 1))
            for index in range(4)
        )
        second_costs = tuple(
            _candidate(index, 5, cost=WorkloadCost(llm=9 if index % 2 == 0 else 1))
            for index in range(4)
        )

        reference_bins = _reference_bins(first_costs, (2, 2))
        first_plan = planner.plan(first_costs, reference_bins=reference_bins, step=0)
        second_plan = planner.plan(second_costs, reference_bins=reference_bins, step=0)

        self.assertEqual(set(first_plan.selected_keys), set(second_plan.selected_keys))
        self.assertNotEqual(first_plan.local_sample_keys(0), second_plan.local_sample_keys(0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_falls_back_to_reference_packing_without_dropping_samples(self) -> None:
        """Feature: Reference-packing fallback.
        Description: Drive workload-aware greedy placement into a dead end.
        Expectation: The known-feasible canonical grouping preserves all samples.
        """
        planner = DynamicPackingPlanner(
            cost_model=lambda metadata: metadata.cost,
            data_parallel_size=1, seq_len=10, local_batch_size=2,
        )
        candidates = tuple(_candidate(index, tokens) for index, tokens in enumerate((6, 2, 2, 5, 3, 2)))
        reference_bins = _reference_bins(candidates, (3, 3))

        plan = planner.plan(candidates, reference_bins=reference_bins, step=3)

        self.assertEqual(set(plan.selected_keys), {candidate.key for candidate in candidates})
        self.assertEqual([packing_bin.pack_tokens for packing_bin in plan.local_batches[0]], [10, 10])
        self.assertEqual(
            tuple(tuple(key.dataset_index for key in packing_bin.sample_keys)
                  for packing_bin in plan.local_batches[0]),
            ((0, 1, 2), (3, 4, 5)),
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_rejects_a_balanced_plan_that_does_not_conserve_selection(self) -> None:
        """Feature: Plan conservation validation.
        Description: Inject a balanced plan that omits selected keys.
        Expectation: The final invariant rejects the incomplete plan.
        """
        planner = DynamicPackingPlanner(
            cost_model=lambda metadata: metadata.cost,
            data_parallel_size=1, seq_len=10, local_batch_size=1,
        )
        candidates = (_candidate(0, 5), _candidate(1, 5))
        reference_bins = _reference_bins(candidates, (2,))

        with (
                patch.object(planner.balancing_algorithm, "assign", return_value=((candidates[0].key,),)),
                self.assertRaisesRegex(ValueError, "conserve every selected sample key exactly once"),
        ):
            planner.plan(candidates, reference_bins=reference_bins, step=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_candidate_gather_order_does_not_change_the_plan(self) -> None:
        """Feature: Deterministic distributed planning.
        Description: Plan candidates in forward and reverse gathered order.
        Expectation: Canonical stream positions produce the same plan and ID.
        """
        planner = DynamicPackingPlanner(
            cost_model=lambda metadata: metadata.cost,
            data_parallel_size=2, seq_len=12, local_batch_size=2,
        )
        candidates = tuple(
            _candidate(
                index,
                tokens,
                cost=WorkloadCost(encoder=encoder_cost, llm=tokens),
                reader_rank=index % 2,
            )
            for index, (tokens, encoder_cost) in enumerate(
                ((8, 2), (4, 9), (6, 1), (6, 4), (4, 3), (8, 8), (5, 6), (7, 7))
            )
        )

        reference_bins = _reference_bins(candidates, (2, 2, 2, 2))
        forward = planner.plan(candidates, reference_bins=reference_bins, step=9)
        reversed_plan = planner.plan(tuple(reversed(candidates)), reference_bins=reference_bins, step=9)

        self.assertEqual(forward, reversed_plan)
        self.assertEqual(forward.plan_id, reversed_plan.plan_id)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_oversized_sample_errors_by_default(self) -> None:
        """Feature: Oversized sample policy.
        Description: Select a sample larger than the configured sequence length.
        Expectation: The default policy rejects overflow during planning.
        """
        planner = DynamicPackingPlanner(
            cost_model=lambda metadata: metadata.cost,
            data_parallel_size=1, seq_len=10, local_batch_size=1,
        )
        samples = (_candidate(0, 12),)
        with self.assertRaisesRegex(ValueError, "requires 12 tokens, exceeding seq_len=10"):
            planner.plan(samples, reference_bins=_reference_bins(samples, (1,)), step=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_single_policy_places_oversized_sample_alone(self) -> None:
        """Feature: Singleton oversized sample policy.
        Description: Enable explicit singleton placement for an oversized sample.
        Expectation: No other sample is packed with the oversized sample.
        """
        planner = DynamicPackingPlanner(
            cost_model=lambda metadata: metadata.cost,
            data_parallel_size=1,
            seq_len=10,
            local_batch_size=1,
            oversized_policy="single",
        )
        samples = (_candidate(0, 12),)

        plan = planner.plan(samples, reference_bins=_reference_bins(samples, (1,)), step=0)

        packing_bin = plan.local_batches[0][0]
        self.assertTrue(packing_bin.oversized)
        self.assertEqual(packing_bin.pack_tokens, 12)
        self.assertEqual(packing_bin.sample_keys[0], SampleKey(0, 0))
        self.assertEqual(len(packing_bin.sample_keys), 1)


if __name__ == "__main__":
    unittest.main()
