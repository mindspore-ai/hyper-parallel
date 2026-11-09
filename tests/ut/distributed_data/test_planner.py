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
from collections import Counter
from typing import Optional
from unittest.mock import patch

from hyper_parallel.distributed_data.balancing_algorithm import LPTBalancingAlgorithm
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import (
    SampleKey,
    SampleMetadata,
    WorkloadCost,
)


def _candidate(
        dataset_index: int,
        pack_tokens: int,
        *,
        cost: WorkloadCost = WorkloadCost(),
        reader_rank: int = 0,
        global_sample_position: Optional[int] = None,
) -> SampleMetadata:
    return SampleMetadata(
        pack_tokens=pack_tokens, cost=cost, sample_id=dataset_index,
        key=SampleKey(
            reader_rank, dataset_index, dataset_index if global_sample_position is None else global_sample_position,
        ),
    )


def _reference_bins(candidates, bin_sizes) -> tuple[tuple[SampleKey, ...], ...]:
    """Return the original grouping already selected by the producer."""
    ordered = tuple(sorted(candidates, key=lambda item: item.key.global_sample_position))
    bins = []
    offset = 0
    for size in bin_sizes:
        bins.append(tuple(item.key for item in ordered[offset:offset + size]))
        offset += size
    return tuple(bins)


class TestDynamicPackingPlanner(unittest.TestCase):
    """Verify exact-set sample balancing, fallback, overflow, and stability."""

    def test_explicit_target_slots_preserve_round_costs_with_multiple_bins(self) -> None:
        """Feature: Explicit window destinations independent of physical DP size.
        Description: Plan two DP2 rounds with two bins per destination, then one ordinary round.
        Expectation: Window cost is 9 + 8, and the next call still uses two physical ranks.
        """
        costs = (8, 1, 1, 1, 7, 1, 1, 1)
        samples = tuple(_candidate(index, 1, cost=WorkloadCost(llm=cost)) for index, cost in enumerate(costs))
        reference = tuple((item.key,) for item in samples)
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=2, seq_len=1, cost_model=lambda item: item.cost,
        )
        with patch.object(planner.balancing_algorithm, "assign", return_value=reference) as assign:
            plan = planner.plan(samples, reference_bins=reference, step=6, target_slot_count=4)
        self.assertEqual(assign.call_args.kwargs["data_parallel_size"], 4)
        self.assertEqual(assign.call_args.kwargs["objective_group_size"], 2)
        self.assertEqual([len(batch) for batch in plan.local_batches], [2, 2, 2, 2])
        self.assertEqual([cost.llm for cost in plan.rank_costs], [9, 2, 8, 2])
        self.assertEqual(planner.last_balance_decision["accepted_objective"], 17)
        self.assertEqual(plan.selected_keys, tuple(item.key for item in samples))
        self.assertEqual(planner.data_parallel_size, 2)
        self.assertEqual(planner.distributed_bin_count, 4)
        self.assertEqual(planner.distributed_token_budget, 4)
        with patch.object(planner.balancing_algorithm, "assign", return_value=reference[:4]):
            plan = planner.plan(samples[:4], reference_bins=reference[:4], step=8)
        self.assertEqual([len(batch) for batch in plan.local_batches], [2, 2])
        self.assertEqual(planner.last_balance_decision["accepted_objective"], 9)

    def test_target_slots_require_complete_nonempty_rounds(self) -> None:
        """Feature: Target-slot validation.
        Description: Supply invalid slot counts or a reference that does not fill the requested slots.
        Expectation: Invalid counts fail before cost estimation, and incomplete references are rejected.
        """
        samples = tuple(_candidate(index, 1) for index in range(2))
        reference = tuple((item.key,) for item in samples)
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=1, seq_len=1, cost_model=lambda item: item.cost,
        )
        with patch.object(planner, "cost_model") as estimate:
            for count in (True, False, 0, -2, 1, 3, 2.0, "4"):
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, "target_slot_count"):
                    planner.plan(samples, reference_bins=reference, step=0, target_slot_count=count)
            estimate.assert_not_called()
        with self.assertRaisesRegex(ValueError, "target_slot_count.*non-empty bins"):
            planner.plan(samples, reference_bins=reference, step=0, target_slot_count=4)

    def test_changing_window_sizes_and_assignment_failure_preserve_dp_size(self) -> None:
        """Feature: Planner reuse across differently sized windows.
        Description: Plan three, two, and one DP4 rounds after an assignment exception.
        Expectation: Each result has the correct round count, summed makespan, membership, and step indices.
        """
        costs = (12, 1, 2, 3, 11, 4, 5, 6, 10, 7, 8, 9)
        samples = tuple(_candidate(index, 1, cost=WorkloadCost(llm=cost)) for index, cost in enumerate(costs))
        reference = tuple((item.key,) for item in samples)
        planner = DynamicPackingPlanner(
            data_parallel_size=4, local_batch_size=1, seq_len=1, cost_model=lambda item: item.cost,
        )
        with patch.object(planner.balancing_algorithm, "assign", side_effect=RuntimeError("assignment failed")):
            with self.assertRaisesRegex(RuntimeError, "assignment failed"):
                planner.plan_window(samples, reference_bins=reference, step=0)
        for rounds, expected_cost in ((3, 24), (2, 16), (1, 12)):
            count = rounds * 4
            with self.subTest(rounds=rounds):
                plans = planner.plan_window(samples[:count], reference_bins=reference[:count], step=6)
                self.assertEqual([plan.step for plan in plans], list(range(6, 6 + rounds)))
                self.assertEqual([plan.data_parallel_size for plan in plans], [4] * rounds)
                self.assertEqual(Counter(key for plan in plans for key in plan.selected_keys),
                                 Counter(item.key for item in samples[:count]))
                self.assertEqual(planner.last_balance_decision["accepted_objective"], expected_cost)
                self.assertEqual(set(planner.last_sample_costs), {item.key for item in samples[:count]})
                self.assertEqual(planner.data_parallel_size, 4)
                self.assertEqual(planner.distributed_bin_count, 4)

    def test_schedule_intact_packs_after_greedy_fallback(self) -> None:
        """Feature: Intact-pack scheduling inside accumulation windows.
        Description: Keep twelve packs intact after assignment falls back to the reference.
        Expectation: Three DP4 rounds cost 24 instead of 33, with deterministic membership and step indices.
        """
        costs = (12, 1, 2, 3, 11, 4, 5, 6, 10, 7, 8, 9)
        samples = tuple(_candidate(index, 1, cost=WorkloadCost(llm=cost)) for index, cost in enumerate(costs))
        reference = tuple((item.key,) for item in samples)
        planner = DynamicPackingPlanner(
            data_parallel_size=4, local_batch_size=1, seq_len=1, cost_model=lambda item: item.cost,
        )
        with patch.object(planner.balancing_algorithm, "assign", return_value=reference):
            plans = planner.plan_window(samples, reference_bins=reference, step=7)
            self.assertEqual([plan.step for plan in plans], [7, 8, 9])
            self.assertEqual([[cost.llm for cost in plan.rank_costs] for plan in plans],
                             [[12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]])
            self.assertEqual(planner.last_balance_decision["original_objective"], 33)
            self.assertEqual(planner.last_balance_decision["accepted_objective"], 24)
            self.assertEqual(Counter(key for plan in plans for key in plan.selected_keys),
                             Counter(item.key for item in samples))
            self.assertEqual(plans, planner.plan_window(tuple(reversed(samples)), reference_bins=reference, step=7))
            planner.min_balance_gain = 0.3
            rejected = planner.plan_window(samples, reference_bins=reference, step=7)
            self.assertEqual(
                tuple(batch[0].sample_keys for plan in rejected for batch in plan.local_batches), reference,
            )
            self.assertFalse(planner.last_balance_decision["accepted"])

    def test_window_scheduling_respects_scope_and_equal_cost(self) -> None:
        """Feature: Scheduling scope and benefit gate.
        Description: Use a single round, multiple local bins, or identical-cost packs.
        Expectation: Unsupported cases skip scheduling and zero-gain windows retain the original order.
        """
        samples = tuple(_candidate(index, 1, cost=WorkloadCost(llm=1)) for index in range(4))
        reference = tuple((item.key,) for item in reversed(samples))
        for dp_size, local_size in ((4, 1), (1, 2)):
            planner = DynamicPackingPlanner(
                data_parallel_size=dp_size, local_batch_size=local_size, seq_len=1,
                cost_model=lambda item: item.cost,
            )
            with self.subTest(dp=dp_size, local=local_size), patch.object(
                    planner.balancing_algorithm, "assign", return_value=reference,
            ), patch.object(planner, "_schedule_window_bins", side_effect=AssertionError("Unexpected scheduling")):
                plans = planner.plan_window(samples, reference_bins=reference, step=0)
                self.assertEqual(tuple(bin_plan.sample_keys for plan in plans for batch in plan.local_batches
                                       for bin_plan in batch), reference)
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=1, seq_len=1, cost_model=lambda item: item.cost,
        )
        plans = planner.plan_window(samples, reference_bins=reference, step=0)
        self.assertFalse(planner.last_balance_decision["accepted"])
        self.assertEqual(tuple(batch[0].sample_keys for plan in plans for batch in plan.local_batches), reference)

    def test_window_scheduling_keeps_better_unscheduled_balance_candidate(self) -> None:
        """Feature: Supported variance-objective compatibility.
        Description: LPT produces costs 200/2/3/3 where descending sorting worsens normalized variance.
        Expectation: The unscheduled candidate still clears the ten-percent gate against 203/2/1/2.
        """
        samples = tuple(_candidate(index, tokens, cost=WorkloadCost(llm=cost)) for index, (tokens, cost) in enumerate(
            ((1, 200), (4, 2), *((1, 1),) * 6),
        ))
        reference = tuple(
            tuple(samples[index].key for index in indices) for indices in ((0, 2, 3, 4), (1,), (5,), (6, 7))
        )
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=1, seq_len=4, cost_model=lambda item: item.cost,
            balancing_algorithm=LPTBalancingAlgorithm(objective="balance"), min_balance_gain=0.1,
        )
        plans = planner.plan_window(samples, reference_bins=reference, step=0)
        self.assertEqual([[cost.llm for cost in plan.rank_costs] for plan in plans], [[200, 2], [3, 3]])
        self.assertTrue(planner.last_balance_decision["accepted"])
        self.assertGreater(planner.last_balance_decision["relative_gain"], 0.1)

    def test_scheduling_does_not_hide_invalid_subclass_assignments(self) -> None:
        """Feature: Validation before scheduling.
        Description: A custom LPT subclass loses samples while retaining the expected bin count.
        Expectation: Membership validation rejects it even when a schedulable reference is available.
        """
        class InvalidLPT(LPTBalancingAlgorithm):
            """Return malformed candidates to exercise mandatory subclass auditing."""

            def assign(self, samples: tuple, **kwargs: object) -> tuple:
                """Repeat one occurrence instead of conserving the selected set."""
                del kwargs
                return ((samples[0].key,),) * len(samples)

        samples = tuple(_candidate(index, 1, cost=WorkloadCost(llm=index + 1)) for index in range(4))
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=1, seq_len=1, cost_model=lambda item: item.cost,
            balancing_algorithm=InvalidLPT(), validate=False,
        )
        with self.assertRaisesRegex(ValueError, "conserve every selected sample key"):
            planner.plan_window(samples, reference_bins=tuple((item.key,) for item in samples), step=0)

    def test_custom_non_lpt_policy_keeps_its_window_order(self) -> None:
        """Feature: Custom assignment ownership.
        Description: Supply a non-LPT strategy whose intact packs could benefit from sorting.
        Expectation: The planner respects its order and does not add LPT-specific scheduling.
        """
        class ReferencePolicy:
            """Keep original sample grouping with a makespan objective."""

            @staticmethod
            def assign(samples: tuple, *, reference_bins: tuple, **kwargs: object) -> tuple:
                """Return the selected reference without reordering it."""
                del samples, kwargs
                return reference_bins

            @staticmethod
            def objective(costs: tuple) -> float:
                """Score a round by its largest backbone cost."""
                return max(cost.llm for cost in costs)

        samples = tuple(_candidate(index, 1, cost=WorkloadCost(llm=cost))
                        for index, cost in enumerate((9, 1, 8, 2)))
        reference = tuple((item.key,) for item in samples)
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=1, seq_len=1, cost_model=lambda item: item.cost,
            balancing_algorithm=ReferencePolicy(),
        )
        plans = planner.plan_window(samples, reference_bins=reference, step=0)
        self.assertEqual(tuple(batch[0].sample_keys for plan in plans for batch in plan.local_batches), reference)
        self.assertEqual(planner.last_balance_decision["accepted_objective"], 17)

    def test_planning_requires_bound_sample_keys(self) -> None:
        """Feature: Sample occurrence binding.
        Description: Construct an invalid key and pass unbound metadata directly to the planner.
        Expectation: Invalid or missing keys fail before workload estimation.
        """
        with self.assertRaisesRegex(ValueError, "SampleMetadata.key"):
            SampleMetadata(4, key="invalid")
        planner = DynamicPackingPlanner(
            data_parallel_size=1, seq_len=8, local_batch_size=1, cost_model=lambda metadata: metadata.cost,
        )
        with patch.object(planner, "cost_model") as estimate:
            with self.assertRaisesRegex(ValueError, "Planner samples must have a SampleKey"):
                planner.plan((SampleMetadata(4),), reference_bins=((SampleKey(0, 0),),), step=0)
            estimate.assert_not_called()

    def test_repeated_dataset_indices_remain_distinct_occurrences(self) -> None:
        """Feature: Sample occurrence identity.
        Description: Plan two occurrences of the same Dataset index.
        Expectation: Both SampleKey values survive as independently selected samples.
        """
        samples = tuple(
            SampleMetadata(2, key=SampleKey(0, 7, position))
            for position in range(2)
        )
        planner = DynamicPackingPlanner(
            cost_model=lambda metadata: metadata.cost,
            data_parallel_size=2, seq_len=8, local_batch_size=1,
        )
        plan = planner.plan(samples, reference_bins=_reference_bins(samples, (1, 1)), step=0)
        self.assertEqual(set(plan.selected_keys), {sample.key for sample in samples})
        self.assertEqual(len(plan.selected_keys), 2)

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
                    [sample.cost for sample in samples],
                    [WorkloadCost(encoder=index + 1) for index in range(len(samples))],
                )

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
                self.assertEqual(planner.last_sample_costs, {sample.key: sample.cost for sample in samples})
                if costs[0] == 1:
                    self.assertEqual(
                        tuple(batch[0].sample_keys for batch in plan.local_batches), reference_bins,
                    )

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

    def test_candidate_gather_order_does_not_change_the_plan(self) -> None:
        """Feature: Deterministic distributed planning.
        Description: Plan candidates in forward and reverse gathered order.
        Expectation: Canonical stream positions produce identical assignments and costs.
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
