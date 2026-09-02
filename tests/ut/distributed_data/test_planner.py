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
"""Tests for step sample selection and exact-set balanced placement."""

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
from hyper_parallel.distributed_data.step_sample_selection import StepSampleSelector
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


def _select(
        candidates: tuple[BufferedSampleMetadata, ...],
        *,
        seq_len: int,
        bin_count: int,
        end_of_stream: bool = False,
        oversized_policy: str = "error",
):
    selector = StepSampleSelector(
        seq_len=seq_len,
        distributed_bin_count=bin_count,
        oversized_policy=oversized_policy,
    )
    return selector.select(candidates, end_of_stream=end_of_stream)


class TestStepSampleSelector(unittest.TestCase):
    """Verify that canonical step membership is independent of lookahead."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_selects_only_the_canonical_stream_prefix(self) -> None:
        """Feature: Canonical stream-prefix selection.
        Description: Include a future complementary sample in the candidate window.
        Expectation: The future sample cannot replace the current prefix.
        """
        candidates = tuple(_candidate(index, tokens) for index, tokens in enumerate((6, 6, 4, 4)))

        selection = _select(candidates, seq_len=10, bin_count=2)

        self.assertIsNotNone(selection)
        self.assertEqual(tuple(item.key.dataset_index for item in selection.samples), (0, 1, 2))
        self.assertEqual(selection.reference_bins, ((SampleKey(0, 0),), (SampleKey(0, 1), SampleKey(0, 2))))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_extra_lookahead_does_not_change_step_membership(self) -> None:
        """Feature: Lookahead-independent selection.
        Description: Select from short and extended candidate windows.
        Expectation: The same canonical prefix produces the same sample-ID set.
        """
        candidates = tuple(_candidate(index, tokens) for index, tokens in enumerate((6, 6, 4, 4, 2, 8)))

        short = _select(candidates[:4], seq_len=10, bin_count=2)
        long = _select(tuple(reversed(candidates)), seq_len=10, bin_count=2)

        self.assertEqual(short, long)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_waits_for_end_of_stream_before_accepting_partial_last_bin(self) -> None:
        """Feature: Partial final-bin selection.
        Description: Select an underfilled bin before and after end of stream.
        Expectation: The bin freezes only when no later sample can arrive.
        """
        candidates = (_candidate(0, 4),)

        self.assertIsNone(_select(candidates, seq_len=10, bin_count=1, end_of_stream=False))
        selection = _select(candidates, seq_len=10, bin_count=1, end_of_stream=True)
        self.assertEqual(tuple(item.key for item in selection.samples), (SampleKey(0, 0),))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_merges_strided_readers_by_global_sample_position(self) -> None:
        """Feature: Multi-reader stream reconstruction.
        Description: Merge candidates using their global sample positions.
        Expectation: Reader gather order does not alter the shared epoch stream.
        """
        candidates = (
            _candidate(2, 5, reader_rank=4, global_sample_position=1),
            _candidate(1, 5, reader_rank=0, global_sample_position=0),
        )

        selection = _select(candidates, seq_len=10, bin_count=1)

        self.assertEqual(tuple(item.key for item in selection.samples), (SampleKey(0, 1), SampleKey(4, 2)))


class TestDynamicPackingPlanner(unittest.TestCase):
    """Verify exact-set sample balancing, fallback, overflow, and stability."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_balances_individual_samples_and_conserves_selection(self) -> None:
        """Feature: Sample-level load balancing.
        Description: Place complementary selected samples across data-parallel bins.
        Expectation: Every selected sample is conserved and every bin is full.
        """
        planner = DynamicPackingPlanner(data_parallel_size=2, seq_len=10, local_batch_size=1)
        candidates = tuple(_candidate(index, tokens) for index, tokens in enumerate((7, 3, 7, 3)))
        selection = _select(candidates, seq_len=10, bin_count=2)

        plan = planner.plan(selection, step=0)

        self.assertEqual(set(plan.selected_keys), {candidate.key for candidate in candidates})
        for constructor in plan.constructors:
            packing_bin = constructor.bins[0]
            self.assertEqual(packing_bin.pack_tokens, 10)
            self.assertEqual(len(packing_bin.samples), 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_workload_cost_changes_placement_not_step_membership(self) -> None:
        """Feature: Workload-aware sample placement.
        Description: Change sample costs while preserving packing footprints.
        Expectation: Cost changes placement without changing step membership.
        """
        planner = DynamicPackingPlanner(data_parallel_size=2, seq_len=10, local_batch_size=1)
        first_costs = tuple(
            _candidate(index, 5, cost=WorkloadCost(encoder=100 if index == 0 else 1))
            for index in range(4)
        )
        second_costs = tuple(
            _candidate(index, 5, cost=WorkloadCost(encoder=100 if index == 1 else 1))
            for index in range(4)
        )

        first_selection = _select(first_costs, seq_len=10, bin_count=2)
        second_selection = _select(second_costs, seq_len=10, bin_count=2)
        first_plan = planner.plan(first_selection, step=0)
        second_plan = planner.plan(second_selection, step=0)

        self.assertEqual(set(first_plan.selected_keys), set(second_plan.selected_keys))
        self.assertNotEqual(first_plan.constructor_for(0).sample_keys, second_plan.constructor_for(0).sample_keys)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_falls_back_to_reference_packing_without_dropping_samples(self) -> None:
        """Feature: Reference-packing fallback.
        Description: Drive workload-aware greedy placement into a dead end.
        Expectation: The known-feasible canonical grouping preserves all samples.
        """
        planner = DynamicPackingPlanner(data_parallel_size=1, seq_len=10, local_batch_size=2)
        candidates = tuple(_candidate(index, tokens) for index, tokens in enumerate((6, 2, 2, 5, 3, 2)))
        selection = _select(candidates, seq_len=10, bin_count=2)

        plan = planner.plan(selection, step=3)

        self.assertEqual(set(plan.selected_keys), {candidate.key for candidate in candidates})
        self.assertEqual([packing_bin.pack_tokens for packing_bin in plan.constructors[0].bins], [10, 10])
        self.assertEqual(
            tuple(tuple(sample.key.dataset_index for sample in packing_bin.samples)
                  for packing_bin in plan.constructors[0].bins),
            ((0, 1, 2), (3, 4, 5)),
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_rejects_a_balanced_plan_that_does_not_conserve_selection(self) -> None:
        """Feature: Plan conservation validation.
        Description: Inject a balanced plan that omits selected keys.
        Expectation: The final invariant rejects the incomplete plan.
        """
        planner = DynamicPackingPlanner(data_parallel_size=1, seq_len=10, local_batch_size=1)
        candidates = (_candidate(0, 5), _candidate(1, 5))
        selection = _select(candidates, seq_len=10, bin_count=1)

        with (
                patch.object(planner, "_freeze_bins", return_value=()),
                self.assertRaisesRegex(ValueError, "conserve the frozen step sample set exactly"),
        ):
            planner.plan(selection, step=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_candidate_gather_order_does_not_change_the_plan(self) -> None:
        """Feature: Deterministic distributed planning.
        Description: Plan candidates in forward and reverse gathered order.
        Expectation: Canonical stream positions produce the same plan and ID.
        """
        planner = DynamicPackingPlanner(data_parallel_size=2, seq_len=12, local_batch_size=2)
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

        forward_selection = _select(candidates, seq_len=12, bin_count=4)
        reverse_selection = _select(tuple(reversed(candidates)), seq_len=12, bin_count=4)
        forward = planner.plan(forward_selection, step=9)
        reversed_plan = planner.plan(reverse_selection, step=9)

        self.assertEqual(forward, reversed_plan)
        self.assertEqual(forward.plan_id, reversed_plan.plan_id)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_oversized_sample_errors_by_default(self) -> None:
        """Feature: Oversized sample policy.
        Description: Select a sample larger than the configured sequence length.
        Expectation: The default policy rejects overflow during step selection.
        """
        with self.assertRaisesRegex(ValueError, "requires 12 tokens, exceeding seq_len=10"):
            _select((_candidate(0, 12),), seq_len=10, bin_count=1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_single_policy_places_oversized_sample_alone(self) -> None:
        """Feature: Singleton oversized sample policy.
        Description: Enable explicit singleton placement for an oversized sample.
        Expectation: No other sample is packed with the oversized sample.
        """
        planner = DynamicPackingPlanner(
            data_parallel_size=1,
            seq_len=10,
            local_batch_size=1,
            oversized_policy="single",
        )
        selection = _select(
            (_candidate(0, 12), _candidate(1, 2)),
            seq_len=10,
            bin_count=1,
            oversized_policy="single",
        )

        plan = planner.plan(selection, step=0)

        packing_bin = plan.constructors[0].bins[0]
        self.assertTrue(packing_bin.oversized)
        self.assertEqual(packing_bin.pack_tokens, 12)
        self.assertEqual(packing_bin.samples[0].key, SampleKey(0, 0))
        self.assertEqual(len(packing_bin.samples), 1)


if __name__ == "__main__":
    unittest.main()
