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
"""Window-aware greedy placement and bounded feasible packing refinements."""

import random
import unittest
from collections import Counter
from unittest.mock import patch

from hyper_parallel.distributed_data import (
    CausalAttentionCostModel, FeasibleLPTBalancingAlgorithm, LPTBalancingAlgorithm,
)
from hyper_parallel.distributed_data.balancing_algorithm import (
    _AssignmentBin, _WindowLPT, resolve_balancing_algorithm,
)
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import PackingConstraints, SampleKey, SampleMetadata, WorkloadCost


def _full_pair(scale: int = 1) -> tuple:
    samples = tuple(
        SampleMetadata(tokens * scale, key=SampleKey(0, 0, index), cost=WorkloadCost(llm=cost),
                       packing_costs={"encoder": 1})
        for index, (tokens, cost) in enumerate(zip((6, 2, 2, 5, 3, 2), (90, 70, 50, 100, 80, 60)))
    )
    return samples, (tuple(item.key for item in samples[:3]), tuple(item.key for item in samples[3:]))


def _random_full_bins(count: int, seed: int, capacity: int = 64) -> tuple:
    """Generate deterministic full-capacity bins with per-document quadratic costs."""
    generator = random.Random(seed)
    samples, bins = [], []
    for _ in range(count):
        cuts = [0, *sorted(generator.sample(range(1, capacity), 5)), capacity]
        keys = []
        for start, end in zip(cuts, cuts[1:]):
            length = end - start
            key = SampleKey(0, 0, len(samples))
            samples.append(SampleMetadata(length, key=key, cost=WorkloadCost(llm=length ** 2)))
            keys.append(key)
        bins.append(tuple(keys))
    return tuple(samples), tuple(bins)


class TestFeasibleLPT(unittest.TestCase):
    """Check useful repairs and bounded fallback without accelerator hardware."""

    def test_padding_aware_windows_preserve_capacity_budget_and_scoring(self) -> None:
        """Feature: Padding-aware window planning.
        Description: Refine three-round windows at multiple DP sizes and repair budgets.
        Expectation: Capacity, occurrences, work budget and physical PAD costs remain consistent.
        """
        for dp_size, budget in ((4, 1), (4, 64), (8, 64)):
            with self.subTest(dp_size=dp_size, budget=budget):
                samples, reference = _random_full_bins(dp_size * 3, seed=17, capacity=24)
                lengths = {item.key: item.pack_tokens for item in samples}
                policy = FeasibleLPTBalancingAlgorithm(max_repair_pairs=budget)
                planner = DynamicPackingPlanner(
                    data_parallel_size=dp_size, local_batch_size=1, seq_len=32,
                    cost_model=CausalAttentionCostModel(), balancing_algorithm=policy,
                )
                with patch.object(planner.balancing_algorithm, "_repair_bins",
                                  wraps=planner.balancing_algorithm._repair_bins) as repair:
                    plans = planner.plan_window(samples, reference_bins=reference, step=0)
                self.assertLessEqual(repair.call_count, budget)
                self.assertIsNone(policy.padding_cost)
                self.assertEqual(Counter(key for plan in plans for key in plan.selected_keys), Counter(lengths.keys()))
                objectives = []
                for plan in plans:
                    costs = []
                    for batch, cost in zip(plan.local_batches, plan.rank_costs):
                        selected = [lengths[key] for key in batch[0].sample_keys]
                        self.assertTrue(0 < sum(selected) <= 32)
                        expected = sum(length ** 2 for length in selected) + (32 - sum(selected)) ** 2
                        self.assertEqual(cost.llm, expected)
                        costs.append(expected)
                    objectives.append(max(costs))
                self.assertEqual(planner.last_balance_decision["accepted_objective"], sum(objectives))
                self.assertLessEqual(sum(objectives), planner.last_balance_decision["original_objective"])
                self.assertEqual(plans, planner.plan_window(tuple(reversed(samples)), reference_bins=reference, step=0))

    def test_padding_cost_changes_the_selected_partition(self) -> None:
        """Feature: Padding-aware assignment.
        Description: Compare partitions whose source-only and padded makespans disagree.
        Expectation: The selected partition reduces combined document and PAD cost.
        """
        samples = tuple(SampleMetadata(length, key=SampleKey(0, index))
                        for index, length in enumerate((1, 1, 1, 1, 2)))
        reference = (tuple(item.key for item in samples[2:]), tuple(item.key for item in samples[:2]))
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=1, seq_len=12, cost_model=CausalAttentionCostModel(),
        )
        plan = planner.plan(samples, reference_bins=reference, step=0)
        self.assertEqual([batch[0].pack_tokens for batch in plan.local_batches], [3, 3])
        self.assertEqual(sorted(cost.llm for cost in plan.rank_costs), [84, 86])
        self.assertTrue(planner.last_balance_decision["accepted"])

    def test_two_full_bins_improve_without_accumulation(self) -> None:
        """Feature: Single-round feasibility repair.
        Description: Repair the capacity-ten greedy dead end with two physical DP ranks.
        Expectation: Cost drops from 240 to 230 with exact capacities and repeated indices preserved.
        """
        samples, reference = _full_pair()
        options = {"reference_bins": reference, "constraints": PackingConstraints(10),
                   "data_parallel_size": 2, "local_batch_size": 1}
        self.assertIs(LPTBalancingAlgorithm().assign(samples, **options), reference)
        self.assertIs(type(resolve_balancing_algorithm()), FeasibleLPTBalancingAlgorithm)
        for validate in (False, True):
            with self.subTest(validate=validate):
                planner = DynamicPackingPlanner(
                    data_parallel_size=2, local_batch_size=1, seq_len=10, validate=validate,
                    cost_model=lambda item: item.cost,
                )
                plan = planner.plan(samples, reference_bins=reference, step=0)
                self.assertEqual([batch[0].pack_tokens for batch in plan.local_batches], [10, 10])
                self.assertEqual(Counter(plan.selected_keys), Counter(item.key for item in samples))
                self.assertEqual(planner.last_balance_decision["original_objective"], 240)
                self.assertEqual(planner.last_balance_decision["accepted_objective"], 230)
                self.assertTrue(planner.last_balance_decision["accepted"])
                planner.min_balance_gain = 0.1
                rejected = planner.plan(samples, reference_bins=reference, step=0)
                self.assertEqual(tuple(batch[0].sample_keys for batch in rejected.local_batches), reference)

    def test_multi_bin_repairs_are_deterministic_and_conservative(self) -> None:
        """Feature: Bounded multi-bin repair.
        Description: Repack fixed-seed full domains of four or twelve bins.
        Expectation: Accepted plans conserve every occurrence and capacity with a non-worsening objective.
        """
        for count in (4, 12):
            failures = accepted = 0
            for seed in range(6):
                samples, reference = _random_full_bins(count, seed)
                options = {"reference_bins": reference, "constraints": PackingConstraints(64),
                           "data_parallel_size": count, "local_batch_size": 1}
                failed = LPTBalancingAlgorithm().assign(samples, **options) is reference
                failures += failed
                planner = DynamicPackingPlanner(
                    data_parallel_size=count, local_batch_size=1, seq_len=64,
                    cost_model=lambda item: item.cost,
                )
                plan = planner.plan(samples, reference_bins=reference, step=0)
                accepted += failed and planner.last_balance_decision["accepted"]
                with self.subTest(count=count, seed=seed):
                    self.assertEqual([batch[0].pack_tokens for batch in plan.local_batches], [64] * count)
                    self.assertEqual(Counter(plan.selected_keys), Counter(item.key for item in samples))
                    self.assertLessEqual(planner.last_balance_decision["accepted_objective"],
                                         planner.last_balance_decision["original_objective"])
                    self.assertEqual(plan, planner.plan(tuple(reversed(samples)), reference_bins=reference, step=0))
            self.assertGreater(failures, 0)
            self.assertGreater(accepted, 0)

    def test_unsupported_domains_keep_the_original_lpt_policy(self) -> None:
        """Feature: Repair scope limits.
        Description: Use stage budgets or multiple bins per rank.
        Expectation: Unsupported cases retain ordinary LPT without running subset-sum repair.
        """
        pair, reference = _full_pair()
        four, four_reference = _random_full_bins(4, 0)
        cases = (
            (pair, reference, PackingConstraints(10, packing_budgets={"encoder": 10}), 2, 1),
            (four, four_reference, PackingConstraints(64), 2, 2),
        )
        algorithm = FeasibleLPTBalancingAlgorithm()
        for samples, bins, constraints, dp_size, local_size in cases:
            options = {"reference_bins": bins, "constraints": constraints,
                       "data_parallel_size": dp_size, "local_batch_size": local_size}
            expected = LPTBalancingAlgorithm().assign(samples, **options)
            with self.subTest(dp=dp_size, local=local_size, constraints=constraints), patch.object(
                    algorithm, "_repair_bins", side_effect=AssertionError("Unsupported repair"),
            ):
                self.assertEqual(algorithm.assign(samples, **options), expected)

    def test_one_megatoken_pair_repairs_many_documents(self) -> None:
        """Feature: Long-context feasibility repair.
        Description: Repair two full 1M packs containing 262 documents after an LPT dead end.
        Expectation: Makespan improves while every document and both exact capacities are preserved.
        """
        capacity = 1 << 20
        samples, reference = _full_pair(scale=100_000)
        samples = list(samples)
        bins = [list(keys) for keys in reference]
        short_length, longer_count = divmod(capacity - 1_000_000, 128)
        for keys in bins:
            for index in range(128):
                tokens = short_length + int(index < longer_count)
                key = SampleKey(0, len(samples), len(samples))
                samples.append(SampleMetadata(tokens, key=key, cost=WorkloadCost()))
                keys.append(key)
        reference = tuple(tuple(keys) for keys in bins)
        self.assertIs(LPTBalancingAlgorithm().assign(
            samples, reference_bins=reference, constraints=PackingConstraints(capacity),
            data_parallel_size=2, local_batch_size=1,
        ), reference)
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=1, seq_len=capacity,
            cost_model=lambda item: item.cost,
        )
        plan = planner.plan(samples, reference_bins=reference, step=0)
        self.assertEqual([batch[0].pack_tokens for batch in plan.local_batches], [capacity, capacity])
        self.assertEqual(Counter(plan.selected_keys), Counter(item.key for item in samples))
        self.assertEqual(planner.last_balance_decision["original_objective"], 240)
        self.assertEqual(planner.last_balance_decision["accepted_objective"], 230)
        self.assertTrue(planner.last_balance_decision["accepted"])

    def test_table_budget_rejects_large_token_domains(self) -> None:
        """Feature: Bounded subset-sum allocation.
        Description: Scale a known dead end beyond the 256 MiB table budget.
        Expectation: The feasible reference is returned without constructing an oversized table.
        """
        samples, reference = _full_pair(scale=50_000_000)
        result = FeasibleLPTBalancingAlgorithm().assign(
            samples, reference_bins=reference, constraints=PackingConstraints(500_000_000),
            data_parallel_size=2, local_batch_size=1,
        )
        self.assertIs(result, reference)

    def test_successful_greedy_assignments_are_retained(self) -> None:
        """Feature: Successful LPT compatibility.
        Description: Balance equal-length documents whose greedy packing succeeds.
        Expectation: The repair policy retains exactly the ordinary LPT candidate.
        """
        samples = tuple(SampleMetadata(5, key=SampleKey(0, index, index), cost=WorkloadCost(llm=cost))
                        for index, cost in enumerate((9, 9, 1, 1)))
        reference = ((samples[0].key, samples[1].key), (samples[2].key, samples[3].key))
        options = {"reference_bins": reference, "constraints": PackingConstraints(10),
                   "data_parallel_size": 2, "local_batch_size": 1}
        algorithm = FeasibleLPTBalancingAlgorithm()
        with patch.object(algorithm, "_repair_bins", wraps=algorithm._repair_bins):
            self.assertEqual(algorithm.assign(samples, **options), LPTBalancingAlgorithm().assign(samples, **options))

    def test_large_windows_share_one_pair_budget(self) -> None:
        """Feature: Repair beyond sixteen packs.
        Description: Balance three rounds at DP8 and DP32 with a seven-pair budget.
        Expectation: Repair runs, conserves all samples and capacities, and never exceeds seven checks per window.
        """
        for dp_size, seed, capacity in ((8, 38, 64), (32, 0, 65536)):
            samples, reference = _random_full_bins(dp_size * 3, seed, capacity=capacity)
            algorithm = FeasibleLPTBalancingAlgorithm(max_repair_pairs=7)
            planner = DynamicPackingPlanner(
                data_parallel_size=dp_size, local_batch_size=1, seq_len=capacity,
                cost_model=lambda item: item.cost, balancing_algorithm=algorithm,
            )
            with self.subTest(dp_size=dp_size), patch.object(
                    algorithm, "_repair_bins", wraps=algorithm._repair_bins,
            ) as repair:
                plans = planner.plan_window(samples, reference_bins=reference, step=0)
                self.assertGreater(repair.call_count, 0)
                self.assertLessEqual(repair.call_count, 7)
            self.assertEqual(Counter(key for plan in plans for key in plan.selected_keys),
                             Counter(item.key for item in samples))
            self.assertEqual([batch[0].pack_tokens for plan in plans for batch in plan.local_batches],
                             [capacity] * (dp_size * 3))
            if dp_size == 8:
                self.assertNotEqual(
                    Counter(frozenset(batch[0].sample_keys) for plan in plans for batch in plan.local_batches),
                    Counter(map(frozenset, reference)),
                )
            self.assertLess(planner.last_balance_decision["accepted_objective"],
                            planner.last_balance_decision["original_objective"])
            self.assertEqual(plans, planner.plan_window(tuple(reversed(samples)), reference_bins=reference, step=0))
            planner.min_balance_gain = 0.99
            rejected = planner.plan_window(samples, reference_bins=reference, step=0)
            self.assertEqual(
                tuple(batch[0].sample_keys for plan in rejected for batch in plan.local_batches), reference,
            )

    def test_unsuccessful_checks_consume_budget(self) -> None:
        """Feature: Hard repair work limit.
        Description: Every two-pack attempt returns its unchanged feasible input.
        Expectation: The configured number of distinct attempts exhausts the budget even without improvements.
        """
        samples, reference = _random_full_bins(24, 38)
        algorithm = FeasibleLPTBalancingAlgorithm(max_repair_pairs=5)
        with patch.object(LPTBalancingAlgorithm, "assign", return_value=reference), patch.object(
                algorithm, "_repair_bins", side_effect=lambda _items, _capacity, bins, *_context: bins,
        ) as repair:
            result = algorithm.assign(
                samples, reference_bins=reference, constraints=PackingConstraints(64),
                data_parallel_size=24, local_batch_size=1, objective_group_size=8,
            )
        self.assertIs(result, reference)
        self.assertEqual(repair.call_count, 5)
        self.assertEqual(len({frozenset(call.args[2]) for call in repair.call_args_list}), 5)

    def test_pair_gain_cannot_worsen_window_objective(self) -> None:
        """Feature: Complete-window repair objective.
        Description: Changing costs 100/1 to 60/41 lowers the pair maximum but raises the window score 102 to 140.
        Expectation: The locally beneficial pair is rejected and the original window remains intact.
        """
        samples = tuple(
            SampleMetadata(1, key=SampleKey(0, index), cost=WorkloadCost(llm=cost))
            for index, cost in enumerate((60, 40, 49, 50, 1, 1, 1, 0))
        )
        reference = tuple(tuple(item.key for item in samples[start:start + 2]) for start in range(0, 8, 2))
        algorithm = FeasibleLPTBalancingAlgorithm()
        with patch.object(LPTBalancingAlgorithm, "assign", return_value=reference), patch.object(
                algorithm, "_repair_candidates", return_value=iter(((0, 3),)),
        ):
            result = algorithm.assign(
                samples, reference_bins=reference, constraints=PackingConstraints(2),
                data_parallel_size=4, local_batch_size=1, objective_group_size=2,
            )
        self.assertIs(result, reference)

    def test_candidate_selection_stays_sparse_at_large_dp(self) -> None:
        """Feature: Sparse bottleneck selection.
        Description: Generate candidates for three rounds with thousands of packs each.
        Expectation: Only a few partners per round bottleneck are considered, independently of DP size.
        """
        algorithm = FeasibleLPTBalancingAlgorithm()
        for dp_size in (8, 2048):
            costs = [WorkloadCost(llm=dp_size * 3 - index) for index in range(dp_size * 3)]
            leaders = {0, dp_size, dp_size * 2}
            pairs = list(algorithm._repair_candidates(costs, dp_size))
            with self.subTest(dp_size=dp_size):
                self.assertLessEqual(len(pairs), 36)
                self.assertTrue(all(left in leaders or right in leaders for left, right in pairs))
                self.assertTrue(all(left < right for left, right in pairs))
                self.assertEqual(set().union(*(set(pair) & leaders for pair in pairs)), leaders)

    def test_repair_budget_and_round_size_require_positive_integers(self) -> None:
        """Feature: Repair configuration validation.
        Description: Supply invalid work budgets or window round sizes.
        Expectation: Invalid inputs fail with descriptive errors.
        """
        for value in (0, -1, True, 1.5):
            with self.subTest(budget=value), self.assertRaisesRegex(ValueError, "max_repair_pairs"):
                FeasibleLPTBalancingAlgorithm(max_repair_pairs=value)
        samples, reference = _full_pair()
        for value in (0, -1, True, 1.5, 3):
            with self.subTest(group_size=value), self.assertRaisesRegex(ValueError, "objective_group_size"):
                FeasibleLPTBalancingAlgorithm().assign(
                    samples, reference_bins=reference, constraints=PackingConstraints(10),
                    data_parallel_size=2, local_batch_size=1, objective_group_size=value,
                )

    def test_custom_assignment_retains_existing_signature(self) -> None:
        """Feature: Custom policy compatibility.
        Description: A feasible-LPT subclass overrides assign without the new optional window argument.
        Expectation: The planner respects its existing signature and still schedules complete packs.
        """
        class CustomAssignment(FeasibleLPTBalancingAlgorithm):
            """Keep original document groups with the existing policy signature."""

            # The legacy override deliberately omits the optional window argument.
            def assign(  # pylint: disable=arguments-differ
                    self, samples: tuple, *, reference_bins: tuple, constraints: PackingConstraints,
                    data_parallel_size: int, local_batch_size: int,
            ) -> tuple:
                """Return reference groups without accepting window-specific arguments."""
                del samples, constraints, data_parallel_size, local_batch_size
                return reference_bins

        samples, reference = _random_full_bins(24, 38)
        planner = DynamicPackingPlanner(
            data_parallel_size=8, local_batch_size=1, seq_len=64, cost_model=lambda item: item.cost,
            balancing_algorithm=CustomAssignment(),
        )
        plans = planner.plan_window(samples, reference_bins=reference, step=0)
        self.assertEqual(Counter(frozenset(batch[0].sample_keys) for plan in plans for batch in plan.local_batches),
                         Counter(map(frozenset, reference)))

    def test_partial_pairs_preserve_feasibility(self) -> None:
        """Feature: Underfilled pair feasibility.
        Description: Repack partial bins across three rounds with multiple fixed seeds.
        Expectation: Plans remain feasible, deterministic and non-worsening while conserving occurrences.
        """
        for seed in range(12):
            samples, reference = _random_full_bins(6, seed, capacity=24)
            planner = DynamicPackingPlanner(data_parallel_size=2, local_batch_size=1, seq_len=28,
                                            cost_model=lambda item: item.cost)
            plans = planner.plan_window(samples, reference_bins=reference, step=0)
            self.assertEqual(Counter(key for plan in plans for key in plan.selected_keys),
                             Counter(item.key for item in samples))
            self.assertTrue(all(0 < batch[0].pack_tokens <= 28 for plan in plans for batch in plan.local_batches))
            self.assertLessEqual(planner.last_balance_decision["accepted_objective"],
                                 planner.last_balance_decision["original_objective"])
            self.assertEqual(plans, planner.plan_window(tuple(reversed(samples)), reference_bins=reference, step=0))

    def test_successful_underfilled_greedy_can_improve(self) -> None:
        """Feature: Refinement after successful greedy placement.
        Description: Compare an underfilled LPT assignment before and after pair repair.
        Expectation: Refinement lowers the sum of scheduled round maxima.
        """
        samples, reference = _random_full_bins(6, 1, capacity=24)
        algorithm = FeasibleLPTBalancingAlgorithm()
        options = {"reference_bins": reference, "constraints": PackingConstraints(28),
                   "data_parallel_size": 6, "local_batch_size": 1, "objective_group_size": 2}
        with patch.object(algorithm, "_repair_bin_pairs", side_effect=lambda _lookup, bins, *_rest: bins):
            initial = algorithm.assign(samples, **options)
        self.assertIsNot(initial, reference)
        refined = algorithm.assign(samples, **options)
        lookup = {sample.key: sample for sample in samples}

        def score(bins: tuple) -> float:
            """Evaluate synchronous rounds independently of the planner."""
            costs = [sum(lookup[key].cost.llm for key in keys) for keys in bins]
            return sum(sorted(costs, reverse=True)[::2])

        self.assertLess(score(refined), score(initial))

    def test_window_placement_targets_scheduled_rounds(self) -> None:
        """Feature: Window-aware placement scoring.
        Description: Compare placing a short sample alongside heavy peers or in a lighter round.
        Expectation: Placement prefers heavy peers when that avoids raising another round maximum.
        """
        item = SampleMetadata(2, key=SampleKey(0, 0), cost=WorkloadCost(llm=4))
        loads = [100, 36, 36, 36, 25, 25, 25, 25, 16, 16, 16, 16]
        algorithm = _WindowLPT(4)
        heavy = algorithm._placement_score(_AssignmentBin(1, 0), item, 100, loads, [0] * 12)
        light = algorithm._placement_score(_AssignmentBin(8, 0), item, 100, loads, [0] * 12)
        self.assertEqual(heavy[0], 141)
        self.assertEqual(light[0], 145)
        self.assertLess(heavy, light)
