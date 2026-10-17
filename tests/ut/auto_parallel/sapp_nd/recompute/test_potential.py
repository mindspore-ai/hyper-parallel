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
"""Tests for sorting, retaining and exporting recompute plans at fixed parallel degrees.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/recompute/test_potential.py -v
"""
import csv
import json
import os
import tempfile
import unittest
from typing import Any, Optional, Tuple

import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.recompute.candidate import LayerRange, RecomputeChoice
from hyper_parallel.auto_parallel.sapp_nd.recompute.front import LayerOption
from hyper_parallel.auto_parallel.sapp_nd.recompute.potential import (
    keep_recompute_potential,
    recompute_next_plot_groups,
    recompute_plot_data,
    sort_recompute_results,
    write_recompute_csv,
)

# The degrees every plan of one fixed strategy reports.
_DEGREES = (("DP", 4), ("PP", 2))
# A second strategy's degrees, which no alternative of the first may claim.
_OTHER = (("DP", 2), ("PP", 4))


def _option(link: float = 0.0) -> LayerOption:
    """An option, costs aside, which offloads *link* bytes a micro-batch."""
    return LayerOption(recompute=frozenset(), memory_per_micro_batch=1.0, memory_once=0.0,
                       forward_time=1.0, backward_time=2.0, link_bandwidth=link)


def _plan(
    modes: str,
    score: Optional[float] = 1.0,
    memory: float = 100.0,
    degrees: Tuple[Tuple[str, int], ...] = _DEGREES,
    kind: Any = None,
    link: float = 0.0,
) -> RecomputeChoice:
    """A plan whose layers run *modes*, one letter a layer: f for full, o for off, s for selective.

    The letters may be separated by spaces, which read better and keep a plan of three
    layers from spelling a word.
    """
    names = {"f": "full", "o": "off", "s": "selective"}
    modes = modes.replace(" ", "")
    ranges = tuple(
        LayerRange(index, 1, kind, _option(link), names[letter])
        for index, letter in enumerate(modes)
    )
    return RecomputeChoice(ranges=ranges, stage_memory=(memory,), stage_savings=(float(len(modes)),),
                           score=score, dimensions=degrees, score_parts=(score or 0.0, 0.0))


class TestSortRecomputeResults(unittest.TestCase):
    """The order the cost model's estimate puts the plans in."""

    def test_the_fastest_estimate_comes_first_and_memory_breaks_a_tie(self):
        """
        Feature: sort_recompute_results.
        Description: Three plans, two of them scored the same.
        Expectation: Ascending score, the lighter of the tied pair first,
            and the input sequence is left alone.
        """
        slow = _plan("o o o", score=3.0)
        tied_heavy = _plan("o o f", score=1.0, memory=90.0)
        tied_light = _plan("o f o", score=1.0, memory=80.0)
        results = [slow, tied_heavy, tied_light]
        self.assertEqual(sort_recompute_results(results), [tied_light, tied_heavy, slow])
        self.assertEqual(results, [slow, tied_heavy, tied_light])

    def test_a_plan_without_a_finite_estimate_is_refused(self):
        """
        Feature: sort_recompute_results.
        Description: A plan with no score, and one scored infinite.
        Expectation: Both are refused, rather than sorted anywhere.
        """
        with self.assertRaises(ValueError):
            sort_recompute_results([_plan("o o", score=None)])
        with self.assertRaises(ValueError):
            sort_recompute_results([_plan("o o", score=float("inf"))])


class TestKeepRecomputePotential(unittest.TestCase):
    """Retaining the fastest plan that recomputes one more layer."""

    def test_the_fastest_one_more_full_plan_is_appended(self):
        """
        Feature: keep_recompute_potential.
        Description: Two layers off, with alternatives recomputing one of
            them and one recomputing both.
        Expectation: The fastest of the one-more-full plans is appended and
            returned; the two-more-full plan is left for a later step.
        """
        current = _plan("o o", score=5.0)
        quick = _plan("f o", score=2.0)
        results = [_plan("o f", score=3.0), quick, _plan("f f", score=1.0)]
        kept = [current]
        self.assertIs(keep_recompute_potential(current, kept, results), quick)
        self.assertEqual(kept, [current, quick])

    def test_another_strategy_an_offloading_plan_and_one_already_kept_are_skipped(self):
        """
        Feature: keep_recompute_potential.
        Description: The fastest alternative belongs to another strategy, the
            next states the plan already retained, and the one after it
            offloads.
        Expectation: Only the remaining plan of this strategy is appended,
            and a step with nothing left to turn up returns None.
        """
        current = _plan("f o o", score=4.0)
        kept = [_plan("o o o", score=5.0), current]
        results = [_plan("f f o", score=1.0, degrees=_OTHER), _plan("f o o", score=2.0),
                   _plan("f o f", score=2.5, link=8.0), _plan("f f o", score=3.0)]
        appended = keep_recompute_potential(current, kept, results)
        self.assertEqual([item.mode for item in appended.ranges], ["full", "full", "off"])
        self.assertEqual(appended.score, 3.0)
        self.assertEqual(kept[-1], appended)
        self.assertIsNone(keep_recompute_potential(appended, kept, results))

    def test_a_plan_changing_the_kind_or_several_layers_is_skipped(self):
        """
        Feature: keep_recompute_potential.
        Description: Alternatives that turn a layer of another kind up, that
            turn two layers up at once, and that turn one down instead.
        Expectation: None of them qualifies, so nothing is appended.
        """
        current = _plan("o o s", score=5.0, kind="dense")
        results = [_plan("f o s", score=1.0, kind="moe"), _plan("f f s", score=2.0, kind="dense"),
                   _plan("o o o", score=3.0, kind="dense")]
        kept = [current]
        self.assertIsNone(keep_recompute_potential(current, kept, results))
        self.assertEqual(kept, [current])


class TestRecomputeNextPlotGroups(unittest.TestCase):
    """Selecting the plans a comparison plot shows for each strategy."""

    @staticmethod
    def _configs(count: int) -> list:
        """*count* ranked configurations, scored 10, 20 and so on."""
        configs = []
        for index in range(count):
            dims = Dim.Dimensions([(Dim.DP, 4), (Dim.PP, 2)], all_dims=[Dim.DP, Dim.PP])
            dims.rank = index + 1
            configs.append((dims, 100.0, 10.0 * (index + 1), []))
        return configs

    def test_every_plan_below_the_next_score_and_the_first_above_it(self):
        """
        Feature: recompute_next_plot_groups.
        Description: Two strategies scored 10 and 20, the first holding
            plans at 10, 15 and 25.
        Expectation: The first group holds the two plans below 20 and the
            first at or above it; the last group takes the mean size of the
            groups before it.
        """
        configs = self._configs(2)
        first, second = configs[0][0], configs[1][0]
        results = {
            first: [_plan("o o", score=10.0), _plan("f o", score=15.0), _plan("f f", score=25.0)],
            second: [_plan("o o", score=20.0), _plan("f o", score=22.0), _plan("f f", score=24.0)],
        }
        groups = recompute_next_plot_groups(configs, results)
        self.assertEqual([config for config, _ in groups], [first, second])
        self.assertEqual([choice.score for choice in groups[0][1]], [10.0, 15.0, 25.0])
        self.assertEqual([choice.score for choice in groups[1][1]], [20.0, 22.0, 24.0])

    def test_two_plans_are_kept_where_the_scores_tie(self):
        """
        Feature: recompute_next_plot_groups.
        Description: A strategy whose own plan already scores the next
            strategy's score.
        Expectation: Its group still shows an alternative beside it.
        """
        configs = self._configs(2)
        first, second = configs[0][0], configs[1][0]
        results = {
            first: [_plan("o o", score=20.0), _plan("f o", score=21.0)],
            second: [_plan("o o", score=20.0)],
        }
        groups = recompute_next_plot_groups(configs, results)
        self.assertEqual([choice.score for choice in groups[0][1]], [20.0, 21.0])
        self.assertEqual([choice.score for choice in groups[1][1]], [20.0])

    def test_the_number_of_strategies_shown_is_bounded(self):
        """
        Feature: recompute_next_plot_groups.
        Description: Three strategies with a limit of two, then a limit of
            zero, then no strategy at all.
        Expectation: The limit holds, and nothing is selected without one.
        """
        configs = self._configs(3)
        results = {config: [_plan("o o", score=score)] for config, _, score, _ in configs}
        self.assertEqual(len(recompute_next_plot_groups(configs, results, top_num=2)), 2)
        self.assertEqual(recompute_next_plot_groups(configs, results, top_num=0), [])
        self.assertEqual(recompute_next_plot_groups([], results), [])


class TestRecomputePlotData(unittest.TestCase):
    """The records a comparison plot draws."""

    def test_each_record_states_the_plan_the_trainer_reads(self):
        """
        Feature: recompute_plot_data.
        Description: One strategy with a plan per layer, a plan of one mode
            for the whole model, and a plan the optimizer could not make.
        Expectation: The per-layer record states the mode and the layers
            that run another, the whole-model record its mode alone, and
            the missing plan has neither.
        """
        dims = Dim.Dimensions([(Dim.DP, 4)], all_dims=[Dim.DP])
        dims.rank = 1
        whole = RecomputeChoice(ranges=(), stage_memory=(80.0,), stage_savings=(0.0,), mode="full",
                                score=2.0, dimensions=_DEGREES, score_parts=(2.0, 0.0))
        missing = RecomputeChoice(ranges=(), stage_memory=(90.0,), stage_savings=(), score=3.0,
                                  dimensions=_DEGREES, score_parts=(3.0, 0.0))
        records = recompute_plot_data([(dims, [_plan("f f o o o", score=1.0), whole, missing])])[0][1]
        self.assertEqual(records[0]["mode"], "full")
        self.assertEqual(records[0]["layers"], {"2-4": "off"})
        self.assertEqual(records[0]["full_layers"], 2)
        self.assertEqual((records[1]["mode"], records[1]["layers"], records[1]["full_layers"]),
                         ("full", {}, None))
        self.assertEqual((records[2]["mode"], records[2]["layers"], records[2]["full_layers"]),
                         (None, None, None))
        self.assertEqual([record["score"] for record in records], [1.0, 2.0, 3.0])
        self.assertEqual([record["memory"] for record in records], [100.0, 80.0, 90.0])


class TestWriteRecomputeCsv(unittest.TestCase):
    """The CSV a retained sequence is exported to."""

    def test_each_step_names_the_layer_it_changed(self):
        """
        Feature: write_recompute_csv.
        Description: One strategy's sequence of three plans, each
            recomputing one more layer.
        Expectation: A row a step, the degrees in their own columns, the
            first step naming no changed layer and each later step naming
            the layer it turned up, and the mode the trainer reads stated
            explicitly on every row.
        """
        dims = Dim.Dimensions([(Dim.DP, 4), (Dim.PP, 2)], all_dims=[Dim.DP, Dim.PP])
        dims.rank = 3
        sequence = [_plan("o o o", score=5.0), _plan("f o o", score=4.0), _plan("f f o", score=3.0)]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "nested", "recompute_potentials.csv")
            write_recompute_csv({dims: sequence}, path)
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["TOP"] for row in rows], ["3", "3", "3"])
        self.assertEqual([row["step"] for row in rows], ["0", "1", "2"])
        self.assertEqual([row["DP"] for row in rows], ["4", "4", "4"])
        self.assertEqual([row["PP"] for row in rows], ["2", "2", "2"])
        self.assertEqual([row["changed_layer"] for row in rows], ["", "0", "1"])
        self.assertEqual([row["full_layers"] for row in rows], ["0", "1", "2"])
        self.assertEqual([row["score"] for row in rows], ["5.0", "4.0", "3.0"])
        self.assertEqual([row["plan_scope"] for row in rows], ["per_layer"] * 3)
        self.assertEqual([row["recompute_mode"] for row in rows], ["off", "full", "full"])
        self.assertEqual(json.loads(rows[2]["recompute_layers"]), {"2": "off"})

    def test_a_whole_model_plan_and_an_offloading_plan_state_their_scope(self):
        """
        Feature: write_recompute_csv.
        Description: A strategy whose plan covers the whole model, one that
            offloads, and one the optimizer could not make.
        Expectation: Each row says how wide its plan is, and the CSV is
            written afresh over an earlier one.
        """
        dims = Dim.Dimensions([(Dim.DP, 4)], all_dims=[Dim.DP])
        dims.rank = 1
        whole = RecomputeChoice(ranges=(), stage_memory=(80.0,), stage_savings=(0.0,), mode="selective",
                                score=2.0, dimensions=_DEGREES)
        missing = RecomputeChoice(ranges=(), stage_memory=(90.0,), stage_savings=(), score=3.0,
                                  dimensions=_DEGREES)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "recompute_potentials.csv")
            write_recompute_csv({dims: [_plan("o o", score=9.0)] * 4}, path)
            write_recompute_csv({dims: [_plan("o o", score=1.0, link=8.0), whole, missing]}, path)
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["plan_scope"] for row in rows], ["offload", "whole_model", "unavailable"])
        self.assertEqual(rows[1]["recompute_mode"], "selective")
        self.assertEqual(rows[2]["recompute_mode"], "")
        self.assertEqual(rows[2]["full_layers"], "")


if __name__ == "__main__":
    unittest.main()
