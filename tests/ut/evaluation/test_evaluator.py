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
"""Unit tests for the standalone evaluation leaf package."""

import math
import unittest
from typing import Optional

import torch

from hyper_parallel.evaluation import (
    Evaluator,
    LossMetric,
    LossStatistics,
    PerplexityMetric,
    ReductionContext,
)
from hyper_parallel.evaluation.types import KIND_LM_NLL, KIND_OBJECTIVE, EvalStepResult


def _stats(numerator: float, denominator: float, kind: str = KIND_LM_NLL) -> dict:
    """Build one named statistic with explicit additive values."""
    return {
        "foundation": LossStatistics(
            numerator=torch.tensor(numerator, dtype=torch.float64),
            denominator=torch.tensor(denominator, dtype=torch.float64),
            kind=kind,
        )
    }


class TestLossMetric(unittest.TestCase):
    """Token-weighted aggregation over multiple batches."""

    def test_token_weighted_mean_matches_hand_computation(self):
        """Unequal batches must be weighted by tokens, not averaged per batch."""
        metric = LossMetric()
        metric.reset()
        # Two batches with different token counts: 10 tokens at loss 2.0 and
        # 30 tokens at loss 4.0. Per-batch averaging would give 3.0.
        metric.update(_stats(numerator=20.0, denominator=10.0))
        metric.update(_stats(numerator=120.0, denominator=30.0))
        metric.synchronize(ReductionContext(), torch.device("cpu"))
        self.assertAlmostEqual(metric.compute()["loss"], 140.0 / 40.0, places=12)

    def test_compute_is_pure_and_repeatable(self):
        """A second compute call must not re-accumulate or re-reduce."""
        metric = LossMetric()
        metric.reset()
        metric.update(_stats(numerator=6.0, denominator=3.0))
        metric.synchronize(ReductionContext(), torch.device("cpu"))
        first = metric.compute()
        second = metric.compute()
        self.assertEqual(first, second)

    def test_compute_requires_synchronize(self):
        """Skipping the reduce is a programming error, not a silent zero."""
        metric = LossMetric()
        metric.reset()
        metric.update(_stats(numerator=1.0, denominator=1.0))
        with self.assertRaises(RuntimeError):
            metric.compute()

    def test_reset_drops_state(self):
        """Reusing an instance across runs starts from zero."""
        metric = LossMetric()
        metric.reset()
        metric.update(_stats(numerator=4.0, denominator=2.0))
        metric.synchronize(ReductionContext(), torch.device("cpu"))
        metric.reset()
        metric.update(_stats(numerator=9.0, denominator=3.0))
        metric.synchronize(ReductionContext(), torch.device("cpu"))
        self.assertAlmostEqual(metric.compute()["loss"], 3.0, places=12)

    def test_zero_denominator_yields_no_value(self):
        """An empty evaluation reports nothing rather than dividing by zero."""
        metric = LossMetric()
        metric.reset()
        metric.synchronize(ReductionContext(), torch.device("cpu"))
        self.assertEqual(metric.compute(), {})

    def test_kind_change_is_rejected(self):
        """A statistic cannot change meaning between batches."""
        metric = LossMetric()
        metric.reset()
        metric.update(_stats(numerator=1.0, denominator=1.0, kind=KIND_LM_NLL))
        metric.update(_stats(numerator=1.0, denominator=1.0, kind=KIND_OBJECTIVE))
        with self.assertRaises(ValueError):
            metric.synchronize(ReductionContext(), torch.device("cpu"))


class TestPerplexityMetric(unittest.TestCase):
    """PPL derives from the reduced NLL and refuses non-NLL statistics."""

    def test_perplexity_is_exp_of_nll(self):
        """PPL equals exp(loss) for a declared NLL statistic."""
        metric = PerplexityMetric()
        metric.reset()
        metric.update(_stats(numerator=2.0 * 5.0, denominator=5.0))
        metric.synchronize(ReductionContext(), torch.device("cpu"))
        self.assertAlmostEqual(metric.compute()["perplexity"], math.exp(2.0), places=9)

    def test_non_nll_statistic_is_rejected(self):
        """Objective terms that include auxiliary loss must not become PPL."""
        metric = PerplexityMetric()
        metric.reset()
        metric.update(_stats(numerator=3.0, denominator=1.0, kind=KIND_OBJECTIVE))
        metric.synchronize(ReductionContext(), torch.device("cpu"))
        with self.assertRaises(ValueError):
            metric.compute()

    def test_loss_and_perplexity_share_one_reduction(self):
        """Both metrics read the same reduced NLL without double counting."""
        loss_metric = LossMetric()
        ppl_metric = PerplexityMetric()
        for metric in (loss_metric, ppl_metric):
            metric.reset()
            metric.update(_stats(numerator=9.0, denominator=3.0))
            metric.synchronize(ReductionContext(), torch.device("cpu"))
        self.assertAlmostEqual(loss_metric.compute()["loss"], 3.0, places=12)
        self.assertAlmostEqual(
            ppl_metric.compute()["perplexity"], math.exp(3.0), places=9
        )


class TestEvaluator(unittest.TestCase):
    """The loop bounds rounds and reduces exactly once."""

    def test_round_limit_stops_the_loop(self):
        """max_batches caps the number of forwards."""
        calls = []

        def step(round_index: int) -> EvalStepResult:
            """Record the round and return one unit-weight statistic."""
            calls.append(round_index)
            return EvalStepResult(
                statistics=_stats(numerator=1.0, denominator=1.0),
                sample_count=torch.tensor(1.0, dtype=torch.float64),
            )

        metric = LossMetric()
        evaluator = Evaluator(step, [metric], ReductionContext(), torch.device("cpu"))
        report = evaluator.run(max_batches=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(report.num_batches, 3)
        self.assertEqual(report.num_samples, 3)
        self.assertAlmostEqual(report.metrics["loss"], 1.0, places=12)

    def test_none_result_ends_the_loop(self):
        """A step that returns None reports exhaustion."""
        state = {"rounds": 0}

        def step(round_index: int) -> Optional[EvalStepResult]:
            """Yield two rounds, then report exhaustion."""
            del round_index
            if state["rounds"] >= 2:
                return None
            state["rounds"] += 1
            return EvalStepResult(
                statistics=_stats(numerator=2.0, denominator=1.0),
                sample_count=torch.tensor(1.0, dtype=torch.float64),
            )

        evaluator = Evaluator(step, [LossMetric()], ReductionContext(), torch.device("cpu"))
        report = evaluator.run()
        self.assertEqual(report.num_batches, 2)
        self.assertAlmostEqual(report.metrics["loss"], 2.0, places=12)

    def test_empty_evaluation_is_an_error(self):
        """Consuming nothing is a configuration failure, not a zero loss."""

        def step(round_index: int) -> None:
            """Report an immediately exhausted source."""
            del round_index

        evaluator = Evaluator(step, [LossMetric()], ReductionContext(), torch.device("cpu"))
        with self.assertRaises(RuntimeError):
            evaluator.run()


if __name__ == "__main__":
    unittest.main()
