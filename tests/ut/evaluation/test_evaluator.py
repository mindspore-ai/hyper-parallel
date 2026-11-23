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
from unittest.mock import patch
from typing import Mapping, Optional

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
        with self.assertRaises(ValueError):
            metric.update(_stats(numerator=1.0, denominator=1.0, kind=KIND_OBJECTIVE))


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


class TestSharedReduction(unittest.TestCase):
    """Built-in metrics and sample counts share one numerical collective."""

    def test_compute_never_reduces_twice(self):
        class _InheritedLoss(LossMetric):
            """Keep the ordinary accumulation lifecycle in a derived metric."""

        def gather(outputs: list, value: object, **kwargs: object) -> None:
            """Replicate a control record across the two mocked lanes."""
            del kwargs
            outputs[:] = [value, value]

        def reduce(value: torch.Tensor, **kwargs: object) -> None:
            """Sum two identical mocked numerical contributions."""
            del kwargs
            value.mul_(2)

        metrics = [LossMetric(), PerplexityMetric(), _InheritedLoss(name="inherited_loss")]
        results = iter([EvalStepResult(statistics=_stats(6, 3), sample_count=torch.tensor(1)), None])
        with (patch("hyper_parallel.evaluation.aggregation.dist.all_gather_object", side_effect=gather),
              patch("hyper_parallel.evaluation.aggregation.dist.all_reduce", side_effect=reduce) as reduction):
            evaluator = Evaluator(lambda _: next(results), metrics, ReductionContext(object(), 2), torch.device("cpu"))
            report = evaluator.run()
            for metric in metrics:
                self.assertEqual(metric.compute(), metric.compute())
            self.assertEqual(reduction.call_count, 1)
        self.assertEqual(report.num_samples, 2)
        self.assertEqual(report.num_tokens, 6)
        self.assertEqual(report.metrics["loss"], 2)
        self.assertEqual(report.metrics["inherited_loss"], 2)


class TestMetricProtocol(unittest.TestCase):
    """A LossMetric subclass joins the shared reduction by declaration, not type."""

    def test_subclass_uses_the_shared_reduction(self):
        class _SubclassMetric(LossMetric):
            """Inherit the protocol but fail if its own reduction ever runs."""

            @property
            def supports_shared_reduction(self) -> bool:
                """Explicitly consume the shared statistics despite the custom hook."""
                return True

            def synchronize(self, context: ReductionContext, device: torch.device) -> None:
                """Reject a separate reduction for this shared-statistics metric."""
                del context, device
                raise AssertionError("a shared-reduction metric must not reduce again")

        results = iter([EvalStepResult(statistics=_stats(6, 3), sample_count=torch.tensor(1)), None])
        evaluator = Evaluator(lambda _: next(results), [_SubclassMetric()],
                              ReductionContext(), torch.device("cpu"))
        report = evaluator.run()
        self.assertEqual(report.metrics["loss"], 2.0)

    def test_custom_update_keeps_its_accumulated_values(self):
        """Shared raw statistics must not replace a subclass's transformed totals."""
        class _DoubleLoss(LossMetric):
            """Use a different numerator while retaining token weighting."""

            def update(self, statistics: Mapping[str, LossStatistics]) -> None:
                """Double each numerator before accumulation."""
                super().update({
                    name: LossStatistics(stat.numerator * 2, stat.denominator, stat.kind)
                    for name, stat in statistics.items()
                })

        results = iter([EvalStepResult(statistics=_stats(6, 3), sample_count=torch.tensor(1)), None])
        evaluator = Evaluator(lambda _: next(results), [LossMetric(), _DoubleLoss(name="double_loss")],
                              ReductionContext(), torch.device("cpu"))
        report = evaluator.run()
        self.assertEqual(report.metrics, {"loss": 2.0, "double_loss": 4.0})

    def test_custom_synchronize_is_not_skipped(self):
        """A subclass may require its own synchronization before computing."""
        class _SynchronizedLoss(LossMetric):
            """Record whether the custom synchronization hook was invoked."""

            def __init__(self) -> None:
                """Start with no synchronized state."""
                super().__init__()
                self.synchronized = False

            def synchronize(self, context: ReductionContext, device: torch.device) -> None:
                """Reduce the metric and publish its synchronization state."""
                super().synchronize(context, device)
                self.synchronized = True

        metric = _SynchronizedLoss()
        results = iter([EvalStepResult(statistics=_stats(6, 3), sample_count=torch.tensor(1)), None])
        report = Evaluator(lambda _: next(results), [metric], ReductionContext(), torch.device("cpu")).run()
        self.assertTrue(metric.synchronized)
        self.assertEqual(report.metrics["loss"], 2.0)


class TestReductionContext(unittest.TestCase):
    """A context must describe a group that can actually supply its contributors."""

    def test_nonpositive_world_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            ReductionContext(world_size=0)

    def test_world_size_without_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "process group"):
            ReductionContext(world_size=2)

    def test_group_size_mismatch_is_rejected(self):
        with patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_world_size", return_value=3):
            with self.assertRaisesRegex(ValueError, "does not match"):
                ReductionContext(group=object(), world_size=2)


if __name__ == "__main__":
    unittest.main()
