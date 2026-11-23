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
"""Bounded evaluation loop.

The evaluator owns exactly three things: iterating the injected step for a
bounded number of rounds, feeding the resulting additive statistics into the
metrics, and triggering the single global reduction. Everything that knows
about text, labels, token counting, or the model's forward contract stays in
the Trainer-injected step -- this module is a leaf package and cannot import
``hyper_parallel.trainer``.
"""

from __future__ import annotations

__all__ = ["EvaluationReport", "Evaluator"]

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from hyper_parallel.evaluation.aggregation import accumulate_statistics, reduce_scalar, reduce_statistics
from hyper_parallel.evaluation.metrics.base import Metric
from hyper_parallel.evaluation.types import EvalStepResult, LossStatistics, ReductionContext


@dataclass
class EvaluationReport:
    """Outcome of one bounded evaluation pass.

    Args:
        metrics: Flat metric name to value.
        num_samples: Real samples consumed, summed over contributors.
        num_tokens: Global valid-token denominator.
        num_batches: Forward rounds executed by this rank.
        seconds: Wall time spent in the loop, for throughput reporting.
        status: ``ok``, or a marker explaining why a derived value is absent
            (for example ``perplexity_overflow``).
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    num_tokens: float = 0.0
    num_batches: int = 0
    seconds: float = 0.0
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a plain serializable mapping."""
        return {
            "metrics": dict(self.metrics),
            "num_samples": self.num_samples,
            "num_tokens": self.num_tokens,
            "num_batches": self.num_batches,
            "seconds": self.seconds,
            "status": self.status,
        }


class Evaluator:
    """Run a bounded zero-gradient evaluation loop over injected steps.

    Args:
        step: Callable returning one :class:`EvalStepResult`. It receives the
            round index and is responsible for fetching its own batch, so the
            evaluator never sees iterators or modality details.
        metrics: Metric instances in configuration order.
        context: Contributor group used for the single global reduction.
        device: Device holding reduction scalars.
        compute_samples: Whether ``sample_count`` is unique per rank and can be
            summed across the group as well.
    """

    def __init__(
            self,
            step: Callable[[int], EvalStepResult],
            metrics: List[Metric],
            context: ReductionContext,
            device: "torch.device",
            *,
            compute_samples: bool = True,
    ) -> None:
        """Store the step, metrics, and reduction context."""
        if not metrics:
            raise ValueError("Evaluator requires at least one metric")
        self._step = step
        self.metrics = list(metrics)
        self.context = context
        self.device = device
        self.compute_samples = compute_samples
        self._num_batches = 0
        self._totals: Dict[str, LossStatistics] = {}

    def reset(self) -> None:
        """Reset every metric, the run counters, and any resettable step."""
        for metric in self.metrics:
            metric.reset()
        self._num_batches = 0
        self._totals = {}
        # A step that owns iteration state (a DataLoader cursor) can rewind it,
        # so repeated ``run`` calls evaluate the same range instead of an
        # already-exhausted source.
        step_reset = getattr(self._step, "reset", None)
        if callable(step_reset):
            step_reset()

    def _synchronize_metrics(self, sample_total: torch.Tensor) -> float:
        """Share the built-in metrics' reduction and return the global sample count."""
        # A setter alone is insufficient: subclasses may accumulate transformed
        # statistics or require their own synchronization before compute().
        shared_metrics = [
            metric for metric in self.metrics
            if metric.supports_shared_reduction
        ]
        reduced = None
        if shared_metrics:
            reduced = reduce_statistics(
                [self._totals], self.context, self.device,
                sample_count=sample_total if self.compute_samples else None,
            )
        for metric in self.metrics:
            if metric in shared_metrics:
                metric.set_reduced_statistics(reduced)
            else:
                metric.synchronize(self.context, self.device)

        if self.compute_samples and reduced is not None:
            num_samples = reduced.num_samples
        else:
            if self.compute_samples:
                sample_total = reduce_scalar(sample_total, self.context)
            num_samples = float(sample_total.cpu())

        return num_samples

    def _compute_metrics(self) -> tuple[Dict[str, float], str, float]:
        """Read synchronized values while representing perplexity overflow as status."""
        metrics_dict: Dict[str, float] = {}
        status = "ok"
        num_tokens = 0.0
        for metric in self.metrics:
            for name, value in metric.compute().items():
                if isinstance(value, float) and value == float("inf"):
                    # JSON has no infinity literal; record the overflow and keep
                    # the finite NLL the loss metric already reported.
                    status = "perplexity_overflow"
                    continue
                metrics_dict[name] = value
            num_tokens = max(num_tokens, metric.num_tokens)

        return metrics_dict, status, num_tokens

    def run(self, max_batches: Optional[int] = None) -> EvaluationReport:
        """Execute up to ``max_batches`` rounds, then reduce once.

        Args:
            max_batches: Round limit for this rank; ``None`` runs until the
                injected step reports an empty round.

        Returns:
            The evaluation report.

        Raises:
            RuntimeError: The configured metrics produce no usable value,
                because no real sample or valid token was seen.
        """
        self.reset()
        started = time.perf_counter()

        sample_total = torch.zeros((), dtype=torch.int64, device=self.device)
        round_index = 0
        while max_batches is None or round_index < max_batches:
            result = self._step(round_index)
            round_index += 1
            if result is None:
                break
            self._num_batches += 1
            if result.sample_count is not None:
                sample_total = sample_total + result.sample_count.detach().to(device=self.device, dtype=torch.int64)
            accumulate_statistics(self._totals, result.statistics)
            for metric in self.metrics:
                metric.update(result.statistics)
            if not result.statistics and (
                    result.sample_count is None or not bool(result.sample_count.any())):
                # A round with neither statistics nor samples means the source
                # is exhausted; stop rather than spinning on empty rounds.
                break

        num_samples = self._synchronize_metrics(sample_total)
        metrics_dict, status, num_tokens = self._compute_metrics()

        if num_samples == 0 and not metrics_dict:
            raise RuntimeError(
                "evaluation consumed no real samples; check the configured data "
                "range, split, and batch plan"
            )

        seconds = time.perf_counter() - started
        return EvaluationReport(
            metrics=metrics_dict,
            num_samples=int(num_samples),
            num_tokens=num_tokens,
            num_batches=self._num_batches,
            seconds=seconds,
            status=status,
        )
