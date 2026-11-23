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
"""Metrics accumulate additive statistics and read a shared reduced result.

The lifecycle is ``reset -> update -> synchronize -> compute``. ``update``
accumulates detached scalars in bounded memory, ``synchronize`` does
the single global reduction (through :mod:`hyper_parallel.evaluation.aggregation`),
and ``compute`` is a pure read that may be called repeatedly without triggering
a second collective.

Splitting it this way keeps the expensive, order-dependent work in one place:
the evaluator decides when the one reduction happens, and a metric that is
queried twice (logging plus output file) cannot silently double-count.
"""

from __future__ import annotations

__all__ = [
    "LossMetric",
    "Metric",
    "PerplexityMetric",
    "build_metrics",
]

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional

from hyper_parallel.evaluation.aggregation import ReducedStatistics, accumulate_statistics, reduce_statistics
from hyper_parallel.evaluation.types import KIND_LM_NLL, LossStatistics, ReductionContext

if TYPE_CHECKING:
    import torch


class Metric(ABC):
    """Base class for one named evaluation metric."""

    @abstractmethod
    def reset(self) -> None:
        """Drop accumulated contributions so the instance can be reused."""

    @abstractmethod
    def update(self, statistics: Mapping[str, LossStatistics]) -> None:
        """Record one batch's additive statistics.

        Args:
            statistics: Named additive statistics from one evaluation step.
        """

    @abstractmethod
    def synchronize(self, context: ReductionContext, device: "torch.device") -> None:
        """Perform the single global reduction for this run.

        Args:
            context: Contributor group and its size.
            device: Device that holds any reduction scalars.
        """

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Return this metric's flat output, without any collective.

        Returns:
            Metric name to value.
        """

    @property
    def supports_shared_reduction(self) -> bool:
        """Whether the Evaluator may supply raw globally reduced statistics.

        Opting in requires ``set_reduced_statistics`` and promises that the
        metric needs no custom accumulated state or synchronization hook.
        """
        return False

    @property
    def num_tokens(self) -> float:
        """Global valid-token denominator seen by this metric, or 0 before reduce."""
        return 0.0


class LossMetric(Metric):
    """Token-weighted global mean of one named statistic.

    Args:
        statistic: Statistic name produced by the evaluation step, for example
            ``foundation`` for the language-model objective.
        name: Output key; defaults to ``"loss"`` so a single-statistic run
            reports an unsurprising field.
    """

    def __init__(self, statistic: str = "foundation", name: str = "loss") -> None:
        """Record the statistic this metric reads and its output key."""
        self.statistic = statistic
        self.name = name
        self._totals: Dict[str, LossStatistics] = {}
        self._reduced: Optional[ReducedStatistics] = None

    def reset(self) -> None:
        """Drop the accumulated contributions and any reduced view."""
        self._totals = {}
        self._reduced = None

    def update(self, statistics: Mapping[str, LossStatistics]) -> None:
        """Accumulate one batch's additive statistics."""
        if self.statistic in statistics:
            accumulate_statistics(self._totals, statistics)

    def synchronize(self, context: ReductionContext, device: "torch.device") -> None:
        """Reduce the recorded contributions exactly once."""
        self._reduced = reduce_statistics([self._totals], context, device)

    def set_reduced_statistics(self, statistics: ReducedStatistics) -> None:
        """Consume the Evaluator shared reduction without issuing another collective."""
        self._reduced = statistics

    @property
    def supports_shared_reduction(self) -> bool:
        """Share raw statistics only while the accumulation lifecycle is unchanged.

        Subclasses overriding ``update`` or ``synchronize`` retain their own
        lifecycle. They can explicitly override this property if consuming the
        shared raw statistics is sufficient instead.
        """
        return (
            getattr(self.update, "__func__", None) is LossMetric.update
            and getattr(self.synchronize, "__func__", None) is LossMetric.synchronize
        )

    @property
    def num_tokens(self) -> float:
        """Global valid-token denominator, or 0 before the reduce runs."""
        return 0.0 if self._reduced is None else self._reduced.num_tokens

    def compute(self) -> Dict[str, float]:
        """Return the global weighted mean, or an empty mapping when empty."""
        if self._reduced is None:
            raise RuntimeError(
                f"{type(self).__name__}.compute() requires a prior synchronize(); "
                "the metric lifecycle is reset -> update -> synchronize -> compute"
            )
        mean = self._reduced.mean(self.statistic)
        return {} if mean is None else {self.name: mean}


class PerplexityMetric(LossMetric):
    """Perplexity derived from a genuine language-model NLL statistic.

    Perplexity is computed from the same reduced NLL the loss metric reports;
    it never accumulates or reduces a second time. A statistic that does not
    declare ``kind="lm_nll"`` is rejected, because exponentiating an objective
    that also contains auxiliary terms produces a number without meaning.

    Args:
        statistic: NLL statistic name.
        name: Output key; defaults to ``"perplexity"``.
    """

    def __init__(self, statistic: str = "foundation", name: str = "perplexity") -> None:
        """Record the NLL statistic and the perplexity output key."""
        super().__init__(statistic=statistic, name=name)

    def compute(self) -> Dict[str, float]:
        """Return ``exp(NLL)``, flagging exponent overflow instead of emitting inf."""
        if self._reduced is None:
            raise RuntimeError(
                f"{type(self).__name__}.compute() requires a prior synchronize(); "
                "the metric lifecycle is reset -> update -> synchronize -> compute"
            )
        if self.statistic not in self._reduced.kinds:
            return {}
        if not self._reduced.is_lm_nll(self.statistic):
            raise ValueError(
                f"perplexity requires a statistic declared kind={KIND_LM_NLL!r}; "
                f"statistic {self.statistic!r} is not one. Configure the loss adapter "
                "to declare a pure NLL statistic, or drop the perplexity metric"
            )
        mean = self._reduced.mean(self.statistic)
        if mean is None:
            return {}
        try:
            value = math.exp(mean)
        except OverflowError:
            return {self.name: math.inf}
        if not math.isfinite(value):
            return {self.name: math.inf}
        return {self.name: value}


def build_metrics(
        targets: Iterable[Any],
) -> List[Metric]:
    """Instantiate the configured metric targets and reject duplicate keys.

    Args:
        targets: Configured ``Target`` objects, in evaluation order.

    Returns:
        Fresh metric instances.

    Raises:
        TypeError: A target did not build a :class:`Metric`.
        ValueError: Two metrics produce the same output key.
    """
    metrics: List[Metric] = []
    seen: Dict[str, str] = {}
    for target in targets:
        metric = target.build()
        if not isinstance(metric, Metric):
            raise TypeError(
                f"evaluation metric target built {type(metric).__name__}, "
                "which is not an evaluation.Metric"
            )
        name = getattr(metric, "name", type(metric).__name__)
        if name in seen:
            raise ValueError(
                f"evaluation metrics produce a duplicate output key {name!r} "
                f"({seen[name]} and {type(metric).__name__})"
            )
        seen[name] = type(metric).__name__
        metrics.append(metric)
    return metrics
