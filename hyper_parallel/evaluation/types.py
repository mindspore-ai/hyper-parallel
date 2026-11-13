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
"""Evaluation value types shared by steps, aggregation, metrics, and results.

The types are deliberately plain: a step reports *additive statistics* (a
numerator and a denominator with a declared meaning), never a per-batch mean.
Means and derived values are computed once, after every contribution has been
reduced, so the result cannot depend on how work was split across ranks.
"""

from __future__ import annotations

__all__ = [
    "EvalStepResult",
    "LossStatistics",
    "ReductionContext",
]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    # Type-checking import only: ``evaluation`` is a leaf package and must not
    # import everything ``torch`` drags in at module import time.
    import torch  # pylint: disable=forbidden-backend-import


#: A statistic that is a genuine causal-LM negative log likelihood and can be
#: exponentiated into a perplexity.
KIND_LM_NLL = "lm_nll"
#: Any other objective term (aux loss, MTP, regularizers). Never feeds PPL.
KIND_OBJECTIVE = "objective"


@dataclass
class LossStatistics:
    """One additive, named loss statistic.

    Args:
        numerator: Device scalar with this contribution's loss total.
        denominator: Device non-negative scalar with the weight matching
            ``numerator`` -- the valid token count for token-weighted losses.
        kind: ``lm_nll`` or ``objective``. Only ``lm_nll`` may produce a
            perplexity.
    """

    numerator: "torch.Tensor"
    denominator: "torch.Tensor"
    kind: str = KIND_OBJECTIVE

    def __post_init__(self) -> None:
        """Reject an undeclared statistic kind at construction time."""
        if self.kind not in (KIND_LM_NLL, KIND_OBJECTIVE):
            raise ValueError(
                f"LossStatistics.kind must be {KIND_LM_NLL!r} or {KIND_OBJECTIVE!r}, "
                f"got {self.kind!r}"
            )


@dataclass
class EvalStepResult:
    """Additive outcome of one evaluation forward.

    Args:
        statistics: Named addable statistics produced by this step.
        sample_count: Device scalar with the real (non-dummy) samples this rank
            consumed. It stays on device so a long evaluation does not
            synchronize once per batch; the evaluator reduces it at the end.
            Dummy rows that exist only to align collective rounds contribute 0.
    """

    statistics: Dict[str, LossStatistics] = field(default_factory=dict)
    sample_count: "torch.Tensor | None" = None


@dataclass
class ReductionContext:
    """Where statistic contributions are unique and how to reduce them.

    P0 only builds a data-parallel context: every rank contributes disjoint
    tokens, so a sum over the DP group is exact. TP/CP/EP bring replicated or
    partially-replicated statistics and need their own normalization before
    this point; the run-mode capability check rejects them until then.

    Args:
        group: Process group covering the disjoint contributors, or ``None``
            when there is nothing to reduce (single process).
        world_size: Number of contributors in ``group``.
    """

    group: Optional[Any] = None
    world_size: int = 1

    @property
    def distributed(self) -> bool:
        """Whether a collective is required."""
        return self.group is not None and self.world_size > 1
