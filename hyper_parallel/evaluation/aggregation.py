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
"""Statistics aggregation for standalone evaluation.

Aggregation happens once, after the last batch, over device scalars: each rank
sums its own numerator/denominator pair to a single device value, the pair is
reduced over the contributor group, and only the final scalars become Python
numbers. No per-batch host synchronization and no per-batch global reduction
is performed, because both change the cost of the loop without changing the
result.

The module is a leaf: it depends on ``torch`` and the process group handed to
it, never on ``hyper_parallel.trainer``.
"""

from __future__ import annotations

__all__ = ["ReducedStatistics", "reduce_scalar", "reduce_statistics"]

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import torch  # pylint: disable=forbidden-backend-import


from hyper_parallel.evaluation.types import KIND_LM_NLL, LossStatistics, ReductionContext


@dataclass
class ReducedStatistics:
    """Globally reduced view of the accumulated statistics.

    Args:
        num_tokens: Global denominator of the token-weighted statistic named
            ``foundation``, or 0 when there is none. Reported as
            ``running_ratio`` metadata, not as a divisor for other statistics.
        kinds: Statistic name to its declared kind, for deriving PPL.
        numerators: Global numerator per statistic name.
        denominators: Global denominator per statistic name. Each statistic
            keeps its own weight: a run that reports an auxiliary objective over
            a different target set must not be divided by the language-model
            token count.
    """

    numerators: Dict[str, float]
    num_tokens: float
    kinds: Dict[str, str]
    denominators: Dict[str, float]

    def mean(self, name: str) -> float | None:
        """Return the global weighted mean for one statistic.

        Args:
            name: Statistic name.

        Returns:
            Numerator divided by that statistic's own global denominator, or
            ``None`` when it has no valid contribution.
        """
        denominator = self.denominators.get(name, 0.0)
        numerator = self.numerators.get(name)
        if numerator is None or denominator <= 0:
            return None
        return numerator / denominator

    def is_lm_nll(self, name: str) -> bool:
        """Whether a statistic may be exponentiated into a perplexity."""
        return self.kinds.get(name) == KIND_LM_NLL


def _to_device_scalar(value: "torch.Tensor | float | int", device: "torch.device") -> "torch.Tensor":
    """Normalize one accumulator contribution to an FP64 device scalar.

    FP64 accumulation is cheap here -- one scalar per statistic -- and removes
    the objection that a long evaluation drifts because of FP32 summing.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float64)
    return torch.tensor(float(value), dtype=torch.float64, device=device)


def _split_fp64(value: "torch.Tensor") -> "torch.Tensor":
    """Split FP64 scalars into an exact ``(hi, lo)`` FP32 pair.

    HCCL collectives do not support FP64, so a device sum cannot be reduced in
    its accumulation dtype. ``hi`` is the FP32 rounding of the value and ``lo``
    the FP32 rounding of the remainder: together they carry about 48 mantissa
    bits, and summing each half separately across a few thousand ranks keeps the
    error far below the FP32 epsilon of the result.
    """
    hi = value.to(torch.float32)
    lo = (value - hi.to(torch.float64)).to(torch.float32)
    return torch.stack([hi, lo])


def reduce_scalar(value: "torch.Tensor", context: ReductionContext) -> "torch.Tensor":
    """Sum one FP64 device scalar over the contributor group when distributed.

    The accumulation stays FP64 on every rank; only the wire format is FP32
    pairs, so the collective works on backends that reject FP64 (HCCL).
    """
    if not context.distributed:
        return value
    pair = _split_fp64(value)
    torch.distributed.all_reduce(pair, op=torch.distributed.ReduceOp.SUM, group=context.group)
    return pair[0].to(torch.float64) + pair[1].to(torch.float64)


def reduce_statistics(
        contributions: Iterable[Mapping[str, LossStatistics]],
        context: ReductionContext,
        device: "torch.device",
) -> ReducedStatistics:
    """Sum per-batch statistics and reduce them over the contributor group.

    Args:
        contributions: Statistics from every completed batch, in order.
        context: Contributor group and its size.
        device: Device that holds the scalars; every rank must use the same
            device kind.

    Returns:
        The globally reduced statistics.

    Raises:
        ValueError: One statistic reports a negative denominator, which means
            the step counted weights it did not actually add.
    """
    numerators: Dict[str, torch.Tensor] = {}
    denominators: Dict[str, torch.Tensor] = {}
    kinds: Dict[str, str] = {}

    zero = torch.zeros((), dtype=torch.float64, device=device)
    for contribution in contributions:
        for name, stat in contribution.items():
            declared = kinds.setdefault(name, stat.kind)
            if declared != stat.kind:
                raise ValueError(
                    f"statistic {name!r} changed kind between batches: "
                    f"{declared!r} -> {stat.kind!r}"
                )
            numerators[name] = numerators.get(name, zero.clone()) + _to_device_scalar(stat.numerator, device)
            denominators[name] = (
                denominators.get(name, zero.clone()) + _to_device_scalar(stat.denominator, device)
            )

    reduced_numerators: Dict[str, float] = {}
    reduced_denominators: Dict[str, float] = {}
    for name, numerator in numerators.items():
        denominator = denominators[name]
        if float(denominator.cpu()) < 0.0:
            raise ValueError(
                f"statistic {name!r} accumulated a negative denominator "
                f"({float(denominator.cpu())}); its step must not subtract weight"
            )
        global_numerator = reduce_scalar(numerator, context)
        global_denominator = reduce_scalar(denominator, context)
        reduced_numerators[name] = float(global_numerator.cpu())
        reduced_denominators[name] = float(global_denominator.cpu())

    # The language-model token count is reported as run metadata. It is not a
    # divisor for other statistics: an auxiliary objective over a different
    # target set keeps its own denominator.
    reduced_tokens = reduced_denominators.get("foundation", 0.0)

    return ReducedStatistics(
        numerators=reduced_numerators,
        num_tokens=reduced_tokens,
        kinds=kinds,
        denominators=reduced_denominators,
    )
