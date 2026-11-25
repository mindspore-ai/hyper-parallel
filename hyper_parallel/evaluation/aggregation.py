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

__all__ = ["ReducedStatistics", "accumulate_statistics", "accumulation_dtype", "reduce_scalar", "reduce_statistics"]

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import torch
import torch.distributed as dist


from hyper_parallel.evaluation.types import LossStatistics, ReductionContext


@dataclass
class ReducedStatistics:
    """Globally reduced view of the accumulated statistics.

    Args:
        num_tokens: Global denominator of the token-weighted statistic named
            ``foundation``, or 0 when there is none. Reported as
            ``running_ratio`` metadata, not as a divisor for other statistics.
        nll_flags: Statistic name to its boolean NLL declaration, for deriving PPL.
        numerators: Global numerator per statistic name.
        denominators: Global denominator per statistic name. Each statistic
            keeps its own weight: a run that reports an auxiliary objective over
            a different target set must not be divided by the language-model
            token count.
    """

    numerators: Dict[str, float]
    num_tokens: float
    nll_flags: Dict[str, bool]
    denominators: Dict[str, float]
    num_samples: float = 0.0

    def __post_init__(self) -> None:
        """Reject non-boolean declarations before a metric can consume them."""
        for name, flag in self.nll_flags.items():
            if not isinstance(flag, bool):
                raise ValueError(f"statistic {name!r} is_lm_nll must be bool, got {flag!r}")

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
        """Whether a statistic may be exponentiated into a perplexity.

        Args:
            name: Statistic name whose language-model NLL declaration is queried.

        Returns:
            The statistic's declaration, or ``False`` when the name is absent.
        """
        return self.nll_flags.get(name, False)


@dataclass
class _AccumulatedStatistics(LossStatistics):
    """Internal compensated totals without changing the step's scalar contract."""

    numerator_residual: "torch.Tensor | None" = None
    denominator_residual: "torch.Tensor | None" = None


def accumulation_dtype(device: "torch.device") -> "torch.dtype":
    """Choose the scalar dtype used by the accumulation helpers.

    Args:
        device: Device on which the statistics are accumulated.

    Returns:
        FP32 on Ascend NPU, where callers retain compensation separately;
        FP64 on other devices.
    """
    return torch.float32 if device.type == "npu" else torch.float64


def _to_device_scalar(value: "torch.Tensor | float | int", device: "torch.device") -> "torch.Tensor":
    """Normalize one contribution without requesting unsupported NPU FP64."""
    dtype = accumulation_dtype(device)
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype)
    return torch.tensor(float(value), dtype=dtype, device=device)


def _compensated_add(
        total: torch.Tensor, residual: torch.Tensor, value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add one FP32 contribution with Neumaier's magnitude-aware correction."""
    updated = total + value
    error = torch.where(total.abs() >= value.abs(), (total - updated) + value, (value - updated) + total)
    return updated, residual + error


def _accumulate_component(
        previous: "LossStatistics | None", contribution: LossStatistics,
        component: str, device: torch.device,
) -> tuple[torch.Tensor, "torch.Tensor | None"]:
    """Merge one scalar and any compensation retained by a previous accumulation."""
    source = getattr(contribution, component).detach().to(device=device)
    value = _to_device_scalar(source, device)
    residual_name = f"{component}_residual"
    incoming_residual = getattr(contribution, residual_name, None)
    total = torch.zeros_like(value) if previous is None else _to_device_scalar(getattr(previous, component), device)
    residual = getattr(previous, residual_name, None)
    if value.dtype == torch.float64:
        total = total + value
        for correction in (residual, incoming_residual):
            if correction is not None:
                total = total + _to_device_scalar(correction, device)
        return total, None
    residual = torch.zeros_like(value) if residual is None else _to_device_scalar(residual, device)
    total, residual = _compensated_add(total, residual, value)
    if not source.is_floating_point():
        correction = (source - value.to(source.dtype)).to(value.dtype)
        total, residual = _compensated_add(total, residual, correction)
    if incoming_residual is not None:
        total, residual = _compensated_add(total, residual, _to_device_scalar(incoming_residual, device))
    return total, residual


def accumulate_statistics(
        totals: Dict[str, LossStatistics], contribution: Mapping[str, LossStatistics],
        *, device: "torch.device | None" = None,
) -> None:
    """Accumulate detached scalars in bounded memory, retaining FP32 roundoff.

    Args:
        totals: Mutable statistic-name mapping updated with the accumulated sums.
        contribution: Additive numerator and denominator values for this update,
            including any compensation retained by earlier accumulation.
        device: Device for the totals. When omitted, each statistic uses the
            device of its incoming numerator.
    """
    for name, stat in contribution.items():
        previous = totals.get(name)
        if previous is not None and previous.is_lm_nll != stat.is_lm_nll:
            raise ValueError(f"statistic {name!r} changed is_lm_nll between batches")
        target = stat.numerator.device if device is None else device
        numerator, numerator_residual = _accumulate_component(previous, stat, "numerator", target)
        denominator, denominator_residual = _accumulate_component(previous, stat, "denominator", target)
        totals[name] = _AccumulatedStatistics(
            numerator, denominator, stat.is_lm_nll, numerator_residual, denominator_residual,
        )


def _split_scalar(value: torch.Tensor, residual: "torch.Tensor | None" = None) -> torch.Tensor:
    """Keep native FP64, compensated FP32, or integer totals in two FP32 channels."""
    hi = value.to(torch.float32)
    lo = (value - hi.to(value.dtype)).to(torch.float32)
    if residual is not None:
        lo = lo + residual.to(torch.float32)
    return torch.stack([hi, lo])


def reduce_scalar(value: "torch.Tensor", context: ReductionContext) -> "torch.Tensor":
    """Sum a device scalar and return a CPU FP64 value without losing NPU residuals.

    Args:
        value: Local scalar tensor to sum across the reduction group.
        context: Process group and world size; a local context needs no collective.

    Returns:
        Global sum as a CPU FP64 scalar tensor.
    """
    if not context.distributed:
        return value.cpu().to(torch.float64)
    pair = _split_scalar(value)
    dist.all_reduce(pair, op=dist.ReduceOp.SUM, group=context.group)
    return pair.cpu().to(torch.float64).sum(dim=0)


def _reduce_values(
        values: list[torch.Tensor], residuals: list[torch.Tensor], context: ReductionContext,
) -> list[float]:
    """Validate totals, reduce once when needed, and reconstruct on CPU in FP64."""
    # Validate before any wire conversion: a small negative FP64 denominator
    # could otherwise underflow to zero, hiding an invalid contribution.
    pairs = [
        torch.stack([value.cpu().to(torch.float64), residual.cpu().to(torch.float64)])
        for value, residual in zip(values, residuals)
    ]
    native = torch.stack(pairs, dim=1)
    invalid = bool((native.sum(dim=0)[1:-1:2] < 0).any()) or not bool(torch.isfinite(native).all())
    if context.distributed:
        pairs = [_split_scalar(value, residual) for value, residual in zip(values, residuals)]
        payload = torch.stack(pairs, dim=1)
        invalid = invalid or not bool(torch.isfinite(payload).all())
    else:
        # Native FP64 totals can exceed the FP32 wire range. A local run needs
        # neither that conversion nor its range restriction.
        payload = native
    errors = [invalid]
    if context.distributed:
        errors = [None] * context.world_size
        dist.all_gather_object(errors, invalid, group=context.group)
    if any(errors):
        raise ValueError("evaluation statistics contain a negative denominator or non-finite value")
    if context.distributed:
        dist.all_reduce(payload, op=dist.ReduceOp.SUM, group=context.group)
    return payload.cpu().to(torch.float64).sum(dim=0).tolist()


def reduce_statistics(
        contributions: Iterable[Mapping[str, LossStatistics]],
        context: ReductionContext,
        device: "torch.device",
        sample_count: "torch.Tensor | None" = None,
) -> ReducedStatistics:
    """Sum per-batch statistics and reduce them over the contributor group.

    Args:
        contributions: Statistics from every completed batch, in order.
        context: Contributor group and its size.
        device: Device that holds the scalars; every rank must use the same
            device kind.
        sample_count: Optional sample count packed into the same final reduction.

    Returns:
        The globally reduced statistics.

    Raises:
        ValueError: One statistic reports a negative denominator, which means
            the step counted weights it did not actually add.
    """
    totals: Dict[str, LossStatistics] = {}
    for contribution in contributions:
        accumulate_statistics(totals, contribution, device=device)
    nll_flags = {name: stat.is_lm_nll for name, stat in totals.items()}

    # Empty lanes still participate with zeros, and every lane uses the same
    # ordered statistic schema before entering the numerical collective.
    schemas = [nll_flags]
    if context.distributed:
        schemas = [None] * context.world_size
        dist.all_gather_object(schemas, nll_flags, group=context.group)
    global_nll_flags = {}
    for schema in schemas:
        for name, flag in schema.items():
            previous = global_nll_flags.setdefault(name, flag)
            if previous != flag:
                raise ValueError(f"statistic {name!r} has inconsistent is_lm_nll across ranks")
    names = sorted(global_nll_flags)
    values = []
    residuals = []
    zero = torch.zeros((), dtype=accumulation_dtype(device), device=device)
    for name in names:
        stat = totals.get(name, LossStatistics(zero, zero, global_nll_flags[name]))
        for component in ("numerator", "denominator"):
            values.append(getattr(stat, component))
            residual = getattr(stat, f"{component}_residual", None)
            residuals.append(zero if residual is None else residual)
    values.append(zero if sample_count is None else sample_count.to(device=device))
    residuals.append(zero)
    # Counts remain int64 until split: converting them to an NPU FP32 scalar
    # first would already lose odd counts above 2**24.
    payload = _reduce_values(values, residuals, context)
    reduced_numerators = {name: payload[2 * index] for index, name in enumerate(names)}
    reduced_denominators = {name: payload[2 * index + 1] for index, name in enumerate(names)}
    # The language-model token count is reported as run metadata. It is not a
    # divisor for other statistics: an auxiliary objective over a different
    # target set keeps its own denominator.
    reduced_tokens = reduced_denominators.get("foundation", 0.0)

    return ReducedStatistics(
        numerators=reduced_numerators,
        num_tokens=reduced_tokens,
        nll_flags=global_nll_flags,
        denominators=reduced_denominators,
        num_samples=payload[-1],
    )
