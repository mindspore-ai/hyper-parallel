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
"""Cold-path result exchange and failure diagnostics for Multicore owners."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.distributed as dist

FailureReport = tuple[str, str]


class CollectiveConvergenceError(RuntimeError):
    """Raised when peers cannot exchange collective lifecycle results."""


def exchange(value: Any, *, group: Any, failure_prefix: str) -> list[Any]:
    """Exchange picklable results, preserving the original communication error as cause.

    Args:
        value: Picklable local result.
        group: Explicit group; None selects WORLD when initialized.
        failure_prefix: Caller-owned operation context for communication failures.

    Returns:
        Results in group rank order, or the local result when distributed is uninitialized.
    """
    try:
        if not dist.is_initialized():
            return [value]
        values = [None] * dist.get_world_size(group)
        dist.all_gather_object(values, value, group=group)
        return values
    except Exception as error:  # pylint: disable=broad-exception-caught
        exception_type, message = failure_report(error)
        summary = message if exception_type == "RuntimeError" else f"{exception_type}: {message}"
        raise CollectiveConvergenceError(f"{failure_prefix}: {summary}") from error


def failure_report(error: Exception, *, message: str | None = None) -> FailureReport:
    """Build the picklable exception type and message for one local failure.

    Args:
        error: Local exception raised by the collective stage.
        message: Optional user-facing replacement for the exception text.
    """
    return type(error).__name__, str(error) if message is None else message


def format_collective_failures(
    failures: list[FailureReport | None],
    *,
    rank_label: str = "rank",
    ranks_label: str = "ranks",
) -> str:
    """Format only failed ranks, grouping peers with identical diagnostics.

    Args:
        failures: Failure report or ``None`` in collective-group rank order.
        rank_label: Label for one failing rank.
        ranks_label: Label for multiple ranks with the same failure.

    Returns:
        Compact failure descriptions in first-rank order.
    """
    grouped: dict[FailureReport, list[int]] = {}
    for rank, failure in enumerate(failures):
        if failure is not None:
            grouped.setdefault(failure, []).append(rank)
    descriptions = []
    for (exception_type, message), ranks in grouped.items():
        diagnostic = message if exception_type == "RuntimeError" else f"{exception_type}: {message}"
        owner = f"{rank_label}={ranks[0]}" if len(ranks) == 1 else f"{ranks_label}={ranks}"
        descriptions.append(f"{owner}: {diagnostic}")
    return "; ".join(descriptions)


def collective_call(
    callback: Callable[[], None],
    *,
    stage: str,
    rank_label: str = "rank",
    ranks_label: str = "ranks",
    group: Any,
) -> None:
    """Run one lifecycle stage and converge local failures across peers.

    Args:
        callback: Operation executed by every rank in the same order.
        stage: User-facing description of that stage.
        rank_label: Label for one failing collective-group rank.
        ranks_label: Label for multiple ranks with the same failure.
        group: Explicit collective group; None selects WORLD when initialized.
    """
    failure = None
    try:
        callback()
    # Peers must see ordinary local errors before entering the next collective.
    except Exception as error:  # pylint: disable=broad-exception-caught
        failure = error
    report = None if failure is None else failure_report(failure)
    failures = exchange(report, group=group, failure_prefix=f"{stage} failed while converging {rank_label} results")
    if any(report is not None for report in failures):
        details = format_collective_failures(failures, rank_label=rank_label, ranks_label=ranks_label)
        raise RuntimeError(f"{stage} failed on {details}") from failure
