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
"""Format balancing statistics without inspecting tensors or communicating."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


def _layout_lines(label: str, ranks: Sequence[int], layout: Sequence[Any], bin_label: str = "mb") -> list[str]:
    """Format per-rank pack lengths, sample counts, costs and optional padding."""
    lines = [f"  {label}:"]
    for rank, bins in zip(ranks, layout):
        microbatches = []
        for index, stats in enumerate(bins):
            microbatches.append(
                f"{bin_label}{index}: seq_len={stats['seq_len']}, samples={stats['samples']}, cost={stats['cost']:.12g}"
            )
            if "padding" in stats:
                microbatches[-1] += f", padding={stats['padding']}"
        lines.append(f"  dp{rank}: " + " | ".join(microbatches))
    return lines


def _placement_lines(stats: dict[str, Any], ranks: Sequence[int], bin_label: str = "mb") -> list[str]:
    lines = []
    before, after = stats.get("bins_before"), stats.get("bins_after")
    if before is not None:
        lines.extend(_layout_lines("before balance", ranks, before, bin_label))
    for rank, sent, received in zip(ranks, stats.get("send_samples", ()), stats.get("recv_samples", ())):
        lines.append(f"dp{rank}: send {sent} samples, recv {received} samples")
    if after is not None:
        lines.extend(_layout_lines("after balance", ranks, after, bin_label))
    return lines


def format_balance_stats(stats: dict[str, Any], step: int) -> str:
    """Format original/accepted microbatch layouts with raw predicted LLM costs.

    Args:
        stats: Sample counts, sequence lengths, predicted costs and transfers
            produced by a balancing loader. Native windows include per-round layouts.
        step: One-based iteration number, within the epoch for native loaders.

    Returns:
        One multiline message. Costs retain the planner's units; they are not milliseconds.
    """
    lines = [f"[dp-balance] iteration {step} :"]
    if "rounds" in stats:
        rounds = stats["rounds"]
        lines.append(
            f"  epoch={stats['epoch']}, micro_batches={rounds[0]['micro_batch']}-{rounds[-1]['micro_batch']}, "
            f"planner_rank={stats['planner_rank']}, constructor_ranks={stats['constructor_ranks']}"
        )
    if "relative_gain" in stats:
        lines.append(
            f"  cost_component=llm, objective={stats['objective']}, original={stats['original_objective']:.12g}, "
            f"candidate={stats['candidate_objective']:.12g}, relative_gain={stats['relative_gain']:.12g}, "
            f"min_balance_gain={stats['min_balance_gain']:.12g}, accepted={stats['accepted']}"
        )
        if "accepted_objective" in stats:
            lines.append(f"  accepted_objective={stats['accepted_objective']:.12g}")
    ranks = stats["group_ranks"]
    if "rounds" in stats:
        for round_stats in stats["rounds"]:
            lines.append(f"  micro_batch {round_stats['micro_batch']}:")
            lines.extend(_placement_lines(round_stats, ranks, bin_label="pack"))
    else:
        lines.extend(_placement_lines(stats, ranks))
    return "\n".join(lines)


def log_balance_stats(stats: dict[str, Any], step: int) -> None:
    """Default log callback; the loader invokes it only on global rank zero.

    Args:
        stats: Before/after costs, bin counters and transfer counts.
        step: One-based delivered step number.
    """
    logger.info("%s", format_balance_stats(stats, step))


__all__ = ["format_balance_stats", "log_balance_stats"]
