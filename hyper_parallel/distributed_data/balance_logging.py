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
"""Format local balancing statistics without inspecting tensors or communicating."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


def _layout_lines(label: str, ranks: Sequence[int], layout: Sequence[Any]) -> list[str]:
    lines = [f"  {label}:"]
    for rank, bins in zip(ranks, layout):
        microbatches = []
        for index, stats in enumerate(bins):
            fields = []
            displayed = {"seq_len", "samples", "cost"}
            if all(name in stats for name in ("vae_gen", "vae_cond", "vit")):
                fields.append(
                    f"images(vae_gen={stats['vae_gen']}, vae_cond={stats['vae_cond']}, vit={stats['vit']})"
                )
                displayed.update(("vae_gen", "vae_cond", "vit"))
            sequence = f"seq_len={stats['seq_len']}"
            if "P" in stats and "D" in stats:
                sequence += f" (P={stats['P']}, D={stats['D']})"
                displayed.update(("P", "D"))
            fields.extend((sequence, f"samples={stats['samples']}"))
            if "pixel_numel" in stats:
                fields.append(f"pixel_numel={stats['pixel_numel']}")
                displayed.add("pixel_numel")
            fields.append(f"cost={stats['cost']:.12g}")
            fields.extend(f"{name}={value}" for name, value in stats.items() if name not in displayed)
            microbatches.append(f"mb{index}: " + ", ".join(fields))
        lines.append(f"  dp{rank}: " + " | ".join(microbatches))
    return lines


def format_balance_stats(stats: dict[str, Any], step: int, max_steps: int | None = None) -> str:
    """Format original/accepted microbatch layouts with raw predicted LLM costs.

    Args:
        stats: Statistics produced by the local balancing loader. Image counts,
            P/D and pixel elements are optional application ``bin_stats_fn`` fields.
        step: One-based iteration number.
        max_steps: Optional iteration limit, displayed as ``iteration step/total``.

    Returns:
        One multiline message. Costs retain the planner's units; they are not milliseconds.
    """
    iteration = str(step) if max_steps is None else f"{step}/{max_steps}"
    lines = [f"[dp-balance] iteration {iteration} :"]
    if "relative_gain" in stats:
        lines.append(
            f"  cost_component=llm, objective={stats['objective']}, original={stats['original_objective']:.12g}, "
            f"candidate={stats['candidate_objective']:.12g}, relative_gain={stats['relative_gain']:.12g}, "
            f"min_balance_gain={stats['min_balance_gain']:.12g}, accepted={stats['accepted']}"
        )
    ranks = stats["group_ranks"]
    before, after = stats.get("bins_before"), stats.get("bins_after")
    if before is not None:
        lines.extend(_layout_lines("before balance", ranks, before))
    for rank, sent, received in zip(ranks, stats["send_samples"], stats["recv_samples"]):
        lines.append(f"dp{rank}: send {sent} samples, recv {received} samples")
    if after is not None:
        lines.extend(_layout_lines("after balance", ranks, after))
    return "\n".join(lines)


def log_balance_stats(stats: dict[str, Any], step: int, max_steps: int | None = None) -> None:
    """Default log callback; the loader invokes it only on global rank zero.

    Args:
        stats: Before/after costs, bin counters and transfer counts.
        step: One-based delivered step number.
        max_steps: Optional training step limit.
    """
    logger.info("%s", format_balance_stats(stats, step, max_steps))


__all__ = ["format_balance_stats", "log_balance_stats"]
