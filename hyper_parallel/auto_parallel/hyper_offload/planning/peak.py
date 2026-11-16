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
"""Peak-driven residency planning using the existing trace and action IR."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import accumulate

from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import (
    ResidencyActionType,
    ResidencySchedule,
)
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import ActivationTrace
from hyper_parallel.auto_parallel.hyper_offload.planning._peak_model import (
    EvictionGap,
    PeakProfile,
    build_profile,
)
from hyper_parallel.auto_parallel.hyper_offload.planning.communication_mask import (
    TransferPlacement,
    limit_prefetch_advance,
    limit_release_delay,
    mask_communications,
    release_checkpoints,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreWeights:
    """Weight freed bytes above coverage to avoid many tiny transfer intervals."""

    size: float = 2.0
    cover_start: float = 1.0
    cover_end: float = 1.0
    neighbor_impact: float = 1.0


def _overlap_fraction(gap: EvictionGap, selected: list[EvictionGap]) -> float:
    intervals = sorted(
        (max(gap.first, other.first), min(gap.last, other.last)) for other in selected
    )
    covered, cursor = 0, gap.first
    for start, end in intervals:
        if end > max(cursor, start):
            covered += end - max(cursor, start)
            cursor = end
    return covered / max(1, gap.last - gap.first)


def _choose_gap(
    eligible: list[EvictionGap],
    peak: int,
    selected: list[EvictionGap],
    weights: ScoreWeights,
) -> EvictionGap:
    max_size = max(gap.size_bytes for gap in eligible)
    max_left = max(peak - gap.first for gap in eligible)
    max_right = max(gap.end - peak for gap in eligible)

    def score(gap: EvictionGap) -> tuple:
        """Rank candidates with deterministic size and position tie breaks."""
        value = (
            weights.size * gap.size_bytes / max_size
            + weights.cover_start * (peak - gap.first) / max_left
            + weights.cover_end * (gap.end - peak) / max_right
            - weights.neighbor_impact * _overlap_fraction(gap, selected)
        )
        return value, gap.size_bytes, gap.end - gap.start, gap.start, -gap.storage_id

    return max(eligible, key=score)


def _select_gaps(
    resident: list[int], gaps: list[EvictionGap], limit: int, weights: ScoreWeights,
    preferred: set[tuple[int, int, int]] | None = None,
) -> list[EvictionGap]:
    remaining, selected = list(gaps), []
    delta = [0] * (len(resident) + 1)
    for gap in gaps:
        delta[gap.start + 1] += 1
        delta[gap.end] -= 1
    covering = list(accumulate(delta[:-1]))
    while remaining:
        # An unavoidable operator peak must not hide other reducible peaks.
        reducible = (
            slot
            for slot, count in enumerate(covering)
            if count and resident[slot] > limit
        )
        peak = max(reducible, key=lambda slot: (resident[slot], slot), default=None)
        if peak is None:
            break
        eligible = [gap for gap in remaining if gap.start < peak < gap.end]
        native = [
            gap for gap in eligible
            if preferred and (gap.storage_id, gap.start, gap.end) in preferred
        ]
        chosen = _choose_gap(native or eligible, peak, selected, weights)
        selected.append(chosen)
        remaining.remove(chosen)
        for slot in range(chosen.start + 1, chosen.end):
            resident[slot] -= chosen.size_bytes
            covering[slot] -= 1
    return selected


def _emit_schedule(
    placements: list[TransferPlacement], last_uses: dict[int, int], retained: set[int],
    checkpoint_ops: set[int],
) -> ResidencySchedule:
    schedule = ResidencySchedule()
    for placement in sorted(
        placements,
        key=lambda item: (item.prefetch_before, item.gap.end, item.gap.storage_id),
    ):
        sid = placement.gap.storage_id
        schedule.add_pre(placement.prefetch_before, ResidencyActionType.COPY_H2D, sid)
        # Retire the old snapshot before a consumer may mutate this storage.
        # The existing pool defers reuse until the H2D completion event.
        schedule.add_pre(
            placement.prefetch_before, ResidencyActionType.RELEASE_HOST, sid
        )
    posts = []
    for placement in placements:
        sid = placement.gap.storage_id
        posts.append((placement.copy_after, 0, sid, ResidencyActionType.COPY_D2H))
        deadline = (
            placement.release_after
            if placement.release_deadline is None
            else placement.release_deadline
        )
        # A hard fence on an unrelated wrapper-only op expands management.
        # An earlier native boundary is safe, but never cross the last use.
        while deadline > placement.gap.start and deadline not in checkpoint_ops:
            deadline -= 1
        posts.extend(
            (op, 1, sid, ResidencyActionType.TRY_RELEASE_DEVICE)
            for op in release_checkpoints(placement)
            if op in checkpoint_ops and op < deadline
        )
        posts.append(
            (deadline, 1, sid, ResidencyActionType.RELEASE_DEVICE)
        )
    posts.extend(
        (op, 1, sid, ResidencyActionType.RELEASE_DEVICE)
        for sid, op in last_uses.items()
        if sid not in retained
    )
    for op, _, sid, kind in sorted(posts):
        schedule.add_post(op, kind, sid)
    return schedule


class PeakResidencyPlanner:
    """Select idle storage gaps covering the current peak, then mask transfers.

    Prefer trace-supplied native autograd intervals when they cover the peak;
    use other candidates when needed to satisfy the same memory budget.

    Args:
        enable_communication_masking: Delay releases and advance H2D copies in
            available memory. Runtime events enforce transfer dependencies.
        weights: Normalized size, coverage and neighboring-lifetime weights.
    """

    def __init__(
        self,
        *,
        enable_communication_masking: bool = True,
        weights: ScoreWeights | None = None,
    ) -> None:
        """Configure peak selection and optional communication masking."""
        if not isinstance(enable_communication_masking, bool):
            raise TypeError("enable_communication_masking must be a bool")
        self.enable_communication_masking = enable_communication_masking
        self.weights = weights if weights is not None else ScoreWeights()
        if not all(
            math.isfinite(value) and value >= 0 for value in vars(self.weights).values()
        ):
            raise ValueError("peak score weights must be finite and nonnegative")

    def _reserve_framework_workspace(
        self, trace: ActivationTrace, profile: PeakProfile, selected: list[EvictionGap], limit: int,
    ) -> int:
        """Select extra native intervals and return the workspace-adjusted limit.

        Args:
            trace: Recorded framework allocation sizes and native interval hints.
            profile: Residency profile updated by the extra selections.
            selected: Existing selections extended with extra native intervals.
            limit: Original activation-residency limit in bytes.
        """
        workspace = trace.framework_allocation_bytes
        if isinstance(workspace, bool) or not isinstance(workspace, int) or workspace < 0:
            raise ValueError("framework_allocation_bytes must be a nonnegative integer")
        if not workspace:
            return limit
        # Framework dtype conversions are outside activation ownership.
        # Leave room for their large contiguous allocation plus allocator
        # split slack, using only native-save intervals for extra offload.
        limit = max(0, limit - 2 * workspace)
        preferred = set(trace.preferred_eviction_gaps)
        remaining = [
            gap for gap in profile.gaps
            if gap not in selected and (gap.storage_id, gap.start, gap.end) in preferred
        ]
        selected.extend(_select_gaps(profile.resident_bytes, remaining, limit, self.weights, preferred))
        return limit

    def build(self, trace: ActivationTrace) -> ResidencySchedule:
        """Build actions compatible with the baseline replay executor.

        The limit covers trace-managed storage bytes, not total device memory.
        Irreducible operator peaks are reported; a schedule is not an OOM or
        performance guarantee. Input trace data is never modified.

        Args:
            trace: Warmup operations, storage lifetimes and allocation metadata.
        """
        limit = trace.memory_limit_bytes
        if limit is None:
            return ResidencySchedule()
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("memory_limit_bytes must be a nonnegative integer or None")
        profile = build_profile(trace)
        selected = _select_gaps(
            profile.resident_bytes, profile.gaps, limit, self.weights, set(trace.preferred_eviction_gaps)
        )
        limit = self._reserve_framework_workspace(trace, profile, selected, limit)
        if self.enable_communication_masking:
            placements = mask_communications(selected, profile.resident_bytes, limit)
            limit_prefetch_advance(
                placements, profile.resident_bytes, [op.duration_ms for op in trace.ops], trace.h2d_bandwidth_gbps
            )
            limit_release_delay(
                placements, [op.duration_ms for op in trace.ops], trace.d2h_bandwidth_gbps
            )
        else:
            placements = [
                TransferPlacement(gap, gap.start, gap.start, gap.end)
                for gap in selected
            ]
        peak = max(profile.resident_bytes, default=0)
        if peak > limit:
            logger.warning(
                "Peak planner cannot meet budget: planned peak=%d bytes, limit=%d bytes",
                peak,
                limit,
            )
        logger.info(
            "Peak planner selected %d/%d gaps; planned peak=%d bytes, budget=%d bytes",
            len(selected),
            len(profile.gaps),
            peak,
            limit,
        )
        # Optional checks must not turn wrapper-only detach/framework reads
        # into required replay operations or expand managed tensor bindings.
        checkpoint_ops = {
            index for index, op in enumerate(trace.ops)
            if op.name != "detach.default" and not op.name.startswith("framework_read:")
        }
        return _emit_schedule(placements, profile.last_uses, trace.retained_sids, checkpoint_ops)
