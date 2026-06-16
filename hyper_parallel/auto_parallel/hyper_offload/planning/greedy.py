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
"""Greedy planner for storage residency actions."""

from __future__ import annotations

import logging
import pprint
from dataclasses import dataclass

from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencyActionType, ResidencySchedule
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import (
    AccessKind,
    ActivationTrace,
    StorageAccess,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EvictionCandidate:
    """An open interval between two adjacent accesses."""

    storage_id: int
    size_bytes: int
    copy_start: StorageAccess
    release_start: StorageAccess
    end: StorageAccess

    @property
    def distance(self) -> int:
        """Return the distance."""
        return self.end.op_id - self.release_start.op_id

    @property
    def start_interior(self) -> int:
        """Return the start interior."""
        return self.release_start.op_id + 1

    @property
    def end_interior(self) -> int:
        """Return the end interior."""
        return self.end.op_id - 1


@dataclass
class _Footprint:
    """Resident-memory footprint and derived eviction data for one storage."""

    size_bytes: int
    first_op_id: int
    last_op_id: int
    candidacies: list[_EvictionCandidate]
    last_access: StorageAccess


class GreedyResidencyPlanner:
    """Plan residency by evicting long access gaps until the budget is met."""

    def build(self, trace: ActivationTrace) -> ResidencySchedule:
        """Build the residency schedule from an activation trace.

        Args:
            trace: Activation trace with ops, storage sizes and memory limit.

        Returns:
            A :class:`ResidencySchedule` with pre/post eviction actions.
        """
        schedule = ResidencySchedule()
        limit = trace.memory_limit_bytes if trace.memory_limit_bytes is not None else float("inf")
        if not trace.ops or limit == float("inf"):
            return schedule

        accesses_by_storage = self._group_accesses_by_storage(trace)

        resident_bytes, candidates, releasable = self._build_resident_footprint(
            trace, accesses_by_storage
        )

        selected = self._greedy_select(candidates, resident_bytes, limit)

        for candidate in selected:
            self._add_eviction_actions(schedule, candidate)
        self._emit_release_actions(schedule, selected, trace, releasable)

        peak = max(resident_bytes, default=0)
        self._log_planning_summary(trace, schedule, selected, candidates, peak, limit)

        return schedule

    # ------------------------------------------------------------------
    # Phase 1: group accesses by storage
    # ------------------------------------------------------------------

    @staticmethod
    def _group_accesses_by_storage(
        trace: ActivationTrace,
    ) -> dict[int, list[StorageAccess]]:
        """Return a mapping from storage id to its ordered list of accesses."""
        accesses_by_storage: dict[int, list[StorageAccess]] = {}
        for op in trace.ops:
            for access in op.accesses:
                accesses_by_storage.setdefault(access.storage_id, []).append(access)
        return accesses_by_storage

    # ------------------------------------------------------------------
    # Phase 2: compute resident footprint, eviction candidates & releasable
    # ------------------------------------------------------------------

    @classmethod
    def _build_resident_footprint(
        cls,
        trace: ActivationTrace,
        accesses_by_storage: dict[int, list[StorageAccess]],
    ) -> tuple[list[int], list[_EvictionCandidate], dict[int, StorageAccess]]:
        """Compute the per-op resident-memory array and enumerate eviction candidates.

        Args:
            trace: Activation trace.
            accesses_by_storage: Per-storage access lists (from :meth:`_group_accesses_by_storage`).

        Returns:
            A triple ``(resident_bytes, candidates, releasable)`` where
            *resident_bytes* is indexed by op id,
            *candidates* are sorted for greedy selection, and
            *releasable* maps each storage id to its last access.
        """
        max_op_id = max(
            (access.op_id for op in trace.ops for access in op.accesses),
            default=0,
        )
        resident_bytes = [0] * (max_op_id + 1)
        unsorted_candidates: list[_EvictionCandidate] = []
        releasable: dict[int, StorageAccess] = {}

        for sid, accesses in accesses_by_storage.items():
            size_bytes = trace.storage_sizes.get(sid)
            if size_bytes is None or not accesses:
                continue

            footprint = cls._process_storage_accesses(sid, size_bytes, accesses)
            for op_id in range(footprint.first_op_id, footprint.last_op_id + 1):
                resident_bytes[op_id] += size_bytes

            releasable[sid] = footprint.last_access
            unsorted_candidates.extend(footprint.candidacies)

        unsorted_candidates.sort(
            key=lambda candidate: (
                candidate.distance,
                candidate.size_bytes,
                -candidate.release_start.op_id,
            ),
            reverse=True,
        )

        return resident_bytes, unsorted_candidates, releasable

    @classmethod
    def _process_storage_accesses(
        cls,
        sid: int,
        size_bytes: int,
        accesses: list[StorageAccess],
    ) -> _Footprint:
        """Build the per-storage footprint: eviction candidates and extent info.

        Args:
            sid: Storage id.
            size_bytes: Size of the storage in bytes.
            accesses: Ordered access list for this storage.

        Returns:
            A :class:`_Footprint` with size, extent, eviction candidates and
            last access.
        """
        ordered = sorted(accesses, key=lambda access: access.op_id)
        first = ordered[0].op_id
        last = ordered[-1].op_id
        candidacies: list[_EvictionCandidate] = []

        for release_index, (release_start, end) in enumerate(
            zip(ordered, ordered[1:], strict=False)
        ):
            copy_start = cls._earliest_safe_copy_start(ordered, release_index)
            candidate = _EvictionCandidate(sid, size_bytes, copy_start, release_start, end)
            if candidate.start_interior <= candidate.end_interior:
                candidacies.append(candidate)

        return _Footprint(
            size_bytes=size_bytes,
            first_op_id=first,
            last_op_id=last,
            candidacies=candidacies,
            last_access=ordered[-1],
        )

    # ------------------------------------------------------------------
    # Phase 3: greedy selection
    # ------------------------------------------------------------------

    @staticmethod
    def _greedy_select(
        candidates: list[_EvictionCandidate],
        resident_bytes: list[int],
        limit: float,
    ) -> list[_EvictionCandidate]:
        """Select eviction candidates greedily by longest gap first.

        Args:
            candidates: Pre-sorted candidates (longest gap first).
            resident_bytes: Mutable per-op resident bytes (modified in-place).
            limit: Memory budget in bytes.

        Returns:
            List of selected candidates.
        """
        selected: list[_EvictionCandidate] = []
        for candidate in candidates:
            if max(resident_bytes, default=0) <= limit:
                break

            affected_ops = range(candidate.start_interior, candidate.end_interior + 1)
            if not any(resident_bytes[op_id] > limit for op_id in affected_ops):
                continue

            selected.append(candidate)
            for op_id in affected_ops:
                resident_bytes[op_id] -= candidate.size_bytes

        return selected

    # ------------------------------------------------------------------
    # Phase 4: emit release actions
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_release_actions(
        schedule: ResidencySchedule,
        selected: list[_EvictionCandidate],
        trace: ActivationTrace,
        releasable: dict[int, StorageAccess],
    ) -> None:
        """Emit release-device and (conditional) release-host post-actions.

        Args:
            schedule: Schedule to mutate.
            selected: Eviction candidates chosen by greedy selection.
            trace: Original activation trace (for ``retained_sids``).
            releasable: Mapping from storage id to its last access.
        """
        evicted_sids = {candidate.storage_id for candidate in selected}
        for sid, access in releasable.items():
            if sid in trace.retained_sids:
                continue
            schedule.add_post(
                access.op_id,
                ResidencyActionType.RELEASE_DEVICE,
                sid,
            )
            if sid in evicted_sids:
                schedule.add_post(
                    access.op_id,
                    ResidencyActionType.RELEASE_HOST,
                    sid,
                )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_planning_summary(
        trace: ActivationTrace,
        schedule: ResidencySchedule,
        selected: list[_EvictionCandidate],
        candidates: list[_EvictionCandidate],
        peak: int,
        limit: float,
    ) -> None:
        """Log the planning result at info and debug levels."""
        logger.info(
            "Residency planner selected %d/%d gaps; simulated peak %.2f MiB, budget %.2f MiB",
            len(selected),
            len(candidates),
            peak / 1024**2,
            limit / 1024**2,
        )
        logger.debug(
            "Schedule: %d pre-actions, %d post-actions",
            sum(len(v) for v in schedule.pre.values()),
            sum(len(v) for v in schedule.post.values()),
        )
        GreedyResidencyPlanner._log_schedule_by_op(trace, schedule)

    @staticmethod
    def _log_schedule_by_op(
        trace: ActivationTrace,
        schedule: ResidencySchedule,
    ) -> None:
        """Log the schedule details for each op (debug-level)."""
        for idx, op in enumerate(trace.ops):
            logger.debug("  OP %d name=%s", idx, op.name)
            for access in op.accesses:
                logger.debug(
                    "    access sid=%d %s",
                    access.storage_id,
                    access.kind.name,
                )

            pre_actions = schedule.pre_actions(idx)
            if pre_actions:
                logger.debug(
                    "    PRE %s",
                    GreedyResidencyPlanner._format_actions(pre_actions, "    PRE "),
                )

            post_actions = schedule.post_actions(idx)
            if post_actions:
                logger.debug(
                    "    POST %s",
                    GreedyResidencyPlanner._format_actions(post_actions, "    POST "),
                )

    @staticmethod
    def _earliest_safe_copy_start(accesses: list[StorageAccess], release_index: int) -> StorageAccess:
        """Return the earliest D2H point that still captures current data.

        D2H may move before later reads because they do not mutate storage.
        It must not move before the latest write at or before the release
        point, otherwise the host copy could become stale.
        """
        for index in range(release_index, -1, -1):
            if accesses[index].kind == AccessKind.WRITE:
                return accesses[index]
        return accesses[0]

    @staticmethod
    def _format_actions(actions, prefix: str) -> str:
        items = [(action.kind.name, action.storage_id) for action in actions]
        formatted = pprint.pformat(items, compact=True, width=80)
        return formatted.replace("\n ", "\n" + " " * len(prefix))

    @staticmethod
    def _add_eviction_actions(schedule: ResidencySchedule, candidate: _EvictionCandidate) -> None:
        """Add pre/post schedule actions for a single eviction candidate."""
        schedule.add_post(
            candidate.copy_start.op_id,
            ResidencyActionType.COPY_D2H,
            candidate.copy_start.storage_id,
        )
        schedule.add_post(
            candidate.release_start.op_id,
            ResidencyActionType.RELEASE_DEVICE,
            candidate.release_start.storage_id,
        )
        schedule.add_pre(
            candidate.end.op_id,
            ResidencyActionType.COPY_H2D,
            candidate.end.storage_id,
        )
