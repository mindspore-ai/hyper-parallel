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
"""Reserved, Host-ready, and consumed offsets for exact data replay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class DistributedDatasetState:
    """Checkpoint-safe state recorded only after successful optimizer steps."""

    consumed_offset: int
    next_step: int
    source_size: int
    local_batches_per_step: int
    version: int = 1

    def to_dict(self) -> dict[str, int]:
        """Convert state to a plain checkpoint dictionary."""
        return {
            "version": self.version,
            "consumed_offset": self.consumed_offset,
            "next_step": self.next_step,
            "source_size": self.source_size,
            "local_batches_per_step": self.local_batches_per_step,
        }

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any]) -> "DistributedDatasetState":
        """Validate and construct state from a checkpoint dictionary.

        Args:
            state_dict: Serialized distributed-dataset state.

        Returns:
            Validated checkpoint state.
        """
        required = {"version", "consumed_offset", "next_step", "source_size", "local_batches_per_step"}
        missing = required - set(state_dict)
        if missing:
            raise ValueError(f"Distributed dataset state is missing keys: {sorted(missing)}.")
        values = {name: state_dict[name] for name in required}
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values.values()):
            raise ValueError("Distributed dataset state values must all be integers.")
        return cls(**values)


@dataclass
class _PendingWindow:
    local_batch_offset_start: int
    local_batch_offset_end: int
    plan_id: str | None = None
    ready: bool = False
    delivered: bool = False


class DatasetStateTracker:
    """Track reservation, Host readiness, and successful consumption."""

    VERSION = 1

    def __init__(self, source_size: int, local_batches_per_step: int, prefetch_steps: int) -> None:
        """Initialize offset state and the maximum number of unconsumed windows."""
        if source_size < 0:
            raise ValueError(f"source_size must be non-negative, but got {source_size}.")
        if local_batches_per_step < 1 or prefetch_steps < 1:
            raise ValueError(
                "local_batches_per_step and prefetch_steps must be positive, "
                f"got {local_batches_per_step} and {prefetch_steps}."
            )
        self._source_size = source_size
        self._local_batches_per_step = local_batches_per_step
        self._prefetch_steps = prefetch_steps
        self._consumed_offset = 0
        self._next_step = 0
        self._pending: deque[_PendingWindow] = deque()
        self._lock = Lock()

    @property
    def consumed_offset(self) -> int:
        """Return the offset after the last successful optimizer step."""
        with self._lock:
            return self._consumed_offset

    @property
    def reserved_offset(self) -> int:
        """Return the offset after all scheduled Host-prefetch windows."""
        with self._lock:
            return self._reserved_offset_unlocked()

    @property
    def ready_offset(self) -> int:
        """Return the offset after contiguous windows ready in Host memory."""
        with self._lock:
            offset = self._consumed_offset
            for window in self._pending:
                if not window.ready:
                    break
                offset = window.local_batch_offset_end
            return offset

    @property
    def next_step(self) -> int:
        """Return the logical step index for the next scheduled window."""
        with self._lock:
            return self._next_step + len(self._pending)

    @property
    def has_delivered_unconsumed(self) -> bool:
        """Return whether a delivered step still awaits successful consumption."""
        with self._lock:
            return bool(self._pending and self._pending[0].delivered)

    @property
    def can_prefetch(self) -> bool:
        """Return whether the bounded Host-ready queue has a free step slot."""
        with self._lock:
            queued_windows = sum(not window.delivered for window in self._pending)
            return queued_windows < self._prefetch_steps

    @property
    def has_pending(self) -> bool:
        """Return whether any planned or scheduled step remains unconsumed."""
        with self._lock:
            return bool(self._pending)

    def can_reserve_full_step(self) -> bool:
        """Return whether the source contains another complete local-batch step."""
        with self._lock:
            return self._reserved_offset_unlocked() + self._local_batches_per_step <= self._source_size

    def reserve(self) -> tuple[int, int, int]:
        """Reserve the next bounded prefetch window.

        Returns:
            ``(step, local_batch_offset_start, local_batch_offset_end)`` for the producer task.
        """
        with self._lock:
            queued_windows = sum(not window.delivered for window in self._pending)
            if queued_windows >= self._prefetch_steps:
                raise ValueError(f"At most {self._prefetch_steps} Host-prefetch steps may be queued.")
            start = self._reserved_offset_unlocked()
            if start + self._local_batches_per_step > self._source_size:
                raise ValueError("Local-batch source does not contain another complete optimizer step.")
            end = start + self._local_batches_per_step
            step = self._next_step + len(self._pending)
            self._pending.append(_PendingWindow(start, end))
            return step, start, end

    def mark_ready(self, local_batch_offset_start: int, local_batch_offset_end: int) -> None:
        """Mark one fully read and Host-preprocessed window as ready.

        Args:
            local_batch_offset_start: Inclusive local-batch offset of the reserved window.
            local_batch_offset_end: Exclusive local-batch offset of the reserved window.
        """
        with self._lock:
            for window in self._pending:
                if (window.local_batch_offset_start, window.local_batch_offset_end) != (
                    local_batch_offset_start,
                    local_batch_offset_end,
                ):
                    continue
                if window.ready:
                    raise ValueError(
                        f"Host-prefetch window [{local_batch_offset_start}, {local_batch_offset_end}) is already ready."
                    )
                window.ready = True
                return
        raise ValueError(
            f"Host-prefetch window [{local_batch_offset_start}, {local_batch_offset_end}) was not reserved."
        )

    def mark_delivered(
        self,
        plan_id: str,
        local_batch_offset_start: int,
        local_batch_offset_end: int,
    ) -> None:
        """Attach a completed plan ID to the oldest scheduled window.

        Args:
            plan_id: Deterministic identifier of the delivered step plan.
            local_batch_offset_start: Inclusive local-batch offset of the delivered window.
            local_batch_offset_end: Exclusive local-batch offset of the delivered window.
        """
        with self._lock:
            if not self._pending:
                raise ValueError("Cannot deliver a step when no prefetch window is pending.")
            window = self._pending[0]
            if not window.ready:
                raise ValueError("Cannot deliver a distributed-data step before its Host data is ready.")
            if window.delivered:
                raise ValueError("The current prefetched step was already delivered and must be committed first.")
            if not isinstance(plan_id, str) or not plan_id:
                raise ValueError(f"plan_id must be a non-empty string, but got {plan_id!r}.")
            if (
                local_batch_offset_start != window.local_batch_offset_start
                or local_batch_offset_end != window.local_batch_offset_end
            ):
                raise ValueError(
                    f"Delivered local-batch offsets [{local_batch_offset_start}, {local_batch_offset_end}) "
                    f"do not match reserved window "
                    f"[{window.local_batch_offset_start}, {window.local_batch_offset_end})."
                )
            window.plan_id = plan_id
            window.delivered = True

    def commit(self, plan_id: str) -> None:
        """Commit the oldest delivered plan after optimizer-step success.

        Args:
            plan_id: Deterministic identifier of the successfully consumed plan.
        """
        with self._lock:
            if not self._pending or not self._pending[0].delivered:
                raise ValueError("No delivered distributed-data step is available to commit.")
            window = self._pending[0]
            if plan_id != window.plan_id:
                raise ValueError(f"Plans must commit in order: expected {window.plan_id}, but got {plan_id}.")
            self._consumed_offset = window.local_batch_offset_end
            self._next_step += 1
            self._pending.popleft()

    def state_dict(self) -> dict[str, int]:
        """Return checkpoint state at the successfully consumed offset only."""
        with self._lock:
            return DistributedDatasetState(
                consumed_offset=self._consumed_offset,
                next_step=self._next_step,
                source_size=self._source_size,
                local_batches_per_step=self._local_batches_per_step,
                version=self.VERSION,
            ).to_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore consumed state and discard any runtime look-ahead.

        Args:
            state_dict: Checkpoint state recorded after a successful optimizer step.
        """
        state = DistributedDatasetState.from_dict(state_dict)
        if state.version != self.VERSION:
            raise ValueError(f"Unsupported distributed dataset state version {state.version}.")
        if state.source_size != self._source_size:
            raise ValueError(
                f"Source size changed from checkpoint {state.source_size} to {self._source_size}."
            )
        if state.local_batches_per_step != self._local_batches_per_step:
            raise ValueError(
                f"local_batches_per_step changed from checkpoint {state.local_batches_per_step} "
                f"to {self._local_batches_per_step}."
            )
        if state.consumed_offset < 0 or state.consumed_offset > self._source_size:
            raise ValueError(
                f"Invalid consumed_offset {state.consumed_offset} for source size {self._source_size}."
            )
        expected_step = state.consumed_offset // self._local_batches_per_step
        if state.consumed_offset % self._local_batches_per_step or state.next_step != expected_step:
            raise ValueError(
                f"State consumed_offset {state.consumed_offset} and next_step {state.next_step} are not step-aligned."
            )
        with self._lock:
            if self._pending:
                raise ValueError("Cannot restore distributed dataset state while prefetch work is pending.")
            self._consumed_offset = state.consumed_offset
            self._next_step = state.next_step

    def _reserved_offset_unlocked(self) -> int:
        return self._pending[-1].local_batch_offset_end if self._pending else self._consumed_offset
