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
"""Prepared and consumed offsets for exact data replay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributedDatasetState:
    """Checkpoint-safe state recorded only after successful optimizer steps."""

    consumed_offset: int
    next_step: int
    metadata_size: int
    samples_per_step: int
    version: int = 1

    def to_dict(self) -> dict[str, int]:
        """Convert state to a plain checkpoint dictionary."""
        return {
            "version": self.version,
            "consumed_offset": self.consumed_offset,
            "next_step": self.next_step,
            "metadata_size": self.metadata_size,
            "samples_per_step": self.samples_per_step,
        }

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any]) -> "DistributedDatasetState":
        """Validate and construct state from a checkpoint dictionary."""
        required = {"version", "consumed_offset", "next_step", "metadata_size", "samples_per_step"}
        missing = required - set(state_dict)
        if missing:
            raise ValueError(f"Distributed dataset state is missing keys: {sorted(missing)}.")
        values = {name: state_dict[name] for name in required}
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values.values()):
            raise ValueError("Distributed dataset state values must all be integers.")
        return cls(**values)


@dataclass
class _PendingWindow:
    cursor_start: int
    cursor_end: int
    replay_id: str | None = None
    delivered: bool = False


class DatasetStateTracker:
    """Track bounded preparation separately from successfully consumed data."""

    VERSION = 1

    def __init__(self, metadata_size: int, samples_per_step: int, prefetch_steps: int) -> None:
        """Initialize offset state and the maximum number of unconsumed windows."""
        if metadata_size < 0:
            raise ValueError(f"metadata_size must be non-negative, but got {metadata_size}.")
        if samples_per_step < 1 or prefetch_steps < 1:
            raise ValueError(
                f"samples_per_step and prefetch_steps must be positive, got {samples_per_step} and {prefetch_steps}."
            )
        self._metadata_size = metadata_size
        self._samples_per_step = samples_per_step
        self._prefetch_steps = prefetch_steps
        self._consumed_offset = 0
        self._next_step = 0
        self._pending: deque[_PendingWindow] = deque()

    @property
    def consumed_offset(self) -> int:
        """Return the offset after the last successful optimizer step."""
        return self._consumed_offset

    @property
    def prepared_offset(self) -> int:
        """Return the offset after all reserved preparation windows."""
        return self._pending[-1].cursor_end if self._pending else self._consumed_offset

    @property
    def next_step(self) -> int:
        """Return the logical step index for the next scheduled window."""
        return self._next_step + len(self._pending)

    @property
    def has_delivered_unconsumed(self) -> bool:
        """Return whether a delivered step still awaits successful consumption."""
        return bool(self._pending and self._pending[0].delivered)

    @property
    def can_prefetch(self) -> bool:
        """Return whether another bounded look-ahead window may be scheduled."""
        return len(self._pending) < self._prefetch_steps

    @property
    def has_pending(self) -> bool:
        """Return whether any planned or scheduled step remains unconsumed."""
        return bool(self._pending)

    def can_reserve_full_step(self) -> bool:
        """Return whether the metadata source contains another complete local step."""
        return self.prepared_offset + self._samples_per_step <= self._metadata_size

    def reserve(self) -> tuple[int, int, int]:
        """Reserve the next bounded prefetch window.

        Returns:
            ``(step, cursor_start, cursor_end)`` for the producer task.
        """
        if not self.can_prefetch:
            raise ValueError(f"At most {self._prefetch_steps} unconsumed steps may be prepared.")
        if not self.can_reserve_full_step():
            raise ValueError("Metadata source does not contain another complete optimizer step.")
        start = self.prepared_offset
        end = start + self._samples_per_step
        step = self._next_step + len(self._pending)
        self._pending.append(_PendingWindow(start, end))
        return step, start, end

    def mark_delivered(self, replay_id: str, cursor_start: int, cursor_end: int) -> None:
        """Attach a completed replay ID to the oldest scheduled window."""
        if not self._pending:
            raise ValueError("Cannot deliver a step when no prefetch window is pending.")
        window = self._pending[0]
        if window.delivered:
            raise ValueError("The current prefetched step was already delivered and must be committed first.")
        if not isinstance(replay_id, str) or not replay_id:
            raise ValueError(f"replay_id must be a non-empty string, but got {replay_id!r}.")
        if cursor_start != window.cursor_start or cursor_end != window.cursor_end:
            raise ValueError(
                f"Delivered cursor [{cursor_start}, {cursor_end}) does not match reserved window "
                f"[{window.cursor_start}, {window.cursor_end})."
            )
        window.replay_id = replay_id
        window.delivered = True

    def commit(self, replay_id: str) -> None:
        """Commit the oldest delivered plan after optimizer-step success."""
        if not self._pending or not self._pending[0].delivered:
            raise ValueError("No delivered distributed-data step is available to commit.")
        window = self._pending[0]
        if replay_id != window.replay_id:
            raise ValueError(f"Plans must commit in order: expected {window.replay_id}, but got {replay_id}.")
        self._consumed_offset = window.cursor_end
        self._next_step += 1
        self._pending.popleft()

    def state_dict(self) -> dict[str, int]:
        """Return checkpoint state at the successfully consumed offset only."""
        return DistributedDatasetState(
            consumed_offset=self._consumed_offset,
            next_step=self._next_step,
            metadata_size=self._metadata_size,
            samples_per_step=self._samples_per_step,
            version=self.VERSION,
        ).to_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore consumed state and discard any runtime look-ahead."""
        if self._pending:
            raise ValueError("Cannot restore distributed dataset state while prefetch work is pending.")
        state = DistributedDatasetState.from_dict(state_dict)
        if state.version != self.VERSION:
            raise ValueError(f"Unsupported distributed dataset state version {state.version}.")
        if state.metadata_size != self._metadata_size:
            raise ValueError(
                f"Metadata size changed from checkpoint {state.metadata_size} to {self._metadata_size}."
            )
        if state.samples_per_step != self._samples_per_step:
            raise ValueError(
                f"samples_per_step changed from checkpoint {state.samples_per_step} "
                f"to {self._samples_per_step}."
            )
        if state.consumed_offset < 0 or state.consumed_offset > self._metadata_size:
            raise ValueError(
                f"Invalid consumed_offset {state.consumed_offset} for metadata size {self._metadata_size}."
            )
        expected_step = state.consumed_offset // self._samples_per_step
        if state.consumed_offset % self._samples_per_step or state.next_step != expected_step:
            raise ValueError(
                f"State consumed_offset {state.consumed_offset} and next_step {state.next_step} are not step-aligned."
            )
        self._consumed_offset = state.consumed_offset
        self._next_step = state.next_step
