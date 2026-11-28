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
"""Invocation-local Top-K and KV state for DSA MTP iteration sharing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


def _as_ints(values: torch.Tensor) -> tuple[int, ...]:
    """Convert packed cumulative endpoints to a host tuple for validation."""
    return tuple(int(value) for value in values.detach().cpu().tolist())


def _flatten_positions(
    position_ids: torch.Tensor | None, *, batch_size: int, seq_length: int,
) -> tuple[int, ...] | None:
    """Normalize optional position IDs to the batch-major query-row order."""
    if position_ids is None:
        return None
    if position_ids.ndim != 2 or position_ids.shape[1] != seq_length:
        raise ValueError("MTP IndexShare position_ids must have shape [1 or batch, query_length]")
    if position_ids.shape[0] == 1 and batch_size > 1:
        position_ids = position_ids.expand(batch_size, -1)
    if position_ids.shape[0] != batch_size:
        raise ValueError("MTP IndexShare position_ids batch dimension does not match query")
    return tuple(int(value) for value in position_ids.detach().reshape(-1).cpu().tolist())


@dataclass(frozen=True)
class MTPIndexShareView:
    """First-step KV bank and Top-K rows aligned to one later query."""

    topk_indices: torch.Tensor
    latent_kv: torch.Tensor
    key_rope: torch.Tensor
    key_ends: torch.Tensor
    kv_prefix_length: int


@dataclass(frozen=True)
class MTPIndexShareState:
    """Graph-preserving producer tensors from MTP step zero.

    The integer Top-K selection is discrete. ``latent_kv`` and ``key_rope`` are
    deliberately retained without detaching so gradients from every consumer
    accumulate into the first step's key/value projections.
    """

    topk_indices: torch.Tensor
    latent_kv: torch.Tensor
    key_rope: torch.Tensor
    query_ends: torch.Tensor
    key_ends: torch.Tensor
    query_positions: tuple[int, ...] | None
    kv_prefix_length: int
    index_topk: int

    def view_for_query(
        self,
        query_ends: torch.Tensor,
        *,
        query_positions: torch.Tensor | None,
        query_batch_size: int,
        query_seq_length: int,
        index_topk: int,
    ) -> MTPIndexShareView:
        """Select producer Top-K rows for this step and retain its entire KV bank."""
        if self.index_topk != index_topk:
            raise ValueError("MTP steps must use the same DSA index_topk")
        source_ends = _as_ints(self.query_ends)
        target_ends = _as_ints(query_ends)
        if len(source_ends) != len(target_ends):
            raise ValueError("MTP IndexShare requires matching packed query segment counts")
        source_lengths = tuple(end - (source_ends[i - 1] if i else 0) for i, end in enumerate(source_ends))
        target_lengths = tuple(end - (target_ends[i - 1] if i else 0) for i, end in enumerate(target_ends))
        if any(length < 0 for length in target_lengths) or any(
            target > source for source, target in zip(source_lengths, target_lengths)
        ):
            raise ValueError("Later MTP query segments must be prefixes of the first-step query segments")

        rows: list[int] = []
        target_positions = _flatten_positions(
            query_positions, batch_size=query_batch_size, seq_length=query_seq_length,
        )
        if target_positions is not None and self.query_positions is not None:
            target_start = 0
            for document, length in enumerate(target_lengths):
                available = [
                    (self.query_positions[row], row)
                    for row in range(sum(source_lengths[:document]), sum(source_lengths[:document + 1]))
                ]
                for offset in range(length):
                    position = target_positions[target_start + offset]
                    # Reuse the latest first-step route that is causal for this
                    # query. Thus step-zero h4 Top-K is valid for a later h5 query.
                    candidates = [(source_position, row) for source_position, row in available
                                  if source_position <= position]
                    if not candidates:
                        raise ValueError("MTP query has no causal first-step Top-K row in its document")
                    row = max(candidates)[1]
                    rows.append(row)
                target_start += length
        else:
            source_start = 0
            for source_length, target_length in zip(source_lengths, target_lengths):
                if source_length == 0 and target_length:
                    raise ValueError("MTP query has no first-step Top-K rows in its packed document")
                rows.extend(source_start + min(offset, source_length - 1) for offset in range(target_length))
                source_start += source_length

        source_row_count = sum(source_lengths)
        if source_row_count != self.topk_indices.shape[0]:
            raise ValueError("First-step Top-K rows do not match its packed query endpoints")
        expected_rows = sum(target_lengths)
        if len(rows) != expected_rows:
            raise ValueError("MTP query-to-Top-K mapping does not cover every consumer row")
        row_ids = torch.tensor(rows, dtype=torch.long, device=self.topk_indices.device)
        topk = self.topk_indices.index_select(0, row_ids)
        if tuple(self.key_ends.shape) != tuple(query_ends.shape):
            raise ValueError("MTP steps must preserve packed document boundaries for the shared KV bank")
        return MTPIndexShareView(topk, self.latent_kv, self.key_rope, self.key_ends, self.kv_prefix_length)


@dataclass
class MTPIndexShareContext:
    """Forward-local carrier; never store it on a module or reuse across batches."""

    enabled: bool = True
    states: dict[str, MTPIndexShareState] = field(default_factory=dict)
    dense_keys: set[str] = field(default_factory=set)
    expected_keys: tuple[str, ...] = ()
    _recompute_depth: int = 0

    @contextmanager
    def recomputing(self) -> Iterator[None]:
        """Mark a non-reentrant checkpoint replay for the duration of its call."""
        self._recompute_depth += 1
        try:
            yield
        finally:
            self._recompute_depth -= 1

    @property
    def is_recomputing(self) -> bool:
        """Combine this module's checkpoint marker with HP's outer replay state."""
        from hyper_parallel.core.activation_memory.recompute_state import is_recomputing

        return self._recompute_depth > 0 or is_recomputing()

    def publish(self, key: str, state: MTPIndexShareState) -> None:
        """Publish step-zero tensors once, retaining the original graph on replay."""
        if not self.enabled:
            return
        if self.is_recomputing:
            if key not in self.states:
                raise RuntimeError("MTP IndexShare producer was recomputed without its forward state")
            return
        if key in self.states or key in self.dense_keys:
            raise RuntimeError(f"MTP IndexShare key {key!r} was published more than once in one forward")
        self.states[key] = state

    def mark_dense(self, key: str) -> None:
        """Disable sharing for this attention path during a dense warm-up forward."""
        if self.enabled:
            if not self.is_recomputing:
                self.dense_keys.add(key)

    def consumer_state(self, key: str, step: int) -> MTPIndexShareState | None:
        """Return the step-zero state, or report a missing producer immediately."""
        if not self.enabled or step == 0 or key in self.dense_keys:
            return None
        if step < 0:
            raise ValueError("mtp_iteration must be a nonnegative integer")
        state = self.states.get(key)
        if state is None:
            raise RuntimeError(f"MTP iteration {step} requires step-zero state for attention {key!r}")
        return state

    def validate_complete(self) -> None:
        """Fail if the model adapter did not route the state through its DSA blocks."""
        missing = [key for key in self.expected_keys if key not in self.states and key not in self.dense_keys]
        if missing:
            raise RuntimeError(f"MTP IndexShare was enabled but step zero did not publish: {missing}")

