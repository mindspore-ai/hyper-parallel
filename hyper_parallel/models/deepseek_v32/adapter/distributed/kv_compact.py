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
"""Document-aware KV compaction for the DeepSeek-V3.2 DSA context-parallel path.

TND packing isolates documents in attention: a query never attends beyond its
own document. The causal KV prefix of a query block therefore only needs to
start at the *earliest active document*, not at position 0. For a query block
``[s, e)`` the block keeps ``[d0, e)`` where ``d0`` is the start of the
document containing ``s`` (the largest cumulative document end that is
``<= s``). AllGather still fetches the full chunks, so network bytes are
unchanged; what shrinks is the operators' physical KV input and their
temporary working set.

The three DSA operators (lightning indexer, sparse flash attention, indexer
KL loss) share one coordinate carrier, the cumulative document ends
(``actual_seq_qlen`` / ``actual_seq_kvlen``). Compaction therefore rewrites
the KV-side ends to the compacted window and slices the KV rows; the operator
bodies and the document-local Top-K coordinates stay untouched.

Backward maps compacted gradients back to the gathered layout: selected
positions scatter-add (overlapping windows accumulate), everything else gets
zeros. A compacted gradient is never simply zero-padded at the tail.
"""
from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch  # pylint: disable=forbidden-backend-import

__all__ = [
    "CompactPlan",
    "build_compact_plan",
    "get_compact_plan",
    "interval_select",
    "compact_pair_grad",
]

# Plans are rebuilt per forward; every layer of one micro-batch shares the
# same boundaries, so a small content-addressed cache avoids rebuilding the
# plan 60+ times per step. Keys are the full metadata tuples, so a plan can
# never leak from one micro-batch into another with different content.
_COMPACT_PLAN_CACHE: ContextVar[Optional["OrderedDict[tuple, Optional[CompactPlan]]"]] = ContextVar(
    "dsa_compact_plan_cache", default=None
)
_COMPACT_PLAN_CACHE_LIMIT = 128


@dataclass(frozen=True, eq=False)
class CompactPlan:
    """One forward's compaction decision, immutable once built.

    Attributes:
        batch_size: Batch rows of the gathered KV stream.
        row_stride: Per-row length of the gathered stream (``cp_size *
            local_seq_len``); row ``b`` owns flat positions
            ``[b * row_stride, (b + 1) * row_stride)``.
        windows: Half-open flat ranges kept from the gathered stream, one per
            batch row under fixed CP. Ranges may overlap when one buffer
            serves several query blocks (dynamic pairing); the backward
            accumulates overlaps.
        kv_ends: int32 CPU tensor of cumulative document ends over the
            compacted token stream, the ``actual_seq_kvlen`` the operators
            receive. Empty documents are dropped and the ends are deduplicated.
        q_ends: int32 CPU tensor of cumulative query-token counts over the
            same active documents, the rebased ``actual_seq_qlen``. The TND
            operators pair the q and KV sub-sequences of one document, so
            both sides must drop the dead-head documents together.
        total_tokens: Total compacted tokens, equal to ``kv_ends[-1]``.
    """

    batch_size: int
    row_stride: int
    windows: Tuple[Tuple[int, int], ...]
    kv_ends: torch.Tensor
    q_ends: torch.Tensor
    total_tokens: int
    # Per-device materializations of the selection index; mutable content in
    # an otherwise frozen plan.
    _index_cache: dict = field(default_factory=dict, compare=False, repr=False)

    def select_index(self, device: torch.device) -> torch.Tensor:
        """Return the flat int64 selection index materialized on ``device``.

        Args:
            device: Device the selection index is gathered on.

        Returns:
            The concatenated window ranges as an int64 index tensor; the
            per-device materialization is cached on the plan.
        """
        key = device
        index = self._index_cache.get(key)
        if index is None:
            pieces = [
                torch.arange(start, end, dtype=torch.int64)
                for start, end in self.windows
            ]
            index = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.int64)
            index = index.to(device)
            self._index_cache[key] = index
        return index


def _normalize_document_ends(
    actual_seq_kvlen: Sequence[int] | torch.Tensor | Iterable[int],
    total_tokens: int,
) -> List[int]:
    """Normalize cumulative document ends: dedupe, drop empties, validate.

    Args:
        actual_seq_kvlen: Cumulative document end positions over the whole
            ``[batch_size * global_seq_len]`` flat stream.
        total_tokens: Expected covered tokens (``batch_size *
            global_seq_len``).

    Returns:
        The strictly increasing list of document ends.

    Raises:
        ValueError: If the ends decrease, exceed the stream, or do not cover
            it exactly.
    """
    ends: List[int] = []
    previous = 0
    for raw_end in actual_seq_kvlen:
        end = int(raw_end)
        if end < previous:
            raise ValueError(
                f"document ends must be non-decreasing, got {end} after {previous}"
            )
        if end > total_tokens:
            raise ValueError(
                f"document end {end} exceeds the packed stream of {total_tokens} tokens"
            )
        if end > previous:
            ends.append(end)
            previous = end
    if previous != total_tokens:
        raise ValueError(
            f"document ends cover {previous} tokens, expected {total_tokens}; "
            "compaction requires full-length coverage"
        )
    return ends


def _row_document_ends(ends: Sequence[int], row_base: int, row_length: int) -> List[int]:
    """Return the within-row document ends for one batch row.

    A document crossing a row boundary is split at the row edge: rows are
    isolated in attention, so the within-row segment behaves as its own
    document. The returned ends tile ``[0, row_length)`` exactly — the
    document holding the row's last token contributes ``row_length`` even
    though its global end lies beyond the row.
    """
    row_limit = row_base + row_length
    row_ends = [end - row_base for end in ends if row_base < end <= row_limit]
    if not row_ends or row_ends[-1] != row_length:
        # The last document of the row extends past the row edge: close the
        # row's document list at the edge.
        row_ends.append(row_length)
    return row_ends


def build_compact_plan(
    actual_seq_kvlen: Sequence[int] | torch.Tensor | Iterable[int],
    *,
    batch_size: int,
    global_seq_len: int,
    local_q_start: int,
    local_q_end: int,
) -> Optional[CompactPlan]:
    """Build the compaction plan for one query block under fixed CP.

    Every batch row contributes the same query range ``[local_q_start,
    local_q_end)`` (CP chunk offsets), but each row owns its own documents,
    so the earliest active document differs per row. Row ``b``'s window is
    ``[d0_b, local_q_end)`` where ``d0_b`` is the start of the document
    containing the row's first query token.

    Args:
        actual_seq_kvlen: Cumulative document ends over the whole flat stream.
        batch_size: Batch rows.
        global_seq_len: Unsharded per-row length (``cp_size *
            local_seq_len``).
        local_q_start: Within-row start of this rank's query block.
        local_q_end: Within-row end of the query block; also the causal
            prefix end this rank's KV needs.

    Returns:
        The plan, or ``None`` when every row's earliest active document
        starts at 0 — the causal prefix already is the active window, so the
        caller keeps the plain prefix path (no layout or metadata changes).

    Raises:
        ValueError: If the boundaries are malformed (see
            :func:`_normalize_document_ends`) or the query block is empty or
            outside the row.
    """
    if not 0 <= local_q_start < local_q_end <= global_seq_len:
        raise ValueError(
            f"query block [{local_q_start}, {local_q_end}) must lie inside "
            f"[0, {global_seq_len})"
        )
    total_tokens = batch_size * global_seq_len
    ends = _normalize_document_ends(actual_seq_kvlen, total_tokens)

    windows: List[Tuple[int, int]] = []
    kv_counts: List[int] = []
    q_counts: List[int] = []
    running = 0
    q_running = 0
    all_trivial = True
    for row in range(batch_size):
        row_base = row * global_seq_len
        query_start = row_base + local_q_start
        # Start of the document containing the row's first query token:
        # the largest document end at or before it, clamped to the row.
        d0 = 0
        for end in ends:
            if end <= query_start:
                d0 = end
            else:
                break
        within_row_start = max(d0, row_base) - row_base
        all_trivial = all_trivial and within_row_start == 0
        # A row whose earliest active document starts at 0 keeps its full
        # causal prefix as the window; only the dead head is dropped.
        windows.append(
            (row * global_seq_len + within_row_start, row * global_seq_len + local_q_end)
        )
        # Document token counts inside [within_row_start, local_q_end). The
        # same documents' query-token counts inside [local_q_start,
        # local_q_end) keep the TND sub-sequences paired: dead-head documents
        # (no KV and no queries in the window) drop from both sides together.
        row_ends = _row_document_ends(ends, row_base, global_seq_len)
        previous = 0
        for end in row_ends:
            count = min(end, local_q_end) - max(previous, within_row_start)
            if count > 0:
                running += count
                kv_counts.append(running)
            q_count = min(end, local_q_end) - max(previous, local_q_start)
            if q_count > 0:
                q_running += q_count
                q_counts.append(q_running)
            previous = end

    if all_trivial:
        # The earliest active document starts at 0 in every row: the causal
        # prefix already is the active window, keep the plain prefix path.
        return None
    if running != sum(end - start for start, end in windows):
        raise RuntimeError(
            "compaction plan is inconsistent: metadata counts do not cover "
            "the selected windows"
        )
    return CompactPlan(
        batch_size=batch_size,
        row_stride=global_seq_len,
        windows=tuple(windows),
        kv_ends=torch.tensor(kv_counts, dtype=torch.int32),
        q_ends=torch.tensor(q_counts, dtype=torch.int32),
        total_tokens=running,
    )


def _plan_cache() -> "OrderedDict[tuple, Optional[CompactPlan]]":
    """Return this context's plan cache, creating it on first use."""
    cache = _COMPACT_PLAN_CACHE.get()
    if cache is None:
        cache = OrderedDict()
        _COMPACT_PLAN_CACHE.set(cache)
    return cache


def get_compact_plan(
    actual_seq_kvlen: Sequence[int] | torch.Tensor | Iterable[int],
    *,
    batch_size: int,
    global_seq_len: int,
    local_q_start: int,
    local_q_end: int,
) -> Optional[CompactPlan]:
    """Return the cached compaction plan for this metadata, building it once.

    Every DSA layer of one micro-batch calls with identical metadata, so the
    plan is built once per step per rank and shared across layers. The cache
    is content-addressed and bounded; a stale plan from another micro-batch
    can never be served because the boundaries are part of the key.

    Args:
        actual_seq_kvlen: Cumulative KV document end positions over the
            whole flat stream.
        batch_size: Number of packed sequences in the batch.
        global_seq_len: Per-rank query-block length times the CP world size.
        local_q_start: First global query token of this rank's block.
        local_q_end: One past the last global query token of this rank's
            block.

    Returns:
        The plan describing this rank's compact KV windows.
    """
    key = (
        tuple(int(value) for value in actual_seq_kvlen),
        batch_size,
        global_seq_len,
        local_q_start,
        local_q_end,
    )
    cache = _plan_cache()
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    plan = build_compact_plan(
        actual_seq_kvlen,
        batch_size=batch_size,
        global_seq_len=global_seq_len,
        local_q_start=local_q_start,
        local_q_end=local_q_end,
    )
    cache[key] = plan
    while len(cache) > _COMPACT_PLAN_CACHE_LIMIT:
        cache.popitem(last=False)
    return plan


class _CompactPrefixPair(torch.autograd.Function):
    """Autograd bridge for the window selection and its transpose.

    The forward selects the plan's windows from the flat gathered KV; the
    backward is the exact transpose: selected positions scatter their
    gradients back (accumulating overlaps), everything else gets zeros. Each
    forward saves the selection index it used, so every backward replays its
    own plan regardless of cache state.
    """

    @staticmethod
    def forward(ctx: Any, kv_flat: torch.Tensor, select_index: torch.Tensor) -> torch.Tensor:
        """Select ``select_index`` rows from the flat gathered KV.

        Args:
            ctx: Autograd context receiving the saved selection index.
            kv_flat: Flat gathered KV ``[batch_size * row_stride, ...]``.
            select_index: Flat int64 selection index of the plan's windows.

        Returns:
            The compacted KV ``[total_tokens, ...]``.
        """
        ctx.save_for_backward(select_index)
        ctx.kv_shape = kv_flat.shape
        return kv_flat.index_select(0, select_index)

    @staticmethod
    def backward(ctx: Any, grad_selected: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Scatter-add the compacted gradient back to the gathered layout.

        Args:
            ctx: Autograd context holding the saved selection index.
            grad_selected: Gradient of the compacted KV output.

        Returns:
            The gradient with respect to the flat gathered KV (overlapping
            windows accumulate) and ``None`` for the selection index.
        """
        (select_index,) = ctx.saved_tensors
        grad_flat = torch.zeros(
            ctx.kv_shape, dtype=grad_selected.dtype, device=grad_selected.device
        )
        if grad_selected.numel():
            grad_flat.index_put_((select_index,), grad_selected, accumulate=True)
        return grad_flat, None


def interval_select(kv: torch.Tensor, plan: CompactPlan) -> torch.Tensor:
    """Select the plan's windows from one gathered KV tensor.

    Args:
        kv: Gathered KV tensor ``[batch_size, row_stride, ...]``.
        plan: The compaction plan describing this forward's windows.

    Returns:
        The compacted KV as ``[1, total_tokens, ...]``: TND-packed with the
        plan's ``kv_ends`` as its coordinate carrier. No full-length natural
        copy is materialized; only the selected tokens are touched.

    Raises:
        ValueError: If the tensor shape disagrees with the plan.
    """
    if kv.shape[0] != plan.batch_size or kv.shape[1] != plan.row_stride:
        raise ValueError(
            f"interval_select expects [{plan.batch_size}, {plan.row_stride}, ...], "
            f"got {tuple(kv.shape)}"
        )
    trailing = tuple(kv.shape[2:])
    flat = kv.reshape(plan.batch_size * plan.row_stride, *trailing)
    selected = _CompactPrefixPair.apply(flat, plan.select_index(flat.device))
    return selected.reshape(1, plan.total_tokens, *trailing)


def compact_pair_grad(
    grad_kv: torch.Tensor,
    plan: CompactPlan,
    kv_shape: Sequence[int],
) -> torch.Tensor:
    """Map a compacted KV gradient back to the gathered layout.

    Overlapping windows accumulate their gradients; positions no window
    selected get zeros. This is the exact transpose of
    :func:`interval_select` and the backward :class:`_CompactPrefixPair`
    applies.

    Args:
        grad_kv: Gradient of the compacted KV ``[total_tokens, ...]``.
        plan: The forward's plan.
        kv_shape: Shape of the flat gathered KV the windows selected from.

    Returns:
        The gradient with respect to the flat gathered KV.
    """
    grad_flat = torch.zeros(kv_shape, dtype=grad_kv.dtype, device=grad_kv.device)
    if grad_kv.numel():
        grad_flat.index_put_(
            (plan.select_index(grad_kv.device),), grad_kv, accumulate=True
        )
    return grad_flat
