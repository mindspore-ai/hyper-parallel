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
"""MoonEP-backed expert-parallel MoE execution.

MoonEP owns the token movement *and* the layout: each rank receives the rows its
own experts must compute plus, when the router skews past the per-rank capacity
``NvS = S * K``, the overflow of remote experts it recomputes from locally
available weights (MoonEP's redundant-expert balancing).  ``moonep_plan`` builds
that layout on the host; this module drives the kernels and keeps the path
differentiable:

* ``_MoonEPDispatch`` scatters rows and route weights; its backward is the very
  same combine, so gradients return to the source rows and weights.
* ``_MoonEPCombine`` gathers expert outputs; its backward is a re-dispatch.
* The caller supplies ``expert_forward``, called once per receive-buffer segment
  with that segment's rows and global expert id - which is where redundant
  segments get another rank's weights.

Not covered here: MoonEP's ``prefetch_weight`` / ``reduce_grad``.  Those expect
the expert weights to live in MoonEP's symmetric memory (``[E+B, M, N]`` VMM
tensors carrying the owner/slot source VAs), which is a parameter-layout change
for the framework rather than a call site, so this module takes the redundant
experts' weights from the caller instead.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.distributed.expert_parallel.moonep_plan import build_moonep_plan

__all__ = ["MoonEPExpertParallel"]

logger = logging.getLogger(__name__)

# Process-wide MoonEP buffer cache, keyed by group + buffer geometry.
_SHARED_BUFFERS: dict[tuple, dict[int, object]] = {}
# Opt-in cross-layer scratch sharing (HP_MOONEP_SHARED_SCRATCH=1), see
# MoonEPExpertParallel.__init__ for the safety condition.
_SHARED_ROW_SCRATCH: dict[tuple, list[torch.Tensor]] = {}
_SHARED_ROW_TURN: dict[tuple, int] = {}
_SHARED_COMBINE_SCRATCH: dict[tuple, torch.Tensor] = {}
_SHARED_SCRATCH_WARNED = [False]

# Opt-in per-segment wall-clock breakdown (HP_MOONEP_PROFILE=1).  Each segment
# is bracketed with a device synchronize, so enabling it serialises the pipeline
# and is meant for offline attribution, never for production runs.  Rank 0 of
# each EP group prints the per-call averages every ``_PROF_WINDOW`` layer calls
# and resets the accumulators.
_PROF_ENABLED = os.environ.get("HP_MOONEP_PROFILE", "0") == "1"
_PROF_WINDOW = 480
_PROF: dict[str, list[float]] = {}
_PROF_CALLS = [0]
_PROF_MAXVIO = [0.0, 0, 0.0]  # sum, count, max


def _prof_start() -> float | None:
    """Start a profiled segment (None when profiling is off)."""
    if not _PROF_ENABLED:
        return None
    torch.npu.synchronize()
    return time.perf_counter()


def _prof_mark(name: str, start: float | None) -> None:
    """Close a profiled segment started by :func:`_prof_start`."""
    if start is None:
        return
    torch.npu.synchronize()
    slot = _PROF.setdefault(name, [0.0, 0])
    slot[0] += time.perf_counter() - start
    slot[1] += 1


def _prof_report(transport: MoonEPExpertParallel) -> None:
    """Print and reset the accumulated breakdown every ``_PROF_WINDOW`` calls."""
    if not _PROF_ENABLED:
        return
    _PROF_CALLS[0] += 1
    if _PROF_CALLS[0] % _PROF_WINDOW or transport.rank != 0 or not _PROF:
        return
    parts = [f"{name}={slot[0] / max(slot[1], 1) * 1e3:.2f}ms"
             for name, slot in sorted(_PROF.items())]
    if _PROF_MAXVIO[1]:
        parts.append(f"maxvio_avg={_PROF_MAXVIO[0] / _PROF_MAXVIO[1]:.3f}")
        parts.append(f"maxvio_max={_PROF_MAXVIO[2]:.3f}")
    print(f"[moonep-prof] calls={_PROF_CALLS[0]} " + " ".join(parts), flush=True)
    _PROF.clear()
    _PROF_MAXVIO[0] = 0.0
    _PROF_MAXVIO[1] = 0
    _PROF_MAXVIO[2] = 0.0


def _moonep_runtime():
    """Import the installed MoonEP Ascend runtime lazily (optional dependency)."""
    # Keep the optional NPU dependency out of ordinary HyperParallel imports.
    try:
        from moonep import (  # pylint: disable=import-outside-toplevel
            Buffer,
            MoonEPCommPlan,
        )
        from npu_patch import (  # pylint: disable=import-outside-toplevel
            ShmemRuntime,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The MoonEP EP path requires the Ascend MoonEP package built for the current "
            "SoC and torch/torch_npu versions. Build and install it from MoonEP/npu_patch "
            f"as described in MoonEP/README_910C.md. Import failed: {exc}"
        ) from exc
    return Buffer, ShmemRuntime, MoonEPCommPlan


@dataclass
class _PlanState:
    """Per-call state shared by the dispatch/combine autograd pair."""

    transport: object
    plan: object
    received_rows: int
    layout: list[tuple[int, int, int]]  # (expert id, start row, end row)
    local_counts: torch.Tensor          # [epn] rows per local expert, zeros included
    needed_remote: torch.Tensor         # experts whose rows land off their owner


def _all_ranks_plan_cache_hit(local_hit: bool, device: torch.device,
                              group: Any, world_size: int) -> bool:
    """Return whether every rank in the EP group has a valid cached plan.

    A plan depends on routing counts from every rank, so local route fingerprints
    are sufficient only when every rank reports a cache hit.
    """
    hit_count = torch.tensor([int(local_hit)], dtype=torch.int32, device=device)
    dist.all_reduce(hit_count, group=group)
    return int(hit_count.item()) == world_size


def _segment_layout(cu: torch.Tensor, copies: torch.Tensor,
                    experts: int) -> tuple[list[tuple[int, int, int]], int]:
    """Decode the receive buffer into ``(expert, start, end)`` segments.

    Groups ``[0, E)`` are global-expert groups; groups ``[E, E + B)`` are the
    prefetch slots named by ``experts_to_copy``.  MoonEP's plan places every
    delivered row inside one of them, so walking the boundaries enumerates the
    whole buffer.

    Args:
        cu: This rank's ``[E + B]`` group boundaries.
        copies: This rank's ``[B]`` prefetch expert ids.
        experts: Total routed experts.

    Returns:
        ``(segments, received_rows)``.
    """
    ends = cu.long()
    # Group boundaries are a plain cumulative sum, so the occupied groups are one
    # `nonzero` away - walking all E + B of them in Python costs milliseconds at
    # E=896 while only ~E/R of them hold rows.
    starts = torch.cat([torch.zeros(1, dtype=ends.dtype), ends[:-1]])
    counts = ends - starts
    occupied = torch.nonzero(counts > 0, as_tuple=False).flatten()
    slots = (occupied - experts).clamp(min=0)
    expert_ids = torch.where(occupied < experts, occupied, copies[slots].long())
    if bool((expert_ids < 0).any()):
        raise RuntimeError(
            "MoonEP plan puts rows in a prefetch slot that names no expert; the "
            "plan and the buffer disagree.")
    layout = [(int(expert), int(start), int(start + count))
              for expert, start, count in zip(expert_ids, starts[occupied], counts[occupied])]
    return layout, int(ends[-1])


class _MoonEPDispatch(torch.autograd.Function):
    """Scatter rows and route weights; backward is the matching combine."""

    @staticmethod
    # pylint: disable=arguments-differ,abstract-method
    def forward(ctx: Any, rows: torch.Tensor, weights: torch.Tensor,
                state: _PlanState) -> tuple[torch.Tensor, torch.Tensor]:
        """Move ``rows`` (and their route weights) to their experts' owners."""
        rows_local, weights_local = state.transport._dispatch_raw(rows, weights, state)
        ctx.state = state
        return rows_local, weights_local

    @staticmethod
    # pylint: disable=arguments-differ,abstract-method
    def backward(ctx: Any, grad_rows: torch.Tensor,
                 grad_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Return the output gradient to its source rows and route weights."""
        state = ctx.state
        # One combine serves both transposes: the row gather, and - through the
        # plan's weight path - the route-weight gather.
        tick = _prof_start()
        grad_x, grad_w = state.transport._combine_raw(
            grad_rows, state, grad_weights, want_weights=grad_weights is not None)
        _prof_mark("bwd_combine", tick)
        return grad_x, grad_w, None


class _MoonEPCombine(torch.autograd.Function):
    """Gather expert outputs; backward is a re-dispatch."""

    @staticmethod
    # pylint: disable=arguments-differ,abstract-method
    def forward(ctx: Any, rows_local: torch.Tensor,
                state: _PlanState) -> torch.Tensor:
        """Gather the per-segment expert outputs back to their source rows."""
        ctx.state = state
        combined, _weights = state.transport._combine_raw(rows_local, state, None)
        return combined

    @staticmethod
    # pylint: disable=arguments-differ,abstract-method
    def backward(ctx: Any, grad_rows: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Re-dispatch the output gradient so each expert sees its own rows."""
        state = ctx.state
        tick = _prof_start()
        rows_local, _weights = state.transport._dispatch_raw(grad_rows, None, state)
        _prof_mark("bwd_dispatch", tick)
        return rows_local, None


class MoonEPExpertParallel:
    """Run one MoE layer's expert computation over MoonEP's token transport.

    Args:
        group: EP process group.
        num_experts: Total routed experts ``E``.
        hidden: Row width ``H``.
        rank: This rank's EP index (defaults to its rank in *group*).
        num_sms: Kernel blocks MoonEP may use.

    Note:
        MoonEP sizes its symmetric chunks from the routed-slot count, so one
        Buffer is cached per ``rows`` this rank can present.
    """

    def __init__(self, group: Any, num_experts: int, hidden: int,
                 rank: int | None = None, num_sms: int = 32) -> None:
        """Bind the EP group and size this rank's expert shard.

        Args:
            group: EP process group.
            num_experts: Total routed experts ``E``.
            hidden: Row width ``H``.
            rank: This rank's EP index (defaults to its rank in *group*).
            num_sms: Kernel blocks MoonEP may use.
        """
        self.group = group
        self.ep_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group) if rank is None else rank
        self.num_experts = num_experts
        self.hidden = hidden
        self.num_sms = num_sms
        if num_experts % self.ep_size:
            raise ValueError(
                f"num_experts ({num_experts}) must divide EP size ({self.ep_size})")
        self.local_experts = num_experts // self.ep_size
        # One MoonEP buffer per (group, shape) for the whole process: every MoE
        # layer of a model presents the same shape, and each buffer carves its
        # chunks out of one shmem heap that is sized for the first buffer.  A
        # per-layer buffer therefore exhausts the heap part-way through a deep
        # model.  Everything below (row scratch, plan state) stays per layer:
        # activation-checkpoint recompute drives forwards and backwards of
        # different layers interleaved, so sharing scratch would let one layer
        # overwrite rows another layer's backward still needs.
        self._buffers: dict[int, object] = _SHARED_BUFFERS.setdefault(self._buffer_key(), {})
        self._scratch_namespace = self._buffer_key()
        # Scratch reused across calls: the receive rows are materialised (the
        # symmetric buffer is rewritten by the backward re-dispatch) and the
        # combine input must be a full [NvS, H] tensor.  Allocating both per call
        # costs hundreds of MB of memset/copy at MoE shape, and `combine` only
        # ever reads the rows the plan names, so the tail can stay stale.
        # Two row buffers, used alternately: the backward re-dispatch overwrites
        # whatever the forward returned, and the expert ops upstream still need
        # those values for their own backward.
        if os.environ.get("HP_MOONEP_SHARED_SCRATCH", "0") == "1":
            # Opt-in: one scratch set per rows-key shared by every layer.  Only
            # safe when full activation checkpointing serialises layer
            # forwards/backwards (SAC recomputes one layer at a time, so no two
            # layers' rows are live together); without it, a later layer's
            # forward would overwrite rows an earlier layer's backward needs.
            if not _SHARED_SCRATCH_WARNED[0]:
                _SHARED_SCRATCH_WARNED[0] = True
                logger.warning(
                    "HP_MOONEP_SHARED_SCRATCH=1: sharing MoonEP row/combine "
                    "scratch across layers.  This is only safe with FULL "
                    "activation checkpointing; without it the training results "
                    "are silently wrong.")
            self._row_scratch = _SHARED_ROW_SCRATCH
            self._row_turn = _SHARED_ROW_TURN
            self._combine_scratch = _SHARED_COMBINE_SCRATCH
        else:
            self._row_scratch: dict[tuple, list[torch.Tensor]] = {}
            self._row_turn: dict[tuple, int] = {}
            self._combine_scratch: dict[tuple, torch.Tensor] = {}
        # Plan reuse across activation-checkpoint recompute: full SAC runs every
        # layer's forward twice per step with identical routing. Cache entries
        # are rank-local, so reuse is allowed only after an EP-wide hit vote.
        # Disable with HP_MOONEP_PLAN_CACHE=0.
        self._plan_cache: tuple[bytes, object, _PlanState] | None = None
        self._plan_cache_on = os.environ.get("HP_MOONEP_PLAN_CACHE", "1") == "1"

    def _buffer_key(self) -> tuple:
        """Identify the buffer geometry; identical layers share one buffer."""
        return (id(self.group), self.hidden, self.num_experts, self.ep_size,
                self.local_experts, self.num_sms)

    def _buffer_for(self, rows: int):
        """Return this process's MoonEP buffer for *rows* routed slots."""
        buffer = self._buffers.get(rows)
        if buffer is not None:
            return buffer
        buffer_type, shmem_runtime, _plan_type = _moonep_runtime()
        buffer = buffer_type(
            rows, self.hidden, 1, self.num_experts, self.ep_size,
            B=self.local_experts, num_sms=self.num_sms, use_udma=False,
            group=self.group, explicitly_destroy=True,
        )
        if not shmem_runtime.is_initialized():
            shmem_runtime.init_with_buffer(group=self.group, enable_udma=False)
        self._buffers[rows] = buffer
        return buffer

    def _make_plan(self, counts: torch.Tensor, route_ids: torch.Tensor, rows: int,
                   route_ids_cpu: torch.Tensor | None = None):
        """Build this step's plan and decode its receive layout.

        Args:
            counts: ``[E]`` rows per expert on this rank.
            route_ids: ``[rows]`` global expert id of each row, in the caller's
                row order.
            rows: Routed slots this rank presents.
            route_ids_cpu: Optional pre-copied CPU int32 route ids (reused from
                the plan-cache fingerprint to save a second D2H copy).
        """
        _buffer_type, _shmem_runtime, plan_type = _moonep_runtime()
        if route_ids.dim() != 1 or route_ids.numel() != rows:
            raise ValueError(
                f"the MoonEP plan needs one expert id per row ({rows}), got shape "
                f"{tuple(route_ids.shape)}")
        if counts.numel() != self.num_experts:
            # ``bincount`` grows when an id is outside the expert range, so this
            # also catches out-of-range routing.
            raise ValueError(
                f"routing produced {counts.numel()} expert buckets for "
                f"{self.num_experts} experts; every route id must be in "
                f"[0, {self.num_experts})")
        counts_all = [torch.empty_like(counts, dtype=torch.int32)
                      for _ in range(self.ep_size)]
        dist.all_gather(counts_all, counts.to(torch.int32), group=self.group)
        counts_all = torch.stack(counts_all)
        if _PROF_ENABLED:
            # Routing skew metric (same definition as the comm benchmarks):
            # MaxVio = max per-expert load / mean - 1, over the group-wide
            # totals.  Tracked per _make_plan call (once per layer per step
            # with the plan cache on), reported as mean/max by _prof_report.
            totals = counts_all.sum(dim=0).float()
            mv = float(totals.max() / totals.mean() - 1.0)
            _PROF_MAXVIO[0] += mv
            _PROF_MAXVIO[1] += 1
            _PROF_MAXVIO[2] = max(_PROF_MAXVIO[2], mv)
        fields = build_moonep_plan(
            counts_all, counts.to(torch.int32), route_ids.to(torch.int32),
            self.rank, self.local_experts, rows, route_ids_cpu=route_ids_cpu)
        device = counts.device
        plan = plan_type(
            dst=fields["dst"], experts_to_copy=fields["experts_to_copy"],
            zero_fill_ranges=fields["zero_fill_ranges"],
            remote_stats=fields["remote_stats"], N=rows, R=self.ep_size,
            E=self.num_experts, B=self.local_experts, NvS=rows, K=1,
            dup_groups=torch.zeros((rows, 3), dtype=torch.int32, device=device),
            dup_loffs=torch.zeros((rows,), dtype=torch.int32, device=device),
            dup_counts=torch.zeros((2,), dtype=torch.int32, device=device),
        )
        layout, received = _segment_layout(
            fields["cu_seqlens"].cpu(), fields["experts_to_copy"][self.rank].cpu(),
            self.num_experts)
        if received > rows:
            raise RuntimeError(
                f"the MoonEP plan delivers {received} rows to rank {self.rank} but the "
                f"receive buffer holds {rows}; the plan and the buffer disagree")
        # Per-local-expert row counts (zero where a local expert got nothing):
        # the form a grouped GEMM wants next to its weight table.
        ends = fields["cu_seqlens"].to(torch.int64)
        lo = self.rank * self.local_experts
        local_counts = ends[lo:lo + self.local_experts].clone()
        if self.local_experts > 1:
            local_counts[1:] = local_counts[1:] - ends[lo:lo + self.local_experts - 1]
        return plan, _PlanState(self, plan, received, layout, local_counts,
                                fields["needed_remote"])

    def _dispatch_raw(self, rows: torch.Tensor, weights, state: _PlanState):
        """Scatter ``rows`` (and optional ``weights``) to their experts' owners."""
        buffer = self._buffer_for(int(state.plan.N))
        route_weights = None if weights is None else weights.reshape(-1, 1).contiguous()
        rows_buffer, weights_buffer, _cu, _plan = buffer.dispatch(
            rows, route_weights_sk=route_weights, plan=state.plan, zero_copy=True)
        # Materialise the received rows: the symmetric buffer is written again
        # by the backward re-dispatch, and autograd refuses to keep a view whose
        # base gets modified (a second Buffer for the backward would remove this
        # copy).
        rows = int(state.plan.N)
        scratch_key = (self._scratch_namespace, rows, rows_buffer.device, rows_buffer.dtype)
        pair = self._row_scratch.get(scratch_key)
        if pair is None:
            pair = [torch.empty_like(rows_buffer[:rows]) for _ in range(2)]
            self._row_scratch[scratch_key] = pair
        turn = self._row_turn.get(scratch_key, 0)
        self._row_turn[scratch_key] = 1 - turn
        scratch = pair[turn]
        rows_local = scratch[:state.received_rows]
        rows_local.copy_(rows_buffer[:state.received_rows])
        weights_local = (None if weights_buffer is None
                         else weights_buffer[:state.received_rows].clone())
        if os.environ.get("HP_MOONEP_SELFCHECK") == "1":
            self._selfcheck(rows, weights, rows_local, weights_local, state)
        return rows_local, weights_local

    @staticmethod
    def _selfcheck(rows: torch.Tensor, weights, rows_local: torch.Tensor,
                   weights_local, state: _PlanState) -> None:
        """Verify the plan's slot encoding against what the kernel actually moved.

        Row ``i`` of the source is encoded to land at ``dst[i] % NvS`` of its
        destination's buffer; on the receiving rank that is row
        ``dst[i] % NvS`` of the receive buffer.  This checks the whole encoding
        (expert choice, group offset, slot) rather than inferring it indirectly
        from the layer's output.
        """
        num_rows = int(state.plan.N)
        dest = (state.plan.dst // num_rows).long()
        slot = (state.plan.dst % num_rows).long()
        mine = dest == state.transport.rank
        if int(mine.sum()) == 0:
            return
        expected_rows = rows_local[slot[mine]]
        got_rows = rows[mine]
        row_bad = int((expected_rows != got_rows).any(dim=1).sum())
        weight_bad = -1
        if weights is not None and weights_local is not None:
            weight_bad = int((weights_local[slot[mine]] != weights[mine]).sum())
        print(f"[selfcheck][rank {state.transport.rank}] rows_here={int(mine.sum())} "
              f"row_mismatch={row_bad} weight_mismatch={weight_bad}", flush=True)

    def _combine_raw(self, rows_local: torch.Tensor, state: _PlanState, weights_local,
                     want_weights: bool = False):
        """Gather local rows (and optionally a weight gradient) back to sources."""
        buffer = self._buffer_for(int(state.plan.N))
        rows = int(state.plan.N)
        scratch_key = (self._scratch_namespace, rows, rows_local.device, torch.bfloat16)
        padded_rows = self._combine_scratch.get(scratch_key)
        if padded_rows is None:
            padded_rows = torch.zeros((rows, self.hidden), dtype=torch.bfloat16,
                                      device=rows_local.device)
            self._combine_scratch[scratch_key] = padded_rows
        tick = _prof_start()
        padded_rows[:rows_local.shape[0]].copy_(rows_local.to(torch.bfloat16))
        padded_weights = None
        if want_weights and weights_local is not None:
            padded_weights = torch.zeros((rows,), dtype=torch.float32,
                                         device=rows_local.device)
            padded_weights[:weights_local.shape[0]].copy_(weights_local)
        _prof_mark("combine_pad", tick)
        tick = _prof_start()
        combined, gathered, _event = buffer.combine(
            plan=state.plan, hidden_nvsh=padded_rows, route_weights_nvs=padded_weights)
        _prof_mark("combine_kernel", tick)
        if gathered is None:
            return combined, None
        return combined, gathered.reshape(-1)

    def forward(self, rows: torch.Tensor, route_ids: torch.Tensor,
                route_weights: torch.Tensor, expert_forward: Callable | None,
                output_dtype: torch.dtype | None = None,
                expert_forward_many: Callable | None = None,
                expert_forward_factory: Callable | None = None,
                requires_route_weights: bool = False) -> torch.Tensor:
        """Dispatch, run experts segment by segment, and combine.

        Args:
            rows: ``[rows, H]`` routed rows for this rank.
            route_ids: ``[rows]`` global expert id of each row.
            route_weights: ``[rows]`` per-row routing weight.
            expert_forward: ``expert_forward(segment_rows, expert_id,
                segment_weights)`` returning that segment's output, with the
                routing weight applied where the caller's expert semantics want
                it; called once per received segment, in receive-buffer order,
                which is also the order ``combine`` needs.  May be ``None`` when
                ``expert_forward_factory`` is given.
            output_dtype: Cast the combined rows to this dtype (default keeps
                MoonEP's bf16 output).
            expert_forward_many: Optional batched alternative,
                ``expert_forward_many(rows_local, local_counts, weights_local)``
                returning the whole ``[received_rows, H]`` output.  It is used
                when this rank only holds its *own* experts (no rebalancing), so
                the segments line up with its local weight table and the caller
                can run one grouped GEMM instead of a kernel per segment.
                ``expert_forward`` stays the fallback.
            requires_route_weights: Require weight transport for callbacks that
                consume routing scores, regardless of the global optimization flag.
            expert_forward_factory: Optional
                ``expert_forward_factory(needed_remote) -> expert_forward``.
                Called right after the plan is built with the set of experts
                whose rows land off their owner, so the callback it returns can
                be specialised for this step's remote segments (e.g. gather
                only those experts' weights).

        Returns:
            ``[rows, H]`` outputs in the caller's original row order.
        """
        tick = _prof_start()
        route_ids_cpu = (route_ids.to("cpu", torch.int32)
                         if self._plan_cache_on else None)
        fingerprint = (route_ids_cpu.numpy().tobytes()
                       if route_ids_cpu is not None else None)
        cached = self._plan_cache if fingerprint is not None else None
        local_cache_hit = (fingerprint is not None and cached is not None
                           and cached[0] == fingerprint)
        cache_hit = (
            _all_ranks_plan_cache_hit(
                local_cache_hit, route_ids.device, self.group, self.ep_size,
            )
            if fingerprint is not None else False
        )
        if cache_hit:
            # SAC recompute replay: same routing as this layer's first forward,
            # on every rank, so the cached global plan is still valid.
            _plan, state = cached[1], cached[2]
        else:
            counts = torch.bincount(route_ids.long(), minlength=self.num_experts)
            # The fingerprint's D2H copy doubles as the planner's input copy.
            _plan, state = self._make_plan(counts, route_ids, int(rows.shape[0]),
                                           route_ids_cpu=(route_ids_cpu if os.environ.get(
                                               "HP_MOONEP_REUSE_ROUTE_CPU", "1") == "1" else None))
            if fingerprint is not None:
                self._plan_cache = (fingerprint, _plan, state)
        _prof_mark("plan", tick)
        tick = _prof_start()
        if expert_forward_factory is not None:
            # The factory sees the plan's remote-expert set, so the caller can
            # gather only the weights the off-owner segments actually need
            # instead of the whole expert table.
            expert_forward = expert_forward_factory(state.needed_remote)
        _prof_mark("wgather", tick)
        tick = _prof_start()
        # The caller weights the combined rows locally, so the route weights
        # are never read on the receive side; shipping them would only buy a
        # redundant transport path plus its backward weight gather
        # (HP_MOONEP_ROUTE_WEIGHTS=1 restores the old behaviour).
        transport_weights = (route_weights
                             if requires_route_weights or os.environ.get("HP_MOONEP_ROUTE_WEIGHTS", "0") == "1"
                             else None)
        rows_local, weights_local = _MoonEPDispatch.apply(rows, transport_weights, state)
        _prof_mark("dispatch", tick)
        tick = _prof_start()

        grouped = (expert_forward_many is not None
                   and os.environ.get("HP_MOONEP_GROUPED", "0") == "1")
        if grouped:
            # Hybrid: one grouped GEMM over the local-expert block (the bulk
            # of the rows) plus per-segment callbacks for remote segments
            # (prefetch slots and unprefetched remote groups).  The previous
            # all-or-nothing trigger (_owns_every_segment) almost never fires
            # because capacity == mean makes rebalancing the norm (measured 0%
            # at sigma=0.2..1.0); the hybrid fires on every layout.  Local
            # experts occupy a contiguous group range, so their rows form one
            # contiguous block described by state.local_counts.
            lo = self.rank * self.local_experts
            local_segments = [(e, s, t) for e, s, t in state.layout
                              if lo <= e < lo + self.local_experts]
            remote_segments = [(e, s, t) for e, s, t in state.layout
                               if not lo <= e < lo + self.local_experts]
            # empty, not zeros: the layout's segments tile [0, received) exactly
            # (empty groups contribute zero-width segments), so every row is
            # written below and a memset would be pure waste.
            output_factory = (torch.empty if os.environ.get("HP_MOONEP_EMPTY_OUTPUT", "1") == "1"
                              else torch.zeros)
            out_local = output_factory(rows_local.shape[0], self.hidden, dtype=torch.bfloat16,
                                    device=rows_local.device)
            if local_segments:
                start = local_segments[0][1]
                end = local_segments[-1][2]
                wseg = None if weights_local is None else weights_local[start:end]
                out_local[start:end] = expert_forward_many(
                    rows_local[start:end], state.local_counts, wseg)
            for expert, start, end in remote_segments:
                wseg = None if weights_local is None else weights_local[start:end]
                out_local[start:end] = expert_forward(
                    rows_local[start:end], expert, wseg).float()
            _prof_mark("experts", tick)
        else:
            # bf16 staging: the combine consumes bf16 rows anyway (the padded
            # buffer is bf16 and its backward never reads these values, only
            # re-dispatches the gradient), so fp32 here would just double the
            # per-layer autograd retention at MoE shapes.
            # empty, not zeros: the layout's segments tile [0, received) exactly
            # (empty groups contribute zero-width segments), so every row is
            # written below and a memset would be pure waste.
            output_factory = (torch.empty if os.environ.get("HP_MOONEP_EMPTY_OUTPUT", "1") == "1"
                              else torch.zeros)
            out_local = output_factory(rows_local.shape[0], self.hidden, dtype=torch.bfloat16,
                                    device=rows_local.device)
            for expert, start, end in state.layout:
                # The callback receives the segment's own transported weights so
                # it decides where the routing weight enters its expert math.
                wseg = None if weights_local is None else weights_local[start:end]
                out_local[start:end] = expert_forward(
                    rows_local[start:end], expert, wseg).float()
            _prof_mark("experts", tick)
        tick = _prof_start()
        combined = _MoonEPCombine.apply(out_local, state)
        _prof_mark("combine", tick)
        if output_dtype is not None:
            combined = combined.to(output_dtype)
        _prof_report(self)
        return combined

    def _owns_every_segment(self, state: _PlanState) -> bool:
        """Whether the receive layout is exactly this rank's own experts in order.

        True when the router stayed inside the per-rank capacity: every segment
        then belongs to a local expert, they arrive in ascending local-expert
        order, and the grouped GEMM's weight table lines up with the segments.
        Any rebalanced (redundant-expert) layout breaks that and falls back to
        the per-segment callback.
        """
        lo = self.rank * self.local_experts
        occupied = [index for index, count in enumerate(state.local_counts.tolist()) if count]
        return [expert for expert, _start, _end in state.layout] == [lo + index
                                                                    for index in occupied]
