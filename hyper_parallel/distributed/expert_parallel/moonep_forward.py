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
"""MoonEP as the token transport of the framework's routed-experts branch.

This is the production seam: :func:`moonep_routed_forward` has the same
contract as :func:`~hyper_parallel.distributed.expert_parallel.experts.ep_routed_forward`
- route locally, move the routed rows to the ranks that own their experts, run
the experts, bring the outputs back in token order and weight them - so
``build_ep_compute`` can select it without the archetypes or the model adapters
knowing which transport ran.

Two things differ from the a2a path, both of them MoonEP's design rather than
this module's choice:

* MoonEP's per-rank capacity is the *average* share of the routing, so a real
  router always overloads someone.  The overflow is recomputed on a rank that
  has room, from that expert's weights - the redundant-expert feature.  Those
  weights have to be visible group-wide, but only the few experts some rank
  actually prefetches: they are gathered differentiably into a compact table
  (``_gather_needed``), so the redundant segments' gradients land on the owner
  and balanced routing gathers nothing.
* Rows travel in the caller's order together with their expert id, so there is
  no expert-major sort and no index exchange: the plan decides where each row
  goes, and the combine returns them in the order they were presented.

``HP_MOONEP_EP=1`` selects this path from ``build_ep_compute``.  The transport
itself lives in :mod:`~hyper_parallel.distributed.expert_parallel.moonep_ep`.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hyper_parallel.components.functional.npu_grouped_swiglu import npu_grouped_swiglu
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.expert_parallel.experts import (
    apply_expert_projection,
    ep_routed_forward,
    resolve_swiglu_weights,
)
from hyper_parallel.distributed.expert_parallel.moonep_ep import MoonEPExpertParallel

__all__ = ["moonep_ep_enabled", "moonep_routed_forward"]

logger = logging.getLogger(__name__)

_ENV_ON_VALUES = ("1", "true", "yes", "on")


def moonep_ep_enabled() -> bool:
    """Whether the framework's EP path should move tokens with MoonEP.

    Returns:
        True when ``HP_MOONEP_EP`` is set to a truthy value (or ``auto``).
    """
    return os.environ.get("HP_MOONEP_EP", "").strip().lower() in (
        _ENV_ON_VALUES + ("auto",))


def _moonep_ep_auto() -> bool:
    """Whether shape-based routing between MoonEP and the a2a path is on."""
    return os.environ.get("HP_MOONEP_EP", "").strip().lower() == "auto"


def _moonep_min_rows() -> int:
    """Row threshold below which ``auto`` routes to the a2a path.

    Measured on Qwen3-30B-A3B (8 卡，2026-10-22，对称 grouped 口径，LIMO)：
    MoonEP 在 8k tokens/rank（65536 行）慢 3.3%，12k（98304 行）反超 -3.0%，
    16k -4.8%——交叉点位于 8k~12k tokens/rank 之间，阈值取实测安全的 98304
    行（12288 tokens/rank）。旧口径（a2a 未开 grouped GEMM）的 ~2k tokens/rank
    交叉点已不再适用。
    """
    return int(os.environ.get("HP_MOONEP_MIN_ROWS", "98304"))


_WGATHER_STREAM: list = [None]  # lazy per-process side stream for the gather
_WGATHER_GROUPS: dict[int, Any] = {}  # id(ep_group) -> dedicated gather group


def _wgather_async() -> bool:
    """Whether the redundant-expert weight gather runs on a side stream.

    Off by default.  The first attempt shared the EP group's HCCL communicator
    between the side stream (weight gather) and the main stream (plan's counts
    all_gather) and deadlocked in the first backward (2026-10-22): HCCL
    communicators are not safe for concurrent streams.  The current
    implementation gives the gather its own communicator
    (:func:`_wgather_group`), created once per EP group.
    """
    return os.environ.get("HP_MOONEP_WGATHER_ASYNC", "0") == "1"


def _wgather_stream() -> Any:
    """Lazy side stream overlapping the weight gather with dispatch/GEMM."""
    if _WGATHER_STREAM[0] is None:
        _WGATHER_STREAM[0] = torch.npu.Stream()
    return _WGATHER_STREAM[0]


def _wgather_group(ep_group: Any) -> Any:
    """Dedicated communicator for the weight gather (created once per EP group).

    HCCL communicators are not safe for concurrent streams, so the overlapped
    gather must not share the EP group's communicator with the plan's counts
    all_gather.  ``new_group`` keeps the rank order of the source group, so the
    compact-table layout (owner = rank // epn) is unchanged.  Every rank calls
    this at the same layer/step boundary, keeping the collective creation
    matched.
    """
    key = id(ep_group)
    group = _WGATHER_GROUPS.get(key)
    if group is None:
        group = dist.new_group(ranks=dist.get_process_group_ranks(ep_group))
        _WGATHER_GROUPS[key] = group
    return group


def _transport_for(module: Any, ep_group: Any, hidden: int) -> MoonEPExpertParallel:
    """Return this module's transport, built on first use and reused afterwards.

    The transport caches MoonEP buffers and cross-call scratch keyed by row
    count, so rebuilding it per forward would defeat both.  It is keyed by the
    process group as well: an adapter may be applied to a different mesh
    between runs (e.g. a re-shard), and a transport is bound to its group.

    Args:
        module: MoE block whose experts carry the EP shard.
        ep_group: EP process group.
        hidden: Row width.

    Returns:
        The transport bound to ``module``.
    """
    local_experts = int(module.experts.local_expert_count)
    global_experts = local_experts * dist.get_world_size(ep_group)
    num_sms = int(os.environ.get("HP_MOONEP_NUM_SMS", "32"))
    if num_sms <= 0:
        raise ValueError("HP_MOONEP_NUM_SMS must be positive")
    key = (id(ep_group), hidden, global_experts, num_sms)
    cached = getattr(module, "_moonep_transport", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    transport = MoonEPExpertParallel(ep_group, global_experts, hidden, num_sms=num_sms)
    module._moonep_transport = (key, transport)  # pylint: disable=protected-access
    return transport


class _WeightView:
    """Expert-weight accessor: local shard for own experts, compact table else.

    The compact table holds only the experts some rank prefetches this step
    (see :func:`_gather_needed`); ``position`` maps a global expert id to its
    row there.  Indexing stays differentiable, so a redundant segment's
    gradient flows back through the gather to the owner.

    When ``ready`` is a side stream, the compact table was produced there (the
    weight gather overlaps dispatch/local GEMM); the first compact read on the
    consuming stream synchronises once and records the cross-stream use.
    """

    __slots__ = ("_synced", "compact", "epn", "lo", "local", "packed", "position", "ready")

    def __init__(self, local: torch.Tensor, compact: torch.Tensor | None,
                 position: torch.Tensor | None, lo: int, epn: int,
                 ready: Any = None, packed: bool = False) -> None:
        self.local = local
        self.compact = compact
        self.position = position
        self.lo = lo
        self.epn = epn
        self.ready = ready
        self.packed = packed
        self._synced = ready is None

    def _sync(self) -> None:
        """Make the compact table visible to the current stream (once)."""
        if self._synced:
            return
        current = torch.npu.current_stream()
        current.wait_stream(self.ready)
        # The table was allocated on the side stream; keep its storage alive
        # until the consuming stream is done with it.
        self.compact.record_stream(current)
        self._synced = True

    def __getitem__(self, expert: int) -> torch.Tensor:
        if self.lo <= expert < self.lo + self.epn:
            return self.local[expert - self.lo]
        if self.position is None or int(self.position[expert]) < 0:
            raise KeyError(
                f"expert {expert} is neither local [{self.lo}, "
                f"{self.lo + self.epn}) nor in this step's gathered set; the "
                "plan's remote-expert set and the weight gather disagree")
        self._sync()
        position = int(self.position[expert])
        if self.packed:
            width = self.local[0].numel()
            return self.compact.narrow(0, position, width).view(self.local.shape[1:])
        return self.compact[position]


def _gather_needed(weight: torch.Tensor, group: Any,
                   needed: torch.Tensor, ready: Any = None) -> _WeightView:
    """Gather only the *needed* experts' weights into a compact table.

    Every rank's local experts stay local; only experts that some rank
    prefetches this step are gathered, and every rank can compute that union
    (and therefore the compact layout) from the replicated ``experts_to_copy``
    without extra communication.  A rank contributes the needed experts it
    owns, padded to the widest contribution so the collective stays matched;
    padding rows are never referenced.  Worst case degenerates to the full
    table; balanced routing gathers nothing.

    Note: the first fusion attempt (one all_gather + per-table regroup copies)
    was reverted (2026-10-22): the regroup copy and its backward slices cost
    more than the two saved collectives (4k LIMO profile: 4.61 s/step baseline,
    4.90 fused).  The regroup-free variant now lives in ``_gather_needed_fused``
    (shared flat buffer, per-table views, interleaved positions); this per-table
    path remains as the fallback for tables with mismatched row widths/dtypes
    (e.g. the fused ``gate_up_proj`` layout).

    Args:
        weight: Plain tensor or DTensor ``[num_experts, ...]`` (EP-sharded).
        group: EP process group.
        needed: Sorted unique expert ids (CPU, int64) the group must make
            visible -- identical on every rank.

    Returns:
        A :class:`_WeightView` over the local shard and the compact table.
    """
    local = _local_shard(weight, group)
    ep_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    epn = local.shape[0]
    lo = rank * epn
    if needed.numel() == 0:
        return _WeightView(local, None, None, lo, epn)
    owners = torch.div(needed, epn, rounding_mode="floor")
    counts = torch.zeros(ep_size, dtype=torch.long)
    counts.scatter_add_(0, owners, torch.ones_like(owners))
    width = int(counts.max())
    mine = owners == rank
    my_rows = local[(needed[mine] - lo).long()]
    if my_rows.shape[0] < width:
        pad = local.new_zeros((width - my_rows.shape[0],) + local.shape[1:])
        contribution = torch.cat([my_rows, pad], dim=0)
    else:
        contribution = my_rows
    gathered = torch.distributed.nn.functional.all_gather(contribution, group=group)
    compact = torch.cat(list(gathered), dim=0)          # [R * width, ...]
    within = torch.zeros_like(needed)
    for other in range(ep_size):
        mask = owners == other
        within[mask] = torch.arange(int(mask.sum()))
    position = torch.full((epn * ep_size,), -1, dtype=torch.long)
    position[needed] = owners * width + within
    return _WeightView(local, compact, position, lo, epn, ready=ready)


def _gather_needed_fused(locals_: list[torch.Tensor], group: Any,
                         needed: torch.Tensor,
                         ready: Any = None) -> list[_WeightView]:
    """One differentiable all_gather for same-dtype tables of arbitrary row widths.

    Each rank contributes flattened tables in order. Per-table offsets and
    expert row widths address the shared flat result without regroup copies.
    Narrowing a remote row preserves the gather's gradient dependency.

    Args:
        locals_: Per-table local shards (plain tensors, same dtype).
        group: EP process group.
        needed: Sorted unique expert ids (CPU, int64), identical on every rank.

    Returns:
        One :class:`_WeightView` per table, all viewing the same gather buffer.
    """
    ep_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    epn = locals_[0].shape[0]
    lo = rank * epn
    owners = torch.div(needed, epn, rounding_mode="floor")
    counts = torch.zeros(ep_size, dtype=torch.long)
    counts.scatter_add_(0, owners, torch.ones_like(owners))
    width = int(counts.max())
    mine = owners == rank
    my_ids = (needed[mine] - lo).long()
    flats = []
    for local in locals_:
        my_rows = local[my_ids]
        if my_rows.shape[0] < width:
            pad = local.new_zeros((width - my_rows.shape[0],) + local.shape[1:])
            my_rows = torch.cat([my_rows, pad], dim=0)
        flats.append(my_rows.reshape(-1))
    widths = [local[0].numel() for local in locals_]
    stride = width * sum(widths)
    gathered = torch.distributed.nn.functional.all_gather(
        torch.cat(flats), group=group)
    flat_all = torch.cat(list(gathered), dim=0)
    within = torch.zeros_like(needed)
    for other in range(ep_size):
        mask = owners == other
        within[mask] = torch.arange(int(mask.sum()))
    views = []
    offset = 0
    for local, row_width in zip(locals_, widths):
        position = torch.full((epn * ep_size,), -1, dtype=torch.long)
        position[needed] = owners * stride + offset + within * row_width
        views.append(_WeightView(local, flat_all, position, lo, epn, ready=ready, packed=True))
        offset += width * row_width
    return views


def _needed_tables(
    experts: Any,
    group: Any,
    needed: torch.Tensor,
    async_stream: Any = None,
) -> tuple[_WeightView, _WeightView | None, _WeightView]:
    """Gather only the off-owner experts' tables named by the plan.

    ``needed`` is the plan's remote-expert set (every expert whose rows land
    on a rank that does not own it, including those left in regular groups
    without a prefetch slot); it is deterministic across ranks.  When every
    table shares one row width and dtype (separate SwiGLU layout) the gather
    runs as a single fused collective (:func:`_gather_needed_fused`). Setting
    HP_MOONEP_PACKED_GATHER=1 also allows unequal row widths with one dtype;
    otherwise the implementation falls back to one collective per table.

    With ``async_stream`` the gather is issued on that side stream (overlapping
    dispatch and the local-expert GEMM); the returned views synchronise the
    consuming stream on first compact read (:meth:`_WeightView._sync`).
    """
    needed = needed.cpu().long()
    gate, up, down = resolve_swiglu_weights(experts)
    weights = [gate, down] if up is None else [gate, up, down]
    if async_stream is not None:
        async_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(async_stream):
            return _needed_tables_impl(weights, group, needed, ready=async_stream)
    return _needed_tables_impl(weights, group, needed, ready=None)


def _needed_tables_impl(
    weights: list[torch.Tensor],
    group: Any,
    needed: torch.Tensor,
    ready: Any,
) -> tuple[_WeightView, _WeightView | None, _WeightView]:
    """Gather ``weights``' off-owner rows; ``ready`` tags the producing stream."""
    locals_ = [_local_shard(weight, group) for weight in weights]
    fusible = (os.environ.get("HP_MOONEP_FUSED_GATHER", "1") == "1"
               and needed.numel() > 0
               and len({local.dtype for local in locals_}) == 1
               and (len({int(local[0].numel()) for local in locals_}) == 1
                    or os.environ.get("HP_MOONEP_PACKED_GATHER", "0") == "1"))
    if fusible:
        views = _gather_needed_fused(locals_, group, needed, ready=ready)
    else:
        views = [_gather_needed(weight, group, needed, ready=ready)
                 for weight in weights]
    if len(views) == 2:
        return views[0], None, views[1]
    return views[0], views[1], views[2]


def _grouped_local_segments(experts: Any, group: Any) -> Callable:
    """Batched expert math for the local-expert block (one grouped SwiGLU GEMM).

    The block is contiguous in the receive buffer and described by
    ``local_counts`` (zeros kept), so it lines up with this rank's local
    weight table.  Routing weights are *not* applied here: the caller weights
    the combined rows afterwards, exactly like the per-segment path.  Handles
    both the separate ``w1/w3`` and the fused ``gate_up_proj`` weight layouts.
    """
    def run(rows_local: torch.Tensor, local_counts: torch.Tensor,
            _weights: torch.Tensor) -> torch.Tensor:
        gate, up, down = resolve_swiglu_weights(experts)
        gate_up = _local_shard(gate, group)
        down_w = _local_shard(down, group)
        if up is not None:
            gate_up = torch.cat((gate_up, _local_shard(up, group)), dim=1)
        if bool(getattr(experts, "is_transposed", False)):
            # The holder keeps weights as [in, out]; npu_grouped_swiglu wants
            # [out, in] (it transposes again for the kernel).
            gate_up = gate_up.transpose(1, 2).contiguous()
            down_w = down_w.transpose(1, 2).contiguous()
        # grouped_matmul requires matching x/weight dtypes (bf16 here); the
        # combine accumulates in fp32 on its own, and the framework's grouped
        # path runs bf16 end to end as well.
        rows = rows_local.to(gate_up.dtype)
        return npu_grouped_swiglu(
            rows, gate_up, down_w, local_counts.to(rows_local.device))

    return run


def _tables_anchor(tables: Any) -> torch.Tensor | float:
    """Zero-valued graph dependency on every gathered table.

    The forward all_gather is collective on every rank, but a rank that
    prefetches nothing never reads its compact table, so the gather's backward
    (a reduce_scatter) would fire only on *some* ranks and deadlock the step.
    Anchoring the layer's output to a zero read of each compact table keeps
    the backward collective on every rank at negligible cost.
    """
    if tables is None:
        return 0.0
    anchor = None
    for view in tables:
        if view is not None and view.compact is not None:
            view._sync()  # pylint: disable=protected-access
            term = view.compact.reshape(-1)[0].float() * 0.0
            anchor = term if anchor is None else anchor + term
    return 0.0 if anchor is None else anchor


def _local_shard(weight: torch.Tensor, group: Any) -> torch.Tensor:
    """Return the expert-owned part of ``weight`` for the group gather.

    EP sharding keeps each rank's experts in a contiguous ``Shard(0)`` of the
    stacked table, so the local shard plus a group all-gather reconstructs the
    global table.  Any other placement (TP splitting the hidden dim, a
    replication) would make the concatenation on dim 0 mean something else, so
    those are rejected instead of gathered into a wrong table.

    Args:
        weight: A plain tensor or an EP/TP-sharded ``DTensor``.
        group: The EP process group the table is gathered over.

    Returns:
        A plain tensor this rank owns.

    Raises:
        NotImplementedError: If the tensor is sharded on any dim but the expert dim.
    """
    if not isinstance(weight, DTensor):
        return weight
    names = tuple(getattr(weight.device_mesh, "mesh_dim_names", ()) or ())
    for index, placement in enumerate(weight.placements):
        if isinstance(placement, Replicate):
            continue
        if isinstance(placement, Shard) and placement.dim == 0:
            continue
        raise NotImplementedError(
            f"the MoonEP EP path gathers expert tables along dim 0, but this expert "
            f"weight is distributed with {placement!r} over "
            f"{names[index] if names else 'an unnamed mesh'} "
            f"(mesh={tuple(weight.device_mesh.mesh_shape)}); only Shard(0) over the "
            "expert dimension is supported"
        )
    local = weight.to_local()
    # The gather is over the EP group, so it reconstructs the table only when
    # each rank holds exactly its EP chunk.  EP and FSDP can both shard dim 0
    # (``StridedShard``); then the local chunk is smaller than E / ep and the
    # missing pieces live on ranks this gather does not touch.
    group_size = dist.get_world_size(group)
    if local.shape[0] * group_size != weight.shape[0]:
        raise NotImplementedError(
            f"the MoonEP EP path gathers expert tables over {group_size} EP ranks, but "
            f"the expert dim is split {weight.shape[0] // max(local.shape[0], 1)} ways "
            f"(global {weight.shape[0]}, local {local.shape[0]}); keep FSDP off the "
            "expert dimension (experts EP-sharded only) or disable HP_MOONEP_EP"
        )
    return local


def _segment_output(
    experts: Any,
    segment: torch.Tensor,
    expert: int,
    tables: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor],
) -> torch.Tensor:
    """Expert math for one received segment, mirroring the local expert forward.

    The routing weight is *not* applied here: the caller weights the combined
    rows, exactly as the a2a path does, so both transports produce the same
    quantity for the same weights.

    Args:
        experts: The module's expert holder (for activation and gate hooks).
        segment: ``[rows, H]`` rows delivered for ``expert``.
        expert: Global expert id.
        tables: ``(gate, up, down)`` expert-weight views (``_WeightView``);
            ``up`` is None for a fused ``gate_up_proj`` layout.

    Returns:
        ``[rows, H]`` expert output.
    """
    gate, up, down = tables
    rows = segment.float()
    activation = getattr(experts, "_ep_act_fn", None) or F.silu
    transposed = bool(getattr(experts, "is_transposed", False))
    if up is None:
        gate_up = apply_expert_projection(rows, gate[expert].float(), transposed)
        apply_gate = getattr(experts, "_ep_apply_gate", None)
        if apply_gate is None:
            gate_states, up_states = gate_up.chunk(2, dim=-1)
            hidden = activation(gate_states) * up_states
        else:
            hidden = apply_gate(gate_up)
    else:
        hidden = activation(
            apply_expert_projection(rows, gate[expert].float(), transposed))
        hidden = hidden * apply_expert_projection(rows, up[expert].float(), transposed)
    return apply_expert_projection(hidden, down[expert].float(), transposed)


def moonep_routed_forward(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    router_fn: Callable,
    ep_group: Any,
) -> torch.Tensor:
    """Routed-experts branch over MoonEP, with the a2a path's contract.

    Args:
        module: MoE block whose ``experts`` hold this rank's EP shard.
        hidden_states: ``[B, S, H]`` local tokens.
        router_fn: Router adapter; returns ``(topk_indices [T, K], topk_weights [T, K])``.
        ep_group: EP process group.

    Returns:
        ``[B, S, H]`` routed-expert output, weights applied, token order kept.
    """
    output_shape = tuple(hidden_states.shape)
    hidden = hidden_states.shape[-1]
    flat_states = hidden_states.reshape(-1, hidden)
    token_count = flat_states.shape[0]

    topk_indices, topk_weights = router_fn(module, hidden_states)
    # Routers hand back ``[tokens, K]``; accept a ``[B, S, K]`` form as well so
    # the contract matches ``ep_routed_forward`` for either.
    topk_indices = topk_indices.reshape(-1, topk_indices.shape[-1])
    topk_weights = topk_weights.reshape(-1, topk_weights.shape[-1])
    experts_per_token = topk_indices.shape[1]

    if _moonep_ep_auto() and token_count * experts_per_token < _moonep_min_rows():
        # Below the measured crossover the fixed plan/glue cost dominates the
        # transport win; hand the whole branch back to the a2a implementation
        # (identical contract, no MoonEP buffer is built).
        return ep_routed_forward(
            module, hidden_states,
            router_fn=lambda _module, _states: (topk_indices, topk_weights),
            ep_group=ep_group)
    # Row ``t * K + k`` is token ``t``'s k-th route; MoonEP keeps that order and
    # hands the rows back the same way, so the aggregation only needs the token
    # each row came from.
    token_of_row = torch.arange(token_count, device=flat_states.device).repeat_interleave(
        experts_per_token)
    route_ids = topk_indices.reshape(-1).to(torch.int32)
    route_weights = topk_weights.reshape(-1).to(torch.float32)

    _warn_on_ignored_knobs()
    if getattr(module.experts, "_ep_use_grouped_gemm", False) and "grouped" not in _WARNED:
        _WARNED.add("grouped")
        logger.warning(
            "MoonEP uses grouped GEMM for local segments when HP_MOONEP_GROUPED=1; "
            "remote segments use the gathered-weight per-segment path"
        )
    transport = _transport_for(module, ep_group, hidden)
    tables_box: dict = {}

    def _factory(needed_remote: torch.Tensor) -> Callable:
        if _wgather_async():
            tables_box["tables"] = _needed_tables(
                module.experts, _wgather_group(ep_group), needed_remote,
                async_stream=_wgather_stream())
        else:
            tables_box["tables"] = _needed_tables(module.experts, ep_group,
                                                  needed_remote)
        return lambda segment, expert, _weights: _segment_output(
            module.experts, segment, expert, tables_box["tables"])

    combined = transport.forward(
        flat_states[token_of_row].to(torch.bfloat16),
        route_ids,
        route_weights,
        expert_forward=None,
        expert_forward_factory=_factory,
        expert_forward_many=_grouped_local_segments(module.experts, ep_group),
        output_dtype=torch.float32,
    )
    # Keep the gathered tables' backward collective on every rank (see
    # _tables_anchor), then weight as the a2a path does.
    combined = combined + _tables_anchor(tables_box.get("tables"))
    weighted = combined.to(flat_states.dtype) * route_weights.reshape(-1, 1).to(flat_states.dtype)
    return torch.zeros(
        token_count, hidden, dtype=flat_states.dtype, device=flat_states.device,
    ).index_add(0, token_of_row, weighted).view(*output_shape)


_WARNED: set = set()


def _warn_on_ignored_knobs() -> None:
    """Tell the user once when an a2a-only knob is set on the MoonEP path.

    ``HP_EP_CAPACITY_FACTOR`` bounds rows per expert and ``HP_EP_DISPATCH_CHUNKS``
    splits the a2a exchange; MoonEP replaces both with its own capacity and
    rebalancing, so honoring them here would silently double-limit the routing.
    """
    knobs = ("HP_EP_CAPACITY_FACTOR", "HP_EP_DISPATCH_CHUNKS")
    for knob in knobs:
        if not os.environ.get(knob, "").strip() or knob in _WARNED:
            continue
        _WARNED.add(knob)
        logger.warning(
            "%s is ignored on the MoonEP EP path (HP_MOONEP_EP=1): MoonEP applies its "
            "own capacity and rebalancing", knob,
        )
