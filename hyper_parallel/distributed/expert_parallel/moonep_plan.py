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
"""Host-side MoonEP communication plan, vectorised.

MoonEP's own host planner is a reference implementation: it walks tokens one at
a time in Python, which is fine for a one-off check and far too slow to rebuild
every training step.  This module builds the same plan with tensor ops.

The plan is what makes MoonEP's rebalancing work (see
``docs/moonep_ep_integration.md``): each destination ranks' capacity is
``NvS = S * K``, which equals the *average* share of the global routing, so a
real router always overloads someone.  The planner moves the overflow onto
spare capacity on other ranks and records it in ``experts_to_copy`` (the
receiving rank computes those rows from prefetched weights) - MoonEP's
redundant-expert feature.

Scope: ``K = 1`` (one route per row), no token padding (``token_padding = 1``,
the Ascend convention) and no dedup, which is exactly what this repository's
bridge passes to MoonEP.  Fields mirror ``test_npu/planing_reference.py``
one-to-one, and ``tests``-style checks compare them field by field.

When the compiled ``npu_patch._C.plan_host`` fast path is available it is used
first; set ``HP_MOONEP_PLANNER=python`` to force the NumPy/Torch version.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

__all__ = ["build_moonep_plan"]

logger = logging.getLogger(__name__)

_HOST_PLAN_LOADED = False
_HOST_PLAN_FN = None


def _rebalance(total: np.ndarray, epn: int, cap: int, ep_size: int) -> np.ndarray:
    """Return ``alloc[E, R]``: how many of each expert's rows go to each rank.

    Starts with every expert's rows on its owner and moves the overloaded home
    groups' rows onto the roomiest destinations, taking each home's largest
    remaining expert first - the same policy (and therefore the same plan) as
    MoonEP's reference planner.  Runs on numpy: the plan is a few hundred tiny
    integer ops, and numpy's per-op overhead is an order of magnitude below
    torch's CPU dispatch.

    Args:
        total: ``[E]`` rows per expert across all ranks.
        epn: Local experts per rank.
        cap: Per-rank receive capacity (``NvS``).
        ep_size: Number of EP ranks.

    Returns:
        ``[E, ep_size]`` allocation counts.
    """
    experts = total.shape[0]
    owner = np.arange(experts) // epn
    alloc = np.zeros((experts, ep_size), dtype=total.dtype)
    alloc[np.arange(experts), owner] = total

    # Two-sided fill: `extra` is what each home group must shed, `spare` what
    # each destination can still take.  Both are updated as rows move, so a
    # home never donates rows it does not hold and a destination never exceeds
    # its capacity - the bookkeeping the reference planner keeps in `balance`.
    load = alloc.sum(axis=0)               # rows each rank holds
    # Slack stays *signed*: a rank that still owes rows has negative slack even
    # after shedding some of them.  Clamping it at zero used to let a rank that
    # was still over capacity become a receiver, which left it -- and with it
    # the receive buffer -- above `NvS`.
    spare = cap - load
    for _round in range(2 * ep_size + 1):
        if int(load.max()) <= cap:
            break
        home = int(load.argmax())
        room = spare.copy()
        room[home] = -1  # a rank cannot offload onto itself
        dest = int(room.argmax())
        move = min(int(load[home] - cap), int(spare[dest]))
        if move <= 0:
            break  # nowhere left to put the remainder
        while move > 0:
            held = alloc[home * epn:(home + 1) * epn, home]
            local = int(held.argmax())
            take = min(int(held[local]), move)
            if take <= 0:
                break
            expert = home * epn + local
            alloc[expert, dest] += take
            alloc[expert, home] -= take
            move -= take
            load[home] -= take
            load[dest] += take
            spare[home] += take
            spare[dest] -= take
    return alloc


def _host_plan_fn():
    """Return the compiled C++ host planner when it is available."""
    global _HOST_PLAN_LOADED, _HOST_PLAN_FN
    if not _HOST_PLAN_LOADED:
        _HOST_PLAN_LOADED = True
        backend = os.environ.get("HP_MOONEP_PLANNER", "host").strip().lower()
        if backend != "python":
            try:
                from npu_patch.planning_host import (  # pylint: disable=import-outside-toplevel
                    build_plan_host,
                )
                _HOST_PLAN_FN = build_plan_host
            except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
                logger.debug("MoonEP C++ host planner unavailable: %s", exc)
    return _HOST_PLAN_FN


def build_moonep_plan(
    counts_all: torch.Tensor,
    local_counts: torch.Tensor,
    route_ids: torch.Tensor,
    rank: int,
    epn: int,
    nvs: int,
    route_ids_cpu: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build a MoonEP plan, preferring the compiled C++ host planner.

    The C++ path is optional: environments without MoonEP's rebuilt ``_C``
    extension transparently fall back to the NumPy/Torch implementation.
    ``HP_MOONEP_PLANNER=python`` forces that fallback for A/B testing.
    ``route_ids_cpu`` is an optional pre-copied CPU int32 view of
    ``route_ids`` (e.g. from the plan-cache fingerprint) that saves the C++
    path a second D2H copy.
    """
    host_plan = _host_plan_fn()
    strict_host = os.environ.get("HP_MOONEP_STRICT_HOST_PLANNER", "0") == "1"
    if strict_host and host_plan is None:
        raise RuntimeError("C++ host planner required for this experiment but unavailable")
    if host_plan is not None:
        try:
            return host_plan(counts_all, local_counts, route_ids, rank, epn, nvs,
                             route_ids_cpu=route_ids_cpu)
        except Exception as exc:  # pylint: disable=broad-except
            if strict_host:
                raise RuntimeError("C++ host planner failed during controlled experiment") from exc
            logger.debug("MoonEP C++ host planner failed (%s); using Python path", exc)
    return _build_moonep_plan_python(
        counts_all, local_counts, route_ids, rank, epn, nvs)


def _build_moonep_plan_python(
    counts_all: torch.Tensor,
    local_counts: torch.Tensor,
    route_ids: torch.Tensor,
    rank: int,
    epn: int,
    nvs: int,
) -> dict[str, torch.Tensor]:
    """Build MoonEP's comm plan for this rank, matching the reference exactly.

    Args:
        counts_all: ``[R, E]`` routed rows per rank and expert.
        local_counts: This rank's ``[E]`` row counts.
        route_ids: ``[rows]`` global expert id of each row *in the caller's row
            order*.  Rows need not be sorted by expert: the plan encodes the
            slot each row lands on, so any order is fine as long as the ids say
            where each row belongs.
        rank: This rank's EP index.
        epn: Local experts per rank.
        nvs: Per-rank receive capacity (``S * K``).

    Returns:
        A dict with the plan's tensors: ``dst`` (``[rows]`` per-row destination,
        encoded ``dest * NvS + group offset + index in group``), ``cu_seqlens``
        (this rank's ``[E + B]`` group boundaries), ``experts_to_copy``
        (``[R, B]``, ``-1`` where a rank copies nothing), ``remote_stats``
        (``[2]``) and ``zero_fill_ranges`` (``[E + B, 2]``), plus
        ``local_span`` = ``(start, end)`` of the segment this rank computes for
        its own experts.
    """
    device = counts_all.device
    world, experts = counts_all.shape
    cap = int(nvs)
    prefetch_slots = epn
    # The plan's layout assumes one expert partition per rank and exactly the
    # rows this rank presents.  A mismatch here would encode destinations
    # outside the buffer, so reject it before any tensor is built.
    if world < 1 or experts < 1:
        raise ValueError(f"counts_all must be [R, E] with R, E >= 1, got {tuple(counts_all.shape)}")
    if epn < 1 or experts != world * epn:
        raise ValueError(
            f"expert count {experts} must equal R * epn ({world} * {epn})")
    if cap < 0:
        raise ValueError(f"per-rank capacity must be non-negative, got {cap}")
    if route_ids.dim() != 1 or route_ids.numel() != cap:
        raise ValueError(
            f"route_ids must hold one id per routed row ({cap}), got shape {tuple(route_ids.shape)}")
    if local_counts.numel() != experts:
        raise ValueError(
            f"local_counts must describe {experts} experts, got {local_counts.numel()}")
    # The whole plan is integer bookkeeping over an [R, E] count table plus the
    # per-row ids, so it runs on the host: measured at 64k rows / E=896, the
    # device version spends ~40 ms in gather/sort fallbacks while the host does
    # the same work in a few milliseconds.  Only the finished tensors move back
    # (dst is 256 KB).  Value checks run after the transfer: every int() on a
    # device tensor is a D2H sync.
    counts_all = counts_all.cpu()
    local_counts = local_counts.cpu()
    route_ids = route_ids.cpu()
    counts_np = counts_all.numpy()
    local_np = local_counts.numpy()
    if int(local_counts.sum()) != cap:
        raise ValueError(
            f"local_counts sum to {int(local_counts.sum())} rows but the capacity is {cap}")
    if cap and (int(route_ids.min()) < 0 or int(route_ids.max()) >= experts):
        raise ValueError(
            f"route_ids must name experts in [0, {experts}), got "
            f"[{int(route_ids.min())}, {int(route_ids.max())}]")
    # MoonEP's capacity premise is global: R * NvS rows shared evenly.  If any
    # rank presents a different row count, the migration loop can leave a rank
    # over capacity and its overflow slots would be encoded into the *next*
    # rank's buffer -- the overloaded rank raises, but the others would never
    # notice.  Reject the malformed input before any encoding happens.
    rows_per_rank = counts_np.sum(axis=1)
    if not bool((rows_per_rank == cap).all()):
        raise ValueError(
            f"every rank must present exactly NvS ({cap}) rows, got per-rank "
            f"rows {rows_per_rank.tolist()}")
    total = counts_np.sum(axis=0)
    alloc = _rebalance(total, epn, cap, world)
    alloc_cumsum = alloc.cumsum(axis=1)

    # Every expert whose rows land on a rank that does not own it.  Rows may
    # sit in a *regular group* without a prefetch slot (when a destination's
    # eligible remotes outnumber its slots), so weight gathering must cover
    # this set -- not just the prefetch picks in ``experts_to_copy``.
    needed_remote = np.nonzero(
        ((alloc > 0) & ((np.arange(experts) // epn)[:, None]
                        != np.arange(world)[None, :])).any(axis=1)
    )[0]

    # Prefetch picks: the remote experts with the most rows, largest first,
    # for every destination in one sort over the [E, R] grid instead of a
    # per-rank loop.  Keys pack (rows, expert id) so ties break on the larger
    # expert, the same rule as the reference planner.
    owner = np.arange(experts) // epn
    eligible = (alloc > 0) & (owner[:, None] != np.arange(world)[None, :])
    keys = alloc * (experts + 1) + np.arange(experts)[:, None]
    masked = np.where(eligible, keys, -1)
    order = np.argsort(-masked, axis=0)
    top_vals = np.take_along_axis(masked, order[:prefetch_slots], axis=0)
    valid = top_vals >= 0
    copies = np.where(valid, top_vals % (experts + 1), -1).T.copy().astype(np.int32)
    remote_stats = np.zeros((world, 2), dtype=np.int32)
    # The reference planner reports every remote expert headed for each
    # destination (before truncating to the available prefetch slots).
    remote_stats[:, 0] = eligible.sum(axis=0).astype(np.int32)
    # Count each prefetch, including two experts owned by the same rank: a
    # fancy-index ``+=`` drops duplicates, so a rank that lends two experts
    # to one destination was reported as lending one.
    remote_stats[:, 1] += np.bincount(
        owner[copies.T[valid]], minlength=world).astype(np.int32)
    # The prefetched experts give up their own group; their rows travel to the
    # slot instead, so the two must not both be counted.  Both tables below used
    # to be filled by a per-(rank, slot) Python loop (~900 iterations, several
    # milliseconds); scattering over the whole grid does the same in one op.
    valid_slots = copies >= 0
    slot_experts = np.maximum(copies, 0).astype(np.int64)
    # Invalid slots are clamped to expert 0 only to index the gathers.  They must
    # not be scattered into the [R, E] tables: a rank that prefetches expert 0
    # *and* has a spare slot would otherwise write ``False`` into expert 0's
    # mask, so expert 0's group would be counted in addition to its prefetch
    # slot and the rank would need more rows than ``NvS``.  The reference
    # planner skips a ``-1`` slot; the dummy column gives the vectorised version
    # the same rule.
    scatter_cols = np.where(valid_slots, copies, experts).astype(np.int64)
    prefetch_mask = np.zeros((world, experts + 1), dtype=bool)
    np.put_along_axis(prefetch_mask, scatter_cols, valid_slots, axis=1)
    prefetch_mask = prefetch_mask[:, :experts]

    # Group layout per destination: experts 0..E-1 (those being prefetched give
    # up their group rows to the prefetch slot), then the prefetch slots.
    alloc_t = np.ascontiguousarray(alloc.T)
    group_counts = np.where(alloc_t.astype(bool) & ~prefetch_mask, alloc_t, 0)
    # Rows each prefetch slot carries: expert e's count on its owner, i.e. row
    # `dest` of alloc transposed, gathered by the chosen expert ids.
    prefetch_counts = np.take_along_axis(alloc_t, slot_experts, axis=1)
    prefetch_counts = np.where(valid_slots, prefetch_counts, 0)
    all_counts = np.concatenate([group_counts, prefetch_counts], axis=1)  # [R, E + B]
    # group_start[d, g] is where group g begins; expert offsets index that.
    group_start = np.zeros_like(all_counts)
    group_start[:, 1:] = all_counts.cumsum(axis=1)[:, :-1]
    cu_seqlens = all_counts.cumsum(axis=1).astype(np.int32)

    # A prefetched expert's rows live in its prefetch slot, not in its (empty)
    # group, so its offset has to come from the slot.
    expert_off = np.zeros((world, experts + 1), dtype=group_start.dtype)
    expert_off[:, :experts] = np.where(
        group_counts > 0, group_start[:, :experts], 0)
    # A prefetched expert's rows sit in its slot, so its offset comes from there;
    # an invalid slot writes to the dummy column and leaves every expert's
    # offset untouched.
    np.put_along_axis(
        expert_off, scatter_cols,
        np.where(valid_slots, group_start[:, experts:], 0), axis=1)
    expert_off = expert_off[:, :experts]

    # Per-row destination: owner rank of the expert, plus where the row lands.
    # Each row's slot inside its expert's group: its index among the rows of the
    # same expert on this rank, in the order the caller presented them, plus the
    # earlier source ranks' whole count of that expert.  This block stays in
    # torch: its int32 sort is ~2x faster than numpy's at 64k+ rows, and the
    # zero-copy from_numpy bridge keeps the rest of the plan in numpy.
    experts_per_row = route_ids
    # Stable sort keeps the caller's row order inside an expert group, which is
    # the canonical local-index rule used by MoonEP's reference planner (and by
    # the C++ host fast path).  An unstable sort may permute equal ids and
    # change ``dst`` even though the plan remains self-consistent.
    sorted_ids, order = torch.sort(experts_per_row, stable=True)
    starts = torch.from_numpy(
        (local_np.cumsum() - local_np).astype(np.int32))
    local_index = torch.empty_like(experts_per_row)
    local_index[order] = (torch.arange(experts_per_row.numel(), dtype=torch.int32)
                          - starts[sorted_ids])
    if rank > 0:
        head = torch.from_numpy(
            counts_np.cumsum(axis=0)[rank - 1].astype(np.int32))
    else:
        head = torch.zeros(experts, dtype=torch.int32)
    global_index = head[experts_per_row] + local_index
    # `searchsorted(..., right=True)` is "how many cumulative allocations are
    # <= g"; with R small the plain comparison is cheaper.
    alloc_cumsum_t = torch.from_numpy(np.ascontiguousarray(alloc_cumsum))
    expert_off_t = torch.from_numpy(np.ascontiguousarray(expert_off))
    dest = (alloc_cumsum_t[experts_per_row] <= global_index.unsqueeze(1)).sum(dim=1)
    previous = torch.where(
        dest > 0, alloc_cumsum_t[experts_per_row, (dest - 1).clamp(min=0)],
        torch.zeros_like(dest))
    dst = (dest * nvs
           + expert_off_t[dest, experts_per_row]
           + (global_index - previous)).to(torch.int32)

    zero_fill = np.zeros((experts + prefetch_slots, 2), dtype=np.int32)
    lo = rank * epn
    span = (int(group_start[rank, lo]), int(cu_seqlens[rank, lo + epn - 1]))
    return {
        "dst": dst.contiguous().to(device),
        "cu_seqlens": torch.from_numpy(np.ascontiguousarray(cu_seqlens[rank])).to(device),
        "experts_to_copy": torch.from_numpy(np.ascontiguousarray(copies)).to(device),
        "remote_stats": torch.from_numpy(np.ascontiguousarray(remote_stats[rank])).to(device),
        "zero_fill_ranges": torch.from_numpy(zero_fill).to(device),
        "local_span": span,
        "alloc": torch.from_numpy(np.ascontiguousarray(alloc.T)).to(device),
        "needed_remote": torch.from_numpy(np.ascontiguousarray(needed_remote)).to(device),
    }
