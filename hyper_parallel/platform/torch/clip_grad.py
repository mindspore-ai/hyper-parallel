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
"""Distributed-aware gradient clipping for parallel training.

Communication is driven by each parameter's DTensorSpec (device_mesh +
placements) rather than any specific parallelism strategy, so a single
implementation covers FSDP, HSDP, TP+FSDP, and other DTensor-expressed
parallelisms.

Collective safety aligned with FSDP1; numerical precision aligned with
FSDP2's ``_NormPartial`` norm computation path:

* Gradient norms from sharded parameters are all-reduced across the
  corresponding shard process group.
* Non-sharded / replicated norms contribute locally without communication.
* **All ranks participate in the same collectives** regardless of local
  gradient availability, preventing collective-misalignment deadlocks.

The finite p-norm is bit-exact with upstream only for the pure-FSDP,
single-dtype case; mixed sharded + replicated is mathematically correct
but intentionally diverges (see ``_total_norm_fsdp2_aligned``).

Note: PP does not use DTensor layout for gradients today.  Cross-stage
norm aggregation will require an additional manual all-reduce and is
left for future work.
"""
import functools
import math
import warnings
from collections import defaultdict, namedtuple
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Partial

try:
    from torch.utils._foreach_utils import (
        _device_has_foreach_support,
        _group_tensors_by_device_and_dtype,
        _has_foreach_support,
    )
except ImportError:
    _device_has_foreach_support = None  # type: ignore[assignment]
    _group_tensors_by_device_and_dtype = None  # type: ignore[assignment]
    _has_foreach_support = None  # type: ignore[assignment]

__all__: list[str] = ["clip_grad_norm_"]


# (id(mesh) or None, shard_dims) -> list of local grads for norm computation
_GradGroupKey = Tuple[Optional[int], Tuple[int, ...]]

# (mesh_dim_index, dist.ReduceOp, needs_manual_avg)
_PartialReduceInfo = Tuple[int, "dist.ReduceOp", bool]

# Result of _build_grad_groups; tuple-unpacking compatible with prior 7-tuple.
_GradGroups = namedtuple(
    "_GradGroups",
    "grad_groups all_grads norm_grads key_per_grad mesh_cache device "
    "has_dtensor_grad",
)


# ---------------------------------------------------------------------------
# Reduce-op mapping
# ---------------------------------------------------------------------------

_REDUCE_OP_AVG_SUPPORTED = hasattr(dist.ReduceOp, "AVG")

_STR_TO_REDUCE_OP: Dict[str, "dist.ReduceOp"] = {
    "sum": dist.ReduceOp.SUM,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
}
if _REDUCE_OP_AVG_SUPPORTED:
    _STR_TO_REDUCE_OP["avg"] = dist.ReduceOp.AVG


def _str_to_reduce_op(op_str: str) -> Tuple["dist.ReduceOp", bool]:
    """Map a ``Partial`` placement's *reduce_op* string to ``dist.ReduceOp``.

    Returns ``(reduce_op, needs_manual_avg)`` where *needs_manual_avg*
    is ``True`` when ``"avg"`` is requested but the backend does not
    support ``dist.ReduceOp.AVG`` — the caller should use SUM and
    manually divide by the group size.
    """
    lower = op_str.lower()
    if lower == "avg" and not _REDUCE_OP_AVG_SUPPORTED:
        return dist.ReduceOp.SUM, True
    op = _STR_TO_REDUCE_OP.get(lower)
    if op is None:
        raise ValueError(
            f"Unsupported Partial reduce_op: {op_str!r}. "
            f"Supported: {sorted(set(list(_STR_TO_REDUCE_OP) + ['avg']))}"
        )
    return op, False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_parameters(
    parameters: Union["torch.nn.Module", torch.Tensor, Iterable[torch.Tensor]],
) -> List[torch.Tensor]:
    """Normalize *parameters* to a flat list of tensors.

    * ``torch.nn.Module``  -> ``list(module.parameters())``
    * single ``torch.Tensor`` -> ``[tensor]``
    * iterable of tensors    -> ``list(iterable)``
    """
    if isinstance(parameters, torch.nn.Module):
        return list(parameters.parameters())
    if isinstance(parameters, torch.Tensor):
        return [parameters]
    return list(parameters)


def _param_device(param: torch.Tensor) -> torch.device:
    """Return the local device of *param* (unwrap DTensor if needed)."""
    if isinstance(param, DTensor):
        return param._local_tensor.device  # pylint: disable=protected-access
    return param.device


def _get_grad_obj(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
    """Return the gradient object for *param*.

    Checks ``param.main_grad`` first (used when
    ``MixedPrecisionPolicy.apply_grad_on_fp32_main_grad=True``),
    falling back to ``param.grad``.
    """
    grad = getattr(param, "main_grad", None)
    if grad is not None:
        return grad
    return param.grad


def _get_local_grad(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
    """Return the local gradient tensor, or ``None`` if absent.

    Supports ``main_grad`` for fp32 mixed-precision training.
    """
    if not param.requires_grad:
        return None
    grad = _get_grad_obj(param)
    if grad is None:
        return None
    if isinstance(grad, DTensor):
        return grad._local_tensor  # pylint: disable=protected-access
    return grad


def _get_param_mesh_info(
    param: torch.nn.Parameter,
) -> Tuple[
    Optional[object],
    Tuple[int, ...],
    Tuple[_PartialReduceInfo, ...],
]:
    """Derive DeviceMesh, Shard dims and Partial info from DTensorSpec.

    Checks the *gradient's* spec first; falls back to the *parameter's*
    spec when the gradient is a plain tensor on a DTensor parameter
    (common after FSDP/HSDP backward where ``param.grad`` is stored as
    the local shard tensor).

    Returns ``(mesh, shard_dims, partial_info)`` where *partial_info*
    is a tuple of ``(mesh_dim, dist.ReduceOp, needs_manual_avg)``
    triples that respect the ``Partial`` placement's ``reduce_op``
    attribute.  *needs_manual_avg* is ``True`` when ``"avg"`` was
    requested but the backend lacks ``dist.ReduceOp.AVG`` support.
    """
    grad = _get_grad_obj(param)
    # Prefer grad's spec (most accurate); fall back to param's.
    spec_source = grad if isinstance(grad, DTensor) else param
    if not isinstance(spec_source, DTensor):
        return None, (), ()

    shard_dims = tuple(
        i for i, p in enumerate(spec_source.placements)
        if p.is_shard()
    )
    partial_info = tuple(
        (i, *_str_to_reduce_op(p.reduce_op))
        for i, p in enumerate(spec_source.placements)
        if isinstance(p, Partial)
    )
    return spec_source.device_mesh, shard_dims, partial_info


def _sum_p_norms(
    dev_grads: List[torch.Tensor],
    norm_type: float,
    device: torch.device,
    total: torch.Tensor,
) -> None:
    """Accumulate sum-of-p-th-powers for *dev_grads* into *total*."""
    for g in dev_grads:
        n = torch.linalg.vector_norm(g, norm_type)
        total.add_(n.to(device=device) ** norm_type)


def _foreach_p_norms(
    grads: List[torch.Tensor],
    norm_type: float,
    device: torch.device,
) -> torch.Tensor:
    """Fast path: fuse per-tensor norms via ``_foreach_norm``.

    Restricted to float32 tensors to preserve the same numerical
    precision as ``vector_norm(dtype=float32)``.  Non-float32 tensors
    and backends that raise ``RuntimeError`` fall back to per-tensor
    ``vector_norm``.
    """
    total = torch.tensor(0.0, device=device, dtype=torch.float32)
    grouped = _group_tensors_by_device_and_dtype(
        [[g.detach() for g in grads]],
    )
    for (dev, _), ([dev_grads], _) in grouped.items():
        if (
            dev_grads[0].dtype == torch.float32
            and _has_foreach_support(dev_grads, dev)
        ):
            try:
                per_norms = torch._foreach_norm(  # pylint: disable=W0212
                    dev_grads, norm_type,
                )
            except RuntimeError:
                per_norms = None
            if per_norms is not None:
                total.add_(
                    torch.stack([
                        n.to(device=device) ** norm_type
                        for n in per_norms
                    ]).sum(),
                )
            else:
                _sum_p_norms(dev_grads, norm_type, device, total)
        else:
            _sum_p_norms(dev_grads, norm_type, device, total)
    return total


def _per_tensor_norms(
    grads: List[torch.Tensor],
    norm_type: float,
    device: torch.device,
) -> List[torch.Tensor]:
    """Return per-tensor norms as a list of scalar tensors on *device*."""
    if not grads:
        return []

    if _group_tensors_by_device_and_dtype is None or not hasattr(torch, "_foreach_norm"):
        return [
            torch.linalg.vector_norm(g.detach(), norm_type).to(device=device)
            for g in grads
        ]

    norms: List[torch.Tensor] = []
    grouped = _group_tensors_by_device_and_dtype(
        [[g.detach() for g in grads]],
    )
    for (dev, _), ([dev_grads], _) in grouped.items():
        if dev_grads and _has_foreach_support(dev_grads, dev):
            try:
                per_norms = torch._foreach_norm(  # pylint: disable=W0212
                    dev_grads, norm_type,
                )
            except RuntimeError:
                per_norms = None
            if per_norms is not None:
                norms.extend(
                    [n.to(device=device) for n in per_norms],
                )
                continue
        norms.extend([
            torch.linalg.vector_norm(g, norm_type).to(device=device)
            for g in dev_grads
        ])
    return norms


def _compute_local_norm(  # pylint: disable=R0911
    grads: List[torch.Tensor],
    norm_type: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute the combined norm of *grads* locally in FP32.

    When *grads* is empty, returns the **identity element** for the
    subsequent all-reduce so that this rank contributes a neutral value
    (aligned with FSDP1's ``_zero_scalar`` approach):

    * ``inf``   -> 0    (neutral for MAX; norms are non-negative)
    * ``-inf``  -> +inf (neutral for MIN)
    * ``0``     -> 0    (neutral for SUM)
    * finite    -> 0    (neutral for SUM)
    """
    if not grads:
        if norm_type == -math.inf:
            return torch.tensor(
                float("inf"), device=device, dtype=torch.float32,
            )
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    if norm_type == math.inf:
        norms = [
            torch.linalg.vector_norm(g.detach(), math.inf)
            for g in grads
        ]
        return torch.stack(norms).max().to(device)

    if norm_type == -math.inf:
        norms = [
            torch.linalg.vector_norm(g.detach(), -math.inf)
            for g in grads
        ]
        return torch.stack(norms).min().to(device)

    if norm_type == 0:
        norms = [
            torch.linalg.vector_norm(g.detach(), 0)
            for g in grads
        ]
        return torch.stack(norms).sum().to(device)

    # Finite p-norm: return sum of p-th powers.
    if (
        len(grads) > 1
        and _group_tensors_by_device_and_dtype is not None
        and hasattr(torch, "_foreach_norm")
    ):
        return _foreach_p_norms(grads, norm_type, device)

    # Scalar fallback when foreach utilities are unavailable.
    norms = [
        torch.linalg.vector_norm(g.detach(), norm_type)
        for g in grads
    ]
    norm_powers = [n.to(device=device) ** norm_type for n in norms]
    return torch.stack(norm_powers).sum()


# ---------------------------------------------------------------------------
# Total norm aggregation with collectives
# ---------------------------------------------------------------------------

def _get_total_norm(
    grad_groups: Dict[_GradGroupKey, List[torch.Tensor]],
    norm_type: float,
    mesh_cache: Dict[int, object],
    device: torch.device,
    norm_grads: List[torch.Tensor],
    key_per_grad: List[_GradGroupKey],
) -> torch.Tensor:
    """Compute total gradient norm with per-group all-reduce.

    ``norm_grads`` (parallel to ``key_per_grad``) holds the tensor whose
    norm to take per parameter; only the finite p-norm path consumes it.
    """
    if norm_type == math.inf:
        return _total_norm_inf(
            grad_groups, norm_type, mesh_cache, device,
            dist.ReduceOp.MAX,
        )

    if norm_type == -math.inf:
        return _total_norm_inf(
            grad_groups, norm_type, mesh_cache, device,
            dist.ReduceOp.MIN,
        )

    if norm_type == 0:
        return _total_norm_sum(
            grad_groups, norm_type, mesh_cache, device,
        )

    # Finite p-norm: FSDP2-aligned sequence.
    total_p = _total_norm_fsdp2_aligned(
        grad_groups, norm_type, mesh_cache, device,
        norm_grads, key_per_grad,
    )
    return total_p ** (1.0 / norm_type)


def _total_norm_inf(  # pylint: disable=R0913,R0917
    grad_groups, norm_type, mesh_cache, device, reduce_op,
):
    """Shared logic for inf / -inf norms."""
    group_norms: List[torch.Tensor] = []
    for (mesh_id, shard_dims), grads in grad_groups.items():
        local_norm = _compute_local_norm(grads, norm_type, device)
        if mesh_id is not None:
            mesh = mesh_cache[mesh_id]
            for dim in shard_dims:
                dist.all_reduce(
                    local_norm, op=reduce_op,
                    group=mesh.get_group(dim),
                )
        group_norms.append(local_norm)
    if not group_norms:
        if norm_type == -math.inf:
            return torch.tensor(float("inf"), device=device)
        return torch.tensor(0.0, device=device)
    stacked = torch.stack(group_norms)
    return stacked.max() if reduce_op == dist.ReduceOp.MAX else stacked.min()


def _total_norm_sum(grad_groups, norm_type, mesh_cache, device):
    """Shared logic for finite norms and L0 (all use SUM all-reduce)."""
    total = torch.tensor(0.0, device=device)
    for (mesh_id, shard_dims), grads in grad_groups.items():
        local_val = _compute_local_norm(grads, norm_type, device)
        if mesh_id is not None:
            mesh = mesh_cache[mesh_id]
            for dim in shard_dims:
                dist.all_reduce(
                    local_val, op=dist.ReduceOp.SUM,
                    group=mesh.get_group(dim),
                )
        total.add_(local_val)
    return total


def _reduction_signature(grad_groups, mesh_cache):
    """Bucket grad-group keys by the *process group(s)* they reduce over.

    Two keys that reduce over the same set of process groups (same global
    ranks per shard dim) must be **pooled** so their norms accumulate in
    one stack -- matching FSDP2's single foreach-norm + single reduce and
    keeping the loss bit-exact even when params live on several distinct
    ``DeviceMesh`` objects that share the same DP process group (common
    for multi-component models).  Keys that reduce over *different*
    process groups (TP+FSDP heterogeneous sharding, expert parallel) get
    separate buckets, each reduced over its own group.

    Returns ``(key_to_sig, sig_groups, sig_order)``:

    * ``key_to_sig``  -- ``key -> signature`` (hashable; ``()`` = replicate
      / no communication).
    * ``sig_groups``  -- ``signature -> list[ProcessGroup]`` to all-reduce.
    * ``sig_order``   -- signatures in first-seen (parameter) order, so all
      ranks issue the same collectives in the same order.
    """
    key_to_sig: Dict[_GradGroupKey, Tuple] = {}
    sig_groups: Dict[Tuple, List[object]] = {}
    sig_order: List[Tuple] = []
    for mesh_id, shard_dims in grad_groups:
        if shard_dims and mesh_id is not None:
            mesh = mesh_cache[mesh_id]
            groups = [mesh.get_group(dim) for dim in shard_dims]
            sig = tuple(
                tuple(dist.get_process_group_ranks(group)) for group in groups
            )
        else:
            groups = []
            sig = ()
        key_to_sig[(mesh_id, shard_dims)] = sig
        if sig not in sig_groups:
            sig_groups[sig] = groups
            sig_order.append(sig)
    return key_to_sig, sig_groups, sig_order


def _total_norm_fsdp2_aligned(grad_groups, norm_type, mesh_cache, device,
                              norm_grads, key_per_grad):
    """FSDP2-aligned norm for finite p-norms.

    Grads are bucketed by the *process group* they reduce over (see
    :func:`_reduction_signature`); ``norm_grads`` supplies, in global
    parameter order, the Partial-reduced (already-global) view for Partial
    grads and the raw local grad otherwise.  Each bucket does ONE ``stack``
    → ONE ``vector_norm`` → ``^p`` → ONE ``all_reduce SUM`` per shard dim,
    and the buckets are summed locally.

    Reduction rules:

    * Sharded bucket -- reduce over its own group(s).  Same-group params
      across distinct ``DeviceMesh`` objects pool into one reduce; distinct
      groups (TP+FSDP heterogeneous, expert parallel) stay separate -- the
      per-group convention also used by :func:`_total_norm_inf` /
      :func:`_total_norm_sum` and by VeOmni / torchtitan.
    * Replicate bucket (signature ``()``) -- contribute norm² locally with
      NO communication; reducing it over the shard group would over-count
      it ``shard_world_size`` times (the FSDP1 convention).

    Empty-but-present buckets still issue their all_reduce (identity ``0``
    for SUM) so all ranks run the same collectives.

    Bit-exact with upstream ``torch.nn.utils.clip_grad_norm_`` only for the
    pure-FSDP, single-bucket, single-dtype case.  Mixed shard + replicate is
    mathematically correct but intentionally diverges (upstream folds the
    replicate norm into one ``_NormPartial`` reduce and over-counts it);
    mixed-dtype follows upstream's :func:`_per_tensor_norms` device/dtype
    regrouping rather than strict global order.

    Returns the global sum of p-th powers (caller takes p-th root).
    """
    key_to_sig, sig_groups, sig_order = _reduction_signature(
        grad_groups, mesh_cache,
    )

    # Bucket norms by reduction signature, preserving global parameter order.
    sig_grads: Dict[Tuple, List[torch.Tensor]] = {sig: [] for sig in sig_order}
    for grad, key in zip(norm_grads, key_per_grad):
        sig_grads[key_to_sig[key]].append(grad)

    total_p = torch.tensor(0.0, device=device, dtype=torch.float32)
    for sig in sig_order:
        grads = sig_grads[sig]
        if grads:
            norms = _per_tensor_norms(grads, norm_type, device)
            local_p = torch.linalg.vector_norm(
                torch.stack(norms).to(torch.float32), norm_type,
            ) ** norm_type
        else:
            local_p = torch.tensor(0.0, device=device, dtype=torch.float32)

        for group in sig_groups[sig]:
            dist.all_reduce(local_p, op=dist.ReduceOp.SUM, group=group)

        total_p = total_p + local_p

    return total_p


def _build_coalesce_buffer(
    param_infos: List[Tuple],
    indices: List[int],
) -> Tuple[List[torch.Tensor], List[int], List[bool], List[int]]:
    """Build flat fp32 chunks for one coalesce group.

    Returns ``(chunks, chunk_sizes, has_grad, active_indices)``.
    Frozen params are skipped; trainable grad-free params contribute
    zeros so the collective matches ranks that have a grad.
    """
    chunks: List[torch.Tensor] = []
    chunk_sizes: List[int] = []
    has_grad: List[bool] = []
    active_indices: List[int] = []

    for idx in indices:
        param = param_infos[idx][0]
        local_grad = param_infos[idx][1]
        if local_grad is not None:
            chunks.append(
                local_grad.detach().reshape(-1).to(torch.float32),
            )
            chunk_sizes.append(local_grad.numel())
            has_grad.append(True)
            active_indices.append(idx)
        elif param.requires_grad:
            local_p = (
                param._local_tensor  # pylint: disable=W0212
                if isinstance(param, DTensor) else param.data
            )
            numel = local_p.numel()
            chunks.append(
                torch.zeros(
                    numel, device=local_p.device,
                    dtype=torch.float32,
                ),
            )
            chunk_sizes.append(numel)
            has_grad.append(False)
            active_indices.append(idx)

    return chunks, chunk_sizes, has_grad, active_indices


def _coalesce_partial_reduce(  # pylint: disable=R0914
    param_infos: List[Tuple],
    mesh_cache: Dict[int, object],
) -> Dict[int, torch.Tensor]:
    """Coalesce Partial all-reduces: O(N) collectives → O(G).

    Groups parameters sharing the same ``(mesh, partial_info)`` and
    flattens their gradients (or zeros for trainable grad-free params)
    into a single fp32 buffer.  **One** ``all_reduce`` per buffer
    replaces the previous per-parameter collective calls.

    For TP+FSDP (all params share the same mesh / placements), this
    turns ~200 individual all-reduces into 1 — saving 10-20 ms per
    training step at typical HCCS/NCCL latencies.

    Frozen params (``requires_grad=False``) are consistently grad-free
    across all ranks and are excluded from the buffer to avoid wasting
    bandwidth.

    All buffers use float32 to guarantee dtype consistency across ranks
    in mixed-precision training (grad may be fp16/bf16 while param is
    fp32).

    Returns a dict mapping *param_infos* index → reduced gradient view
    (1-D fp32 slice of the coalesced buffer).  Only entries for params
    with actual gradients are included.
    """
    # Group by Partial coalesce key: (mesh_id, partial_info)
    coalesce_groups: Dict[
        Tuple, List[int],
    ] = defaultdict(list)
    for idx, info in enumerate(param_infos):
        mesh, partial_info = info[2], info[3]
        if partial_info:
            if mesh is None:
                raise RuntimeError(
                    "clip_grad_norm_: parameter has Partial placements "
                    "but no DeviceMesh. This is a DTensor invariant "
                    "violation."
                )
            pck = (id(mesh), partial_info)
            coalesce_groups[pck].append(idx)

    reduced: Dict[int, torch.Tensor] = {}

    for (mesh_id, partial_info), indices in coalesce_groups.items():
        mesh = mesh_cache[mesh_id]
        chunks, chunk_sizes, has_grad, active_indices = (
            _build_coalesce_buffer(param_infos, indices)
        )

        if not chunks:
            continue  # all params frozen, no collective needed

        # Sanity check: same mesh → same device.  Fail fast on
        # misconfigured inputs rather than silent NCCL errors.
        buf_device = chunks[0].device
        for chunk in chunks[1:]:
            if chunk.device != buf_device:
                raise RuntimeError(
                    f"clip_grad_norm_: parameters in the same Partial "
                    f"coalesce group are on different devices "
                    f"({buf_device} vs {chunk.device}). All parameters "
                    f"sharing the same DeviceMesh must reside on the "
                    f"same local device."
                )

        buf = torch.cat(chunks)

        for pdim, reduce_op, needs_avg in partial_info:
            group = mesh.get_group(pdim)
            dist.all_reduce(buf, op=reduce_op, group=group)
            if needs_avg:
                buf /= dist.get_world_size(group=group)

        # Extract views for params with actual gradients.
        offset = 0
        for i, idx in enumerate(active_indices):
            numel = chunk_sizes[i]
            if has_grad[i]:
                reduced[idx] = buf[offset:offset + numel]
            offset += numel

    return reduced


def _build_grad_groups(  # pylint: disable=R0914
    params: List[torch.Tensor],
) -> Tuple[
    Dict[_GradGroupKey, List[torch.Tensor]],
    List[torch.Tensor],
    List[torch.Tensor],
    List[_GradGroupKey],
    Dict[int, object],
    torch.device,
    bool,
]:
    """Classify parameters into grad groups and pre-reduce Partial grads.

    Group structure is derived from *parameter* DTensorSpecs (always
    present on every rank) rather than gradients (which may be ``None``
    on some ranks).  This ensures every rank enters the same set of
    collectives, preventing deadlocks (aligned with FSDP1 where all
    ranks unconditionally execute the same all-reduce path).

    Partial gradients are reduced via a **coalesced** all-reduce
    (see ``_coalesce_partial_reduce``), turning O(N) per-parameter
    collectives into O(G) where G is the number of distinct
    ``(mesh, partial_info)`` groups (typically 1 for TP+FSDP).

    Returns
        ``(grad_groups, all_grads, norm_grads, key_per_grad, mesh_cache,
        device, has_dtensor_grad)``.  ``grad_groups`` maps each
        ``(mesh_id, shard_dims)`` key to its grads; ``all_grads`` is the
        flat list of raw local grads (global parameter order) scaled
        in-place by the clip step; ``norm_grads`` is parallel to
        ``all_grads`` but holds the Partial-reduced view for Partial grads
        (the value whose norm is taken on the finite-p path);
        ``key_per_grad`` is parallel to both and maps each grad back to its
        group key.
    """
    # --- Phase 1: classify all parameters ---
    param_infos: List[Tuple] = []
    mesh_cache: Dict[int, object] = {}
    device: Optional[torch.device] = None

    for param in params:
        mesh, shard_dims, partial_info = _get_param_mesh_info(param)
        key: _GradGroupKey = (
            id(mesh) if mesh is not None else None, shard_dims,
        )
        if mesh is not None:
            mesh_cache[id(mesh)] = mesh
        if device is None:
            device = _param_device(param)
        local_grad = _get_local_grad(param)
        param_infos.append(
            (param, local_grad, mesh, partial_info, key),
        )

    if device is None:
        device = torch.device("cpu")

    # --- Phase 2: coalesced Partial reduction (O(N) → O(G)) ---
    reduced = _coalesce_partial_reduce(param_infos, mesh_cache)

    # --- Phase 3: build grad_groups ---
    grad_groups: Dict[_GradGroupKey, List[torch.Tensor]] = defaultdict(
        list,
    )
    all_grads: List[torch.Tensor] = []
    # norm_grads / key_per_grad are parallel to all_grads (see Returns): the
    # norm-input view (Partial-reduced where coalesced, else raw local) and
    # the group key per grad, kept in global parameter order.
    norm_grads: List[torch.Tensor] = []
    key_per_grad: List[_GradGroupKey] = []
    has_dtensor_grad = False

    for idx, info in enumerate(param_infos):
        param, local_grad, key = info[0], info[1], info[4]
        if local_grad is None:
            # Ensure the key exists so the Shard norm all-reduce is
            # entered even when this rank has no grads for the group.
            if key not in grad_groups:
                grad_groups[key] = []
            continue

        grad_obj = _get_grad_obj(param)
        if isinstance(grad_obj, DTensor):
            has_dtensor_grad = True
        all_grads.append(local_grad)
        key_per_grad.append(key)
        if idx in reduced:
            grad_groups[key].append(reduced[idx])
            norm_grads.append(reduced[idx])
        else:
            grad_groups[key].append(local_grad)
            norm_grads.append(local_grad)

    return _GradGroups(
        grad_groups, all_grads, norm_grads, key_per_grad,
        mesh_cache, device, has_dtensor_grad,
    )


def _clip_grads_with_norm_(
    all_grads: List[torch.Tensor],
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    """Scale gradients in-place so the total norm <= *max_norm*."""
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    if _group_tensors_by_device_and_dtype is not None:
        grouped_grads = _group_tensors_by_device_and_dtype(
            [all_grads],
        )
        for (device, dtype), ([device_grads], _) in grouped_grads.items():
            use_foreach = (
                foreach is None and _has_foreach_support(device_grads, device)
            ) or (foreach and _device_has_foreach_support(device))
            if use_foreach:
                torch._foreach_mul_(  # pylint: disable=W0212
                    device_grads,
                    clip_coef_clamped.to(device=device, dtype=dtype),
                )
            elif foreach:
                raise RuntimeError(
                    f"foreach=True was passed, but can't use the "
                    f"foreach API on {device.type} tensors"
                )
            else:
                clip_coef_clamped_cast = clip_coef_clamped.to(device=device, dtype=dtype)
                for g in device_grads:
                    g.mul_(clip_coef_clamped_cast)
    else:
        # Fallback when _foreach_utils is unavailable.
        if foreach:
            raise RuntimeError(
                "foreach=True was passed, but "
                "torch.utils._foreach_utils is not available"
            )
        for grad in all_grads:
            grad.mul_(clip_coef_clamped.to(grad.device, grad.dtype))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_grad_norm_(
    parameters: Union[
        "torch.nn.Module", torch.Tensor, Iterable[torch.Tensor],
    ],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    """Compute and clip gradient norm for distributed models.

    Drop-in replacement for the standard ``clip_grad_norm_`` that
    correctly handles DTensor-sharded parameters by deriving
    communication from each parameter's DTensorSpec.

    .. warning:: This function uses collective communications.  It
        **must be called on all ranks** to avoid deadlocks.  Aligned
        with FSDP1: every rank participates in the same collectives
        regardless of local gradient availability.

    Communication is derived from each parameter's DTensorSpec:

    * ``Shard`` on mesh dim *d* -- all-reduce norm statistics
      across ``device_mesh.get_group(d)``
    * ``Partial`` on mesh dim *d* -- all-reduce gradient values
      using the placement's ``reduce_op`` before norm computation
    * ``Replicate`` / plain tensor -- no communication

    This covers FSDP, HSDP, TP+FSDP, and any combination expressible
    via DTensor placements.  PP cross-stage norm aggregation is not
    yet handled (requires manual all-reduce across stages).

    Args:
        parameters: An ``nn.Module``, a single ``Tensor``, or an
            iterable of ``Tensor`` s whose gradients to clip.
        max_norm: Maximum allowed gradient norm.
        norm_type: Type of the norm (default ``2.0``).
        error_if_nonfinite: If ``True``, raise a ``RuntimeError``
            when the total norm is non-finite.  Default ``False``.
        foreach: Use the faster foreach-based implementation for the
            gradient clipping step.  If ``None``, use the foreach
            implementation for devices that support it and silently
            fall back to the per-tensor implementation for others.
            Default ``None``.

    Returns:
        The total (unclipped) gradient norm as a scalar tensor,
        cast to the promoted dtype of all gradient tensors.
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    params = _normalize_parameters(parameters)

    (
        grad_groups, all_grads, norm_grads, key_per_grad,
        mesh_cache, device, has_dtensor_grad,
    ) = _build_grad_groups(params)

    # -- Norm + clip (all ranks participate) --------------------------------
    # _compute_local_norm returns identity elements for empty groups,
    # so the subsequent all-reduce is safe and semantically neutral.
    total_norm = _get_total_norm(
        grad_groups, norm_type, mesh_cache, device,
        norm_grads, key_per_grad,
    )

    if error_if_nonfinite and torch.logical_or(
        total_norm.isnan(), total_norm.isinf()
    ):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To "
            "disable this error and scale the gradients by the "
            "non-finite norm anyway, set "
            "`error_if_nonfinite=False`"
        )

    if all_grads:
        # Disable foreach for dtensor-backed grads to avoid dispatch issues.
        effective_foreach = False if has_dtensor_grad and foreach is None else foreach
        _clip_grads_with_norm_(
            all_grads, max_norm, total_norm, effective_foreach,
        )

    # Promote return dtype to match gradient dtypes (FSDP1 convention).
    # When this rank has no gradients, return in the default FP32 dtype
    # (same as FSDP1's behavior to avoid extra communication).
    if not all_grads:
        warnings.warn(
            "clip_grad_norm_ called on this rank with no gradients -- "
            "returning the local norm in the default dtype "
            f"{total_norm.dtype}",
            stacklevel=2,
        )
        return total_norm

    total_norm_dtype = functools.reduce(
        torch.promote_types,
        [g.dtype for g in all_grads],
    )
    # Return global all-reduced norm, consistent with torchtitan's
    # full_tensor() approach — .item() returns the correct global value.
    return total_norm.to(total_norm_dtype)
