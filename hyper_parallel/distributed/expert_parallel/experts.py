# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""expert_parallel.experts: EP expert execution and binding.

- Routed-experts pipeline (``ep_routed_forward``, 05 §6.4.8): router (local
  chunk) -> a2a dispatch (extended EP group) -> local SwiGLU (complete
  expert weights, no internal communication) -> a2a combine -> weighted
  aggregation. SP-in -> SP-out, all communication cohesive inside. NO
  shared-expert/gate branch — that composition is the caller's job.
- Expert entry point (``bind_local_expert_forward`` /
  ``resolve_swiglu_weights``): installs ``experts.forward`` (via the forward
  rewriter's bound-forward install point, 05 §15.2.3) so nested FSDP hooks
  unshard/reshard around the local SwiGLU.
- Interface helpers (``require_attrs`` / ``describe_moe_module``):
  build-time interface assertions with teaching errors, and a structural
  diagnostic for mapping a concrete MoE module to an archetype.

Nothing in this module probes model structure with getattr fallback chains.
Split out of components/distributed/ep_utils.py in stage 4e.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F

from hyper_parallel.components.functional.npu_grouped_swiglu import (
    npu_grouped_swiglu,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _install_bound_forward,
)
from hyper_parallel.distributed.expert_parallel.routing import (
    _resolve_capacity_order,
    apply_capacity_limit,
)
from hyper_parallel.distributed.expert_parallel.collectives import (
    ep_exchange_counts,
    ep_all_to_all,
    ep_all_to_all_async,
    ep_split_sizes,
)
from hyper_parallel.distributed.expert_parallel.routing import (
    MOE_ROUTER_ADAPTERS,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EPDispatch:
    """Prepared tensors and split sizes for one routed expert exchange."""

    source_indices: torch.Tensor
    expert_weights: torch.Tensor
    dispatch_order: torch.Tensor
    states: torch.Tensor
    expert_indices: torch.Tensor
    send_counts: list[int]
    receive_counts: list[int]
    unpermute_indices: Optional[torch.Tensor] = None
    routing_weights: Optional[torch.Tensor] = None


def resolve_swiglu_weights(
    experts: Any,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Resolve SwiGLU weights from the stacked holder (three layouts).

    Returns (w_gate, w_up, w_down):
    - Separate naming: gate_proj/up_proj/down_proj or w1/w3/w2 (all
      [E, I, H]/[E, H, I]);
    - **fused layout** (after the HF 2025 refactor, D-11): gate_up_proj
      [E, 2I, H] + down_proj [E, H, I] -> returns (gate_up_proj, None,
      down_proj); w_up=None marks the fused case (the compute side chunks
      out gate/up).
    """
    w_fused = getattr(experts, "gate_up_proj", None)
    if w_fused is None:
        w_fused = getattr(experts, "gate_and_up_projs", None)
    w_down = getattr(experts, "down_proj", None)
    w_down = w_down if w_down is not None else getattr(experts, "down_projs", None)
    w_down = w_down if w_down is not None else getattr(experts, "w2", None)
    if w_fused is not None and w_down is not None:
        return w_fused, None, w_down

    w_gate = getattr(experts, "gate_proj", None)
    w_gate = w_gate if w_gate is not None else getattr(experts, "w1", None)
    w_up = getattr(experts, "up_proj", None)
    w_up = w_up if w_up is not None else getattr(experts, "w3", None)
    if w_gate is None or w_up is None or w_down is None:
        raise NotImplementedError(
            f"{type(experts).__name__}: only SwiGLU experts are supported "
            "(fused gate_up_proj, or separate gate_proj/up_proj/down_proj, "
            "w1/w2/w3 three-matrix layouts); use an EP-aware MoE module instead"
        )
    return w_gate, w_up, w_down


# ``ArgSort`` has no int32/int64 AICore kernel and silently falls back to
# AI_CPU (runtime warning), which is expensive on the EP dispatch path where it
# sits before the all-to-all.  Every key sorted here is a small non-negative
# integer and float32 represents integers up to 2**24 exactly, so sorting a
# float32 key preserves the order and lets the kernel run on AICore.  Opt-in
# while it is being measured (HP_EP_SORT_FP32=1).
_SORT_KEY_FP32_LIMIT = 2 ** 24
_SORT_FP32_ENABLED = os.environ.get("HP_EP_SORT_FP32", "0") == "1"

# The dispatch tags every routed slot with its source token as
# ``arange(token_count).repeat_interleave(top_k)``.  That eager call lowers to a
# *single-block* vector kernel: on one NPU it costs 9.25 ms per call at
# T=8192 / K=8 (one vector core out of 48, ~58 MB/s of writes), i.e. 36 x 9.25 ms
# = 0.33 s per step on the 18-layer testbed (16% of a 2.05 s step) and 1.07 s per
# step on the 256-card shape.
#
# ``expand`` (broadcast copy) and ``div`` (integer floor division) build the
# numerically identical int64 tensor in 0.034 / 0.041 ms -- 270x cheaper, all
# cores.  Measured end to end on the 18-layer testbed, 3 interleaved rounds each
# (min-of-run step time, order rotated): legacy 2.049 / 2.057 / 2.065 s vs
# expand 2.016 / 2.016 / 2.026 s, i.e. -1.6% step time and +2.4% throughput.
# The gap between the kernel's 0.33 s and the 0.04 s actually recovered is the
# usual one: a device kernel span is not the same as recoverable step time, and
# the eager kernel spent ~half of its span overlapped with communication.
_SOURCE_INDEX_MODES = ("legacy", "expand", "div")
_SOURCE_INDEX_MODE = os.environ.get("HP_EP_SOURCE_INDEX", "expand").lower()
# Dispatch counter for the sparse capacity-limit log (see ep_routed_dispatch).
_CAPACITY_SYNC_COUNT = 0


def _argsort_keys(keys: torch.Tensor, *, bound: int) -> torch.Tensor:
    """Sort non-negative integer keys, on AICore when the range allows.

    Args:
        keys: Non-negative integer sort keys.
        bound: Exclusive upper bound on every key, known statically by the
            caller so no device sync is needed to check the float32 range.

    Returns:
        The int64 permutation that sorts ``keys``.
    """
    if _SORT_FP32_ENABLED and 0 < bound <= _SORT_KEY_FP32_LIMIT:
        return keys.to(torch.float32).argsort()
    return keys.argsort()


def _expert_token_counts(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Count routed tokens per expert with device ops only.

    ``torch.bincount`` on NPU reads the input's min and max back to the host
    (input validation and output sizing), so every call drains the device queue
    and blocks the host. The counts here are consumed either fully on device
    (grouped GEMM group list) or -- where the ragged all-to-all genuinely needs
    them -- as a single host list, so accumulating them with ``scatter_add_``
    keeps the same values at one device kernel and no host stall.

    Args:
        indices: Expert indices, any shape; values must be in ``[0, num_experts)``.
        num_experts: Length of the returned histogram (``minlength``).

    Returns:
        int64 counts of shape ``[num_experts]`` on ``indices.device``.
    """
    flat = indices.to(torch.int64).reshape(-1)
    counts = torch.zeros(num_experts, dtype=torch.int64, device=flat.device)
    return counts.scatter_add_(0, flat, torch.ones_like(flat))


def _local_swiglu_expert_forward(experts, dispatched_states, local_expert_indices):
    """Compute dispatched tokens with the local stacked SwiGLU experts.

    This function is installed as ``experts.forward`` for the HF-native EP
    path. Calling the expert module, instead of indexing its parameters from
    the parent MoE forward, allows nested FSDP forward hooks to unshard and
    reshard expert parameters around the local computation.
    """
    token_order = _argsort_keys(
        local_expert_indices, bound=experts.local_expert_count)
    sorted_states = dispatched_states[token_order]
    local_expert_counts = _expert_token_counts(
        local_expert_indices, experts.local_expert_count)
    if getattr(experts, "_ep_use_grouped_gemm", False):
        grouped_forward = getattr(experts, "forward_expert_major", None)
        if callable(grouped_forward):
            sorted_output = grouped_forward(sorted_states, local_expert_counts)
        else:
            gate_weight, up_weight, down_weight = resolve_swiglu_weights(experts)
            if up_weight is not None:
                raise ValueError(
                    "EP grouped GEMM currently requires packed gate_up_proj weights"
                )
            sorted_output = npu_grouped_swiglu(
                sorted_states,
                gate_weight,
                down_weight,
                local_expert_counts,
            )
        output = torch.empty_like(sorted_output)
        output[token_order] = sorted_output
        return output

    gate_weight, up_weight, down_weight = resolve_swiglu_weights(experts)
    # The eager path needs host ints: read them in one transfer rather than
    # draining the queue once per local expert.
    expert_token_counts = local_expert_counts.tolist()
    sorted_outputs = []
    token_start = 0
    for local_expert_index in range(experts.local_expert_count):
        expert_token_count = expert_token_counts[local_expert_index]
        expert_states = sorted_states[token_start:token_start + expert_token_count]
        if up_weight is None:
            gate_up_states = F.linear(  # pylint: disable=not-callable
                expert_states,
                gate_weight[local_expert_index],
            )
            apply_gate = getattr(experts, "_ep_apply_gate", None)
            if apply_gate is None:
                gate_states, up_states = gate_up_states.chunk(2, dim=-1)
                activation = getattr(experts, "_ep_act_fn", F.silu)
                activated_states = activation(gate_states) * up_states
            else:
                activated_states = apply_gate(gate_up_states)
        else:
            gate_states = F.linear(  # pylint: disable=not-callable
                expert_states, gate_weight[local_expert_index]
            )
            up_states = F.linear(  # pylint: disable=not-callable
                expert_states, up_weight[local_expert_index]
            )
            activation = getattr(experts, "_ep_act_fn", F.silu)
            activated_states = activation(gate_states) * up_states
        sorted_outputs.append(
            F.linear(  # pylint: disable=not-callable
                activated_states,
                down_weight[local_expert_index],
            )
        )
        token_start += expert_token_count

    sorted_output = torch.cat(sorted_outputs)
    output = torch.empty_like(sorted_output)
    output[token_order] = sorted_output
    return output


def _get_global_expert_count(module):
    """Return the model-level routed expert count for an MoE module."""
    if hasattr(module.experts, "num_experts"):
        return module.experts.num_experts
    if hasattr(module, "num_experts"):
        return module.num_experts
    if hasattr(module, "config") and hasattr(module.config, "num_experts"):
        return module.config.num_experts
    if hasattr(module, "config") and hasattr(module.config, "n_routed_experts"):
        return module.config.n_routed_experts
    raise ValueError(
        f"{type(module).__name__}: cannot determine the global routed expert count"
    )


def bind_local_expert_forward(
    module: Any,
    ep_size: int,
    use_grouped_gemm: bool = False,
    apply_gate: Optional[Callable] = None,
) -> None:
    """Install the local expert compute entry used by TP-extend-EP.

    Called by the EP compute factory (archetype or user-written) at apply
    time: sets ``module.experts.local_expert_count`` and installs
    ``experts.forward`` (via the forward rewriter's bound-forward install
    point) so nested FSDP hooks unshard/reshard around the local SwiGLU
    computation. ``apply_gate`` supplies model-specific fused gate/up
    semantics when the default activation-times-up rule is insufficient.
    """
    global_expert_count = _get_global_expert_count(module)
    if global_expert_count % ep_size != 0:
        raise ValueError(
            f"num_experts ({global_expert_count}) must be divisible by ep_size ({ep_size})"
        )
    module.experts.local_expert_count = global_expert_count // ep_size
    activation = getattr(module.experts, "act_fn", None)
    if activation is None:
        hidden_act = getattr(getattr(module, "config", None), "hidden_act", "silu")
        activation = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
            "swish": F.silu,
        }.get(hidden_act)
        if activation is None:
            raise ValueError(
                f"{type(module).__name__}: unsupported expert activation {hidden_act!r}; "
                "provide experts.act_fn or extend the EP activation registry"
            )
    if apply_gate is not None and use_grouped_gemm:
        raise ValueError("custom expert gate activation is not supported by grouped GEMM")
    # ``_ep_*`` is the contract the local compute path reads back off the module's
    # experts child; setting them here is the point of the binder.
    # pylint: disable=protected-access
    module.experts._ep_act_fn = activation
    module.experts._ep_apply_gate = apply_gate
    module.experts._ep_use_grouped_gemm = use_grouped_gemm
    # The forward write itself lives in the forward rewriter (05 §15.2.3:
    # the single MethodType/assignment site); this binder only sets the
    # companion attributes above.
    _install_bound_forward(module.experts, _local_swiglu_expert_forward)


def _resolve_capacity_factor(module: Any) -> Optional[float]:
    """Return the configured expert capacity factor, or ``None`` when it is off.

    Resolution order: the ``HP_EP_CAPACITY_FACTOR`` env override (handy for a sweep without
    editing YAML), then the MoE block's own ``capacity_factor`` attribute (set by the EP recipe
    from a ``plan_overrides[].replace`` forward implementation), then off. Off is the default, so an
    unconfigured run keeps the previous behaviour exactly.

    Args:
        module: MoE block (``None`` is tolerated for callers without one).

    Returns:
        A positive float, or ``None`` when the capacity limit is disabled.
    """
    override = os.environ.get("HP_EP_CAPACITY_FACTOR", "").strip()
    if override:
        try:
            value = float(override)
        except ValueError:
            logger.warning(
                "HP_EP_CAPACITY_FACTOR=%r is not a number; capacity limit stays off", override)
            return None
        return value if value > 0 else None
    value = getattr(getattr(module, "experts", None), "capacity_factor", None)
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def _routed_slot_token_ids(token_count: int, experts_per_token: int, device: Any) -> torch.Tensor:
    """Return the source token of every routed slot, as an index tensor.

    ``arange(token_count).repeat_interleave(experts_per_token)`` and both fast
    paths produce the same int64 tensor: slot ``token * K + i`` holds ``token``.

    Args:
        token_count: Number of local tokens ``T`` in this dispatch.
        experts_per_token: Routed experts per token ``K``.
        device: Device to build the tensor on.

    Returns:
        Contiguous int64 tensor of shape ``[T * K]``.

    Raises:
        ValueError: If ``HP_EP_SOURCE_INDEX`` names an unknown mode.
    """
    mode = _SOURCE_INDEX_MODE
    if mode not in _SOURCE_INDEX_MODES:
        raise ValueError(
            f"HP_EP_SOURCE_INDEX must be one of {_SOURCE_INDEX_MODES}, but got {mode!r}"
        )
    if mode == "expand" and experts_per_token > 0:
        # ``reshape`` may hand back a stride-0 view when the token dim is 1, so
        # materialize: the callers index with the result and expect a flat buffer.
        return (
            torch.arange(token_count, device=device)
            .unsqueeze(1)
            .expand(token_count, experts_per_token)
            .reshape(-1)
            .contiguous()
        )
    if mode == "div" and experts_per_token > 0:
        return torch.arange(token_count * experts_per_token, device=device).div(
            experts_per_token, rounding_mode="floor"
        )
    return torch.arange(token_count, device=device).repeat_interleave(experts_per_token)


def _prepare_ep_dispatch(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    local_expert_count: int,
    global_expert_count: int,
    ep_size: int,
    ep_group: Any,
    keep: Optional[torch.Tensor] = None,
    use_fused_moe_token_permute_and_unpermute: bool = False,
) -> _EPDispatch:
    """Sort routed tokens and exchange per-rank dispatch counts.

    ``keep`` is an optional ``[T, K]`` bool mask (see
    :func:`routing.apply_capacity_limit`).  It compacts the routed slots *before* anything is
    exchanged: dropped slots never become rows, which is the entire point of a capacity limit,
    because a rank's expert buffers are sized by the rows it receives.  The rest of the
    pipeline is already ragged-safe -- the expert compute is row-based, and the combine
    accumulates with ``index_add_`` over ``source_indices`` -- so a token with fewer than ``K``
    surviving slots simply receives fewer contributions.
    """
    flattened_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    token_count = flattened_states.shape[0]
    experts_per_token = topk_indices.shape[1]
    if keep is None:
        expert_indices = topk_indices.reshape(-1)
        expert_weights = topk_weights.reshape(-1)
        source_indices = _routed_slot_token_ids(
            token_count, experts_per_token, flattened_states.device)
    else:
        flat_keep = keep.reshape(-1)
        expert_indices = topk_indices.reshape(-1)[flat_keep]
        expert_weights = topk_weights.reshape(-1)[flat_keep]
        source_indices = _routed_slot_token_ids(
            token_count, experts_per_token, flattened_states.device)[flat_keep]
    destination_ranks = torch.div(expert_indices, local_expert_count, rounding_mode="floor")
    use_fused_moe_token_permute_and_unpermute = (
        use_fused_moe_token_permute_and_unpermute and flattened_states.device.type == "npu" and keep is None
    )
    unpermute_indices = None
    if use_fused_moe_token_permute_and_unpermute:
        # Keep optional NPU dependencies out of the generic dispatch path.
        from hyper_parallel.components.functional.moe_token_permute import (  # pylint: disable=C0415
            moe_token_permute,
        )

        dispatched_states, unpermute_indices = moe_token_permute(
            flattened_states, topk_indices.to(torch.int32),
        )
        dispatch_order = torch.empty_like(unpermute_indices, dtype=torch.int64)
        dispatch_order.scatter_(
            0, unpermute_indices.to(torch.int64),
            torch.arange(unpermute_indices.numel(), device=unpermute_indices.device),
        )
    else:
        dispatch_order = _argsort_keys(
            destination_ranks * global_expert_count + expert_indices,
            bound=ep_size * global_expert_count,
        )
        dispatched_states = flattened_states[source_indices[dispatch_order]].contiguous()
    dispatched_indices = expert_indices[dispatch_order].unsqueeze(-1).contiguous()
    # A fixed shape does not imply a fixed router; use this invocation's counts.
    send_counts_tensor = _expert_token_counts(destination_ranks, ep_size)
    receive_counts_tensor = ep_exchange_counts(send_counts_tensor, ep_group)
    send_counts = ep_split_sizes(send_counts_tensor, dispatched_states.shape[0])
    receive_counts = ep_split_sizes(receive_counts_tensor)
    return _EPDispatch(
        source_indices=source_indices,
        expert_weights=expert_weights,
        dispatch_order=dispatch_order,
        states=dispatched_states,
        expert_indices=dispatched_indices,
        send_counts=send_counts,
        receive_counts=receive_counts,
        unpermute_indices=unpermute_indices,
        routing_weights=topk_weights if use_fused_moe_token_permute_and_unpermute else None,
    )


def _resolve_dispatch_chunks(ep_size: int) -> int:
    """Use at most two target-rank groups for the ordinary EP pipeline.

    Two groups provide the next dispatch to overlap without fragmenting expert
    GEMMs into one batch per peer. A single-rank group needs only one chunk.

    Args:
        ep_size: Number of ranks in the expert-parallel group.

    Returns:
        One chunk for EP1, otherwise two chunks.
    """
    return min(2, ep_size)


def _chunk_rank_range(group: int, ep_size: int, chunk_count: int) -> tuple[int, int]:
    """Half-open EP rank range covered by chunk group ``group``.

    The ``chunk_count`` groups partition the EP ranks into contiguous,
    as-equal-as-possible ranges, so a group is always non-empty when
    ``chunk_count <= ep_size`` (which :func:`_resolve_dispatch_chunks`
    guarantees).

    Args:
        group: Chunk group index.
        ep_size: Size of the EP group.
        chunk_count: Number of chunk groups.

    Returns:
        ``(first_rank, last_rank)``, the first inclusive / last exclusive rank.
    """
    return group * ep_size // chunk_count, (group + 1) * ep_size // chunk_count


def _chunk_rank_group(ep_rank: int, ep_size: int, chunk_count: int) -> int:
    """Chunk group owning ``ep_rank`` (the inverse of :func:`_chunk_rank_range`).

    Args:
        ep_rank: This rank's index in the EP group.
        ep_size: Size of the EP group.
        chunk_count: Number of chunk groups.

    Returns:
        The group index whose range contains ``ep_rank``.
    """
    return ((ep_rank + 1) * chunk_count - 1) // ep_size


class _EPDispatchChunks(NamedTuple):
    """Chunk plan of one routed exchange (see :func:`_ep_dispatch_chunks`).

    ``send_counts[c]`` / ``recv_counts[c]`` are full-length (``ep_size``) count
    lists for chunk ``c``; ``row_ranges[c]`` is the chunk's half-open slice of
    the expert-major stream.  That slice is the same for the dispatch input and
    for the combined output, because both are ordered by destination rank.
    """

    send_counts: list
    recv_counts: list
    row_ranges: list


def _ep_dispatch_chunks(
    send_counts: list,
    receive_counts: list,
    ep_size: int,
    chunk_count: int,
    ep_rank: int,
) -> _EPDispatchChunks:
    """Split the routed exchange into ``chunk_count`` self-consistent slices.

    Chunk ``c`` of rank ``r`` carries exactly the rows ``r`` dispatches to rank
    group ``(c + group(r)) % chunk_count`` -- a contiguous slice of the sorted
    expert-major stream -- and expects its reply from rank group
    ``(group(r) - c) % chunk_count``.  Every count is derived from the counts
    the caller already exchanged in :func:`_prepare_ep_dispatch`; no extra
    collective is needed to learn a chunk's traffic.

    The per-chunk rotation is what makes the pipeline meaningful: pinned to the
    rank's own destination group, a chunk would receive *all* of the rank's rows
    (a rank receives exactly the rows addressed to it) and the other chunks
    would have nothing to compute on.  Rotating by the rank group spreads both
    the sends and the receives of every rank evenly over the chunks.

    Args:
        send_counts: Rows this rank dispatches to each EP rank.
        receive_counts: Rows this rank receives from each EP rank.
        ep_size: Size of the EP group.
        chunk_count: Number of chunks (``2 <= chunk_count <= ep_size``).
        ep_rank: This rank's index in the EP group.

    Returns:
        The :class:`_EPDispatchChunks` plan, in chunk order.
    """
    row_offsets = [0]
    for count in send_counts:
        row_offsets.append(row_offsets[-1] + count)
    own_group = _chunk_rank_group(ep_rank, ep_size, chunk_count)
    send_plan, recv_plan, row_ranges = [], [], []
    for chunk in range(chunk_count):
        first, last = _chunk_rank_range(
            (chunk + own_group) % chunk_count, ep_size, chunk_count)
        send_plan.append(
            [send_counts[rank] if first <= rank < last else 0 for rank in range(ep_size)])
        row_ranges.append((row_offsets[first], row_offsets[last]))
        first, last = _chunk_rank_range(
            (own_group - chunk) % chunk_count, ep_size, chunk_count)
        recv_plan.append(
            [receive_counts[rank] if first <= rank < last else 0 for rank in range(ep_size)])
    return _EPDispatchChunks(send_plan, recv_plan, row_ranges)


def _run_ep_local_experts(
    module: Any,
    dispatched_states: torch.Tensor,
    dispatched_indices: torch.Tensor,
    send_counts: list[int],
    receive_counts: list[int],
    ep_group: Any,
    expert_offset: int,
) -> torch.Tensor:
    """Dispatch tokens, run local experts, and return combined expert outputs."""
    received_states = ep_all_to_all(dispatched_states, send_counts, receive_counts, ep_group)
    received_indices = ep_all_to_all(
        dispatched_indices, send_counts, receive_counts, ep_group
    ).squeeze(-1)
    local_outputs = module.experts(received_states, received_indices - expert_offset)
    return ep_all_to_all(local_outputs.contiguous(), receive_counts, send_counts, ep_group)


def _dispatch_chunk(
    dispatched_states: torch.Tensor,
    dispatched_expert_indices: torch.Tensor,
    chunk_send_counts: list[int],
    chunk_recv_counts: list[int],
    ep_group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Issue one chunk's hidden-state and expert-index exchanges lazily.

    Args:
        dispatched_states: Hidden rows belonging to this chunk.
        dispatched_expert_indices: Matching expert IDs, shaped [rows, 1].
        chunk_send_counts: Rows sent to each peer for this chunk.
        chunk_recv_counts: Rows received from each peer for this chunk.
        ep_group: Expert-parallel process group.

    Returns:
        Received hidden rows and expert IDs. The squeeze is a view; their first
        payload consumers order the collective wait.
    """
    received_states = ep_all_to_all_async(
        dispatched_states, chunk_send_counts, chunk_recv_counts, ep_group)
    received_indices = ep_all_to_all_async(
        dispatched_expert_indices, chunk_send_counts, chunk_recv_counts, ep_group).squeeze(-1)
    return received_states, received_indices


def _aggregate_ep_outputs(
    combined_outputs: torch.Tensor,
    expert_weights: torch.Tensor,
    source_indices: torch.Tensor,
    dispatch_order: torch.Tensor,
    output_shape: tuple[int, int, int],
    unpermute_indices: Optional[torch.Tensor] = None,
    routing_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Undo expert-major routing and aggregate weighted top-k outputs."""
    if unpermute_indices is not None:
        # Keep optional NPU dependencies out of the generic aggregation path.
        from hyper_parallel.components.functional.moe_token_unpermute import (  # pylint: disable=C0415
            moe_token_unpermute,
        )

        return moe_token_unpermute(combined_outputs, unpermute_indices, routing_weights).view(*output_shape)
    weighted_outputs = combined_outputs * expert_weights[dispatch_order].unsqueeze(-1)
    output = torch.zeros(
        output_shape[0] * output_shape[1],
        output_shape[-1],
        dtype=weighted_outputs.dtype,
        device=weighted_outputs.device,
    )
    output.index_add_(0, source_indices[dispatch_order], weighted_outputs)
    return output.to(combined_outputs.dtype).view(*output_shape)


def _aggregate_ep_chunks(
    combined_chunks: list[torch.Tensor],
    row_ranges: list[tuple[int, int]],
    expert_weights: torch.Tensor,
    source_indices: torch.Tensor,
    dispatch_order: torch.Tensor,
    output_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Aggregate the per-chunk combine outputs into the routed output.

    Same arithmetic as :func:`_aggregate_ep_outputs`, applied to each chunk's
    contiguous slice of the expert-major stream.  Every chunk accumulates into
    the one output buffer through ``index_add_`` (which sums duplicate source
    slots), so the routed output is assembled once, after the last chunk, and no
    stream-sized copy is needed to concatenate the chunks back into stream
    order.  Reading a chunk is also what waits for its combine exchange, so the
    wait of chunk ``c`` lands after chunk ``c + 1``'s experts have been issued.

    Args:
        combined_chunks: Per-chunk combine tensors, consumed in plan order.
        row_ranges: Matching half-open slices of the expert-major stream.
        expert_weights: Flattened routing weights, ``[tokens * top_k]``.
        source_indices: Source token of every routed slot, ``[tokens * top_k]``.
        dispatch_order: The dispatch permutation, ``[tokens * top_k]``.
        output_shape: Shape of the routed output.

    Returns:
        The routed output, shape ``output_shape``.
    """
    reference = combined_chunks[0]
    output = torch.zeros(
        output_shape[0] * output_shape[1],
        output_shape[-1],
        dtype=torch.promote_types(reference.dtype, expert_weights.dtype),
        device=reference.device,
    )
    for combined, (row_start, row_end) in zip(combined_chunks, row_ranges):
        rows = dispatch_order[row_start:row_end]
        # Reading a chunk is what orders its stream on the pending combine.
        weighted_outputs = combined * expert_weights[rows].unsqueeze(-1)
        output.index_add_(0, source_indices[rows], weighted_outputs)
    return output.to(reference.dtype).view(*output_shape)


def _run_ep_local_experts_chunked(
    module: Any,
    plan: _EPDispatchChunks,
    dispatched_states: torch.Tensor,
    dispatched_indices: torch.Tensor,
    ep_group: Any,
    expert_offset: int,
) -> list[torch.Tensor]:
    """Dispatch, run the local experts and combine, one chunk at a time.

    The chunks are software-pipelined: the next chunk's dispatch exchange is
    issued *before* the current chunk's experts run, so the GEMM overlaps the
    in-flight exchange, and each chunk's combine is issued right after its GEMM,
    so it overlaps the next chunk's GEMM.  Only chunk 0's dispatch and the last
    chunk's combine stay fully exposed.

    Neither wait is forced here -- the experts' first read materializes the
    dispatch, and the aggregation is what reads a combine -- so the chunked path
    keeps the lazy-wait property of :func:`ep_all_to_all_async`.

    Args:
        module: MoE block exposing ``experts`` (with ``local_expert_count``).
        plan: Chunk plan from :func:`_ep_dispatch_chunks`.
        dispatched_states: Expert-major hidden states, ``[rows, H]``.
        dispatched_indices: Matching expert indices, ``[rows, 1]``.
        ep_group: Extended EP process group.
        expert_offset: First global expert index owned by this rank.

    Returns:
        One combine output per chunk, in plan order (their wait is pending).
    """
    chunk_count = len(plan.row_ranges)

    def issue(chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Issue chunk ``chunk``'s dispatch exchange over its stream slice.

        Args:
            chunk: Chunk index into ``plan``.

        Returns:
            The chunk's pending ``(received_states, received_indices)``.
        """
        row_start, row_end = plan.row_ranges[chunk]
        return _dispatch_chunk(
            dispatched_states[row_start:row_end],
            dispatched_indices[row_start:row_end],
            plan.send_counts[chunk],
            plan.recv_counts[chunk],
            ep_group,
        )

    incoming = issue(0)
    combined_chunks = []
    for chunk in range(chunk_count):
        # Issued before the experts of this chunk run: that exchange is what
        # their GEMM overlaps.
        following = issue(chunk + 1) if chunk + 1 < chunk_count else None
        received_states, received_indices = incoming
        local_outputs = module.experts(
            received_states,
            received_indices - expert_offset)
        combined_chunks.append(ep_all_to_all_async(
            local_outputs.contiguous(),
            plan.recv_counts[chunk],
            plan.send_counts[chunk],
            ep_group,
        ))
        incoming = following
    return combined_chunks


def _chunked_ep_forward(
    module: Any,
    dispatch: "_EPDispatch",
    ep_group: Any,
    ep_size: int,
    ep_rank: int,
    chunk_count: int,
    expert_offset: int,
    output_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Run routed forward over the bounded target-rank groups as a software pipeline.

    Split out of :func:`ep_routed_forward` so that entry point stays readable; the schedule
    itself is unchanged (see :func:`_run_ep_local_experts_chunked`).
    """
    plan = _ep_dispatch_chunks(
        dispatch.send_counts, dispatch.receive_counts, ep_size, chunk_count, ep_rank)
    combined_chunks = _run_ep_local_experts_chunked(
        module,
        plan,
        dispatch.states,
        dispatch.expert_indices,
        ep_group,
        expert_offset,
    )
    return _aggregate_ep_chunks(
        combined_chunks,
        plan.row_ranges,
        dispatch.expert_weights,
        dispatch.source_indices,
        dispatch.dispatch_order,
        output_shape,
    )


def ep_routed_forward(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    router_fn: Callable,
    ep_group: Any,
    use_fused_moe_token_permute_and_unpermute: bool = True,
) -> torch.Tensor:
    """Routed-experts pipeline: SP-in (local chunk) -> all communication
    inside -> SP-out. **Routed branch only.**

    This primitive deliberately does NOT handle shared experts / scalar
    gates / branch merging — that composition is model semantics and belongs
    to the caller (an ep_compute.py archetype or a user-written factory,
    accuracy_fix_plan.md §3). There is no ``tp_group`` parameter: if the
    caller invokes a nested-boundary submodule (e.g. ``module.shared_expert``),
    that submodule's own boundary performs its TP communication — the
    **nested-boundary call contract**: the nested boundary receives the
    parent's local layout, exclusively owns its parameter layout and TP
    communication, and returns its declared out_dst layout. The caller MUST
    NOT repeat a compensating collective over the nested boundary's mesh.

    Communication flow (isomorphic to Megatron token_dispatcher.py
    MoEAlltoAllTokenDispatcher):
    router (local chunk, no communication) -> a2a dispatch (extended EP
    group, including TP ranks) -> local SwiGLU (complete expert weights, no
    internal communication, no Partial) -> a2a combine (returns over the
    same group) -> weighted aggregation.

    Input hidden [B, S/tp, H] (local sequence chunk, boundary identity);
    output [B, S/tp, H] (complete, boundary identity).

    ``router_fn`` is supplied BY THE CALLER (explicit choice, e.g. an entry
    of MOE_ROUTER_ADAPTERS picked by name in the factory code).

    With multiple EP ranks the exchange uses two contiguous target-rank groups
    and is software-pipelined (see
    :func:`_run_ep_local_experts_chunked`): the math is the same up to the
    grouping of the experts' GEMM, and the routed all-to-all -- the one
    collective of the step with no independent work to hide behind -- gets the
    chunk GEMMs to overlap with. EP1 retains the single-chunk schedule.

    ``use_fused_moe_token_permute_and_unpermute`` replaces the generic sort/index-add routing
    with NPU token permute/unpermute. It defaults to enabled on NPU and does
    not require grouped GEMM. CPU/GPU, chunked dispatch and capacity-limited
    dispatch retain the generic permutation path.

    Extended EP group = the ep axis of the derived expert mesh (flatten
    ep_size consecutive ranks: first span the TP group, then extend to
    adjacent dp/cp ranks; MindSpeed TP-extend-EP / Megatron etp=1 + ep
    homogeneous across TP). Expert weights are only Shard(0) along the
    expert dim -- each rank holds num_experts/ep_size complete experts, so
    there is no all_gather/reduce_scatter pair.
    """
    ep_size = ep_group.size()
    ep_rank = dist.get_rank(group=ep_group)
    local_expert_count = module.experts.local_expert_count
    global_expert_count = local_expert_count * ep_size
    expert_offset = ep_rank * local_expert_count
    # Resolved before the dispatch preparation so a mistyped knob fails before
    # any collective is issued.
    chunk_count = _resolve_dispatch_chunks(ep_size)

    output_shape = tuple(hidden_states.shape)
    topk_indices, topk_weights = router_fn(module, hidden_states)  # [T, K]
    dispatch = _prepare_ep_dispatch(
        hidden_states,
        topk_indices,
        topk_weights,
        local_expert_count=local_expert_count,
        global_expert_count=global_expert_count,
        ep_size=ep_size,
        ep_group=ep_group,
        use_fused_moe_token_permute_and_unpermute=(
            use_fused_moe_token_permute_and_unpermute
            and chunk_count == 1 and _resolve_capacity_factor(module) is None
        ),
    )
    if chunk_count > 1:
        return _chunked_ep_forward(
            module, dispatch, ep_group, ep_size, ep_rank, chunk_count, expert_offset,
            output_shape,
        )
    combined_expert_outputs = _run_ep_local_experts(
        module,
        dispatch.states,
        dispatch.expert_indices,
        dispatch.send_counts,
        dispatch.receive_counts,
        ep_group,
        expert_offset,
    )
    return _aggregate_ep_outputs(
        combined_expert_outputs,
        dispatch.expert_weights,
        dispatch.source_indices,
        dispatch.dispatch_order,
        output_shape,
        unpermute_indices=dispatch.unpermute_indices,
        routing_weights=dispatch.routing_weights,
    )


class _EPRoutedChunks(NamedTuple):
    """Pending tensors for the two target-rank groups.

    Each received_states/received_indices entry is an issued exchange whose first
    payload consumer orders the wait. The plan retains its traffic and row slice.
    """

    plan: _EPDispatchChunks
    received_states: list
    received_indices: list


class EPRoutedState(NamedTuple):
    """Routed state between asynchronous dispatch and experts+combine.

    The received tensors materialize at their first non-view consumers. With two
    target-rank groups, received_states and received_indices stay None and the
    per-group tensors travel in chunks. Full send/receive counts remain available.
    """

    source_token_indices: torch.Tensor
    flattened_expert_weights: torch.Tensor
    dispatch_order: torch.Tensor
    received_states: Optional[Any]
    received_indices: Optional[Any]
    send_counts: list
    receive_counts: list
    output_shape: tuple
    chunks: Optional[_EPRoutedChunks] = None


def _capacity_limited_slots(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    capacity_factor: float,
    global_expert_count: int,
) -> torch.Tensor:
    """Bound every expert's routed slots and report the drop rate sparsely.

    Split out of :func:`ep_routed_dispatch` to keep that entry point readable; the reporting
    cadence is part of the contract (see the comment inside).
    """
    keep, dropped = apply_capacity_limit(
        topk_indices, capacity_factor, global_expert_count, topk_weights)
    # Draining the drop count is a host sync, so pay it on the first dispatch and then
    # sparsely -- never once per layer, which is what made the first sweep look +90%.
    global _CAPACITY_SYNC_COUNT  # pylint: disable=global-statement
    _CAPACITY_SYNC_COUNT += 1
    if _CAPACITY_SYNC_COUNT == 1 or _CAPACITY_SYNC_COUNT % 200 == 0:
        total_slots = topk_indices.numel()
        dropped_slots = int(dropped.item())
        logger.info(
            "EP capacity limit: factor=%s order=%s dropped %.2f%% of routed slots "
            "(%d experts, call %d)",
            capacity_factor, _resolve_capacity_order(),
            100.0 * dropped_slots / max(total_slots, 1),
            global_expert_count, _CAPACITY_SYNC_COUNT,
        )
    return keep


def _issue_chunked_dispatch(
    dispatch: "_EPDispatch",
    ep_group: Any,
    ep_size: int,
    ep_rank: int,
    chunk_count: int,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
) -> "EPRoutedState":
    """Issue every chunk's split exchange up front and keep the handles pending.

    Split out of :func:`ep_routed_dispatch`; the pending-handle contract is unchanged.
    """
    plan = _ep_dispatch_chunks(
        dispatch.send_counts, dispatch.receive_counts, ep_size, chunk_count, ep_rank)
    chunk_states, chunk_indices = [], []
    for chunk in range(chunk_count):
        row_start, row_end = plan.row_ranges[chunk]
        chunk_state, chunk_index = _dispatch_chunk(
            dispatch.states[row_start:row_end],
            dispatch.expert_indices[row_start:row_end],
            plan.send_counts[chunk],
            plan.recv_counts[chunk],
            ep_group,
        )
        chunk_states.append(chunk_state)
        chunk_indices.append(chunk_index)
    return EPRoutedState(
        dispatch.source_indices,
        dispatch.expert_weights,
        dispatch.dispatch_order,
        None,
        None,
        dispatch.send_counts,
        dispatch.receive_counts,
        (batch_size, sequence_length, hidden_size),
        _EPRoutedChunks(plan, chunk_states, chunk_indices),
    )


def ep_routed_dispatch(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    router_fn: Callable,
    ep_group: Any,
) -> EPRoutedState:
    """Route locally and issue hidden-state and expert-index exchanges lazily.

    With multiple EP ranks, both target-rank groups are issued here. Independent
    shared-expert work may run before ep_routed_experts_and_combine consumes the
    received tensors and thereby orders their collective waits.

    Args:
        module: MoE block exposing ``experts`` (with ``local_expert_count``).
        hidden_states: Local sequence chunk, shape ``[B, S, H]``.
        router_fn: Router adapter, called as ``router_fn(module, hidden_states)``.
        ep_group: Extended EP process group.

    Returns:
        The pending :class:`EPRoutedState` to hand to
        :func:`ep_routed_experts_and_combine`.
    """
    ep_size = ep_group.size()
    local_expert_count = module.experts.local_expert_count
    global_expert_count = local_expert_count * ep_size
    # Resolved before the dispatch preparation so a mistyped knob fails before
    # any collective is issued.
    chunk_count = _resolve_dispatch_chunks(ep_size)

    batch_size, sequence_length, hidden_size = hidden_states.shape
    topk_indices, topk_weights = router_fn(module, hidden_states)
    capacity_factor = _resolve_capacity_factor(module)
    keep = None
    if capacity_factor is not None:
        keep = _capacity_limited_slots(
            topk_indices, topk_weights, capacity_factor, global_expert_count)
    dispatch = _prepare_ep_dispatch(
        hidden_states,
        topk_indices,
        topk_weights,
        local_expert_count=local_expert_count,
        global_expert_count=global_expert_count,
        ep_size=ep_size,
        ep_group=ep_group,
        keep=keep,
    )
    source_token_indices = dispatch.source_indices
    flattened_expert_weights = dispatch.expert_weights
    dispatch_order = dispatch.dispatch_order
    dispatched_states = dispatch.states
    dispatched_expert_indices = dispatch.expert_indices
    send_counts = dispatch.send_counts
    receive_counts = dispatch.receive_counts
    if chunk_count > 1:
        return _issue_chunked_dispatch(
            dispatch, ep_group, ep_size, dist.get_rank(group=ep_group), chunk_count,
            batch_size, sequence_length, hidden_size,
        )
    received_states = ep_all_to_all_async(
        dispatched_states, send_counts, receive_counts, ep_group)
    received_indices = ep_all_to_all_async(
        dispatched_expert_indices, send_counts, receive_counts, ep_group).squeeze(-1)
    return EPRoutedState(
        source_token_indices,
        flattened_expert_weights,
        dispatch_order,
        received_states,
        received_indices,
        send_counts,
        receive_counts,
        (batch_size, sequence_length, hidden_size),
    )


def ep_routed_experts_and_combine(
    module: Any,
    state: EPRoutedState,
    ep_group: Any,
) -> torch.Tensor:
    """Consume issued token exchanges, compute local experts and combine.

    Received-index arithmetic and the experts' first hidden-state reads order
    their respective collective waits. Per-group outputs accumulate in source
    token order using the original routing weights.

    Args:
        module: MoE block exposing ``experts`` (with ``local_expert_count``).
        state: State returned by :func:`ep_routed_dispatch`.
        ep_group: Extended EP process group.

    Returns:
        The routed branch output, shape ``state.output_shape``.
    """
    expert_offset = dist.get_rank(group=ep_group) * module.experts.local_expert_count
    if state.chunks is not None:
        plan = state.chunks.plan
        combined_chunks = []
        for chunk, (received_states, received_indices) in enumerate(
                zip(state.chunks.received_states, state.chunks.received_indices)):
            local_outputs = module.experts(
                received_states,
                received_indices - expert_offset)
            combined_chunks.append(ep_all_to_all_async(
                local_outputs.contiguous(),
                plan.recv_counts[chunk],
                plan.send_counts[chunk],
                ep_group,
            ))
        return _aggregate_ep_chunks(
            combined_chunks,
            plan.row_ranges,
            state.flattened_expert_weights,
            state.source_token_indices,
            state.dispatch_order,
            state.output_shape,
        )
    local_outputs = module.experts(
        state.received_states,
        state.received_indices - expert_offset)
    combined_expert_outputs = ep_all_to_all_async(
        local_outputs.contiguous(), state.receive_counts, state.send_counts, ep_group)
    return _aggregate_ep_outputs(
        combined_expert_outputs,
        state.flattened_expert_weights,
        state.source_token_indices,
        state.dispatch_order,
        state.output_shape,
    )
def _merge_shared_none(
    module: Any,
    hidden_states: torch.Tensor,
    routed: torch.Tensor,
) -> torch.Tensor:
    """Routed branch only (no shared expert)."""
    del module, hidden_states
    return routed


def _merge_shared_additive(
    module: Any,
    hidden_states: torch.Tensor,
    routed: torch.Tensor,
) -> torch.Tensor:
    """Compose ``routed + module.shared_experts(x)`` (DeepSeek-V3 / GLM-4-MoE).

    ``module.shared_experts(...)`` is a planned nested TP boundary: its exit
    already performs the TP reduction (nested-boundary call contract) — do NOT
    all-reduce its return value.
    """
    return routed + module.shared_experts(hidden_states)


def _merge_shared_gated(
    module: Any,
    hidden_states: torch.Tensor,
    routed: torch.Tensor,
) -> torch.Tensor:
    """Compose ``routed + sigmoid(shared_expert_gate(x)) * shared_expert(x)``.

    Qwen2-MoE layout.  ``module.shared_expert(...)`` is a planned nested TP
    boundary — the reduction contract is documented on
    :func:`_merge_shared_additive`.
    """
    shared = module.shared_expert(hidden_states)
    gate = torch.sigmoid(module.shared_expert_gate(hidden_states))
    return routed + gate * shared


# The three shared-branch merges for the codegen inline EP body
# (``moe_ep_forward``), keyed by STRUCTURAL mode.  They mirror the combine
# closures the native archetype factories in ``recipes.py`` define inline;
# codegen imports callable-free plans, so the generated body selects the
# composition here by string key.
SHARED_MERGE_MODES: dict[str, Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "none": _merge_shared_none,
    "additive": _merge_shared_additive,
    "gated": _merge_shared_gated,
}


def moe_ep_forward(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    router_kind: str,
    shared: str = "none",
    ep_group: Any,
) -> torch.Tensor:
    """Shared routed-MoE forward surfaced to generated inline EP artifacts.

    The single framework-generic EP body used across model families in
    place of per-family strategy strings: runs the routed branch and merges
    the shared branch according to ``shared``. Router semantics are selected
    by ``router_kind`` — a STRUCTURAL key (``softmax_topk`` /
    ``topk_router_module`` / ``sigmoid_group``), never a model name.

    ``shared`` selects an entry of :data:`SHARED_MERGE_MODES`, mirroring
    the combine closures the native archetype factories define inline:
      * ``"none"``     — routed output only.
      * ``"additive"`` — ``routed + module.shared_experts(x)``
                         (DeepSeek-V3 / GLM-4-MoE).
      * ``"gated"``    — ``routed + sigmoid(module.shared_expert_gate(x)) *
                         module.shared_expert(x)`` (Qwen2-MoE).

    Keeps the same numeric contract as ``ep_routed_forward`` plus the
    ``combine`` step of the runtime EP recipes.
    """
    merge = SHARED_MERGE_MODES.get(shared)
    if merge is None:
        raise ValueError(
            f"moe_ep_forward: unknown shared merge mode {shared!r} "
            f"(expected one of {sorted(SHARED_MERGE_MODES)})"
        )
    routed = ep_routed_forward(
        module,
        hidden_states,
        router_fn=MOE_ROUTER_ADAPTERS[router_kind],
        ep_group=ep_group,
    )
    return merge(module, hidden_states, routed)


def require_attrs(module: Any, *names: str, owner: str = "") -> None:
    """Assert that ``module`` has every attribute in ``names``; raise a
    teaching ValueError listing the module's ACTUAL children otherwise.

    Used by EP compute factories (archetype or user-written) at apply time:
    the factory body runs ONCE at apply time (before the wrapped forward
    ever executes), so an interface mismatch fails the model build in
    seconds instead of surfacing as a runtime AttributeError at step N.
    The check is structural (the names the implementation will call exist);
    it does not prove semantic correctness — that is vouched by numeric
    verification.
    """
    missing = [n for n in names if not hasattr(module, n)]
    if not missing:
        return
    children = [n for n, _ in module.named_children()]
    who = f"{owner} " if owner else ""
    raise ValueError(
        f"{who}expects MoE module attribute(s) {missing} on "
        f"{type(module).__name__}, but they do not exist; the module's "
        f"actual children are {children}. Pick the matching EP archetype "
        f"(see the archetype table in ep_compute.py), or write your own "
        f"factory (reference: examples/distributed/ep_factories.py) — the "
        f"names your compute_fn calls on module.<child> must match the "
        f"model's actual attribute names"
    )


def describe_moe_module(module: Any) -> str:
    """Structural diagnostic for a MoE module: child submodules, direct
    parameter shapes, and expert-related attributes — the facts needed to
    pick an EP archetype or write a custom factory. Returns the report;
    also logged at INFO."""
    lines = [f"MoE module: {type(module).__name__}"]
    children = list(module.named_children())
    lines.append(f"children ({len(children)}):")
    for name, child in children:
        lines.append(f"  - {name}: {type(child).__name__}")
    direct_params = list(module.named_parameters(recurse=False))
    if direct_params:
        lines.append("direct parameters:")
        for name, p in direct_params:
            lines.append(f"  - {name}: shape={tuple(p.shape)}")
    experts = getattr(module, "experts", None)
    if experts is not None:
        lines.append(
            f"experts: {type(experts).__name__}, "
            f"local_expert_count={getattr(experts, 'local_expert_count', '<unset>')}"
        )
    report = "\n".join(lines)
    return report
