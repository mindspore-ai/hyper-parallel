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

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F

from hyper_parallel.components.ops import (
    grouped_matmul, grouped_swiglu, moe_token_permute, moe_token_unpermute, swiglu,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _install_bound_forward,
)
from hyper_parallel.distributed.expert_parallel.routing import (
    apply_capacity_limit,
)
from hyper_parallel.distributed.expert_parallel.collectives import (
    ep_exchange_counts,
    ep_all_to_all,
    ep_all_to_all_async,
    ep_split_sizes,
)


_SORT_KEY_FP32_LIMIT = 2**24


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


def _argsort_keys(keys: torch.Tensor, *, bound: int) -> torch.Tensor:
    """Sort nonnegative expert IDs below ``bound`` without losing precision."""
    # Ascend can use an AICore sort when the integer IDs are exact in FP32.
    if 0 < bound <= _SORT_KEY_FP32_LIMIT:
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
    token_order = _argsort_keys(local_expert_indices, bound=experts.local_expert_count)
    sorted_states = dispatched_states[token_order]
    local_expert_counts = _expert_token_counts(
        local_expert_indices, experts.local_expert_count)
    if getattr(experts, "_ep_use_grouped_gemm", False):
        grouped_forward = getattr(experts, "forward_expert_major", None)
        if callable(grouped_forward):
            sorted_output = grouped_forward(sorted_states, local_expert_counts)
        else:
            gate_weight, up_weight, down_weight = resolve_swiglu_weights(experts)
            if up_weight is None:
                boundaries = local_expert_counts.cumsum(0)
                packed = grouped_matmul(sorted_states, gate_weight.transpose(1, 2), boundaries)
                sorted_output = grouped_matmul(swiglu(packed), down_weight.transpose(1, 2), boundaries)
            else:
                sorted_output = grouped_swiglu(
                    sorted_states, gate_weight, down_weight, up_weight, local_expert_counts,
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
    if getattr(module.experts, "requires_ep_grouped_gemm", False) and not use_grouped_gemm:
        raise ValueError(
            f"{type(module.experts).__name__} requires use_grouped_gemm=True in the EP method replacement"
        )
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
    """Read the configured expert capacity multiplier."""
    value = getattr(getattr(module, "experts", None), "capacity_factor", None)
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def _routed_slot_token_ids(
    token_count: int, experts_per_token: int, device: Any,
) -> torch.Tensor:
    """Return the source token of every routed slot, as an index tensor.

    Slot ``token * K + i`` holds the source index ``token``.

    Args:
        token_count: Number of local tokens ``T`` in this dispatch.
        experts_per_token: Routed experts per token ``K``.
        device: Device to build the tensor on.

    Returns:
        Contiguous int64 tensor of shape ``[T * K]``.

    """
    # Make a contiguous buffer even for a single token with multiple slots.
    return (
        torch.arange(token_count, device=device)
        .unsqueeze(1)
        .expand(token_count, experts_per_token)
        .reshape(-1)
        .contiguous()
    )


def _prepare_ep_dispatch(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    local_expert_count: int,
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
        expert_weights = topk_weights.reshape(-1).to(dtype=flattened_states.dtype)
        source_indices = _routed_slot_token_ids(
            token_count, experts_per_token, flattened_states.device)
    else:
        flat_keep = keep.reshape(-1)
        expert_indices = topk_indices.reshape(-1)[flat_keep]
        expert_weights = topk_weights.reshape(-1)[flat_keep].to(dtype=flattened_states.dtype)
        source_indices = _routed_slot_token_ids(
            token_count, experts_per_token, flattened_states.device)[flat_keep]
    use_fused_moe_token_permute_and_unpermute = (
        use_fused_moe_token_permute_and_unpermute
        and flattened_states.device.type == "npu" and keep is None
    )
    unpermute_indices = None
    if use_fused_moe_token_permute_and_unpermute:
        dispatched_states, dispatch_order, unpermute_indices = moe_token_permute(
            flattened_states, topk_indices,
        )
    else:
        # Global IDs are contiguous across destination ranks.
        dispatch_order = _argsort_keys(expert_indices, bound=local_expert_count * ep_size)
        dispatched_states = flattened_states[source_indices[dispatch_order]].contiguous()
    dispatched_indices = expert_indices[dispatch_order].unsqueeze(-1).contiguous()
    destination_ranks = torch.div(expert_indices, local_expert_count, rounding_mode="floor")
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
        return moe_token_unpermute(combined_outputs, unpermute_indices, routing_weights).view(*output_shape)
    weighted_outputs = combined_outputs * expert_weights[dispatch_order].unsqueeze(-1)
    output = torch.zeros(
        output_shape[0] * output_shape[1],
        output_shape[-1],
        dtype=weighted_outputs.dtype,
        device=weighted_outputs.device,
    )
    output.index_add_(0, source_indices[dispatch_order], weighted_outputs)
    return output.view(*output_shape)


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

    ``use_fused_moe_token_permute_and_unpermute`` replaces the generic sort/index-add routing
    with NPU token permute/unpermute. It defaults to enabled on NPU and does
    not require grouped GEMM. CPU/GPU and capacity-limited dispatch use the
    generic permutation path.

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
    expert_offset = ep_rank * local_expert_count
    output_shape = tuple(hidden_states.shape)
    dispatch = _plan_routed_slots(
        module, hidden_states, router_fn, ep_size, ep_group,
        fused_permute=use_fused_moe_token_permute_and_unpermute,
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


class EPRoutedState(NamedTuple):
    """Pending routed-experts state between dispatch and experts+combine.

    Produced by :func:`ep_routed_dispatch` and consumed by
    :func:`ep_routed_experts_and_combine`.  ``received_states`` /
    ``received_indices`` may still be in flight until their first non-view use.
    """

    source_token_indices: torch.Tensor
    flattened_expert_weights: torch.Tensor
    dispatch_order: torch.Tensor
    received_states: torch.Tensor
    received_indices: torch.Tensor
    send_counts: list[int]
    receive_counts: list[int]
    output_shape: tuple[int, int, int]


def _plan_routed_slots(
    module: Any,
    hidden_states: torch.Tensor,
    router_fn: Callable,
    ep_size: int,
    ep_group: Any,
    fused_permute: bool = False,
) -> _EPDispatch:
    """Route once and compact the same capacity-limited slots for all schedules."""
    local_expert_count = module.experts.local_expert_count
    global_expert_count = local_expert_count * ep_size
    topk_indices, topk_weights = router_fn(module, hidden_states)
    capacity_factor = _resolve_capacity_factor(module)
    keep = None
    if capacity_factor is not None:
        keep, _ = apply_capacity_limit(
            topk_indices, capacity_factor, global_expert_count, topk_weights,
        )
    return _prepare_ep_dispatch(
        hidden_states, topk_indices, topk_weights,
        local_expert_count=local_expert_count,
        ep_size=ep_size,
        ep_group=ep_group,
        keep=keep,
        use_fused_moe_token_permute_and_unpermute=fused_permute,
    )


def ep_routed_dispatch(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    router_fn: Callable,
    ep_group: Any,
) -> EPRoutedState:
    """Route locally and **launch** the token exchange without waiting.

    Same routing/dispatch preparation as :func:`ep_routed_forward`, but the two
    token exchanges are issued through :func:`ep_all_to_all_async`, whose wait
    is deferred to the first non-view consumer.  A caller that runs independent
    work between this call and :func:`ep_routed_experts_and_combine` (for
    DeepSeek-V3 style blocks: the shared-expert MLP, which depends only on
    ``hidden_states``) therefore overlaps that work with the in-flight
    exchange instead of serializing behind it.

    Args:
        module: MoE block exposing ``experts`` (with ``local_expert_count``).
        hidden_states: Local sequence chunk, shape ``[B, S, H]``.
        router_fn: Router adapter, called as ``router_fn(module, hidden_states)``.
        ep_group: Extended EP process group.

    Returns:
        The pending :class:`EPRoutedState` to hand to
        :func:`ep_routed_experts_and_combine`.
    """
    batch_size, sequence_length, hidden_size = hidden_states.shape
    dispatch = _plan_routed_slots(module, hidden_states, router_fn, ep_group.size(), ep_group)
    send_counts = dispatch.send_counts
    receive_counts = dispatch.receive_counts
    received_states = ep_all_to_all_async(dispatch.states, send_counts, receive_counts, ep_group)
    received_indices = ep_all_to_all_async(
        dispatch.expert_indices, send_counts, receive_counts, ep_group,
    ).squeeze(-1)
    return EPRoutedState(
        dispatch.source_indices,
        dispatch.expert_weights,
        dispatch.dispatch_order,
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
    """Run the local experts on a dispatched state and exchange the outputs back.

    The dispatch exchange reached ``state`` from :func:`ep_routed_dispatch` and
    is waited here -- at the first expert read, i.e. after whatever work the
    caller ran in between -- and the combine exchange is issued asynchronously
    as well, so its wait lands on the weighted aggregation instead of blocking
    right after the local experts.

    Args:
        module: MoE block exposing ``experts`` (with ``local_expert_count``).
        state: State returned by :func:`ep_routed_dispatch`.
        ep_group: Extended EP process group.

    Returns:
        The routed branch output, shape ``state.output_shape``.
    """
    expert_offset = dist.get_rank(group=ep_group) * module.experts.local_expert_count
    local_outputs = module.experts(
        state.received_states,
        state.received_indices - expert_offset)
    combined_expert_outputs = ep_all_to_all_async(
        local_outputs.contiguous(), state.receive_counts, state.send_counts, ep_group,
    )
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


# Shared-branch merges used by model adapters and native EP recipes.
SHARED_MERGE_MODES: dict[str, Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "none": _merge_shared_none,
}


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
