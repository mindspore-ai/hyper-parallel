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
"""ep_utils: public EP primitives for MoE local-region compute (05 §6.4.7, D-09).

The framework owns the MECHANISM; the semantic composition (which router,
whether a shared expert exists, gate formulas, how branches merge) belongs to
the explicitly-selected semantic implementation — a built-in archetype
factory from ep_compute.py or a user-written factory (accuracy_fix_plan.md
§3). Nothing in this module probes model structure with getattr fallback
chains.

Public primitives:
1. **Backend-dispatched all_to_all** (``ep_all_to_all``): NCCL/HCCL use the
   ragged a2a (``_EPAllToAllUneven``, zero-padding); gloo and other backends
   that do not support ragged a2a use pad-to-max + ``all_to_all_single``
   (``_EPAllToAllPadded``). Both paths are numerically equivalent (padding
   only adds filler rows that do not participate in computation).
2. **Router adapter registry** (``MOE_ROUTER_ADAPTERS``): routing semantics
   are model-specific (softmax/sigmoid, top-k normalization, scaling), while
   the expert MLP structure is uniform (SwiGLU).
   ``(module, hidden) -> (topk_idx [T,K] int64, topk_w [T,K] float)``.
   A factory picks an adapter BY NAME in its own code — the choice is
   explicit, never inferred.
3. **Routed-experts pipeline** (``ep_routed_forward``, 05 §6.4.8): router
   (local chunk) -> a2a dispatch (extended EP group) -> local SwiGLU
   (complete expert weights, no internal communication) -> a2a combine ->
   weighted aggregation. SP-in -> SP-out, all communication cohesive inside.
   NO shared-expert/gate branch — that composition is the caller's job.
4. **Expert entry point** (``bind_local_expert_forward`` /
   ``resolve_swiglu_weights``): installs ``experts.forward`` so nested FSDP
   hooks unshard/reshard around the local SwiGLU.
5. **Interface helpers** (``require_attrs`` / ``describe_moe_module``):
   build-time interface assertions with teaching errors, and a structural
   diagnostic for mapping a concrete MoE module to an archetype.
"""

import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hyper_models.components.ops.npu_grouped_swiglu import (
    npu_grouped_swiglu,
)

# ────────────────────────────────────────────────────────────────────────────
# Backend-dispatched all_to_all
# ────────────────────────────────────────────────────────────────────────────

# Measured (2026-07-20): gloo supports equal-length all_to_all_single
# (including int64), but not the ragged list-based all_to_all; NCCL/HCCL
# support both.
_UNEVEN_A2A_BACKENDS = ("nccl", "hccl")


def _backend_supports_uneven_a2a(group) -> bool:
    return dist.get_backend(group) in _UNEVEN_A2A_BACKENDS


class _EPAllToAllUneven(torch.autograd.Function):  # pylint: disable=abstract-method
    """Ragged all_to_all (NCCL/HCCL production path): split by send/recv counts.

    forward:  split(x, send_counts) -> dist.all_to_all(out_list, in_list) -> cat
    backward: swap send/recv counts and run the ragged all_to_all again
              (a2a is self-inverse).
    """

    @staticmethod
    def forward(ctx, x, send_counts, recv_counts, group):  # pylint: disable=arguments-differ
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        out = x.new_empty((sum(recv_counts),) + tuple(x.shape[1:]))
        dist.all_to_all(list(out.split(recv_counts)),
                        list(x.split(send_counts)), group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        grad = _EPAllToAllUneven.apply(
            grad_output.contiguous(), ctx.recv_counts, ctx.send_counts, ctx.group)
        return grad, None, None, None


class _EPAllToAllPadded(torch.autograd.Function):  # pylint: disable=abstract-method
    """pad-to-max + all_to_all_single (gloo test path).

    forward:  pad each dest chunk to the global max(counts) (a2a_single
              requires equal-length chunks per rank -> pad_to must be
              globally consistent, obtained via all_reduce MAX)
              -> a2a_single -> unpad by recv_counts;
    backward: pad by recv_counts -> a2a_single (equal-length self-inverse)
              -> unpad by send_counts.
    """

    @staticmethod
    def _pad_and_exchange(x, counts, pad_to, group):
        """Pad each chunk to pad_to, run equal-length a2a_single, return [ep*pad_to, ...]."""
        chunks = []
        for chunk, n in zip(x.split(counts), counts):
            if n < pad_to:
                pad = x.new_zeros((pad_to - n,) + tuple(x.shape[1:]))
                chunk = torch.cat([chunk, pad])
            chunks.append(chunk)
        send = torch.cat(chunks).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return recv

    @staticmethod
    def _unpad(recv, counts, pad_to):
        """Take the valid rows from the equal-length buffer per counts and cat."""
        pieces = []
        for i, n in enumerate(counts):
            if n > 0:
                pieces.append(recv[i * pad_to: i * pad_to + n])
        if not pieces:
            return recv.new_zeros((0,) + tuple(recv.shape[1:]))
        return torch.cat(pieces)

    @staticmethod
    def forward(ctx, x, send_counts, recv_counts, group):  # pylint: disable=arguments-differ
        """Exchange padded expert-token chunks and retain counts for backward."""
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        local_max = max([*send_counts, *recv_counts, 1])
        pad_to = x.new_tensor([local_max], dtype=torch.int64)
        dist.all_reduce(pad_to, op=dist.ReduceOp.MAX, group=group)
        ctx.pad_to = pad_to = int(pad_to.item())
        recv = _EPAllToAllPadded._pad_and_exchange(x, send_counts, pad_to, group)
        return _EPAllToAllPadded._unpad(recv, recv_counts, pad_to)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        # backward = reversed a2a: pad by recv_counts -> a2a_single -> unpad by send_counts
        recv = _EPAllToAllPadded._pad_and_exchange(
            grad_output.contiguous(), ctx.recv_counts, ctx.pad_to, ctx.group)
        grad = _EPAllToAllPadded._unpad(recv, ctx.send_counts, ctx.pad_to)
        return grad, None, None, None


def ep_all_to_all(x, send_counts, recv_counts, group):
    """Unified entry for EP token exchange (autograd-differentiable).

    send_counts/recv_counts: list[int], length ep_size, row counts per dest/src rank.
    NCCL/HCCL -> ragged a2a (zero-padding); other backends (gloo test path) -> pad-to-max.
    """
    if _backend_supports_uneven_a2a(group):
        return _EPAllToAllUneven.apply(x, send_counts, recv_counts, group)
    return _EPAllToAllPadded.apply(x, send_counts, recv_counts, group)


# ────────────────────────────────────────────────────────────────────────────
# Router adapter registry
# ────────────────────────────────────────────────────────────────────────────

def _softmax_topk_router(module, hidden_states):
    """default adapter: softmax -> topk -> normalize by sum (Mixtral/Qwen3 semantics).

    top_k source: config.num_experts_per_tok / config.top_k / module.top_k (default 2);
    normalization switch: config.norm_topk_prob (default True).
    """
    gate = getattr(module, "gate", None)
    if gate is None:
        gate = getattr(module, "router", None)
    if gate is None:
        raise AttributeError(
            f"{type(module).__name__}: router not found (neither gate nor router "
            "attribute exists); please register a custom MOE_ROUTER_ADAPTERS entry"
        )
    cfg = getattr(module, "config", None)
    logits = gate(hidden_states)
    logits = logits.view(-1, logits.shape[-1])
    top_k = (getattr(cfg, "num_experts_per_tok", None)
             or getattr(cfg, "top_k", None)
             or getattr(module, "top_k", 2))
    weights = logits.softmax(-1)
    topk_w, topk_idx = weights.topk(int(top_k), dim=-1)
    if getattr(cfg, "norm_topk_prob", True):
        topk_w = topk_w / topk_w.sum(-1, keepdim=True).clamp_min(1e-20)
    return topk_idx, topk_w


def _topk_router_module(module, hidden_states):
    """Qwen2/Qwen3/Mixtral adapter: gate is a TopKRouter module (after the HF 2025
    refactor); forward directly returns (logits, scores [T,K], indices [T,K])
    -- take the latter two."""
    gate = getattr(module, "gate", None)
    if gate is None:
        gate = getattr(module, "router", None)
    out = gate(hidden_states)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        _, scores, indices = out
        return indices, scores
    raise TypeError(
        f"{type(module).__name__}: TopKRouter should return (logits, scores, indices), "
        f"got {type(out).__name__} -- use the default adapter or register a custom adapter"
    )


def _sigmoid_group_router(module, hidden_states):
    """deepseekv3/glm4moe adapter: sigmoid + e_score_correction_bias +
    group-limited topk + (optional) normalization + routed_scaling_factor
    (step-by-step consistent with HF DeepseekV3MoE.route_tokens_to_experts /
    Glm4MoeMoE).

    Parameter source: the module's own attributes take precedence
    (n_group/topk_group/top_k/norm_topk_prob/routed_scaling_factor), falling
    back to module.config; when n_group is missing or <=1, the group-limited
    filter is skipped.
    """
    gate = getattr(module, "gate", None)
    if gate is None:
        gate = getattr(module, "router", None)
    cfg = getattr(module, "config", None)

    def _attr(name, default=None):
        v = getattr(module, name, None)
        if v is None and cfg is not None:
            v = getattr(cfg, name, None)
        return default if v is None else v

    logits = gate(hidden_states)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = logits.view(-1, logits.shape[-1]).float()
    e_total = logits.shape[-1]
    scores = logits.sigmoid()
    bias = getattr(gate, "e_score_correction_bias", None)
    scores_for_choice = scores + bias if bias is not None else scores

    n_group = int(_attr("n_group", 0) or 0)
    topk_group = int(_attr("topk_group", 0) or 0)
    top_k = int(_attr("top_k", None) or _attr("num_experts_per_tok", 2))
    if n_group > 1 and topk_group > 0:
        group_scores = (scores_for_choice
                        .view(-1, n_group, e_total // n_group)
                        .topk(2, dim=-1)[0].sum(dim=-1))
        group_idx = group_scores.topk(topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (group_mask.unsqueeze(-1)
                      .expand(-1, n_group, e_total // n_group)
                      .reshape(-1, e_total))
        scores_for_choice = scores_for_choice.masked_fill(
            ~score_mask.bool(), float("-inf"))

    topk_idx = scores_for_choice.topk(top_k, dim=-1, sorted=False)[1]
    topk_w = scores.gather(1, topk_idx)
    if _attr("norm_topk_prob", False):
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
    topk_w = topk_w * float(_attr("routed_scaling_factor", 1.0))
    return topk_idx, topk_w


# {adapter_name: (module, hidden) -> (topk_idx [T,K] int64, topk_w [T,K] float)}
# A compute factory picks an adapter BY NAME in its own code (the choice is
# explicit — the framework never infers routing semantics). Keys use the
# canonical arch spelling from ShardingPlanner._get_architecture
# (config.architectures lowercased with suffix stripped, e.g. "qwen3moe");
# underscored aliases cover the config.model_type path (e.g. "qwen3_moe").
MOE_ROUTER_ADAPTERS = {
    "default": _softmax_topk_router,
    "qwen2moe": _topk_router_module,
    "qwen2_moe": _topk_router_module,
    "qwen3moe": _topk_router_module,
    "qwen3_moe": _topk_router_module,
    "mixtral": _topk_router_module,
    "deepseekv3": _sigmoid_group_router,
    "deepseek_v3": _sigmoid_group_router,
    "glm4moe": _sigmoid_group_router,
    "glm4_moe": _sigmoid_group_router,
}


# ────────────────────────────────────────────────────────────────────────────
# EP forward for HF-native MoE (D-09c)
# ────────────────────────────────────────────────────────────────────────────

def resolve_swiglu_weights(experts):
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


def _local_swiglu_expert_forward(experts, dispatched_states, local_expert_indices):
    """Compute dispatched tokens with the local stacked SwiGLU experts.

    This function is installed as ``experts.forward`` for the HF-native EP
    path. Calling the expert module, instead of indexing its parameters from
    the parent MoE forward, allows nested FSDP forward hooks to unshard and
    reshard expert parameters around the local computation.
    """
    gate_weight, up_weight, down_weight = resolve_swiglu_weights(experts)
    token_order = local_expert_indices.argsort()
    sorted_states = dispatched_states[token_order]
    local_expert_counts = torch.bincount(
        local_expert_indices,
        minlength=experts.local_expert_count,
    )
    if getattr(experts, "_ep_use_grouped_gemm", False):
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

    sorted_outputs = []
    token_start = 0
    for local_expert_index in range(experts.local_expert_count):
        expert_token_count = int(local_expert_counts[local_expert_index])
        expert_states = sorted_states[token_start:token_start + expert_token_count]
        if up_weight is None:
            gate_states, up_states = F.linear(  # pylint: disable=not-callable
                expert_states,
                gate_weight[local_expert_index],
            ).chunk(2, dim=-1)
        else:
            gate_states = F.linear(  # pylint: disable=not-callable
                expert_states, gate_weight[local_expert_index]
            )
            up_states = F.linear(  # pylint: disable=not-callable
                expert_states, up_weight[local_expert_index]
            )
        activation = getattr(experts, "_ep_act_fn", F.silu)
        sorted_outputs.append(
            F.linear(  # pylint: disable=not-callable
                activation(gate_states) * up_states,
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


def bind_local_expert_forward(module, ep_size, use_grouped_gemm=False):
    """Install the local expert compute entry used by TP-extend-EP.

    Called by the EP compute factory (archetype or user-written) at apply
    time: sets ``module.experts.local_expert_count`` and installs
    ``experts.forward`` so nested FSDP hooks unshard/reshard around the
    local SwiGLU computation.
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
    module.experts._ep_act_fn = activation
    module.experts._ep_use_grouped_gemm = use_grouped_gemm
    module.experts.forward = types.MethodType(
        _local_swiglu_expert_forward,
        module.experts,
    )


# ────────────────────────────────────────────────────────────────────────────
# Routed-experts pipeline primitive (D-10, 05 §6.4.8, isomorphic to Megatron
# MoEAlltoAllTokenDispatcher + expert_tensor_parallel_size=1)
# ────────────────────────────────────────────────────────────────────────────

def ep_routed_forward(module, hidden_states, *, router_fn, ep_group):
    """Routed-experts pipeline: SP-in (local chunk) -> all communication
    inside -> SP-out. **Routed branch only.**

    This primitive deliberately does NOT handle shared experts / scalar
    gates / branch merging — that composition is model semantics and belongs
    to the caller (an ep_compute.py archetype or a user-written factory,
    accuracy_fix_plan.md §3). There is no ``tp_group`` parameter: if the
    caller invokes a nested-boundary submodule (e.g. ``module.shared_expert``),
    that submodule's own boundary performs its TP communication — the
    **nested-boundary call contract**:

    1. the input is the parent local region's current logical local layout;
    2. the nested boundary exclusively owns its parameter layout and its TP
       communication (entry/exit via its own PrecompiledBoundary);
    3. the return value is already the nested boundary's out_dst logical
       layout (e.g. under SP: the complete per-token values of the local
       sequence chunk);
    4. the caller MUST NOT repeat any compensating collective
       (all-reduce / reduce-scatter / all-gather) on the returned value
       over the nested boundary's mesh.

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

    batch_size, sequence_length, hidden_size = hidden_states.shape
    flattened_states = hidden_states.reshape(-1, hidden_size)
    token_count = flattened_states.shape[0]

    # 1. router (local chunk): each rank's chunk differs -> no token duplication
    topk_indices, topk_weights = router_fn(module, hidden_states)  # [T, K]
    experts_per_token = topk_indices.shape[1]
    flattened_expert_indices = topk_indices.reshape(-1)
    flattened_expert_weights = topk_weights.reshape(-1).to(flattened_states.dtype)
    source_token_indices = torch.arange(
        token_count,
        device=flattened_states.device,
    ).repeat_interleave(experts_per_token)
    destination_ranks = torch.div(
        flattened_expert_indices,
        local_expert_count,
        rounding_mode="floor",
    )

    # 2. sort by (dest, expert) + counts exchange (extended EP group)
    dispatch_order = (
        destination_ranks * global_expert_count + flattened_expert_indices
    ).argsort()
    dispatched_states = flattened_states[source_token_indices[dispatch_order]].contiguous()
    dispatched_expert_indices = flattened_expert_indices[
        dispatch_order
    ].unsqueeze(-1).contiguous()
    send_counts_tensor = torch.bincount(destination_ranks, minlength=ep_size)
    receive_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(
        receive_counts_tensor,
        send_counts_tensor,
        group=ep_group,
    )
    send_counts = send_counts_tensor.tolist()
    receive_counts = receive_counts_tensor.tolist()

    # 3. a2a dispatch over the extended EP group (tokens carry the full H,
    #    sent to the rank holding the expert)
    received_states = ep_all_to_all(
        dispatched_states,
        send_counts,
        receive_counts,
        ep_group,
    )
    received_global_expert_indices = ep_all_to_all(
        dispatched_expert_indices,
        send_counts,
        receive_counts,
        ep_group,
    ).squeeze(-1)

    # 4. Call the local expert module so nested FSDP hooks unshard before
    #    accessing expert weights and reshard after the SwiGLU computation.
    local_expert_outputs = module.experts(
        received_states,
        received_global_expert_indices - expert_offset,
    )

    # 5. a2a combine over the extended EP group -> inverse perm -> weighted
    #    aggregation by topk_w
    combined_expert_outputs = ep_all_to_all(
        local_expert_outputs.contiguous(),
        receive_counts,
        send_counts,
        ep_group,
    )
    weighted_expert_outputs = combined_expert_outputs * flattened_expert_weights[
        dispatch_order
    ].unsqueeze(-1)
    # ``dispatch_order`` is globally expert-major, so index-add accumulates a
    # token's selected experts in ascending expert order. This matches the HF
    # Qwen3 expert loop and avoids BF16 drift from summing in router top-k order.
    output = torch.zeros(
        token_count,
        hidden_size,
        dtype=weighted_expert_outputs.dtype,
        device=weighted_expert_outputs.device,
    )
    output.index_add_(
        0,
        source_token_indices[dispatch_order],
        weighted_expert_outputs,
    )
    output = output.view(
        batch_size,
        sequence_length,
        hidden_size,
    )
    return output


# ────────────────────────────────────────────────────────────────────────────
# Interface helpers (accuracy_fix_plan.md §3 E2): build-time interface
# assertions with teaching errors + a structural diagnostic
# ────────────────────────────────────────────────────────────────────────────

def require_attrs(module, *names, owner: str = ""):
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


def describe_moe_module(module) -> str:
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
