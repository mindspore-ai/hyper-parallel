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
"""ep_utils: EP passthrough for HF-native MoE (05 §6.4.7, D-09).

Three parts:
1. **Backend-dispatched all_to_all** (``_ep_all_to_all``): NCCL/HCCL use the
   ragged a2a (``_EPAllToAllUneven``, zero-padding); gloo and other backends
   that do not support ragged a2a use pad-to-max + ``all_to_all_single``
   (``_EPAllToAllPadded``). Both paths are numerically equivalent (padding
   only adds filler rows that do not participate in computation).
2. **Router adapter registry** (``MOE_ROUTER_ADAPTERS``): routing semantics
   are model-specific (softmax/sigmoid, top-k normalization, scaling), while
   the expert MLP structure is uniform (SwiGLU).
   ``(module, hidden) -> (topk_idx [T,K] int64, topk_w [T,K] float)``.
3. **TP-extend-EP forward** (``_hf_native_ep_compute``, 05 §6.4.8): router
   (local chunk) -> a2a dispatch (extended EP group) -> local SwiGLU
   (complete expert weights, no internal communication) -> a2a combine ->
   weighted aggregation (+ shared_experts).
   All local tensor, shared by production/validate dual-mode (local_region
   tolerant semantics). No all_gather/reduce_scatter -- expert weights are
   not TP-sharded (Megatron expert_tensor_parallel_size=1 +
   expert_model_parallel_size homogeneous across TP).
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

# ────────────────────────────────────────────────────────────────────────────
# Backend-dispatched all_to_all
# ────────────────────────────────────────────────────────────────────────────

# Measured (2026-07-20): gloo supports equal-length all_to_all_single
# (including int64), but not the ragged list-based all_to_all; NCCL/HCCL
# support both.
_UNEVEN_A2A_BACKENDS = ("nccl", "hccl")


def _backend_supports_uneven_a2a(group) -> bool:
    return dist.get_backend(group) in _UNEVEN_A2A_BACKENDS


class _EPAllToAllUneven(torch.autograd.Function):
    """Ragged all_to_all (NCCL/HCCL production path): split by send/recv counts.

    forward:  split(x, send_counts) -> dist.all_to_all(out_list, in_list) -> cat
    backward: swap send/recv counts and run the ragged all_to_all again
              (a2a is self-inverse).
    """

    @staticmethod
    def forward(ctx, x, send_counts, recv_counts, group):
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        out = x.new_empty((sum(recv_counts),) + tuple(x.shape[1:]))
        dist.all_to_all(list(out.split(recv_counts)),
                        list(x.split(send_counts)), group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad = _EPAllToAllUneven.apply(
            grad_output.contiguous(), ctx.recv_counts, ctx.send_counts, ctx.group)
        return grad, None, None, None


class _EPAllToAllPadded(torch.autograd.Function):
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
    def forward(ctx, x, send_counts, recv_counts, group):
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        local_max = max(max(send_counts), max(recv_counts), 1)
        pad_to = x.new_tensor([local_max], dtype=torch.int64)
        dist.all_reduce(pad_to, op=dist.ReduceOp.MAX, group=group)
        ctx.pad_to = pad_to = int(pad_to.item())
        recv = _EPAllToAllPadded._pad_and_exchange(x, send_counts, pad_to, group)
        return _EPAllToAllPadded._unpad(recv, recv_counts, pad_to)

    @staticmethod
    def backward(ctx, grad_output):
        # backward = reversed a2a: pad by recv_counts -> a2a_single -> unpad by send_counts
        recv = _EPAllToAllPadded._pad_and_exchange(
            grad_output.contiguous(), ctx.recv_counts, ctx.pad_to, ctx.group)
        grad = _EPAllToAllPadded._unpad(recv, ctx.send_counts, ctx.pad_to)
        return grad, None, None, None


def _ep_all_to_all(x, send_counts, recv_counts, group):
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
    """qwen3moe/mixtral adapter: gate is a TopKRouter module (after the HF 2025
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
# The planner picks an adapter by arch name (unregistered archs fall back to
# "default"). The arch name comes from _get_architecture (config.architectures
# lowercased with suffix stripped, e.g. "qwen3moe"); underscored aliases cover
# the config.model_type path (e.g. "qwen3_moe").
MOE_ROUTER_ADAPTERS = {
    "default": _softmax_topk_router,
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

def _swiglu_weights(experts):
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


# ────────────────────────────────────────────────────────────────────────────
# TP-extend-EP forward (D-10, 05 §6.4.8, isomorphic to Megatron
# MoEAlltoAllTokenDispatcher + expert_tensor_parallel_size=1)
# ────────────────────────────────────────────────────────────────────────────

def _hf_native_ep_compute(module, hidden_states, *, router_fn,
                          ep_group, tp_group=None):
    """TP-extend-EP forward: SP-in (local chunk) -> all communication inside
    the region -> SP-out.

    Communication flow (isomorphic to Megatron token_dispatcher.py
    MoEAlltoAllTokenDispatcher):
    router (local chunk, no communication) -> a2a dispatch (extended EP
    group, including TP ranks) -> local SwiGLU (complete expert weights, no
    internal communication, no Partial) -> a2a combine (returns over the
    same group) -> weighted aggregation.

    Input hidden [B, S/tp, H] (local sequence chunk, boundary identity);
    output [B, S/tp, H] (complete, boundary identity).

    Extended EP group = the ep axis of the derived expert mesh (flatten
    ep_size consecutive ranks: first span the TP group, then extend to
    adjacent dp/cp ranks; MindSpeed TP-extend-EP / Megatron etp=1 + ep
    homogeneous across TP). Expert weights are only Shard(0) along the
    expert dim -- each rank holds num_experts/ep_size complete experts, so
    there is no all_gather/reduce_scatter pair.
    """
    w_gate, w_up, w_down = _swiglu_weights(module.experts)
    ep_size = ep_group.size()
    ep_rank = dist.get_rank(group=ep_group)
    e_local = w_gate.shape[0]
    e_global = e_local * ep_size
    expert_offset = ep_rank * e_local

    b, s, h = hidden_states.shape
    x = hidden_states.reshape(-1, h)
    t = x.shape[0]

    # 1. router (local chunk): each rank's chunk differs -> no token duplication
    topk_idx, topk_w = router_fn(module, hidden_states)      # [T, K]
    k = topk_idx.shape[1]
    flat_idx = topk_idx.reshape(-1)
    flat_w = topk_w.reshape(-1).to(x.dtype)
    token_of = torch.arange(t, device=x.device).repeat_interleave(k)
    dest = torch.div(flat_idx, e_local, rounding_mode="floor")

    # 2. sort by (dest, expert) + counts exchange (extended EP group)
    perm = (dest * e_global + flat_idx).argsort()
    send_x = x[token_of[perm]].contiguous()
    send_eid = flat_idx[perm].unsqueeze(-1).contiguous()
    send_counts_t = torch.bincount(dest, minlength=ep_size)
    recv_counts_t = torch.empty_like(send_counts_t)
    dist.all_to_all_single(recv_counts_t, send_counts_t, group=ep_group)
    send_counts = send_counts_t.tolist()
    recv_counts = recv_counts_t.tolist()

    # 3. a2a dispatch over the extended EP group (tokens carry the full H,
    #    sent to the rank holding the expert)
    recv_x = _ep_all_to_all(send_x, send_counts, recv_counts, ep_group)
    recv_eid = _ep_all_to_all(
        send_eid, send_counts, recv_counts, ep_group).squeeze(-1)

    # 4. local SwiGLU (complete expert weights) -> output complete (no Partial,
    #    no reduction point)
    order = recv_eid.argsort()
    sorted_x = recv_x[order]
    local_counts = torch.bincount(recv_eid - expert_offset, minlength=e_local)
    ys, pos = [], 0
    for i in range(e_local):
        n = int(local_counts[i])
        xi = sorted_x[pos:pos + n]
        if w_up is None:
            # fused layout (D-11): chunk gate/up out of gate_up_proj [E, 2I, H]
            gate, up = F.linear(xi, w_gate[i]).chunk(2, dim=-1)
            ys.append(F.linear(F.silu(gate) * up, w_down[i]))
        else:
            ys.append(F.linear(F.silu(F.linear(xi, w_gate[i]))
                               * F.linear(xi, w_up[i]), w_down[i]))
        pos += n
    y_sorted = torch.cat(ys) if ys else recv_x.new_zeros((0, h))
    y = torch.empty_like(y_sorted)
    y[order] = y_sorted

    # 5. a2a combine over the extended EP group -> inverse perm -> weighted
    #    aggregation by topk_w
    ret_x = _ep_all_to_all(y.contiguous(), recv_counts, send_counts, ep_group)
    out_flat = torch.zeros_like(ret_x)
    out_flat[perm] = ret_x
    out = (out_flat * flat_w.unsqueeze(-1)).view(t, k, h).sum(1).view(b, s, h)

    # 6. shared_experts (if present): chunk x TP-sharded weights -> Partial ->
    #    TP-group reduction
    shared = getattr(module, "shared_experts", None)
    if shared is not None:
        if tp_group is None:
            raise RuntimeError(
                "shared_experts on the EP path requires tp_group "
                "(Partial reduction of the TP-sharded output)")
        shared_out = shared(hidden_states)
        dist.all_reduce(shared_out, group=tp_group)
        out = out + shared_out
    return out
