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

"""expert_parallel.routing: MoE router adapters (05 §6.4.7, D-09).

Routing semantics are model-specific (softmax/sigmoid, top-k normalization,
scaling), while the expert MLP structure is uniform (SwiGLU). A factory picks
an adapter from ``MOE_ROUTER_ADAPTERS`` BY NAME in its own code — the choice
is explicit, never inferred. Each adapter maps
``(module, hidden) -> (topk_idx [T,K] int64, topk_w [T,K] float)``.

Split out of components/distributed/ep_utils.py in stage 4e.
"""

import torch


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
