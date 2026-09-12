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
"""FLOPs-per-token estimation from model-configuration geometry.

Backend-free, pure-Python estimator used by the trainer's throughput / MFU
metrics. It reads HF-style config attributes (a config object or a plain
mapping both work) and applies the standard 6N convention:

- every matmul parameter contributes ``6 * params`` FLOPs per token
  (forward + backward);
- MoE models count only the *activated* experts (top-k routed + shared),
  plus the router gate projection;
- the LM-head logits projection (``hidden x vocab``) is included;
- when ``seq_len`` is known, the attention score/weight matmuls add
  ``6 * layers * heads * (qk_head_dim + v_head_dim) * seq_len`` per token.
  Following the torchtitan convention, causal-attention sparsity is NOT
  accounted for;
- activation-checkpoint recompute is never counted as useful FLOPs.

Multi-head lat attention (MLA, DeepSeek-V2/V3 and Kimi families) and GQA/MHA
projection layouts are both recognized; MoE fields from the DeepSeek
(``n_routed_experts`` / ``n_shared_experts`` / ``first_k_dense_replace``) and
Qwen-MoE (``num_experts`` / ``num_experts_per_tok`` /
``shared_expert_intermediate_size`` / ``decoder_sparse_step``) conventions are
both accepted. The estimate is architecture-agnostic and approximate by
design — model families that need an exact value may expose
``model.hp_flops_per_token``, which :func:`resolve_flops_per_token` prefers.
"""

from collections.abc import Mapping
from typing import Any, Optional


_LAYER_FIELDS = ("num_hidden_layers", "n_layers", "num_layers")
_HIDDEN_FIELDS = ("hidden_size", "dim", "d_model", "n_embd")
_HEAD_FIELDS = ("num_attention_heads", "n_heads", "n_head")
_KV_HEAD_FIELDS = ("num_key_value_heads", "n_kv_heads")
_ROUTED_EXPERT_FIELDS = ("n_routed_experts", "num_experts")
_TOPK_FIELDS = ("num_experts_per_tok", "moe_topk", "moe_router_topk", "top_k", "topk")


def _read(config: Any, *names: str) -> Optional[Any]:
    """Return the first present, non-None field from a config object or mapping."""
    for name in names:
        if isinstance(config, Mapping):
            value = config.get(name)
        else:
            value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _read_float(config: Any, *names: str) -> Optional[float]:
    """Return the first present field coerced to ``float``, else ``None``."""
    value = _read(config, *names)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attention_geometry(config: Any) -> Optional[tuple[int, int, int, float]]:
    """Return ``(num_heads, qk_head_dim, v_head_dim, per-layer projection params)``."""
    hidden = _read_float(config, *_HIDDEN_FIELDS)
    heads = _read_float(config, *_HEAD_FIELDS)
    if not hidden or not heads:
        return None
    heads = int(heads)
    kv_lora_rank = _read_float(config, "kv_lora_rank")
    if kv_lora_rank:
        qk_nope_dim = _read_float(config, "qk_nope_head_dim") or 0.0
        qk_rope_dim = _read_float(config, "qk_rope_head_dim") or 0.0
        qk_dim = qk_nope_dim + qk_rope_dim
        v_dim = _read_float(config, "v_head_dim") or qk_dim
        q_lora_rank = _read_float(config, "q_lora_rank")
        if q_lora_rank:
            q_params = hidden * q_lora_rank + q_lora_rank * heads * qk_dim
        else:
            q_params = hidden * heads * qk_dim
        kv_a_params = hidden * (kv_lora_rank + qk_rope_dim)
        kv_b_params = kv_lora_rank * heads * (qk_nope_dim + v_dim)
        o_params = heads * v_dim * hidden
        params = q_params + kv_a_params + kv_b_params + o_params
    else:
        head_dim = _read_float(config, "head_dim") or hidden / heads
        kv_heads = _read_float(config, *_KV_HEAD_FIELDS) or heads
        qk_dim = v_dim = head_dim
        q_params = hidden * heads * head_dim
        kv_params = 2 * hidden * kv_heads * head_dim
        o_params = heads * head_dim * hidden
        params = q_params + kv_params + o_params
    return heads, int(qk_dim), int(v_dim), params


def _mlp_active_params(config: Any, layers: int, hidden: float) -> Optional[float]:
    """Return per-token MLP parameters summed over all layers (MoE: activated only)."""
    intermediate = _read_float(config, "intermediate_size")
    routed_experts = _read_float(config, *_ROUTED_EXPERT_FIELDS)
    if not routed_experts:
        if not intermediate:
            return None
        return layers * 3 * hidden * intermediate
    topk = _read_float(config, *_TOPK_FIELDS)
    moe_ffn = _read_float(config, "moe_intermediate_size") or intermediate
    if not topk or not moe_ffn:
        return None
    dense_first = min(int(_read_float(config, "first_k_dense_replace") or 0), layers)
    sparse_step = max(int(_read_float(config, "decoder_sparse_step") or 1), 1)
    span = layers - dense_first
    moe_layers = span if sparse_step == 1 else (span + sparse_step - 1) // sparse_step
    dense_layers = layers - moe_layers
    per_moe_layer = topk * 3 * hidden * moe_ffn + hidden * routed_experts
    shared_experts = _read_float(config, "n_shared_experts")
    if shared_experts:
        per_moe_layer += shared_experts * 3 * hidden * moe_ffn
    else:
        shared_ffn = _read_float(config, "shared_expert_intermediate_size")
        if shared_ffn:
            per_moe_layer += 3 * hidden * shared_ffn
    total = moe_layers * per_moe_layer
    if intermediate:
        total += dense_layers * 3 * hidden * intermediate
    return total


def estimate_flops_per_token(config: Any, seq_len: Optional[int] = None) -> Optional[float]:
    """Estimate training FLOPs per token (6N convention) from config geometry.

    Args:
        config: HF-style model config; attribute access or plain mapping.
        seq_len: Sequence length used to add the attention score/weight
            quadratic term; ``None`` reports the linear 6N part only.

    Returns:
        Estimated FLOPs per token, or ``None`` when the config lacks the
        geometry fields needed for an honest estimate.
    """
    layers = _read_float(config, *_LAYER_FIELDS)
    hidden = _read_float(config, *_HIDDEN_FIELDS)
    if not layers or not hidden:
        return None
    layers = int(layers)
    attention = _attention_geometry(config)
    mlp_params = _mlp_active_params(config, layers, hidden)
    if attention is None or mlp_params is None:
        return None
    heads, qk_dim, v_dim, attn_params = attention
    active_params = layers * attn_params + mlp_params
    vocab = _read_float(config, "vocab_size")
    if vocab:
        active_params += hidden * vocab
    flops = 6.0 * active_params
    if seq_len:
        flops += 6.0 * layers * heads * (qk_dim + v_dim) * int(seq_len)
    return flops


def batch_seq_len(micro_batches: Any) -> Optional[int]:
    """Return the sequence length of the first micro-batch carrying ``input_ids``.

    Args:
        micro_batches: One micro-batch or a list of mapping micro-batches.

    Returns:
        The trailing ``input_ids`` dimension, or ``None`` when unavailable.
    """
    if micro_batches is None:
        return None
    if isinstance(micro_batches, Mapping):
        micro_batches = [micro_batches]
    for batch in micro_batches:
        if not isinstance(batch, Mapping):
            continue
        shape = getattr(batch.get("input_ids"), "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[-1])
    return None


def resolve_flops_per_token(
    model: Any,
    model_config: Any = None,
    seq_len: Optional[int] = None,
) -> Optional[float]:
    """Resolve FLOPs per token: the model's own property wins, else estimate.

    Args:
        model: Built model instance; a family may expose
            ``hp_flops_per_token`` with an exact derivation.
        model_config: HF-style config used by the generic estimator;
            defaults to ``model.config``.
        seq_len: Sequence length for the attention quadratic term.

    Returns:
        FLOPs per token, or ``None`` when neither source can provide one.
    """
    value = getattr(model, "hp_flops_per_token", None) if model is not None else None
    if value:
        return float(value)
    if model_config is None and model is not None:
        model_config = getattr(model, "config", None)
    if model_config is None:
        return None
    return estimate_flops_per_token(model_config, seq_len)
