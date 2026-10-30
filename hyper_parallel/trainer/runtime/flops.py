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
"""Analytical model-FLOPs estimation for trainer throughput metrics."""

from __future__ import annotations

from typing import Any, Optional


# Gated FFNs use three GEMMs (gate, up, down); a plain FFN uses two.
_GATED_ACTS = frozenset({"silu", "swiglu", "fusedswiglu"})


def estimate_flops_per_token(config: Any, seq_length: float) -> Optional[float]:
    """Estimate training FLOPs per token for a decoder-only model.

    The estimate counts forward, weight-gradient and data-gradient GEMMs
    (3x) and two FLOPs per multiply-accumulate (2x), so every GEMM carries a
    factor of six. Causal attention counts only the lower triangle via
    ``q * seq_length``.

    Args:
        config: Hugging Face model configuration.
        seq_length: Representative tokens per sequence. For packed or
            variable-length batches, pass ``global_tokens / global_samples``.

    Returns:
        Estimated FLOPs per token, or ``None`` when the model family is not
        supported. Communication, recomputation, normalization, dropout,
        RoPE and optimizer work are not included.
    """
    if getattr(config, "model_type", None) != "qwen3_moe":
        return None

    hidden = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)
    head_dim = int(getattr(config, "head_dim", None) or hidden // heads)
    intermediate = int(config.intermediate_size)
    moe_intermediate = int(config.moe_intermediate_size)
    topk = int(config.num_experts_per_tok)
    vocab = int(config.vocab_size)

    query = heads * head_dim
    key_value = kv_heads * head_dim

    # Per token, per standard-attention layer.
    attention = 6.0 * (
        hidden * (query + key_value + key_value)
        + query * float(seq_length)
        + query * hidden
    )

    sigma = 3.0 if getattr(config, "hidden_act", "") in _GATED_ACTS else 2.0
    dense_mlp = 6.0 * hidden * intermediate * sigma
    moe_mlp = 6.0 * hidden * moe_intermediate * topk * sigma

    # Match Qwen3MoeDecoderLayer: MoE when the layer is not in
    # ``mlp_only_layers`` and ``(layer_idx + 1) % decoder_sparse_step == 0``.
    mlp_only = set(getattr(config, "mlp_only_layers", None) or ())
    sparse_step = max(1, int(getattr(config, "decoder_sparse_step", 1) or 1))
    num_moe = sum(
        1
        for layer_idx in range(layers)
        if layer_idx not in mlp_only
        and int(getattr(config, "num_experts", 0) or 0) > 0
        and (layer_idx + 1) % sparse_step == 0
    )
    num_dense = layers - num_moe

    logits = 6.0 * hidden * vocab
    return (
        attention * layers
        + dense_mlp * num_dense
        + moe_mlp * num_moe
        + logits
    )
