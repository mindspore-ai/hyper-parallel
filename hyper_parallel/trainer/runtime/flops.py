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

from typing import Any, Optional, Tuple

# Qwen3-MoE always projects its dense MLP and every routed expert through three
# GEMMs (gate, up, down). ``hidden_act`` only selects the elementwise activation
# applied to the gate branch, so it must not change the projection count.
_GATED_FFN_PROJECTIONS = 3.0


def _qwen3_moe_coefficients(config: Any) -> Optional[Tuple[float, float]]:
    """Split per-token FLOPs into a length-free part and a per-position part.

    Causal attention charges ``query * seq_length`` per layer, so a sequence of
    length ``S`` costs ``tokens * base + S * S * attention`` FLOPs. Keeping the
    two coefficients apart lets callers accumulate ``sum(S_i ** 2)`` over the
    real attention boundaries instead of assuming one representative length.

    Args:
        config: Hugging Face model configuration.

    Returns:
        ``(base, attention)`` FLOPs per token and per token-position, or
        ``None`` when the model family is not supported.
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

    # Per token, per standard-attention layer: QKV and output projections.
    attention_base = 6.0 * (hidden * (query + key_value + key_value) + query * hidden)
    # Causal attention only attends the lower triangle, i.e. ``query * S``.
    attention_position = 6.0 * query

    projections = _GATED_FFN_PROJECTIONS
    dense_mlp = 6.0 * hidden * intermediate * projections
    moe_mlp = 6.0 * hidden * moe_intermediate * topk * projections

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
    base = (
        attention_base * layers
        + dense_mlp * num_dense
        + moe_mlp * num_moe
        + logits
    )
    return base, attention_position * layers


def estimate_flops_per_token(config: Any, seq_length: float) -> Optional[float]:
    """Estimate training FLOPs per token for a decoder-only model.

    The estimate counts forward, weight-gradient and data-gradient GEMMs
    (3x) and two FLOPs per multiply-accumulate (2x), so every GEMM carries a
    factor of six. Causal attention counts only the lower triangle via
    ``q * seq_length``.

    Args:
        config: Hugging Face model configuration.
        seq_length: Representative tokens per sequence. Use this helper only
            when every sequence in the step has that length; otherwise prefer
            :func:`estimate_step_flops`, which accumulates the attention term
            over the real sequence boundaries.

    Returns:
        Estimated FLOPs per token, or ``None`` when the model family is not
        supported. Communication, recomputation, normalization, dropout,
        RoPE and optimizer work are not included.
    """
    coefficients = _qwen3_moe_coefficients(config)
    if coefficients is None:
        return None
    base, attention_position = coefficients
    return base + attention_position * float(seq_length)


def estimate_step_flops(
        config: Any,
        input_tokens: float,
        sum_seq_len_sq: float,
) -> Optional[float]:
    """Estimate training FLOPs for one optimizer step.

    Attention FLOPs scale with the square of each sequence length, so a step is
    described by the model-input token count ``sum(S_i)`` and the length squares
    ``sum(S_i ** 2)``. For packed batches, ``S_i`` are the packed segment
    lengths; for plain batches they are the per-row token counts.

    Args:
        config: Hugging Face model configuration.
        input_tokens: Model-input tokens consumed by the step, i.e. tokens that
            reach attention and the FFN, including prompt tokens masked out of
            the loss.
        sum_seq_len_sq: Sum of the squared length of every sequence that
            reaches attention in the step.

    Returns:
        Estimated step FLOPs, or ``None`` when the model family is not
        supported. Communication, recomputation, normalization, dropout,
        RoPE and optimizer work are not included.
    """
    coefficients = _qwen3_moe_coefficients(config)
    if coefficients is None:
        return None
    base, attention_position = coefficients
    return base * float(input_tokens) + attention_position * float(sum_seq_len_sq)
