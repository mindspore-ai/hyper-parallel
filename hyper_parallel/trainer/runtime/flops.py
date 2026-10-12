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
"""Architecture-based LLM training FLOPs, following Megatron and MindFormers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _get(config: Any, *names: str, default: Any = None) -> Any:
    """Read common HF/Megatron geometry names without consulting model names."""
    for name in names:
        value = config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)
        if value is not None:
            return value
    return default


def _dimension(value: Any, name: str, *, minimum: int = 1) -> int:
    """Validate static model geometry before entering the training loop."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class TransformerFlops:
    """Forward FLOPs coefficients for token-linear and attention-quadratic work.

    Full training uses factor 3 for forward, input and weight gradients.
    These model estimates exclude optimizer, communication, recomputation
    and kernel padding; they do not measure hardware utilization.
    """

    per_token: float
    per_attention_pair: float

    def estimate(self, tokens: Any, squared_lengths: Any) -> Any:
        """Evaluate using executed tokens and sum of squared attention lengths.

        Args:
            tokens: Count of tokens processed by projections/MLPs, not loss-mask count.
            squared_lengths: Sum of L_i**2 over independent attention sequences.

        Returns:
            Numeric or device scalar FLOPs; tensor inputs remain on device.
        """
        return 3 * (self.per_token * tokens + self.per_attention_pair * squared_lengths)


def _moe_pattern(config: Any, layers: int) -> list[bool]:
    """Resolve executed dense/expert layers from structural configuration."""
    types = _get(config, "mlp_layer_types")
    if types is not None:
        if len(types) != layers or any(kind not in ("dense", "sparse", "moe") for kind in types):
            raise ValueError("mlp_layer_types must contain one dense/sparse/moe entry per layer")
        return [kind != "dense" for kind in types]
    experts = _get(config, "n_routed_experts", "num_local_experts", "num_experts", "num_moe_experts", default=0)
    if not experts:
        return [False] * layers
    _dimension(experts, "num_experts")
    frequency = _get(config, "moe_layer_freq")
    if isinstance(frequency, (list, tuple)):
        if len(frequency) != layers or any(value not in (0, 1) for value in frequency):
            raise ValueError("moe_layer_freq must contain one 0/1 entry per layer")
        return [bool(value) for value in frequency]
    sparse_step = _get(config, "decoder_sparse_step")
    frequency = _dimension(frequency if frequency is not None else sparse_step or 1, "moe_layer_freq")
    first_dense = _dimension(_get(config, "first_k_dense_replace", default=0), "first_k_dense_replace", minimum=0)
    dense_only = _get(config, "mlp_only_layers", default=())
    offset = 1 if sparse_step is not None else 0
    return [index >= first_dense and index not in dense_only and (index + offset) % frequency == 0
            for index in range(layers)]


def _attention_terms(config: Any, hidden: int, heads: int) -> tuple[float, float]:
    """Return projection and full-attention multiply-add coefficients."""
    kv_rank = _get(config, "kv_lora_rank")
    if kv_rank is not None or _get(config, "multi_latent_attention", default=False):
        kv_rank = _dimension(kv_rank, "kv_lora_rank")
        qk_dim = _dimension(_get(config, "qk_nope_head_dim", "qk_head_dim"), "qk_head_dim")
        rope = _dimension(_get(config, "qk_rope_head_dim", "qk_pos_emb_head_dim"), "rope_dim", minimum=0)
        value_dim = _dimension(_get(config, "v_head_dim"), "v_head_dim")
        q_rank = _get(config, "q_lora_rank")
        query = hidden * heads * (qk_dim + rope) if q_rank is None else (
            _dimension(q_rank, "q_lora_rank") * (hidden + heads * (qk_dim + rope) + 1)
        )
        projection = query + kv_rank * (hidden + heads * (qk_dim + value_dim) + 1)
        projection += hidden * rope + heads * value_dim * hidden
        return projection, heads * (qk_dim + rope + value_dim)
    head_dim = _get(config, "head_dim", "kv_channels")
    if head_dim is None and hidden % heads:
        raise ValueError("num_attention_heads must divide hidden_size when head_dim is absent")
    head_dim = _dimension(head_dim if head_dim is not None else hidden // heads, "head_dim")
    groups = _dimension(_get(config, "num_key_value_heads", "num_query_groups", default=heads), "num_query_groups")
    if _get(config, "group_query_attention") is False:
        groups = heads
    if heads % groups:
        raise ValueError("num_query_groups must divide num_attention_heads")
    query = heads * head_dim
    gate = query if _get(config, "attention_output_gate", default=False) else 0
    return hidden * (2 * query + 2 * groups * head_dim + gate), 2 * query


def _mlp_term(config: Any, hidden: int, intermediate: int, pattern: list[bool]) -> float:
    """Count active experts, never the total stored expert parameter count."""
    activation = _get(config, "hidden_act", "activation_function", default="gelu")
    gated = _get(config, "swiglu", default=activation in ("silu", "swiglu", "fusedswiglu", "geglu"))
    expansion = 3 if gated else 2
    moe_layers = sum(pattern)
    term = expansion * hidden * intermediate * (len(pattern) - moe_layers)
    if not moe_layers:
        return term
    moe_width = _dimension(_get(config, "moe_intermediate_size", "moe_ffn_hidden_size", default=intermediate),
                           "moe_ffn_hidden_size")
    topk = _dimension(_get(config, "num_experts_per_tok", "moe_router_topk"), "moe_router_topk")
    experts = _get(config, "n_routed_experts", "num_local_experts", "num_experts", "num_moe_experts")
    if experts is not None and topk > _dimension(experts, "num_experts"):
        raise ValueError("moe_router_topk cannot exceed num_experts")
    shared = _get(config, "shared_expert_intermediate_size", "moe_shared_expert_intermediate_size")
    if shared is None:
        shared = (_get(config, "n_shared_experts", default=0) or 0) * moe_width
    shared = _dimension(shared, "shared_expert_intermediate_size", minimum=0)
    configured_latent = _get(config, "moe_latent_size")
    latent = _dimension(hidden if configured_latent is None else configured_latent, "moe_latent_size")
    routed = expansion * latent * moe_width * topk
    if configured_latent is not None:
        routed += 2 * hidden * latent
    return term + moe_layers * (routed + expansion * hidden * shared)


def transformer_flops(config: Any, *,
                      mtp_layer_types: tuple[bool, ...] = ()) -> TransformerFlops | None:
    """Build reusable dense/MoE MHA/GQA/MLA FLOPs coefficients from architecture.

    Args:
        config: Flat HF/Megatron-compatible causal decoder geometry, not a model name.
        mtp_layer_types: Executed MTP decoder MLPs, True for MoE and False for dense.
            Checkpoint MTP configuration alone does not imply execution.

    Returns:
        Coefficients, or None for missing/unsupported structure. Composite,
        sparse/linear attention and cross-attention are not estimated.

    Raises:
        ValueError: If a supported architecture has invalid dimensions.
    """
    required = [_get(config, *names) for names in (
        ("hidden_size",), ("num_hidden_layers", "num_layers", "depth"),
        ("num_attention_heads", "num_heads"), ("intermediate_size", "ffn_hidden_size"),
    )]
    if any(value is None for value in required):
        return None
    unsupported = ("text_config", "vision_config", "audio_config", "hybrid_layer_pattern",
                   "experimental_attention_variant", "index_topk", "compress_ratios", "sliding_window",
                   "add_cross_attention", "is_encoder_decoder")
    if any(_get(config, name) for name in unsupported):
        return None
    layer_types = _get(config, "layer_types")
    if layer_types and any(kind not in ("full_attention", "attention") for kind in layer_types):
        return None
    hidden, layers, heads, intermediate = (
        _dimension(value, name) for value, name in zip(required, ("hidden_size", "num_layers", "num_heads", "ffn_size"))
    )
    pattern = _moe_pattern(config, layers) + list(mtp_layer_types)
    projection, attention = _attention_terms(config, hidden, heads)
    depth = len(mtp_layer_types)
    linear = projection * (layers + depth) + _mlp_term(config, hidden, intermediate, pattern)
    linear += depth * (3 * hidden + 2 * hidden * hidden)
    vocab = _get(config, "padded_vocab_size", "vocab_size")
    if vocab is None:
        return None
    linear += hidden * _dimension(vocab, "vocab_size") * (depth + 1)
    return TransformerFlops(2 * linear, attention * (layers + depth))


class TransformerFlopsEstimator:
    """Default full-training estimator for a structurally supported text decoder.

    Geometry determines the estimate without a model-name whitelist.
    Unknown structures and partially frozen models omit TFLOP/s rather
    than publishing a dense estimate for a different workload.
    """

    def __init__(self, model_config: Any, model: Any = None) -> None:
        """Resolve constant geometry once; keep no model or activation references."""
        mtp_types = []
        for layer in getattr(getattr(model, "mtp", None), "layers", ()):
            mlp = getattr(getattr(layer, "transformer_layer", None), "mlp", None)
            if mlp is None:
                self.coefficients = None
                return
            mtp_types.append(hasattr(mlp, "experts"))
        self.coefficients = transformer_flops(model_config, mtp_layer_types=tuple(mtp_types))
        if callable(getattr(model, "parameters", None)) and any(
                not parameter.requires_grad for parameter in model.parameters()):
            self.coefficients = None

    def __call__(self, batch: Mapping[str, Any], *, cp_size: int = 1) -> Any:
        """Return full-CP micro-batch FLOPs, or None when work is not fully known.

        Forward ``actual_seq_len`` (batch-one cumulative ends) or dense-mask
        ``cu_seq_lens`` determines sum(L_i**2); tensor metadata
        stays on device. Projection work still counts all processed input tokens.
        Packed boundaries must cover the executed batch, as required by batching.
        """
        shape = getattr(batch.get("input_ids"), "shape", ())
        if self.coefficients is None or len(shape) != 2:
            return None
        batch_size, local_length = shape
        length = local_length * cp_size
        squared_lengths = batch_size * length * length
        boundaries = batch.get("cu_seq_lens")
        sequence_ends = batch.get("actual_seq_len")
        if sequence_ends is not None:
            if batch_size != 1 or not isinstance(sequence_ends, (tuple, list)) or not sequence_ends:
                return None
            if any(not isinstance(end, int) for end in sequence_ends) or sequence_ends[-1] != length:
                return None
            lengths = [right - left for left, right in zip((0, *sequence_ends[:-1]), sequence_ends)]
            if any(size <= 0 for size in lengths):
                return None
            squared_lengths = sum(size * size for size in lengths)
        elif boundaries is not None:
            if getattr(boundaries, "ndim", None) != 1:
                return None
            lengths = boundaries[1:] - boundaries[:-1]
            squared_lengths = lengths.long().square().sum()
        return self.coefficients.estimate(batch_size * length, squared_lengths)
