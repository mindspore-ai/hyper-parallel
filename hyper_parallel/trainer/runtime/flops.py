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
"""Composable architecture-based model FLOPs, following Megatron and MindFormers."""

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

    Components can be summed to describe text, vision, audio and projection
    branches. A frozen encoder run under no_grad uses factor 1, a fully trained
    component uses 3 (forward, input gradient, weight gradient). A frozen weight
    with a required input gradient uses 2 for parameter GEMMs; core attention
    retains both activation gradients and uses 3. These are model estimates, excluding
    optimizer, communication, recompute and kernel padding; they are not MFU.
    """

    per_token: float
    per_attention_pair: float

    def estimate(self, tokens: Any, squared_lengths: Any, *, forward_backward_factor: int = 3) -> Any:
        """Evaluate using executed tokens and sum of squared attention lengths.

        Args:
            tokens: Count of tokens processed by projections/MLPs, not loss-mask count.
            squared_lengths: Sum of L_i**2 over independent attention sequences.
            forward_backward_factor: 1/2/3 for forward / forward+dgrad / full training.

        Returns:
            Numeric or device scalar FLOPs; tensor inputs remain on device.
        """
        if forward_backward_factor not in (1, 2, 3):
            raise ValueError("forward_backward_factor must be 1, 2 or 3")
        # QK/PV operands are activations: freezing weights does not remove either input gradient.
        attention_factor = 1 if forward_backward_factor == 1 else 3
        return (forward_backward_factor * self.per_token * tokens
                + attention_factor * self.per_attention_pair * squared_lengths)


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


def transformer_flops(config: Any, *, causal: bool = True, include_logits: bool = True,
                      mtp_layer_types: tuple[bool, ...] = ()) -> TransformerFlops | None:
    """Build reusable dense/MoE MHA/GQA/MLA FLOPs coefficients from architecture.

    Args:
        config: Flat HF/Megatron-compatible component geometry, not a model name.
        causal: Whether core attention uses the references' half-square causal estimate.
        include_logits: Include the vocabulary projection for a language decoder.
        mtp_layer_types: Executed MTP decoder MLPs, True for MoE and False for dense.
            Checkpoint MTP configuration alone does not imply execution.

    Returns:
        Coefficients, or None for missing/unsupported structure. Composite,
        sparse/linear attention and cross-attention require a complete estimator.

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
    if include_logits:
        vocab = _get(config, "padded_vocab_size", "vocab_size")
        if vocab is None:
            return None
        linear += hidden * _dimension(vocab, "vocab_size") * (depth + 1)
    return TransformerFlops(2 * linear, (1 if causal else 2) * attention * (layers + depth))


class TransformerFlopsEstimator:
    """Default full-training estimator for a structurally supported text decoder.

    A custom ``TrainerConfig.flops_estimator`` can compose ``transformer_flops``
    coefficients with vision/audio/projector work, actual modality lengths and
    freeze factors. The Trainer never selects a model family or discards unknown
    components to report a misleading full-model TFLOP/s value.
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

        Global packed ``cu_seq_lens`` determines sum(L_i**2); tensor metadata
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
        if boundaries is not None:
            if getattr(boundaries, "ndim", None) != 1:
                return None
            lengths = boundaries[1:] - boundaries[:-1]
            squared_lengths = lengths.long().square().sum()
        return self.coefficients.estimate(batch_size * length, squared_lengths)


@dataclass(frozen=True)
class FlopsComponent:
    """One declared executed branch in a multimodal FLOPs estimate.

    ``input_key`` refers to [batch, sequence, ...] processed tokens, not raw
    images/waveforms. ``lengths_key`` can instead supply full, unpadded sequence
    lengths. ``config_path`` selects component geometry (e.g. text_config) from
    the model config. A linear/MLP projector uses ``linear_dimensions`` instead.
    Exactly one geometry source must be specified. Encoder patch embedding is
    a separate linear component; include every executed branch explicitly.
    """

    input_key: str
    config_path: str | None = None
    linear_dimensions: tuple[int, ...] = ()
    lengths_key: str | None = None
    causal: bool = False
    include_logits: bool = False
    forward_backward_factor: int = 3
    cp_partitioned: bool = False
    optional_input: bool = False

    def coefficients(self, model_config: Any) -> TransformerFlops:
        """Resolve one complete component; reject unsupported declared geometry."""
        if bool(self.config_path) == bool(self.linear_dimensions):
            raise ValueError("Each FLOPs component needs exactly one of config_path or linear_dimensions")
        if not self.input_key or self.forward_backward_factor not in (1, 2, 3):
            raise ValueError("FLOPs component requires input_key and forward_backward_factor in (1, 2, 3)")
        if self.linear_dimensions:
            if len(self.linear_dimensions) < 2:
                raise ValueError("linear_dimensions must include at least input and output sizes")
            dimensions = [_dimension(size, "linear_dimensions") for size in self.linear_dimensions]
            return TransformerFlops(2 * sum(left * right for left, right in zip(dimensions[:-1], dimensions[1:])), 0)
        config = model_config
        for name in self.config_path.split("."):
            config = _get(config, name)
        coefficients = transformer_flops(config, causal=self.causal, include_logits=self.include_logits)
        if coefficients is None:
            raise ValueError(f"Unsupported FLOPs component geometry at {self.config_path!r}")
        return coefficients

    def workload(self, batch: Mapping[str, Any], cp_size: int) -> tuple[Any, Any] | None:
        """Return full-CP token and quadratic counts using the declared batch layout."""
        if self.lengths_key is not None:
            lengths = batch.get(self.lengths_key)
            if lengths is None:
                return (0, 0) if self.optional_input else None
            if getattr(lengths, "ndim", None) != 1:
                raise ValueError("FLOPs component sequence lengths must be a one-dimensional tensor")
            lengths = lengths.long()
            return lengths.sum(), lengths.square().sum()
        value = batch.get(self.input_key)
        if value is None:
            return (0, 0) if self.optional_input else None
        shape = getattr(value, "shape", ())
        if len(shape) not in (2, 3):
            raise ValueError("FLOPs component input must contain processed [batch, sequence, ...] tokens")
        samples, length = shape[:2]
        length *= cp_size if self.cp_partitioned else 1
        return samples * length, samples * length * length


class CompositeFlopsEstimator:
    """Compose full model FLOPs for dense/MoE text, vision/audio and projectors.

    Components describe execution and freeze state, not model brand names. The
    returned count covers a full logical CP micro-batch and is identical across
    its TP/CP replicas. The meter de-duplicates these replicas. New architectures
    can provide a different callable through the same configuration interface.
    """

    def __init__(self, model_config: Any, components: list[FlopsComponent]) -> None:
        """Validate and cache component geometry once, outside the training loop."""
        if not components:
            raise ValueError("CompositeFlopsEstimator requires all executed components")
        self.components = [(component, component.coefficients(model_config)) for component in components]

    def __call__(self, batch: Mapping[str, Any], *, cp_size: int = 1) -> Any:
        """Sum complete components, or omit the total if any required input is absent."""
        total = 0
        for component, coefficients in self.components:
            workload = component.workload(batch, cp_size)
            if workload is None:
                return None
            total = total + coefficients.estimate(*workload, forward_backward_factor=component.forward_backward_factor)
        return total
