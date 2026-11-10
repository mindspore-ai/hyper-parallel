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
"""PaLM model-FLOPs accounting for training throughput metrics.

The definition follows PaLM (Chowdhery et al., 2022, Appendix B). Per trained
token the model performs ``6 * N`` FLOPs in the matrix multiplications of the
``N`` parameters that token touches (2 forward, 4 backward) plus
``12 * L * H * Q * T`` FLOPs in self-attention: ``QK^T`` and ``PV`` over the
full ``T x T`` score matrix, without a causal-mask discount. Embedding lookups,
element-wise work, activation recomputation and optimizer updates are not model
FLOPs; their time still counts in the step time that MFU divides by.

Two extensions follow torchtitan's model-FLOPs helpers: gated delta-rule
linear-attention layers add their state recurrence, and vision encoders are
left out so a multimodal model reports decoder-only MFU, because vision cost
scales with image patches rather than with the text sequence.
"""

from __future__ import annotations

__all__ = [
    "AttentionLayer",
    "DeltaRuleLayer",
    "MatmulParamCount",
    "ModelFlopsProfile",
    "attention_layers_from_config",
    "count_matmul_params",
]

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Optional, Sequence, Union

import torch  # pylint: disable=forbidden-backend-import

_ROUTED_EXPERT_PATH = re.compile(r"(?:^|\.)experts\.")
_TOP_K_FIELDS = ("num_experts_per_tok", "moe_router_topk", "top_k")
_NUM_EXPERT_FIELDS = ("num_experts", "n_routed_experts", "num_local_experts")
_VISION_MODULE_NAMES = frozenset(
    {"visual", "vision_tower", "vision_model", "vision_encoder", "multi_modal_projector", "mm_projector"}
)
_SPARSE_ATTENTION_FIELDS = ("index_topk", "compress_ratio", "compress_ratios")


def _decoder_config(model_config: Any) -> Any:
    """Return the text-decoder config of a composite config, else the config itself."""
    text_config = getattr(model_config, "text_config", None)
    return model_config if text_config is None else text_config


def _first_positive_int(config: Any, names: Iterable[str]) -> Optional[int]:
    """Return the first positive integer field among ``names``, if any."""
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


@dataclass(frozen=True)
class AttentionLayer:
    """Attention geometry of one decoder layer.

    Attributes:
        num_heads: Query heads. Grouped key/value heads do not reduce the score
            or value products, so only the query head count matters.
        qk_head_dim: Per-head width of the query-key contraction.
        v_head_dim: Per-head width of the value contraction.
        window: Sliding-window length, or ``None`` for full attention.
    """

    num_heads: int
    qk_head_dim: int
    v_head_dim: int
    window: Optional[int] = None

    def forward_flops_per_token(self, seq_len: float) -> float:
        """Return forward ``QK^T`` and ``PV`` FLOPs per token.

        Args:
            seq_len: Training sequence length.

        Returns:
            ``2 * heads * (qk_head_dim + v_head_dim) * attended_tokens``.
        """
        attended_tokens = seq_len if self.window is None else min(seq_len, self.window)
        return 2.0 * self.num_heads * (self.qk_head_dim + self.v_head_dim) * attended_tokens


@dataclass(frozen=True)
class DeltaRuleLayer:
    """Gated delta-rule linear attention of one decoder layer.

    Every value head carries a ``key_head_dim x value_head_dim`` state that each
    token reads, updates and queries: three products of that size. The
    projections and the short convolution are parameters already counted in
    ``N``; decay, gating and normalization are element-wise.

    Attributes:
        num_heads: Value heads, each holding one recurrent state.
        key_head_dim: Per-head key width.
        value_head_dim: Per-head value width.
    """

    num_heads: int
    key_head_dim: int
    value_head_dim: int

    def forward_flops_per_token(self, seq_len: float) -> float:
        """Return forward recurrence FLOPs per token, independent of ``seq_len``.

        Args:
            seq_len: Training sequence length, unused by the recurrence.

        Returns:
            ``2 * 3 * heads * key_head_dim * value_head_dim``.
        """
        del seq_len
        return 6.0 * self.num_heads * self.key_head_dim * self.value_head_dim


SequenceMixer = Union[AttentionLayer, DeltaRuleLayer]


@dataclass(frozen=True)
class MatmulParamCount:
    """Matrix-multiply parameters a token touches, as counted on one rank.

    Attributes:
        matmul_params: Parameters of rank >= 2 other than input embedding
            tables and vision encoders, with routed experts weighted by
            ``top_k / num_experts``.
        output_head_params: The part of ``matmul_params`` in the output head.
        vision_params: Vision-encoder parameters left out of ``matmul_params``.
        unsupported: Reasons the formula would misstate this model.
    """

    matmul_params: Fraction
    output_head_params: Fraction
    vision_params: Fraction = Fraction(0)
    unsupported: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelFlopsProfile:
    """Static inputs of the PaLM model-FLOPs formula for one model.

    Attributes:
        matmul_params: Active matrix-multiply parameters per token, ``N``.
        output_head_params: Part of ``matmul_params`` outside the decoder
            layers, which layer-wise activation recomputation does not replay.
        attention_layers: Sequence mixer of every decoder layer.
    """

    matmul_params: float
    output_head_params: float
    attention_layers: tuple[SequenceMixer, ...]

    def _attention_forward_flops(self, seq_len: float) -> float:
        """Return the forward attention FLOPs per token summed over layers."""
        return sum(layer.forward_flops_per_token(seq_len) for layer in self.attention_layers)

    def flops_per_token(self, seq_len: float) -> float:
        """Return PaLM model FLOPs per trained token, forward and backward.

        Args:
            seq_len: Training sequence length ``T``.

        Returns:
            ``6 * N`` plus ``12 * H * Q * T`` for every softmax-attention layer
            and three times the forward recurrence of every delta-rule layer.
        """
        return 6.0 * self.matmul_params + 3.0 * self._attention_forward_flops(seq_len)

    def recompute_flops_per_token(self, seq_len: float) -> float:
        """Return forward FLOPs replayed per token when every decoder layer is recomputed.

        Args:
            seq_len: Training sequence length ``T``.

        Returns:
            Forward matmul FLOPs of the decoder layers plus their forward attention.
        """
        decoder_params = self.matmul_params - self.output_head_params
        return 2.0 * decoder_params + self._attention_forward_flops(seq_len)


def _output_head_ids(model_parts: Sequence[torch.nn.Module]) -> set[int]:
    """Return ids of output-projection weights, which are matmuls even when tied."""
    head_ids = set()
    for part in model_parts:
        get_output_embeddings = getattr(part, "get_output_embeddings", None)
        head = get_output_embeddings() if callable(get_output_embeddings) else None
        if getattr(head, "weight", None) is not None:
            head_ids.add(id(head.weight))
        for name, module in part.named_modules():
            if name.split(".")[-1] == "lm_head" and getattr(module, "weight", None) is not None:
                head_ids.add(id(module.weight))
    return head_ids


def _embedding_ids(model_parts: Sequence[torch.nn.Module]) -> set[int]:
    """Return ids of embedding-table weights, which are looked up, not multiplied."""
    return {
        id(module.weight)
        for part in model_parts
        for module in part.modules()
        if isinstance(module, torch.nn.Embedding)
    }


def _unique_named_parameters(model_parts: Sequence[torch.nn.Module]) -> Iterable[tuple[str, torch.nn.Parameter]]:
    """Yield each parameter of the model parts once, under its first name."""
    seen = set()
    for part in model_parts:
        for name, param in part.named_parameters():
            if id(param) not in seen:
                seen.add(id(param))
                yield name, param


def _token_share(name: str, expert_ratio: Optional[Fraction]) -> Optional[Fraction]:
    """Return the share of a parameter each token uses, or ``None`` when it is unknown."""
    if _ROUTED_EXPERT_PATH.search(name) and "shared_expert" not in name:
        return expert_ratio
    return Fraction(1)


def count_matmul_params(model_parts: Sequence[torch.nn.Module], model_config: Any) -> MatmulParamCount:
    """Count the matrix-multiply parameters each token touches on this rank.

    ``Parameter.numel()`` of a sharded DTensor is its global size, so FSDP, TP
    and EP shards are counted once per parameter. Pipeline stages hold disjoint
    parameters; the caller sums the stage counts. Parameters under a vision
    encoder or multimodal projector are set aside, so the count describes the
    text decoder.

    Args:
        model_parts: Model chunks held by this rank.
        model_config: Model configuration providing the MoE routing fields.

    Returns:
        Active matmul, output-head and vision-encoder parameter counts with any
        reasons the formula does not cover this model.
    """
    config = _decoder_config(model_config)
    top_k = _first_positive_int(config, _TOP_K_FIELDS)
    num_experts = _first_positive_int(config, _NUM_EXPERT_FIELDS)
    expert_ratio = Fraction(top_k, num_experts) if top_k and num_experts else None
    embedding_ids = _embedding_ids(model_parts)
    head_ids = _output_head_ids(model_parts)

    matmul_params = Fraction(0)
    output_head_params = Fraction(0)
    vision_params = Fraction(0)
    unsupported = set()
    for name, param in _unique_named_parameters(model_parts):
        pieces = name.split(".")
        if any(piece.startswith("mtp") for piece in pieces):
            unsupported.add("multi-token prediction layers are present")
        if _VISION_MODULE_NAMES.intersection(pieces):
            vision_params += param.numel()
            continue
        if param.dim() < 2 or (id(param) in embedding_ids and id(param) not in head_ids):
            continue
        share = _token_share(name, expert_ratio)
        if share is None:
            unsupported.add("routed experts are present but the model config has no top-k / expert count")
            continue
        matmul_params += param.numel() * share
        if id(param) in head_ids:
            output_head_params += param.numel() * share
    if getattr(model_config, "vision_config", None) is not None and vision_params == 0:
        unsupported.add("the model config has a vision encoder whose parameters could not be identified")
    return MatmulParamCount(matmul_params, output_head_params, vision_params, tuple(sorted(unsupported)))


def _legacy_layer_types(config: Any, num_layers: int) -> list[str]:
    """Derive layer kinds for configs that predate ``layer_types``.

    Qwen2-style configs gate sliding windows with ``use_sliding_window`` and
    keep the first ``max_window_layers`` layers on full attention; Mistral-style
    configs without that switch apply a set ``sliding_window`` to every layer.
    """
    window = _first_positive_int(config, ("sliding_window",))
    if hasattr(config, "use_sliding_window"):
        if not (config.use_sliding_window and window):
            return ["full_attention"] * num_layers
        full_layers = getattr(config, "max_window_layers", None) or 0
        return ["full_attention" if index < full_layers else "sliding_attention" for index in range(num_layers)]
    return ["sliding_attention" if window else "full_attention"] * num_layers


def _delta_rule_layer(config: Any) -> Optional[DeltaRuleLayer]:
    """Return the gated delta-rule geometry named by Qwen3-Next style fields."""
    num_heads = _first_positive_int(config, ("linear_num_value_heads",))
    key_head_dim = _first_positive_int(config, ("linear_key_head_dim",))
    value_head_dim = _first_positive_int(config, ("linear_value_head_dim",))
    if num_heads is None or key_head_dim is None or value_head_dim is None:
        return None
    return DeltaRuleLayer(num_heads, key_head_dim, value_head_dim)


def _attention_head_dims(config: Any, num_heads: int) -> Optional[tuple[int, int]]:
    """Return the query-key and value head widths, MLA split widths included."""
    qk_nope = _first_positive_int(config, ("qk_nope_head_dim",))
    qk_rope = _first_positive_int(config, ("qk_rope_head_dim",))
    v_head_dim = _first_positive_int(config, ("v_head_dim",))
    if qk_nope is not None and qk_rope is not None and v_head_dim is not None:
        return qk_nope + qk_rope, v_head_dim
    head_dim = _first_positive_int(config, ("head_dim",))
    if head_dim is None:
        hidden_size = _first_positive_int(config, ("hidden_size",))
        if hidden_size is None:
            return None
        head_dim = hidden_size // num_heads
    return head_dim, head_dim


def _layer_mixer(config: Any, layer_type: str, attention: AttentionLayer) -> tuple[Optional[SequenceMixer], str]:
    """Return the sequence mixer of one layer type, or ``None`` with the reason it is not covered."""
    if layer_type == "full_attention":
        return attention, ""
    if layer_type == "sliding_attention":
        window = _first_positive_int(config, ("sliding_window",))
        if window is None:
            return None, "sliding_attention layers have no sliding_window"
        return AttentionLayer(attention.num_heads, attention.qk_head_dim, attention.v_head_dim, window), ""
    if layer_type == "linear_attention":
        delta_rule = _delta_rule_layer(config)
        if delta_rule is None:
            return None, ("linear_attention layers lack linear_num_value_heads / linear_key_head_dim / "
                          "linear_value_head_dim")
        return delta_rule, ""
    return None, f"sequence-mixer layer type {layer_type!r} is not covered by the formula"


def attention_layers_from_config(model_config: Any) -> tuple[tuple[SequenceMixer, ...], Optional[str]]:
    """Read the per-layer sequence-mixer geometry from a model configuration.

    Args:
        model_config: Hugging Face style model configuration.

    Returns:
        The sequence mixers, and a reason when the geometry is not covered.
    """
    config = _decoder_config(model_config)
    num_layers = _first_positive_int(config, ("num_hidden_layers",))
    num_heads = _first_positive_int(config, ("num_attention_heads",))
    if num_layers is None or num_heads is None:
        return (), "the model config has no num_hidden_layers / num_attention_heads"
    sparse_fields = [name for name in _SPARSE_ATTENTION_FIELDS if getattr(config, name, None)]
    if sparse_fields:
        return (), f"sparse attention ({', '.join(sparse_fields)}) is not covered by the PaLM formula"
    head_dims = _attention_head_dims(config, num_heads)
    if head_dims is None:
        return (), "the model config has no head_dim / hidden_size"

    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        layer_types = _legacy_layer_types(config, num_layers)
    if len(layer_types) != num_layers:
        return (), f"layer_types lists {len(layer_types)} layers but num_hidden_layers is {num_layers}"

    attention = AttentionLayer(num_heads, *head_dims)
    layers: list[SequenceMixer] = []
    for layer_type in layer_types:
        mixer, reason = _layer_mixer(config, layer_type, attention)
        if mixer is None:
            return (), reason
        layers.append(mixer)
    return tuple(layers), None
