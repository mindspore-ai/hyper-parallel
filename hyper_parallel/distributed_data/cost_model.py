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
"""Replaceable workload estimates, independent of packing hard limits."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from hyper_parallel.distributed_data.schema import SampleMetadata, WorkloadCost


class CostModel(Protocol):
    """Deterministic, stateless CPU estimate evaluated on the centralized planner.

    Implementations consume lightweight ``metadata.features``, not image tensors.
    They must not mutate metadata, perform distributed collectives, or change
    ``pack_tokens`` / ``packing_costs``. Return components in comparable units.
    """

    def __call__(self, metadata: SampleMetadata) -> WorkloadCost:
        """Estimate one selected sample's additive workload."""


def _configuration_dict(config: Any) -> dict[str, Any]:
    """Accept an ordinary dictionary or a configuration with ``to_dict``."""
    if isinstance(config, Mapping):
        return dict(config)
    if callable(getattr(config, "to_dict", None)):
        value = config.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise ValueError("model_config must be a mapping or a configuration exposing to_dict().")


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, but got {value!r}.")
    return value


@dataclass(frozen=True)
class BackboneFlopsConfig:
    """Latent-attention decoder dimensions, independent of model libraries.

    Only the shared backbone decoder layers are modeled. Diffusion input/output
    stacks, the language-model head, ViT, VAE, and their projectors are excluded.
    ``mlp_layer_types`` follows the actual layer order: ``dense`` or ``sparse``.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    mlp_layer_types: tuple[str, ...]
    kv_lora_rank: int
    q_lora_rank: int | None
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    moe_intermediate_size: int = 0
    n_routed_experts: int = 0
    num_experts_per_tok: int = 0
    n_shared_experts: int = 0
    attn_block_mode: str = "tail"
    diff_time_embed_mode: str = "add"

    def __post_init__(self) -> None:
        """Reject dimensions or layer patterns not covered by this estimate."""
        for name in (
                "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                "kv_lora_rank", "v_head_dim",
        ):
            _integer(getattr(self, name), name)
        for name in ("qk_nope_head_dim", "qk_rope_head_dim", "n_shared_experts"):
            _integer(getattr(self, name), name, 0)
        if self.q_lora_rank is not None:
            _integer(self.q_lora_rank, "q_lora_rank")
        if self.qk_nope_head_dim + self.qk_rope_head_dim == 0:
            raise ValueError("The combined query/key head dimension must be positive.")
        if (
                not isinstance(self.mlp_layer_types, tuple)
                or len(self.mlp_layer_types) != self.num_hidden_layers
                or any(kind not in ("dense", "sparse") for kind in self.mlp_layer_types)
        ):
            raise ValueError("mlp_layer_types must contain one 'dense' or 'sparse' entry per backbone layer.")
        for name in ("moe_intermediate_size", "n_routed_experts", "num_experts_per_tok"):
            _integer(getattr(self, name), name, int("sparse" in self.mlp_layer_types))
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok cannot exceed n_routed_experts.")
        if self.attn_block_mode not in ("tail", "blockmask"):
            raise ValueError("attn_block_mode must be 'tail' or 'blockmask'.")
        if self.diff_time_embed_mode not in ("add", "prepend", "adaln"):
            raise ValueError("diff_time_embed_mode must be 'add', 'prepend', or 'adaln'.")

    @classmethod
    def from_model_config(cls, model_config: Any) -> "BackboneFlopsConfig":
        """Read the final model configuration after architecture overrides.

        Args:
            model_config: Configuration with explicit decoder dimensions and layer types.
        """
        config = _configuration_dict(model_config)
        text = _configuration_dict(config.get("text_config", config))
        required = (
            "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
            "kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        )
        missing = [name for name in required if name not in text]
        if missing:
            raise ValueError(f"Backbone cost model is missing configuration fields: {missing}.")
        pattern = text.get("mlp_layer_types")
        if not isinstance(pattern, (list, tuple)):
            raise ValueError("model_config must provide mlp_layer_types in actual decoder layer order.")
        qk_dim = (
            _integer(text["qk_nope_head_dim"], "qk_nope_head_dim", 0)
            + _integer(text["qk_rope_head_dim"], "qk_rope_head_dim", 0)
        )
        if text.get("qk_head_dim", qk_dim) != qk_dim:
            raise ValueError("qk_head_dim must equal qk_nope_head_dim + qk_rope_head_dim.")
        return cls(
            **{name: text[name] for name in required},
            mlp_layer_types=tuple(pattern),
            **{name: text.get(name, 0) for name in (
                "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok", "n_shared_experts",
            )},
            attn_block_mode=config.get("attn_block_mode", "tail"),
            diff_time_embed_mode=config.get("diff_time_embed_mode", "add"),
        )


class DefaultCostModel:
    """Shape-based FLOPs for a shared latent-attention and expert backbone.

    Args:
        model_config: Final model configuration or :class:`BackboneFlopsConfig`.
            Required to derive the backbone's arithmetic workload.
        training_multiplier: Constant forward-to-training FLOPs proxy: 1 for
            forward, 3 for forward/backward, or 4 for full-layer recomputation.
            It does not change balancing decisions when shared by all samples.

    Note:
        MAC=2 FLOPs. The attention area per independent raw item is
        ``P**2 / 2 + D * (P + D)``; blockwise attention adds half the square of each
        conditional-image placeholder run. ``P`` includes control tokens and
        conditional-image tokens, not just natural-language text. Costs add
        across packed items; there is no attention between different items.

        Linear terms include MLA Q/KV/O projections, dense SwiGLU, routed top-k
        and shared experts, and router projection. Norms, activations, softmax,
        attention tile padding, communication and expert-kernel efficiency are
        excluded. This is theoretical arithmetic, not a latency predictor.
    """

    def __init__(self, model_config: Any, *, training_multiplier: float = 3.0) -> None:
        """Compile workload coefficients from the effective decoder configuration.

        Args:
            model_config: Explicit decoder dimensions and attention layout.
            training_multiplier: Forward-to-training arithmetic scale.
        """
        if model_config is None:
            raise ValueError("DefaultCostModel requires model_config; provide it or supply a custom cost_model.")
        if (
                not isinstance(training_multiplier, (int, float))
                or isinstance(training_multiplier, bool)
                or not math.isfinite(training_multiplier)
                or training_multiplier <= 0
        ):
            raise ValueError("training_multiplier must be finite and positive.")
        self.training_multiplier = float(training_multiplier)
        self.config = (
            model_config if isinstance(model_config, BackboneFlopsConfig)
            else BackboneFlopsConfig.from_model_config(model_config)
        )
        self.linear_flops_per_token = 0
        self.attention_flops_per_pair = 0
        self._compile_coefficients()
        identity = {"config": asdict(self.config), "training_multiplier": self.training_multiplier}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        self.model_id = f"backbone-flops-v1-{digest}"

    def _compile_coefficients(self) -> None:
        config = self.config
        hidden, heads = config.hidden_size, config.num_attention_heads
        qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        query_mac = hidden * heads * qk_dim if config.q_lora_rank is None else (
            hidden * config.q_lora_rank + config.q_lora_rank * heads * qk_dim
        )
        projection_mac = (
            query_mac
            + hidden * (config.kv_lora_rank + config.qk_rope_head_dim)
            + config.kv_lora_rank * heads * (config.qk_nope_head_dim + config.v_head_dim)
            + heads * config.v_head_dim * hidden
        )
        dense_layers = config.mlp_layer_types.count("dense")
        sparse_layers = config.num_hidden_layers - dense_layers
        dense_mac = 3 * hidden * config.intermediate_size
        sparse_mac = (
            3 * hidden * config.moe_intermediate_size * (config.num_experts_per_tok + config.n_shared_experts)
            + hidden * config.n_routed_experts
        )
        self.linear_flops_per_token = 2 * (
            config.num_hidden_layers * projection_mac + dense_layers * dense_mac + sparse_layers * sparse_mac
        )
        self.attention_flops_per_pair = 2 * config.num_hidden_layers * heads * (qk_dim + config.v_head_dim)

    def forward_flops(self, metadata: SampleMetadata) -> float:
        """Estimate decoder forward FLOPs from CPU-only per-raw-item metadata.

        Args:
            metadata: One independent item's token counts and attention segments.
        """
        prefix = _integer(metadata.features.get("P"), "metadata.features['P']", 0)
        diffusion = _integer(metadata.features.get("D"), "metadata.features['D']", 0)
        if prefix + diffusion != metadata.pack_tokens:
            raise ValueError("metadata features P + D must equal pack_tokens before optional timestep-token insertion.")
        if diffusion and self.config.diff_time_embed_mode == "prepend":
            diffusion += 1
        pairs = 0.5 * prefix * prefix + diffusion * (prefix + diffusion)
        if self.config.attn_block_mode == "blockmask":
            runs = metadata.features.get("cond_image_token_lengths")
            if not isinstance(runs, (list, tuple)):
                raise ValueError("Blockwise attention requires cond_image_token_lengths (empty for no images).")
            lengths = [_integer(length, "conditional-image run length") for length in runs]
            if sum(lengths) > prefix:
                raise ValueError("Conditional-image run lengths cannot exceed the prefix token count P.")
            pairs += 0.5 * sum(length * length for length in lengths)
        return float(self.linear_flops_per_token * (prefix + diffusion) + self.attention_flops_per_pair * pairs)

    def __call__(self, metadata: SampleMetadata) -> WorkloadCost:
        """Return backbone-only workload without changing physical packing caps."""
        return WorkloadCost(llm=self.training_multiplier * self.forward_flops(metadata))


def resolve_cost_model(cost_model: CostModel | None, model_config: Any = None) -> CostModel:
    """Use an explicit callback or construct the default from model dimensions.

    Args:
        cost_model: Optional user-supplied workload callback.
        model_config: Required backbone configuration when the callback is absent.
    """
    if cost_model is None:
        return DefaultCostModel(model_config)
    if not callable(cost_model):
        raise ValueError("cost_model must be callable: SampleMetadata -> WorkloadCost.")
    return cost_model


__all__ = ["BackboneFlopsConfig", "CostModel", "DefaultCostModel"]
