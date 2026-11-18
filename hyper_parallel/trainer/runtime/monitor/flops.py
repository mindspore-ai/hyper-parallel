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
"""Versioned standard-block FLOPs composition, independent of model family names."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FlopsBlock:
    """One exclusively owned logical compute block, before parallel partitioning."""

    name: str
    kind: str
    input_size: int = 0
    output_size: int = 0
    heads: int = 0
    head_dim: int = 0
    multiplicity: int = 1
    backward_multiplier: float = 2.0
    kernel_size: int = 0

    def __post_init__(self) -> None:
        if (
            (not isinstance(self.multiplicity, int) or isinstance(self.multiplicity, bool))
            or self.multiplicity < 1
            or not math.isfinite(self.backward_multiplier)
            or self.backward_multiplier < 0
        ):
            raise ValueError("FLOPs block multiplicity and backward work must be nonnegative")
        if self.kind not in ("linear", "attention", "depthwise_conv1d", "gated_delta_rule"):
            raise ValueError(f"Unsupported standard FLOPs block: {self.kind}")
        if self.kind == "linear" and (self.input_size <= 0 or self.output_size <= 0):
            raise ValueError("Linear FLOPs dimensions must be positive")
        if self.kind in ("attention", "gated_delta_rule") and (self.heads <= 0 or self.head_dim <= 0):
            raise ValueError("Attention FLOPs dimensions must be positive")
        if self.kind == "depthwise_conv1d" and (self.input_size <= 0 or self.kernel_size <= 0):
            raise ValueError("Depthwise convolution requires positive channels and kernel size")

    def estimate(self, tokens: float, causal_pairs: float) -> float:
        """Count forward plus required backward work without recomputation."""
        if self.kind == "linear":
            forward = 2 * tokens * self.input_size * self.output_size
        elif self.kind == "attention":
            forward = 4 * causal_pairs * self.heads * self.head_dim
        elif self.kind == "depthwise_conv1d":
            forward = 2 * tokens * self.input_size * self.kernel_size
        elif self.kind == "gated_delta_rule":
            # Megatron's standard GDN core convention for equal key/value head dimensions.
            # Chunk/WY/solve overhead and recomputation belong to kernel work, not model MFU.
            forward = 8 * tokens * self.heads * self.head_dim**2
        else:
            raise ValueError(f"Unsupported standard FLOPs block: {self.kind}")
        return forward * (1 + self.backward_multiplier) * self.multiplicity


class BlockFlopsEstimator:
    """Require exclusive block names and explicit completeness before reporting MFU."""

    version = "standard-blocks-v2-causal-triangle-gdn-megatron"

    def __init__(self, blocks: list[FlopsBlock], missing: tuple[str, ...] = ()) -> None:
        """Validate exclusive block ownership and preserve missing coverage."""
        self.blocks, self.missing = blocks, missing
        names = [block.name for block in blocks]
        if len(set(names)) != len(names):
            raise ValueError("FLOPs blocks require unique ownership names")

    def estimate(self, tokens: float, sum_seq_len_sq: float) -> dict[str, float]:
        """Use the same triangular approximation as the existing trainer estimator."""
        if tokens < 0 or sum_seq_len_sq < 0:
            raise ValueError("FLOPs input geometry must be nonnegative")
        if self.missing:
            return {}
        return {block.name: block.estimate(tokens, sum_seq_len_sq / 2) for block in self.blocks}


def _gated_delta_blocks(config: Any, name: str, hidden: int) -> BlockFlopsEstimator:
    """Compose the standard equal-head-dimension GDN contract used by the hybrid text model."""
    fields = (
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    )
    values = [getattr(config, field, None) for field in fields]
    if any((not isinstance(value, int) or isinstance(value, bool)) or value <= 0 for value in values):
        return BlockFlopsEstimator([], (name + ".gdn_dimensions",))
    key_heads, value_heads, key_dim, value_dim, kernel = values
    if value_heads % key_heads or key_dim != value_dim:
        return BlockFlopsEstimator([], (name + ".gdn_head_layout",))
    key_width, value_width = key_heads * key_dim, value_heads * value_dim
    qkv_width = 2 * key_width + value_width
    return BlockFlopsEstimator(
        [
            FlopsBlock(name + ".in_proj_qkv", "linear", hidden, qkv_width),
            FlopsBlock(name + ".in_proj_z", "linear", hidden, value_width),
            FlopsBlock(name + ".in_proj_b", "linear", hidden, value_heads),
            FlopsBlock(name + ".in_proj_a", "linear", hidden, value_heads),
            FlopsBlock(name + ".conv1d", "depthwise_conv1d", input_size=qkv_width, kernel_size=kernel),
            FlopsBlock(name + ".delta_rule", "gated_delta_rule", heads=value_heads, head_dim=value_dim),
            FlopsBlock(name + ".out_proj", "linear", value_width, hidden),
        ]
    )


def transformer_blocks(config: Any, attention_gates: dict[int, bool] | None = None) -> BlockFlopsEstimator:
    """Compose standard blocks from structural fields; reject unknown attention kernels."""
    config = getattr(config, "text_config", config)
    required = ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size")
    if any(not hasattr(config, name) for name in required):
        return BlockFlopsEstimator([], ("transformer_dimensions",))
    hidden, layers, heads = config.hidden_size, config.num_hidden_layers, config.num_attention_heads
    if any(not isinstance(value, int) or value <= 0 for value in (hidden, layers, heads, config.vocab_size)):
        return BlockFlopsEstimator([], ("invalid_transformer_dimensions",))
    dim = getattr(config, "head_dim", None) or hidden // heads
    kv = getattr(config, "num_key_value_heads", None) or heads
    blocks, missing = [], []
    types = getattr(config, "layer_types", None)
    if types is not None and len(types) != layers:
        return BlockFlopsEstimator([], ("layer_types_length",))
    attention_gates = attention_gates or {}
    for index in range(layers):
        name = f"layers.{index}"
        layer_type = types[index] if types else "full_attention"
        if layer_type == "linear_attention":
            gdn = _gated_delta_blocks(config, name + ".linear_attn", hidden)
            blocks.extend(gdn.blocks)
            missing.extend(gdn.missing)
        elif layer_type not in ("full_attention", "attention"):
            missing.append(f"{name}.{types[index]}")
        elif hasattr(config, "kv_lora_rank") or hasattr(config, "linear_num_key_heads") and not types:
            missing.append(f"{name}.nonstandard_attention")
        else:
            gated = attention_gates.get(
                index, getattr(config, "attn_output_gate", getattr(config, "attention_output_gate", False))
            )
            blocks.extend(
                [
                    FlopsBlock(name + ".q", "linear", hidden, heads * dim),
                    FlopsBlock(name + ".kv", "linear", hidden, 2 * kv * dim),
                    FlopsBlock(name + ".o", "linear", heads * dim, hidden),
                    FlopsBlock(name + ".attention", "attention", heads=heads, head_dim=dim),
                ]
            )
            if gated:
                blocks.append(FlopsBlock(name + ".attention_gate", "linear", hidden, heads * dim))
        experts = getattr(config, "num_experts", 0)
        is_moe = (
            experts > 0
            and index not in (getattr(config, "mlp_only_layers", None) or ())
            and (index + 1) % max(1, getattr(config, "decoder_sparse_step", 1)) == 0
        )
        intermediate = (
            getattr(config, "moe_intermediate_size", None) if is_moe else getattr(config, "intermediate_size", None)
        )
        if intermediate is None:
            missing.append(name + ".ffn_dimensions")
            continue
        topk = getattr(config, "num_experts_per_tok", 1) if is_moe else 1
        for projection in ("gate", "up", "down"):
            inputs, outputs = (intermediate, hidden) if projection == "down" else (hidden, intermediate)
            blocks.append(FlopsBlock(name + ".ffn." + projection, "linear", inputs, outputs, multiplicity=topk))
        if is_moe:
            blocks.append(FlopsBlock(name + ".router", "linear", hidden, experts))
        shared = getattr(config, "shared_expert_intermediate_size", 0) if is_moe else 0
        if shared:
            for projection in ("gate", "up", "down"):
                inputs, outputs = (shared, hidden) if projection == "down" else (hidden, shared)
                blocks.append(FlopsBlock(name + ".shared." + projection, "linear", inputs, outputs))
            blocks.append(FlopsBlock(name + ".shared_gate", "linear", hidden, 1))
    blocks.append(FlopsBlock("lm_head", "linear", hidden, config.vocab_size))
    return BlockFlopsEstimator(blocks, tuple(missing))


@dataclass(frozen=True)
class DevicePeak:
    """Deployment-provided dense peak; never infer a SKU or sparsity multiplier."""

    sku: str
    dtype: str
    tflops: float
    source: str
    sparse: bool = False

    def __post_init__(self) -> None:
        if (
            not self.sku
            or not self.source
            or self.dtype not in ("float16", "bfloat16", "float32")
            or not math.isfinite(self.tflops)
            or self.tflops <= 0
            or self.sparse
        ):
            raise ValueError("MFU requires a sourced, finite, positive dense peak for the actual compute dtype")


def model_blocks(model: Any, config: Any) -> BlockFlopsEstimator:
    """Prefer an explicit adapter manifest; reject unmodelled gradient/architecture paths."""
    manifest = getattr(model, "monitor_flops_blocks", None)
    if manifest is not None:
        return BlockFlopsEstimator([FlopsBlock(**entry) for entry in manifest])
    if getattr(model, "peft_config", None) or any(not p.requires_grad for p in model.parameters()):
        return BlockFlopsEstimator([], ("frozen_or_peft_gradient_paths_require_adapter",))
    # Projection metadata survives parameter sharding; reading it needs no tensor hooks or data copies.
    text_config = getattr(config, "text_config", config)
    heads = getattr(text_config, "num_attention_heads", 0)
    dim = getattr(text_config, "head_dim", None)
    if dim is None and heads:
        dim = getattr(text_config, "hidden_size", 0) // heads
    gates = {}
    if heads and dim:
        for module in model.modules():
            index = getattr(module, "layer_idx", None)
            q_width = getattr(getattr(module, "q_proj", None), "out_features", None)
            if isinstance(index, int) and not isinstance(index, bool) and q_width in (heads * dim, 2 * heads * dim):
                gates[index] = q_width == 2 * heads * dim
    estimator = transformer_blocks(config, gates)
    if getattr(config, "vision_config", None) is not None:
        return BlockFlopsEstimator(estimator.blocks, (*estimator.missing, "vision_compute_requires_adapter"))
    return estimator
