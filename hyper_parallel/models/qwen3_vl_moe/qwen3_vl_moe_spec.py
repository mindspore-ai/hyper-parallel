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
"""Qwen3 VL MoE identity with complete public axial vision, GQA, experts and MRoPE families."""

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules import (
    AxialVisionTransformer,
    CausalLMHead,
    GroupedQueryAttention,
    GroupedExperts,
    RMSNorm,
    RotaryEmbedding,
)
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.replacement import ModuleReplacementRequest
from hyper_parallel.models.registry import register_custom_model


def get_replacement_rules(*, model: nn.Module, context: Mapping[str, Any]) -> tuple[ModuleReplacementRequest, ...]:
    """Declare routed GQA decoder boundaries and the complete deepstack visual encoder."""
    del context
    conditional = hasattr(model.config, "vision_config")
    prefix = "model.language_model" if conditional else "model"
    rules = [
        ModuleReplacementRequest(
            match=(prefix + ".rotary_emb",), target=RotaryEmbedding, kwargs={"mrope_interleaved": True}
        ),
        ModuleReplacementRequest(
            match=(prefix + ".layers.*.self_attn",),
            target=GroupedQueryAttention,
            kwargs={"fuse_qkv": False, "use_fused_ops": False},
        ),
        ModuleReplacementRequest(
            match=(prefix + ".layers.*.mlp.experts",), target=GroupedExperts, kwargs={"use_grouped_gemm": False}
        ),
        ModuleReplacementRequest(
            match=(
                prefix + ".layers.*.input_layernorm",
                prefix + ".layers.*.post_attention_layernorm",
                prefix + ".norm",
            ),
            target=RMSNorm,
        ),
        ModuleReplacementRequest(match=("lm_head",), target=CausalLMHead),
    ]
    if conditional:
        rules.append(ModuleReplacementRequest(match=("model.visual",), target=AxialVisionTransformer))
    return tuple(rules)


def get_fsdp_wrap_modules(model: nn.Module) -> tuple[str, ...]:
    """Expose complete visual blocks and merger as independent FSDP execution units."""
    visual = getattr(model.model, "visual", None)
    if visual is None:
        return ()
    return tuple(f"model.visual.blocks.{index}" for index in range(len(visual.blocks))) + ("model.visual.merger",)


def get_fsdp_excluded_subtrees(model: nn.Module) -> tuple[str, ...]:
    """Keep the visual container outside generic language-decoder inference."""
    return ("model.visual",) if hasattr(model.model, "visual") else ()


def get_fsdp_execution_order(model: nn.Module, module_fqns: tuple[str, ...]) -> tuple[str, ...]:
    """Follow the source VLM's embedding-before-vision first-forward execution order."""
    del model
    embedding = "model.language_model.embed_tokens"
    return (
        (embedding,) + tuple(name for name in module_fqns if name != embedding)
        if embedding in module_fqns
        else module_fqns
    )


MODEL_SPEC = ModelAdapterSpec(
    architecture="Qwen3VLMoeForConditionalGeneration",
    model_type="qwen3_vl_moe",
    replacements=get_replacement_rules,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
    fsdp_execution_order=get_fsdp_execution_order,
)

register_custom_model(
    "Qwen3VLMoeForConditionalGeneration",
    "hyper_parallel.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
    "Qwen3VLMoeForConditionalGeneration",
)
