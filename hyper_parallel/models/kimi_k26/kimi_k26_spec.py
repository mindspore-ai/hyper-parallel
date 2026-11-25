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
"""Kimi K2.6 native identity composing public MLA, experts, complete vision and spatial projection."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules import (
    CausalLMHead,
    GroupedExperts,
    LearnedAxialVisionTransformer,
    MultiLatentAttention,
    RMSNorm,
    RotaryEmbedding,
    SpatialPatchProjection,
)
from hyper_parallel.components.modules.common.swiglu_mlp import SwiGLUMLP
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_custom_model, register_model_adapter
from hyper_parallel.models.replacement import ModuleReplacementRequest


def get_replacement_rules(*, model: nn.Module, context: Mapping[str, Any]) -> tuple[ModuleReplacementRequest, ...]:
    """Declare actual native text/vision/projector boundaries while retaining separate Muon matrices."""
    fused = context.get("device_type") == "npu"
    config = model.config.text_config
    prefix = "model.language_model"
    dense = (
        min(config.first_k_dense_replace, config.num_hidden_layers)
        if config.n_routed_experts
        else config.num_hidden_layers
    )
    rules = [
        ModuleReplacementRequest(match=("lm_head",), target=CausalLMHead),
        ModuleReplacementRequest(match=(prefix + ".rotary_emb",), target=RotaryEmbedding),
        ModuleReplacementRequest(
            match=(prefix + ".layers.*.self_attn",),
            target=MultiLatentAttention,
            kwargs={"strategy": "expanded_ulysses", "overlap": False, "fuse_latent_qkv": False,
                    "key_expansion_layout": "bnsd"},
        ),
        ModuleReplacementRequest(
            match=(
                prefix + ".layers.*.input_layernorm",
                prefix + ".layers.*.post_attention_layernorm",
                prefix + ".norm",
            ),
            target=RMSNorm,
            kwargs={"use_fused_ops": fused},
        ),
        ModuleReplacementRequest(match=("model.vision_tower",), target=LearnedAxialVisionTransformer),
        ModuleReplacementRequest(match=("model.mm_projector",), target=SpatialPatchProjection),
    ]
    if dense:
        rules.append(
            ModuleReplacementRequest(
                match=tuple(f"{prefix}.layers.{i}.mlp" for i in range(dense)),
                target=SwiGLUMLP,
                kwargs={"state_layout": "separate"},
            )
        )
    if dense < config.num_hidden_layers:
        rules.append(
            ModuleReplacementRequest(
                match=(prefix + ".layers.*.mlp.experts",), target=GroupedExperts, kwargs={"use_grouped_gemm": fused}
            )
        )
        if config.n_shared_experts:
            rules.append(
                ModuleReplacementRequest(
                    match=(prefix + ".layers.*.mlp.shared_experts",),
                    target=SwiGLUMLP,
                    kwargs={"state_layout": "separate"},
                )
            )
    return tuple(rules)


def get_fsdp_wrap_modules(model: nn.Module) -> tuple[str, ...]:
    """Own each complete vision block and the complete language-width projector."""
    return tuple(f"model.vision_tower.layers.{i}" for i in range(len(model.model.vision_tower.layers))) + (
        "model.mm_projector",
    )


def get_fsdp_excluded_subtrees(model: nn.Module) -> tuple[str, ...]:
    """Keep vision modules outside language decoder inference."""
    del model
    return ("model.vision_tower", "model.mm_projector")


MODEL_SPEC = ModelAdapterSpec(
    architecture="Kimi_K25ForConditionalGeneration",
    model_type="kimi_k26",
    replacements=get_replacement_rules,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
)
LEGACY_SPEC = ModelAdapterSpec(
    architecture="KimiK25ForConditionalGeneration",
    model_type="kimi_k26_legacy",
    replacements=get_replacement_rules,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
)
register_model_adapter(LEGACY_SPEC)

for architecture in ("Kimi_K25ForConditionalGeneration", "KimiK25ForConditionalGeneration"):
    register_custom_model(
        architecture, "hyper_parallel.models.kimi_k26.modeling_kimi_k26", "Kimi_K25ForConditionalGeneration"
    )


LEGACY_TARGETS = {
    "hyper_parallel.models.kimi_k26.adapter.distributed.context_parallel.bind_context_parallel": (
        "hyper_parallel.models.kimi_k26.modeling_kimi_k26.bind_context_parallel", {},
    ),
    "hyper_parallel.models.kimi_k26.adapter.distributed.context_parallel.build_cp_attention_mask": (
        "hyper_parallel.distributed.context_parallel.attention.build_cp_attention_mask", {},
    ),
    "hyper_parallel.models.kimi_k26.adapter.distributed.ep_compute.kimi_k26_ep_compute_fn": (
        "hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn", {},
    ),
    "hyper_parallel.models.kimi_k26.adapter.loss.KimiChunkedCausalLMLoss": (
        "hyper_parallel.components.losses.ChunkedCausalLMLoss", {"labels_are_shifted": False},
    ),
    "hyper_parallel.models.kimi_k26.adapter.data.": (
        "hyper_parallel.models.kimi_k26.dataset_preprocessing.", {},
    ),
    "hyper_parallel.models.kimi_k26.adapter.validation.": (
        "hyper_parallel.models.kimi_k26.validation.", {},
    ),
}
