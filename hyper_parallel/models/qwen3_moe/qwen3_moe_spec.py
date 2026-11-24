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
"""Qwen3-MoE source boundaries and public module-family defaults."""
from importlib import import_module
from types import ModuleType

from hyper_parallel.components.modules import GroupedExperts, GroupedQueryAttention, RMSNorm, RotaryEmbedding
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_custom_model
from hyper_parallel.models.replacement import ModuleReplacementRequest


def get_replacement_rules() -> tuple[ModuleReplacementRequest, ...]:
    """Preserve the frozen recipe's packed QKV and expert NS matrix boundaries."""
    return (
        ModuleReplacementRequest(match=("model.rotary_emb",), target=RotaryEmbedding),
        ModuleReplacementRequest(
            match=("model.layers.*.self_attn",),
            target=GroupedQueryAttention,
            kwargs={"fuse_qkv": True, "ulysses_degree": 1, "use_fused_ops": False},
        ),
        ModuleReplacementRequest(
            match=("model.layers.*.mlp.experts",), target=GroupedExperts, kwargs={"use_grouped_gemm": False}
        ),
        ModuleReplacementRequest(
            match=("model.layers.*.input_layernorm", "model.layers.*.post_attention_layernorm", "model.norm"),
            target=RMSNorm,
            kwargs={"use_fused_ops": False},
        ),
    )


def _codegen_provider() -> ModuleType:
    """Use the shared renderer without model-specific forwarding modules."""
    return import_module("hyper_parallel.codegen.inline.component_source")


MODEL_SPEC = ModelAdapterSpec(
    architecture="Qwen3MoeForCausalLM",
    model_type="qwen3_moe",
    replacements=get_replacement_rules,
    codegen=_codegen_provider,
)
register_custom_model(
    "Qwen3MoeForCausalLM", "hyper_parallel.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeForCausalLM"
)


LEGACY_TARGETS = {
    "hyper_parallel.models.qwen3_moe.adapter.conversion.module_replacement.replace_qwen3_moe_rms_norm": (
        "hyper_parallel.components.modules.RMSNorm", {"use_fused_ops": True},
    ),
    "hyper_parallel.models.qwen3_moe.adapter.conversion.module_replacement.replace_qwen3_moe_flash_attention": (
        "hyper_parallel.components.modules.GroupedQueryAttention", {},
    ),
    "hyper_parallel.models.qwen3_moe.adapter.conversion.module_replacement.replace_qwen3_moe_grouped_experts": (
        "hyper_parallel.components.modules.GroupedExperts", {},
    ),
    "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn": (
        "hyper_parallel.distributed.expert_parallel.recipes.topk_router_ep_compute_fn", {},
    ),
}
