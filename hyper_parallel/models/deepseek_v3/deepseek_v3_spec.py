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
"""DeepSeek V3 and V2 compatibility identities using public MLA, expert and MLP families."""

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules import GroupedExperts, MultiLatentAttention, RMSNorm, RotaryEmbedding
from hyper_parallel.components.modules.common.swiglu_mlp import SwiGLUMLP
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter
from hyper_parallel.models.replacement import ModuleReplacementRequest


def get_replacement_rules(*, model: nn.Module, context: Mapping[str, Any]) -> tuple[ModuleReplacementRequest, ...]:
    """Declare architecture-owned boundaries while preserving the existing packed latent-Q/KV matrix."""
    del context
    config = model.config
    depth = config.num_hidden_layers
    dense = min(config.first_k_dense_replace, depth) if config.n_routed_experts else depth
    rules = [
        ModuleReplacementRequest(
            match=("model.rotary_emb",), target=RotaryEmbedding,
            kwargs={"output_format": "complex" if config.model_type == "deepseek_v2" else "cos_sin"},
        ),
        ModuleReplacementRequest(
            match=("model.layers.*.self_attn",),
            target=MultiLatentAttention,
            kwargs={
                "strategy": "expanded_ulysses",
                "overlap": False,
                "fuse_latent_qkv": config.model_type != "deepseek_v2",
                "key_expansion_layout": "bnsd" if config.model_type == "deepseek_v2" else "bsnd",
            },
        ),
        ModuleReplacementRequest(
            match=("model.layers.*.input_layernorm", "model.layers.*.post_attention_layernorm", "model.norm"),
            target=RMSNorm,
        ),
    ]
    if dense:
        rules.append(
            ModuleReplacementRequest(
                match=tuple(f"model.layers.{index}.mlp" for index in range(dense)),
                target=SwiGLUMLP,
                kwargs={"state_layout": "separate"},
            )
        )
    if dense < depth:
        rules.append(
            ModuleReplacementRequest(
                match=("model.layers.*.mlp.experts",), target=GroupedExperts, kwargs={"use_grouped_gemm": False}
            )
        )
        if config.n_shared_experts:
            rules.append(
                ModuleReplacementRequest(
                    match=("model.layers.*.mlp.shared_experts",), target=SwiGLUMLP, kwargs={"state_layout": "separate"}
                )
            )
    return tuple(rules)


MODEL_SPEC = ModelAdapterSpec(
    architecture="DeepseekV3ForCausalLM", model_type="deepseek_v3", replacements=get_replacement_rules
)
V2_SPEC = ModelAdapterSpec(
    architecture="DeepseekV2ForCausalLM", model_type="deepseek_v2", replacements=get_replacement_rules
)
register_model_adapter(V2_SPEC)


LEGACY_TARGETS = {
    "hyper_parallel.models.deepseek_v3.adapter.conversion.module_replacement.replace_linear": (
        "hyper_parallel.components.quantization.modules.replacement.replace_linear",
        {},
    ),
    "hyper_parallel.models.deepseek_v3.adapter.conversion.module_replacement.replace_grouped_experts": (
        "hyper_parallel.components.quantization.modules.replacement.replace_grouped_experts",
        {},
    ),
}
