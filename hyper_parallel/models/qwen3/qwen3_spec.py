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
"""Qwen3 default family declarations, independent of recipe configuration."""

from importlib import import_module
from types import ModuleType

from hyper_parallel.components.modules.attention.grouped_query_attention import GroupedQueryAttention
from hyper_parallel.components.modules.common.rms_norm import RMSNorm
from hyper_parallel.components.modules.common.rotary_embedding import RotaryEmbedding
from hyper_parallel.components.modules.common.swiglu_mlp import SwiGLUMLP, parameter_sharding_rules
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.replacement import ModuleReplacementRequest


def get_replacement_rules() -> tuple[ModuleReplacementRequest, ...]:
    """Declare public families while retaining Muon's independent Q/K/V and gate/up matrices."""
    return (
        ModuleReplacementRequest(match=("model.rotary_emb",), target=RotaryEmbedding),
        ModuleReplacementRequest(
            match=("model.layers.*.self_attn",),
            target=GroupedQueryAttention,
            kwargs={"fuse_qkv": False, "use_fused_ops": False},
        ),
        ModuleReplacementRequest(match=("model.layers.*.mlp",), target=SwiGLUMLP, kwargs={"state_layout": "separate"}),
        ModuleReplacementRequest(
            match=("model.layers.*.input_layernorm", "model.layers.*.post_attention_layernorm", "model.norm"),
            target=RMSNorm,
        ),
    )


def _codegen_provider() -> ModuleType:
    """Retain existing code generation while defaults use public families directly."""
    return import_module("hyper_parallel.codegen.inline.component_source")


MODEL_SPEC = ModelAdapterSpec(
    architecture="Qwen3ForCausalLM",
    model_type="qwen3",
    replacements=get_replacement_rules,
    sharding_rules=parameter_sharding_rules,
    codegen=_codegen_provider,
)


# Historical YAML targets resolve to public families; new declarations use canonical names.
LEGACY_TARGETS = {
    "hyper_parallel.models.qwen3.adapter.conversion.module_replacement.replace_qwen3_rms_norm": (
        "hyper_parallel.components.modules.RMSNorm",
        {"use_fused_ops": True},
    ),
    "hyper_parallel.models.qwen3.adapter.conversion.module_replacement.replace_qwen3_flash_attention": (
        "hyper_parallel.components.modules.GroupedQueryAttention",
        {},
    ),
    "hyper_parallel.models.qwen3.adapter.conversion.module_replacement.replace_qwen3_swiglu_mlp": (
        "hyper_parallel.components.modules.SwiGLUMLP",
        {},
    ),
}
