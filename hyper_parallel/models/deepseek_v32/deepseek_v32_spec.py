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
"""DeepSeek-V3.2 source boundaries and public DSA/MLP/expert/state families."""

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules import DeepseekSparseAttention, GroupedExperts, RMSNorm, RotaryEmbedding
from hyper_parallel.components.modules.common.swiglu_mlp import SwiGLUMLP
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.replacement import ModuleReplacementRequest


def get_replacement_rules(*, model: nn.Module, context: Mapping[str, Any]) -> tuple[ModuleReplacementRequest, ...]:
    """Declare real dense/routed boundaries from architectural layer counts, without installing components."""
    config = model.config
    depth = config.num_hidden_layers
    dense = min(config.first_k_dense_replace, depth)
    rules = [
        ModuleReplacementRequest(match=("model.rotary_emb",), target=RotaryEmbedding),
        ModuleReplacementRequest(
            match=("model.layers.*.self_attn",),
            target=DeepseekSparseAttention,
            kwargs={"kv_cp_size": context.get("cp_size", 1), "kv_cp_strategy": "allgather", "cp_comm_overlap": True},
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
        rules.append(ModuleReplacementRequest(match=("model.layers.*.mlp.experts",), target=GroupedExperts))
        if config.n_shared_experts:
            rules.append(
                ModuleReplacementRequest(
                    match=("model.layers.*.mlp.shared_experts",), target=SwiGLUMLP, kwargs={"state_layout": "separate"}
                )
            )
    return tuple(rules)


MODEL_SPEC = ModelAdapterSpec(
    architecture="DeepseekV32ForCausalLM", model_type="deepseek_v32", replacements=get_replacement_rules
)


LEGACY_TARGETS = {
    "hyper_parallel.models.deepseek_v32.adapter.conversion.module_replacement.replace_deepseek_v32_rms_norm": (
        "hyper_parallel.components.modules.RMSNorm",
        {"use_fused_ops": True},
    ),
    "hyper_parallel.models.deepseek_v32.adapter.conversion.module_replacement.replace_deepseek_v32_sparse_attention": (
        "hyper_parallel.components.modules.DeepseekSparseAttention",
        {},
    ),
    "hyper_parallel.models.deepseek_v32.adapter.conversion.module_replacement.replace_deepseek_v32_grouped_experts": (
        "hyper_parallel.components.modules.GroupedExperts",
        {},
    ),
}
