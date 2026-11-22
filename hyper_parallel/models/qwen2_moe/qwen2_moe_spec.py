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
"""Qwen2-MoE source boundaries, shared-gate ownership and default public families."""

from typing import Any

from hyper_parallel.components.modules import GroupedExperts, GroupedQueryAttention, RMSNorm, RotaryEmbedding
from hyper_parallel.components.modules.common.swiglu_mlp import SwiGLUMLP
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.replacement import ModuleReplacementRequest


def get_replacement_rules() -> tuple[ModuleReplacementRequest, ...]:
    """Retain separate attention/shared-expert matrices and source-ordered expert execution."""
    return (
        ModuleReplacementRequest(match=("model.rotary_emb",), target=RotaryEmbedding),
        ModuleReplacementRequest(
            match=("model.layers.*.self_attn",),
            target=GroupedQueryAttention,
            kwargs={"fuse_qkv": False, "use_fused_ops": False},
        ),
        ModuleReplacementRequest(
            match=("model.layers.*.mlp.experts",), target=GroupedExperts, kwargs={"use_grouped_gemm": False}
        ),
        ModuleReplacementRequest(
            match=("model.layers.*.mlp.shared_expert",), target=SwiGLUMLP, kwargs={"state_layout": "separate"}
        ),
        ModuleReplacementRequest(
            match=("model.layers.*.input_layernorm", "model.layers.*.post_attention_layernorm", "model.norm"),
            target=RMSNorm,
        ),
    )


def build_parameter_sharding_rules() -> list[tuple[list[str], Any]]:
    """Keep the scalar shared-expert gate replicated rather than splitting its output dimension."""
    # Planner discovery imports model specs, so parameter-role import must wait until invocation.
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=import-outside-toplevel
        ParamRole,
    )

    return [(["shared_expert_gate"], ParamRole.REPLICATED)]


MODEL_SPEC = ModelAdapterSpec(
    architecture="Qwen2MoeForCausalLM",
    model_type="qwen2_moe",
    replacements=get_replacement_rules,
    sharding_rules=build_parameter_sharding_rules,
)
