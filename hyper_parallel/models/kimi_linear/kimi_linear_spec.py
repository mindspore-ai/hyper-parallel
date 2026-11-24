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
"""Native Kimi Linear boundaries with separate NoPE MLA and unbounded channel-delta families."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules import ChannelDeltaAttention, GroupedExperts, MultiLatentAttention, RMSNorm
from hyper_parallel.components.modules.common.swiglu_mlp import SwiGLUMLP
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.replacement import ModuleReplacementRequest


def get_replacement_rules(*, model: nn.Module, context: Mapping[str, Any]) -> tuple[ModuleReplacementRequest, ...]:
    """Select actual attention/MLP roles without borrowing the bounded K3 decay or output gate."""
    del context
    config = model.config
    rules = [
        ModuleReplacementRequest(
            match=("model.layers.*.input_layernorm", "model.layers.*.post_attention_layernorm", "model.norm"),
            target=RMSNorm,
        )
    ]
    for kind, target, arguments in (
        ("linear_attention", ChannelDeltaAttention, {}),
        (
            "full_attention",
            MultiLatentAttention,
            {"strategy": "expanded_ulysses", "overlap": False, "fuse_latent_qkv": False, "use_nope": True},
        ),
    ):
        boundaries = tuple(
            f"model.layers.{index}.self_attn" for index, value in enumerate(config.layer_types) if value == kind
        )
        if boundaries:
            rules.append(ModuleReplacementRequest(match=boundaries, target=target, kwargs=arguments))
    dense = tuple(
        f"model.layers.{index}.mlp" for index, value in enumerate(config.mlp_layer_types) if value != "sparse"
    )
    if dense:
        rules.append(ModuleReplacementRequest(match=dense, target=SwiGLUMLP, kwargs={"state_layout": "separate"}))
    if "sparse" in config.mlp_layer_types:
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
    architecture="KimiLinearForCausalLM", model_type="kimi_linear", replacements=get_replacement_rules
)
