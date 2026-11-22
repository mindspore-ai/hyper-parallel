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
"""K3 declarations using public computation families and a model-owned temporal vision family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules import (
    CausalLMHead,
    GatedMultiLatentAttention,
    KimiDeltaAttention,
    RMSNorm,
    SiTUMLP,
    SpatialPatchProjection,
)
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.kimi_k3.components.latent_moe import KimiLatentMoE
from hyper_parallel.models.kimi_k3.components.vision import TemporalVisionTransformer
from hyper_parallel.models.kimi_k3.source_compat import prepare_model_import
from hyper_parallel.models.registry import register_custom_model
from hyper_parallel.models.replacement import ModuleReplacementRequest


def build_parameter_sharding_rules() -> list[tuple[list[str], Any]]:
    """Return KDA parameter roles not covered by generic naming rules."""
    return [
        (["block_sparse_moe.gate"], ParamRole.MOE_GATE),
        (["A_log", "dt_bias"], ParamRole.REPLICATED),
        (["q_conv1d", "k_conv1d", "v_conv1d"], ParamRole.COLWISE),
        (["q_a_proj", "kv_a_proj_with_mqa"], ParamRole.REPLICATED),
        (["self_attention_res_proj", "mlp_res_proj", "output_attn_res_proj"],
         ParamRole.REPLICATED),
        (["f_a_proj", "g_a_proj"], ParamRole.REPLICATED),
        (["f_b_proj", "b_proj", "g_proj", "g_b_proj"], ParamRole.COLWISE),
        (["vision_tower.", "mm_projector."], ParamRole.REPLICATED),
    ]



def get_replacement_rules(*, model: nn.Module, context: Mapping[str, Any]) -> tuple[ModuleReplacementRequest, ...]:
    """Retain independent source matrices, latent MoE routing and attention-residual composition."""
    del context
    config = model.config.text_config
    prefix = "language_model.model"
    if config.hidden_act != "situ" or not config.mla_use_output_gate:
        raise ValueError("K3 declarations require the released SiTU and output-gated MLA structure")
    rules = [
        ModuleReplacementRequest(
            match=(
                prefix + ".norm",
                prefix + ".output_attn_res_norm",
                prefix + ".layers.*.input_layernorm",
                prefix + ".layers.*.post_attention_layernorm",
                prefix + ".layers.*.self_attention_res_norm",
                prefix + ".layers.*.mlp_res_norm",
            ),
            target=RMSNorm,
            kwargs={"cast_order": "before_weight"},
        ),
        ModuleReplacementRequest(match=(prefix + ".layers.*.mlp",), target=SiTUMLP),
        ModuleReplacementRequest(match=(prefix + ".layers.*.block_sparse_moe",), target=KimiLatentMoE),
        ModuleReplacementRequest(match=("vision_tower",), target=TemporalVisionTransformer),
        ModuleReplacementRequest(match=("mm_projector",), target=SpatialPatchProjection),
        ModuleReplacementRequest(match=("language_model.lm_head",), target=CausalLMHead),
    ]
    for is_linear, target, options in (
        (True, KimiDeltaAttention, {"backend": "eager", "ulysses_degree": None}),
        (
            False,
            GatedMultiLatentAttention,
            {"fuse_latent_qkv": False, "source_contract": "tensor", "strategy": "expanded_ulysses", "overlap": False},
        ),
    ):
        names = tuple(
            f"{prefix}.layers.{index}.self_attn"
            for index in range(config.num_hidden_layers)
            if bool(config.is_kda_layer(index)) == is_linear
        )
        if names:
            rules.append(ModuleReplacementRequest(match=names, target=target, kwargs=options))
    return tuple(rules)


def get_fsdp_wrap_modules(model: nn.Module) -> tuple[str, ...]:
    """Expose visual blocks and projector as units separate from the language decoder."""
    return tuple(f"vision_tower.encoder.blocks.{index}" for index in range(len(model.vision_tower.encoder.blocks))) + (
        "mm_projector",
    )


def get_fsdp_excluded_subtrees(model: nn.Module) -> tuple[str, ...]:
    """Exclude the visual architecture from language decoder discovery."""
    del model
    return ("vision_tower", "mm_projector")


MODEL_SPEC = ModelAdapterSpec(
    architecture="KimiK3ForConditionalGeneration",
    model_type="kimi_k3",
    source_import=prepare_model_import,
    replacements=get_replacement_rules,
    sharding_rules=build_parameter_sharding_rules,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
)
register_custom_model(
    "KimiK3ForConditionalGeneration", "hyper_parallel.models.kimi_k3.modeling_kimi_k3", "KimiK3ForConditionalGeneration"
)


LEGACY_TARGETS = {
    "hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel.": (
        "hyper_parallel.distributed.context_parallel.wrappers.", {},
    ),
    "hyper_parallel.models.kimi_k3.adapter.mla.KimiMultiLatentAttention": (
        "hyper_parallel.components.modules.GatedMultiLatentAttention", {"source_contract": "tensor"},
    ),
    "hyper_parallel.models.kimi_k3.adapter.data.runtime.KimiK3PackedRuntime": (
        "hyper_parallel.data.batching.packed_sequence_dataset_adapter.PackedSequenceDatasetAdapter", {},
    ),
}
