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
"""Declare native text and visual parameter ownership and FSDP units."""
from __future__ import annotations

from typing import Any

from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import CP, EP, TP, ModuleShardingSpec


def build_parameter_sharding_rules() -> list[tuple[list[str], Any]]:
    """Return KDA parameter roles not covered by generic naming rules."""
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )

    return [
        (["vision_tower", "mm_projector", "self_attention_res_proj", "mlp_res_proj"], ParamRole.REPLICATED),
        (["q_a_proj", "kv_a_proj_with_mqa", "linear_qkv"], ParamRole.REPLICATED),
        (["block_sparse_moe.gate.weight"], ParamRole.MOE_GATE),
        (["q_conv1d", "k_conv1d", "v_conv1d"], ParamRole.COLWISE),
        (["A_log", "dt_bias"], ParamRole.COLWISE),
        (["f_a_proj", "g_a_proj"], ParamRole.REPLICATED),
        (["f_b_proj", "b_proj", "g_proj", "g_b_proj"], ParamRole.COLWISE),
    ]


def build_training_plan_overrides(config: Any) -> dict[str, ModuleShardingSpec]:
    """Declare parameter-only boundaries for K3 modules without generic templates.

    These branches retain replicated model-parallel ownership and remain
    FSDP-managed. Existing attention, MoE and norm templates are retained.

    Args:
        config: Text configuration or multimodal configuration containing
            text_config and optional vision_config.

    Returns:
        Module-pattern overrides for AttnRes parameters,
        replicated across TP, CP and EP; ordinary FSDP management remains active.
    """
    text = getattr(config, "text_config", config)
    prefix = "language_model." if hasattr(config, "text_config") else ""
    parameter_names = {}
    # The encoder provider gives the shared planner sole ownership of visual roots.
    if text.attn_res_block_size is not None:
        parameter_names.update({
            f"{prefix}model.layers.*.self_attention_res_proj": ("weight",),
            f"{prefix}model.layers.*.mlp_res_proj": ("weight",),
            f"{prefix}model.output_attn_res_proj": ("weight",),
        })
    return {
        pattern: ModuleShardingSpec(params={
            name: {TP: Replicate(), CP: Replicate(), EP: Replicate()} for name in names
        })
        for pattern, names in parameter_names.items()
    }

def get_fsdp_wrap_modules(model: Any) -> tuple[str, ...]:
    """Bound unshard size by visual blocks, the projector, and text decoder layers.

    Args:
        model: Native text model or multimodal model with language_model and
            vision_tower subtrees.

    Returns:
        Ordered module FQNs for explicit FSDP wrapping in forward execution order.
    """
    if not hasattr(model, "language_model"):
        return tuple(f"model.layers.{idx}" for idx in range(len(model.model.layers)))
    blocks = tuple(f"vision_tower.encoder.blocks.{idx}" for idx in range(len(model.vision_tower.encoder.blocks)))
    decoder = tuple(f"language_model.model.layers.{idx}" for idx in range(len(model.language_model.model.layers)))
    return ("vision_tower.patch_embed",) + blocks + ("vision_tower.encoder.final_layernorm", "mm_projector") + decoder


def get_fsdp_excluded_subtrees(model: Any) -> tuple[str, ...]:
    """Keep explicitly declared units out of automatic FSDP discovery.

    Args:
        model: Text or multimodal model whose wrapping units are listed explicitly.

    Returns:
        Text-backbone and, when present, vision roots excluded from automatic
        discovery; these roots are not exemptions from explicit FSDP wrapping.
    """
    subtrees = ("language_model.model",) if hasattr(model, "language_model") else ("model",)
    return (("vision_tower",) + subtrees) if hasattr(model, "vision_tower") else subtrees


def get_fsdp_execution_order(model: Any, module_fqns: tuple[str, ...]) -> tuple[str, ...]:
    """Visit visual units before language decoder units for every microbatch.

    Args:
        model: Text or multimodal model defining the declared wrapping order.
        module_fqns: Selected FSDP units, including separately wrapped expert units.

    Returns:
        Selected declared units with expert children after their decoder layer,
        followed by otherwise unrecognized FQNs in their supplied order.
    """
    selected = set(module_fqns)
    ordered = []
    for name in get_fsdp_wrap_modules(model):
        if name in selected:
            ordered.append(name)
        expert_name = f"{name}.block_sparse_moe.experts"
        if expert_name in selected:
            ordered.append(expert_name)
    return tuple(ordered) + tuple(name for name in module_fqns if name not in ordered)
