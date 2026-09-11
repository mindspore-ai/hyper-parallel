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
"""Register the DeepSeek-V4.1 validation model and adapter providers."""

from typing import Any

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import (
    register_custom_model,
    register_model_adapter,
)


def _load_replacements():
    """Return the model's replacement factories lazily."""
    from hyper_parallel.models.deepseek_v41.adapter import replacements  # pylint: disable=C0415

    return replacements


def _load_expert_parallel():
    """Return the model's EP forward factory lazily."""
    from hyper_parallel.models.deepseek_v41.adapter import expert_parallel  # pylint: disable=C0415

    return expert_parallel


def _load_context_parallel():
    """Return the model's shared-attention CP wrapper lazily."""
    from hyper_parallel.models.deepseek_v41.adapter import context_parallel  # pylint: disable=C0415

    return context_parallel


def _load_sharding_rules():
    """Return V4.1 parameter-role overrides."""
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )

    return [
        (["indexer.q_b_proj", "indexer.weights_proj"], ParamRole.COLWISE),
        (["q_a_proj", "kv_proj", "compressor", "indexer.wk", "indexer.k_norm"],
         ParamRole.REPLICATED),
        (["engram", "attn_hc", "ffn_hc"], ParamRole.REPLICATED),
        (["q_b_proj", "o_a_proj", "sinks"], ParamRole.COLWISE),
        ("o_b_proj", ParamRole.ROWWISE),
    ]


def _get_visual_fsdp_wrap_modules(model: Any) -> tuple[str, ...]:
    """Return V4.1 visual execution units with bounded unshard size."""
    module_by_fqn = dict(model.named_modules())
    vision_blocks = tuple(
        module_fqn
        for module_fqn in module_by_fqn
        if module_fqn.startswith("model.vision.blocks.")
        and module_fqn.count(".") == 3
    )
    aligner = (
        ("model.aligner",)
        if module_by_fqn.get("model.aligner") is not None
        else ()
    )
    return vision_blocks + aligner


def _get_fsdp_wrap_modules(model: Any) -> tuple[str, ...]:
    """Return visual units and mixed-mesh Engram execution units."""
    module_by_fqn = dict(model.named_modules())
    engram_units = tuple(
        module_fqn
        for module_fqn in module_by_fqn
        if module_fqn.endswith(".engram") or module_fqn.endswith(".engram.wkv")
    )
    return _get_visual_fsdp_wrap_modules(model) + engram_units


def _get_fsdp_excluded_subtrees(model: Any) -> tuple[str, ...]:
    """Keep the ViT hierarchy out of HF decoder-container discovery."""
    module_by_fqn = dict(model.named_modules())
    return ("model.vision",) if module_by_fqn.get("model.vision") is not None else ()


def _get_fsdp_execution_order(
        model: Any,
        module_fqns: tuple[str, ...],
) -> tuple[str, ...]:
    """Return visual and per-decoder child units in V4.1 forward order."""
    visual_fqns = _get_visual_fsdp_wrap_modules(model)
    selected_fqns = set(module_fqns)
    execution_order = [
        module_fqn for module_fqn in visual_fqns if module_fqn in selected_fqns
    ]
    layer_fqns = tuple(
        module_fqn
        for module_fqn in module_fqns
        if module_fqn.startswith("model.layers.")
        and module_fqn.count(".") == 2
    )
    for layer_fqn in layer_fqns:
        execution_order.append(layer_fqn)
        for suffix in ("engram", "engram.wkv", "mlp.experts"):
            child_fqn = f"{layer_fqn}.{suffix}"
            if child_fqn in selected_fqns:
                execution_order.append(child_fqn)
    execution_order.extend(
        module_fqn for module_fqn in module_fqns if module_fqn not in execution_order
    )
    return tuple(execution_order)


register_custom_model(
    "DeepseekV41ForCausalLM",
    "hyper_parallel.models.deepseek_v41.modeling_deepseek_v41",
    "DeepseekV41CroppedForCausalLM",
)

DEEPSEEK_V41_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="DeepseekV41ForCausalLM",
    model_type="deepseek_v41",
    replacements=_load_replacements,
    context_parallel=_load_context_parallel,
    expert_parallel=_load_expert_parallel,
    sharding_rules=_load_sharding_rules,
    fsdp_wrap_modules=_get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=_get_fsdp_excluded_subtrees,
    fsdp_execution_order=_get_fsdp_execution_order,
)
register_model_adapter(DEEPSEEK_V41_ADAPTER_SPEC)


__all__ = ["DEEPSEEK_V41_ADAPTER_SPEC"]
