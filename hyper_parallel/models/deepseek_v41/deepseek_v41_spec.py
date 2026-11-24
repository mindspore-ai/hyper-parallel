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
"""Declare native V4.1 identity, architecture policies and public component lifecycle bindings."""

from collections.abc import Callable
from functools import partial
from typing import Any, Union

import torch

from hyper_parallel.components.modules.engram.host_engram_table import HostEngramTable
from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.distributed.encoder_parallel.attention import encoder_attention_checkpoint_contexts
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole
from hyper_parallel.models.deepseek_v41.shared_state import attention_role, configure_shared_state
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionTower


from hyper_parallel.components.optim.headwise import build_reshape_fn as build_headwise_reshape_fn
from hyper_parallel.models.adapter_spec import ModelAdapterSpec, RecomputePolicy
from hyper_parallel.models.registry import register_custom_model
from hyper_parallel.components.modules.engram.host_state import (
    build_engram_external_state,
)


# Keep decoder renames scoped to text layers: vision and Engram retain native names.
_V41_RENAMES = (
    ('^embed\\.weight$', 'embed_tokens.weight'),
    ('^head\\.weight$', 'lm_head.weight'),
    ('^norm\\.weight$', 'norm.weight'),
    ('^hc_head_fn$', 'hc_head.hc_fn'),
    ('^hc_head_base$', 'hc_head.hc_base'),
    ('^hc_head_scale$', 'hc_head.hc_scale'),
    ('^layers\\.(\\d+)\\.attn_norm\\.', 'layers.\\1.input_layernorm.'),
    ('^layers\\.(\\d+)\\.ffn_norm\\.', 'layers.\\1.post_attention_layernorm.'),
    ('^layers\\.(\\d+)\\.hc_attn_fn$', 'layers.\\1.attn_hc.fn'),
    ('^layers\\.(\\d+)\\.hc_attn_base$', 'layers.\\1.attn_hc.base'),
    ('^layers\\.(\\d+)\\.hc_attn_scale$', 'layers.\\1.attn_hc.scale'),
    ('^layers\\.(\\d+)\\.hc_ffn_fn$', 'layers.\\1.ffn_hc.fn'),
    ('^layers\\.(\\d+)\\.hc_ffn_base$', 'layers.\\1.ffn_hc.base'),
    ('^layers\\.(\\d+)\\.hc_ffn_scale$', 'layers.\\1.ffn_hc.scale'),
    ('^layers\\.(\\d+)\\.attn\\.', 'layers.\\1.self_attn.'),
    ('^layers\\.(\\d+)\\.ffn\\.', 'layers.\\1.mlp.'),
    ('^layers\\.(\\d+)\\.self_attn\\.attn_sink$', 'layers.\\1.self_attn.sinks'),
    ('^layers\\.(\\d+)\\.self_attn\\.indexer\\.wq_b\\.', 'layers.\\1.self_attn.indexer.q_b_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.wq_a\\.', 'layers.\\1.self_attn.q_a_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.wq_b\\.', 'layers.\\1.self_attn.q_b_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.wkv\\.', 'layers.\\1.self_attn.kv_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.wo_a\\.', 'layers.\\1.self_attn.o_a_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.wo_b\\.', 'layers.\\1.self_attn.o_b_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.q_norm\\.', 'layers.\\1.self_attn.q_a_norm.'),
    ('^layers\\.(\\d+)\\.mlp\\.shared_experts\\.w1\\.', 'layers.\\1.mlp.shared_experts.gate_proj.'),
    ('^layers\\.(\\d+)\\.mlp\\.shared_experts\\.w2\\.', 'layers.\\1.mlp.shared_experts.down_proj.'),
    ('^layers\\.(\\d+)\\.mlp\\.shared_experts\\.w3\\.', 'layers.\\1.mlp.shared_experts.up_proj.'),
)


def build_deepseek_v41_checkpoint_mapping() -> list[Any]:
    """Build model-owned V4.1 names using Transformers conversion operators.

    Upstream V4 rules may rename arbitrary ``.attn.``, ``.norm.`` and router
    biases. V4.1 retains those native names in vision, compressors and gates,
    so its mapping must not depend on upstream V4 pattern spelling or scope.

    Returns:
        Fresh name transforms and expert merge operations for the native model.
    """
    try:
        # Family registration is lazy and remains usable without Transformers installed.
        from transformers.core_model_loading import (  # pylint: disable=import-outside-toplevel
            Concatenate, MergeModulelist, WeightConverter, WeightRenaming,
        )
    except ImportError:
        return []

    mapping = [WeightRenaming(source, target) for source, target in _V41_RENAMES]
    mapping.extend([
        WeightConverter(
            source_patterns=["mlp.experts.*.w1.weight", "mlp.experts.*.w3.weight"],
            target_patterns="mlp.experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            source_patterns="mlp.experts.*.w2.weight",
            target_patterns="mlp.experts.down_proj",
            operations=[MergeModulelist(dim=0)],
        ),
    ])
    return mapping



def register_deepseek_v41_checkpoint_mapping() -> bool:
    """Register mappings for the production V4.1 classes."""
    try:
        from transformers.conversion_mapping import (  # pylint: disable=C0415
            register_checkpoint_conversion_mapping,
        )
    except ImportError:
        return False

    mapping = build_deepseek_v41_checkpoint_mapping()
    register_checkpoint_conversion_mapping("deepseek_v41", mapping, overwrite=True)
    register_checkpoint_conversion_mapping(
        "DeepseekV41ForCausalLM",
        mapping,
        overwrite=True,
    )
    # The root mapping already covers this nested base model. Registering an
    # empty class-specific mapping prevents its translated DeepseekV4Config
    # from adding a second, V4-only scoped conversion pipeline.
    register_checkpoint_conversion_mapping(
        "DeepseekV41Model",
        [],
        overwrite=True,
    )
    return True



def build_parameter_sharding_rules() -> list[tuple[Union[list[str], str], Any]]:
    """Return V4.1 parameter-role overrides."""
    return [
        (["image_start", "image_end", "image_newline"], ParamRole.REPLICATED),
        # The multimodal aligner runs on complete, CP-replicated image
        # features before their rows are inserted into local decoder shards.
        # Its w1/w2 pair is a pointwise projector, not an attention boundary;
        # force both linears to stay replicated so the generic colwise/rowwise
        # heuristic cannot require a mathematically invalid CP attention wrapper.
        (["aligner.w1", "aligner.w2"], ParamRole.REPLICATED),
        (["indexer.q_b_proj", "indexer.weights_proj"], ParamRole.COLWISE),
        (
            ["q_a_proj", "kv_proj", "compressor", "indexer.wk", "indexer.k_norm"],
            ParamRole.REPLICATED,
        ),
        (["engram", "attn_hc", "ffn_hc"], ParamRole.REPLICATED),
        (["dspark"], ParamRole.REPLICATED),
        (["q_b_proj", "o_a_proj", "sinks"], ParamRole.COLWISE),
        ("o_b_proj", ParamRole.ROWWISE),
    ]


def _get_visual_fsdp_wrap_modules(model: Any) -> tuple[str, ...]:
    """Return V4.1 visual execution units with bounded unshard size."""
    module_by_fqn = dict(model.named_modules())
    vision_blocks = tuple(
        module_fqn
        for module_fqn in module_by_fqn
        if module_fqn.startswith("model.vision.blocks.") and module_fqn.count(".") == 3
    )
    aligner = ("model.aligner",) if module_by_fqn.get("model.aligner") is not None else ()
    if not vision_blocks:
        return aligner
    return ("model.vision.patch_embed",) + vision_blocks + ("model.vision.norm",) + aligner


def get_fsdp_wrap_modules(model: Any) -> tuple[str, ...]:
    """Return visual units and homogeneous-mesh Engram child units.

    Args:
        model: Model being built or inspected.
    """
    module_by_fqn = dict(model.named_modules())
    engram_units = tuple(
        module_fqn
        for module_fqn in module_by_fqn
        if module_fqn.endswith(".engram.wkv")
        or (module_fqn.endswith(".engram.embed") and not isinstance(module_by_fqn[module_fqn], HostEngramTable))
    )
    return _get_visual_fsdp_wrap_modules(model) + engram_units


def get_fsdp_excluded_subtrees(model: Any) -> tuple[str, ...]:
    """Keep the ViT hierarchy out of HF decoder-container discovery.

    Args:
        model: Model being built or inspected.
    """
    module_by_fqn = dict(model.named_modules())
    return ("model.vision",) if module_by_fqn.get("model.vision") is not None else ()


def get_fsdp_execution_order(
    model: Any,
    module_fqns: tuple[str, ...],
) -> tuple[str, ...]:
    """Return visual and per-decoder child units in V4.1 forward order.

    Args:
        model: Model being built or inspected.
        module_fqns: Fully qualified module names in execution order.
    """
    visual_fqns = _get_visual_fsdp_wrap_modules(model)
    selected_fqns = set(module_fqns)
    execution_order = [module_fqn for module_fqn in visual_fqns if module_fqn in selected_fqns]
    layer_fqns = tuple(
        module_fqn
        for module_fqn in module_fqns
        if module_fqn.startswith("model.layers.") and module_fqn.count(".") == 2
    )
    for layer_fqn in layer_fqns:
        execution_order.append(layer_fqn)
        for suffix in ("engram.embed", "engram.wkv", "mlp.experts"):
            child_fqn = f"{layer_fqn}.{suffix}"
            if child_fqn in selected_fqns:
                execution_order.append(child_fqn)
    execution_order.extend(module_fqn for module_fqn in module_fqns if module_fqn not in execution_order)
    return tuple(execution_order)


_ATTENTION_ROLES = ("full", "reindex", "reuse")

#: The attention roles that recompute with their layer, per tier.
_RECOMPUTED_ROLES = {
    "none": (),
    "reuse": ("reuse",),
    "reuse_reindex": ("reuse", "reindex"),
    "all": _ATTENTION_ROLES,
}

#: The tier the model ships with: the layers that only consume cross-layer state
#: recompute, their producers stay outside the region and keep their
#: activations. Higher tiers recompute more attention for less memory, lower
#: tiers recompute less attention for less step time.
ATTENTION_RECOMPUTE_TIER = "reuse"


def _attention_stays_resident(module_fqn: str, module: Any, *, resident: tuple[str, ...]) -> bool:
    """Keep the attention of a role that does not recompute out of the region."""
    del module_fqn
    return attention_role(module) in resident


def _configure_recompute(model: Any, *, roles: tuple[str, ...]) -> None:
    """Configure decoder shared state and complete visual-block replay."""
    configure_shared_state(model, roles)
    for module in model.modules():
        if isinstance(module, DeepseekV41VisionTower):
            for block in module.blocks:
                if isinstance(block, CheckpointWrapper):
                    block.checkpoint_kwargs.update(
                        context_fn=encoder_attention_checkpoint_contexts,
                        # All ranks must replay every Ulysses/FSDP collective.
                        early_stop=False,
                    )


def build_recompute_policy() -> RecomputePolicy:
    """Return the V4.1 recompute regions used by normal training.

    Every decoder layer and every visual block is one region. The attention
    roles listed for :data:`ATTENTION_RECOMPUTE_TIER` recompute with their layer; the remaining
    roles stay outside the region, which is what the tiers differ in. The
    model-owned hook then pins the discrete selection of the recomputed index
    sources, so recomputing one neither chooses a different key set nor
    publishes its state a second time; published cross-layer entries stay alive
    until the graph drops the state in backward.
    """
    recomputed = _RECOMPUTED_ROLES[ATTENTION_RECOMPUTE_TIER]
    resident = tuple(role for role in _ATTENTION_ROLES if role not in recomputed)
    return RecomputePolicy(
        region_patterns=("model.layers.*", "model.vision.blocks.*"),
        exclude_predicate=(partial(_attention_stays_resident, resident=resident) if resident else None),
        on_applied=partial(_configure_recompute, roles=recomputed),
    )


def _load_expert_parallel():
    """Resolve finite legacy EP recipes from the complete model-owned MoE family."""
    # Native family construction has optional HMC dependencies; spec discovery must stay lazy.
    from hyper_parallel.models.deepseek_v41.components import (  # pylint: disable=C0415
        modality_moe,
    )

    return modality_moe


def _load_pipeline():
    """Load the model-owned stage protocol without importing the model eagerly."""
    from hyper_parallel.models.deepseek_v41.pipeline import get_pipeline_spec  # pylint: disable=C0415

    return get_pipeline_spec()


def _load_validation_spec():
    """Return model-owned parity and self-consistency declarations lazily."""
    from hyper_parallel.models.deepseek_v41.validation.model_validation_spec import (  # pylint: disable=C0415
        get_validation_spec,
    )

    return get_validation_spec()


def _load_encoder_parallel_spec():
    """Load DeepSeek construction rules and its model-bound bridge factory lazily."""
    from hyper_parallel.models.deepseek_v41.components.vision_transformer import (  # pylint: disable=C0415
        get_deepseek_v41_encoder_parallel_spec,
    )

    return get_deepseek_v41_encoder_parallel_spec()


MODEL_SPEC = ModelAdapterSpec(
    architecture="DeepseekV41ForCausalLM",
    model_type="deepseek_v41",
    expert_parallel=_load_expert_parallel,
    external_state=build_engram_external_state,
    sharding_rules=build_parameter_sharding_rules,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
    fsdp_execution_order=get_fsdp_execution_order,
    pipeline_parallel=_load_pipeline,
    recompute=build_recompute_policy,
    validation=_load_validation_spec,
    encoder_parallel=_load_encoder_parallel_spec,
)

register_custom_model(
    "DeepseekV41ForCausalLM",
    "hyper_parallel.models.deepseek_v41.modeling_deepseek_v41",
    "DeepseekV41ForCausalLM",
)
register_deepseek_v41_checkpoint_mapping()

__all__ = ["MODEL_SPEC"]


def build_muon_reshape_fn(
    attention_head_dim: int,
    indexer_head_dim: int,
) -> Callable[[str, torch.Tensor], list[torch.Tensor]]:
    """Build zero-copy query/key views using the configured model head widths.

    Args:
        attention_head_dim: Backbone and DSpark attention query/key head width.
        indexer_head_dim: Lightning Indexer query/key head width.

    Returns:
        Callable accepted by Muon's existing ``reshape_fn`` advanced option.
    """
    for width in (attention_head_dim, indexer_head_dim):
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("Head widths must be positive integers")
    widths = {name: attention_head_dim for name in ("q_b_proj", "kv_proj", "wq_b", "wkv")}
    widths.update({"indexer.q_b_proj": indexer_head_dim, "indexer.wk": indexer_head_dim})
    patterns = {
        prefix + name + ".weight": width for prefix in ("self_attn.", "*.self_attn.") for name, width in widths.items()
    }
    return build_headwise_reshape_fn(patterns)


# Preserve published data configuration targets during removal of model-owned adapter packages.
LEGACY_TARGETS = {
    ("hyper_parallel.models.deepseek_v41.adapter.distributed."
     "moe_engram_expert_parallel.deepseek_v41_engram_compute_fn"): (
        "hyper_parallel.components.modules.engram.parallel_engram.engram_ep_compute_fn", {},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel.": (
        "hyper_parallel.models.deepseek_v41.components.modality_moe.", {},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel.": (
        "hyper_parallel.models.deepseek_v41.components.vision_transformer.", {},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe.DeepseekV41MegaMoeExperts": (
        "hyper_parallel.components.modules.GroupedExperts", {"megakernel": True},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.distributed.mega_gate.DeepseekV41MegaGate": (
        "hyper_parallel.models.deepseek_v41.components.modality_router.ModalityTopKRouter",
        {"use_fused_ops": True},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc.DeepseekV41HyperMegaMhc": (
        "hyper_parallel.components.modules.PipelinedMhc", {"shifted": True, "use_fused_ops": True},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.data.image_processor.": (
        "hyper_parallel.models.deepseek_v41.dataset_preprocessing.image_processor.",
        {},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.data.text_transform.": (
        "hyper_parallel.models.deepseek_v41.dataset_preprocessing.text_transform.",
        {},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.data.encoding.": (
        "hyper_parallel.models.deepseek_v41.dataset_preprocessing.encoding.",
        {},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.data.runtime.": (
        "hyper_parallel.models.deepseek_v41.dataset_adapter.",
        {},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.data.transform_fn.": (
        "hyper_parallel.models.deepseek_v41.dataset_preprocessing.transform_fn.",
        {},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.data.processor.": (
        "hyper_parallel.models.deepseek_v41.dataset_preprocessing.processor.",
        {},
    ),
    "hyper_parallel.models.deepseek_v41.adapter.data.runtime.DeepseekV41Runtime": (
        "hyper_parallel.models.deepseek_v41.dataset_adapter.DeepseekV41DatasetAdapter",
        {},
    ),
}

LEGACY_TARGETS["hyper_parallel.models.deepseek_v41.adapter.validation."] = (
    "hyper_parallel.models.deepseek_v41.validation.",
    {},
)
