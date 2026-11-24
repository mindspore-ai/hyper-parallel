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
"""Qwen3.5 dense text and VLM identities with complete public vision and hybrid-attention families."""

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.checkpoint.export_policy import export_text_checkpoint
from hyper_parallel.components.modules import (
    AxialVisionTransformer,
    CausalLMHead,
    GatedDeltaNet,
    GatedGroupedQueryAttention,
    OffsetRMSNorm,
    RotaryEmbedding,
)
from hyper_parallel.components.modules.common.swiglu_mlp import SwiGLUMLP
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.distributed.context_parallel import wrappers as context_wrappers
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderParallelAdapterSpec
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_custom_model, register_model_adapter
from hyper_parallel.models.replacement import ModuleReplacementRequest


def get_replacement_rules(*, model: nn.Module, context: Mapping[str, Any]) -> tuple[ModuleReplacementRequest, ...]:
    """Declare actual hybrid layer roles and replace the entire visual boundary when present."""
    del context
    conditional = hasattr(model.config, "vision_config")
    config = model.config.text_config if conditional else model.config
    prefix = "model.language_model" if conditional else "model"
    rules = [
        ModuleReplacementRequest(
            match=(prefix + ".rotary_emb",), target=RotaryEmbedding, kwargs={"mrope_interleaved": True}
        ),
        ModuleReplacementRequest(
            match=(prefix + ".layers.*.mlp",), target=SwiGLUMLP, kwargs={"state_layout": "separate"}
        ),
        ModuleReplacementRequest(
            match=(
                prefix + ".layers.*.input_layernorm",
                prefix + ".layers.*.post_attention_layernorm",
                prefix + ".norm",
            ),
            target=OffsetRMSNorm,
        ),
        ModuleReplacementRequest(match=("lm_head",), target=CausalLMHead),
    ]
    full = tuple(
        f"{prefix}.layers.{index}.self_attn"
        for index, kind in enumerate(config.layer_types)
        if kind == "full_attention"
    )
    linear = tuple(
        f"{prefix}.layers.{index}.linear_attn"
        for index, kind in enumerate(config.layer_types)
        if kind == "linear_attention"
    )
    if full:
        rules.append(
            ModuleReplacementRequest(
                match=full, target=GatedGroupedQueryAttention, kwargs={"fuse_qkv": False, "use_fused_ops": False}
            )
        )
    if linear:
        rules.append(ModuleReplacementRequest(match=linear, target=GatedDeltaNet, kwargs={"ulysses_degree": None}))
    if conditional:
        rules.append(ModuleReplacementRequest(match=("model.visual",), target=AxialVisionTransformer))
    return tuple(rules)


def get_fsdp_wrap_modules(model: nn.Module) -> tuple[str, ...]:
    """Expose complete visual blocks and merger as independent FSDP execution units."""
    visual = getattr(model.model, "visual", None)
    if visual is None:
        return ()
    return tuple(f"model.visual.blocks.{index}" for index in range(len(visual.blocks))) + ("model.visual.merger",)


def get_fsdp_excluded_subtrees(model: nn.Module) -> tuple[str, ...]:
    """Keep the visual container outside generic language-decoder inference."""
    return ("model.visual",) if hasattr(model.model, "visual") else ()


def get_fsdp_execution_order(model: nn.Module, module_fqns: tuple[str, ...]) -> tuple[str, ...]:
    """Follow the source VLM's embedding-before-vision first-forward execution order."""
    del model
    embedding = "model.language_model.embed_tokens"
    return (
        (embedding,) + tuple(name for name in module_fqns if name != embedding)
        if embedding in module_fqns
        else module_fqns
    )


def get_encoder_parallel_spec() -> EncoderParallelAdapterSpec:
    """Declare complete public vision ownership and its dataset token contract."""
    # Optional vision support stays outside text-only specification discovery.
    from hyper_parallel.components.modules.vision.axial_vision_transformer import (  # pylint: disable=C0415
        AxialVisionEncoderRuntime,
        CPVisionAttention,
    )

    def bind(model: nn.Module) -> AxialVisionEncoderRuntime:
        """Bind public encoder execution to the architecture's image-token identity."""
        if isinstance(model.model, HSDPModule):
            raise NotImplementedError("Encoder execution requires child FSDP units rather than the complete model core")
        return AxialVisionEncoderRuntime(model.model.visual, model.config.image_token_id)

    return EncoderParallelAdapterSpec(("model.visual",), "model.visual.blocks.*.attn", CPVisionAttention, bind)


def get_context_parallel() -> Any:
    """Expose shared source-decoder metadata forwarding without another attention algorithm."""
    return context_wrappers


MODEL_SPEC = ModelAdapterSpec(
    architecture="Qwen3_5ForConditionalGeneration",
    encoder_parallel=get_encoder_parallel_spec,
    model_type="qwen3_5",
    replacements=get_replacement_rules,
    context_parallel=get_context_parallel,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
    fsdp_execution_order=get_fsdp_execution_order,
)
TEXT_SPEC = ModelAdapterSpec(
    architecture="Qwen3_5ForCausalLM",
    model_type="qwen3_5_text",
    replacements=get_replacement_rules,
    context_parallel=get_context_parallel,
    checkpoint_export=export_text_checkpoint,
)
register_model_adapter(MODEL_SPEC)
register_model_adapter(TEXT_SPEC)

register_custom_model("Qwen3_5ForCausalLM", "hyper_parallel.models.qwen3_5.modeling_qwen3_5", "Qwen3_5ForCausalLM")
register_custom_model(
    "Qwen3_5ForConditionalGeneration",
    "hyper_parallel.models.qwen3_5.modeling_qwen3_5",
    "Qwen3_5ForConditionalGeneration",
)


LEGACY_TARGETS = {
    "hyper_parallel.models.qwen3_5.adapter.distributed.context_parallel.qwen3_5_packed_decoder_wrapper": (
        "hyper_parallel.distributed.context_parallel.wrappers.packed_linear_decoder_wrapper", {},
    ),
    "hyper_parallel.models.qwen3_5.adapter.linear_attention_decoder.qwen3_5_linear_attention_decoder_wrapper": (
        "hyper_parallel.distributed.context_parallel.wrappers.packed_linear_decoder_wrapper", {},
    ),
    "hyper_parallel.models.qwen3_5.adapter.linear_attention_decoder.qwen3_5_packed_model_wrapper": (
        "hyper_parallel.distributed.context_parallel.wrappers.packed_linear_model_wrapper", {},
    ),
    "hyper_parallel.models.qwen3_5.adapter.data.runtime.Qwen35PackedRuntime": (
        "hyper_parallel.data.batching.packed_sequence_dataset_adapter.PackedSequenceDatasetAdapter", {},
    ),
}
