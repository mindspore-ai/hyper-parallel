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
"""HF model orchestration composing the public terminal family for pre-shifted training loss."""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM as HFTextModel
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration as HFMultimodalModel

from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMOutput
from hyper_parallel.components.losses.router_load_balancing import token_choice_load_balancing_loss
from hyper_parallel.components.modules import CausalLMHead
from hyper_parallel.components.modules.vision.axial_vision_transformer import AxialVisionEncoderRuntime
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderFeature
from hyper_parallel.models import HyperAutoModelForImageTextToText
from hyper_parallel.models.qwen3_5_moe.vlm_cp import QwenVLMCPInputs


def select_local_image_features(records: list[EncoderFeature], geometry: QwenVLMCPInputs,
                               hidden: torch.Tensor, input_ids: torch.Tensor,
                               image_token_id: int) -> torch.Tensor:
    """Insert only image rows intersecting this rank's global token interval.

    Args:
        records: Image work items paired with their encoded features.
        geometry: Global token interval assigned to this CP rank.
        hidden: Local text embeddings receiving image features.
        input_ids: Local token IDs identifying image placeholders.
        image_token_id: Token ID used for image placeholders.

    Returns:
        Local embeddings with image placeholders replaced by image features.
    """
    rows = []
    for item, features in sorted(records, key=lambda record: record[0].decoder_token_start):
        begin = max(geometry.start, item.decoder_token_start)
        end = min(geometry.stop, item.decoder_token_start + item.output_rows)
        if begin < end:
            rows.append(features[begin - item.decoder_token_start:end - item.decoder_token_start])
    mask = input_ids.eq(image_token_id).unsqueeze(-1).expand_as(hidden)
    features = torch.cat(rows).to(hidden.dtype) if rows else hidden.new_empty((0, hidden.shape[-1]))
    if int(mask[..., 0].sum()) != features.shape[0]:
        raise ValueError("CP-local image placeholders do not match routed image features")
    return hidden.masked_scatter(mask, features)


def _validate_cp_forward_inputs(model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Check the supported uncached CP forward arguments."""
    if args or kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
        raise ValueError("Qwen VLM CP supports keyword-only uncached training")
    router_enabled = kwargs.get("output_router_logits")
    if router_enabled is None:
        router_enabled = getattr(getattr(model.config, "text_config", model.config), "output_router_logits", False)
    if router_enabled:
        raise NotImplementedError("Qwen VLM CP does not yet aggregate router auxiliary loss")
    if kwargs.get("return_dict") is False:
        raise ValueError("Qwen VLM CP requires return_dict=True")
    logits_to_keep = kwargs.get("logits_to_keep", 0)
    if isinstance(logits_to_keep, torch.Tensor) or logits_to_keep != 0:
        raise ValueError("Qwen VLM CP requires logits_to_keep=0")


def forward_encoder_plan(model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Compose owned encoder execution with HF's native multimodal fusion protocol."""
    plan = kwargs.pop("encoder_parallel_plan")
    if kwargs.get("image_grid_thw") is None and plan.decoder_items:
        kwargs["image_grid_thw"] = torch.tensor([item.metadata["grid"] for item in plan.decoder_items])
    if plan.decoder_layout.cp_size > 1 or plan.decoder_layout.sequence_parallel:
        raise ValueError("Partitioned Qwen encoder execution requires explicit vlm_cp_inputs")
    if args:
        if len(args) != 1 or "input_ids" in kwargs:
            raise ValueError("Encoder execution accepts one positional input_ids or keyword inputs")
        kwargs["input_ids"] = args[0]
    if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
        raise ValueError("Encoder parallel execution supports uncached training")
    if kwargs.get("mm_encoder_outputs") is not None:
        raise ValueError("An encoder plan cannot be combined with caller-provided media outputs")
    if kwargs.get("inputs_embeds") is None:
        kwargs["inputs_embeds"] = model.get_input_embeddings()(kwargs["input_ids"])
    payload = kwargs.pop("encoder_parallel_inputs", None)
    records, anchor = plan.encode_and_route(kwargs if payload is None else payload)
    kwargs.pop("pixel_values", None)
    if records:
        kwargs["mm_encoder_outputs"] = {
            "image": BaseModelOutputWithPooling(pooler_output=tuple(feature for _, feature in records)),
        }
    output = model._forward_unpartitioned(**kwargs)
    if anchor is not None:
        if getattr(output, "loss_sum", None) is not None:
            output.loss_sum = output.loss_sum + anchor
        elif getattr(output, "loss", None) is not None:
            output.loss = output.loss + anchor
        else:
            output.logits = output.logits + anchor
    return output


def forward_partitioned_vlm(model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Execute the architecture's class-owned media fusion over one decoder sequence window."""
    bridge = AxialVisionEncoderRuntime(model.model.visual, model.config.image_token_id)
    geometry = kwargs.pop("vlm_cp_inputs", None)
    if geometry is None:
        return model._forward_unpartitioned(*args, **kwargs)
    if bool((geometry.input_ids[~geometry.attention_mask.bool()] == model.config.image_token_id).any()):
        raise ValueError("Qwen VLM CP padding must not contain image placeholders")
    plan = kwargs.get("encoder_parallel_plan")
    if plan is not None:
        images = plan.decoder_items
        if kwargs.get("image_grid_thw") is None and images:
            kwargs["image_grid_thw"] = torch.tensor([item.metadata["grid"] for item in images])
    else:
        images = bridge.parse_images({**kwargs, "vlm_global_input_ids": geometry.input_ids})
    if geometry.cp_size == 1:
        if "encoder_parallel_plan" in kwargs:
            return forward_encoder_plan(model, args, kwargs)
        if "chunk_loss_targets" in kwargs:
            return model._forward_unpartitioned(**kwargs)
        return model._forward_unpartitioned(*args, **kwargs)
    token_types = geometry.token_type_ids
    if token_types is None:
        token_types = geometry.input_ids.eq(model.config.image_token_id).long()
    positions, _ = model.model.get_rope_index(
        input_ids=geometry.input_ids, mm_token_type_ids=token_types,
        image_grid_thw=kwargs.get("image_grid_thw"), attention_mask=geometry.attention_mask,
    )
    _validate_cp_forward_inputs(model, args, kwargs)
    kwargs["use_cache"] = False
    input_ids = kwargs.pop("input_ids")
    kwargs.pop("labels", None)
    kwargs.pop("mm_token_type_ids", None)
    kwargs.pop("attention_mask", None)
    plan = kwargs.pop("encoder_parallel_plan", None)
    hidden = model.get_input_embeddings()(input_ids)
    if plan is not None:
        payload = kwargs.pop("encoder_parallel_inputs", None)
        records, anchor = plan.encode_and_route(kwargs if payload is None else payload)
    elif images:
        visual = model.model.visual
        features = visual(kwargs["pixel_values"].to(visual.dtype),
                          grid_thw=kwargs["image_grid_thw"], return_dict=True).pooler_output
        records, offset = [], 0
        for item in images:
            records.append((item, features[offset:offset + item.output_rows]))
            offset += item.output_rows
        # A CP rank may own no image rows but must participate in vision backward/FSDP.
        anchor = features.sum() * 0.0
    else:
        records, anchor = [], None
    hidden = select_local_image_features(records, geometry, hidden, input_ids, model.config.image_token_id)
    for field in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "position_ids"):
        kwargs.pop(field, None)
    chunked = kwargs.get("chunk_loss_targets") is not None
    output = model._forward_unpartitioned(
        input_ids=None, inputs_embeds=hidden,
        position_ids=positions[..., geometry.start:geometry.stop].contiguous(),
        attention_mask={"full_attention": None, "linear_attention": None}, **kwargs,
    )
    if chunked:
        if anchor is not None:
            output.loss_sum = output.loss_sum + anchor
        return output
    # Sum/clamped count keeps empty-supervision ranks finite, including their gradients.
    targets = geometry.targets.reshape(-1)
    output.loss = functional.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                                           targets, ignore_index=-100, reduction="sum")
    output.loss = output.loss / targets.ne(-100).sum().clamp_min(1)
    if anchor is not None:
        output.loss = output.loss + anchor
    return output


def bind_qwen_vlm_parallel(model: Any, *, cp_enabled: bool = True) -> None:
    """Bind execution coordinates without replacing model or visual forward methods."""
    if not callable(getattr(model, "_forward_unpartitioned", None)):
        raise TypeError("Qwen VLM parallel execution requires the registered native conditional-generation class")
    if not isinstance(cp_enabled, bool):
        raise TypeError("cp_enabled must be bool")
    model._hp_vlm_cp_enabled = cp_enabled  # pylint: disable=protected-access


def build_qwen_vlm(pretrained_model_name_or_path: str, *, distributed_setup: Any = None,
                   **kwargs: Any) -> Any:
    """Build a registered native Qwen VLM and bind its class-owned CP protocol.

    Args:
        pretrained_model_name_or_path: Model checkpoint path.
        distributed_setup: Shared parallel setup with the decoder CP degree.
        **kwargs: Arguments forwarded to the standard image-text model builder.

    Returns:
        The native multimodal model, retaining source execution for CP=1.
    """
    model = HyperAutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path, distributed_setup=distributed_setup, **kwargs,
    )
    if model.config.model_type not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError("build_qwen_vlm requires a Qwen dense or MoE multimodal model")
    cp_enabled = distributed_setup is not None and distributed_setup.mesh_context.cp_size > 1
    bind_qwen_vlm_parallel(model, cp_enabled=cp_enabled)
    return model


def _forward_chunked(
    model: Any,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    chunk_size: int,
    ignore_index: int,
    kwargs: dict[str, Any],
) -> ChunkedCausalLMOutput:
    """Compose the actual backbone and terminal family; media fusion stays in the source architecture."""
    if kwargs.pop("labels", None) is not None or kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
        raise ValueError("chunked training requires pre-shifted targets without labels or KV cache")
    retained = kwargs.pop("logits_to_keep", 0)
    if isinstance(retained, torch.Tensor) or retained != 0 or kwargs.pop("return_dict", True) is False:
        raise ValueError("chunked training requires complete token coverage and return_dict=True")
    kwargs["use_cache"] = False
    config = getattr(model.config, "text_config", model.config)
    is_moe = hasattr(config, "num_experts")
    router_logits = kwargs.pop("output_router_logits", None)
    if is_moe:
        router_logits = config.output_router_logits if router_logits is None else router_logits
        kwargs["output_router_logits"] = router_logits
    outputs = model.model(return_dict=True, **kwargs)
    if not getattr(model.lm_head, "supports_chunked_cross_entropy", False):
        raise TypeError("chunked training requires the CausalLMHead family")
    result = model.lm_head(
        outputs.last_hidden_state, targets=targets, loss_mask=mask, chunk_size=chunk_size, ignore_index=ignore_index
    )
    aux_loss = None
    if is_moe and router_logits:
        aux_loss = token_choice_load_balancing_loss(
            outputs.router_logits, config.num_experts, config.num_experts_per_tok, kwargs.get("attention_mask")
        )
    return ChunkedCausalLMOutput(
        loss_sum=result.loss_sum,
        valid_token_count=result.valid_token_count,
        aux_loss=aux_loss,
        aux_loss_coef=float(getattr(config, "router_aux_loss_coef", 0.0)),
        router_logits=getattr(outputs, "router_logits", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
    )


class Qwen3_5MoeForCausalLM(HFTextModel):
    """Retain source construction and ordinary outputs, with explicit chunked training composition."""

    supports_chunked_loss = True

    @classmethod

    def _can_set_experts_implementation(cls) -> bool:
        """Retain the native source capability when HF inspects this orchestration module."""
        return HFTextModel._can_set_experts_implementation()

    def __init__(self, config: Any) -> None:
        """Install the terminal family while preserving source weight tying and initialization."""
        super().__init__(config)
        self.lm_head = CausalLMHead(module=self.lm_head)

    def forward(
        self,
        *args: Any,
        chunk_loss_targets: torch.Tensor | None = None,
        chunk_loss_mask: torch.Tensor | None = None,
        chunk_loss_chunk_size: int = 1024,
        chunk_loss_ignore_index: int = -100,
        **kwargs: Any,
    ) -> Any:
        """Dispatch normal source calls or compose backbone hidden states with pre-shifted loss targets."""
        if chunk_loss_targets is None:
            return super().forward(*args, **kwargs)
        if args:
            if len(args) != 1 or "input_ids" in kwargs:
                raise ValueError("chunked training accepts a single positional input_ids argument")
            kwargs["input_ids"] = args[0]
        return _forward_chunked(
            self, chunk_loss_targets, chunk_loss_mask, chunk_loss_chunk_size, chunk_loss_ignore_index, kwargs
        )


class Qwen3_5MoeForConditionalGeneration(HFMultimodalModel):
    """Retain source construction and ordinary outputs, with explicit chunked training composition."""

    supports_chunked_loss = True

    @classmethod

    def _can_set_experts_implementation(cls) -> bool:
        """Retain the native source capability when HF inspects this orchestration module."""
        return HFMultimodalModel._can_set_experts_implementation()

    def __init__(self, config: Any) -> None:
        """Install the terminal family while preserving source weight tying and initialization."""
        super().__init__(config)
        self.lm_head = CausalLMHead(module=self.lm_head)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute ordinary source/chunked calls or the owned post-media CP path."""
        if "vlm_cp_inputs" in kwargs:
            return forward_partitioned_vlm(self, args, kwargs)
        if "encoder_parallel_plan" in kwargs:
            return forward_encoder_plan(self, args, kwargs)
        return self._forward_unpartitioned(*args, **kwargs)

    def _forward_unpartitioned(
        self,
        *args: Any,
        chunk_loss_targets: torch.Tensor | None = None,
        chunk_loss_mask: torch.Tensor | None = None,
        chunk_loss_chunk_size: int = 1024,
        chunk_loss_ignore_index: int = -100,
        **kwargs: Any,
    ) -> Any:
        """Dispatch normal source calls or compose backbone hidden states with pre-shifted loss targets."""
        if chunk_loss_targets is None:
            return HFMultimodalModel.forward(self, *args, **kwargs)
        if args:
            if len(args) != 1 or "input_ids" in kwargs:
                raise ValueError("chunked training accepts a single positional input_ids argument")
            kwargs["input_ids"] = args[0]
        return _forward_chunked(
            self, chunk_loss_targets, chunk_loss_mask, chunk_loss_chunk_size, chunk_loss_ignore_index, kwargs
        )
