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
"""Model-owned forward installation for Qwen dense and MoE VLM CP."""

from functools import partial, wraps
from typing import Any

import torch
from torch.nn import functional

from hyper_parallel.distributed.encoder_parallel.contracts import EncoderFeature
from hyper_parallel.models import HyperAutoModelForImageTextToText
from hyper_parallel.models.qwen3_5_moe.vlm_cp import QwenVLMCPInputs
from hyper_parallel.models.qwen3_5_moe.adapter.encoder_parallel import QwenVisionBridge
from hyper_parallel.models.qwen3_5_moe.adapter.vlm_chunk_loss import _chunk_loss_forward


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


def install_qwen_vlm_cp_forward(model: Any, *, cp_enabled: bool = True) -> None:
    """Keep the HF decoder/head, replacing only image insertion and CP loss alignment.

    Args:
        model: Conditional-generation model receiving the batch forward adapter.
        cp_enabled: Whether the decoder uses context parallelism.
    """
    if getattr(model, "_hp_qwen_vlm_forward_installed", False):
        raise ValueError("Qwen VLM CP forward is already installed")
    original_forward = model.forward
    bridge = QwenVisionBridge(model)

    @wraps(original_forward)
    def forward(*args: Any, **kwargs: Any) -> Any:
        """Run the local decoder slice under the outer FSDP forward hooks."""
        geometry = kwargs.pop("vlm_cp_inputs", None)
        if geometry is None:
            return original_forward(*args, **kwargs)
        if bool((geometry.input_ids[~geometry.attention_mask.bool()] == model.config.image_token_id).any()):
            raise ValueError("Qwen VLM CP padding must not contain image placeholders")
        images = bridge.parse_images({**kwargs, "vlm_global_input_ids": geometry.input_ids})
        if geometry.cp_size == 1:
            if "chunk_loss_targets" in kwargs:
                return _chunk_loss_forward(model, **kwargs)
            return original_forward(*args, **kwargs)
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
            records, anchor = plan.encode_and_route(kwargs)
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
        forward_impl = original_forward
        chunked = "chunk_loss_targets" in kwargs
        if chunked:
            # Use the VLM loss adapter directly so either binding order is safe.
            forward_impl = partial(_chunk_loss_forward, model)
        output = forward_impl(
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

    model.forward = forward
    # Exercise the shared internal integration contract; no public equivalent exists.
    model._hp_vlm_cp_enabled = cp_enabled  # pylint: disable=protected-access
    model._hp_qwen_vlm_forward_installed = True  # pylint: disable=protected-access


def build_qwen_vlm(pretrained_model_name_or_path: str, *, distributed_setup: Any = None,
                   **kwargs: Any) -> Any:
    """Build a Qwen VLM and install its CP forward independently of batching.

    Args:
        pretrained_model_name_or_path: Model checkpoint path.
        distributed_setup: Shared parallel setup with the decoder CP degree.
        **kwargs: Arguments forwarded to the standard image-text model builder.

    Returns:
        The model-owned batch adapter, retaining the native forward for CP=1.
    """
    model = HyperAutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path, distributed_setup=distributed_setup, **kwargs,
    )
    if model.config.model_type not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError("build_qwen_vlm requires a Qwen dense or MoE multimodal model")
    cp_enabled = distributed_setup is not None and distributed_setup.mesh_context.cp_size > 1
    install_qwen_vlm_cp_forward(model, cp_enabled=cp_enabled)
    return model
