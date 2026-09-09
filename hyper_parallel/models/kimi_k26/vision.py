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
"""Reuse the official Kimi vision stack with trainable HF text and SDPA."""

from __future__ import annotations

from importlib import import_module
from types import MethodType
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import
from transformers import AutoModelForImageTextToText, PretrainedConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from transformers.utils import ModelOutput

from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import get_model_adapter, register_model_adapter


def _kimi_k26_sharding_rules() -> list[tuple[list[str], ParamRole]]:
    """Keep vision/projector on the dense mesh and reuse Deepseek MLA naming rules."""
    return [
        (["vision_tower.", "mm_projector."], ParamRole.REPLICATED),
        *get_model_adapter("deepseek_v3").sharding_rules(),
    ]


def _vision_patch_projection(module: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    """Express the one-kernel-per-patch Conv2d as an equivalent linear projection."""
    # Kimi already supplies individual patches; this avoids the NPU BF16
    # convolution weight-gradient kernel while preserving weights and shapes.
    # pylint: disable-next=not-callable
    return F.linear(pixels.flatten(1), module.weight.flatten(1), module.bias).unsqueeze(-1).unsqueeze(-1)


def _vision_position_embedding(
    module: torch.nn.Module, x: torch.Tensor, grid_thws: torch.Tensor,
) -> torch.Tensor:
    """Retain official spatial/temporal embedding math without forced compilation."""
    embeddings = []
    for frames, height, width in grid_thws.tolist():
        if frames > module.num_frames:
            raise ValueError("Image/video grid exceeds the configured temporal position table")
        if (height, width) == module.weight.shape[:-1]:
            spatial = module.weight.flatten(end_dim=1)
        else:
            spatial = F.interpolate(
                module.weight.permute(2, 0, 1).unsqueeze(0),
                size=(height, width), mode=module.interpolation_mode,
            ).squeeze(0).permute(1, 2, 0).flatten(end_dim=1)
        temporal = spatial if frames == 1 else spatial.unsqueeze(0).repeat(frames, 1, 1) + module.time_weight[:frames]
        embeddings.append(temporal.reshape(-1, temporal.shape[-1]))
    return x + torch.cat(embeddings)


def _vision_rotary_frequencies(
    module: torch.nn.Module, grid_thws: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute official interleaved x/y rotary angles using real arithmetic."""
    frequencies = module.theta_base ** (-torch.arange(0, module.dim, 4, device=device).float() / module.dim)
    angles = []
    for frames, height, width in grid_thws.tolist():
        positions = torch.arange(height * width, device=device)
        x_angles = torch.outer((positions % width).float(), frequencies)
        y_angles = torch.outer((positions // width).float(), frequencies)
        angles.append(torch.stack((x_angles, y_angles), dim=-1).flatten(-2).repeat(frames, 1))
    angles = torch.cat(angles).unsqueeze(1)
    return angles.cos(), angles.sin()


def _vision_attention(
    module: torch.nn.Module,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rope_freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Run noncausal SDPA independently for each packed image/video segment."""
    del max_seqlen
    query, key, value = module.wqkv(x).view(
        -1, 3, module.num_heads, module.hidden_size_per_attention_head,
    ).unbind(dim=1)
    cosine, sine = rope_freqs_cis
    rotated = []
    for tensor in (query, key):
        pairs = tensor.float().unflatten(-1, (-1, 2))
        real, imaginary = pairs.unbind(-1)
        rotated.append(torch.stack((real * cosine - imaginary * sine,
                                    real * sine + imaginary * cosine), dim=-1).flatten(-2).to(tensor.dtype))
    query, key = rotated[0], rotated[1]
    boundaries = cu_seqlens.tolist()
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        outputs.append(F.scaled_dot_product_attention(  # pylint: disable=not-callable
            query[start:end].transpose(0, 1).unsqueeze(0),
            key[start:end].transpose(0, 1).unsqueeze(0),
            value[start:end].transpose(0, 1).unsqueeze(0),
            dropout_p=0.0, is_causal=False,
        ).squeeze(0).transpose(0, 1).flatten(-2))
    return module.wo(torch.cat(outputs))


def register_kimi_k26_multimodal(config_path: str) -> type[PreTrainedModel]:
    """Register an HF VLM class composed from the local official model assets.

    Args:
        config_path: Directory containing official Kimi configuration/modeling Python files.

    Returns:
        A conditional-generation class reusing the official multimodal forward.
    """
    official_class = get_class_from_dynamic_module(
        "modeling_kimi_k25.KimiK25ForConditionalGeneration", config_path,
    )
    official = import_module(official_class.__module__)

    class KimiK26ForConditionalGeneration(official_class):
        """Official vision/projector/fusion with a native trainable text backbone."""

        _supports_sdpa = True

        def __init__(self, config: PretrainedConfig) -> None:
            """Compose all configured layers without constructing the remote text model."""
            official.KimiK25PreTrainedModel.__init__(self, config)
            self.vision_tower = official.MoonViT3dPretrainedModel(official.VisionTowerConfig(config.vision_config))
            self.vision_tower.encoder.gradient_checkpointing = False
            projector_config = official.ProjectorConfig(config.vision_config)
            projector_class = {"identity": official.IdentityMap, "mlp": official.MLP,
                               "patchmerger": official.PatchMergerMLP}[projector_config.mm_projector_type]
            self.mm_projector = (projector_class() if projector_config.mm_projector_type == "identity"
                                 else projector_class(projector_config))
            self.language_model = DeepseekV3ForCausalLM(config.text_config)
            self.post_init()
            self.vision_tower.to(dtype=self.language_model.dtype)
            self.mm_projector.to(dtype=self.language_model.dtype)
            self.vision_tower.patch_embed.proj.forward = MethodType(
                _vision_patch_projection, self.vision_tower.patch_embed.proj,
            )
            self.vision_tower.patch_embed.pos_emb.forward = MethodType(
                _vision_position_embedding, self.vision_tower.patch_embed.pos_emb,
            )
            self.vision_tower.encoder.rope_2d.get_freqs_cis = MethodType(
                _vision_rotary_frequencies, self.vision_tower.encoder.rope_2d,
            )
            for block in self.vision_tower.encoder.blocks:
                block.attention_qkvpacked = MethodType(_vision_attention, block)

        def forward(
            self, *args: Any, image_grid_thw: torch.Tensor | None = None,
            grid_thws: torch.Tensor | None = None, **kwargs: Any,
        ) -> tuple | ModelOutput:
            """Accept the shared VLM batch grid name as well as the official alias."""
            return super().forward(
                *args, grid_thws=grid_thws if grid_thws is not None else image_grid_thw, **kwargs,
            )

    AutoModelForImageTextToText.register(official_class.config_class, KimiK26ForConditionalGeneration, exist_ok=True)
    register_model_adapter(ModelAdapterSpec(
        architecture="KimiK26ForConditionalGeneration", model_type="kimi_k25",
        sharding_rules=_kimi_k26_sharding_rules,
    ))
    return KimiK26ForConditionalGeneration
