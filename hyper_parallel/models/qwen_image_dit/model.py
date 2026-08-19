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
"""Real Qwen-Image DiT model for diffusion training."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import QwenImageTransformer2DModel
from torch import nn


@dataclass
class DiTOutput:
    """DiT model output compatible with BaseTrainer loss extraction."""
    loss: Optional[torch.Tensor] = None
    sample: Optional[torch.Tensor] = None


class QwenImageDiT(nn.Module):
    """Real Qwen-Image DiT model with layer truncation for validation.

    Loads the actual QwenImageTransformer2DModel from HuggingFace,
    but truncates transformer blocks to specified num_layers.
    """

    def __init__(self, config, init_device="npu"):
        super().__init__()
        # Support both dict and object configs
        if isinstance(config, dict):
            cfg = config
        else:
            cfg = {
                "model_name": getattr(config, "model_name", "Qwen/Qwen-Image"),
                "subfolder": getattr(config, "subfolder", "transformer"),
                "num_layers": getattr(config, "num_layers", 2),
            }

        model_name = cfg.get("model_name", "Qwen/Qwen-Image")
        subfolder = cfg.get("subfolder", "transformer")
        num_layers = cfg.get("num_layers", 2)

        # Load real model config from HuggingFace
        model_config = QwenImageTransformer2DModel.load_config(
            model_name, subfolder=subfolder
        )
        model_config["num_layers"] = num_layers

        # Initialize weights on the target device to use device RNG (NPU vs CPU),
        # matching VeOmni's with torch.device(init_device) context so that
        # seed=42 produces identical initial weights across frameworks.
        with torch.device(init_device):
            self.model = QwenImageTransformer2DModel.from_config(model_config)


        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Model params: {total_params:.1f}M")

    def forward(self, latent, timestep, condition, target_noise=None, **kwargs):  # pylint: disable=unused-argument
        """Forward pass.

        Args:
            latent: (B, C, H, W) VAE latent tensor.
            timestep: (B,) int64 diffusion timestep.
            condition: (B, seq_len, cond_dim) text/image condition.
            target_noise: (B, C, H, W) optional target noise for loss.
            **kwargs: absorbs framework-specific args (e.g. use_cache).

        Returns:
            DiTOutput with loss (if target_noise provided) and sample.
        """
        b, c, h, w = latent.shape

        # Convert (B, C, H, W) -> (B, H*W, C) for QwenImageTransformer2DModel
        latent_flat = latent.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # img_shapes for RoPE position encoding
        # Format: [[(1, H, W)]] — matches VeOmni's _normalize_img_shapes output
        img_shapes = [[(1, h, w)]] * b

        # Forward through real model
        out = self.model(
            hidden_states=latent_flat,
            timestep=timestep,
            encoder_hidden_states=condition,

            img_shapes=img_shapes,
        )

        # Convert output back to (B, C, H, W)
        pred_noise = out.sample  # (B, H*W, C)
        pred_noise_spatial = pred_noise.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # Compute loss if target provided
        if target_noise is not None:
            loss = F.mse_loss(pred_noise_spatial, target_noise)
            return DiTOutput(loss=loss, sample=pred_noise_spatial)

        return DiTOutput(sample=pred_noise_spatial)
