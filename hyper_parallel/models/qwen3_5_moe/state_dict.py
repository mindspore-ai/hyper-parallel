# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""HF ↔ hyper state-dict adapter for the Qwen3.5-MoE family.

Splits HF's fused per-expert tensors into hyper's per-expert layout so
``BaseTrainer._load_weights`` can stay generic. Registered on the
``qwen3_5_moe`` ModelSpec.
"""

__all__ = ["Qwen3_5MoeStateDictAdapter"]

# pylint: disable=C0103  # HF transformers class-name convention (Qwen3_5*)
import re
from typing import Dict, Optional

import torch

from hyper_parallel.models.qwen3_5_moe.checkpoint import load_hf_qwen3_5_moe_state_dict


class Qwen3_5MoeStateDictAdapter:
    """State-dict adapter for Qwen3.5-MoE (35B-A3B and friends)."""

    @staticmethod
    def load_hf_state_dict(
        weights_path: str,
        model_config,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Read an HF safetensors checkpoint and return a hyper-named state dict."""
        return load_hf_qwen3_5_moe_state_dict(
            weights_path,
            num_experts=model_config.num_experts,
            hidden_size=model_config.hidden_size,
            moe_intermediate_size=model_config.moe_intermediate_size,
            num_hidden_layers=model_config.num_hidden_layers,
            dtype=dtype,
        )

    @staticmethod
    def save_hf_state_dict(
        state_dict: Dict[str, torch.Tensor],
        model_config,
    ) -> Dict[str, torch.Tensor]:
        """Convert hyper state_dict to HF format for checkpoint export.

        Inverse of load_hf_state_dict: repacks per-expert tensors back to
        HF packed format and remaps keys to HF convention.
        """
        del model_config  # unused (HF format keeps packed expert layout intact)
        hf_sd = {}

        for key, tensor in state_dict.items():
            if key == "lm_head.weight":
                hf_sd[key] = tensor
                continue
            # Packed-experts pass-through with prefix remap.
            match = re.match(
                r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$",
                key,
            )
            if match:
                layer_idx = int(match.group(1))
                kind = match.group(2)
                hf_sd[f"model.language_model.layers.{layer_idx}.mlp.experts.{kind}"] = tensor
                continue
            if key.startswith("model."):
                tail = key[len("model."):]
                hf_sd[f"model.language_model.{tail}"] = tensor
            else:
                hf_sd[key] = tensor

        return hf_sd
