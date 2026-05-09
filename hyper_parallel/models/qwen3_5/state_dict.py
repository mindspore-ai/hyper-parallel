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
"""HF ↔ hyper state-dict adapter for the dense Qwen3.5 family.

Owns the model-specific HF tensor renaming so ``BaseTrainer._load_weights``
can stay generic. Registered on the ``qwen3_5`` ModelSpec.
"""
# pylint: disable=C0103  # HF transformers class-name convention (Qwen3_5*)
from typing import Dict, Optional

import torch

from hyper_parallel.models.qwen3_5.checkpoint import load_hf_qwen3_5_state_dict


class Qwen3_5StateDictAdapter:
    """State-dict adapter for the dense Qwen3.5 family (Base + Instruct)."""

    def load_hf_state_dict(
        self,
        weights_path: str,
        model_config,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Read an HF safetensors checkpoint and return a hyper-named state dict."""
        return load_hf_qwen3_5_state_dict(
            weights_path,
            num_hidden_layers=model_config.num_hidden_layers,
            dtype=dtype,
        )

    def save_hf_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        model_config,
    ) -> Dict[str, torch.Tensor]:
        """Convert hyper state_dict to HF format for checkpoint export.

        Maps hyper tensor names back to HF Qwen3.5 convention:
        - model.embed_tokens.weight → model.language_model.embed_tokens.weight
        - model.layers.{i}.* → model.language_model.layers.{i}.*
        - model.norm.weight → model.language_model.norm.weight
        - lm_head.weight → lm_head.weight (unchanged)
        """
        del model_config  # name mapping is config-independent
        hf_sd = {}
        for key, tensor in state_dict.items():
            if key == "lm_head.weight":
                hf_sd[key] = tensor
            elif key.startswith("model."):
                tail = key[len("model."):]
                hf_sd[f"model.language_model.{tail}"] = tensor
            else:
                hf_sd[key] = tensor
        return hf_sd

__all__ = ["Qwen3_5StateDictAdapter"]
