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
"""HF ↔ hyper state-dict adapter for the VL-MoE model."""

__all__ = ["VLStateDictAdapter"]

from typing import Dict, Optional

import torch

from hyper_parallel.models.vl_moe.checkpoint import load_hf_vl_moe_state_dict


class VLStateDictAdapter:
    """Load/save adapter registered on the ``vl_moe`` ModelSpec."""

    @staticmethod
    def load_hf_state_dict(
        weights_path: str,
        model_config,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Read an HF safetensors checkpoint and return hyper-named tensors."""
        include_visual = bool(getattr(model_config, "vl", False))
        text_config = model_config.text_config if include_visual else model_config
        vision_config = getattr(model_config, "vision_config", None)
        return load_hf_vl_moe_state_dict(
            weights_path,
            num_hidden_layers=text_config.num_hidden_layers,
            include_visual=include_visual,
            vision_depth=getattr(vision_config, "depth", None),
            dtype=dtype,
        )

    @staticmethod
    def save_hf_state_dict(
        state_dict: Dict[str, torch.Tensor],
        model_config,
    ) -> Dict[str, torch.Tensor]:
        """Map hyper keys back to HF names."""
        del model_config
        hf_sd = {}
        for key, tensor in state_dict.items():
            if key.startswith("model.") or key == "lm_head.weight":
                hf_sd[key] = tensor
            elif key.startswith(("embed_tokens.", "layers.", "norm.")):
                hf_sd[f"model.language_model.{key}"] = tensor
        return hf_sd