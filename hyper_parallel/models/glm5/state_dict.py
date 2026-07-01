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
"""State-dict adapter for GLM5 dense Phase-1 checkpoints."""
from typing import Dict, Optional

import torch

from hyper_parallel.models.glm5.checkpoint import load_hf_glm5_state_dict


class GLM5StateDictAdapter:
    """HF ↔ hyper state-dict adapter for the GLM5 dense training skeleton."""

    def load_hf_state_dict(
        self,
        weights_path: str,
        model_config,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        return load_hf_glm5_state_dict(
            weights_path,
            num_hidden_layers=model_config.num_hidden_layers,
            num_experts=getattr(model_config, "num_experts", None),
            dtype=dtype,
        )

    def save_hf_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        model_config,
    ) -> Dict[str, torch.Tensor]:
        del model_config
        return dict(state_dict)


__all__ = ["GLM5StateDictAdapter"]
