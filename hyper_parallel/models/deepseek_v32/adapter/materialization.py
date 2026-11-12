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
"""Rebuild DeepSeek-V3.2 rotary buffers after empty materialization."""

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32RotaryEmbedding

from hyper_parallel.models.materialization import register_rebuildable_buffer


def register_rotary_materialization(model: nn.Module) -> None:
    """Retain native CPU RoPE initialization outside storage replaced by to_empty."""
    for module in model.modules():
        if not isinstance(module, DeepseekV32RotaryEmbedding):
            continue
        with torch.device("cpu"):
            reference = DeepseekV32RotaryEmbedding(module.config)
        for name, value in reference.named_buffers(recurse=False):
            register_rebuildable_buffer(
                module, name, value=value.to(dtype=getattr(module, name).dtype), persistent=False,
            )


__all__ = ["register_rotary_materialization"]
