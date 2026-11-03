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
"""Rebuild Qwen dense and MoE rotary buffers after empty materialization."""

import torch
from torch import nn

from hyper_parallel.models.materialization import register_rebuildable_buffer


_TEXT_ROTARY_CLASSES = {"Qwen3_5TextRotaryEmbedding", "Qwen3_5MoeTextRotaryEmbedding"}
_VISION_ROTARY_CLASSES = {"Qwen3_5VisionRotaryEmbedding", "Qwen3_5MoeVisionRotaryEmbedding"}


def register_rotary_materialization(model: nn.Module) -> None:
    """Retain native CPU RoPE initialization outside storage replaced by to_empty."""
    for module in model.modules():
        class_name = type(module).__name__
        with torch.device("cpu"):
            if class_name in _TEXT_ROTARY_CLASSES:
                reference = type(module)(module.config, device="cpu")
            elif class_name in _VISION_ROTARY_CLASSES:
                reference = type(module)(module.dim, module.theta)
            else:
                continue
        for name, value in reference.named_buffers(recurse=False):
            register_rebuildable_buffer(
                module, name, value=value.to(dtype=getattr(module, name).dtype), persistent=False,
            )
