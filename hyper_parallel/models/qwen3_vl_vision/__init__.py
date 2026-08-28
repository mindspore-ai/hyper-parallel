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
"""Shared Qwen3-VL vision tower."""
from hyper_parallel.models.qwen3_vl_vision.model import (
    Qwen3VLMoeVisionAttention,
    Qwen3VLMoeVisionConfig,
    Qwen3VLMoeVisionDecoder,
    Qwen3VLMoeVisionMLP,
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeVisionOutput,
    Qwen3VLMoeVisionPatchEmbed,
    Qwen3VLMoeVisionPatchMerger,
    Qwen3VLMoeVisionRotaryEmbedding,
    Qwen3VLMoeVisionSdpaCore,
)

__all__ = [
    "Qwen3VLMoeVisionAttention",
    "Qwen3VLMoeVisionConfig",
    "Qwen3VLMoeVisionDecoder",
    "Qwen3VLMoeVisionMLP",
    "Qwen3VLMoeVisionModel",
    "Qwen3VLMoeVisionOutput",
    "Qwen3VLMoeVisionPatchEmbed",
    "Qwen3VLMoeVisionPatchMerger",
    "Qwen3VLMoeVisionRotaryEmbedding",
    "Qwen3VLMoeVisionSdpaCore",
]
