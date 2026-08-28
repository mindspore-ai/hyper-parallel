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
"""Shared neural-network layers and linear-attention operator wrappers."""

__all__ = [
    "GroupQueryAttention",
    "GatedDeltaNet",
    "SwiGLUMLP",
    "MoE",
    "MoEExperts",
    "SharedExpertMoE",
    "TopKRouter",
    "RMSNorm",
    "RMSNormGated",
    "RotaryEmbedding",
    "MultiModalRotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
]

from hyper_parallel.models.modules.attention import GroupQueryAttention
from hyper_parallel.models.modules.feed_forward import SwiGLUMLP
from hyper_parallel.models.modules.linear_attention import GatedDeltaNet
from hyper_parallel.models.modules.moe import MoE, MoEExperts, SharedExpertMoE, TopKRouter
from hyper_parallel.models.modules.rmsnorm import RMSNorm, RMSNormGated
from hyper_parallel.models.modules.rope import (
    MultiModalRotaryEmbedding,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
