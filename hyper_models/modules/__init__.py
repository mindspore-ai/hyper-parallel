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
"""Public high-performance module interfaces."""

from hyper_models.modules.dsa_attention import DSAAttention, DeepseekV32DSAAttention
from hyper_models.modules.gqa_attention import GatedGQAAttention, GQAAttention
from hyper_models.modules.grouped_experts import GroupedExperts
from hyper_models.modules.mhc import MhcPostModule, MhcPostProcessModule, MhcPreModule
from hyper_models.modules.mla_attention import MLAAttention
from hyper_models.modules.rms_norm import OffsetRMSNorm, RMSNorm
from hyper_models.modules.shared_expert import SharedExpert
from hyper_models.modules.swiglu_mlp import SwiGLUMLP

__all__ = [
    "DeepseekV32DSAAttention",
    "DSAAttention",
    "GQAAttention",
    "GatedGQAAttention",
    "GroupedExperts",
    "MhcPostModule",
    "MhcPostProcessModule",
    "MhcPreModule",
    "MLAAttention",
    "OffsetRMSNorm",
    "RMSNorm",
    "SharedExpert",
    "SwiGLUMLP",
]
