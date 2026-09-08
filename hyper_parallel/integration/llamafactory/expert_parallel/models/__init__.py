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

"""Model-specific expert-parallel patches."""

from .qwen3_vl_moe import QWEN3_VL_MOE_EXPERT_PARALLEL_PATCH
from .registry import ExpertParallelModelPatch

EXPERT_PARALLEL_MODEL_PATCHES = (
    QWEN3_VL_MOE_EXPERT_PARALLEL_PATCH,
)


def get_expert_parallel_model_patches() -> tuple[ExpertParallelModelPatch, ...]:
    """Return model-specific EP patches known to the LlamaFactory integration."""
    return EXPERT_PARALLEL_MODEL_PATCHES


__all__ = [
    "EXPERT_PARALLEL_MODEL_PATCHES",
    "ExpertParallelModelPatch",
    "get_expert_parallel_model_patches",
]
