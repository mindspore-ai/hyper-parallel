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
"""Model-specific context-parallel patches."""

from .qwen3_vl import QWEN3VL_MOE_CONTEXT_PARALLEL_PATCH
from .registry import ContextParallelModelPatch

CONTEXT_PARALLEL_MODEL_PATCHES = (QWEN3VL_MOE_CONTEXT_PARALLEL_PATCH,)


def get_context_parallel_model_patches() -> tuple[ContextParallelModelPatch, ...]:
    """Return model-specific CP patches known to the LlamaFactory integration."""
    return CONTEXT_PARALLEL_MODEL_PATCHES


__all__ = [
    "CONTEXT_PARALLEL_MODEL_PATCHES",
    "ContextParallelModelPatch",
    "get_context_parallel_model_patches",
]
