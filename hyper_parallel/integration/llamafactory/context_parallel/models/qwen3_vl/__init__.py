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
"""Qwen3-VL context-parallel patches."""

from ..registry import ContextParallelModelPatch
from .qwen3vl_forward import (
    _enable_qwen3vl_moe_attention_patch,
    _enable_qwen3vl_moe_visual_embeds_patch,
    is_qwen3vl_moe_model,
)


def _prepare_qwen3vl_moe_context_parallel(model, hp_args, get_mesh) -> None:
    """Apply all Qwen3VL-MoE-specific CP patches under one model match."""
    _enable_qwen3vl_moe_visual_embeds_patch(model, hp_args)
    _enable_qwen3vl_moe_attention_patch(model, get_mesh(), hp_args)


QWEN3VL_MOE_CONTEXT_PARALLEL_PATCH = ContextParallelModelPatch(
    name="qwen3vl_moe",
    supports=is_qwen3vl_moe_model,
    prepare=_prepare_qwen3vl_moe_context_parallel,
)

__all__ = [
    "QWEN3VL_MOE_CONTEXT_PARALLEL_PATCH",
    "_enable_qwen3vl_moe_attention_patch",
    "_enable_qwen3vl_moe_visual_embeds_patch",
    "is_qwen3vl_moe_model",
]
