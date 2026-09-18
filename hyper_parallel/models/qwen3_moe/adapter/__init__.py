# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Public Qwen3-MoE adapter surface.

Module conversion lives in ``conversion/``, model-forward and loss contracts
in ``runtime/``, and CP/EP implementations in ``distributed/``.
"""

from hyper_parallel.models.qwen3_moe.adapter.registration import (
    QWEN3_MOE_ADAPTER_SPEC,
)
from hyper_parallel.models.qwen3_moe.adapter.conversion.module_replacement import (
    replace_qwen3_moe_flash_attention,
    replace_qwen3_moe_grouped_experts,
    replace_qwen3_moe_rms_norm,
)

__all__ = [
    "QWEN3_MOE_ADAPTER_SPEC",
    "replace_qwen3_moe_flash_attention",
    "replace_qwen3_moe_grouped_experts",
    "replace_qwen3_moe_rms_norm",
]
