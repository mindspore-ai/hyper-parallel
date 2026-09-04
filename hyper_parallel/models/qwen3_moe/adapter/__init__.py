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
"""qwen3_moe.adapter: public handwritten extension surface for Qwen3-MoE.

Only the supported extension contract is exported here; underscore-named
functions/classes/modules remain internal implementation details (adjust
doc §7.1). M2 provides the structure replacements onto the generic
``modules`` entries; the attention mask/cache contract lives in
``attention.py`` and the CP/EP distributed rules in ``distributed/``
(M3).
"""

from hyper_parallel.models.qwen3_moe.adapter.registration import (
    QWEN3_MOE_ADAPTER_SPEC,
)
from hyper_parallel.models.qwen3_moe.adapter.replacements import (
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
