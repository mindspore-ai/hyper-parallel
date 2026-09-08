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
"""qwen3_moe.adapter.distributed: Qwen3-MoE CP/EP rule providers.

New home (adjust doc §5.1) of the Qwen3-MoE context-parallel wrappers and
the expert-parallel compute archetype factory. This package is the flat
export surface the model adapter spec exposes; the generic CP collectives,
the EP combine skeleton and the router adapters stay in
``auto_models/distributed/{context_parallel,expert_parallel}``.

The CP wrappers come in two families: the fused wrappers
(``context_parallel.py``) wrap the ``modules.GQAAttention`` built by
``replace_qwen3_moe_flash_attention`` and only swap its
``attention_interface``; the async wrappers (``context_parallel_async.py``,
moved out of the generic ``wrappers.py`` in M5) target the original
HuggingFace-structure attention module and rewrite its whole forward.
"""

from hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel import (
    qwen3_moe_flash_attention_cp_mask_wrapper,
    qwen3_moe_flash_attention_cp_wrapper,
    qwen3_moe_flash_attention_ulysses_cp_wrapper,
)
from hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel_async import (
    qwen3_moe_async_colossal_cp_wrapper,
    qwen3_moe_async_hybrid_cp_wrapper,
    qwen3_moe_async_ulysses_cp_wrapper,
)
from hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel import (
    qwen3moe_ep_compute_fn,
)

__all__ = [
    "qwen3_moe_async_colossal_cp_wrapper",
    "qwen3_moe_async_hybrid_cp_wrapper",
    "qwen3_moe_async_ulysses_cp_wrapper",
    "qwen3_moe_flash_attention_cp_mask_wrapper",
    "qwen3_moe_flash_attention_cp_wrapper",
    "qwen3_moe_flash_attention_ulysses_cp_wrapper",
    "qwen3moe_ep_compute_fn",
]
