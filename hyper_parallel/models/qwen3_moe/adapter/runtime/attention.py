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
"""Qwen attention adapter using the common BNSD/TND FA contract."""

from typing import Any
import torch
from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward


def run_qwen3_moe_flash_attention(
    module: torch.nn.Module, query_states: torch.Tensor, key_states: torch.Tensor,
    value_states: torch.Tensor, attention_mask: torch.Tensor | None,
    dropout: float = 0.0, scaling: float | None = None, **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Forward packed lengths, causal/window options and masks without dropping metadata."""
    if attention_mask is not None and attention_mask.ndim == 4:
        # Preserve HF cache-capacity masks while sharing all actual FA semantics.
        attention_mask = attention_mask[..., :key_states.shape[2]]
    return npu_fusion_attention_forward(module, query_states, key_states, value_states,
                                        attention_mask, dropout=dropout, scaling=scaling, **kwargs)


def qwen3_moe_flash_attention_forward(module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
    """Call the concrete family module through its normal Module boundary."""
    return module(*args, **kwargs)
