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
"""Qwen3 dense mask convention for Ascend fused attention."""

from __future__ import annotations

from typing import Any

# This adapter targets the PyTorch-native Transformers Qwen3 implementation
# and intentionally shares its tensor and module contracts.
# pylint: disable=forbidden-backend-import
import torch
from torch import nn

_COMPRESSED_CAUSAL_MASK_SIZE = 2048
_COMPRESSED_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


def _get_compressed_causal_mask(device: torch.device) -> torch.Tensor:
    """Return the cached mask required by NPU left-up causal sparse mode."""
    mask = _COMPRESSED_CAUSAL_MASKS.get(device)
    if mask is None:
        mask = torch.triu(
            torch.ones(
                (_COMPRESSED_CAUSAL_MASK_SIZE, _COMPRESSED_CAUSAL_MASK_SIZE),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )
        _COMPRESSED_CAUSAL_MASKS[device] = mask
    return mask


def run_qwen3_flash_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run Qwen3 grouped-query attention with the Ascend fused kernel.

    Boolean masks use the Transformers convention, where ``True`` marks an
    allowed pair. The NPU kernel expects the inverse convention. Additive
    masks use zero for allowed positions and a nonzero value for blocked
    positions.

    Args:
        module: Owning attention module.
        query: Query states in BNSD layout.
        key: Key states in BNSD layout.
        value: Value states in BNSD layout.
        attention_mask: Optional Transformers-format attention mask.
        dropout: Attention dropout probability.
        scaling: Attention score scaling factor.
        **kwargs: Additional attention arguments accepted for HF compatibility.

    Returns:
        Attention output in BSND layout and ``None`` for attention weights.
    """
    # torch_npu is optional outside Ascend environments.
    import torch_npu  # pylint: disable=C0415

    del kwargs

    if attention_mask is None:
        attention_mask = _get_compressed_causal_mask(query.device)
        sparse_mode = 2
    else:
        if attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.logical_not(attention_mask).to(query.device)
        else:
            attention_mask = attention_mask.bool().to(query.device)
        sparse_mode = 0

    sparse_kwargs = {}
    if (
        sparse_mode == 0
        and getattr(module, "is_causal", True)
        and query.shape[-2] == key.shape[-2]
    ):
        sparse_kwargs["next_tockens"] = 0

    attention_output = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        head_num=query.shape[1],
        input_layout="BNSD",
        atten_mask=attention_mask,
        keep_prob=1 - dropout,
        scale=scaling,
        sparse_mode=sparse_mode,
        **sparse_kwargs,
    )[0]
    return attention_output.transpose(1, 2).contiguous(), None


__all__ = ["run_qwen3_flash_attention"]
