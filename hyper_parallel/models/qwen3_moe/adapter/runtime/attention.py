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
"""attention: the Qwen3-MoE mask/cache/forward attention contract.

Holds only what is specific to Qwen3-MoE (adjust doc §5.2/§5.3): the
compressed causal mask required by the NPU left-up causal sparse mode and
the Transformers-to-Ascend mask convention conversion. The projection
pipeline (fused QKV, Q/K norm, RoPE, cache update, output projection) is
owned by the generic ``modules.GQAAttention``; the RMSNorm helper merged
into ``functional/rms_norm.py`` — no second family copy lives here.

This module stays importable on CPU-only checkouts: ``torch_npu`` is
imported lazily inside the kernel entry point.
"""

from __future__ import annotations

from typing import Any

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


def run_qwen3_moe_flash_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run Qwen3-MoE grouped-query attention with the Ascend Flash Attention kernel.

    Boolean model-facing masks follow the Transformers convention: ``True``
    marks an allowed query-key pair. This function performs the sole conversion
    to the Ascend kernel convention, where ``True`` marks a blocked pair.

    This is the ``attention_interface`` that
    ``replacements.replace_qwen3_moe_flash_attention`` hands to
    ``modules.GQAAttention`` so the replaced module keeps the Qwen mask/cache
    calling convention (compressed causal mask, ``sparse_mode`` selection).
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


def qwen3_moe_flash_attention_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the Qwen3-MoE attention forward contract through the replacement module.

    ``replace_qwen3_moe_flash_attention`` constructs a
    ``modules.GQAAttention`` whose forward implements this contract (adjust
    doc §5.3: the projection/output pipeline is taken over by the generic
    module; the kernel call delegates to ``run_qwen3_moe_flash_attention``).
    This delegate keeps the historical function-form entry for callers that
    still address the forward as a free function.

    Args:
        module: The replaced attention module (a ``modules.GQAAttention``).
        hidden_states: Input hidden states.
        position_embeddings: Cosine and sine rotary embedding tensors.
        attention_mask: Optional mask in the Transformers convention.
        past_key_values: Optional model cache passed through to projection.
        **kwargs: Additional fused attention arguments.

    Returns:
        The attention output and optional attention weights.
    """
    return module.forward(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        **kwargs,
    )
