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
"""HF registry bridge for cumulative-length NPU attention, without model-forward rewrites."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AttentionInterface, AttentionMaskInterface

from hyper_parallel.components.functional.npu_fusion_attention import (
    npu_fusion_attention_forward,
    resolve_packed_sequence_lengths,
)


def _packed_attention_mask(*, attention_mask: torch.Tensor | None = None, **kwargs: Any) -> None:
    """Leave document visibility to TND, rejecting conflicting custom mask policies."""
    if attention_mask is not None or kwargs.get("use_vmap", False):
        raise ValueError("Packed NPU attention requires document lengths without an explicit or custom mask")


def _packed_attention_forward(
    module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    attention_mask: torch.Tensor | None, **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Keep native HF attention usable with the same packed kernel as family members."""
    if attention_mask is not None:
        raise ValueError("Packed NPU attention requires document lengths without an explicit mask")
    query_lengths, _ = resolve_packed_sequence_lengths(
        kwargs, query.shape[0] * query.shape[2], key.shape[0] * key.shape[2])
    if query_lengths is None:
        raise ValueError("Packed NPU attention requires global cumulative document lengths")
    return npu_fusion_attention_forward(module, query, key, value, None, **kwargs)


def register_npu_packed_attention() -> str:
    """Register HF attention and mask callbacks before constructing the model.

    Use as the nested factory for ``model.attn_implementation`` and pair with
    ``PackedAttentionRuntime``. Both native HF and replaced attention consume
    cumulative document lengths. No dense mask is created by HF, and existing
    eager/SDPA registrations are unchanged. Cached decoding is outside this
    packed-training contract.

    Returns:
        The registered Transformers attention implementation name.
    """
    name = "hyper_npu_packed"
    AttentionInterface.register(name, _packed_attention_forward)
    AttentionMaskInterface.register(name, _packed_attention_mask)
    return name
