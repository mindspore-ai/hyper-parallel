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
"""Local fused attention boundary shared by K3 text and media adapters."""

from __future__ import annotations

import importlib
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.dtensor.dtensor import DTensor


def npu_attention(owner: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  allowed: torch.Tensor | None, *, causal: bool, scaling: float,
                  dropout: float = 0.0, sequence_ends: tuple[int, ...] | None = None) -> torch.Tensor:
    """Return BSND attention while preserving explicit mask and segment semantics.

    Args:
        owner: Attention module passed to the shared fusion-attention adapter.
        query: Local NPU queries with shape ``[B, N, S_q, D_qk]``.
        key: Local NPU keys with shape ``[B, N, S_k, D_qk]``.
        value: Local NPU values with shape ``[B, N, S_k, D_v]``.
        allowed: Optional attention mask; boolean True marks visible keys.
        causal: Whether each query is restricted to its causal context.
        scaling: Multiplier applied to query-key attention scores.
        dropout: Attention dropout probability.
        sequence_ends: Optional cumulative segment ends shared by queries and
            keys, isolating independent segments in flattened packed inputs.

    Returns:
        Attention values with shape ``[B, S_q, N, D_v]``.
    """
    if any(isinstance(tensor, DTensor) for tensor in (query, key, value)):
        raise TypeError("K3 fused attention requires local tensors; parallel adapters own communication")
    if query.device.type != "npu":
        raise ValueError("K3 npu_fusion attention requires Ascend NPU tensors")
    # The shared helper imports torch_npu, which remains optional for eager users.
    module = importlib.import_module("hyper_parallel.components.functional.npu_fusion_attention")
    options = {"is_causal": causal, "next_tokens": 0 if causal else 1048576}
    if sequence_ends is not None:
        options.update(actual_seq_qlen=sequence_ends, actual_seq_kvlen=sequence_ends)
    return module.npu_fusion_attention_forward(
        owner, query, key, value, allowed, dropout=dropout, scaling=scaling, **options,
    )[0]
