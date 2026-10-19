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

"""DeepSeek-V4.1 query/key head views for the existing Muon reshape hook."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch  # pylint: disable=forbidden-backend-import


def _reshape_heads(
        param_fqn: str,
        ns_input: torch.Tensor,
        *,
        attention_head_dim: int,
        indexer_head_dim: int,
) -> list[torch.Tensor]:
    """Expose query/key head views while preserving the working tensor's storage."""
    suffix = param_fqn.split("self_attn.", 1)[-1]
    if "self_attn." not in param_fqn:
        return [ns_input]
    if suffix in ("q_b_proj.weight", "kv_proj.weight", "wq_b.weight", "wkv.weight"):
        width = attention_head_dim
    elif suffix in ("indexer.q_b_proj.weight", "indexer.wk.weight"):
        width = indexer_head_dim
    else:
        return [ns_input]
    if ns_input.ndim != 2 or ns_input.shape[0] % width:
        raise ValueError(f"Head-wise Muon requires a 2D matrix with rows divisible by {width}: {param_fqn}")
    return [ns_input.view(-1, width, ns_input.shape[1])]


def build_reshape_fn(
        attention_head_dim: int,
        indexer_head_dim: int,
) -> Callable[[str, torch.Tensor], list[torch.Tensor]]:
    """Build zero-copy query/key views using the configured model head widths.

    Args:
        attention_head_dim: Backbone and DSpark attention query/key head width.
        indexer_head_dim: Lightning Indexer query/key head width.

    Returns:
        Callable accepted by Muon's existing ``reshape_fn`` advanced option.
    """
    for width in (attention_head_dim, indexer_head_dim):
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("Head widths must be positive integers")
    return partial(_reshape_heads, attention_head_dim=attention_head_dim, indexer_head_dim=indexer_head_dim)
