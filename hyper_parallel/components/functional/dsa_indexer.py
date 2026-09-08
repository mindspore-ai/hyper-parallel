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
"""DSA sparse-attention indexer."""

from typing import Tuple

import torch  # pylint: disable=forbidden-backend-import
from einops import rearrange
import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import


def dsa_indexer(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    merge_weight: torch.Tensor,
    actual_seq_qlen: torch.Tensor,
    actual_seq_kvlen: torch.Tensor,
    sparse_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate top-k token indices for DSA sparse attention.

    Args:
        index_query: Indexer query in BSND layout.
        index_key: Indexer key in BSND layout.
        merge_weight: Per-head indexer weights in BSN layout.
        actual_seq_qlen: Cumulative query lengths as an int32 NPU tensor.
        actual_seq_kvlen: Cumulative key lengths as an int32 NPU tensor.
        sparse_count: Number of token indices selected for each query.

    Returns:
        Top-k indices in ``[T, N_kv, K]`` int32 layout, followed by the query,
        key, and merge-weight tensors in TND, TND, and TN layouts. Positions
        without enough causal candidates are filled with ``-1``. The flattened
        tensors can be passed directly to :func:`dsa_kl_loss`.
    """
    index_query_tnd = rearrange(index_query, "b s n d -> (b s) n d")
    index_key_tnd = rearrange(index_key, "b s n d -> (b s) n d")
    merge_weight_tnd = rearrange(merge_weight, "b s n -> (b s) n")
    topk_indices, _ = torch.ops.custom.npu_lightning_indexer_enhance(
        index_query_tnd,
        index_key_tnd,
        merge_weight_tnd,
        actual_seq_lengths_query=actual_seq_qlen,
        actual_seq_lengths_key=actual_seq_kvlen,
        block_table=None,
        layout_query="TND",
        layout_key="TND",
        sparse_count=sparse_count,
        sparse_mode=3,
        return_value=False,
    )
    return topk_indices, index_query_tnd, index_key_tnd, merge_weight_tnd
