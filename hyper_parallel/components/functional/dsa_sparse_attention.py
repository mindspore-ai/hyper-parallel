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
"""DSA sparse-attention function."""

from typing import Tuple

import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import
from einops import rearrange

from hyper_parallel.components.functional.dsa_ops import _cann_dsa_sparse_attention


def dsa_sparse_attention(
    query_nope: torch.Tensor,
    compressed_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    scale: float,
    actual_seq_qlen: torch.Tensor,
    actual_seq_kvlen: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute NPU DSA sparse flash attention without parameter-sink rescaling.

    Args:
        query_nope: Absorbed MLA query in BSND layout.
        compressed_kv: Compressed MLA key/value in BSND layout.
        query_rope: Query rotary states in BSND layout.
        key_rope: Key rotary states in BSND layout.
        topk_indices: Int32 ``[T, N_kv, K]`` output from :func:`dsa_indexer`.
        scale: Attention score scaling factor.
        actual_seq_qlen: Cumulative query lengths as an int32 NPU tensor.
        actual_seq_kvlen: Cumulative key lengths as an int32 NPU tensor.

    Returns:
        Attention output in BSND layout, followed by the softmax maximum and
        sum required by :func:`dsa_sparse_indexer_kl_loss`. The output head dimension contains
        the compressed-KV dimension followed by zero padding for the RoPE
        dimension required by the sparse-attention backward operator.
    """
    query_tnd = rearrange(query_nope, "b s n d -> (b s) n d")
    key_tnd = rearrange(compressed_kv, "b s n d -> (b s) n d")
    query_rope_tnd = rearrange(query_rope, "b s n d -> (b s) n d")
    key_rope_tnd = rearrange(key_rope, "b s n d -> (b s) n d")
    output, softmax_max, softmax_sum = _cann_dsa_sparse_attention(
        query_tnd,
        key_tnd,
        key_tnd,
        topk_indices,
        scale,
        query_rope=query_rope_tnd,
        key_rope=key_rope_tnd,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_kvlen,
        layout="TND",
    )
    if query_rope.size(-1) > 0:
        output = F.pad(output, [0, query_rope.size(-1)])
    output = rearrange(output, "(b s) n d -> b s n d", b=query_nope.shape[0])
    return output, softmax_max, softmax_sum
