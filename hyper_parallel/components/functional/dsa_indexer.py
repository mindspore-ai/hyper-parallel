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

import torch
from einops import rearrange

from hyper_parallel.tools.logging import get_logger

logger = get_logger()


def _cann_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    actual_seq_qlen: torch.Tensor,
    actual_seq_klen: torch.Tensor,
    sparse_count: int,
) -> torch.Tensor:
    """Call the ops-transformer LightningIndexer V2 with TND metadata."""
    cu_q = torch.cat((actual_seq_qlen.new_zeros(1), actual_seq_qlen))
    cu_k = torch.cat((actual_seq_klen.new_zeros(1), actual_seq_klen))
    kwargs = {
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "layout_q": "TND",
        "layout_k": "TND",
        "mask_mode": 3,
        "cmp_ratio": 1,
        "return_value": 0,
    }
    # The package is optional and is needed only for this selected CANN path.
    from cann_ops_transformer.ops import (  # pylint: disable=import-outside-toplevel
        lightning_indexer,
        lightning_indexer_metadata,
    )

    kwargs["metadata"] = lightning_indexer_metadata(
        query.shape[-2], key.shape[-2], query.shape[-1], sparse_count,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, batch_size=cu_q.numel() - 1,
        max_seqlen_q=int((cu_q[1:] - cu_q[:-1]).max().item()),
        max_seqlen_k=int((cu_k[1:] - cu_k[:-1]).max().item()),
        layout_q="TND", layout_k="TND", mask_mode=3, cmp_ratio=1,
    )
    indices, _ = lightning_indexer(
        query.contiguous(), key.contiguous(), weights.float().contiguous(), sparse_count, **kwargs,
    )
    return indices


def dsa_indexer(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    merge_weight: torch.Tensor,
    actual_seq_qlen: torch.Tensor,
    actual_seq_kvlen: torch.Tensor,
    sparse_count: int,
    *,
    backend: str = "cann",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate top-k token indices for DSA sparse attention.

    Args:
        index_query: Indexer query in BSND layout.
        index_key: Indexer key in BSND layout.
        merge_weight: Per-head indexer weights in BSN layout.
        actual_seq_qlen: Cumulative query lengths as an int32 NPU tensor.
        actual_seq_kvlen: Cumulative key lengths as an int32 NPU tensor.
        sparse_count: Number of token indices selected for each query.
        backend: ``"cann"`` (default) or the legacy ``"omni"`` backend.

    Returns:
        Top-k indices in ``[T, N_kv, K]`` int32 layout, followed by the query,
        key, and merge-weight tensors in TND, TND, and TN layouts. Positions
        without enough causal candidates are filled with ``-1``. The flattened
        tensors can be passed directly to :func:`dsa_kl_loss`.
    """
    if backend not in ("cann", "omni"):
        raise ValueError(f"backend must be 'cann' or 'omni', got {backend!r}")
    index_query_tnd = rearrange(index_query, "b s n d -> (b s) n d")
    index_key_tnd = rearrange(index_key, "b s n d -> (b s) n d")
    merge_weight_tnd = rearrange(merge_weight, "b s n -> (b s) n")
    # Load only the selected optional backend so either can be used independently.
    # pylint: disable=import-outside-toplevel
    if backend == "cann":
        topk_indices = _cann_lightning_indexer(
            index_query_tnd,
            index_key_tnd,
            merge_weight_tnd,
            actual_seq_qlen,
            actual_seq_kvlen,
            sparse_count,
        )
    else:
        logger.warning(
            "dsa_indexer backend='omni' is deprecated and will be removed in a future release; "
            "use backend='cann'."
        )
        import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import

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
    # pylint: enable=import-outside-toplevel
    return topk_indices, index_query_tnd, index_key_tnd, merge_weight_tnd
