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
"""Parameter-sink attention function."""

from typing import Optional, Sequence

import torch  # pylint: disable=forbidden-backend-import
import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import


def sink_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    attention_mask: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    pre_tokens: int = 1048576,
    next_tokens: int = 0,
    keep_prob: float = 1.0,
    sparse_mode: int = 0,
    actual_seq_qlen: Optional[Sequence[int]] = None,
    actual_seq_kvlen: Optional[Sequence[int]] = None,
    sink_num: int = 0,
) -> torch.Tensor:
    """Compute NPU fused parameter-sink attention in TND layout.

    Args:
        query: Query tensor in TND layout.
        key: Key tensor in TND layout with sink tokens prepended.
        value: Value tensor in TND layout with sink tokens prepended.
        num_heads: Number of query attention heads.
        attention_mask: Optional attention mask.
        scale: Scaling factor applied to attention scores.
        pre_tokens: Number of preceding tokens considered by sparse attention.
        next_tokens: Number of following tokens considered by sparse attention.
        keep_prob: Dropout keep probability.
        sparse_mode: Sparse attention mode passed to the custom kernel.
        actual_seq_qlen: Optional cumulative query sequence lengths.
        actual_seq_kvlen: Optional cumulative key/value sequence lengths,
            excluding the prepended sink tokens.
        sink_num: Number of 64-token sink blocks.

    Returns:
        Attention output tensor.
    """
    return torch.ops.custom.npu_flash_attention_score_enhance(
        query,
        key,
        value,
        num_heads,
        pse=None,
        padding_mask=None,
        atten_mask=attention_mask,
        scale=scale,
        keep_prob=keep_prob,
        input_layout="TND",
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        inner_precise=0,
        sparse_mode=sparse_mode,
        prefix=[],
        sink_num=sink_num,
    )[0]
