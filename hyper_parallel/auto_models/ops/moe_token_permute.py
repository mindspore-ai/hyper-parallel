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
"""NPU MoE token permutation function."""

import torch  # pylint: disable=forbidden-backend-import
import torch_npu


def moe_token_permute(
    tokens: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder tokens from token-major to expert-major order.

    Args:
        tokens: Token tensor whose first dimension is the token dimension.
        indices: Selected expert indices with shape ``[num_tokens, top_k]``.

    Returns:
        A tuple containing expert-major tokens and the sorting indices needed
        by :func:`moe_token_unpermute`.
    """
    return torch_npu.npu_moe_token_permute(tokens, indices)
