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
"""Fused FP32 row dot and BF16 probability multiplication for the expert VJP."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _probability_vjp(unweighted, activation, probability, grad_probability, grad_activation,
                     rows, width: tl.constexpr, BLOCK: tl.constexpr, ROW_BLOCK: tl.constexpr,
                     SCALE_ACTIVATION: tl.constexpr):
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    col = tl.arange(0, BLOCK)
    offsets = row[:, None] * width + col[None, :]
    mask = (row[:, None] < rows) & (col[None, :] < width)
    values = tl.load(unweighted + offsets, mask, other=0).to(tl.float32)
    activated = tl.load(activation + offsets, mask, other=0).to(tl.float32)
    score = tl.load(probability + row, row < rows, other=0).to(tl.float32)
    dp = tl.sum(values * activated, axis=1)
    tl.store(grad_probability + row, dp, row < rows)
    tl.store(grad_activation + offsets, values * score[:, None], mask)
    if SCALE_ACTIVATION:
        tl.store(activation + offsets, activated * score[:, None], mask)


def probability_vjp(unweighted: torch.Tensor, activation: torch.Tensor,
                    probabilities: torch.Tensor, *, scale_activation: bool = False
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dp and dA without full FP32 row matrices.

    Args:
        unweighted: Unweighted activation cotangent [received, intermediate].
        activation: Matching contiguous activation matrix.
        probabilities: Contiguous routing weights [received].
        scale_activation: Overwrite activation with p*A after computing dp from
            its original values. The caller must own this consumable buffer,
            with no other users or aliases that require its original contents.

    Returns:
        Probability and activation cotangents. By default activation is unchanged.
        This primitive supports only explicit first-order VJPs.
    """
    if (unweighted.shape != activation.shape or unweighted.ndim != 2
            or not unweighted.is_contiguous() or not activation.is_contiguous()
            or probabilities.shape != (unweighted.shape[0],) or not probabilities.is_contiguous()):
        raise ValueError("Probability VJP requires contiguous matching row matrices and scalar scores.")
    grad_probability = torch.empty_like(probabilities)
    grad_activation = torch.empty_like(unweighted)
    if unweighted.shape[0]:
        block = triton.next_power_of_2(unweighted.shape[1])
        row_block = min(4, max(1, 8192 // block))
        _probability_vjp[(triton.cdiv(unweighted.shape[0], row_block),)](
            unweighted, activation, probabilities, grad_probability, grad_activation,
            unweighted.shape[0], unweighted.shape[1], block, row_block, scale_activation,
        )
    return grad_probability, grad_activation
