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
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Keep the imported kernel structure aligned with upstream FLA.
# pylint: disable=line-too-long,missing-public-type-hints,missing-public-docstring
# pylint: disable=missing-function-docstring,missing-module-docstring,unused-argument
# pylint: disable=unsupported-binary-operation,invalid-name,disallowed-name
# pylint: disable=consider-using-from-import,import-outside-toplevel,no-else-return
# pylint: disable=no-member,use-dict-literal,global-statement,unsubscriptable-object
# pylint: disable=invalid-unary-operand-type,nested-min-max,used-before-assignment

"""Beta-sigmoid gate kernels adapted for triton-ascend on Ascend NPU."""

import torch
import triton
import triton.language as tl

from .ascend_ub_manager import ASCEND_MAX_GRID_DIM, compute_activation_block_size


def _beta_sigmoid_launch_config(
    n_elements: int,
    is_backward: bool = False,
) -> tuple[tuple[int], int]:
    block = compute_activation_block_size(
        n_elements,
        is_backward,
        max_grid=ASCEND_MAX_GRID_DIM,
    )
    return (triton.cdiv(n_elements, block),), block


@triton.jit
def fused_beta_sigmoid_fwd_kernel(
    x,
    y,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    b_x = tl.load(x + offs, mask=mask, other=0).to(tl.float32)
    b_y = scale * tl.sigmoid(b_x)
    tl.store(y + offs, b_y.to(y.dtype.element_ty), mask=mask)


@triton.jit
def fused_beta_sigmoid_bwd_kernel(
    x,
    dy,
    dx,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    b_x = tl.load(x + offs, mask=mask, other=0).to(tl.float32)
    b_dy = tl.load(dy + offs, mask=mask, other=0).to(tl.float32)
    b_y = tl.sigmoid(b_x)
    b_dx = b_dy * scale * b_y * (1.0 - b_y)
    tl.store(dx + offs, b_dx.to(dx.dtype.element_ty), mask=mask)


@torch.compiler.disable
def fused_beta_sigmoid_fwd_npu(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x, dtype=torch.float32)
    n_elements = x.numel()
    grid, block = _beta_sigmoid_launch_config(n_elements, is_backward=False)
    fused_beta_sigmoid_fwd_kernel[grid](
        x,
        y,
        scale,
        n_elements,
        BLOCK_SIZE=block,
    )
    return y


@torch.compiler.disable
def fused_beta_sigmoid_bwd_npu(
    x: torch.Tensor,
    dy: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    x = x.contiguous()
    dy = dy.contiguous()
    dx = torch.empty_like(x)
    n_elements = x.numel()
    grid, block = _beta_sigmoid_launch_config(n_elements, is_backward=True)
    fused_beta_sigmoid_bwd_kernel[grid](
        x,
        dy,
        dx,
        scale,
        n_elements,
        BLOCK_SIZE=block,
    )
    return dx
