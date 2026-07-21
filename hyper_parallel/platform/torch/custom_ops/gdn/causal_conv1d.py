# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Wenshuo Zhao
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Autograd wrapper for the NPU Triton causal depthwise Conv1d kernels."""

from typing import Optional

import torch

from .kernels.causal_conv1d import causal_conv1d_bwd_impl, causal_conv1d_fwd_impl


class CausalConv1dFunction(torch.autograd.Function):
    """Causal Conv1d with ``[B, T, D]`` input and ``[D, W]`` weights."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        num_heads: int,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        activation: Optional[str] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ):
        if weight.ndim != 2 or weight.shape[0] != x.shape[-1]:
            raise ValueError(
                "causal Conv1d expects weight [D, W] matching input [B, T, D], "
                f"got input={tuple(x.shape)}, weight={tuple(weight.shape)}."
            )
        kernel_weight = weight.transpose(0, 1).contiguous()
        placeholder = x.new_empty(0)
        ctx.save_for_backward(
            x,
            kernel_weight,
            bias if bias is not None else placeholder,
            residual if residual is not None else placeholder,
            initial_state if initial_state is not None else placeholder,
        )
        ctx.has_bias = bias is not None
        ctx.has_residual = residual is not None
        ctx.has_initial_state = initial_state is not None
        ctx.num_heads = num_heads
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens

        output, final_state = causal_conv1d_fwd_impl(
            x=x,
            weight=kernel_weight,
            H=num_heads,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            output_final_state=output_final_state,
        )
        output = output.permute(0, 2, 1, 3).reshape_as(x).contiguous()
        return output, final_state

    @staticmethod
    def backward(ctx, dy: torch.Tensor, d_final_state: Optional[torch.Tensor] = None):
        x, kernel_weight, bias, residual, initial_state = ctx.saved_tensors
        bias = bias if ctx.has_bias else None
        residual = residual if ctx.has_residual else None
        initial_state = initial_state if ctx.has_initial_state else None
        grad_output = dy.reshape(
            x.shape[0], x.shape[1], ctx.num_heads, x.shape[2] // ctx.num_heads
        ).permute(0, 2, 1, 3).contiguous()
        dx, dw, db, dr, dh0 = causal_conv1d_bwd_impl(
            x=x,
            dy=grad_output,
            H=ctx.num_heads,
            dht=d_final_state,
            weight=kernel_weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=ctx.activation,
            cu_seqlens=ctx.cu_seqlens,
        )
        return (
            dx,
            dw.transpose(0, 1).contiguous(),
            None,
            db,
            dr,
            dh0,
            None,
            None,
            None,
        )


def causal_conv1d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
):
    """Run fused causal depthwise Conv1d on an NPU tensor."""
    return CausalConv1dFunction.apply(
        x,
        weight,
        num_heads,
        bias,
        residual,
        initial_state,
        activation,
        cu_seqlens,
        output_final_state,
    )


__all__ = ["CausalConv1dFunction", "causal_conv1d_triton"]
