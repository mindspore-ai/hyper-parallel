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
"""Sinkhorn normalization of mHC residual mixing logits."""

from typing import Any, Protocol, Tuple

import torch

from ..factory import _select_for_call
from ..registry import OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol


class SinkhornFn(Protocol):
    """Approximately balance square residual-mixing logits using Sinkhorn iterations.

    Start with row softmax plus eps, normalize columns once, then alternate
    row and column normalization for the remaining sinkhorn_iters - 1 rounds.
    Each denominator includes eps, so row and column sums are approximate.
    The native reference uses the input dtype; logits remain differentiable.
    """

    def __call__(self, x: torch.Tensor, sinkhorn_iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
        """Return the normalized matrix.

        Args:
            x: Square mixing logits [..., streams, streams].
            sinkhorn_iters: Number of normalization rounds, normally at least one.
            eps: Stability term added after softmax and to normalization denominators.

        Returns:
            Balanced mixing matrices with the input shape.
        """


SINKHORN = OpSpec[SinkhornFn](config_key='sinkhorn')


def sinkhorn(x: torch.Tensor, sinkhorn_iters: int = 20, eps: float = 1e-06) -> torch.Tensor:
    """Return the normalized matrix.

    Args:
        x: Square mixing logits [..., streams, streams].
        sinkhorn_iters: Number of normalization rounds, normally at least one.
        eps: Stability term added after softmax and to normalization denominators.

    Returns:
        Balanced mixing matrices with the input shape.

    Notes:
        Approximately balance square residual-mixing logits using Sinkhorn iterations.

        Start with row softmax plus eps, normalize columns once, then alternate
        row and column normalization for the remaining sinkhorn_iters - 1 rounds.
        Each denominator includes eps, so row and column sums are approximate.
        The native reference uses the input dtype; logits remain differentiable.
    """
    operation = _select_for_call(SINKHORN, device=x.device)
    return operation(x, sinkhorn_iters, eps)


class _Sinkhorn(torch.autograd.Function):
    """Autograd bridge for the NPU Sinkhorn custom operator."""

    @staticmethod
    def forward(
        ctx: Any,
        h_res: torch.Tensor,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Run the NPU Sinkhorn forward operator."""
        output, norm_out, sum_out = torch.ops.custom.npu_sinkhorn(
            h_res,
            out_flag=1,
            eps=eps,
            num_iters=sinkhorn_iters,
        )
        ctx.save_for_backward(norm_out, sum_out)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """Run the NPU Sinkhorn backward operator."""
        norm_out, sum_out = ctx.saved_tensors
        grad_h_res = torch.ops.custom.npu_sinkhorn_grad(grad_output, norm_out, sum_out)
        grads = [grad_h_res, None, None]
        return tuple(grads)


def native_sinkhorn(h_res: torch.Tensor, sinkhorn_iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a square mixing matrix with the portable PyTorch algorithm.

    Args:
        h_res: Residual mixing logits.
        sinkhorn_iters: Number of normalization iterations.
        eps: Numerical stability epsilon.

    Returns:
        The normalized residual mixing matrix.
    """
    h_res = h_res.softmax(-1) + eps
    col_sum = h_res.sum(-2, keepdim=True)
    h_res = h_res / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = h_res.sum(-1, keepdim=True)
        h_res = h_res / (row_sum + eps)
        col_sum = h_res.sum(-2, keepdim=True)
        h_res = h_res / (col_sum + eps)
    return h_res


def _load_omni() -> SinkhornFn:
    require_package("omni_training_custom_ops")
    require_symbol(torch.ops.custom, "npu_sinkhorn")
    require_symbol(torch.ops.custom, "npu_sinkhorn_grad")
    return _Sinkhorn.apply


def register_sinkhorn(registry: OpRegistry) -> None:
    """Register basic Torch normalization and Omni's fused implementation."""
    registry.register_family("sinkhorn", specs=(SINKHORN,))
    registry.register_impl(SINKHORN, name="native", devices=("cpu", "cuda", "npu"), load=lambda: native_sinkhorn)
    registry.register_impl(SINKHORN, name="omni", devices=("npu",), load=_load_omni)
