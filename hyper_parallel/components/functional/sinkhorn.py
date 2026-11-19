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
"""Sinkhorn normalization functions used by manifold hyper-connections."""

from typing import Any, Tuple

import torch

try:
    import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import
except ImportError:
    omni_training_custom_ops = None


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
        if omni_training_custom_ops is None:
            raise ImportError("NPU Sinkhorn requires omni_training_custom_ops")
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


class _TorchNpuSinkhorn(torch.autograd.Function):
    """Native Ascend normalization and derivative independent of Pre collapse."""

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
        """Keep only the native normalization caches for backward."""
        # Torch-npu is optional outside the native NPU path.
        import torch_npu  # pylint: disable=import-outside-toplevel

        output, norms, sums = torch_npu.npu_mhc_sinkhorn(logits, out_flag=1, eps=eps, num_iters=iters)
        ctx.save_for_backward(norms, sums)
        return output

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Differentiate logits using the native Sinkhorn backward."""
        # Torch-npu is optional outside the native NPU path.
        import torch_npu  # pylint: disable=import-outside-toplevel

        return torch_npu.npu_mhc_sinkhorn_backward(grad.contiguous(), *ctx.saved_tensors), None, None


def sinkhorn(h_res: torch.Tensor, sinkhorn_iters: int = 20, eps: float = 1e-6,
             *, backend: str = "omni") -> torch.Tensor:
    """Normalize a square mixing matrix with the NPU Sinkhorn operator.

    Args:
        h_res: Residual mixing matrix.
        sinkhorn_iters: Number of normalization iterations.
        eps: Numerical stability epsilon.
        backend: Legacy "omni" (default) or native "torch_npu" binding.

    Returns:
        The normalized residual mixing matrix.
    """
    if backend == "torch_npu":
        return _TorchNpuSinkhorn.apply(h_res, sinkhorn_iters, eps)
    if backend != "omni":
        raise ValueError("Sinkhorn backend must be omni or torch_npu")
    return _Sinkhorn.apply(h_res, sinkhorn_iters, eps)


def sinkhorn_knopps(h_res: torch.Tensor, sinkhorn_iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
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
