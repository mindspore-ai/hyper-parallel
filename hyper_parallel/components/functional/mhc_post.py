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
"""Manifold hyper-connection post-processing functions."""

from typing import Any, Tuple

import torch

from hyper_parallel.tools.logging import get_logger

logger = get_logger()


class _MhcPost(torch.autograd.Function):
    """Autograd bridge for the legacy omni NPU MHC post operator."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        """Run the legacy omni NPU MHC post forward operator."""
        # The package is optional and is needed only for the legacy omni backend.
        # pylint: disable=import-outside-toplevel
        try:
            import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import
        except ImportError as error:
            raise ImportError("MHC backend='omni' requires omni_training_custom_ops") from error
        # pylint: enable=import-outside-toplevel
        ctx.save_for_backward(x, h_res, h_out, h_post)
        return torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post(
            x,
            h_res,
            h_out,
            h_post,
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the legacy omni NPU MHC post backward operator."""
        x, h_res, h_out, h_post = ctx.saved_tensors
        grad_x, grad_h_res, grad_h_out, grad_h_post = torch.ops.custom.npu_ai_infra_mhc_post_grad(
            grad_output.contiguous(),
            x,
            h_res,
            h_out,
            h_post,
        )
        return grad_x, grad_h_res, grad_h_out, grad_h_post


class _TorchNpuMhcPost(torch.autograd.Function):
    """First-order bridge for the native Ascend Post and weighted reduction."""

    @staticmethod
    def forward(ctx: Any, streams: torch.Tensor, combine: torch.Tensor,
                output: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Load the optional torch-npu binding only when selected."""
        import torch_npu  # pylint: disable=import-outside-toplevel

        ctx.save_for_backward(streams, combine, output, post)
        return torch_npu.npu_mhc_post(streams, combine, output, post)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Propagate gradients through the original streams and coefficients."""
        # Torch-npu is an optional dependency for the explicitly selected backend.
        import torch_npu  # pylint: disable=import-outside-toplevel

        return torch_npu.npu_mhc_post_backward(grad.contiguous(), *ctx.saved_tensors)


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    num_stream: int,
    *,
    backend: str = "cann",
) -> torch.Tensor:
    """Combine transformed and residual streams after an MHC-wrapped block.

    Args:
        x: Output of the wrapped block.
        residual: Flattened residual streams.
        h_post: Per-stream output mixing coefficients.
        h_res: Residual stream mixing matrix.
        num_stream: Number of residual streams.
        backend: ``"cann"`` (default) calls the differentiable CANN
            ops-transformer ``mhc_post`` operator and requires fp16/bf16
            ``x``/``residual`` with fp32 ``h_post``/``h_res``; the legacy
            ``"omni"`` backend is deprecated. CPU execution keeps the eager
            fallback. ``"torch_npu"`` selects native Ascend bindings.

    Returns:
        Flattened mixed residual streams.
    """
    if backend not in ("cann", "omni", "torch_npu"):
        raise ValueError(f"backend must be 'cann', 'omni' or 'torch_npu', got {backend!r}")
    x_shape = x.size()
    residual = residual.reshape(x_shape[0], x_shape[1], num_stream, -1)
    if x.device.type == "npu":
        if backend == "torch_npu":
            return _TorchNpuMhcPost.apply(residual, h_res, x, h_post).flatten(2)
        # Load only the selected optional backend so either can be used independently.
        # pylint: disable=import-outside-toplevel
        if backend == "cann":
            from cann_ops_transformer.ops import mhc_post as cann_mhc_post

            # CANN argument order: residual streams, residual matrix, block
            # output, post coefficients.
            return cann_mhc_post(residual, h_res, x, h_post).flatten(2)
        logger.warning(
            "mhc_post backend='omni' is deprecated and will be removed in a future release; "
            "use backend='cann'."
        )
        return _MhcPost.apply(residual, h_res, x, h_post).flatten(2)
    # pylint: enable=import-outside-toplevel
    output = h_post.unsqueeze(-1) * x.unsqueeze(-2)
    output = output + torch.sum(
        h_res.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return output.flatten(2).to(x.dtype)


def weighted_stream_sum(streams: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Aggregate with arbitrary pre coefficients using native Post on NPU.

    The first destination column contains weights and all other columns are
    zero. Returning only that destination implements a weighted stream sum
    without materializing the FP32 product of streams and coefficients.

    Args:
        streams: Residual streams shaped [B,S,N,H].
        weights: FP32 pre coefficients shaped [B,S,N], possibly from an earlier boundary.

    Returns:
        Weighted sum shaped [B,S,H] in the input dtype; CPU uses Torch.
    """
    if streams.ndim != 4 or weights.shape != streams.shape[:-1] or weights.dtype != torch.float32:
        raise ValueError("Weighted stream sum requires [B,S,N,H] streams and FP32 [B,S,N] weights")
    if streams.device.type == "cpu":
        return (streams.float() * weights.unsqueeze(-1)).sum(-2).to(streams.dtype)
    if streams.device.type != "npu" or streams.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Native weighted stream sum requires BF16/FP16 NPU streams")
    count = streams.shape[-2]
    combine = torch.nn.functional.pad(weights.unsqueeze(-1), (0, count - 1))
    zero_output = streams.new_zeros((*streams.shape[:-2], streams.shape[-1]))
    mixed = _TorchNpuMhcPost.apply(streams, combine, zero_output, torch.zeros_like(weights))
    return mixed[..., 0, :].contiguous()
