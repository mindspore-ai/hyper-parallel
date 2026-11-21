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

from importlib import import_module
from typing import Any, Tuple

import torch

from hyper_parallel.tools.logging import get_logger

logger = get_logger()


class _CannMhcPost(torch.autograd.Function):
    """Keep CANN forward and backward behind dispatcher/Meta implementations."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, h_res: torch.Tensor,
        h_out: torch.Tensor, h_post: torch.Tensor,
    ) -> torch.Tensor:
        """Save inputs and dispatch the CANN forward operator.

        Args:
            ctx: Autograd context.
            x: Residual streams.
            h_res: Residual mixing matrix.
            h_out: Block output.
            h_post: Output mixing coefficients.
        """
        # Import the optional package for schema and Meta registration only.
        import_module("cann_ops_transformer.ops.mhc_post")
        ctx.save_for_backward(x, h_res, h_out, h_post)
        return torch.ops.cann_ops_transformer.mhc_post.default(x, h_res, h_out, h_post)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch backward so fake capture never enters the raw extension.

        Args:
            ctx: Autograd context with saved forward inputs.
            grad_output: Gradient of the mixed residual streams.
        """
        import_module("cann_ops_transformer.ops.mhc_post_backward")
        return torch.ops.cann_ops_transformer.mhc_post_backward.default(
            grad_output.contiguous(), *ctx.saved_tensors,
        )


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
        """Run the legacy omni NPU MHC post forward operator.

        Args:
            ctx: Autograd context.
            x: Residual streams.
            h_res: Residual mixing matrix.
            h_out: Block output.
            h_post: Output mixing coefficients.
        """
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
        """Run the legacy omni NPU MHC post backward operator.

        Args:
            ctx: Autograd context with saved forward inputs.
            grad_output: Gradient of the mixed residual streams.
        """
        x, h_res, h_out, h_post = ctx.saved_tensors
        grad_x, grad_h_res, grad_h_out, grad_h_post = torch.ops.custom.npu_ai_infra_mhc_post_grad(
            grad_output.contiguous(),
            x,
            h_res,
            h_out,
            h_post,
        )
        return grad_x, grad_h_res, grad_h_out, grad_h_post


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
        backend: ``"cann"`` (default) dispatches the differentiable CANN
            operator for fp16/bf16 inputs with fp32 mixing coefficients.
            CPU and FP32 execution use the traceable tensor formula; the
            legacy ``"omni"`` backend is deprecated.

    Returns:
        Flattened mixed residual streams.
    """
    if backend not in ("cann", "omni"):
        raise ValueError(f"backend must be 'cann' or 'omni', got {backend!r}")
    x_shape = x.size()
    residual = residual.reshape(x_shape[0], x_shape[1], num_stream, -1)
    if x.device.type == "npu":
        if backend == "cann":
            if x.dtype in (torch.float16, torch.bfloat16):
                # CANN's Python wrapper bypasses dispatch; its registered ops
                # preserve Meta inference during joint forward/backward capture.
                return _CannMhcPost.apply(residual, h_res, x, h_post).flatten(2)
        else:
            logger.warning(
                "mhc_post backend='omni' is deprecated and will be removed in a future release; "
                "use backend='cann'."
            )
            return _MhcPost.apply(residual, h_res, x, h_post).flatten(2)
    # CANN's fused kernel does not support FP32 block/residual inputs.
    output = h_post.unsqueeze(-1) * x.unsqueeze(-2)
    output = output + torch.sum(
        h_res.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return output.flatten(2).to(x.dtype)
