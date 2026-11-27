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
"""Backends for residual mixing after an mHC-wrapped sublayer."""

from __future__ import annotations

from typing import Any, Protocol, Tuple

import torch

from .._dependency import require_package, require_symbol
from ..factory import _select_for_call
from ..registry import OpRegistry
from ..spec import OpSpec


class MhcPostFn(Protocol):
    """Mix transformed and residual streams using learned mHC coefficients.

    Reshape residual into streams and compute, for output stream j,
    h_post[j] * x + sum_i(h_res[i, j] * residual[i]). Flatten output stream
    and hidden axes. The native implementation restores the sublayer output dtype.
    """

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor, num_stream: int,
    ) -> torch.Tensor:
        """Return mixed streams with the stream dimension flattened.

        Args:
            x: Sublayer outputs [batch, sequence, hidden].
            residual: Residual states [batch, sequence, num_stream * hidden].
            h_post: Sublayer mixing coefficients [batch, sequence, num_stream].
            h_res: Residual mixing coefficients [batch, sequence, num_stream, num_stream].
            num_stream: Number of residual streams.

        Returns:
            Mixed residual states [batch, sequence, num_stream * hidden] in the x dtype.
        """


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    num_stream: int,
) -> torch.Tensor:
    """Mix residual streams using the process-wide backend configuration.

    Args:
        x: Sublayer outputs [batch, sequence, hidden].
        residual: Residual states [batch, sequence, num_stream * hidden].
        h_post: Sublayer mixing coefficients [batch, sequence, num_stream].
        h_res: Residual mixing coefficients [batch, sequence, num_stream, num_stream].
        num_stream: Number of residual streams.

    Returns:
        Mixed residual states [batch, sequence, num_stream * hidden] in the x dtype.

    Notes:
        Mix transformed and residual streams using learned mHC coefficients.

        Reshape residual into streams and compute, for output stream j,
        h_post[j] * x + sum_i(h_res[i, j] * residual[i]). Flatten output stream
        and hidden axes. The native implementation restores the sublayer output dtype.
    """
    operation = _select_for_call(MHC_POST, device=x.device)
    return operation(x, residual, h_post, h_res, num_stream)


MHC_POST = OpSpec[MhcPostFn](config_key='mhc_post')


def _native(
    x: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor, num_stream: int,
) -> torch.Tensor:
    streams = residual.reshape(x.shape[0], x.shape[1], num_stream, x.shape[-1])
    transformed = h_post.unsqueeze(-1) * x.unsqueeze(-2)
    mixed = (h_res.unsqueeze(-1) * streams.unsqueeze(-2)).sum(dim=2)
    return (transformed + mixed).flatten(2).to(x.dtype)


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


def _load_native() -> MhcPostFn:
    return _native


def _load_omni() -> MhcPostFn:
    require_package("omni_training_custom_ops")
    require_symbol(torch.ops.custom, "npu_ai_infra_manifold_constrained_hyper_connection_post")
    require_symbol(torch.ops.custom, "npu_ai_infra_mhc_post_grad")

    def run(
        x: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor, num_stream: int,
    ) -> torch.Tensor:
        """Adapt flattened streams to the Omni autograd bridge."""
        if x.dtype == torch.float32:
            raise ValueError("Fused mHC post does not support FP32 inputs; configure ops.mhc_post='native'")
        streams = residual.reshape(x.shape[0], x.shape[1], num_stream, -1)
        return _MhcPost.apply(streams, h_res, x, h_post).flatten(2)

    return run


def _load_cann() -> MhcPostFn:
    require_package("cann_ops_transformer")
    extension = require_package("cann_ops_transformer.ops")
    require_symbol(extension, "mhc_post")
    kernel = extension.mhc_post


    def run(
        x: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor, num_stream: int,
    ) -> torch.Tensor:
        """Adapt flattened streams to CANN's residual-first argument order."""
        if x.dtype == torch.float32:
            raise ValueError("Fused mHC post does not support FP32 inputs; configure ops.mhc_post='native'")
        streams = residual.reshape(x.shape[0], x.shape[1], num_stream, -1)
        return kernel(streams, h_res, x, h_post).flatten(2)

    return run


def register_mhc_post(registry: OpRegistry) -> None:
    """Register Torch residual mixing and the optional CANN and Omni kernels."""
    registry.register_family("mhc_post", specs=(MHC_POST,))
    registry.register_impl(MHC_POST, name="native", devices=("cuda", "cpu", "npu"), load=_load_native)
    registry.register_impl(MHC_POST, name="cann", devices=("npu",), load=_load_cann)
    registry.register_impl(MHC_POST, name="omni", devices=("npu",), load=_load_omni)
