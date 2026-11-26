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

from functools import partial
from typing import Protocol

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.components.functional.mhc_post import mhc_post

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec


class MhcPostFn(Protocol):
    """Combine residual and transformed streams and flatten the result."""

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor, num_stream: int,
    ) -> torch.Tensor:
        """Return mixed streams with the stream dimension flattened."""


MHC_POST = OpSpec[MhcPostFn](config_key="mhc_post")


def _native(
    x: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor, num_stream: int,
) -> torch.Tensor:
    streams = residual.reshape(x.shape[0], x.shape[1], num_stream, x.shape[-1])
    transformed = h_post.unsqueeze(-1) * x.unsqueeze(-2)
    mixed = (h_res.unsqueeze(-1) * streams.unsqueeze(-2)).sum(dim=2)
    return (transformed + mixed).flatten(2).to(x.dtype)


def _load_native() -> MhcPostFn:
    return _native


def _load_omni() -> MhcPostFn:
    require_package("omni_training_custom_ops")
    require_symbol(torch.ops.custom, "npu_ai_infra_manifold_constrained_hyper_connection_post")
    require_symbol(torch.ops.custom, "npu_ai_infra_mhc_post_grad")
    return partial(mhc_post, backend="omni")


def _load_cann() -> MhcPostFn:
    require_package("cann_ops_transformer")
    require_symbol(require_package("cann_ops_transformer.ops"), "mhc_post")
    return partial(mhc_post, backend="cann")


class _TileKernelsMhcPost(torch.autograd.Function):
    """Connect upstream's raw post forward and backward kernels."""

    @staticmethod
    def forward(ctx, x, residual, post, combine, kernels):
        """Save the exact activation and coefficient tensors consumed by the kernel."""
        ctx.save_for_backward(x, residual, post, combine)
        ctx.backward_kernel = kernels.mhc_post_bwd
        return kernels.mhc_post_fwd(x, residual, post, combine)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        """Return all four upstream gradients without deriving another backward."""
        return (*ctx.backward_kernel(*ctx.saved_tensors, gradient.contiguous()), None)


def _load_tile_kernels() -> MhcPostFn:
    kernels = require_package("tile_kernels.mhc")
    for symbol in ("mhc_post_fwd", "mhc_post_bwd"):
        require_symbol(kernels, symbol)

    def run(x, residual, h_post, h_res, num_stream):
        """Restore upstream's explicit stream axis and Hyper's flattened output."""
        streams = residual.reshape(*x.shape[:2], num_stream, x.shape[-1]).contiguous()
        output = _TileKernelsMhcPost.apply(
            x.contiguous(), streams, h_post.unsqueeze(-1).contiguous(), h_res.contiguous(), kernels,
        )
        return output.flatten(2)

    return run


def register_mhc_post(registry: OpRegistry) -> None:
    """Register Torch residual mixing and the optional CANN and Omni kernels."""
    registry.register_family("mhc_post", specs=(MHC_POST,))
    registry.register_impl(MHC_POST, name="native", devices=("cuda", "cpu", "npu"), load=_load_native)
    registry.register_impl(MHC_POST, name="cann", devices=("npu",), load=_load_cann)
    registry.register_impl(MHC_POST, name="omni", devices=("npu",), load=_load_omni)
    registry.register_impl(MHC_POST, name="tile_kernels", devices=("npu",), load=_load_tile_kernels)
