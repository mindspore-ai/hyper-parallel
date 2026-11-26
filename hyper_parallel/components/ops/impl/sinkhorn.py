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

from typing import Protocol

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.components.functional.sinkhorn import sinkhorn, sinkhorn_knopps

from ..registry import OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol


class SinkhornFn(Protocol):
    """Normalize square mixing matrices without changing their shape."""

    def __call__(self, x: torch.Tensor, sinkhorn_iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
        """Return the normalized matrix."""


SINKHORN = OpSpec[SinkhornFn](config_key="sinkhorn")


def _load_omni() -> SinkhornFn:
    require_package("omni_training_custom_ops")
    require_symbol(torch.ops.custom, "npu_sinkhorn")
    require_symbol(torch.ops.custom, "npu_sinkhorn_grad")
    return sinkhorn


class _TileKernelsSinkhorn(torch.autograd.Function):
    """Bind upstream's buffer-writing normalization kernels to autograd."""

    @staticmethod
    def forward(ctx, x, iterations, eps, kernels):
        """Retain logits because the upstream backward recomputes normalization."""
        output = torch.empty_like(x)
        kernels.mhc_sinkhorn_fwd(x, output, iterations, eps)
        ctx.save_for_backward(x)
        ctx.iterations, ctx.eps, ctx.backward_kernel = iterations, eps, kernels.mhc_sinkhorn_bwd
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        """Fill the logit gradient using the matching iteration count and epsilon."""
        (x,) = ctx.saved_tensors
        result = torch.empty_like(x)
        ctx.backward_kernel(gradient.contiguous(), x, result, ctx.iterations, ctx.eps)
        return result, None, None, None


def _load_tile_kernels() -> SinkhornFn:
    kernels = require_package("tile_kernels.mhc")
    for symbol in ("mhc_sinkhorn_fwd", "mhc_sinkhorn_bwd"):
        require_symbol(kernels, symbol)

    def run(x, sinkhorn_iters=20, eps=1e-6):
        """Keep the 4-stream restriction of the Ascend SIMD normalization."""
        if x.shape[-2:] != (4, 4):
            raise ValueError("TileKernels Sinkhorn requires 4-by-4 mixing matrices")
        output = _TileKernelsSinkhorn.apply(x.reshape(-1, 4, 4).contiguous(), sinkhorn_iters, eps, kernels)
        return output.reshape(x.shape)

    return run


def register_sinkhorn(registry: OpRegistry) -> None:
    """Register basic Torch normalization and Omni's fused implementation."""
    registry.register_family("sinkhorn", specs=(SINKHORN,))
    registry.register_impl(SINKHORN, name="native", devices=("cpu", "cuda", "npu"), load=lambda: sinkhorn_knopps)
    registry.register_impl(SINKHORN, name="omni", devices=("npu",), load=_load_omni)
    registry.register_impl(SINKHORN, name="tile_kernels", devices=("npu",), load=_load_tile_kernels)
