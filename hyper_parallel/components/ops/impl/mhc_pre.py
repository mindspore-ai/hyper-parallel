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
"""mHC pre composition, coefficient splitting and stream aggregation."""

from __future__ import annotations

from functools import partial
from typing import Any, Protocol

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from hyper_parallel.components.functional.mhc_pre import mhc_pre as fused_mhc_pre
from hyper_parallel.components.functional.sinkhorn import sinkhorn_knopps

from .._dependency import require_package, require_symbol
from .gemm import kernel_symbol
from ..registry import OpRegistry
from ..spec import OpSpec
from .sinkhorn import _load_tile_kernels as load_tile_sinkhorn


class MhcPreFn(Protocol):
    """Return block input, post/residual coefficients and unchanged residual streams."""

    def __call__(
        self, x: torch.Tensor, phi: torch.Tensor, branch_alpha: torch.Tensor, branch_beta: torch.Tensor,
        num_stream: int, sinkhorn_iters: int = 20, norm_eps: float = 1e-6, hc_eps: float = 1e-6,
        gamma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize, project and mix flattened residual streams."""


class MhcSplitFn(Protocol):
    """Generate pre, post and unnormalized residual coefficients."""

    def __call__(
        self, mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, num_stream: int,
        *, post_multiplier: float = 2.0, epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the projected coefficients without normalizing the residual matrix."""


class MhcApplyFn(Protocol):
    """Aggregate explicit residual streams using the supplied pre coefficients."""

    def __call__(self, hidden: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """Return the weighted sum in the activation dtype."""


MHC_PRE = OpSpec[MhcPreFn](config_key="mhc_pre")
MHC_SPLIT = OpSpec[MhcSplitFn](config_key="mhc_pre", variant="split_mixes")
MHC_PRE_APPLY = OpSpec[MhcApplyFn](config_key="mhc_pre", variant="apply_mix")


def _native_split(mixes, scale, base, num_stream, *, post_multiplier=2.0, epsilon=1e-6):
    pre, post, residual = mixes.split((num_stream, num_stream, num_stream * num_stream), dim=-1)
    pre_bias, post_bias, res_bias = base.reshape(-1).split((num_stream, num_stream, num_stream * num_stream))
    alpha = scale.reshape(-1)
    return (torch.sigmoid(pre * alpha[0] + pre_bias) + epsilon,
            torch.sigmoid(post * alpha[1] + post_bias) * post_multiplier,
            (residual * alpha[2] + res_bias).unflatten(-1, (num_stream, num_stream)))


def _native_apply(hidden, pre):
    return (hidden.float() * pre.unsqueeze(-1)).sum(-2).to(hidden.dtype)


def _compose_pre(split, normalize, collapse) -> MhcPreFn:
    def run(
        x: Any,
        phi: Any,
        branch_alpha: Any,
        branch_beta: Any,
        num_stream: Any,
        sinkhorn_iters: Any=20,
        norm_eps: Any=1e-6,
        hc_eps: Any=1e-6,
        gamma: Any=None,
    ) -> Any:
        """Keep normalization/projection differentiable and parameters owned by the module."""
        normalized = F.rms_norm(x.float(), (x.shape[-1],), eps=norm_eps)
        if gamma is not None:
            normalized = normalized * gamma.reshape(-1).float()
        mixes = F.linear(normalized, phi.float())  # pylint: disable=not-callable
        pre, post, residual = split(mixes, branch_alpha, branch_beta, num_stream, epsilon=hc_eps)
        hidden = x.unflatten(-1, (num_stream, x.shape[-1] // num_stream))
        return collapse(hidden, pre), post, normalize(residual, sinkhorn_iters, hc_eps), x

    return run


class _TileSplit(torch.autograd.Function):
    """Bind the upstream coefficient kernels, including scale and bias gradients."""

    @staticmethod
    def forward(ctx: Any, mixes: Any, scale: Any, base: Any, multiplier: Any, epsilon: Any, kernels: Any) -> Any:
        """Write the three coefficient outputs using the upstream 4-stream ABI."""
        pre = mixes.new_empty(mixes.shape[0], 4)
        post = torch.empty_like(pre)
        residual = mixes.new_empty(mixes.shape[0], 16)
        kernels.mhc_pre_split_mixes_fwd(mixes, scale, base, pre, post, residual, multiplier, epsilon)
        ctx.save_for_backward(mixes, post, scale, base)
        ctx.multiplier, ctx.backward_kernel = multiplier, kernels.mhc_pre_split_mixes_bwd
        return pre, post, residual

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, pre_grad: Any, post_grad: Any, residual_grad: Any) -> Any:
        """Return input, scale and bias gradients from the upstream backward."""
        mixes, post, scale, base = ctx.saved_tensors
        grad_mixes, grad_scale, grad_base = (torch.empty_like(t) for t in (mixes, scale, base))
        ctx.backward_kernel(pre_grad.contiguous(), post_grad.contiguous(), residual_grad.contiguous(),
                            mixes, post, scale, base, grad_mixes, grad_scale, grad_base, ctx.multiplier)
        return grad_mixes, grad_scale, grad_base, None, None, None


class _TileApply(torch.autograd.Function):
    """Bind the upstream stream aggregation and its accumulating backward."""

    @staticmethod
    def forward(ctx: Any, hidden: Any, pre: Any, kernels: Any) -> Any:
        """Keep explicit streams and coefficients for backward."""
        output = hidden.new_empty(hidden.shape[0], hidden.shape[-1])
        kernels.mhc_pre_apply_mix_fwd(hidden, pre, output)
        ctx.save_for_backward(hidden, pre)
        ctx.backward_kernel = kernels.mhc_pre_apply_mix_bwd
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: Any) -> Any:
        """Zero the activation gradient because the upstream kernel adds into it."""
        hidden, pre = ctx.saved_tensors
        hidden_grad = torch.zeros_like(hidden)
        pre_grad = ctx.backward_kernel(gradient.contiguous(), hidden, pre, hidden_grad)
        return hidden_grad, pre_grad, None


def _load_tile_split() -> MhcSplitFn:
    kernels = require_package("tile_kernels.mhc")
    for name in ("mhc_pre_split_mixes_fwd", "mhc_pre_split_mixes_bwd"):
        require_symbol(kernels, name)

    def run(mixes: Any, scale: Any, base: Any, num_stream: Any, *, post_multiplier: Any=2.0, epsilon: Any=1e-6) -> Any:
        """Flatten token axes while preserving the coefficient shapes and gradient links."""
        if num_stream != 4:
            raise ValueError("TileKernels mHC coefficient splitting requires four streams")
        pre, post, residual = _TileSplit.apply(
            mixes.reshape(-1, 24).contiguous(), scale.reshape(-1).contiguous(), base.reshape(-1).contiguous(),
            post_multiplier, epsilon, kernels,
        )
        shape = mixes.shape[:-1]
        return pre.reshape(*shape, 4), post.reshape(*shape, 4), residual.reshape(*shape, 4, 4)

    return run


def _load_tile_apply() -> MhcApplyFn:
    kernels = require_package("tile_kernels.mhc")
    for name in ("mhc_pre_apply_mix_fwd", "mhc_pre_apply_mix_bwd"):
        require_symbol(kernels, name)

    def run(hidden: Any, pre: Any) -> Any:
        """Restore token axes after the upstream three-dimensional stream aggregation."""
        output = _TileApply.apply(hidden.reshape(-1, *hidden.shape[-2:]).contiguous(),
                                  pre.reshape(-1, pre.shape[-1]).contiguous(), kernels)
        return output.reshape(*hidden.shape[:-2], hidden.shape[-1])

    return run


def _load_tile_pre() -> MhcPreFn:
    return _compose_pre(_load_tile_split(), load_tile_sinkhorn(), _load_tile_apply())


def _load_cann() -> MhcPreFn:
    require_symbol(require_package("cann_ops_transformer.ops"), "mhc_pre_sinkhorn")
    return partial(fused_mhc_pre, backend="cann")


class MhcPrenormFn(Protocol):
    """Return projection partials and squared sums; this operation does not apply RMSNorm."""

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        num_splits: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split contiguous 256-column blocks, with earlier splits receiving remainder blocks."""


MHC_PRENORM = OpSpec[MhcPrenormFn](config_key="mhc_pre", variant="prenorm_projection")


def _ranges(width, splits):
    if width <= 0 or width % 256 or not 1 <= splits <= width // 256:
        raise ValueError("Prenorm projection requires positive K divisible by 256 and 1..K/256 splits")
    blocks, extra = divmod(width // 256, splits)
    return [(256 * (i * blocks + min(i, extra)), 256 * ((i + 1) * blocks + min(i + 1, extra))) for i in range(splits)]


def native_prenorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve split dimension only when explicitly requested."""
    projected, squared = [], []
    for start, end in _ranges(x.shape[-1], num_splits or 1):
        columns = x[..., start:end].float()
        projected.append(columns @ weight[..., start:end].float().T)
        squared.append(columns.square().sum(-1))
    return (torch.stack(projected), torch.stack(squared)) if num_splits is not None else (projected[0], squared[0])


class _PrenormProjection(torch.autograd.Function):
    """Attach both projection and squared-sum gradients to upstream forward outputs."""

    @staticmethod
    def forward(ctx: Any, x: Any, weight: Any, splits: Any, kernel: Any) -> Any:
        """Execute the selected backend and retain the tensors needed for backward."""
        shape = (x.shape[0], weight.shape[0])
        prefix = () if splits is None else (splits,)
        projection = torch.empty((*prefix, *shape), device=x.device, dtype=torch.float32)
        squared = torch.empty((*prefix, x.shape[0]), device=x.device, dtype=torch.float32)
        kernel(x, weight, projection, squared, num_splits=splits)
        ctx.save_for_backward(x, weight)
        ctx.splits = splits
        ctx.set_materialize_grads(False)
        return projection, squared

    @staticmethod
    def backward(ctx: Any, grad_projection: Any, grad_squared: Any) -> Any:
        """Return the input gradients required by the operator contract."""
        x, weight = ctx.saved_tensors
        if ctx.splits is None:
            grad_projection = None if grad_projection is None else grad_projection.unsqueeze(0)
            grad_squared = None if grad_squared is None else grad_squared.unsqueeze(0)
        grad_x, grad_w = torch.zeros_like(x, dtype=torch.float32), torch.zeros_like(weight, dtype=torch.float32)
        for i, (start, end) in enumerate(_ranges(x.shape[-1], ctx.splits or 1)):
            if grad_projection is not None:
                grad_x[:, start:end] = grad_projection[i] @ weight[:, start:end].float()
                grad_w[:, start:end] = grad_projection[i].T @ x[:, start:end].float()
            if grad_squared is not None:
                grad_x[:, start:end] += 2 * x[:, start:end].float() * grad_squared[i, :, None]
        return grad_x.to(x.dtype), grad_w.to(weight.dtype), None, None


def load_prenorm() -> MhcPrenormFn:
    """Load DeepGEMM's fused projection/squared-sum forward lazily."""
    kernel = kernel_symbol(require_package("deep_gemm"), "tf32_hc_prenorm_gemm")

    def run(x: Any, weight: Any, *, num_splits: Any=None) -> Any:
        """Apply the operator contract using the selected implementation."""
        return _PrenormProjection.apply(x, weight, num_splits, kernel)

    return run


def register_mhc_pre(registry: OpRegistry) -> None:
    """Register complete Pre and its reusable coefficient/aggregation stages."""
    registry.register_family("mhc_pre", specs=(MHC_PRE, MHC_SPLIT, MHC_PRE_APPLY, MHC_PRENORM))
    registry.register_impl(MHC_PRENORM, name="native", devices=("cpu", "cuda", "npu"), load=lambda: native_prenorm)
    registry.register_impl(MHC_PRENORM, name="deep_gemm", devices=("npu",), load=load_prenorm)
    for spec, native, tile in (
        (MHC_PRE, lambda: _compose_pre(_native_split, sinkhorn_knopps, _native_apply), _load_tile_pre),
        (MHC_SPLIT, lambda: _native_split, _load_tile_split),
        (MHC_PRE_APPLY, lambda: _native_apply, _load_tile_apply),
    ):
        registry.register_impl(spec, name="native", devices=("cpu", "cuda", "npu"), load=native)
        registry.register_impl(spec, name="tile_kernels", devices=("npu",), load=tile)
    registry.register_impl(MHC_PRE, name="cann", devices=("npu",), load=_load_cann)
