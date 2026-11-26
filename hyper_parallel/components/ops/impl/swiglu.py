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
"""Packed and DeepSeek FP32 clamped SwiGLU implementations."""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec
from .quantization import QuantTensor, _native_cast_back, _native_per_token


class SwiGLUFn(Protocol):
    """Apply SiLU(gate) * up to packed Gate/Up projections."""

    def __call__(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Split the packed projection evenly along the given dimension."""


class ClampedSwiGLUFn(Protocol):
    """Compute a FP32 activation from separate, optionally clamped projections."""

    def __call__(self, x: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
        """Clamp the projections for positive limits and return the FP32 result."""


class SwiGLUForwardFn(Protocol):
    """Non-quantized activation with optional routing weights and expert padding."""

    def __call__(self, x: torch.Tensor, fmt: str, **options: Any) -> torch.Tensor:
        """Preserve clamp, count-buffer, expert prefix and routing-weight options."""


class SwiGLUBackwardFn(Protocol):
    """Explicit VJP with optional routing-weight gradient and recomputed activation."""

    def __call__(self, x: torch.Tensor, gradient: torch.Tensor, fmt: str, **options: Any) -> tuple:
        """Return input gradient, optional weight gradient and optional activation in that order."""


class QuantizedSwiGLUFn(Protocol):
    """Fused activation and per-token FP8 conversion without a straight-through gradient."""

    def __call__(self, x: torch.Tensor, fmt: str, num_per_channels: int, **options: Any) -> QuantTensor:
        """Return encoded activation and scaling factors."""


class QuantizedSwiGLUBackwardFn(Protocol):
    """Explicit VJP of FP8 activations plus encoded input gradient."""

    def __call__(self, x: QuantTensor, gradient: torch.Tensor, fmt: str,
                 num_per_channels: int, **options: Any) -> tuple:
        """Return BF16 input VJP, encoded VJP, optional routing VJP and recomputed activation."""


SWIGLU = OpSpec[SwiGLUFn](config_key="swiglu")
SWIGLU_CLAMPED = OpSpec[ClampedSwiGLUFn](config_key="swiglu", variant="clamped_fp32")
SWIGLU_FORWARD = OpSpec[SwiGLUForwardFn](config_key="swiglu", variant="routed")
SWIGLU_BACKWARD = OpSpec[SwiGLUBackwardFn](config_key="swiglu", variant="backward")
SWIGLU_QUANTIZED = OpSpec[QuantizedSwiGLUFn](config_key="swiglu", variant="quantized")
SWIGLU_QUANTIZED_BACKWARD = OpSpec[QuantizedSwiGLUBackwardFn](config_key="swiglu", variant="quantized_backward")


def _native(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if x.shape[dim] % 2:
        raise ValueError("SwiGLU requires an evenly packed Gate/Up dimension")
    gate, up = x.chunk(2, dim=dim)
    return F.silu(gate) * up


def _clamp(x: torch.Tensor, up: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    gate, up = x.float(), up.float()
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return gate, up


def _native_clamped(x: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = _clamp(x, up, limit)
    return F.silu(gate) * up


def _load_native() -> SwiGLUFn:
    return _native


def _load_native_clamped() -> ClampedSwiGLUFn:
    return _native_clamped


def _load_npu() -> SwiGLUFn:
    require_symbol(require_package("torch_npu"), "npu_swiglu")
    # The existing implementation imports the optional NPU package.
    from hyper_parallel.components.functional.swiglu import swiglu  # pylint: disable=import-outside-toplevel
    return swiglu


def _load_npu_clamped() -> ClampedSwiGLUFn:
    kernel = _load_npu()

    def swiglu_clamped(x: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
        """Clamp separate projections in FP32 and apply the packed activation kernel."""
        gate, up = _clamp(x, up, limit)
        return kernel(torch.cat((gate, up), dim=-1), dim=-1)

    return swiglu_clamped


class _TileKernelsSwiGLU(torch.autograd.Function):
    """Bind the upstream non-quantized forward and backward to one autograd node."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, forward_kernel: Any, backward_kernel: Any,
        weights: torch.Tensor | None, options: dict[str, Any], fmt: str | None,
    ) -> torch.Tensor:
        """Retain the packed input and the exact backward selected for this call."""
        ctx.save_for_backward(x, weights)
        ctx.options = options
        ctx.backward_kernel = backward_kernel
        ctx.fmt = fmt or ("bf16" if x.dtype == torch.bfloat16 else "fp32")
        output = forward_kernel(x, ctx.fmt, **options, **({} if weights is None else {"topk_weights": weights}))
        return _clear_padding(output, options, _tile_alignment(options))

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: Any) -> Any:
        """Consume the upstream tuple containing the unquantized input gradient."""
        x, weights = ctx.saved_tensors
        options = {name: value for name, value in ctx.options.items() if name != "clamped_count"}
        dtype = torch.promote_types(x.dtype, grad_output.dtype)
        fmt = "bf16" if dtype == torch.bfloat16 else "fp32"
        result = ctx.backward_kernel(x.to(dtype), grad_output.contiguous().to(dtype), fmt,
                                     **options, **({} if weights is None else {"topk_weights": weights}))
        return result[0], None, None, None if weights is None else result[1], None, None


def _load_tile_kernels() -> SwiGLUFn:
    kernels = require_package("tile_kernels.quant")
    require_symbol(kernels, "swiglu_forward")
    require_symbol(kernels, "swiglu_backward")
    forward_kernel, backward_kernel = kernels.swiglu_forward, kernels.swiglu_backward

    def swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Adapt arbitrary packed axes and leading dimensions to upstream's 2-D ABI."""
        packed = x.movedim(dim, -1)
        width = packed.shape[-1]
        result = _TileKernelsSwiGLU.apply(
            packed.reshape(-1, width).contiguous(), forward_kernel, backward_kernel, None, {}, None,
        )
        return result.reshape(*packed.shape[:-1], width // 2).movedim(-1, dim)

    return swiglu


def _expert_rows(x, prefixes, alignment):
    rows = torch.arange(x.shape[0], device=x.device)
    if prefixes is None:
        return torch.ones_like(rows, dtype=torch.bool)
    starts = F.pad(prefixes[:-1].long(), (1, 0))
    starts = (starts + alignment - 1).div(alignment, rounding_mode="floor") * alignment
    return ((rows[:, None] >= starts) & (rows[:, None] < prefixes)).any(-1)


def _clear_padding(output, options, alignment):
    """Upstream forward skips expert padding rows in newly allocated outputs."""
    prefixes = options.get("psum_num_tokens_per_expert")
    if prefixes is None:
        return output
    data = output[0] if isinstance(output, tuple) else output
    valid = _expert_rows(data, prefixes, alignment)
    if not isinstance(output, tuple):
        return output.masked_fill(~valid[:, None], 0)
    data.view(torch.uint8).masked_fill_(~valid[:, None], 0)
    scale = output[1]
    if scale.dtype == torch.int16:
        column_major = scale.stride(0) == 1 and scale.stride(1) != 1
        exponent = scale.T.view(torch.uint8).T if column_major else scale.view(torch.uint8)
        padded_valid = F.pad(valid, (0, exponent.shape[0] - valid.shape[0]), value=False)
        exponent.masked_fill_(~padded_valid[:, None], 127)
    else:
        scale.masked_fill_(~valid[:, None], 1)
    return output


def _tile_alignment(options):
    if options.get("psum_num_tokens_per_expert") is None:
        return 128
    return require_package("tile_kernels.config").get_token_alignment()


def _native_forward(x, fmt, *, psum_num_tokens_per_expert=None, topk_weights=None,
                    routed_scaling_factor=None, clamp_value=None, clamped_count=None, token_alignment=128):
    if routed_scaling_factor is not None and topk_weights is None:
        raise ValueError("routed_scaling_factor requires topk_weights")
    gate, up = x.float().chunk(2, -1)
    valid = _expert_rows(x, psum_num_tokens_per_expert, token_alignment)
    if clamped_count is not None:
        if clamp_value is None:
            raise ValueError("clamped_count requires clamp_value")
        counts = ((gate > clamp_value).sum(-1), (up > clamp_value).sum(-1), (up < -clamp_value).sum(-1))
        with torch.no_grad():
            clamped_count.add_(torch.stack(tuple((count * valid).sum() for count in counts)
                                           + (valid.sum() * gate.shape[-1],)))
    if clamp_value is not None:
        gate, up = gate.clamp(max=clamp_value), up.clamp(-clamp_value, clamp_value)
    output = F.silu(gate) * up
    if topk_weights is not None:
        output = output * (topk_weights * (1.0 if routed_scaling_factor is None else routed_scaling_factor))[:, None]
    return (output * valid[:, None]).to({"bf16": torch.bfloat16, "fp32": torch.float32}[fmt])


@torch.no_grad()
def _native_backward(x, gradient, fmt, *, psum_num_tokens_per_expert=None, topk_weights=None,
                     routed_scaling_factor=None, clamp_value=None, do_recompute=False, token_alignment=128):
    x = x.reshape(-1, x.shape[-1])
    gradient = gradient.reshape(x.shape[0], x.shape[-1] // 2)
    raw_gate, raw_up = x.float().chunk(2, -1)
    gate, up = raw_gate, raw_up
    if clamp_value is not None:
        gate, up = gate.clamp(max=clamp_value), up.clamp(-clamp_value, clamp_value)
    valid = _expert_rows(x, psum_num_tokens_per_expert, token_alignment)
    sigmoid = gate.sigmoid()
    scale = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    if routed_scaling_factor is not None and topk_weights is None:
        raise ValueError("routed_scaling_factor requires topk_weights")
    effective = valid.float() if topk_weights is None else valid * topk_weights * scale
    upstream = gradient.float() * effective[:, None]
    grad_gate = upstream * up * sigmoid * (1 + gate * (1 - sigmoid))
    grad_up = upstream * gate * sigmoid
    if clamp_value is not None:
        grad_gate = grad_gate.masked_fill(raw_gate > clamp_value, 0)
        grad_up = grad_up.masked_fill(raw_up.abs() > clamp_value, 0)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[fmt]
    result = [torch.cat((grad_gate, grad_up), -1).to(dtype)]
    activation = gate * sigmoid * up
    if topk_weights is not None:
        result.append((gradient.float() * activation).sum(-1) * scale * valid)
    if do_recompute:
        result.append((activation * effective[:, None]).to(dtype))
    return tuple(result)


def _tile_options(options):
    options = dict(options)
    if options.get("topk_weights") is not None:
        options["topk_weights"] = options["topk_weights"].contiguous()
    default_alignment = 128 if options.get("psum_num_tokens_per_expert") is not None else None
    alignment = options.pop("token_alignment", default_alignment)
    if alignment is not None:
        config = require_package("tile_kernels.config")
        if alignment != config.get_token_alignment():
            raise ValueError("token_alignment must match TileKernels' configured expert alignment")
    return options


def _load_routed() -> SwiGLUForwardFn:
    kernels = require_package("tile_kernels.quant")
    for name in ("swiglu_forward", "swiglu_backward"):
        require_symbol(kernels, name)

    def run(x: Any, fmt: Any, **options: Any) -> Any:
        """Apply the operator contract using the selected implementation."""
        x = x.contiguous()
        options = _tile_options(options)
        weighted_grad = getattr(options.get("topk_weights"), "requires_grad", False)
        if torch.is_grad_enabled() and (x.requires_grad or weighted_grad):
            weights = options.pop("topk_weights", None)
            return _TileKernelsSwiGLU.apply(x, kernels.swiglu_forward, kernels.swiglu_backward, weights, options, fmt)
        output = kernels.swiglu_forward(x, fmt, **options)
        return _clear_padding(output, options, _tile_alignment(options))

    return run


def _load_clamped() -> ClampedSwiGLUFn:
    kernel = _load_routed()

    def run(x: Any, up: Any, limit: Any) -> Any:
        """Apply the operator contract using the selected implementation."""
        gate, up = torch.broadcast_tensors(x.float(), up.float())
        packed = torch.cat((gate, up), -1)
        width = packed.shape[-1]
        output = kernel(packed.reshape(-1, width).contiguous(), "fp32",
                        **({"clamp_value": limit} if limit > 0 else {}))
        return output.reshape(*packed.shape[:-1], width // 2)

    return run


def _load_explicit(name):
    kernels = require_package("tile_kernels.quant")
    require_symbol(kernels, name)

    def run(*args: Any, **options: Any) -> Any:
        """Apply the operator contract using the selected implementation."""
        settings = _tile_options(options)
        output = getattr(kernels, name)(*args, **settings)
        if name == "swiglu_forward_and_per_token_cast":
            return _clear_padding(output, settings, _tile_alignment(settings))
        return output

    return run


@torch.no_grad()
def _native_quantized(x, fmt, num_per_channels, **options):
    if fmt != "e4m3":
        raise ValueError("fused SwiGLU quantization requires e4m3")
    cast_options = {name: options.pop(name) for name in ("use_tma_aligned_col_major_sf", "round_sf", "use_packed_ue8m0")
                    if name in options}
    activation = _native_forward(x, "fp32", **options)
    encoded = _native_per_token(activation, fmt, num_per_channels, **cast_options)
    return _clear_padding(encoded, options, options.get("token_alignment", 128))


@torch.no_grad()
def _native_quantized_backward(x, gradient, fmt, num_per_channels, **options):
    if fmt != "e4m3":
        raise ValueError("fused SwiGLU quantization requires e4m3")
    cast_options = {name: options.pop(name) for name in ("use_tma_aligned_col_major_sf", "round_sf", "use_packed_ue8m0")
                    if name in options}
    values = _native_cast_back(x, "fp32", (1, num_per_channels))
    result = _native_backward(values, gradient, "fp32", **options)
    encoded = _native_per_token(result[0], fmt, num_per_channels, **cast_options)
    suffix = list(result[1:])
    if options.get("do_recompute"):
        suffix[-1] = suffix[-1].to(torch.bfloat16)
    return (result[0].to(torch.bfloat16), encoded, *suffix)


def register_swiglu(registry: OpRegistry) -> None:
    """Register packed and clamped SwiGLU under the same selection key."""
    registry.register_family("swiglu", specs=(SWIGLU, SWIGLU_CLAMPED, SWIGLU_FORWARD, SWIGLU_BACKWARD,
                                              SWIGLU_QUANTIZED, SWIGLU_QUANTIZED_BACKWARD))
    registry.register_impl(SWIGLU, name="native", devices=("cuda", "cpu", "npu"), load=_load_native)
    registry.register_impl(SWIGLU, name="torch_npu", devices=("npu",), load=_load_npu)
    registry.register_impl(
        SWIGLU, name="tile_kernels", devices=("npu",), load=_load_tile_kernels,
    )
    registry.register_impl(
        SWIGLU_CLAMPED, name="native", devices=("cuda", "cpu", "npu"), load=_load_native_clamped,
    )
    registry.register_impl(SWIGLU_CLAMPED, name="torch_npu", devices=("npu",), load=_load_npu_clamped)
    registry.register_impl(SWIGLU_CLAMPED, name="tile_kernels", devices=("npu",), load=_load_clamped)
    for spec, native, loader in (
        (SWIGLU_FORWARD, _native_forward, _load_routed),
        (SWIGLU_BACKWARD, _native_backward, lambda: _load_explicit("swiglu_backward")),
        (SWIGLU_QUANTIZED, _native_quantized, lambda: _load_explicit("swiglu_forward_and_per_token_cast")),
        (SWIGLU_QUANTIZED_BACKWARD, _native_quantized_backward,
         lambda: _load_explicit("swiglu_backward_and_per_token_cast")),
    ):
        registry.register_impl(spec, name="native", devices=("cpu", "cuda", "npu"), load=lambda fn=native: fn)
        registry.register_impl(spec, name="tile_kernels", devices=("npu",), load=loader)
