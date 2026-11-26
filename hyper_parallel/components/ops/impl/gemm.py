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
"""Dense GEMM and head contractions, with explicit quantized operand storage."""

from __future__ import annotations

from typing import Any, Protocol, Union
from collections.abc import Callable

import torch

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec
from .quantization import QuantTensor

Operand = Union[torch.Tensor, QuantTensor]
Recipe = tuple[int, int, int]


class GemmFn(Protocol):
    """Multiply physical operands in nt/nn/tn/tt layout, optionally adding C."""

    def __call__(
        self,
        a: Operand,
        b: Operand,
        *,
        layout: str = "nt",
        c: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        alpha: float = 1.0,
        recipe: Recipe = (1, 1, 32),
    ) -> torch.Tensor:
        """Return alpha * op(A) @ op(B) + C; quantized inputs have no VJP."""


class HeadEinsumFn(Protocol):
    """Contract head projections using one of the three MLA head expressions."""

    def __call__(
        self,
        a: Operand,
        b: Operand,
        expression: str,
        *,
        c: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        recipe: Recipe = (1, 1, 32),
    ) -> torch.Tensor:
        """Support bhr,hdr->bhd, bhd,hdr->bhr and bhd,bhr->hdr."""


GEMM = OpSpec[GemmFn](config_key="gemm")
HEAD_EINSUM = OpSpec[HeadEinsumFn](config_key="gemm", variant="head_einsum")


def operand_data(value: Operand) -> torch.Tensor:
    """Return the selection tensor without decoding a quantized operand."""
    return value[0] if isinstance(value, tuple) else value


def transpose_operand(value: Operand) -> Operand:
    """Transpose both data and scale axes, preserving their physical strides."""
    return tuple(t.transpose(-1, -2) for t in value) if isinstance(value, tuple) else value.transpose(-1, -2)


def require_quantized_inference(*values: Operand | None) -> None:
    """Reject gradients through quantized data/scales instead of silently detaching them."""
    if not torch.is_grad_enabled() or not any(isinstance(v, tuple) for v in values):
        return
    tensors = [t for v in values if v is not None for t in (v if isinstance(v, tuple) else (v,))]
    if any(t.requires_grad for t in tensors):
        raise RuntimeError("Quantized DeepGEMM contracts are inference-only; no quantization backward is defined")


def decode_operand(value: Operand, granularity: tuple[int, int] = (1, 32)) -> torch.Tensor:
    """Decode canonical [..., MN, K] FP8/FP4 and K-packed UE8M0 scales for the native reference."""
    if not isinstance(value, tuple):
        return value.float()
    data, scale = value
    if data.dtype == torch.int8:
        axis = -2 if data.stride(-2) == 1 else -1
        raw = data.movedim(axis, -1).contiguous().view(torch.uint8)
        codes = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
        levels = data.new_tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), dtype=torch.float32)
        decoded = (levels[codes & 7] * torch.where(codes < 8, 1.0, -1.0)).movedim(-1, axis)
    else:
        decoded = data.float()
    if scale.dtype == torch.int16:
        exponent = scale.reshape(-1).view(torch.uint8).reshape(*scale.shape[:-1], -1)
        scale = torch.pow(2.0, exponent.float() - 127)
    expanded = scale.float().repeat_interleave(granularity[0], -2).repeat_interleave(granularity[1], -1)
    return decoded * expanded[..., : decoded.shape[-2], : decoded.shape[-1]]


def canonical_operands(a: Operand, b: Operand, layout: str) -> tuple[Operand, Operand]:
    """Normalize physical dense operands to [M,K] and [N,K]."""
    if layout not in ("nt", "nn", "tn", "tt"):
        raise ValueError("GEMM layout must be nt, nn, tn or tt")
    return (transpose_operand(a) if layout[0] == "t" else a, transpose_operand(b) if layout[1] == "n" else b)


def native_gemm(
    a: Operand,
    b: Operand,
    *,
    layout: str = "nt",
    c: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    alpha: float = 1.0,
    recipe: Recipe = (1, 1, 32),
) -> torch.Tensor:
    """Reference F32 accumulation and explicit output rounding."""
    require_quantized_inference(a, b, c)
    left, right = canonical_operands(a, b, layout)
    result = (
        decode_operand(left, (recipe[0], recipe[2])) @ decode_operand(right, (recipe[1], recipe[2])).transpose(-1, -2)
    ) * alpha
    return (result if c is None else result + c.float()).to(output_dtype)


def _output_buffer(shape, device, c, output_dtype, alpha):
    if c is not None:
        if c.dtype != output_dtype:
            raise ValueError("DeepGEMM C must have the requested output dtype")
        if tuple(c.shape) == shape:
            # The native dense API distinguishes accumulation from bias by pointer identity.
            result = c.clone()
            return result, result
        if alpha != 1.0:
            raise ValueError("DeepGEMM does not combine a separate bias with alpha")
    return torch.empty(shape, device=device, dtype=output_dtype), c


class _DenseGemm(torch.autograd.Function):
    """Attach the mathematical VJP to an upstream output-buffer-only BF16 kernel."""

    @staticmethod
    def forward(ctx: Any, a: Any, b: Any, c: Any, layout: Any, output_dtype: Any, alpha: Any, kernel: Any) -> Any:
        """Execute the selected backend and retain the tensors needed for backward."""
        left, right = canonical_operands(a, b, layout)
        result, accumulation = _output_buffer((left.shape[-2], right.shape[-2]), a.device, c, output_dtype, alpha)
        kernel(a, b, result, c=accumulation, alpha=None if alpha == 1.0 else alpha)
        ctx.save_for_backward(a, b, c)
        ctx.layout, ctx.alpha = layout, alpha
        return result

    @staticmethod
    def backward(ctx: Any, grad: Any) -> Any:
        """Return the input gradients required by the operator contract."""
        a, b, c = ctx.saved_tensors
        left, right = canonical_operands(a, b, ctx.layout)
        grad = grad.float()
        grad_a = (grad @ right.float()) * ctx.alpha
        grad_b = (grad.transpose(-1, -2) @ left.float()) * ctx.alpha
        if ctx.layout[0] == "t":
            grad_a = grad_a.transpose(-1, -2)
        if ctx.layout[1] == "n":
            grad_b = grad_b.transpose(-1, -2)
        grad_c = None if c is None else grad.sum_to_size(c.shape).to(c.dtype)
        return grad_a.to(a.dtype), grad_b.to(b.dtype), grad_c, None, None, None, None


def kernel_symbol(extension: object, name: str) -> Callable:
    """Resolve one known ABI symbol after optional-dependency validation."""
    require_symbol(extension, name)
    return getattr(extension, name)


def load_gemm() -> GemmFn:
    """Load fixed DeepGEMM entry points; no arbitrary operator-name dispatch is exposed."""
    extension = require_package("deep_gemm")
    bf16 = {layout: kernel_symbol(extension, f"bf16_gemm_{layout}") for layout in ("nt", "nn", "tn", "tt")}
    quantized = {layout: kernel_symbol(extension, f"fp8_fp4_gemm_{layout}") for layout in ("nt", "nn", "tn", "tt")}

    def run(
        a: Any,
        b: Any,
        *,
        layout: Any="nt",
        c: Any=None,
        output_dtype: Any=torch.bfloat16,
        alpha: Any=1.0,
        recipe: Any=(1, 1, 32),
    ) -> Any:
        """Apply the operator contract using the selected implementation."""
        require_quantized_inference(a, b, c)
        left, right = canonical_operands(a, b, layout)
        if isinstance(a, tuple) != isinstance(b, tuple):
            raise ValueError("Both operands must use the quantized data/scale contract")
        if not isinstance(a, tuple):
            return _DenseGemm.apply(a, b, c, layout, output_dtype, alpha, bf16[layout])
        rows = operand_data(left).shape[-2]
        cols = operand_data(right).shape[-2]
        if operand_data(left).dtype == torch.int8 and operand_data(left).stride(-2) == 1:
            rows *= 2
        if operand_data(right).dtype == torch.int8 and operand_data(right).stride(-2) == 1:
            cols *= 2
        result, accumulation = _output_buffer((rows, cols), a[0].device, c, output_dtype, alpha)
        quantized[layout](a, b, result, c=accumulation, recipe=recipe, alpha=None if alpha == 1.0 else alpha)
        return result

    return run


def _einsum_permutations(expression):
    mapping = {
        "bhr,hdr->bhd": ((1, 0, 2), (0, 1, 2), (1, 0, 2)),
        "bhd,hdr->bhr": ((1, 0, 2), (0, 2, 1), (1, 0, 2)),
        "bhd,bhr->hdr": ((1, 2, 0), (1, 2, 0), (0, 1, 2)),
    }
    if expression not in mapping:
        raise ValueError("Unsupported head einsum expression")
    return mapping[expression]


def native_head_einsum(
    a: Operand,
    b: Operand,
    expression: str,
    *,
    c: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    recipe: Recipe = (1, 1, 32),
) -> torch.Tensor:
    """Decode scales along each expression's contraction axis before multiplication."""
    require_quantized_inference(a, b, c)
    perm_a, perm_b, perm_d = _einsum_permutations(expression)

    def canonical(value: Any, permutation: Any, granularity: Any) -> Any:
        """Canonical."""
        if isinstance(value, tuple):
            return decode_operand(tuple(t.permute(permutation) for t in value), granularity)
        return value.permute(permutation).float()

    left = canonical(a, perm_a, (recipe[0], recipe[2]))
    right = canonical(b, perm_b, (recipe[1], recipe[2]))
    result = (left @ right.transpose(-1, -2)).permute(perm_d)
    return (result if c is None else result + c.float()).to(output_dtype)


class _HeadEinsum(torch.autograd.Function):
    """Differentiate supported contractions without replaying their forward GEMM."""

    @staticmethod
    def forward(ctx: Any, a: Any, b: Any, c: Any, expression: Any, result: Any, kernel: Any) -> Any:
        """Execute the selected backend and retain the tensors needed for backward."""
        kernel(expression, a, b, result, c=c)
        ctx.save_for_backward(a, b, c)
        ctx.expression = expression
        return result

    @staticmethod
    def backward(ctx: Any, grad: Any) -> Any:
        """Return the input gradients required by the operator contract."""
        a, b, c = ctx.saved_tensors
        inputs, output = ctx.expression.split("->")
        left, right = inputs.split(",")
        grad_a = torch.einsum(f"{output},{right}->{left}", grad.float(), b.float()).to(a.dtype)
        grad_b = torch.einsum(f"{output},{left}->{right}", grad.float(), a.float()).to(b.dtype)
        return grad_a, grad_b, None if c is None else grad.to(c.dtype), None, None, None


def load_head_einsum() -> HeadEinsumFn:
    """Load BF16 training and FP8 inference head contractions."""
    extension = require_package("deep_gemm")
    bf16 = kernel_symbol(extension, "einsum")
    fp8 = kernel_symbol(extension, "fp8_einsum")

    def run(
        a: Any,
        b: Any,
        expression: Any,
        *,
        c: Any=None,
        output_dtype: Any=torch.bfloat16,
        recipe: Any=(1, 1, 32),
    ) -> Any:
        """Apply the operator contract using the selected implementation."""
        require_quantized_inference(a, b, c)
        _einsum_permutations(expression)
        inputs, output = expression.split("->")
        dimensions = dict(zip(inputs.replace(",", ""), (*operand_data(a).shape, *operand_data(b).shape)))
        result = torch.empty(tuple(dimensions[i] for i in output), device=operand_data(a).device, dtype=output_dtype)
        if isinstance(a, tuple):
            fp8(expression, a, b, result, c=c, recipe=recipe)
            return result
        return _HeadEinsum.apply(a, b, c, expression, result, bf16)

    return run


def register_gemm(registry: OpRegistry) -> None:
    """Register explicit DeepGEMM implementations without changing auto selection."""
    registry.register_family("gemm", specs=(GEMM, HEAD_EINSUM))
    for spec, native, fused in ((GEMM, native_gemm, load_gemm), (HEAD_EINSUM, native_head_einsum, load_head_einsum)):
        registry.register_impl(spec, name="native", devices=("cpu", "cuda", "npu"), load=lambda fn=native: fn)
        registry.register_impl(spec, name="deep_gemm", devices=("npu",), load=fused)
