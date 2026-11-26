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
"""Expert-major grouped matrix multiplication adapters."""

from __future__ import annotations

from typing import Any

from typing import Protocol

import torch

from .gemm import (
    Operand, Recipe, decode_operand, kernel_symbol, operand_data, require_quantized_inference, transpose_operand,
)
from ..registry import OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol


class GroupedMatmulFn(Protocol):
    """Multiply expert-major rows by weights using cumulative group boundaries."""

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """Multiply expert-major rows without a bias.

        Args:
            x: Token rows of shape ``[T, K]``, already grouped by expert.
            weight: Expert matrices of shape ``[E, K, N]``.
            group_list: One cumulative end per expert. Ends are nondecreasing,
                repeated ends represent empty experts, and the final end is T.

        Returns:
            Token rows of shape ``[T, N]`` in their input order.
        """


GROUPED_MATMUL = OpSpec[GroupedMatmulFn](config_key="grouped_matmul")


def _native(x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
    # One small matmul per expert keeps the reference path independent of
    # grouped kernels without gathering an entire expert weight per token.
    """Multiply token rows by their expert matrices, retaining empty-expert gradients."""
    ends = group_list.tolist()
    starts = [0, *ends[:-1]]
    if len(ends) != weight.shape[0] or (ends[-1] if ends else 0) != x.shape[0]:
        raise ValueError("Grouped matmul boundaries must cover every token and contain one entry per expert")
    if any(start > end for start, end in zip(starts, ends)):
        raise ValueError("Grouped matmul boundaries must be nondecreasing")
    if not ends:
        return x @ weight.sum(dim=0)
    return torch.cat([x[start:end] @ expert for start, end, expert in zip(starts, ends, weight.unbind())])


def _load_native() -> GroupedMatmulFn:
    return _native


def _load_npu() -> GroupedMatmulFn:
    """Load the NPU grouped matmul adapter with cumulative row boundaries."""
    require_symbol(require_package("torch_npu"), "npu_grouped_matmul")
    # The shared autograd adapter imports the optional NPU package.
    from hyper_parallel.components.functional.grouped_matmul import (  # pylint: disable=import-outside-toplevel
        grouped_matmul,
    )

    def run(x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """Translate cumulative row boundaries for the selected grouped matmul kernel."""
        return grouped_matmul(
            x, weight, bias=None, group_list=group_list.to(torch.int64), group_type=0, group_list_type=0,
        )

    return run


def _load_cuda() -> GroupedMatmulFn:
    require_symbol(torch, "_grouped_mm")

    def run(x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """Translate cumulative row boundaries for the selected grouped matmul kernel."""
        return torch._grouped_mm(x, weight, offs=group_list.to(torch.int32))  # pylint: disable=protected-access

    return run


def check_torch_grouped_mm(x: torch.Tensor) -> str | None:
    """Check CUDA requirements shared by grouped matrix multiplication users."""
    if x.dtype != torch.bfloat16:
        return "torch_grouped_mm requires BF16 inputs"
    if torch.cuda.get_device_capability(x.device)[0] < 8:
        return "torch_grouped_mm requires compute capability 8.0 or later"
    return None


class PaddedGroupedGemmFn(Protocol):
    """Multiply M- or K-grouped operands with padded prefix ends."""

    def __call__(
        self,
        a: Operand,
        b: Operand,
        ends: torch.Tensor,
        *,
        layout: str = "nt",
        alignment: int = 256,
        c: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        recipe: Recipe = (1, 1, 32),
    ) -> torch.Tensor:
        """Each group starts at align_up(previous_end, alignment); K-grouped input padding must be zero."""


M_GROUPED_GEMM = OpSpec[PaddedGroupedGemmFn](config_key="grouped_matmul", variant="m_padded")
K_GROUPED_GEMM = OpSpec[PaddedGroupedGemmFn](config_key="grouped_matmul", variant="k_padded")


def _slices(ends, alignment):
    start = 0
    for end in ends.tolist():
        if end < start:
            raise ValueError("Padded prefix ends cannot precede their aligned group starts")
        yield start, end
        start = (end + alignment - 1) // alignment * alignment


def native_m_grouped(
    a: Operand,
    b: Operand,
    ends: torch.Tensor,
    *,
    layout: str = "nt",
    alignment: int = 256,
    c: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    recipe: Recipe = (1, 1, 32),
) -> torch.Tensor:
    """Reference expert-major GEMM with zero output in padded rows."""
    if layout not in ("nt", "nn") or c is not None:
        raise ValueError("M-grouped GEMM supports nt/nn and no accumulation tensor")
    require_quantized_inference(a, b)
    left = decode_operand(a, (recipe[0], recipe[2]))
    right = decode_operand(transpose_operand(b) if layout == "nn" else b, (recipe[1], recipe[2]))
    pieces = []
    previous = 0
    for expert, (start, end) in enumerate(_slices(ends, alignment)):
        pieces.append(left.new_zeros(start - previous, right.shape[-2]))
        pieces.append(left[start:end] @ right[expert].T)
        previous = end
    pieces.append(left.new_zeros(left.shape[0] - previous, right.shape[-2]))
    return torch.cat(pieces).to(output_dtype)


def native_k_grouped(
    a: Operand,
    b: Operand,
    ends: torch.Tensor,
    *,
    layout: str = "tn",
    alignment: int = 256,
    c: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    recipe: Recipe = (1, 1, 32),
) -> torch.Tensor:
    """Reference group reductions along padded K, preserving empty expert gradients."""
    if layout not in ("tn", "nt"):
        raise ValueError("K-grouped GEMM supports tn/nt")
    require_quantized_inference(a, b, c)
    left = decode_operand(transpose_operand(a) if layout == "tn" else a, (recipe[0], recipe[2]))
    right = decode_operand(transpose_operand(b) if layout == "tn" else b, (recipe[1], recipe[2]))
    result = torch.stack([left[:, start:end] @ right[:, start:end].T for start, end in _slices(ends, alignment)])
    return (result if c is None else result + c.float()).to(output_dtype)


class _PaddedGroupedGemm(torch.autograd.Function):
    """Use the opposite grouped contraction for VJPs without reading device prefix ends on host."""

    @staticmethod
    def forward(
        ctx: Any,
        a: Any,
        b: Any,
        ends: Any,
        c: Any,
        mode: Any,
        layout: Any,
        output_dtype: Any,
        kernels: Any,
    ) -> Any:
        """Execute the selected backend and retain the tensors needed for backward."""
        right = b.transpose(-1, -2) if (mode == "m" and layout == "nn") else b
        if mode == "k" and layout == "nt":
            a, right = a.T, right.T
        shape = (a.shape[0], right.shape[-2]) if mode == "m" else (ends.numel(), a.shape[1], right.shape[1])
        result = torch.empty(shape, device=a.device, dtype=output_dtype)
        if mode == "m":
            kernels["m_nt"](a, right, result, ends, use_psum_layout=True, ensure_zero_padding=True)
        else:
            kernels["k_tn"](a, right, result, None, ends, c=c, use_psum_layout=True)
        ctx.save_for_backward(a, right, ends, c)
        ctx.mode, ctx.layout, ctx.kernels = mode, layout, kernels
        return result

    @staticmethod
    def backward(ctx: Any, grad: Any) -> Any:
        """Return the input gradients required by the operator contract."""
        a, b, ends, c = ctx.saved_tensors
        kernels = ctx.kernels
        # DeepGEMM's BF16 VJP kernels consume BF16 operands, even with FP32 forward output.
        if grad.dtype != a.dtype:
            raise RuntimeError("DeepGEMM grouped training requires BF16 output; FP32 output is inference-only")
        grad_a, grad_b = torch.empty_like(a), torch.empty_like(b)
        if ctx.mode == "m":
            kernels["m_nn"](grad, b, grad_a, ends, use_psum_layout=True, ensure_zero_padding=True)
            kernels["k_tn"](grad, a, grad_b, None, ends, use_psum_layout=True)
            if ctx.layout == "nn":
                grad_b = grad_b.transpose(-1, -2)
        else:
            kernels["m_nt"](b, grad, grad_a, ends, use_psum_layout=True, ensure_zero_padding=True)
            kernels["m_nn"](a, grad, grad_b, ends, use_psum_layout=True, ensure_zero_padding=True)
            if ctx.layout == "nt":
                grad_a, grad_b = grad_a.T, grad_b.T
        return grad_a, grad_b, None, None if c is None else grad, None, None, None, None


def _quantized_grouped(a, b, ends, mode, layout, c, output_dtype, recipe, quant):
    data_a, data_b = operand_data(a), operand_data(b)
    if mode == "m":
        n = data_b.shape[-2 if layout == "nt" else -1]
        if data_b.dtype == torch.int8 and data_b.stride(-2 if layout == "nt" else -1) == 1:
            n *= 2
        result = torch.empty((data_a.shape[0], n), device=data_a.device, dtype=output_dtype)
        quant[f"m_{layout}"](a, b, result, ends, recipe=recipe, use_psum_layout=True, ensure_zero_padding=True)
    else:
        axis = 1 if layout == "tn" else 0
        m, n = data_a.shape[axis], data_b.shape[axis]
        if data_a.dtype == torch.int8 and data_a.stride(axis) == 1:
            m *= 2
        if data_b.dtype == torch.int8 and data_b.stride(axis) == 1:
            n *= 2
        result = torch.empty((ends.numel(), m, n), device=data_a.device, dtype=output_dtype)
        quant[f"k_{layout}"](a, b, result, None, ends, c=c, recipe=recipe, use_psum_layout=True)
    return result


def _require_grouped_training_dtype(a, b, c, output_dtype):
    if (output_dtype != a.dtype and torch.is_grad_enabled()
            and any(t is not None and t.requires_grad for t in (a, b, c))):
        raise RuntimeError("DeepGEMM grouped training requires BF16 output")


def load_padded_grouped(mode: str) -> PaddedGroupedGemmFn:
    """Load known M/K grouped symbols, retaining the package's alignment policy."""
    extension = require_package("deep_gemm")
    alignment_fn = kernel_symbol(extension, "get_mk_alignment_for_contiguous_layout")
    kernels = {
        "m_nt": kernel_symbol(extension, "m_grouped_bf16_gemm_nt_contiguous"),
        "m_nn": kernel_symbol(extension, "m_grouped_bf16_gemm_nn_contiguous"),
        "k_tn": kernel_symbol(extension, "k_grouped_bf16_gemm_tn_contiguous"),
    }
    quant = {
        "m_nt": kernel_symbol(extension, "m_grouped_fp8_fp4_gemm_nt_contiguous"),
        "m_nn": kernel_symbol(extension, "m_grouped_fp8_fp4_gemm_nn_contiguous"),
        "k_nt": kernel_symbol(extension, "k_grouped_fp8_gemm_nt_contiguous"),
        "k_tn": kernel_symbol(extension, "k_grouped_fp8_gemm_tn_contiguous"),
    }

    def run(
        a: Any,
        b: Any,
        ends: Any,
        *,
        layout: Any=None,
        alignment: Any=256,
        c: Any=None,
        output_dtype: Any=torch.bfloat16,
        recipe: Any=(1, 1, 32),
    ) -> Any:
        """Apply the operator contract using the selected implementation."""
        layout = layout or ("nt" if mode == "m" else "tn")
        if alignment != alignment_fn():
            raise ValueError("Padded prefix alignment must match DeepGEMM's configured MK alignment")
        if mode == "m" and (layout not in ("nt", "nn") or c is not None):
            raise ValueError("M-grouped GEMM supports nt/nn and no accumulation tensor")
        if mode == "k" and layout not in ("tn", "nt"):
            raise ValueError("K-grouped GEMM supports tn/nt")
        require_quantized_inference(a, b, c)
        if not isinstance(a, tuple):
            _require_grouped_training_dtype(a, b, c, output_dtype)
            return _PaddedGroupedGemm.apply(a, b, ends, c, mode, layout, output_dtype, kernels)
        return _quantized_grouped(a, b, ends, mode, layout, c, output_dtype, recipe, quant)

    return run


def register_padded_grouped_gemm(registry: OpRegistry) -> None:
    """Extend grouped_matmul with explicit padded M/K variants."""
    for spec, native, mode in ((M_GROUPED_GEMM, native_m_grouped, "m"), (K_GROUPED_GEMM, native_k_grouped, "k")):
        registry.register_impl(spec, name="native", devices=("cpu", "cuda", "npu"), load=lambda fn=native: fn)
        registry.register_impl(spec, name="deep_gemm", devices=("npu",), load=lambda key=mode: load_padded_grouped(key))


def register_grouped_matmul(registry: OpRegistry) -> None:
    """Register one grouped GEMM over local expert-major token rows."""
    registry.register_family("grouped_matmul", specs=(GROUPED_MATMUL, M_GROUPED_GEMM, K_GROUPED_GEMM))
    register_padded_grouped_gemm(registry)
    registry.register_impl(GROUPED_MATMUL, name="native", devices=("cuda", "cpu", "npu"), load=_load_native)
    registry.register_impl(GROUPED_MATMUL, name="torch_npu", devices=("npu",), load=_load_npu)
    registry.register_impl(
        GROUPED_MATMUL, name="torch_grouped_mm", devices=("cuda",), load=_load_cuda,
        check_compatibility=check_torch_grouped_mm,
    )
