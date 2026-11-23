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
"""Forward-only symmetric products implemented with Triton-Ascend."""

import importlib
from functools import lru_cache
from typing import Optional

import torch


def _check_layout(tensor: torch.Tensor) -> None:
    """Keep each matrix dense while allowing its last-two-axis transpose."""
    valid = tensor.layout == torch.strided
    if valid and not tensor.is_contiguous():
        shape, strides = tensor.shape, tensor.stride()
        valid = (strides[-2] == 1 and strides[-1] == shape[-2]
                 and (tensor.ndim == 2 or shape[0] <= 1 or strides[0] == shape[-2] * shape[-1]))
    if not valid:
        raise ValueError("symmetric_matmul requires contiguous matrices or their last-two-axis transposes")


def _require_npu(tensor: torch.Tensor) -> None:
    """Reject unsupported devices before loading the optional backend."""
    if tensor.device.type != "npu":
        raise ValueError("symmetric_matmul requires Ascend NPU tensors")


def _check_output(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor, shape: tuple, dtype: torch.dtype) -> None:
    """Reject unsupported destinations before any kernel can write to them."""
    if (out.shape != shape or out.dtype != dtype or out.device != a.device
            or out.layout != torch.strided or not out.is_contiguous()):
        raise ValueError("out must be contiguous with the output shape, dtype and device")
    for operand in (a, b):
        if (out.numel() and operand.numel()
                and out.untyped_storage().data_ptr() == operand.untyped_storage().data_ptr()):
            raise ValueError("out must not share storage with an input")


def _check_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    """Check that the operands describe a supported square product."""
    if a.ndim not in (2, 3) or b.ndim != a.ndim:
        raise ValueError("a and b must both be rank 2 or rank 3")
    if a.shape[:-2] != b.shape[:-2] or a.shape[-1] != b.shape[-2] or a.shape[-2] != b.shape[-1]:
        raise ValueError("Expected a [..., N, K] and b [..., K, N] with identical batch dimensions")
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise ValueError("a and b must have the same FP16 or BF16 dtype")
    if a.device != b.device:
        raise ValueError("a and b must be on the same device")
    _check_layout(a)
    _check_layout(b)


def _validate(
    a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor], out_dtype: Optional[torch.dtype],
) -> tuple[tuple[int, ...], torch.dtype]:
    """Check the forward-only contract using tensor metadata."""
    _check_inputs(a, b)
    shape = (*a.shape[:-2], a.shape[-2], a.shape[-2])
    dtype = a.dtype if out_dtype is None else out_dtype
    if dtype not in (a.dtype, torch.float32):
        raise ValueError("out_dtype must be the input dtype or torch.float32")
    n, k = a.shape[-2:]
    # Batch bases are wide, while the matrix coordinates passed to tiling are still int32.
    if (a.ndim == 2 or a.shape[0] > 0) and max(n * k, n * n) >= 2**31:
        raise ValueError("Per-matrix shape exceeds the symmetric_matmul indexing range")
    if out is not None:
        _check_output(out, a, b, shape, dtype)
    tensors = (a, b) if out is None else (a, b, out)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise ValueError("symmetric_matmul is forward-only; use it under torch.no_grad()")
    return shape, dtype


@lru_cache(maxsize=1)
def _triton_backend():
    """Keep optional compiler/runtime dependencies off the ordinary import path."""
    try:
        return importlib.import_module("hyper_parallel.components.functional._triton.symmetric_matmul")
    except (ImportError, OSError) as error:
        raise ImportError(
            "symmetric_matmul requires compatible Triton-Ascend and torch-npu installations; "
            "see hyper_parallel/components/functional/README.md"
        ) from error


def symmetric_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute a mathematically symmetric ``a @ b`` with Triton-Ascend.

    Args:
        a: FP16/BF16 NPU matrix ``[N, K]`` or batch ``[B, N, K]``.
        b: Matching matrix ``[K, N]`` or batch ``[B, K, N]``. May alias a.
        out: Optional contiguous destination sharing no storage with either input.
        out_dtype: Input dtype by default; ``torch.float32`` retains FP32 accumulators.

    Returns:
        A complete ``[..., N, N]`` matrix. Each rounded lower-triangle value is
        also written to its upper-triangle mirror, giving bitwise-identical triangles.

    Raises:
        ValueError: Unsupported shape, layout, dtype, device, aliasing or active autograd.
        ImportError: The optional Triton-Ascend backend cannot be loaded.

    Note:
        The caller guarantees that the mathematical product is symmetric; this
        condition is not checked. Inputs must be contiguous or last-two-axis
        transposes of contiguous tensors. Broadcasting and backward are unsupported.
        FP32 accumulation is used with no scaling or additive epilogue. Batch offsets
        use 64-bit indices, while matrix and task indices use signed 32-bit indices.
        This function does not change CANN settings. Bitwise parity with native matmul is not
        guaranteed; NPU reference comparisons should set ``CLOSE_MATMUL_K_SHIFT=1``
        before process startup. Compilation and device execution errors propagate.
    """
    shape, dtype = _validate(a, b, out, out_dtype)
    _require_npu(a)
    result = a.new_empty(shape, dtype=dtype) if out is None else out
    if result.numel() == 0:
        return result
    if a.shape[-1] == 0:
        return result.zero_()
    _triton_backend().launch_symmetric_matmul(a, b, result)
    return result
