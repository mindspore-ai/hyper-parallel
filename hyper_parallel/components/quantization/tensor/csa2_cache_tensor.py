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
"""Autograd-bearing physical CSA2 KV cache, backed only by packed bytes."""

from collections.abc import Callable
from typing import Any, Literal, Optional

import torch
from torch import Tensor

from hyper_parallel.components.quantization.ops.npu_fp8 import allow_fully_written_cache_allocation
from hyper_parallel.components.quantization.ops.npu_mxfp4 import pack_indexer_k, unpack_indexer_k
from hyper_parallel.components.quantization.quantizers.main_kv import pack_main_kv, unpack_main_kv
from hyper_parallel.components.quantization.quantizers.swa_kv import pack_swa_kv, unpack_swa_kv
from hyper_parallel.components.quantization.tensor.quantized_tensor import QuantizedTensor

CSA2CacheFormat = Literal["indexer_k", "main_kv", "swa_kv"]
CSA2ScaleFormat = Literal["e8m0", "e4m3", "e8m7"]


class CSA2CacheTensor(QuantizedTensor):
    """Logical floating tensor whose only physical storage is a packed byte row."""

    def __new__(cls, packed: Tensor, logical_shape: tuple[int, ...], logical_dtype: torch.dtype,
                cache_format: CSA2CacheFormat, scale_format: CSA2ScaleFormat,
                requires_grad: bool = False) -> "CSA2CacheTensor":
        """Expose logical shape and dtype without allocating dense storage."""
        return QuantizedTensor.__new__(
            cls, logical_shape, logical_dtype, device=packed.device, requires_grad=requires_grad,
        )

    def __init__(self, packed: Tensor, logical_shape: tuple[int, ...], logical_dtype: torch.dtype,
                 cache_format: CSA2CacheFormat, scale_format: CSA2ScaleFormat,
                 requires_grad: bool = False) -> None:
        """Attach physical bytes after the logical wrapper has been allocated."""
        del logical_shape, logical_dtype, requires_grad
        self.packed = packed
        self.cache_format = cache_format
        self.scale_format = scale_format

    def _new_with_transformed_data(self, transform: Callable[[Tensor], Tensor], *,
                                   requires_grad: bool) -> "CSA2CacheTensor":
        """Preserve the physical bytes when a wrapper transformation is requested."""
        return type(self)(transform(self.packed), tuple(self.shape), self.dtype,
                          self.cache_format, self.scale_format, requires_grad)

    @property
    def packed_nbytes(self) -> int:
        """Physical data bytes, including quantization scales."""
        return self.packed.numel()

    def dequantize(self) -> Tensor:
        """Materialize a floating tensor only at a computation consumer."""
        return _ReadCSA2Cache.apply(self)

    def dequantize_selected(self, indices: Tensor) -> Tensor:
        """Decode selected rows; invalid IDs return zero and receive no gradient.

        The current reference KL uses this for its detached teacher under
        ``no_grad``. Its backward runs only for a differentiable consumer;
        main attention currently uses the full-cache read instead.
        """
        valid_main = self.cache_format == "main_kv" and self.ndim == 3
        valid_swa = self.cache_format == "swa_kv" and self.ndim == 4 and self.shape[1] == 1
        if not (valid_main or valid_swa) or indices.ndim != 3:
            raise ValueError("Selected reads require [B, K, D] Main or [B, 1, K, D] SWA cache.")
        if indices.shape[0] != self.shape[0]:
            raise ValueError("Selected cache indices must share the cache batch size.")
        if indices.dtype not in (torch.int32, torch.int64) or indices.device != self.device:
            raise ValueError("Selected cache indices must be int32 or int64 on the cache device.")
        return _ReadSelectedCSA2Cache.apply(self, indices)


class _WriteCSA2Cache(torch.autograd.Function):
    """Autograd boundary that stores bytes while exposing a logical STE value."""

    @staticmethod
    def forward(ctx: Any, value: Tensor, cache_format: CSA2CacheFormat,
                scale_format: CSA2ScaleFormat) -> CSA2CacheTensor:
        """Pack one logical tensor and retain only its byte representation."""
        del ctx
        with allow_fully_written_cache_allocation():
            if cache_format == "indexer_k":
                packed = pack_indexer_k(value)
            elif cache_format == "main_kv":
                packed = pack_main_kv(value, scale_format)
            else:
                packed = pack_swa_kv(value, scale_format)
        return CSA2CacheTensor(packed, tuple(value.shape), value.dtype,
                               cache_format, scale_format, value.requires_grad)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        """Pass logical gradients through the quantized write boundary."""
        # Packed uint8 bytes are non-differentiable; pass floating gradients back via identity STE.
        del ctx
        return grad_output, None, None


class _ReadCSA2Cache(torch.autograd.Function):
    """Dense cache read for consumers without a selected-row implementation."""

    @staticmethod
    def forward(ctx: Any, cache: CSA2CacheTensor) -> Tensor:
        """Decode the complete physical cache for a compatible consumer."""
        del ctx
        width = cache.shape[-1]
        with allow_fully_written_cache_allocation():
            if cache.cache_format == "indexer_k":
                return unpack_indexer_k(cache.packed, width, cache.dtype)
            if cache.cache_format == "main_kv":
                return unpack_main_kv(cache.packed, width, cache.dtype, cache.scale_format)
            return unpack_swa_kv(cache.packed, width, cache.scale_format)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor]:
        """Return the decoded consumer gradient to the logical cache."""
        # Forward the consumer's floating gradient unchanged to the upstream packed write boundary.
        del ctx
        return (grad_output,)


class _ReadSelectedCSA2Cache(torch.autograd.Function):
    """Sparse packed-row read with row-wise gradient accumulation."""

    @staticmethod
    def forward(ctx: Any, cache: CSA2CacheTensor, indices: Tensor) -> Tensor:
        """Gather packed rows first, then decode only those selected rows."""
        batch = cache.shape[0]
        length, width = cache.shape[-2:]
        valid = (indices >= 0) & (indices < length)
        safe = indices.long().clamp(min=0, max=max(length - 1, 0))
        ctx.save_for_backward(safe, valid)
        ctx.bank_shape = (batch, length, width)
        ctx.is_swa = cache.cache_format == "swa_kv"
        ctx.scale_format = cache.scale_format
        if length == 0 or indices.numel() == 0:
            return torch.zeros((*indices.shape, width), dtype=cache.dtype, device=cache.device)
        with allow_fully_written_cache_allocation():
            offsets = torch.arange(batch, device=cache.device).view(-1, 1, 1) * length
            rows = cache.packed.reshape(batch * length, -1).index_select(0, (safe + offsets).reshape(-1))
            packed = rows.reshape(*indices.shape, cache.packed.shape[-1])
            decoded = (unpack_swa_kv(packed, width, ctx.scale_format) if ctx.is_swa
                       else unpack_main_kv(packed, width, cache.dtype, ctx.scale_format))
            return decoded.masked_fill(~valid.unsqueeze(-1), 0)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        """Accumulate selected-row gradients by row, including repeated IDs."""
        safe, valid = ctx.saved_tensors
        batch, length, width = ctx.bank_shape
        # IndexAdd consumes one index per row. ScatterAdd instead broadcasts
        # each row index over D channels and is particularly slow on Ascend.
        # FP32 accumulation also preserves duplicate-row sums before the cast.
        grad_bank = torch.zeros((batch * length, width), device=grad_output.device, dtype=torch.float32)
        if safe.numel() and length:
            offsets = torch.arange(batch, device=safe.device).view(-1, 1, 1) * length
            row_ids = (safe + offsets).reshape(-1)
            grad_values = grad_output.masked_fill(~valid.unsqueeze(-1), 0).reshape(-1, width).float()
            grad_bank.index_add_(0, row_ids, grad_values)
        grad_bank = grad_bank.reshape(ctx.bank_shape).to(grad_output.dtype)
        return grad_bank.unsqueeze(1) if ctx.is_swa else grad_bank, None


def pack_csa2_cache(value: Tensor, cache_format: CSA2CacheFormat,
                    scale_format: Optional[CSA2ScaleFormat] = None) -> CSA2CacheTensor:
    """Create a packed cache tensor with an identity STE to its producer."""
    if cache_format not in ("indexer_k", "main_kv", "swa_kv"):
        raise ValueError(f"Unsupported CSA2 cache format: {cache_format!r}.")
    default_scale = {"indexer_k": "e8m0", "main_kv": "e4m3", "swa_kv": "e8m0"}
    effective_scale = default_scale[cache_format] if scale_format is None else scale_format
    allowed_scales = {"indexer_k": ("e8m0",), "main_kv": ("e4m3", "e8m7"),
                      "swa_kv": ("e8m0", "e8m7")}
    if effective_scale not in allowed_scales[cache_format]:
        raise ValueError(f"Unsupported {cache_format} scale format: {effective_scale!r}.")
    return _WriteCSA2Cache.apply(value, cache_format, effective_scale)


__all__ = ["CSA2CacheTensor", "pack_csa2_cache"]
