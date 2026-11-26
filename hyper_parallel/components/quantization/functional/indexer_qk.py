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
"""CSA2 Indexer Q/K precision policy and identity-STE QDQ boundary."""

from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch import Tensor

from hyper_parallel.components.quantization.config import CSA2IndexerQKFormat, MX_BLOCK_SIZE_32
from hyper_parallel.components.quantization.ops.npu_mxfp4 import _native_qdq
from hyper_parallel.components.quantization.tensor.csa2_cache_tensor import pack_csa2_cache


class IndexerQKPrecisionPolicy(Protocol):
    """Post-architecture-transform Indexer Query/Key precision transform.

    Query is quantized after RoPE. Owned Key is packed once after the optional
    differentiable CP gather; Full scores its decoded value and Reindex reads
    the same physical cache.
    """

    def apply_query(self, query: Tensor) -> Tensor:
        """Quantize and dequantize one Indexer Query tensor in place of use."""

    def encode_owned_key(self, key: Tensor) -> tuple[Tensor, Tensor]:
        """Return the scoring value and its shared physical cache from one write."""


class _MXFP4QDQFunction(torch.autograd.Function):
    """Quantize to MXFP4 and dequantize back to the input dtype.

    Forward runs the real QDQ operator chain inside the Function so autograd
    does not track it; backward is the identity STE defined here, never by the
    operator. ``npu_dynamic_mx_quant`` / ``npu_anti_mx_quant`` expose no own
    backward contract, so the STE must be supplied at this boundary.
    """

    @staticmethod
    def forward(
        ctx: Any, x: Tensor, block_size: int, qdq: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Return ``x`` re-expressed after one MXFP4 quant/dequant round trip."""
        del ctx
        x_2d = x.reshape(-1, x.shape[-1])  # rows = per-head tokens / key rows
        x_hat_2d = _qdq_2d(x_2d, block_size, qdq)
        return x_hat_2d.reshape(x.shape)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        """Return the identity STE; the format carries no gradient."""
        # Propagate student KL gradients across the QDQ boundary via identity STE.
        del ctx
        return grad_output, None, None


class MXFP4FakeQuantPolicy:
    """Stateless MXFP4 Q/K policy. Holds no parameters, buffers or state keys.

    Quantizes the whole last axis in 32-channel blocks independently,
    so per-head tokens (Indexer Query ``[B, S, H, D]``) and key rows never
    borrow elements across heads or tokens to fill a block.
    """

    def _qdq(self, x: Tensor) -> Tensor:
        """Run one fake QDQ round trip through the autograd boundary."""
        return _MXFP4QDQFunction.apply(x, MX_BLOCK_SIZE_32, _native_qdq)

    def apply_query(self, query: Tensor) -> Tensor:
        """Quantize and dequantize the Indexer Query."""
        return self._qdq(query)

    def encode_owned_key(self, key: Tensor) -> tuple[Tensor, Tensor]:
        """Score from the same single packed write published for Reindex."""
        cache = pack_csa2_cache(key, "indexer_k")
        return cache.dequantize(), cache


def create_indexer_qk_policy(
    scheme: CSA2IndexerQKFormat,
) -> IndexerQKPrecisionPolicy:
    """Build the configured OCP MXFP4 QDQ policy."""

    if not isinstance(scheme, CSA2IndexerQKFormat):
        raise TypeError(
            "create_indexer_qk_policy expects a CSA2IndexerQKFormat, "
            f"got {type(scheme).__name__}."
        )
    return MXFP4FakeQuantPolicy()


def _qdq_2d(
    x_2d: Tensor, block_size: int, qdq: Callable[[Tensor], Tensor]
) -> Tensor:
    """Run one MXFP4 QDQ on a 2D tensor, one block per row segment."""
    if x_2d.shape[-1] <= 0:
        raise ValueError("Indexer Q/K last axis must be positive.")
    if x_2d.shape[-1] % block_size:
        raise ValueError(
            f"Indexer Q/K last axis {x_2d.shape[-1]} is not divisible by "
            f"block_size {block_size}; no cross-row padding is allowed"
        )
    if x_2d.numel() == 0:
        return x_2d
    return qdq(x_2d)


__all__ = [
    "IndexerQKPrecisionPolicy",
    "MXFP4FakeQuantPolicy",
    "create_indexer_qk_policy",
]
