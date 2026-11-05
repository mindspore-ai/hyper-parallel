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
"""Adapters for the optional CANN mHC autograd extension."""

from functools import lru_cache
import importlib
import math
from typing import Callable

import torch  # pylint: disable=forbidden-backend-import
from torch.distributed.tensor import DTensor as TorchDTensor

from hyper_parallel.core.dtensor.dtensor import DTensor


def validate_mhc_backend(backend: str) -> None:
    """Reject unknown mHC implementations without importing optional extensions."""
    if backend not in ("omni", "cann"):
        raise ValueError(f"MHC backend must be 'omni' or 'cann', got {backend!r}")


@lru_cache(maxsize=2)
def _load_cann_mhc_op(name: str) -> Callable:
    """Load the Python autograd wrapper, rather than the raw registered op."""
    # The optional package may JIT compile native bridges when first imported.
    try:
        module = importlib.import_module(f"cann_ops_transformer.ops.mhc.{name}")
    except ImportError as err:
        raise RuntimeError(
            f"MHC backend='cann' requires cann_ops_transformer.ops.mhc.{name}. "
            "See hyper_parallel/components/functional/README.md for the CANN mHC requirements."
        ) from err
    return getattr(module, name)


def _validate_cann_tensors(x: torch.Tensor, *others: torch.Tensor) -> None:
    """Require local NPU tensors without silently discarding distributed layouts."""
    tensors = (x, *others)
    if any(isinstance(tensor, (DTensor, TorchDTensor)) for tensor in tensors):
        raise ValueError("MHC backend='cann' requires local tensors; DTensor sharding is not supported")
    if x.device.type != "npu" or any(tensor.device != x.device for tensor in others):
        raise ValueError("MHC backend='cann' requires all inputs on the same NPU")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"CANN mHC activations must be BF16 or FP16, got {x.dtype}")


def cann_mhc_pre(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    num_stream: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    gamma: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapt packed streams to the CANN A2 fused pre/Sinkhorn contract."""
    if gamma is not None:
        raise ValueError("CANN fused MHC pre does not support a normalization gamma")
    if not all(math.isfinite(value) and value > 0 for value in (norm_eps, hc_eps)):
        raise ValueError("CANN MHC eps values must be finite and positive")
    if num_stream != 4 or sinkhorn_iters != 20:
        raise ValueError("CANN A2 fused MHC pre requires four streams and 20 Sinkhorn iterations")
    if x.ndim != 3 or not 0 < x.shape[-1] < num_stream * 100000 or x.shape[-1] % (num_stream * 128) != 0:
        raise ValueError("CANN A2 fused MHC pre requires [B, S, 4*D] with 0 < D < 100000 divisible by 128")
    if phi.shape != (24, x.shape[-1]) or alpha.numel() != 3 or bias.numel() != 24:
        raise ValueError("CANN MHC pre requires phi [24, 4*D], three alpha values and 24 bias values")
    _validate_cann_tensors(x, phi, alpha, bias)
    streams = x.unflatten(-1, (num_stream, x.shape[-1] // num_stream)).contiguous()
    op = _load_cann_mhc_op("mhc_pre_sinkhorn")
    inputs = (
        streams, phi.float().contiguous(), alpha.float().reshape(3).contiguous(),
        bias.float().reshape(24).contiguous(),
    )
    # The optional wrapper tests requires_grad without checking grad mode;
    # detach in inference/reentrant checkpoint forward to omit backward caches.
    if not torch.is_grad_enabled():
        inputs = tuple(tensor.detach() for tensor in inputs)
    hidden, h_post, h_res = op(*inputs, num_stream, sinkhorn_iters, hc_eps, norm_eps)
    # Keep the residual branch in the ordinary autograd graph. The fused pre
    # backward returns only its own contribution to the shared input gradient.
    return hidden, h_post, h_res.unflatten(-1, (num_stream, num_stream)), x


def cann_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    num_stream: int,
) -> torch.Tensor:
    """Combine streams using CANN's differentiable post operator."""
    if x.ndim != 3 or num_stream != 4 or not 0 < x.shape[-1] < 100000 or x.shape[-1] % 128 != 0:
        raise ValueError("CANN A2 MHC post requires [B, S, D], four streams and 0 < D < 100000 divisible by 128")
    if residual.shape != (*x.shape[:2], num_stream * x.shape[-1]):
        raise ValueError("CANN MHC post residual shape must match [B, S, 4*D]")
    if h_post.shape != (*x.shape[:2], num_stream) or h_res.shape != (*x.shape[:2], num_stream, num_stream):
        raise ValueError("CANN MHC post coefficients must have shapes [B, S, 4] and [B, S, 4, 4]")
    _validate_cann_tensors(x, residual, h_post, h_res)
    if residual.dtype != x.dtype or h_post.dtype != torch.float32 or h_res.dtype != torch.float32:
        raise ValueError("CANN MHC post requires matching activation dtypes and FP32 mixing coefficients")
    streams = residual.unflatten(-1, (num_stream, x.shape[-1])).contiguous()
    op = _load_cann_mhc_op("mhc_post")
    return op(streams, h_res.contiguous(), x.contiguous(), h_post.contiguous()).flatten(2)
