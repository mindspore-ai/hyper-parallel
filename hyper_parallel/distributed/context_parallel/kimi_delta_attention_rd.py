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
"""Cached eager recursive doubling with explicit transport qualification."""
from __future__ import annotations

# pylint: disable=forbidden-backend-import

from dataclasses import dataclass
from functools import lru_cache
import importlib
from typing import Any

import torch
import torch.distributed as dist


@lru_cache(maxsize=1)
def _rd_kernels() -> Any:
    """Load optional Ascend kernels only when an NPU epilogue is executed."""
    return importlib.import_module(
        "hyper_parallel.components.functional._triton.kimi_delta_attention.rd_pointwise"
    )


def _finish_round(product: torch.Tensor, previous: torch.Tensor, save_m: bool = True) -> torch.Tensor | None:
    """Add the previous S and copy the next M in one launch, without fusing GEMM.

    Inputs must be distinct contiguous FP32 [B,H,128,256] buffers. Only the
    newly computed product's S half is modified; input summaries are immutable.

    Args:
        product: Fresh contiguous FP32 packed matrix product [B, H, 128, 256].
        previous: Previous immutable packed S/M tensor with the same layout.
        save_m: Whether a future round consumes the new contiguous M.
    """
    if (product.shape != previous.shape or product.shape[-2:] != (128, 256)
            or product.dtype != torch.float32 or previous.dtype != torch.float32
            or not product.is_contiguous() or not previous.is_contiguous()
            or product.device != previous.device or product.data_ptr() == previous.data_ptr()):
        raise ValueError("Expected distinct contiguous FP32 packed products on the same device")
    matrix = (torch.empty((*product.shape[:-1], 128), dtype=product.dtype, device=product.device)
              if save_m else None)
    if product.device.type == "cpu":
        product[..., :128].add_(previous[..., :128])
        if save_m:
            matrix.copy_(product[..., 128:])
        return matrix
    # Optional Ascend Triton kernels are not needed by CPU protocol checks.
    kernels = _rd_kernels()

    rows = product.numel() // 256
    kernels.kda_rd_finish_round_kernel[((rows + 63) // 64,)](
        product, previous, matrix if save_m else product,
        row_count=rows, block_rows=64, save_matrix=save_m, num_warps=4)
    return matrix


def _finish_to_state(
    product: torch.Tensor, previous: torch.Tensor, save_m: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Emit contiguous final-stage S alongside the unchanged packed epilogue.

    Args:
        product: Fresh contiguous FP32 packed matrix product [B, H, 128, 256].
        previous: Previous immutable packed S/M tensor with the same layout.
        save_m: Whether a future round consumes the new contiguous M.
    """
    state = torch.empty((*product.shape[:-1], 128), dtype=product.dtype, device=product.device)
    matrix = torch.empty_like(state) if save_m else None
    if product.device.type == "cpu":
        product[..., :128].add_(previous[..., :128])
        state.copy_(product[..., :128])
        if save_m:
            matrix.copy_(product[..., 128:])
        return matrix, state
    # Optional Ascend Triton kernels are not needed by CPU protocol checks.
    kernels = _rd_kernels()

    rows = product.numel() // 256
    kernels.kda_rd_finish_to_state_kernel[((rows + 63) // 64,)](
        product, previous, matrix if save_m else product, state, row_count=rows, save_matrix=save_m, num_warps=4)
    return matrix, state


@dataclass(frozen=True)
class TransportChoice:
    """Observable choice of the same-arithmetic public/coalesced transport."""
    direct: bool
    reason: str


def select_transport(torch_version: str, npu_version: str, cann_version: str,
                     device_name: str, available: bool, force_public: bool = False) -> TransportChoice:
    """Select only the qualified eager implementation, never graph support.

    Args:
        torch_version: Installed framework version.
        npu_version: Installed NPU extension version.
        cann_version: Installed CANN toolkit version.
        device_name: Runtime NPU device name.
        available: Whether HCCL eager coalescing is callable.
        force_public: Explicitly select the public batch transport.
    """
    supported = (torch_version.split('+')[0] == '2.10.0'
                 and npu_version.split('+')[0] == '2.10.0'
                 and cann_version in ('9.1.0.beta3', '9.1.0-beta.3')
                 and device_name == 'Ascend910B3' and available)
    if force_public:
        return TransportChoice(False, 'public transport explicitly requested')
    reason = 'qualified eager coalescing' if supported else 'unqualified environment: public fallback'
    return TransportChoice(supported, reason)


class RoundExchange:
    """Keep immutable payload alive until send completion; outputs belong to the call."""

    def __init__(self, group: Any, rank: int, ranks: tuple[int, ...], force_public: bool = False) -> None:
        """Resolve ordered peers and qualify the optional NPU eager transport."""
        if not ranks or tuple(sorted(set(ranks))) != ranks or not 0 <= rank < len(ranks):
            raise ValueError("RD requires ordered unique global peers and a valid local rank")
        self.group, self.rank, self.size, self.peers = group, rank, len(ranks), ranks
        self.manager = getattr(dist.distributed_c10d, "_coalescing_manager", None)
        self.device = torch.device("cpu")
        self.choice = TransportChoice(False, "public CPU qualification transport")
        backend = dist.get_backend(group)
        if backend == "hccl":
            # torch-npu is optional; CPU protocol qualification must not load it.
            torch_npu = importlib.import_module("torch_npu")
            get_cann_version = importlib.import_module("torch_npu.utils").get_cann_version

            self.device = torch.device("npu", torch.npu.current_device())
            self.choice = select_transport(
                torch.__version__, torch_npu.__version__, get_cann_version("CANN"),
                torch.npu.get_device_name(), callable(self.manager), force_public)
        elif backend != "gloo":
            raise ValueError("Cached RD is qualified for HCCL and CPU Gloo only")

    def __call__(self, payload: torch.Tensor, destination: int | None, source: int | None) -> torch.Tensor | None:
        """Exchange group-local peers on the caller stream with completion dependencies."""
        if payload.device != self.device or not payload.is_contiguous():
            raise ValueError('RD payload must be contiguous on the protocol device')
        if any(peer is not None and not 0 <= peer < self.size for peer in (source, destination)):
            raise ValueError('RD peer is outside the process group')
        output = torch.empty_like(payload) if source is not None else None
        if source is None and destination is None:
            return output
        if self.choice.direct:
            with self.manager(group=self.group, device=self.device, async_ops=True) as manager:
                if source is not None:
                    dist.irecv(output, group=self.group, group_src=source)
                if destination is not None:
                    dist.isend(payload, group=self.group, group_dst=destination)
            manager.wait()
        else:
            operations = []
            if source is not None:
                operations.append(dist.P2POp(dist.irecv, output, self.peers[source], self.group))
            if destination is not None:
                operations.append(dist.P2POp(dist.isend, payload, self.peers[destination], self.group))
            for work in dist.batch_isend_irecv(operations):
                work.wait()
        return output


def _validate_forward_summary(state: torch.Tensor, matrix: torch.Tensor) -> None:
    """Reject incompatible boundary inputs before any peer exchange."""
    if (state.ndim != 4 or state.device != matrix.device
            or state.shape != matrix.shape or state.shape[-2:] != (128, 128)
            or matrix.dtype != torch.float32 or state.dtype != torch.float32
            or not matrix.is_contiguous() or not state.is_contiguous()):
        raise ValueError("Cached RD requires contiguous FP32 K=V128 inputs")


def _validate_backward_summary(gradient: torch.Tensor, cache: tuple, rank: int, distances: tuple) -> None:
    """Reject incompatible invocation cache layouts before any peer exchange."""
    if (gradient.ndim != 4 or gradient.shape[-2:] != (128, 128)
            or gradient.dtype != torch.float32 or not gradient.is_contiguous()):
        raise ValueError('RD gradient must be contiguous FP32 B/H/K128/V128')
    expected = sum(rank >= distance for distance in distances)
    if not isinstance(cache, tuple) or len(cache) != expected:
        raise ValueError("Layout RD received an incompatible invocation cache")
    if any(matrix.shape != gradient.shape or matrix.dtype != gradient.dtype or matrix.device != gradient.device
           or not matrix.is_contiguous()
           for matrix in cache):
        raise ValueError('RD cache tensor contract differs from the invocation gradient')


def _compact_transfer(matrix: torch.Tensor) -> torch.Tensor:
    """Keep a contiguous M view from retaining a larger producer allocation."""
    if matrix.untyped_storage().nbytes() != matrix.numel() * matrix.element_size():
        return matrix.clone()
    return matrix


class CachedRecursiveDoubling:
    """Exclusive forward scan and its cached adjoint, preserving the established rounding.

    Caller creates/warm-ups the HCCL group collectively. Inputs are immutable,
    contiguous FP32 B/H/K128/V128 summaries. Keep the returned invocation cache
    with its own autograd context. No shared graph buffers or mutable layer state.
    """

    requires_contiguous_summaries = True

    def __init__(self, group: Any, rank: int, ranks: tuple[int, ...], force_public: bool = False) -> None:
        """Prepare communication metadata and logarithmic scan distances."""
        self.exchange = RoundExchange(group, rank, ranks, force_public)
        self.rank, self.size = self.exchange.rank, self.exchange.size
        self.distances = tuple(1 << level for level in range((self.size - 1).bit_length()))

    @property
    def transport_choice(self) -> TransportChoice:
        """Expose the selected path to manifests and the future planner."""
        return self.exchange.choice

    def forward(self, state: torch.Tensor, matrix: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return the exclusive boundary and immutable pre-round transition cache.

        Args:
            state: Local summary S tensor or kernel output.
            matrix: Contiguous FP32 local transfer M.
        """
        _validate_forward_summary(state, matrix)
        matrix = _compact_transfer(matrix)
        # CP1/2 have no packed stage. For larger CP this preserves the original cat layout.
        packed = torch.cat((state, matrix), dim=-1) if self.size > 2 else None
        before_m, narrow_state = matrix, state
        cache = []
        for distance in self.distances:
            final = distance == self.distances[-1]
            source = self.rank - distance if self.rank >= distance else None
            destination = self.rank + distance if self.rank + distance < self.size else None
            received = self.exchange(narrow_state if final else packed, destination, source)
            if source is not None:
                cache.append(before_m)
                with torch.profiler.record_function("transport/doubling/compose"):
                    if final:
                        narrow_state = before_m @ received + narrow_state
                    else:
                        aggregate = before_m @ received
                        if 2 * distance == self.distances[-1]:
                            before_m, narrow_state = _finish_to_state(aggregate, packed, self.rank >= 2 * distance)
                        else:
                            before_m = _finish_round(aggregate, packed, save_m=self.rank >= 2 * distance)
                        packed = aggregate
            elif not final and 2 * distance == self.distances[-1] and self.rank:
                # Low ranks can stop receiving before the penultimate stage.
                # Their last inclusive S still replaces the original local S.
                narrow_state = packed[..., :128].contiguous()
        received = self.exchange(narrow_state, self.rank + 1 if self.rank + 1 < self.size else None,
                                 self.rank - 1 if self.rank else None)
        return torch.zeros_like(state) if received is None else received, tuple(cache)

    def backward(self, gradient: torch.Tensor, cache: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Transpose the exact recorded forward composition order.

        Args:
            gradient: Contiguous FP32 local boundary-gradient summary G.
            cache: Invocation-owned transfers returned by this protocol forward.
        """
        _validate_backward_summary(gradient, cache, self.rank, self.distances)
        received = self.exchange(gradient, self.rank - 1 if self.rank else None,
                                 self.rank + 1 if self.rank + 1 < self.size else None)
        adjoint = torch.zeros_like(gradient) if received is None else received
        index = len(cache)
        for distance in reversed(self.distances):
            destination = self.rank - distance if self.rank >= distance else None
            source = self.rank + distance if self.rank + distance < self.size else None
            payload = adjoint
            if destination is not None:
                index -= 1
                payload = cache[index].transpose(-2, -1) @ adjoint
            received = self.exchange(payload, destination, source)
            if received is not None:
                # Adjoint is invocation-owned and every send has completed.
                adjoint.add_(received)
        return adjoint
