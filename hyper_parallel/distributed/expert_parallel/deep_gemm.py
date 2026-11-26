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

"""Inference-only MegaMoE symmetric buffer ownership and backend layout utilities.

These interfaces own collective resources or physical encodings, not portable
mathematical operators. Dense and grouped computation lives in components.ops.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

QuantTensor = tuple[torch.Tensor, torch.Tensor]


def _require_inference(*values):
    if not torch.is_grad_enabled():
        return
    for value in values:
        tensors = value if isinstance(value, tuple) else (value,)
        if any(isinstance(tensor, torch.Tensor) and tensor.requires_grad for tensor in tensors):
            raise RuntimeError("MegaMoE and physical weight transforms are inference-only; no backward is defined")


def transform_mega_moe_weights(l1_weights: QuantTensor, l2_weights: QuantTensor,
                               activation: str = "swiglu") -> tuple[QuantTensor, QuantTensor]:
    """Produce backend-owned NZ weight/scale layouts before collective inference."""
    _require_inference(l1_weights, l2_weights)
    return import_module("deep_gemm").transform_weights_for_mega_moe(l1_weights, l2_weights, activation=activation)


def transform_gemm_scales(scale: torch.Tensor, mn: int, k: int, recipe: tuple[int, int, int], *,
                          is_sfa: bool, num_groups: int | None = None, disable_ue8m0_cast: bool = False,
                          psum_layout: torch.Tensor | None = None) -> torch.Tensor:
    """Prepare physical scale storage without changing the caller's global GEMM policy."""
    _require_inference(scale)
    return import_module("deep_gemm").transform_sf_into_required_layout(
        scale, mn, k, recipe, num_groups, is_sfa, disable_ue8m0_cast, psum_layout)


def transform_k_grouped_gemm_scales(scale: torch.Tensor, mn: int, k: int, recipe: tuple[int, int, int], *,
                                    is_sfa: bool, psum_layout: torch.Tensor) -> torch.Tensor:
    """Prepare scale storage for padded K-grouped inference."""
    _require_inference(scale)
    return import_module("deep_gemm").transform_k_grouped_sf_into_required_layout(
        scale, mn, k, recipe, is_sfa, psum_layout)


class MegaMoEBuffer:
    """Own a native SymmBuffer, its borrowed views and its collective execution lifetime.

    All ranks must construct, execute and close in matching order. Callers must
    finish consuming outputs before close; close synchronizes the device before
    releasing native allocation references. Borrowed views must not outlive this owner.
    """

    def __init__(self, group: Any, **options: Any) -> None:
        """Allocate a symmetric input/routing buffer for the supplied process group."""
        self._extension = import_module("deep_gemm")
        self._buffer = self._extension.SymmBuffer(group, **options)

    @property
    def native(self) -> Any:
        """Access native input/routing views while their owner is alive."""
        if self._buffer is None:
            raise RuntimeError("MegaMoE buffer is closed")
        return self._buffer

    def __enter__(self) -> MegaMoEBuffer:
        """Retain ownership for the duration of a collective inference scope."""
        _ = self.native
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release resources on both normal and exceptional scope exit."""
        self.close()

    def forward(self, output: torch.Tensor, l1_weights: QuantTensor, l2_weights: QuantTensor, *,
                shared_l1_weights: QuantTensor | None = None, shared_l2_weights: QuantTensor | None = None,
                cumulative_local_expert_recv_stats: torch.Tensor | None = None,
                recipe: tuple[int, int, int] = (1, 1, 32), activation: str = "swiglu",
                activation_clamp: float | None = None, fast_math: bool = True,
                rms_weight: torch.Tensor | None = None, rms_eps: float | None = None,
                low_latency: bool | None = None) -> torch.Tensor:
        """Execute the native forward collective; every rank must use matching options."""
        native = self.native
        _require_inference(output, l1_weights, l2_weights, shared_l1_weights, shared_l2_weights, rms_weight,
                           native.x, native.x_sf, native.topk_weights)
        self._extension.fp8_fp4_mega_moe(
            output, l1_weights, l2_weights, native, shared_l1_weights=shared_l1_weights,
            shared_l2_weights=shared_l2_weights, cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            recipe=recipe, activation=activation, activation_clamp=activation_clamp, fast_math=fast_math,
            rms_weight=rms_weight, rms_eps=rms_eps, low_latency=low_latency)
        return output

    def close(self) -> None:
        """Wait for outstanding NPU work before dropping the symmetric allocation's owner."""
        if self._buffer is not None:
            torch.npu.synchronize(self._buffer.x.device)
            self._buffer.destroy()
            self._buffer = None
