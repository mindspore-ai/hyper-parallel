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
"""Local zero-state Gated DeltaNet over dense or packed tokens."""

from __future__ import annotations

from typing import Protocol

import torch

from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo
from hyper_parallel.components.modules.gated_delta_net import (
    chunk_gated_delta_rule,
    is_triton_gdn_available,
)

from .._dependency import require_package, require_symbol
from ..registry import ImplementationUnavailable, OpRegistry
from ..spec import OpSpec


class GdnFn(Protocol):
    """Compute GDN without reading or returning recurrent state."""

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor,
        *, chunk_size: int = 64, normalize_qk: bool = True, info: PackedSequenceInfo | None = None,
    ) -> torch.Tensor:
        """Apply local GDN to effective gate and beta inputs, resetting between documents."""


GDN = OpSpec[GdnFn](config_key="gdn", variant="zero_state")


def _implementation(backend: str) -> GdnFn:
    def run(
        x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor,
        *, chunk_size: int = 64, normalize_qk: bool = True, info: PackedSequenceInfo | None = None,
    ) -> torch.Tensor:
        offsets = None
        if info is not None:
            offsets = (info.materialize(x.device, chunk_size).cu_seqlens if backend == "triton" else info.local_offsets)
        output, _ = chunk_gated_delta_rule(
            x, key, value, gate, beta, chunk_size=chunk_size, use_qk_l2norm_in_kernel=normalize_qk,
            backend=backend, cu_seqlens=offsets,
        )
        return output
    return run


def _load_triton() -> GdnFn:
    if not is_triton_gdn_available():
        raise ImplementationUnavailable("requires triton-ascend >= 3.2.1")
    return _implementation("triton")


def _load_ascendc() -> GdnFn:
    kernels = require_package("fla_npu.ops.ascendc")
    require_symbol(kernels, "chunk_gated_delta_rule_fwd")
    require_symbol(kernels, "chunk_gated_delta_rule_bwd")
    return _implementation("ascendc")


def register_gdn(registry: OpRegistry) -> None:
    """Register the shared GDN contract and its optional NPU kernels."""
    registry.register_family("gdn", specs=(GDN,))
    registry.register_impl(GDN, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _implementation("eager"))
    registry.register_impl(GDN, name="triton_ascend", devices=("npu",), load=_load_triton)
    registry.register_impl(GDN, name="ascendc", devices=("npu",), load=_load_ascendc)
