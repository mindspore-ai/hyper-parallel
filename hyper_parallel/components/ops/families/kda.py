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
"""Local zero-state Kimi Delta Attention with raw gate and beta logits."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol

import torch

from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo

from .._dependency import require_package, require_symbol
from ..registry import ImplementationUnavailable, OpRegistry
from ..spec import OpSpec


class KdaFn(Protocol):
    """Compute KDA with normalized Q/K and a lower-bounded gate."""

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor,
        *, a_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float | None,
        chunk_size: int = 64, scale: float | None = None, info: PackedSequenceInfo | None = None,
    ) -> torch.Tensor:
        """Interpret gate and beta as logits and reset recurrence between packed documents."""


KDA = OpSpec[KdaFn](config_key="kda", variant="zero_state")


def _implementation(backend: str) -> KdaFn:
    # Load chunk_kda lazily because its module imports the public kda operator.
    chunk_kda = import_module("hyper_parallel.components.modules.kimi_delta_attention").chunk_kda

    def run(
        x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor,
        *, a_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float | None,
        chunk_size: int = 64, scale: float | None = None, info: PackedSequenceInfo | None = None,
    ) -> torch.Tensor:
        offsets = None
        if info is not None:
            offsets = (info.materialize(x.device, chunk_size).cu_seqlens if backend == "triton" else info.local_offsets)
        output, _ = chunk_kda(
            x, key, value, gate, beta, a_log=a_log, dt_bias=dt_bias, scale=scale, lower_bound=lower_bound,
            chunk_size=chunk_size, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, backend=backend, cu_seqlens=offsets,
        )
        return output
    return run


def _load_triton() -> KdaFn:
    require_package("triton")
    adapter = require_package("hyper_parallel.components.functional.kimi_delta_attention_fla_adapter")
    if not adapter.is_fla_triton_kda_available():
        raise ImplementationUnavailable("requires the FLA Triton-Ascend KDA kernel")
    return _implementation("triton")


def _load_ascendc() -> KdaFn:
    kernels = require_package("fla_npu.ops.ascendc")
    for name in (
        "chunk_kda_fwd_prepare", "chunk_fwd_h", "chunk_kda_fwd_finalize", "chunk_kda_bwd_prepare",
        "chunk_gated_delta_rule_bwd_dhu", "chunk_kda_bwd_finalize",
    ):
        require_symbol(kernels, name)
    return _implementation("ascendc")


def register_kda(registry: OpRegistry) -> None:
    """Register the local KDA contract and its optional NPU kernels."""
    registry.register_family("kda", specs=(KDA,))
    registry.register_impl(KDA, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _implementation("eager"))
    registry.register_impl(KDA, name="triton_ascend", devices=("npu",), load=_load_triton)
    registry.register_impl(KDA, name="ascendc", devices=("npu",), load=_load_ascendc)
