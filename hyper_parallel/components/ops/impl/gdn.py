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

from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import PackedSequenceInfo
from hyper_parallel.components.functional.gated_delta_net import (
    chunk_gated_delta_rule,
    is_triton_gdn_available,
)

from .._dependency import require_package, require_symbol
from ..factory import _select_for_call
from ..registry import ImplementationUnavailable, OpRegistry
from ..spec import OpSpec


class GdnFn(Protocol):
    """Evaluate local gated delta-rule linear attention from a zero recurrent state.

    Gate values are effective log decays and beta values are effective update
    strengths, not logits. Optionally L2-normalize Q/K, then run the chunked
    delta recurrence with state resets at packed document boundaries. This
    interface neither consumes nor returns cross-rank recurrent state.
    """

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor,
        *, chunk_size: int = 64, normalize_qk: bool = True, info: PackedSequenceInfo | None = None,
    ) -> torch.Tensor:
        """Apply local GDN to effective gate and beta inputs, resetting between documents.

        Args:
            x: Queries [batch, sequence, key_heads, key_dim].
            key: Keys with the same layout as x.
            value: Values [batch, sequence, value_heads, value_dim].
            gate: Effective log decay [batch, sequence, value_heads].
            beta: Effective update strengths [batch, sequence, value_heads].
            chunk_size: Number of tokens per recurrence chunk.
            normalize_qk: Whether to L2-normalize queries and keys inside the operation.
            info: Optional local packed-document metadata; each document starts from zero state.

        Returns:
            Attention states [batch, sequence, value_heads, value_dim] without a final recurrent state.
        """


def gdn(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    normalize_qk: bool = True,
    info: PackedSequenceInfo | None = None,
) -> torch.Tensor:
    """Compute zero-state GDN with effective gate and beta inputs.

    Args:
        x: Queries [batch, sequence, key_heads, key_dim].
        key: Keys with the same layout as x.
        value: Values [batch, sequence, value_heads, value_dim].
        gate: Effective log decay [batch, sequence, value_heads].
        beta: Effective update strengths [batch, sequence, value_heads].
        chunk_size: Number of tokens per recurrence chunk.
        normalize_qk: Whether to L2-normalize queries and keys inside the operation.
        info: Optional local packed-document metadata; each document starts from zero state.

    Returns:
        Attention states [batch, sequence, value_heads, value_dim] without a final recurrent state.

    Notes:
        Evaluate local gated delta-rule linear attention from a zero recurrent state.

        Gate values are effective log decays and beta values are effective update
        strengths, not logits. Optionally L2-normalize Q/K, then run the chunked
        delta recurrence with state resets at packed document boundaries. This
        interface neither consumes nor returns cross-rank recurrent state.
    """
    operation = _select_for_call(GDN, device=x.device)
    return operation(x, key, value, gate, beta, chunk_size=chunk_size, normalize_qk=normalize_qk, info=info)


GDN = OpSpec[GdnFn](config_key='gdn', variant='zero_state')


def _implementation(backend: str) -> GdnFn:
    """Bind the GDN backend and translate optional packed document boundaries."""

    def run(
        x: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        *,
        chunk_size: int = 64,
        normalize_qk: bool = True,
        info: PackedSequenceInfo | None = None,
    ) -> torch.Tensor:
        """Run zero-state GDN with per-call document boundaries."""
        if backend != "eager":
            _validate_fused_inputs(x, key, value, gate, beta, chunk_size=chunk_size)
        offsets = None
        if info is not None:
            offsets = info.materialize(x.device, chunk_size).cu_seqlens if backend == "triton" else info.local_offsets
        output, _ = chunk_gated_delta_rule(
            x,
            key,
            value,
            gate,
            beta,
            chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=normalize_qk,
            backend=backend,
            cu_seqlens=offsets,
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


def _validate_fused_inputs(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> None:
    """Reject unsupported inputs in the selected fused implementation."""
    reason = None
    if (
        chunk_size != 64
        or x.shape[-1] != 128
        or value.shape[-1] != 128
        or (x.dtype != torch.bfloat16)
        or (key.dtype != x.dtype)
        or (value.dtype != x.dtype)
        or (beta.dtype != x.dtype)
        or (gate.dtype != torch.float32)
    ):
        reason = 'fused GDN requires chunk size 64, head dimension 128, BF16 Q/K/V/beta and FP32 gate'
    if reason is not None:
        raise ValueError(reason)
