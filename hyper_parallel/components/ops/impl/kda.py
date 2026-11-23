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
from torch.nn import functional as F

from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import PackedSequenceInfo

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


def _inverse_unit_lower_triangular(strict_lower: torch.Tensor) -> torch.Tensor:
    """Invert ``I + strict_lower`` using differentiable forward substitution."""
    chunk_size = strict_lower.shape[-1]
    rows = []
    for row_idx in range(chunk_size):
        base = -strict_lower[..., row_idx, :]
        if rows:
            previous = torch.stack(rows, dim=-2)
            row = torch.einsum(
                "...i,...ij->...j",
                base[..., :row_idx],
                previous,
            )
        else:
            row = torch.zeros_like(base)
        diagonal = F.one_hot(  # pylint: disable=not-callable
            torch.tensor(row_idx, device=strict_lower.device),
            num_classes=chunk_size,
        ).to(strict_lower.dtype)
        rows.append(row + diagonal)
    return torch.stack(rows, dim=-2)


def _prepare_kda_wy(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the cumulative gate, causal mask, and chunkwise WY factors."""
    gate = gate.cumsum(dim=-2)
    causal = torch.ones(
        chunk_size,
        chunk_size,
        dtype=torch.bool,
        device=query.device,
    ).tril()
    strict_causal = causal.logical_xor(
        torch.eye(chunk_size, dtype=torch.bool, device=query.device)
    )

    # Future-position differences can overflow before the final causal mask,
    # leaving NaNs in backward even when the forward result is finite.
    gate_pair = (gate.unsqueeze(-2) - gate.unsqueeze(-3)).masked_fill(~causal.unsqueeze(-1), -torch.inf)
    key_pair = torch.einsum(
        "...ik,...jk,...ijk->...ij",
        key,
        key,
        gate_pair.exp(),
    )
    strict_lower = (
        key_pair * beta.unsqueeze(-1)
    ).masked_fill(~strict_causal, 0)
    inverse = _inverse_unit_lower_triangular(strict_lower)
    transform = inverse * beta.unsqueeze(-2)
    w = transform @ (key * gate.exp())
    u = transform @ value

    return gate, causal, w, u


def _run_kda_chunks(
    query: torch.Tensor,
    key: torch.Tensor,
    gate: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    state: torch.Tensor,
    causal: torch.Tensor,
    scale: float,
    num_chunks: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Apply chunk WY factors to the recurrent state in sequence order."""
    output_chunks = []
    for chunk_idx in range(num_chunks):
        query_i = query[:, :, chunk_idx]
        key_i = key[:, :, chunk_idx]
        gate_i = gate[:, :, chunk_idx]
        w_i = w[:, :, chunk_idx]
        u_i = u[:, :, chunk_idx]

        relative_gate = (gate_i.unsqueeze(-2) - gate_i.unsqueeze(-3)).masked_fill(~causal.unsqueeze(-1), -torch.inf)
        qk = torch.einsum(
            "bhik,bhjk,bhijk->bhij",
            query_i,
            key_i,
            relative_gate.exp(),
        ).masked_fill(~causal, 0)
        value_new = u_i - w_i @ state
        output_inter = (query_i * gate_i.exp()) @ state
        output_chunks.append(output_inter * scale + qk @ value_new * scale)

        last_gate = gate_i[:, :, -1]
        key_decay = key_i * (last_gate.unsqueeze(-2) - gate_i).exp()
        state = state * last_gate.exp().unsqueeze(-1)
        state = state + key_decay.transpose(-1, -2) @ value_new

    return output_chunks, state


def _chunk_view(tensor: torch.Tensor, chunk_size: int) -> torch.Tensor:
    padding = (-tensor.shape[1]) % chunk_size
    tensor = F.pad(tensor, (0, 0) * (tensor.ndim - 2) + (0, padding))
    chunks = tensor.shape[1] // chunk_size
    return tensor.reshape(tensor.shape[0], chunks, chunk_size, *tensor.shape[2:]).permute(
        0, 3, 1, 2, *range(4, tensor.ndim + 1),
    )


def _validate_kda_shapes(
    x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor,
) -> None:
    """Validate token alignment, grouped heads and gate shapes for native KDA."""
    if x.ndim != 4 or x.shape != key.shape or value.ndim != 4 or value.shape[:2] != x.shape[:2]:
        raise ValueError("KDA requires matching [B,T,H,D] query/key and token-aligned values")
    key_heads, key_dim = x.shape[-2:]
    value_heads = value.shape[-2]
    if value_heads % key_heads or gate.shape != (*x.shape[:2], value_heads, key_dim):
        raise ValueError("KDA requires grouped value heads and a per-key-dimension gate")
    if beta.shape != (*x.shape[:2], value_heads):
        raise ValueError("KDA beta must have shape [B,T,value_heads]")


def _validate_kda_gate_parameters(
    a_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float | None, gate_shape: tuple[int, int],
) -> None:
    """Validate the trainable gate dimensions and optional negative lower bound."""
    value_heads, key_dim = gate_shape
    if a_log.numel() != value_heads or dt_bias.numel() != value_heads * key_dim:
        raise ValueError("KDA gate parameters must match value heads and key dimensions")
    if lower_bound is not None and lower_bound >= 0:
        raise ValueError("lower_bound must be negative")


def _native(
    x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor,
    *, a_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float | None,
    chunk_size: int = 64, scale: float | None = None, info: PackedSequenceInfo | None = None,
) -> torch.Tensor:
    """Evaluate the zero-state chunked WY reference without importing model modules."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    _validate_kda_shapes(x, key, value, gate, beta)
    batch, length, key_heads, key_dim = x.shape
    value_heads, value_dim = value.shape[-2:]
    _validate_kda_gate_parameters(a_log, dt_bias, lower_bound, (value_heads, key_dim))
    if info is not None:
        if batch != 1 or length != info.length:
            raise ValueError("Packed KDA requires [1,T,H,D] matching the local token interval")
        outputs = [
            _native(*(tensor[:, start:end] for tensor in (x, key, value, gate, beta)),
                    a_log=a_log, dt_bias=dt_bias, lower_bound=lower_bound, chunk_size=chunk_size, scale=scale)
            for start, end in zip(info.local_offsets, info.local_offsets[1:])
        ]
        return torch.cat(outputs, dim=1)
    output_dtype = value.dtype
    normalized = [
        tensor * (tensor.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt().to(tensor.dtype)
        for tensor in (x, key)
    ]
    x, key = (tensor.float().repeat_interleave(value_heads // key_heads, dim=2) for tensor in normalized)
    shifted = gate.float() + dt_bias.float().reshape(1, 1, value_heads, key_dim)
    coefficient = a_log.float().exp().reshape(1, 1, value_heads, 1)
    gate = (-coefficient * F.softplus(shifted) if lower_bound is None  # pylint: disable=not-callable
            else lower_bound * (coefficient * shifted).sigmoid())
    tensors = (x, key, value.float(), gate, beta.float().sigmoid())
    x, key, value, gate, beta = (_chunk_view(tensor, chunk_size) for tensor in tensors)
    gate, causal, w, u = _prepare_kda_wy(x, key, value, gate, beta, chunk_size)
    state = x.new_zeros((batch, value_heads, key_dim, value_dim))
    effective_scale = key_dim ** -0.5 if scale is None else scale
    outputs, _ = _run_kda_chunks(x, key, gate, w, u, state, causal, effective_scale, x.shape[2])
    return torch.stack(outputs, dim=2).permute(0, 2, 3, 1, 4).reshape(
        batch, -1, value_heads, value_dim,
    )[:, :length].to(output_dtype)


def _implementation(backend: str) -> KdaFn:
    # Existing fused adapters remain optional and are loaded only for a selected backend.
    """Bind the fused KDA backend and translate per-document sequence offsets."""
    chunk_kda = import_module("hyper_parallel.components.functional.kimi_delta_attention").chunk_kda

    def run(
        x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor,
        *, a_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float | None,
        chunk_size: int = 64, scale: float | None = None, info: PackedSequenceInfo | None = None,
    ) -> torch.Tensor:
        """Run the selected fused KDA kernel with per-call document boundaries."""
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
    adapter = require_package("hyper_parallel.components.functional.kimi_delta_attention_triton")
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
    registry.register_impl(KDA, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native)
    registry.register_impl(KDA, name="triton_ascend", devices=("npu",), load=_load_triton)
    registry.register_impl(KDA, name="ascendc", devices=("npu",), load=_load_ascendc)
