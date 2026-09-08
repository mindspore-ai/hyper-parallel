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
"""Pure PyTorch Kimi Delta Attention operators and training layer.

The implementations in this module intentionally favor readable equations over
performance. They are the numerical oracle shared by the standalone layer and
context-parallel wrappers.
"""
# pylint: disable=invalid-name
# This module mirrors the existing Torch model building blocks in this package.
# pylint: disable=forbidden-backend-import
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F


_KDA_BACKENDS = frozenset({"eager", "triton"})
_TRITON_KDA_HEAD_DIM = 128
_TRITON_KDA_CHUNK_SIZE = 64


def _is_triton_kda_input_supported(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    gate: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    a_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    *,
    chunk_size: int,
    lower_bound: Optional[float],
) -> bool:
    """Check the fixed Kimi K3 contract validated by the fused backend."""
    operands = (key, value, gate, beta, a_log, dt_bias)
    if any(tensor is None for tensor in operands):
        return False
    if lower_bound is None:
        return False

    basic_contract = (
        query.device.type == "npu",
        all(tensor.dtype == torch.bfloat16 for tensor in (query, key, value, gate, beta)),
        all(tensor.dtype == torch.float32 for tensor in (a_log, dt_bias)),
        all(tensor.ndim == 4 for tensor in (query, key, value, gate)),
        beta.ndim == 3,
        query.shape == key.shape,
    )
    if not all(basic_contract):
        return False

    batch_size, sequence_length, num_query_heads, key_dim = query.shape
    num_value_heads, value_dim = value.shape[2:]
    shape_contract = (
        value.shape[:2] == (batch_size, sequence_length),
        gate.shape == (batch_size, sequence_length, num_value_heads, key_dim),
        beta.shape == (batch_size, sequence_length, num_value_heads),
        num_value_heads % num_query_heads == 0,
        a_log.numel() == num_value_heads,
        dt_bias.numel() == num_value_heads * key_dim,
        key_dim == value_dim == _TRITON_KDA_HEAD_DIM,
        chunk_size == _TRITON_KDA_CHUNK_SIZE,
        sequence_length % chunk_size == 0,
        -5.0 <= lower_bound < 0,
    )
    if not all(shape_contract):
        return False
    return all(tensor.device == query.device for tensor in operands)


def is_triton_kda_available(
    query: Optional[torch.Tensor] = None,
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    *,
    chunk_size: int = _TRITON_KDA_CHUNK_SIZE,
    lower_bound: Optional[float] = -5.0,
) -> bool:
    """Return whether the validated Triton-Ascend KDA backend is available."""
    if chunk_size != _TRITON_KDA_CHUNK_SIZE:
        return False
    if query is not None and not _is_triton_kda_input_supported(
        query,
        key,
        value,
        gate,
        beta,
        a_log,
        dt_bias,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    ):
        return False
    from hyper_parallel.platform.torch.custom_ops.kda.fla_adapter import (  # pylint: disable=import-outside-toplevel
        is_fla_triton_kda_available,
    )
    return is_fla_triton_kda_available()


def _validate_kda_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    """Validate dense token-first KDA inputs and return their dimensions."""
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4 or gate.dim() != 4:
        raise ValueError("query, key, value, and gate must be rank-4 tensors.")
    if beta.dim() != 3:
        raise ValueError("beta must be a rank-3 tensor.")
    if query.shape != key.shape:
        raise ValueError("query and key must have identical shapes.")

    batch_size, sequence_length, num_k_heads, k_head_dim = query.shape
    value_batch, value_sequence, num_v_heads, v_head_dim = value.shape
    if (value_batch, value_sequence) != (batch_size, sequence_length):
        raise ValueError("value batch and sequence dimensions must match query.")
    if gate.shape != (batch_size, sequence_length, num_v_heads, k_head_dim):
        raise ValueError(
            "gate must have shape [B, S, num_v_heads, k_head_dim], got "
            f"{tuple(gate.shape)}."
        )
    if beta.shape != (batch_size, sequence_length, num_v_heads):
        raise ValueError(
            "beta must have shape [B, S, num_v_heads], got "
            f"{tuple(beta.shape)}."
        )
    if num_v_heads % num_k_heads != 0:
        raise ValueError("num_v_heads must be divisible by num_k_heads.")
    return (
        batch_size,
        sequence_length,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )


def _reshape_gate_bias(
    parameter: torch.Tensor,
    num_v_heads: int,
    k_head_dim: int,
    name: str,
) -> torch.Tensor:
    """Return a gate parameter as ``[num_v_heads, k_head_dim]``."""
    if parameter.numel() != num_v_heads * k_head_dim:
        raise ValueError(
            f"{name} must contain {num_v_heads * k_head_dim} elements, "
            f"got {parameter.numel()}."
        )
    return parameter.reshape(num_v_heads, k_head_dim)


def torch_kda_gate(
    gate_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    """Activate KDA log-decay logits.

    ``lower_bound`` selects the Kimi K3 parameterization
    ``g_min * sigmoid(exp(a_log) * (logits + bias))``. With no lower bound,
    this function implements the earlier Kimi Linear negative-Softplus gate.
    """
    if gate_logits.dim() != 4:
        raise ValueError("gate_logits must have shape [B, S, H, K].")
    num_v_heads, k_head_dim = gate_logits.shape[-2:]
    if a_log.numel() != num_v_heads:
        raise ValueError(
            f"a_log must contain {num_v_heads} elements, got {a_log.numel()}."
        )
    bias = _reshape_gate_bias(dt_bias, num_v_heads, k_head_dim, "dt_bias")
    scale = a_log.float().exp().reshape(1, 1, num_v_heads, 1)
    preactivation = gate_logits.float() + bias.float().reshape(
        1, 1, num_v_heads, k_head_dim
    )
    if lower_bound is None:
        return -scale * F.softplus(preactivation)  # pylint: disable=not-callable
    if lower_bound >= 0:
        raise ValueError(f"lower_bound must be negative, got {lower_bound}.")
    return float(lower_bound) * torch.sigmoid(scale * preactivation)


def _prepare_kda_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    scale: Optional[float],
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    lower_bound: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply the common KDA input transformations in FP32."""
    _, _, num_k_heads, num_v_heads, k_head_dim, _ = _validate_kda_inputs(
        query, key, value, gate, beta
    )
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt(
            (query.float() * query.float()).sum(dim=-1, keepdim=True) + 1e-6
        ).to(query.dtype)
        key = key * torch.rsqrt(
            (key.float() * key.float()).sum(dim=-1, keepdim=True) + 1e-6
        ).to(key.dtype)
    if use_gate_in_kernel:
        if a_log is None or dt_bias is None:
            raise ValueError("a_log and dt_bias are required when gate activation is enabled.")
        gate = torch_kda_gate(
            gate,
            a_log,
            dt_bias,
            lower_bound=lower_bound,
        )
    if use_beta_sigmoid_in_kernel:
        beta = beta.float().sigmoid()

    value_groups = num_v_heads // num_k_heads
    query = query.float().repeat_interleave(value_groups, dim=2)
    key = key.float().repeat_interleave(value_groups, dim=2)
    value = value.float()
    gate = gate.float()
    beta = beta.float()
    effective_scale = k_head_dim ** -0.5 if scale is None else scale
    return query, key, value, gate, beta, float(effective_scale)


def _initial_kda_state(
    value: torch.Tensor,
    k_head_dim: int,
    initial_state: Optional[torch.Tensor],
) -> torch.Tensor:
    """Build or validate a KDA state in FP32."""
    expected_shape = (value.shape[0], value.shape[2], k_head_dim, value.shape[3])
    if initial_state is None:
        return value.new_zeros(expected_shape, dtype=torch.float32)
    if tuple(initial_state.shape) != expected_shape:
        raise ValueError(
            f"initial_state must have shape {expected_shape}, got "
            f"{tuple(initial_state.shape)}."
        )
    return initial_state.float()


def torch_recurrent_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Evaluate the exact token-recurrent KDA equations with PyTorch ops."""
    output_dtype = value.dtype
    query, key, value, gate, beta, scale = _prepare_kda_inputs(
        query,
        key,
        value,
        gate,
        beta,
        a_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        lower_bound=lower_bound,
    )
    state = _initial_kda_state(value, key.shape[-1], initial_state)
    outputs = []
    for token_idx in range(query.shape[1]):
        query_i = query[:, token_idx]
        key_i = key[:, token_idx]
        value_i = value[:, token_idx]
        gate_i = gate[:, token_idx]
        beta_i = beta[:, token_idx]

        state = state * gate_i.exp().unsqueeze(-1)
        state_value = torch.einsum("bhk,bhkv->bhv", key_i, state)
        residual = value_i - state_value
        state = state + torch.einsum(
            "bhk,bhv->bhkv",
            key_i * beta_i.unsqueeze(-1),
            residual,
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", query_i * scale, state))

    output = torch.stack(outputs, dim=1).to(output_dtype)
    return output, state if output_final_state else None


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


def torch_chunk_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Evaluate chunkwise KDA using the UT/WY representation.

    This implementation is algebraically independent from
    :func:`torch_recurrent_kda`; comparing the two catches errors in the
    chunk transform that a second token loop would hide.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    output_dtype = value.dtype
    sequence_length = query.shape[1]
    query, key, value, gate, beta, scale = _prepare_kda_inputs(
        query,
        key,
        value,
        gate,
        beta,
        a_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        lower_bound=lower_bound,
    )
    batch_size, _, num_heads, k_head_dim = query.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, 0, 0, pad_size))
    gate = F.pad(gate, (0, 0, 0, 0, 0, pad_size))
    beta = F.pad(beta, (0, 0, 0, pad_size))
    num_chunks = query.shape[1] // chunk_size

    def chunk_view(tensor: torch.Tensor) -> torch.Tensor:
        """Rearrange a token-first tensor to head-first chunks."""
        shape = tensor.shape
        return tensor.reshape(
            shape[0], num_chunks, chunk_size, shape[2], *shape[3:]
        ).permute(0, 3, 1, 2, *range(4, tensor.dim() + 1))

    query, key, value, gate, beta = [
        chunk_view(tensor) for tensor in (query, key, value, gate, beta)
    ]
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

    gate_pair = gate.unsqueeze(-2) - gate.unsqueeze(-3)
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

    state = _initial_kda_state(
        value.permute(0, 2, 3, 1, 4).reshape(
            batch_size, num_chunks * chunk_size, num_heads, v_head_dim
        ),
        k_head_dim,
        initial_state,
    )
    output_chunks = []
    for chunk_idx in range(num_chunks):
        query_i = query[:, :, chunk_idx]
        key_i = key[:, :, chunk_idx]
        gate_i = gate[:, :, chunk_idx]
        w_i = w[:, :, chunk_idx]
        u_i = u[:, :, chunk_idx]

        relative_gate = gate_i.unsqueeze(-2) - gate_i.unsqueeze(-3)
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

    output = torch.stack(output_chunks, dim=2)
    output = output.permute(0, 2, 3, 1, 4).reshape(
        batch_size,
        num_chunks * chunk_size,
        num_heads,
        v_head_dim,
    )
    output = output[:, :sequence_length].to(output_dtype)
    return output, state if output_final_state else None


def chunk_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
    chunk_size: int = 64,
    backend: str = "eager",
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Dispatch KDA to an explicitly selected eager or Triton backend."""
    backend = backend.lower()
    if backend not in _KDA_BACKENDS:
        raise ValueError(
            f"unsupported KDA backend {backend!r}; "
            f"expected one of {sorted(_KDA_BACKENDS)}."
        )
    if backend == "eager":
        return torch_chunk_kda(
            query,
            key,
            value,
            gate,
            beta,
            a_log=a_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            lower_bound=lower_bound,
            chunk_size=chunk_size,
        )
    if initial_state is not None or output_final_state:
        raise NotImplementedError(
            "KDA backend='triton' currently supports zero initial state without "
            "returning a final state; stateful execution is owned by the CP backend."
        )
    if not (
        use_qk_l2norm_in_kernel
        and use_gate_in_kernel
        and use_beta_sigmoid_in_kernel
    ):
        raise NotImplementedError(
            "KDA backend='triton' requires fused Q/K normalization, gate activation, "
            "and beta sigmoid."
        )
    triton_a_log = a_log.float() if a_log is not None else None
    triton_dt_bias = dt_bias.float() if dt_bias is not None else None
    if not _is_triton_kda_input_supported(
        query,
        key,
        value,
        gate,
        beta,
        triton_a_log,
        triton_dt_bias,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    ):
        raise RuntimeError(
            "KDA backend='triton' received an unsupported input contract; it requires "
            "q/k/v/gate/beta=bf16, a_log/dt_bias=fp32, "
            "head_k_dim=head_v_dim=128, chunk_size=64, and a supported safe gate."
        )

    from hyper_parallel.platform.torch.custom_ops.kda.chunk_kda import (  # pylint: disable=import-outside-toplevel
        fused_chunk_kda,
    )

    output = fused_chunk_kda(
        query,
        key,
        value,
        gate,
        beta,
        a_log=triton_a_log,
        dt_bias=triton_dt_bias,
        scale=scale,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        safe_gate=True,
    )
    return output, None


def _chunk_kda_summary_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad and convert token-first KDA tensors into head-first chunks."""
    sequence_length = key.shape[1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    key = F.pad(key, (0, 0, 0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, 0, 0, pad_size))
    gate = F.pad(gate, (0, 0, 0, 0, 0, pad_size))
    beta = F.pad(beta, (0, 0, 0, pad_size))
    num_chunks = key.shape[1] // chunk_size

    def chunk_view(tensor: torch.Tensor) -> torch.Tensor:
        """Rearrange one token-first tensor to head-first chunks."""
        shape = tensor.shape
        return tensor.reshape(
            shape[0], num_chunks, chunk_size, shape[2], *shape[3:]
        ).permute(0, 3, 1, 2, *range(4, tensor.dim() + 1))

    key, value, gate, beta = (
        chunk_view(tensor) for tensor in (key, value, gate, beta)
    )
    return key, value, gate, beta, num_chunks


def torch_kda_state_summary(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summarize a local KDA segment as ``H_out = M @ H_in + S``.

    The implementation follows the same chunkwise WY equations as
    :func:`torch_chunk_kda`, but computes only the state-independent affine
    transition. ``S`` has shape ``[B, HV, K, V]`` and ``M`` has shape
    ``[B, HV, K, K]``. Query is accepted to share KDA input preprocessing and
    grouped-value validation; the state transition itself is independent of
    query and output scale.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    _, key, value, gate, beta, _ = _prepare_kda_inputs(
        query,
        key,
        value,
        gate,
        beta,
        a_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        lower_bound=lower_bound,
    )
    key, value, gate, beta, num_chunks = _chunk_kda_summary_inputs(
        key,
        value,
        gate,
        beta,
        chunk_size,
    )
    gate = gate.cumsum(dim=-2)
    strict_causal = torch.ones(
        chunk_size,
        chunk_size,
        dtype=torch.bool,
        device=key.device,
    ).tril(diagonal=-1)
    gate_pair = gate.unsqueeze(-2) - gate.unsqueeze(-3)
    key_pair = torch.einsum(
        "...ik,...jk,...ijk->...ij",
        key,
        key,
        gate_pair.exp(),
    )
    strict_lower = (key_pair * beta.unsqueeze(-1)).masked_fill(
        ~strict_causal,
        0,
    )
    inverse = _inverse_unit_lower_triangular(strict_lower)
    transform = inverse * beta.unsqueeze(-2)
    w = transform @ (key * gate.exp())
    u = transform @ value

    batch_size, num_heads, _, _, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    eye = torch.eye(
        k_head_dim,
        dtype=torch.float32,
        device=key.device,
    ).reshape(1, 1, k_head_dim, k_head_dim)
    state_ext = torch.zeros(
        batch_size,
        num_heads,
        k_head_dim,
        v_head_dim,
        dtype=torch.float32,
        device=key.device,
    )
    transition = eye.expand(batch_size, num_heads, -1, -1).clone()

    for chunk_idx in range(num_chunks):
        key_i = key[:, :, chunk_idx]
        gate_i = gate[:, :, chunk_idx]
        w_i = w[:, :, chunk_idx]
        u_i = u[:, :, chunk_idx]
        last_gate = gate_i[:, :, -1]
        key_decay = key_i * (last_gate.unsqueeze(-2) - gate_i).exp()
        transition_i = torch.diag_embed(last_gate.exp())
        transition_i = transition_i - key_decay.transpose(-1, -2) @ w_i
        state_ext_i = key_decay.transpose(-1, -2) @ u_i
        state_ext = transition_i @ state_ext + state_ext_i
        transition = transition_i @ transition

    return state_ext.float(), transition.float()


def torch_apply_kda_state_summary(
    state_ext: torch.Tensor,
    transition: torch.Tensor,
    initial_state: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply a KDA affine state summary to an incoming state."""
    if initial_state is None:
        return state_ext.float()
    return (transition @ initial_state.to(transition) + state_ext).float()


def torch_compose_kda_state_summaries(
    first_state_ext: torch.Tensor,
    first_transition: torch.Tensor,
    second_state_ext: torch.Tensor,
    second_transition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose two chronological KDA summaries, first then second."""
    state_ext = second_transition @ first_state_ext + second_state_ext
    transition = second_transition @ first_transition
    return state_ext, transition


class KimiRMSNormGated(nn.Module):
    """Per-head RMSNorm followed by the sigmoid output gate used by KDA."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize the per-head normalization weight and epsilon."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize in FP32 and apply a sigmoid gate in the input dtype."""
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        normalized = normalized * self.weight.float()
        return (normalized * gate.float().sigmoid()).to(input_dtype)


def _make_short_convolution(channels: int, kernel_size: int) -> nn.Conv1d:
    """Create the depthwise causal convolution used by the training layer."""
    convolution = nn.Conv1d(
        channels,
        channels,
        kernel_size=kernel_size,
        groups=channels,
        padding=kernel_size - 1,
        bias=False,
    )
    convolution.activation = "silu"
    return convolution


class KimiDeltaAttention(nn.Module):
    """Training-only Kimi Delta Attention layer implemented with eager Torch.

    The layer owns the projection, ShortConv, gate, normalization, and output
    parameters used by Kimi K3. Accelerator selection is intentionally outside
    this model module; context-parallel policies may replace ``forward`` while
    preserving these parameters and state-dict names.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 32,
        num_v_heads: Optional[int] = None,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        conv_kernel_size: int = 4,
        chunk_size: int = 64,
        gate_lower_bound: float = -5.0,
        use_full_rank_gate: bool = True,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
    ) -> None:
        """Initialize a configurable, training-only KDA attention layer."""
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be positive.")
        if num_v_heads is None:
            num_v_heads = num_heads
        if num_v_heads <= 0 or num_v_heads % num_heads:
            raise ValueError("num_v_heads must be positive and divisible by num_heads.")
        if head_k_dim <= 0 or head_v_dim <= 0:
            raise ValueError("KDA head dimensions must be positive.")
        if conv_kernel_size <= 0 or chunk_size <= 0:
            raise ValueError("conv_kernel_size and chunk_size must be positive.")
        if gate_lower_bound >= 0:
            raise ValueError("gate_lower_bound must be negative.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_k_heads = num_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.head_dim = head_v_dim
        self.key_dim = num_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.conv_size = conv_kernel_size
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.gate_lower_bound = gate_lower_bound
        self.lower_bound = gate_lower_bound
        self.safe_gate = True
        self.allow_neg_eigval = False
        self.use_short_conv = True
        self.use_full_rank_gate = use_full_rank_gate

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_conv1d = _make_short_convolution(self.key_dim, conv_kernel_size)
        self.k_conv1d = _make_short_convolution(self.key_dim, conv_kernel_size)
        self.v_conv1d = _make_short_convolution(self.value_dim, conv_kernel_size)

        self.A_log = nn.Parameter(torch.empty(num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.zeros(num_v_heads * head_k_dim, dtype=torch.float32)
        )
        with torch.no_grad():
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1, 16).log())

        self.f_a_proj = nn.Linear(hidden_size, head_k_dim, bias=False)
        self.f_b_proj = nn.Linear(head_k_dim, num_v_heads * head_k_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_v_heads, bias=False)
        if use_full_rank_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        else:
            self.g_a_proj = nn.Linear(hidden_size, head_v_dim, bias=False)
            self.g_b_proj = nn.Linear(head_v_dim, self.value_dim, bias=False)
        self.o_norm = KimiRMSNormGated(head_v_dim, eps=rms_norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    @staticmethod
    def _run_short_convolution(
        projected: torch.Tensor,
        convolution: nn.Conv1d,
    ) -> torch.Tensor:
        """Apply one causal depthwise convolution and trim its right padding."""
        sequence_length = projected.shape[1]
        output = convolution(projected.transpose(1, 2))[:, :, :sequence_length]
        return F.silu(output.transpose(1, 2))

    def _output_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project the layer input to the per-head output gate."""
        if self.use_full_rank_gate:
            return self.g_proj(hidden_states)
        return self.g_b_proj(self.g_a_proj(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run fixed-length, training-time KDA without recurrent cache."""
        if use_cache or kwargs.get("cache_params") is not None:
            raise NotImplementedError("KimiDeltaAttention does not support inference cache.")
        if output_attentions:
            raise NotImplementedError("KDA does not materialize attention weights.")
        if kwargs.get("cu_seqlens") is not None:
            raise NotImplementedError("Packed variable-length KDA is not implemented.")
        if attention_mask is not None and not bool(attention_mask.bool().all().item()):
            raise NotImplementedError("Padded KDA batches are not implemented.")

        batch_size, sequence_length, _ = hidden_states.shape
        query = self._run_short_convolution(
            self.q_proj(hidden_states), self.q_conv1d
        ).reshape(batch_size, sequence_length, self.num_heads, self.head_k_dim)
        key = self._run_short_convolution(
            self.k_proj(hidden_states), self.k_conv1d
        ).reshape(batch_size, sequence_length, self.num_heads, self.head_k_dim)
        value = self._run_short_convolution(
            self.v_proj(hidden_states), self.v_conv1d
        ).reshape(batch_size, sequence_length, self.num_v_heads, self.head_v_dim)
        gate = self.f_b_proj(self.f_a_proj(hidden_states)).reshape(
            batch_size, sequence_length, self.num_v_heads, self.head_k_dim
        )
        beta = self.b_proj(hidden_states).reshape(
            batch_size, sequence_length, self.num_v_heads
        )
        output, _ = torch_chunk_kda(
            query,
            key,
            value,
            gate,
            beta,
            a_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=self.gate_lower_bound,
            chunk_size=self.chunk_size,
        )
        output_gate = self._output_gate(hidden_states).reshape(
            batch_size, sequence_length, self.num_v_heads, self.head_v_dim
        )
        output = self.o_norm(output, output_gate)
        return self.o_proj(output.reshape(batch_size, sequence_length, self.value_dim))


__all__ = [
    "KimiDeltaAttention",
    "KimiRMSNormGated",
    "chunk_kda",
    "is_triton_kda_available",
    "torch_apply_kda_state_summary",
    "torch_chunk_kda",
    "torch_compose_kda_state_summaries",
    "torch_kda_gate",
    "torch_kda_state_summary",
    "torch_recurrent_kda",
]
