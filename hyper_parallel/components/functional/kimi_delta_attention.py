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
"""Local KDA math, state application and explicit backend selection."""
from __future__ import annotations
from collections.abc import Sequence
from typing import Optional

import torch
from torch.nn import functional as F

from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import packed_eager
from .kimi_delta_attention_ascendc import ascendc_chunk_kda
from .kimi_delta_attention_triton import fused_chunk_kda, is_fla_triton_kda_available


def _validate_kda_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    """Validate dense token-first KDA inputs and return their dimensions."""
    if (query.dim(), key.dim(), value.dim(), gate.dim()) != (4, 4, 4, 4):
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

    Args:
        gate_logits: Untransformed per-key decay logits.
        a_log: Log decay-rate parameter.
        dt_bias: Per-key time-step bias.
        lower_bound: Negative lower bound of the safe decay gate.
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
    """Evaluate the exact token-recurrent KDA equations with eager ops.

    Args:
        query: Query tensor in batch, token, head, key-dimension order.
        key: Key tensor with the same token and key-head axes as query.
        value: Value tensor in batch, token, head, value-dimension order.
        gate: Decay gate or output gate required by this stage.
        beta: Delta-rule update strengths, or logits when sigmoid fusion is enabled.
        a_log: Log decay-rate parameter.
        dt_bias: Per-key time-step bias.
        scale: Query scaling factor; None selects the head-dimension default.
        initial_state: Optional state entering the first local chunk.
        output_final_state: Whether to return the state after the final chunk.
        use_qk_l2norm_in_kernel: Whether this operator normalizes query and key.
        use_gate_in_kernel: Whether this operator transforms raw decay logits.
        use_beta_sigmoid_in_kernel: Whether this operator transforms beta logits with sigmoid.
        lower_bound: Negative lower bound of the safe decay gate.
    """
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

    return output_chunks, state


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
    cu_seqlens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Evaluate chunkwise KDA using the UT/WY representation.

    This implementation is algebraically independent from
    :func:`torch_recurrent_kda`; comparing the two catches errors in the
    chunk transform that a second token loop would hide.

    Args:
        query: Query tensor in batch, token, head, key-dimension order.
        key: Key tensor with the same token and key-head axes as query.
        value: Value tensor in batch, token, head, value-dimension order.
        gate: Decay gate or output gate required by this stage.
        beta: Delta-rule update strengths, or logits when sigmoid fusion is enabled.
        a_log: Log decay-rate parameter.
        dt_bias: Per-key time-step bias.
        scale: Query scaling factor; None selects the head-dimension default.
        initial_state: Optional state entering the first local chunk.
        output_final_state: Whether to return the state after the final chunk.
        use_qk_l2norm_in_kernel: Whether this operator normalizes query and key.
        use_gate_in_kernel: Whether this operator transforms raw decay logits.
        use_beta_sigmoid_in_kernel: Whether this operator transforms beta logits with sigmoid.
        lower_bound: Negative lower bound of the safe decay gate.
        chunk_size: Number of tokens in a recurrence chunk.
        cu_seqlens: Cumulative document boundaries for packed tokens.
    """
    if cu_seqlens is not None:
        return packed_eager(
            torch_chunk_kda, (query, key, value, gate, beta), cu_seqlens,
            initial_state=initial_state, output_final_state=output_final_state,
            a_log=a_log, dt_bias=dt_bias, scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel, lower_bound=lower_bound, chunk_size=chunk_size,
        )
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
        """Rearrange a token-first tensor to head-first chunks.

        Args:
            tensor: Tensor to reshape or redistribute without changing token order.
        """
        shape = tensor.shape
        return tensor.reshape(
            shape[0], num_chunks, chunk_size, shape[2], *shape[3:]
        ).permute(0, 3, 1, 2, *range(4, tensor.dim() + 1))

    query, key, value, gate, beta = [
        chunk_view(tensor) for tensor in (query, key, value, gate, beta)
    ]
    gate, causal, w, u = _prepare_kda_wy(query, key, value, gate, beta, chunk_size)

    state = _initial_kda_state(
        value.permute(0, 2, 3, 1, 4).reshape(
            batch_size, num_chunks * chunk_size, num_heads, v_head_dim
        ),
        k_head_dim,
        initial_state,
    )
    output_chunks, state = _run_kda_chunks(query, key, gate, w, u, state, causal, scale, num_chunks)

    output = torch.stack(output_chunks, dim=2)
    output = output.permute(0, 2, 3, 1, 4).reshape(
        batch_size,
        num_chunks * chunk_size,
        num_heads,
        v_head_dim,
    )
    output = output[:, :sequence_length].to(output_dtype)
    return output, state if output_final_state else None


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
        """Rearrange one token-first tensor to head-first chunks.

        Args:
            tensor: Tensor to reshape or redistribute without changing token order.
        """
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

    Args:
        query: Query tensor in batch, token, head, key-dimension order.
        key: Key tensor with the same token and key-head axes as query.
        value: Value tensor in batch, token, head, value-dimension order.
        gate: Decay gate or output gate required by this stage.
        beta: Delta-rule update strengths, or logits when sigmoid fusion is enabled.
        a_log: Log decay-rate parameter.
        dt_bias: Per-key time-step bias.
        scale: Query scaling factor; None selects the head-dimension default.
        use_qk_l2norm_in_kernel: Whether this operator normalizes query and key.
        use_gate_in_kernel: Whether this operator transforms raw decay logits.
        use_beta_sigmoid_in_kernel: Whether this operator transforms beta logits with sigmoid.
        lower_bound: Negative lower bound of the safe decay gate.
        chunk_size: Number of tokens in a recurrence chunk.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    _, key, value, gate, beta, _ = _prepare_kda_inputs(
        query, key, value, gate, beta,
        a_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        lower_bound=lower_bound,
    )
    key, value, gate, beta, num_chunks = _chunk_kda_summary_inputs(
        key, value, gate, beta, chunk_size,
    )
    gate = gate.cumsum(dim=-2)
    strict_causal = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=key.device,
    ).tril(diagonal=-1)
    gate_pair = gate.unsqueeze(-2) - gate.unsqueeze(-3)
    key_pair = torch.einsum(
        "...ik,...jk,...ijk->...ij", key, key, gate_pair.exp(),
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
        k_head_dim, dtype=torch.float32, device=key.device,
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
    """Apply a KDA affine state summary to an incoming state.

    Args:
        state_ext: State produced from a zero initial state.
        transition: Linear state transition matrix.
        initial_state: Optional state entering the first local chunk.
    """
    if initial_state is None:
        return state_ext.float()
    return (transition @ initial_state.to(transition) + state_ext).float()


def torch_compose_kda_state_summaries(
    first_state_ext: torch.Tensor,
    first_transition: torch.Tensor,
    second_state_ext: torch.Tensor,
    second_transition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose two chronological KDA summaries, first then second.

    Args:
        first_state_ext: Zero-input state of the earlier sequence interval.
        first_transition: Transition matrix of the earlier sequence interval.
        second_state_ext: Zero-input state of the later sequence interval.
        second_transition: Transition matrix of the later sequence interval.
    """
    state_ext = second_transition @ first_state_ext + second_state_ext
    transition = second_transition @ first_transition
    return state_ext, transition


def apply_kda_state_summary(
    state_ext: torch.Tensor,
    transition: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply a prepared KDA state summary in FP32.

    Args:
        state_ext: Local zero-entry state S.
        transition: Local transfer M; may be omitted for a zero entry.
        initial_state: Incoming FP32 state, or None for a zero entry.

    Returns:
        The outgoing FP32 state.
    """
    if initial_state is None:
        return state_ext
    # Model autocast must not silently quantize a communicated FP32 state.
    with torch.autocast(device_type=state_ext.device.type, enabled=False):
        return torch.matmul(transition.float(), initial_state.float()) + state_ext


def apply_kda_state_gradient_summary(
    grad_state_ext: torch.Tensor,
    transition: torch.Tensor,
    grad_final_state: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply the adjoint summary to a gradient from the next rank.

    Args:
        grad_state_ext: Local gradient contribution G.
        transition: Forward transfer M used by this backend's existing adjoint.
        grad_final_state: Incoming terminal-state gradient, or None for zero exit.
    """
    if grad_final_state is None:
        return grad_state_ext
    with torch.autocast(device_type=grad_state_ext.device.type, enabled=False):
        return (
            torch.matmul(transition.float().transpose(-2, -1), grad_final_state.float())
            + grad_state_ext
        )


_KDA_BACKENDS = frozenset({"eager", "triton", "ascendc"})
_TRITON_KDA_HEAD_DIM = 128
_TRITON_KDA_CHUNK_SIZE = 64


def _has_triton_kda_shapes(
    query: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    chunk_size: int,
) -> bool:
    """Check the fixed dense shapes accepted by the Triton KDA path."""
    batch_size, sequence_length, num_query_heads, key_dim = query.shape
    num_value_heads, value_dim = value.shape[2:]
    return all((
        value.shape[:2] == (batch_size, sequence_length),
        gate.shape == (batch_size, sequence_length, num_value_heads, key_dim),
        beta.shape == (batch_size, sequence_length, num_value_heads),
        num_value_heads % num_query_heads == 0,
        a_log.numel() == num_value_heads,
        dt_bias.numel() == num_value_heads * key_dim,
        key_dim == value_dim == _TRITON_KDA_HEAD_DIM,
        chunk_size == _TRITON_KDA_CHUNK_SIZE,
    ))


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

    shape_supported = _has_triton_kda_shapes(
        query, value, gate, beta, a_log, dt_bias, chunk_size,
    )
    lower_bound_supported = -5.0 <= lower_bound < 0
    if not (shape_supported and lower_bound_supported):
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
    """Return whether the validated Triton-Ascend KDA backend is available.

    Args:
        query: Query tensor in batch, token, head, key-dimension order.
        key: Key tensor with the same token and key-head axes as query.
        value: Value tensor in batch, token, head, value-dimension order.
        gate: Decay gate or output gate required by this stage.
        beta: Delta-rule update strengths, or logits when sigmoid fusion is enabled.
        a_log: Log decay-rate parameter.
        dt_bias: Per-key time-step bias.
        chunk_size: Number of tokens in a recurrence chunk.
        lower_bound: Negative lower bound of the safe decay gate.
    """
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
    return is_fla_triton_kda_available()


def _normalize_kda_backend(backend: str) -> str:
    """Normalize the selected backend and reject unsupported names."""
    backend = backend.lower()
    if backend not in _KDA_BACKENDS:
        raise ValueError(
            f"unsupported KDA backend {backend!r}; "
            f"expected one of {sorted(_KDA_BACKENDS)}."
        )
    return backend


def _validate_fused_state_contract(initial_state, output_final_state, qk_norm, gate, beta, backend) -> None:
    """Keep raw-input dispatch separate from the stateful CP backend contract."""
    if initial_state is not None or output_final_state:
        raise NotImplementedError(
            f"KDA backend={backend!r} currently supports zero initial state without "
            "returning a final state; stateful execution is owned by the CP backend."
        )
    if not (qk_norm and gate and beta):
        raise NotImplementedError(
            f"KDA backend={backend!r} requires fused Q/K normalization, gate activation, "
            "and beta sigmoid."
        )


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
    cu_seqlens: Optional[torch.Tensor | Sequence[int]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Dispatch dense or packed KDA to the explicitly selected local backend.

    Args:
        query: Query tensor in batch, token, head, key-dimension order.
        key: Key tensor with the same token and key-head axes as query.
        value: Value tensor in batch, token, head, value-dimension order.
        gate: Decay gate or output gate required by this stage.
        beta: Delta-rule update strengths, or logits when sigmoid fusion is enabled.
        a_log: Log decay-rate parameter.
        dt_bias: Per-key time-step bias.
        scale: Query scaling factor; None selects the head-dimension default.
        initial_state: Optional state entering the first local chunk.
        output_final_state: Whether to return the state after the final chunk.
        use_qk_l2norm_in_kernel: Whether this operator normalizes query and key.
        use_gate_in_kernel: Whether this operator transforms raw decay logits.
        use_beta_sigmoid_in_kernel: Whether this operator transforms beta logits with sigmoid.
        lower_bound: Negative lower bound of the safe decay gate.
        chunk_size: Number of tokens in a recurrence chunk.
        backend: Explicit local operator implementation to dispatch.
        cu_seqlens: Cumulative document boundaries for packed tokens.
    """
    if query.ndim == 3:
        output, final = chunk_kda(
            *(tensor.unsqueeze(0) for tensor in (query, key, value, gate, beta)),
            a_log=a_log, dt_bias=dt_bias, scale=scale, initial_state=initial_state,
            output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel, use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            lower_bound=lower_bound, chunk_size=chunk_size, backend=backend, cu_seqlens=cu_seqlens)
        return output.squeeze(0), final
    backend = _normalize_kda_backend(backend)
    if backend == "eager":
        return torch_chunk_kda(
            query, key, value, gate, beta,
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
            cu_seqlens=cu_seqlens,
        )
    _validate_fused_state_contract(initial_state, output_final_state, use_qk_l2norm_in_kernel,
                                  use_gate_in_kernel, use_beta_sigmoid_in_kernel, backend)
    if backend == "ascendc":
        output = ascendc_chunk_kda(
            query, key, value, gate, beta,
            a_log=a_log.float() if a_log is not None else None,
            dt_bias=dt_bias.float() if dt_bias is not None else None,
            scale=scale, lower_bound=lower_bound, chunk_size=chunk_size, cu_seqlens=cu_seqlens,
        )
        return output, None
    triton_a_log = a_log.float() if a_log is not None else None
    triton_dt_bias = dt_bias.float() if dt_bias is not None else None
    if not _is_triton_kda_input_supported(
        query, key, value, gate, beta, triton_a_log, triton_dt_bias,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    ):
        raise RuntimeError(
            "KDA backend='triton' received an unsupported input contract; it requires "
            "q/k/v/gate/beta=bf16, a_log/dt_bias=fp32, "
            "head_k_dim=head_v_dim=128, chunk_size=64, and a supported safe gate."
        )


    output = fused_chunk_kda(
        query, key, value, gate, beta,
        a_log=triton_a_log,
        dt_bias=triton_dt_bias,
        scale=scale,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        safe_gate=True,
        cu_seqlens=cu_seqlens,
    )
    return output, None
