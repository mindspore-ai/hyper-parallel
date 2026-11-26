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
"""Local GDN math, state application and explicit backend selection."""
from __future__ import annotations
import importlib
import importlib.metadata
import importlib.util
import re
from typing import Optional

import torch
from torch.nn import functional as F

from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import packed_eager
from .gated_delta_net_ascendc import chunk_gated_delta_rule_ascendc


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize along ``dim``."""
    # ``(x * x).sum`` (MulBackward) instead of ``x.pow(2).sum`` (PowBackward):
    # equal math, NPU yields ULP-different gradients across the two ops.
    # NPU autocast promotes the reduction; the native core still consumes x.dtype.
    return (x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)).to(x.dtype)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    g: torch.Tensor, beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    normalize_qk: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunked gated delta-rule linear attention (eager reference).

    fp32 internal compute, output cast back to input dtype.

    Shapes
    ------
    query / key:  ``(B, S, num_k_heads, head_k_dim)``
    value:        ``(B, S, num_v_heads, head_v_dim)`` (here num_k_heads == num_v_heads)
    g:            ``(B, S, num_v_heads)``
    beta:         ``(B, S, num_v_heads)``

    Args:
        query: Query tensor in batch, token, head, key-dimension order.
        key: Key tensor with the same token and key-head axes as query.
        value: Value tensor in batch, token, head, value-dimension order.
        g: Log decay values for the recurrence.
        beta: Delta-rule update strengths, or logits when sigmoid fusion is enabled.
        chunk_size: Number of tokens in a recurrence chunk.
        initial_state: Optional state entering the first local chunk.
        output_final_state: Whether to return the state after the final chunk.
        normalize_qk: Whether this operator normalizes query and key.
        cu_seqlens: Cumulative document boundaries for packed tokens.
    """
    if cu_seqlens is not None:
        return packed_eager(
            torch_chunk_gated_delta_rule, (query, key, value, g, beta), cu_seqlens,
            initial_state=initial_state, output_final_state=output_final_state,
            normalize_qk=normalize_qk, chunk_size=chunk_size,
        )
    initial_dtype = query.dtype
    if normalize_qk:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    # (B, S, H, D) → (B, H, S, D), fp32
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # (B, H, S, D) → (B, H, n_chunks, chunk_size, D)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def apply_gdn_state_summary(
    state_ext: torch.Tensor,
    transition: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply GDN's existing local affine state program.

    Args:
        state_ext: Local zero-entry state S.
        transition: Local transfer M, omitted only for a zero entry.
        initial_state: Incoming state, or None for a zero entry.
    """
    if initial_state is None:
        return state_ext
    return torch.matmul(transition, initial_state.float()) + state_ext


def apply_gdn_state_gradient_summary(
    grad_state_ext: torch.Tensor,
    transition: Optional[torch.Tensor],
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
    return (
        torch.matmul(transition.transpose(-2, -1), grad_final_state.float())
        + grad_state_ext
    )


_GDN_BACKENDS = frozenset({"eager", "triton", "ascendc"})
_MIN_TRITON_ASCEND_VERSION = (3, 2, 1)
_TRITON_GDN_HEAD_DIM = 128
_TRITON_GDN_CHUNK_SIZE = 64


def _parse_version(version_text: str) -> tuple[int, int, int]:
    """Return a three-component numeric version tuple."""
    parts = []
    for part in version_text.split("+")[0].split(".")[:3]:
        match = re.match(r"\d+", part)
        if match is not None:
            parts.append(int(match.group()))
    return tuple((parts + [0, 0, 0])[:3])


def _is_triton_gdn_input_supported(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
) -> bool:
    """Check the fixed Qwen3.5 GDN contract validated by this backend."""
    if any(tensor is None for tensor in (key, value, g, beta)):
        return False
    if not (
        query.device.type == "npu"
        and query.dtype == key.dtype == value.dtype == beta.dtype == torch.bfloat16
        and g.dtype == torch.float32
        and query.ndim == key.ndim == value.ndim == 4
        and g.ndim == beta.ndim == 3
        and query.shape == key.shape
        and query.shape[:3] == value.shape[:3] == g.shape == beta.shape
        and query.shape[-1] == value.shape[-1] == _TRITON_GDN_HEAD_DIM
    ):
        return False
    return all(
        tensor.device == query.device
        for tensor in (key, value, g, beta)
    )


def is_triton_gdn_available(
    query: Optional[torch.Tensor] = None,
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    chunk_size: int = _TRITON_GDN_CHUNK_SIZE,
) -> bool:
    """Return whether the validated Triton-Ascend GDN backend is available.

    Args:
        query: Query tensor in batch, token, head, key-dimension order.
        key: Key tensor with the same token and key-head axes as query.
        value: Value tensor in batch, token, head, value-dimension order.
        g: Log decay values for the recurrence.
        beta: Delta-rule update strengths, or logits when sigmoid fusion is enabled.
        chunk_size: Number of tokens in a recurrence chunk.
    """
    if chunk_size != _TRITON_GDN_CHUNK_SIZE:
        return False
    if query is not None and not _is_triton_gdn_input_supported(
        query, key, value, g, beta,
    ):
        return False
    try:
        version_text = importlib.metadata.version("triton-ascend")
    except importlib.metadata.PackageNotFoundError:
        return False
    version = _parse_version(version_text)
    if version < _MIN_TRITON_ASCEND_VERSION:
        return False
    try:
        importlib.import_module("triton")
        return importlib.util.find_spec("triton.backends.ascend") is not None
    except (AttributeError, ImportError, ModuleNotFoundError):
        return False


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    backend: str = "eager",
    cu_seqlens: Optional[torch.Tensor] = None,
    use_beta_sigmoid_in_kernel: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Dispatch GDN to an explicitly selected eager, Triton or Ascend C backend.

    Args:
        query: Query tensor in batch, token, head, key-dimension order.
        key: Key tensor with the same token and key-head axes as query.
        value: Value tensor in batch, token, head, value-dimension order.
        g: Log decay values for the recurrence.
        beta: Delta-rule update strengths, or logits when sigmoid fusion is enabled.
        chunk_size: Number of tokens in a recurrence chunk.
        initial_state: Optional state entering the first local chunk.
        output_final_state: Whether to return the state after the final chunk.
        use_qk_l2norm_in_kernel: Whether this operator normalizes query and key.
        backend: Explicit local operator implementation to dispatch.
        cu_seqlens: Cumulative document boundaries for packed tokens.
        use_beta_sigmoid_in_kernel: Interpret beta as raw logits and fuse when supported.
    """
    if query.ndim == 3:
        output, final = chunk_gated_delta_rule(
            *(tensor.unsqueeze(0) for tensor in (query, key, value, g, beta)),
            initial_state=initial_state, output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, chunk_size=chunk_size,
            backend=backend, cu_seqlens=cu_seqlens,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel)
        return output.squeeze(0), final
    backend = backend.lower()
    if backend not in _GDN_BACKENDS:
        raise ValueError(
            f"unsupported GDN backend {backend!r}; "
            f"expected one of {sorted(_GDN_BACKENDS)}."
        )
    if use_beta_sigmoid_in_kernel and backend != "ascendc":
        beta = beta.sigmoid()
    if backend == "eager":
        return torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            normalize_qk=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )
    if backend == "ascendc":
        if initial_state is not None or output_final_state:
            raise NotImplementedError(
                "The local GDN Ascend C entry does not expose recurrent states; state CP uses its staged adapter.")
        if chunk_size != 64:
            raise ValueError("GDN Ascend C requires chunk_size=64.")
        return chunk_gated_delta_rule_ascendc(
            query, key, value, g, beta, cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel)
    if not is_triton_gdn_available(
        query, key, value, g, beta, chunk_size=chunk_size,
    ):
        raise RuntimeError(
            "GDN backend='triton' requires triton-ascend >= 3.2.1 "
            "installation with the Ascend backend and NPU inputs "
            "q/k/v/beta=bf16, g=fp32, head_k_dim=head_v_dim=128, "
            "and chunk_size=64."
        )

    # The vendored Triton kernels are an optional NPU dependency.
    from hyper_parallel.components.functional.gated_delta_net_triton import (  # pylint: disable=import-outside-toplevel
        chunk_gated_delta_rule as triton_chunk_gated_delta_rule,
    )

    return triton_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
