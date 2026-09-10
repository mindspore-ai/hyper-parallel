# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Gated DeltaNet eager oracle and explicit local-backend dispatcher."""
# pylint: disable=forbidden-backend-import,missing-public-type-hints
# pylint: disable=missing-public-docstring,not-callable

import importlib
import importlib.metadata
import importlib.util
import re
from typing import Optional

import torch
from torch.nn import functional as F

_GDN_BACKENDS = frozenset({"eager", "triton"})
_MIN_TRITON_ASCEND_VERSION = (3, 2, 1)
_TRITON_GDN_HEAD_DIM = 128
_TRITON_GDN_CHUNK_SIZE = 64


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize along ``dim``."""
    # ``(x * x).sum`` (MulBackward) instead of ``x.pow(2).sum`` (PowBackward):
    # equal math, NPU yields ULP-different gradients across the two ops.
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Chunked gated delta-rule linear attention (pure torch reference).

    fp32 internal compute, output cast back to input dtype.

    Shapes
    ------
    query / key:  ``(B, S, num_k_heads, head_k_dim)``
    value:        ``(B, S, num_v_heads, head_v_dim)`` (here num_k_heads == num_v_heads)
    g:            ``(B, S, num_v_heads)``
    beta:         ``(B, S, num_v_heads)``
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
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
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

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
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1],
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _parse_version(version_text: str) -> tuple[int, int, int]:
    """Return a three-component numeric version tuple."""
    parts = [
        int(match.group())
        for part in version_text.split("+")[0].split(".")[:3]
        if (match := re.match(r"\d+", part)) is not None
    ]
    return tuple((parts + [0, 0, 0])[:3])


def _is_triton_gdn_input_supported(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
) -> bool:
    """Check the fixed Qwen3.5 GDN contract validated by this backend."""
    if key is None or value is None or g is None or beta is None:
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
        and query.shape[1] % _TRITON_GDN_CHUNK_SIZE == 0
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
    """Return whether the validated Triton-Ascend GDN backend is available."""
    if chunk_size != _TRITON_GDN_CHUNK_SIZE:
        return False
    if query is not None and not _is_triton_gdn_input_supported(
        query, key, value, g, beta
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
):
    """Dispatch GDN to an explicitly selected eager or Triton backend."""
    backend = backend.lower()
    if backend not in _GDN_BACKENDS:
        raise ValueError(
            f"unsupported GDN backend {backend!r}; "
            f"expected one of {sorted(_GDN_BACKENDS)}."
        )
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
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    if not is_triton_gdn_available(
        query, key, value, g, beta, chunk_size=chunk_size
    ):
        raise RuntimeError(
            "GDN backend='triton' requires triton-ascend >= 3.2.1 "
            "installation with the Ascend backend and NPU inputs "
            "q/k/v/beta=bf16, g=fp32, head_k_dim=head_v_dim=128, "
            "chunk_size=64, and sequence length divisible by 64."
        )

    from hyper_parallel.platform.torch.custom_ops.gdn.chunk_gated_delta_rule import (  # pylint: disable=import-outside-toplevel
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
    )



__all__ = ["chunk_gated_delta_rule", "is_triton_gdn_available", "torch_chunk_gated_delta_rule"]
