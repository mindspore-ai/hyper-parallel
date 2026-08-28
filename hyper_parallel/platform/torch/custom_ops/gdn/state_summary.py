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
"""Affine state-summary operations used by GDN State-P2P."""

from typing import Optional

import torch
import triton

from .triton.state_summary import (
    gdn_packed_state_summary_kernel,
    gdn_state_grad_ext_kernel,
)


def _validate_fixed_summary_shape(
    key_dim: int,
    value_dim: int,
    chunk_size: int,
) -> None:
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "Triton GDN state summary requires key_dim=value_dim=128 and "
            "chunk_size=64."
        )


@torch.compiler.disable
def chunk_gated_delta_rule_state_summary_fwd(
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    *,
    chunk_size: int = 64,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the local affine map ``state_out = M @ state_in + S``."""
    if key.ndim != 4 or w.ndim != 4 or u.ndim != 4 or g.ndim != 3:
        raise ValueError("GDN state summary expects key/w/u [B,T,H,D] and g [B,T,H].")
    batch, seq_len, heads, key_dim = key.shape
    value_dim = u.shape[-1]
    _validate_fixed_summary_shape(key_dim, value_dim, chunk_size)
    if block_size not in (64, 128):
        raise ValueError(f"GDN state-summary block_size must be 64 or 128, got {block_size}.")
    if seq_len % chunk_size != 0:
        raise ValueError(
            f"GDN state-summary sequence length {seq_len} must be divisible by {chunk_size}."
        )
    if w.shape != key.shape or u.shape[:3] != key.shape[:3] or g.shape != key.shape[:3]:
        raise ValueError(
            "Incompatible GDN state-summary shapes: "
            f"key={tuple(key.shape)}, w={tuple(w.shape)}, "
            f"u={tuple(u.shape)}, g={tuple(g.shape)}."
        )

    key, w, u, g = (tensor.contiguous() for tensor in (key, w, u, g))
    packed_summary = torch.empty(
        batch,
        heads,
        key_dim,
        value_dim + key_dim,
        device=key.device,
        dtype=torch.float32,
    )
    gdn_packed_state_summary_kernel[
        (triton.cdiv(value_dim + key_dim, block_size), batch * heads)
    ](
        key,
        w,
        u,
        g,
        packed_summary,
        seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BV=block_size,
        NT=seq_len // chunk_size,
    )
    state_ext = packed_summary[..., :value_dim].contiguous()
    transition = packed_summary[..., value_dim:].contiguous()
    return state_ext, transition


@torch.compiler.disable
def chunk_gated_delta_rule_state_gradient_summary_bwd(
    query: torch.Tensor,
    key: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    grad_output: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Return the local-loss contribution to the incoming state gradient."""
    batch, seq_len, heads, key_dim = query.shape
    value_dim = grad_output.shape[-1]
    _validate_fixed_summary_shape(key_dim, value_dim, chunk_size)
    if seq_len % chunk_size != 0:
        raise ValueError(
            f"GDN state-gradient sequence length {seq_len} must be divisible by {chunk_size}."
        )
    qk_shape = (batch, seq_len, heads, key_dim)
    value_shape = (batch, seq_len, heads, value_dim)
    if (
        key.shape != qk_shape
        or w.shape != qk_shape
        or g.shape != qk_shape[:3]
        or grad_output.shape != value_shape
        or dv.shape != value_shape
    ):
        raise ValueError(
            "Incompatible GDN state-gradient summary shapes: "
            f"query={tuple(query.shape)}, key={tuple(key.shape)}, "
            f"w={tuple(w.shape)}, g={tuple(g.shape)}, "
            f"grad_output={tuple(grad_output.shape)}, dv={tuple(dv.shape)}."
        )

    query, key, w, g, grad_output, dv = (
        tensor.contiguous() for tensor in (query, key, w, g, grad_output, dv)
    )
    grad_state_ext = torch.empty(
        batch,
        heads,
        key_dim,
        value_dim,
        device=query.device,
        dtype=torch.float32,
    )
    gdn_state_grad_ext_kernel[(1, batch * heads)](
        query,
        key,
        w,
        g,
        grad_output,
        dv,
        grad_state_ext,
        scale,
        seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BV=128,
        NT=seq_len // chunk_size,
    )
    return grad_state_ext


def apply_gdn_state_summary(
    state_ext: torch.Tensor,
    transition: torch.Tensor,
    initial_state: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply a local affine state summary in FP32."""
    if initial_state is None:
        return state_ext
    return torch.matmul(transition, initial_state.float()) + state_ext


def apply_gdn_state_gradient_summary(
    grad_state_ext: torch.Tensor,
    transition: torch.Tensor,
    grad_final_state: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply the adjoint affine summary to a gradient from the next rank."""
    if grad_final_state is None:
        return grad_state_ext
    return (
        torch.matmul(transition.transpose(-2, -1), grad_final_state.float())
        + grad_state_ext
    )


__all__ = [
    "apply_gdn_state_gradient_summary",
    "apply_gdn_state_summary",
    "chunk_gated_delta_rule_state_gradient_summary_bwd",
    "chunk_gated_delta_rule_state_summary_fwd",
]
