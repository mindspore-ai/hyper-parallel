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
"""Affine KDA state summary built from prepared WY intermediates."""

# pylint: disable=forbidden-backend-import

__all__ = [
    "apply_kda_state_gradient_summary",
    "apply_kda_state_summary",
    "kda_state_gradient_summary_from_prepared",
    "kda_state_summary_forward_from_prepared",
]

from typing import Optional

import torch
import triton


def _validate_prepared_summary_inputs(
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gate: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int]:
    """Validate the fixed Kimi K3 summary contract and return dimensions."""
    if (key.ndim, w.ndim, u.ndim, gate.ndim) != (4, 4, 4, 4):
        raise ValueError("KDA prepared summary expects rank-4 key/w/u/gate tensors.")
    batch, sequence_length, heads, key_dim = key.shape
    value_dim = u.shape[-1]
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "Triton KDA state summary requires key_dim=value_dim=128 and "
            "chunk_size=64."
        )
    if sequence_length <= 0:
        raise ValueError("State summary requires a nonempty sample fragment.")
    if w.shape != key.shape or gate.shape != key.shape:
        raise ValueError(
            "KDA prepared key, w, and gate must have identical shapes, got "
            f"key={tuple(key.shape)}, w={tuple(w.shape)}, gate={tuple(gate.shape)}."
        )
    if u.shape[:3] != key.shape[:3]:
        raise ValueError(
            f"KDA prepared u prefix {tuple(u.shape[:3])} must match "
            f"key prefix {tuple(key.shape[:3])}."
        )
    if not key.is_npu:
        raise RuntimeError("Triton KDA state summary requires Ascend NPU tensors.")
    return batch, sequence_length, heads, key_dim, value_dim


def _validate_gradient_summary_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    w: torch.Tensor,
    gate: torch.Tensor,
    grad_output: torch.Tensor,
    grad_value: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int]:
    """Validate prepared tensors used by the reverse state wavefront."""
    tensors = (query, key, w, gate, grad_output, grad_value)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError(
            "KDA gradient summary expects rank-4 query/key/w/gate/do/dv tensors."
        )
    batch, sequence_length, heads, key_dim = query.shape
    value_dim = grad_output.shape[-1]
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "Triton KDA state-gradient summary requires key_dim=value_dim=128 "
            "and chunk_size=64."
        )
    if sequence_length <= 0:
        raise ValueError("State summary requires a nonempty sample fragment.")
    prepared_shape = (batch, sequence_length, heads, key_dim)
    value_shape = (batch, sequence_length, heads, value_dim)
    if key.shape != prepared_shape or w.shape != prepared_shape:
        raise ValueError(
            "KDA prepared query, key, and w must have identical shapes, got "
            f"query={tuple(query.shape)}, key={tuple(key.shape)}, "
            f"w={tuple(w.shape)}."
        )
    if gate.shape != prepared_shape:
        raise ValueError(
            f"KDA prepared gate shape {tuple(gate.shape)} must match "
            f"query shape {prepared_shape}."
        )
    if grad_output.shape != value_shape or grad_value.shape != value_shape:
        raise ValueError(
            "KDA grad_output and grad_value must have shape "
            f"{value_shape}, got do={tuple(grad_output.shape)}, "
            f"dv={tuple(grad_value.shape)}."
        )
    if not query.is_npu:
        raise RuntimeError(
            "Triton KDA state-gradient summary requires Ascend NPU tensors."
        )
    return batch, sequence_length, heads, key_dim, value_dim


def _launch_kda_state_summary(
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gate: torch.Tensor,
    *,
    chunk_size: int,
    output_transition: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Launch the shared S/M summary used by the state CP protocols."""
    return _launch_kda_paired_state_summary(
        key,
        w,
        u,
        gate,
        chunk_size=chunk_size,
        output_transition=output_transition,
    )


def _launch_kda_state_gradient_summary(
    query: torch.Tensor,
    key: torch.Tensor,
    w: torch.Tensor,
    gate: torch.Tensor,
    grad_output: torch.Tensor,
    grad_value: torch.Tensor,
    scale: float,
    *,
    chunk_size: int,
    block_size: int = 64,
) -> torch.Tensor:
    """Launch the reverse-wavefront Triton-Ascend summary kernel."""
    from ._triton.kimi_delta_attention.state_summary import (  # pylint: disable=import-outside-toplevel
        kda_state_grad_ext_kernel,
    )

    batch, sequence_length, heads, key_dim, value_dim = (
        _validate_gradient_summary_inputs(
            query,
            key,
            w,
            gate,
            grad_output,
            grad_value,
            chunk_size,
        )
    )
    query, key, w, gate, grad_output, grad_value = (
        tensor.contiguous()
        for tensor in (query, key, w, gate, grad_output, grad_value)
    )
    grad_state_ext = torch.empty(
        batch,
        heads,
        key_dim,
        value_dim,
        dtype=torch.float32,
        device=query.device,
    )
    kda_state_grad_ext_kernel[
        (triton.cdiv(value_dim, block_size), batch * heads)
    ](
        query,
        key,
        w,
        gate,
        grad_output,
        grad_value,
        grad_state_ext,
        float(scale),
        sequence_length,
        H=heads,
        H_TOTAL=heads,
        HEAD_OFFSET=0,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BV=block_size,
        num_warps=2,
    )
    return grad_state_ext


@torch.compiler.disable
def _launch_kda_paired_state_summary(
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gate: torch.Tensor,
    *,
    chunk_size: int = 64,
    output_transition: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build Ascend FLA-paired ``S_ext`` and an explicit FP32 transfer ``M``."""
    from ._triton.kimi_delta_attention.state_summary import (  # pylint: disable=import-outside-toplevel
        kda_split_state_summary_kernel,
        kda_state_transition_kernel,
    )

    batch, sequence_length, heads, key_dim, value_dim = (
        _validate_prepared_summary_inputs(key, w, u, gate, chunk_size)
    )
    key, w, u, gate = (
        tensor.contiguous() for tensor in (key, w, u, gate)
    )
    state_ext = torch.empty(
        batch,
        heads,
        key_dim,
        value_dim,
        dtype=torch.float32,
        device=key.device,
    )
    transition = None
    if output_transition:
        transition = torch.empty(
            batch,
            heads,
            key_dim,
            key_dim,
            dtype=torch.float32,
            device=key.device,
        )
    kda_split_state_summary_kernel[(1, batch * heads)](
        key,
        w,
        u,
        gate,
        state_ext,
        transition if transition is not None else state_ext,
        sequence_length,
        H=heads,
        H_TOTAL=heads,
        HEAD_OFFSET=0,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BV=128,
        OUTPUT_MODE=1,
        num_warps=2,
    )
    if output_transition:
        kda_state_transition_kernel[(triton.cdiv(key_dim, 64), batch * heads)](
            key,
            w,
            gate,
            transition,
            sequence_length,
            H=heads,
            K=key_dim,
            BT=chunk_size,
            BV=64,
            num_warps=4,
            num_stages=2,
        )
    return state_ext, transition


@torch.compiler.disable
def kda_state_summary_forward_from_prepared(
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gate: torch.Tensor,
    *,
    chunk_size: int = 64,
    output_transition: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build the local affine summary for state context parallelism.

    ``S_ext`` follows Ascend FLA h's FP32 feedback. ``M`` composes explicit
    chunk transfers with FP32 feedback, following FLA CP.
    ``output_transition=False`` returns only S and skips M allocation/computation.
    It is valid for a zero-entry P2P endpoint whose backward never applies M.

    Args:
        key: Prepared keys, shaped [batch, tokens, heads, key_dim].
        w: Prepared feedback weights with the same shape as key.
        u: Prepared values, shaped [batch, tokens, heads, value_dim].
        gate: Per-chunk cumulative log2 gates matching key.
        chunk_size: Number of tokens per local chunk; the kernel requires 64.
        output_transition: Whether the caller needs the local transition M.

    Returns:
        The FP32 state S and the FP32 transition M, or None when M is omitted.
    """
    return _launch_kda_state_summary(
        key,
        w,
        u,
        gate,
        chunk_size=chunk_size,
        output_transition=output_transition,
    )


@torch.compiler.disable
def kda_state_gradient_summary_from_prepared(
    query: torch.Tensor,
    key: torch.Tensor,
    w: torch.Tensor,
    gate: torch.Tensor,
    grad_output: torch.Tensor,
    grad_value: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Return the local-loss contribution to the incoming KDA state gradient.

    Match Ascend FLA ``dhu`` at zero terminal state: scale ``grad_output``
    before its dot-input cast and preserve the two K64 slabs and subtraction
    grouping. This summary omits the full ``dh``/``dv`` outputs. All inputs
    are read-only; the affine application to a nonzero state is a separate step.

    Args:
        query: Prepared gated queries [B,T,H,K].
        key: Matching prepared gated keys.
        w: Prepared feedback weights matching key.
        gate: Per-chunk cumulative log2 gates matching key.
        grad_output: Original token-output gradients [B,T,H,V].
        grad_value: Local value gradients before recurrent feedback.
        scale: Attention scaling factor.
        chunk_size: Tokens per local chunk; the kernel requires 64.

    Returns:
        FP32 local contribution G, shaped [B,H,K,V].
    """
    return _launch_kda_state_gradient_summary(
        query,
        key,
        w,
        gate,
        grad_output,
        grad_value,
        scale,
        chunk_size=chunk_size,
        block_size=128,
    )


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
    return torch.matmul(transition, initial_state.float()) + state_ext


def apply_kda_state_gradient_summary(
    grad_state_ext: torch.Tensor,
    transition: torch.Tensor,
    grad_final_state: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply the adjoint summary for the reverse P2P state wavefront."""
    if grad_final_state is None:
        return grad_state_ext
    return (
        torch.matmul(transition.transpose(-2, -1), grad_final_state.float())
        + grad_state_ext
    )
