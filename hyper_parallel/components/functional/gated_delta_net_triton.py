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
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Keep the validated kernel adapter close to its upstream implementation.
# pylint: disable=line-too-long,missing-public-type-hints,missing-public-docstring
# pylint: disable=non-google-docstring,disallowed-name,unused-argument,invalid-name
# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=abstract-method,arguments-differ

import warnings
from typing import Any, Optional

import torch
import triton

from ._triton.gated_delta_net.state_summary import (
    gdn_packed_state_summary_kernel, gdn_state_grad_ext_kernel,
)
from ._triton.gated_delta_net.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from ._triton.gated_delta_net.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from ._triton.gated_delta_net.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from ._triton.gated_delta_net.cumsum import chunk_local_cumsum
from ._triton.gated_delta_net.solve_tril import solve_tril
from ._triton.gated_delta_net.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from ._triton.gated_delta_net.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    inv_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return (x * inv_norm).to(x.dtype), inv_norm


def chunk_gated_delta_rule_fwd_prepare(
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
) -> tuple:
    """Compute forward intermediates that do not depend on the initial state.

    Args:
        k: Key activations in token-major layout.
        v: Value activations in token-major layout.
        g: Log-decay gates, cumulative when supplied by prepare.
        beta: Activated delta-rule update weights.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
        chunk_size: Tokens per recurrence chunk.
    """
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=False)
    matrix_a = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    matrix_a = solve_tril(A=matrix_a, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=matrix_a,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    return g, matrix_a, w, u


def chunk_gated_delta_rule_fwd_apply_state(
        k: torch.Tensor,
        g: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
) -> tuple:
    """Apply the recurrent initial state and return local state intermediates.

    Args:
        k: Key activations in token-major layout.
        g: Log-decay gates, cumulative when supplied by prepare.
        w: Prepared key/update projection for the recurrent state.
        u: Prepared value/update projection.
        initial_state: Optional state entering the local sequence.
        output_final_state: Whether to return the terminal state.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
        chunk_size: Tokens per recurrence chunk.
    """
    return chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )


def chunk_gated_delta_rule_fwd_output(
        q: torch.Tensor,
        k: torch.Tensor,
        v_new: torch.Tensor,
        h: torch.Tensor,
        g: torch.Tensor,
        scale: float,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
) -> torch.Tensor:
    """Compute local outputs after recurrent states have been applied.

    Args:
        q: Query activations in token-major layout.
        k: Key activations in token-major layout.
        v_new: Values corrected by their incoming chunk states.
        h: Saved chunk-entry recurrent states.
        g: Log-decay gates, cumulative when supplied by prepare.
        scale: Attention output scaling factor.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
        chunk_size: Tokens per recurrence chunk.
    """
    return chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )


def chunk_gated_delta_rule_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
) -> tuple:
    """Prepare chunks, apply their incoming state and finalize token outputs.

    Args:
        q: Query activations in token-major layout.
        k: Key activations in token-major layout.
        v: Value activations in token-major layout.
        g: Log-decay gates, cumulative when supplied by prepare.
        beta: Activated delta-rule update weights.
        scale: Attention output scaling factor.
        initial_state: Optional state entering the local sequence.
        output_final_state: Whether to return the terminal state.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
        chunk_size: Tokens per recurrence chunk.
    """
    g, matrix_a, w, u = chunk_gated_delta_rule_fwd_prepare(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_apply_state(
        k=k,
        g=g,
        w=w,
        u=u,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    output = chunk_gated_delta_rule_fwd_output(
        q=q,
        k=k,
        v_new=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return g, output, matrix_a, final_state


def chunk_gated_delta_rule_bwd_prepare(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        do: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
) -> tuple:
    """Compute backward intermediates that do not depend on final-state grad.

    Args:
        q: Query activations in token-major layout.
        k: Key activations in token-major layout.
        v: Value activations in token-major layout.
        g: Log-decay gates, cumulative when supplied by prepare.
        beta: Activated delta-rule update weights.
        A: Solved per-chunk WY representation.
        scale: Attention output scaling factor.
        initial_state: Optional state entering the local sequence.
        do: Gradient of the local output.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
        chunk_size: Tokens per recurrence chunk.
    """
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_apply_state(
        k=k,
        g=g,
        w=w,
        u=u,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return w, h, v_new, dv


def chunk_gated_delta_rule_bwd_state(
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        dht: Optional[torch.Tensor],
        do: torch.Tensor,
        dv: torch.Tensor,
        scale: float,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
) -> tuple:
    """Apply the final-state gradient and produce the initial-state gradient.

    Args:
        q: Query activations in token-major layout.
        k: Key activations in token-major layout.
        w: Prepared key/update projection for the recurrent state.
        g: Log-decay gates, cumulative when supplied by prepare.
        initial_state: Optional state entering the local sequence.
        dht: Optional gradient of the terminal recurrent state.
        do: Gradient of the local output.
        dv: Value gradient to propagate through the next local stage.
        scale: Attention output scaling factor.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
        chunk_size: Tokens per recurrence chunk.
    """
    return chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )


def chunk_gated_delta_rule_bwd_finish(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        w: torch.Tensor,
        h: torch.Tensor,
        v_new: torch.Tensor,
        dv: torch.Tensor,
        do: torch.Tensor,
        dh: torch.Tensor,
        scale: float,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
) -> tuple:
    """Finish local tensor gradients after the state-gradient handoff.

    Args:
        q: Query activations in token-major layout.
        k: Key activations in token-major layout.
        v: Value activations in token-major layout.
        g: Log-decay gates, cumulative when supplied by prepare.
        beta: Activated delta-rule update weights.
        A: Solved per-chunk WY representation.
        w: Prepared key/update projection for the recurrent state.
        h: Saved chunk-entry recurrent states.
        v_new: Values corrected by their incoming chunk states.
        dv: Value gradient to propagate through the next local stage.
        do: Gradient of the local output.
        dh: Gradients of chunk-entry recurrent states.
        scale: Attention output scaling factor.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
        chunk_size: Tokens per recurrence chunk.
    """
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype} , should be float32")
    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
    )
    return dq, dk, dv, db, dg


def chunk_gated_delta_rule_bwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
) -> tuple:
    """Reconstruct local intermediates and return raw input gradients.

    Args:
        q: Query activations in token-major layout.
        k: Key activations in token-major layout.
        v: Value activations in token-major layout.
        g: Log-decay gates, cumulative when supplied by prepare.
        beta: Activated delta-rule update weights.
        A: Solved per-chunk WY representation.
        scale: Attention output scaling factor.
        initial_state: Optional state entering the local sequence.
        do: Gradient of the local output.
        dht: Optional gradient of the terminal recurrent state.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
        chunk_size: Tokens per recurrence chunk.
    """
    w, h, v_new, dv = chunk_gated_delta_rule_bwd_prepare(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        scale=scale,
        initial_state=initial_state,
        do=do,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_state(
        q=q,
        k=k,
        w=w,
        g=g,
        initial_state=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dq, dk, dv, db, dg = chunk_gated_delta_rule_bwd_finish(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        w=w,
        h=h,
        v_new=v_new,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return dq, dk, dv, db, dg, dh0


@torch.compiler.disable
@input_guard
def chunk_gated_delta_rule_fwd_prepare_saved(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
        cu_seqlens: torch.Tensor = None,
) -> tuple:
    """Prepare fused GDN forward tensors without consuming the initial state.

    Args:
        q: Query activations in token-major layout.
        k: Key activations in token-major layout.
        v: Value activations in token-major layout.
        g: Log-decay gates, cumulative when supplied by prepare.
        beta: Activated delta-rule update weights.
        scale: Attention output scaling factor.
        use_qk_l2norm_in_kernel: Whether local stages normalize Q/K and differentiate that normalization.
        chunk_size: Tokens per recurrence chunk.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    q_norm, q_inv_norm = q, q.new_empty(0)
    k_norm, k_inv_norm = k, k.new_empty(0)
    if use_qk_l2norm_in_kernel:
        q_norm, q_inv_norm = _l2norm(q)
        k_norm, k_inv_norm = _l2norm(k)

    g_cumsum, matrix_a, w, u = chunk_gated_delta_rule_fwd_prepare(
        k=k_norm,
        v=v,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    return (
        q_norm,
        k_norm,
        q_inv_norm,
        k_inv_norm,
        g_cumsum,
        matrix_a,
        w,
        u,
        scale,
    )


@torch.compiler.disable
@input_guard
def chunk_gated_delta_rule_fwd_apply_state_saved(
        k_norm: torch.Tensor,
        g_cumsum: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        initial_state: torch.Tensor = None,
        output_final_state: bool = True,
        chunk_size: int = 64,
        cu_seqlens: torch.Tensor = None,
) -> tuple:
    """Apply an initial state to fused prepared forward tensors.

    Args:
        k_norm: Normalized key activations.
        g_cumsum: Per-chunk cumulative log-decay gates.
        w: Prepared key/update projection for the recurrent state.
        u: Prepared value/update projection.
        initial_state: Optional state entering the local sequence.
        output_final_state: Whether to return the terminal state.
        chunk_size: Tokens per recurrence chunk.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
    """
    return chunk_gated_delta_rule_fwd_apply_state(
        k=k_norm,
        g=g_cumsum,
        w=w,
        u=u,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )


@torch.compiler.disable
@input_guard
def chunk_gated_delta_rule_fwd_output_saved(
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        g_cumsum: torch.Tensor,
        h: torch.Tensor,
        v_new: torch.Tensor,
        scale: float,
        chunk_size: int = 64,
        cu_seqlens: torch.Tensor = None,
) -> torch.Tensor:
    """Compute fused GDN output from prepared, state-applied tensors.

    Args:
        q_norm: Normalized query activations.
        k_norm: Normalized key activations.
        g_cumsum: Per-chunk cumulative log-decay gates.
        h: Saved chunk-entry recurrent states.
        v_new: Values corrected by their incoming chunk states.
        scale: Attention output scaling factor.
        chunk_size: Tokens per recurrence chunk.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
    """
    return chunk_gated_delta_rule_fwd_output(
        q=q_norm,
        k=k_norm,
        v_new=v_new,
        h=h,
        g=g_cumsum,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )


@torch.compiler.disable
@input_guard
def chunk_gated_delta_rule_fwd_saved(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float = None,
        initial_state: torch.Tensor = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
        cu_seqlens: torch.Tensor = None,
) -> tuple:
    """Run fused GDN forward and return the tensors required by its backward.

    Args:
        q: Query activations in token-major layout.
        k: Key activations in token-major layout.
        v: Value activations in token-major layout.
        g: Log-decay gates, cumulative when supplied by prepare.
        beta: Activated delta-rule update weights.
        scale: Attention output scaling factor.
        initial_state: Optional state entering the local sequence.
        use_qk_l2norm_in_kernel: Whether local stages normalize Q/K and differentiate that normalization.
        chunk_size: Tokens per recurrence chunk.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
    """
    (
        q_norm,
        k_norm,
        q_inv_norm,
        k_inv_norm,
        g_cumsum,
        matrix_a,
        w,
        u,
        scale,
    ) = chunk_gated_delta_rule_fwd_prepare_saved(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_apply_state_saved(
        k_norm,
        g_cumsum,
        w,
        u,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    output = chunk_gated_delta_rule_fwd_output_saved(
        q_norm,
        k_norm,
        g_cumsum,
        h,
        v_new,
        scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    return (
        output.to(q.dtype),
        final_state,
        q_norm,
        k_norm,
        q_inv_norm,
        k_inv_norm,
        g_cumsum,
        matrix_a,
        scale,
    )


@torch.compiler.disable
@input_guard
def chunk_gated_delta_rule_bwd_prepare_saved(
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        v: torch.Tensor,
        g_cumsum: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        initial_state: torch.Tensor,
        grad_output: torch.Tensor,
        scale: float,
        chunk_size: int = 64,
        cu_seqlens: torch.Tensor = None,
) -> tuple:
    """Prepare fused backward tensors before the final-state grad arrives.

    Args:
        q_norm: Normalized query activations.
        k_norm: Normalized key activations.
        v: Value activations in token-major layout.
        g_cumsum: Per-chunk cumulative log-decay gates.
        beta: Activated delta-rule update weights.
        A: Solved per-chunk WY representation.
        initial_state: Optional state entering the local sequence.
        grad_output: Gradient of the local output.
        scale: Attention output scaling factor.
        chunk_size: Tokens per recurrence chunk.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
    """
    return chunk_gated_delta_rule_bwd_prepare(
        q=q_norm,
        k=k_norm,
        v=v,
        g=g_cumsum,
        beta=beta,
        A=A,
        scale=scale,
        initial_state=initial_state,
        do=grad_output,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )


@torch.compiler.disable
@input_guard
def chunk_gated_delta_rule_bwd_state_saved(
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        g_cumsum: torch.Tensor,
        w: torch.Tensor,
        initial_state: torch.Tensor,
        grad_final_state: torch.Tensor,
        grad_output: torch.Tensor,
        dv_local: torch.Tensor,
        scale: float,
        chunk_size: int = 64,
        cu_seqlens: torch.Tensor = None,
) -> tuple:
    """Consume the final-state grad and produce the initial-state grad.

    Args:
        q_norm: Normalized query activations.
        k_norm: Normalized key activations.
        g_cumsum: Per-chunk cumulative log-decay gates.
        w: Prepared key/update projection for the recurrent state.
        initial_state: Optional state entering the local sequence.
        grad_final_state: Optional gradient of the terminal recurrent state.
        grad_output: Gradient of the local output.
        dv_local: Output-stage value gradient before recurrent correction.
        scale: Attention output scaling factor.
        chunk_size: Tokens per recurrence chunk.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
    """
    return chunk_gated_delta_rule_bwd_state(
        q=q_norm,
        k=k_norm,
        w=w,
        g=g_cumsum,
        initial_state=initial_state,
        dht=grad_final_state,
        do=grad_output,
        dv=dv_local,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )


@torch.compiler.disable
@input_guard
def chunk_gated_delta_rule_bwd_finish_saved(
        q: torch.Tensor,
        k: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        v: torch.Tensor,
        g_cumsum: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        w: torch.Tensor,
        h: torch.Tensor,
        v_new: torch.Tensor,
        dv: torch.Tensor,
        grad_output: torch.Tensor,
        dh: torch.Tensor,
        q_inv_norm: torch.Tensor,
        k_inv_norm: torch.Tensor,
        scale: float,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
        cu_seqlens: torch.Tensor = None,
) -> tuple:
    """Finish local fused gradients after the P2P state-gradient handoff.

    Args:
        q: Query activations in token-major layout.
        k: Key activations in token-major layout.
        q_norm: Normalized query activations.
        k_norm: Normalized key activations.
        v: Value activations in token-major layout.
        g_cumsum: Per-chunk cumulative log-decay gates.
        beta: Activated delta-rule update weights.
        A: Solved per-chunk WY representation.
        w: Prepared key/update projection for the recurrent state.
        h: Saved chunk-entry recurrent states.
        v_new: Values corrected by their incoming chunk states.
        dv: Value gradient to propagate through the next local stage.
        grad_output: Gradient of the local output.
        dh: Gradients of chunk-entry recurrent states.
        q_inv_norm: Saved query inverse L2 norms.
        k_inv_norm: Saved key inverse L2 norms.
        scale: Attention output scaling factor.
        use_qk_l2norm_in_kernel: Whether local stages normalize Q/K and differentiate that normalization.
        chunk_size: Tokens per recurrence chunk.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
    """
    dq, dk, dv, dbeta, dg = chunk_gated_delta_rule_bwd_finish(
        q=q_norm,
        k=k_norm,
        v=v,
        g=g_cumsum,
        beta=beta,
        A=A,
        w=w,
        h=h,
        v_new=v_new,
        dv=dv,
        do=grad_output,
        dh=dh,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    if use_qk_l2norm_in_kernel:
        with torch.enable_grad():
            q_leaf = q.detach().requires_grad_(True)
            k_leaf = k.detach().requires_grad_(True)
            q_recomputed, _ = _l2norm(q_leaf)
            k_recomputed, _ = _l2norm(k_leaf)
            dq, dk = torch.autograd.grad(
                (q_recomputed, k_recomputed),
                (q_leaf, k_leaf),
                grad_outputs=(dq, dk),
            )
    del q_inv_norm, k_inv_norm
    return dq, dk, dv, dg, dbeta


@torch.compiler.disable
@input_guard
def chunk_gated_delta_rule_bwd_saved(
        q: torch.Tensor,
        k: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        v: torch.Tensor,
        g_cumsum: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        initial_state: torch.Tensor,
        grad_output: torch.Tensor,
        grad_final_state: torch.Tensor,
        q_inv_norm: torch.Tensor,
        k_inv_norm: torch.Tensor,
        scale: float,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
        cu_seqlens: torch.Tensor = None,
) -> tuple:
    """Run fused GDN backward from a context saved by the forward helper.

    Args:
        q: Query activations in token-major layout.
        k: Key activations in token-major layout.
        q_norm: Normalized query activations.
        k_norm: Normalized key activations.
        v: Value activations in token-major layout.
        g_cumsum: Per-chunk cumulative log-decay gates.
        beta: Activated delta-rule update weights.
        A: Solved per-chunk WY representation.
        initial_state: Optional state entering the local sequence.
        grad_output: Gradient of the local output.
        grad_final_state: Optional gradient of the terminal recurrent state.
        q_inv_norm: Saved query inverse L2 norms.
        k_inv_norm: Saved key inverse L2 norms.
        scale: Attention output scaling factor.
        use_qk_l2norm_in_kernel: Whether local stages normalize Q/K and differentiate that normalization.
        chunk_size: Tokens per recurrence chunk.
        cu_seqlens: Optional cumulative offsets delimiting packed documents.
    """
    w, h, v_new, dv = chunk_gated_delta_rule_bwd_prepare_saved(
        q_norm,
        k_norm,
        v,
        g_cumsum,
        beta,
        A,
        initial_state,
        grad_output,
        scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_state_saved(
        q_norm,
        k_norm,
        g_cumsum,
        w,
        initial_state,
        grad_final_state,
        grad_output,
        dv,
        scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    dq, dk, dv, dg, dbeta = chunk_gated_delta_rule_bwd_finish_saved(
        q,
        k,
        q_norm,
        k_norm,
        v,
        g_cumsum,
        beta,
        A,
        w,
        h,
        v_new,
        dv,
        grad_output,
        dh,
        q_inv_norm,
        k_inv_norm,
        scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    return dq, dk, dv, dg, dbeta, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Autograd wrapper for the Triton-Ascend chunk Gated Delta Rule."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
            ctx: Any,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            scale: float,
            initial_state: torch.Tensor,
            output_final_state: bool,
            cu_seqlens: Optional[torch.LongTensor] = None,
            use_qk_l2norm_in_kernel: bool = False,
            chunk_size: int = 64,
    ) -> tuple:
        """Save the local recurrence inputs and normalization auxiliaries.

        Args:
            ctx: Autograd context owned by this invocation.
            q: Query activations in token-major layout.
            k: Key activations in token-major layout.
            v: Value activations in token-major layout.
            g: Log-decay gates, cumulative when supplied by prepare.
            beta: Activated delta-rule update weights.
            scale: Attention output scaling factor.
            initial_state: Optional state entering the local sequence.
            output_final_state: Whether to return the terminal state.
            cu_seqlens: Optional cumulative offsets delimiting packed documents.
            use_qk_l2norm_in_kernel: Whether local stages normalize Q/K and differentiate that normalization.
            chunk_size: Tokens per recurrence chunk.
        """
        g, output, matrix_a, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )

        saved_initial_state = initial_state if initial_state is not None else q.new_empty(0)
        saved_cu_seqlens = cu_seqlens if cu_seqlens is not None else q.new_empty(0, dtype=torch.long)
        ctx.save_for_backward(q, k, v, g, beta, matrix_a, saved_initial_state, saved_cu_seqlens)
        ctx.has_initial_state = initial_state is not None
        ctx.has_cu_seqlens = cu_seqlens is not None
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return output.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
            ctx: Any,
            do: torch.Tensor,
            dht: torch.Tensor
    ) -> tuple:
        """Differentiate the local recurrence and restore input normalization gradients.

        Args:
            ctx: Autograd context owned by this invocation.
            do: Gradient of the local output.
            dht: Optional gradient of the terminal recurrent state.
        """
        q, k, v, g, beta, matrix_a, initial_state, cu_seqlens = ctx.saved_tensors
        if not ctx.has_initial_state:
            initial_state = None
        if not ctx.has_cu_seqlens:
            cu_seqlens = None
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=matrix_a,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_size=ctx.chunk_size,
        )
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None


@torch.compiler.disable
def chunk_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
        head_first: bool = False,
) -> tuple:
    """Execute the local fused GDN recurrence and its paired backward.

    Args:
        q (Tensor):
            queries of shape `[B, T, H, K]`.
        k (Tensor):
            keys of shape `[B, T, H, K]`.
        v (Tensor):
            values of shape `[B, T, H, V]`.
        g (Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.
    Returns:
        o (Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype}, k current type is {k.dtype}, "
            f"v current type is {v.dtype}, they should be equal"
        )
    if q.dtype == torch.float32:
        raise ValueError(
            "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
        )
    if len(beta.shape) != 3:
        raise ValueError(
            f"beta current shape len is {len(beta.shape)}, beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
        )

    if head_first:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q, _ = _l2norm(q)
        k, _ = _l2norm(k)

    output, final_state = ChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, False, chunk_size,
    )
    return output, final_state


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
    output_transition: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return S and optionally M of ``state_out = M @ state_in + S``.

    Set output_transition=False only when the caller has no consumer for M.
    Both modes retain the same packed kernel; omitting M avoids its output
    copy and saved-state lifetime, not its arithmetic.

    Args:
        key: Prepared key with shape [B,T,H,K].
        w: WY correction with the same shape as key.
        u: Prepared value with shape [B,T,H,V].
        g: Chunk cumulative log-decay with shape [B,T,H].
        chunk_size: Number of tokens in each recurrent chunk.
        block_size: Width of each output tile.
        output_transition: Whether the caller consumes the transition M.

    Returns:
        FP32 additive state S and transition M, or None when M is omitted.
    """
    if (key.ndim, w.ndim, u.ndim, g.ndim) != (4, 4, 4, 3):
        raise ValueError("GDN state summary expects key/w/u [B,T,H,D] and g [B,T,H].")
    batch, seq_len, heads, key_dim = key.shape
    value_dim = u.shape[-1]
    _validate_fixed_summary_shape(key_dim, value_dim, chunk_size)
    if block_size not in (64, 128):
        raise ValueError(f"GDN state-summary block_size must be 64 or 128, got {block_size}.")
    if seq_len <= 0:
        raise ValueError("State summary requires a nonempty sample fragment.")
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
        NT=triton.cdiv(seq_len, chunk_size),
    )
    state_ext = packed_summary[..., :value_dim].contiguous()
    if not output_transition:
        return state_ext, None
    transition = packed_summary[..., value_dim:].contiguous()
    return state_ext, transition


def _validate_gdn_gradient_summary_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    grad_output: torch.Tensor,
    dv: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int]:
    """Validate the reverse summary inputs before launching its kernel."""
    batch, seq_len, heads, key_dim = query.shape
    value_dim = grad_output.shape[-1]
    _validate_fixed_summary_shape(key_dim, value_dim, chunk_size)
    if seq_len <= 0:
        raise ValueError("State summary requires a nonempty sample fragment.")
    qk_shape = (batch, seq_len, heads, key_dim)
    value_shape = (batch, seq_len, heads, value_dim)
    if (key.shape, w.shape, g.shape, grad_output.shape, dv.shape) != (
        qk_shape, qk_shape, qk_shape[:3], value_shape, value_shape
    ):
        raise ValueError(
            "Incompatible GDN state-gradient summary shapes: "
            f"query={tuple(query.shape)}, key={tuple(key.shape)}, "
            f"w={tuple(w.shape)}, g={tuple(g.shape)}, "
            f"grad_output={tuple(grad_output.shape)}, dv={tuple(dv.shape)}."
        )
    return batch, seq_len, heads, key_dim, value_dim


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
    """Return the local-loss contribution to the incoming state gradient.

    Args:
        query: Prepared query with shape [B,T,H,K].
        key: Prepared key with the same shape as query.
        w: WY correction with the same shape as query.
        g: Chunk cumulative log-decay with shape [B,T,H].
        grad_output: Attention output gradient with shape [B,T,H,V].
        dv: Local value-gradient contribution with the same shape as grad_output.
        scale: Attention scaling factor.
        chunk_size: Number of tokens in each recurrent chunk.

    Returns:
        FP32 incoming-state gradient with zero outgoing-state gradient.
    """
    batch, seq_len, heads, key_dim, value_dim = _validate_gdn_gradient_summary_inputs(
        query, key, w, g, grad_output, dv, chunk_size
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
        NT=triton.cdiv(seq_len, chunk_size),
    )
    return grad_state_ext
