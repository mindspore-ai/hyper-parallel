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
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Keep the validated kernel adapter close to its upstream implementation.
# pylint: disable=line-too-long,missing-public-type-hints,missing-public-docstring
# pylint: disable=non-google-docstring,disallowed-name,unused-argument,invalid-name
# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=abstract-method,arguments-differ

import warnings
from typing import Optional

import torch

from .triton.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from .triton.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from .triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .triton.cumsum import chunk_local_cumsum
from .triton.solve_tril import solve_tril
from .triton.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from .triton.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd


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
):
    """Compute forward intermediates that do not depend on the initial state."""
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=False)
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    return g, A, w, u


def chunk_gated_delta_rule_fwd_apply_state(
        k: torch.Tensor,
        g: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
):
    """Apply the recurrent initial state and return local state intermediates."""
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
):
    """Compute local outputs after recurrent states have been applied."""
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
):
    g, A, w, u = chunk_gated_delta_rule_fwd_prepare(
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
    o = chunk_gated_delta_rule_fwd_output(
        q=q,
        k=k,
        v_new=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return g, o, A, final_state


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
):
    """Compute backward intermediates that do not depend on final-state grad."""
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
):
    """Apply the final-state gradient and produce the initial-state gradient."""
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
):
    """Finish local tensor gradients after the state-gradient handoff."""
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
):
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
):
    """Prepare fused GDN forward tensors without consuming the initial state."""
    if scale is None:
        scale = k.shape[-1] ** -0.5

    q_norm, q_inv_norm = q, q.new_empty(0)
    k_norm, k_inv_norm = k, k.new_empty(0)
    if use_qk_l2norm_in_kernel:
        q_norm, q_inv_norm = _l2norm(q)
        k_norm, k_inv_norm = _l2norm(k)

    g_cumsum, A, w, u = chunk_gated_delta_rule_fwd_prepare(
        k=k_norm,
        v=v,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
    )
    return (
        q_norm,
        k_norm,
        q_inv_norm,
        k_inv_norm,
        g_cumsum,
        A,
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
):
    """Apply an initial state to fused prepared forward tensors."""
    return chunk_gated_delta_rule_fwd_apply_state(
        k=k_norm,
        g=g_cumsum,
        w=w,
        u=u,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
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
):
    """Compute fused GDN output from prepared, state-applied tensors."""
    return chunk_gated_delta_rule_fwd_output(
        q=q_norm,
        k=k_norm,
        v_new=v_new,
        h=h,
        g=g_cumsum,
        scale=scale,
        chunk_size=chunk_size,
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
):
    """Run fused GDN forward and return the tensors required by its backward."""
    (
        q_norm,
        k_norm,
        q_inv_norm,
        k_inv_norm,
        g_cumsum,
        A,
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
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_apply_state_saved(
        k_norm,
        g_cumsum,
        w,
        u,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    output = chunk_gated_delta_rule_fwd_output_saved(
        q_norm,
        k_norm,
        g_cumsum,
        h,
        v_new,
        scale,
        chunk_size=chunk_size,
    )
    return (
        output.to(q.dtype),
        final_state,
        q_norm,
        k_norm,
        q_inv_norm,
        k_inv_norm,
        g_cumsum,
        A,
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
):
    """Prepare fused backward tensors before the final-state grad arrives."""
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
):
    """Consume the final-state grad and produce the initial-state grad."""
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
):
    """Finish local fused gradients after the P2P state-gradient handoff."""
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
):
    """Run fused GDN backward from a context saved by the forward helper."""
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
    )
    return dq, dk, dv, dg, dbeta, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Autograd wrapper for the Triton-Ascend chunk Gated Delta Rule."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
            ctx,
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
    ):
        g, o, A, final_state = chunk_gated_delta_rule_fwd(
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
        ctx.save_for_backward(q, k, v, g, beta, A, saved_initial_state, saved_cu_seqlens)
        ctx.has_initial_state = initial_state is not None
        ctx.has_cu_seqlens = cu_seqlens is not None
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
            ctx,
            do: torch.Tensor,
            dht: torch.Tensor
    ):
        q, k, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
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
            A=A,
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
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
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

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        False,
        chunk_size,
    )
    return o, final_state
