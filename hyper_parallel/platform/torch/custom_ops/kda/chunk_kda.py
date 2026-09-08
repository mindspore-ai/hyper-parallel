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
"""Dense Triton-Ascend KDA execution boundaries for Torch training."""
from __future__ import annotations

from typing import Any, Optional

import torch

from hyper_parallel.platform import get_platform

from .fla_adapter import get_fla_kda_staged_ops, run_fla_chunk_kda
from .state_summary import (
    apply_kda_state_gradient_summary,
    apply_kda_state_summary,
    kda_state_gradient_summary_from_prepared,
    kda_state_summary_forward_from_prepared,
)


platform = get_platform()


def _validate_local_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    chunk_size: int,
    lower_bound: float,
) -> None:
    """Validate the fixed-shape local KDA backend before compiling kernels."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or gate.ndim != 4:
        raise ValueError("Fused KDA expects rank-4 query/key/value/gate tensors.")
    if beta.ndim != 3:
        raise ValueError("Fused KDA expects a rank-3 beta tensor.")
    if query.shape != key.shape:
        raise ValueError("Fused KDA query and key must have identical shapes.")
    batch, sequence_length, num_query_heads, key_dim = query.shape
    num_value_heads, value_dim = value.shape[2:]
    if value.shape[:2] != (batch, sequence_length):
        raise ValueError("Fused KDA value must match query batch and sequence dimensions.")
    if gate.shape != (batch, sequence_length, num_value_heads, key_dim):
        raise ValueError("Fused KDA gate has an incompatible shape.")
    if beta.shape != (batch, sequence_length, num_value_heads):
        raise ValueError("Fused KDA beta has an incompatible shape.")
    if num_value_heads % num_query_heads:
        raise ValueError("Fused KDA value heads must be divisible by query heads.")
    if not (
        query.dtype == key.dtype == value.dtype == gate.dtype == beta.dtype
        == torch.bfloat16
    ):
        raise TypeError("Fused KDA requires q/k/v/gate/beta tensors in bfloat16.")
    if a_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise TypeError("Fused KDA requires a_log and dt_bias tensors in float32.")
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "Fused KDA currently requires key_dim=value_dim=128 and chunk_size=64."
        )
    if sequence_length % chunk_size:
        raise ValueError(
            f"Fused KDA sequence length {sequence_length} must be divisible "
            f"by chunk_size {chunk_size}."
        )
    if a_log.numel() != num_value_heads:
        raise ValueError("Fused KDA a_log must contain one value per value head.")
    if dt_bias.numel() != num_value_heads * key_dim:
        raise ValueError("Fused KDA dt_bias must contain one value per gate channel.")
    if not -5.0 <= lower_bound < 0:
        raise ValueError("Fused KDA lower_bound must lie in [-5, 0).")
    if not query.is_npu:
        raise RuntimeError("The fused KDA backend requires Ascend NPU tensors.")
    if any(
        tensor.device != query.device
        for tensor in (key, value, gate, beta, a_log, dt_bias)
    ):
        raise ValueError("Fused KDA inputs must reside on the same NPU device.")
    get_fla_kda_staged_ops()


def _validate_p2p_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    chunk_size: int,
    lower_bound: float,
    cp_rank: int,
    cp_size: int,
) -> None:
    """Validate the fixed-shape KDA P2P backend before communication starts."""
    _validate_local_inputs(
        query,
        key,
        value,
        gate,
        beta,
        a_log,
        dt_bias,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    )
    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"Invalid KDA P2P rank {cp_rank} for cp_size {cp_size}.")


def fused_chunk_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float] = None,
    lower_bound: float = -5.0,
    chunk_size: int = 64,
    safe_gate: bool = True,
) -> torch.Tensor:
    """Run the local Triton-Ascend KDA backend without CP communication."""
    _validate_local_inputs(
        query,
        key,
        value,
        gate,
        beta,
        a_log,
        dt_bias,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    )
    output, _ = run_fla_chunk_kda(
        query,
        key,
        value,
        gate,
        beta,
        a_log=a_log,
        dt_bias=dt_bias.reshape(-1),
        scale=scale,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
    )
    return output


class _KDAStateP2PFunction(torch.autograd.Function):
    """Run fused local KDA around an affine recurrent-state P2P wavefront."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ,too-many-arguments,too-many-locals
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_raw: torch.Tensor,
        beta_raw: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        lower_bound: float,
        chunk_size: int,
        safe_gate: bool,
        cp_group: Any,
        prev_rank: int,
        next_rank: int,
        cp_rank: int,
        cp_size: int,
    ) -> torch.Tensor:
        """Prepare local KDA, propagate its state, and produce token outputs."""
        ops = get_fla_kda_staged_ops()

        rcp_ln2 = 1.4426950408889634

        query, key, value, gate_raw, beta_raw = (
            tensor.contiguous()
            for tensor in (query, key, value, gate_raw, beta_raw)
        )
        query, query_rstd = ops.l2norm_fwd(query)
        key, key_rstd = ops.l2norm_fwd(key)
        beta = ops.fused_beta_sigmoid(beta_raw)
        gate = ops.kda_gate_chunk_cumsum(
            g=gate_raw,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=rcp_ln2,
            chunk_size=chunk_size,
            lower_bound=lower_bound,
        )

        state_shape = (
            query.shape[0],
            value.shape[2],
            query.shape[-1],
            value.shape[-1],
        )
        initial_state = None
        recv_work = None
        if cp_rank > 0:
            initial_state = torch.empty(
                state_shape,
                device=query.device,
                dtype=torch.float32,
            )
            recv_work = platform.irecv(
                initial_state,
                src=prev_rank,
                group=cp_group,
            )

        w, u, _, kg, attention_qk, attention_kk = ops.chunk_kda_fwd_intra(
            q=query,
            k=key,
            v=value,
            gk=gate,
            beta=beta,
            scale=scale,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            disable_recompute=True,
        )
        state_ext = None
        transition = None
        if cp_rank < cp_size - 1:
            state_ext, transition = kda_state_summary_forward_from_prepared(
                kg,
                w,
                u,
                gate,
                chunk_size=chunk_size,
            )
        if recv_work is not None:
            recv_work.wait()

        send_work = None
        send_state = None
        if cp_rank < cp_size - 1:
            final_state = apply_kda_state_summary(
                state_ext,
                transition,
                initial_state,
            )
            send_state = final_state.contiguous()
            send_work = platform.isend(
                send_state,
                dst=next_rank,
                group=cp_group,
            )

        states, value_new, _ = ops.chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=gate,
            initial_state=initial_state,
            output_final_state=False,
            chunk_size=chunk_size,
        )
        output = ops.chunk_gla_fwd_o_gk(
            q=query,
            v=value_new,
            g=gate,
            A=attention_qk,
            h=states,
            scale=scale,
            chunk_size=chunk_size,
        )
        if send_work is not None:
            send_work.wait()

        saved_initial_state = initial_state
        if saved_initial_state is None:
            saved_initial_state = query.new_empty(0, dtype=torch.float32)
        ctx.save_for_backward(
            query,
            query_rstd,
            key,
            key_rstd,
            value,
            gate_raw,
            beta_raw,
            a_log,
            dt_bias,
            attention_qk,
            attention_kk,
            transition if transition is not None else query.new_empty(0),
            saved_initial_state,
        )
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.cp_group = cp_group
        ctx.prev_rank = prev_rank
        ctx.next_rank = next_rank
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        return output.to(value.dtype)

    @staticmethod
    def backward(  # pylint: disable=arguments-differ,too-many-locals
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Reverse the state wavefront, then finish the fused local backward."""
        ops = get_fla_kda_staged_ops()

        rcp_ln2 = 1.4426950408889634

        (
            query,
            query_rstd,
            key,
            key_rstd,
            value,
            gate_raw,
            beta_raw,
            a_log,
            dt_bias,
            attention_qk,
            attention_kk,
            transition,
            saved_initial_state,
        ) = ctx.saved_tensors
        initial_state = saved_initial_state if saved_initial_state.numel() else None
        grad_output = grad_output.contiguous()

        grad_final_state = None
        recv_work = None
        if ctx.cp_rank < ctx.cp_size - 1:
            grad_final_state = torch.empty(
                (
                    query.shape[0],
                    value.shape[2],
                    query.shape[-1],
                    value.shape[-1],
                ),
                device=query.device,
                dtype=torch.float32,
            )
            recv_work = platform.irecv(
                grad_final_state,
                src=ctx.next_rank,
                group=ctx.cp_group,
            )

        beta = ops.fused_beta_sigmoid(beta_raw)
        gate = ops.kda_gate_chunk_cumsum(
            g=gate_raw,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=rcp_ln2,
            chunk_size=ctx.chunk_size,
            lower_bound=ctx.lower_bound,
        )
        w, u, query_gated, key_gated = ops.recompute_w_u_fwd(
            q=query,
            k=key,
            v=value,
            beta=beta,
            A=attention_kk,
            gk=gate,
        )
        states, value_new, _ = ops.chunk_gated_delta_rule_fwd_h(
            k=key_gated,
            w=w,
            u=u,
            gk=gate,
            initial_state=initial_state,
            output_final_state=False,
            chunk_size=ctx.chunk_size,
        )
        grad_attention_qk, grad_value_local = ops.chunk_kda_bwd_dav(
            q=query,
            k=key,
            v=value_new,
            do=grad_output,
            A=attention_qk,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
        )
        grad_state_ext = None
        if ctx.cp_rank > 0:
            grad_state_ext = kda_state_gradient_summary_from_prepared(
                query_gated,
                key_gated,
                w,
                gate,
                grad_output,
                grad_value_local,
                ctx.scale,
                chunk_size=ctx.chunk_size,
            )
        if recv_work is not None:
            recv_work.wait()
        send_work = None
        send_state_gradient = None
        if ctx.cp_rank > 0:
            grad_initial_state = apply_kda_state_gradient_summary(
                grad_state_ext,
                transition,
                grad_final_state,
            )
            send_state_gradient = grad_initial_state.contiguous()
            send_work = platform.isend(
                send_state_gradient,
                dst=ctx.prev_rank,
                group=ctx.cp_group,
            )

        grad_states, _, grad_value = ops.chunk_gated_delta_rule_bwd_dhu(
            q=query_gated,
            k=key_gated,
            w=w,
            gk=gate,
            h0=initial_state,
            dht=grad_final_state,
            do=grad_output,
            dv=grad_value_local,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
        )
        (
            grad_query,
            grad_key,
            grad_value,
            grad_beta,
            grad_gate,
            grad_attention_kk,
        ) = ops.chunk_kda_bwd_wy_dqkg_fused(
            q=query,
            k=key,
            v=value,
            v_new=value_new,
            g=gate,
            beta=beta,
            A=attention_kk,
            h=states,
            do=grad_output,
            dh=grad_states,
            dv=grad_value,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
        )
        grad_query, grad_key, grad_beta, grad_gate = ops.chunk_kda_bwd_intra(
            q=query,
            k=key,
            g=gate,
            beta=beta,
            dAqk=grad_attention_qk,
            dAkk=grad_attention_kk,
            dq=grad_query,
            dk=grad_key,
            db=grad_beta,
            dg=grad_gate,
            chunk_size=ctx.chunk_size,
            safe_gate=ctx.safe_gate,
        )

        num_query_heads = query.shape[2]
        num_value_heads = value.shape[2]
        if num_value_heads > num_query_heads:
            groups = num_value_heads // num_query_heads
            grad_query = grad_query.view(
                *grad_query.shape[:2],
                num_query_heads,
                groups,
                grad_query.shape[-1],
            ).sum(dim=3)
            grad_key = grad_key.view(
                *grad_key.shape[:2],
                num_query_heads,
                groups,
                grad_key.shape[-1],
            ).sum(dim=3)

        grad_gate = ops.chunk_local_cumsum(
            grad_gate,
            chunk_size=ctx.chunk_size,
            reverse=True,
        )
        grad_gate, grad_a_log, grad_dt_bias = ops.kda_gate_bwd(
            g=gate_raw,
            A_log=a_log,
            dt_bias=dt_bias,
            dyg=grad_gate,
            lower_bound=ctx.lower_bound,
        )
        grad_beta = ops.fused_beta_sigmoid_bwd(beta_raw, grad_beta)
        grad_query = ops.l2norm_bwd(query, query_rstd, grad_query)
        grad_key = ops.l2norm_bwd(key, key_rstd, grad_key)
        if send_work is not None:
            send_work.wait()

        return (
            grad_query,
            grad_key,
            grad_value,
            grad_gate,
            grad_beta,
            grad_a_log,
            grad_dt_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def fused_chunk_kda_p2p(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cp_group: Any,
    prev_rank: int,
    next_rank: int,
    cp_rank: int,
    cp_size: int,
    scale: Optional[float] = None,
    lower_bound: float = -5.0,
    chunk_size: int = 64,
    safe_gate: bool = True,
) -> torch.Tensor:
    """Run one sequence-sharded KDA segment with state P2P."""
    _validate_p2p_inputs(
        query,
        key,
        value,
        gate,
        beta,
        a_log,
        dt_bias,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )
    effective_scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    return _KDAStateP2PFunction.apply(
        query,
        key,
        value,
        gate,
        beta,
        a_log,
        dt_bias.reshape(-1),
        effective_scale,
        float(lower_bound),
        chunk_size,
        safe_gate,
        cp_group,
        prev_rank,
        next_rank,
        cp_rank,
        cp_size,
    )


__all__ = ["fused_chunk_kda", "fused_chunk_kda_p2p"]
