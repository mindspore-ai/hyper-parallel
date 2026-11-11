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
"""Explicit local Chunk KDA stages for caller-managed state-scan CP.

These experimental primitives provide no implicit automatic differentiation or
communication. Use ``npu_chunk_kda`` for ordinary differentiable training.
Layouts, optional outputs and supported flags follow FLA-NPU's stage ACLNN ABI.
Metadata is host integer data; tensors must be local, not DTensors.
"""
from typing import Optional, Sequence, Tuple, Union

from hyper_parallel.platform import get_platform

_platform = get_platform()
Tensor = _platform.Tensor
_Metadata = Optional[Union[Sequence[int], Tensor]]
_StageOutputs = Tuple[Optional[Tensor], ...]


def npu_chunk_kda_fwd_prepare(
        q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, scale: float,
        chunk_size: int = 64, *, layout: str = "BNSD",
        a_log: Optional[Tensor] = None, dt_bias: Optional[Tensor] = None,
        cu_seqlens: _Metadata = None, chunk_indices: _Metadata = None,
        epsilon: float = 1e-6, use_qk_l2norm_in_kernel: bool = False, use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False, allow_neg_eigval: bool = False,
        safe_gate: bool = False, lower_bound: float = -5.0, use_exp2: bool = True,
        backward_mode: str = "save") -> _StageOutputs:
    """Return 13 head-major caches from local inputs in ``layout``.

    Order: ``gk, Aqk, Akk, w, u, qg, kg, qg_scaled, q_hat, k_hat,
    q_rstd, k_rstd, beta_eff``. ``gk`` and the last three caches are FP32;
    other caches have q's dtype. Value-head caches use HV, norm caches use HK.
    ``backward_mode`` selects output slots, independently of automatic
    differentiation: ``none`` selects 0,1,3,4,6,7; ``forward`` adds 2;
    ``recompute`` adds 8..12; ``save`` selects all. Other slots are None.
    All four input layouts BNSD/BSND/NTD/TND are passed to ACLNN unchanged.
    Norm caches hold identity values when normalization is disabled; beta_eff
    is the effective FP32 beta, required by sigmoid backward.
    """
    return _platform.custom_ops.npu_chunk_kda_fwd_prepare(
        q, k, v, g, beta, scale, chunk_size, layout, a_log, dt_bias,
        cu_seqlens, chunk_indices, epsilon, use_qk_l2norm_in_kernel,
        use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
        safe_gate, lower_bound, use_exp2, backward_mode,
    )


def npu_chunk_kda_pre_process_fwd(
        k: Tensor, w: Tensor, u: Tensor, *, gk: Tensor, cu_seqlens: _Metadata,
        chunk_size: int = 64) -> Tuple[Tensor]:
    """Return ``(summary,)`` with FP32 shape ``[N,HV,K,V+K]``.

    Inputs are rank-4 BNSD caches (KDA uses gated ``kg`` as k), B=1.
    ``cu_seqlens`` selects one or more packed subintervals, including a
    nonzero starting boundary. The summary stores ``[E | M]``. The caller
    exchanges summaries to construct the next stage's initial_state.
    """
    return _platform.custom_ops.npu_chunk_kda_pre_process_fwd(
        k, w, u, gk, cu_seqlens, chunk_size,
    )


def npu_chunk_kda_fwd_h(
        k: Tensor, w: Tensor, u: Tensor, *, gk: Tensor, initial_state: Optional[Tensor] = None,
        output_final_state: bool = False, chunk_size: int = 64, cu_seqlens: _Metadata = None,
        chunk_indices: _Metadata = None, use_exp2: bool = True,
        state_v_first: bool = False) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Return ``(h, v_new, final_state)`` for rank-4 BNSD caches.

    h has shape [B,Nc,HV,K,V] and input dtype; v_new matches u.
    With state_v_first, h and boundary states store V,K instead of K,V.
    initial_state is [N,HV,K,V], BF16/FP32 as allowed by FLA; final_state
    matches its dtype, or is FP32 without initial_state, and is None unless
    requested. N is batch size or number of packed sequences. Nc counts
    per-sequence chunks, including tails. No state exchange happens here.
    """
    return _platform.custom_ops.npu_chunk_kda_fwd_h(
        k, w, u, gk, initial_state, output_final_state, chunk_size,
        cu_seqlens, chunk_indices, use_exp2, state_v_first,
    )


def npu_chunk_kda_fwd_finalize(
        qg_scaled: Tensor, aqk: Tensor, v_new: Tensor, h: Tensor, *, output_layout: str = "BSND",
        state_v_first: bool = False, cu_seqlens: _Metadata = None, chunk_indices: _Metadata = None) -> Tensor:
    """Return attention output from head-major caches and chunk-major h.

    Output dtype matches qg_scaled. output_layout selects BNSD/BSND for
    rank-4 caches or NTD/TND for rank-3 caches; no input layout conversion
    is performed. chunk width is inferred by FLA from aqk.
    """
    return _platform.custom_ops.npu_chunk_kda_fwd_finalize(
        qg_scaled, aqk, v_new, h, output_layout, state_v_first,
        cu_seqlens, chunk_indices,
    )


def npu_chunk_kda_bwd_prepare(
        aqk: Tensor, v_new: Tensor, d_o: Tensor, h: Tensor, *, scale: float, layout: str = "BNSD",
        chunk_size: int = 64, state_v_first: bool = False, cu_seqlens: _Metadata = None,
        chunk_indices: _Metadata = None) -> Tuple[Tensor, Tensor, Tensor]:
    """Return head-major ``(dAqk, dv_local, dq_raw)``.

    aqk, v_new and d_o all use the specified layout; h is chunk-major.
    dAqk/dq_raw are FP32 and have last dimension chunk_size/K;
    dv_local matches the value dimension and dtype. This is an explicit
    gradient stage, not a request to the automatic differentiation engine.
    """
    return _platform.custom_ops.npu_chunk_kda_bwd_prepare(
        aqk, v_new, d_o, h, scale, layout, chunk_size,
        state_v_first, cu_seqlens, chunk_indices,
    )


def npu_chunk_kda_pre_process_bwd(
        q: Tensor, k: Tensor, w: Tensor, d_o: Tensor, dv: Tensor, scale: float,
        chunk_size: int = 64, *, gk: Tensor, cu_seqlens: _Metadata = None) -> Tensor:
    """Return backward FP32 summary ``[HV,K,V+K]`` for one segment.

    Rank-4 inputs use BNSD; KDA supplies qg/kg as q/k and dv_local as dv.
    cu_seqlens, when supplied, selects the single [begin,end) segment.
    The caller performs a reverse state scan to construct dht.
    """
    return _platform.custom_ops.npu_chunk_kda_pre_process_bwd(
        q, k, w, d_o, dv, scale, chunk_size, gk, cu_seqlens,
    )


def npu_chunk_kda_bwd_dhu(
        q: Tensor, k: Tensor, w: Tensor, d_o: Tensor, dv: Tensor, scale: float,
        chunk_size: int = 64, *, gk: Tensor, h0: Optional[Tensor] = None,
        dht: Optional[Tensor] = None, cu_seqlens: _Metadata = None, chunk_indices: _Metadata = None,
        use_exp2: bool = True, state_v_first: bool = False) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """Return ``(dh, dh0, dv_scan)`` using optional boundary gradients.

    Rank-4 q/k/w/d_o/dv use BNSD; KDA passes qg/kg as q/k. dh always
    stores [B,Nc,HV,K,V], regardless of state_v_first. h0/dht/dh0 use
    [N,HV,K,V] or [N,HV,V,K], with q's dtype. dh0 is None unless h0 is
    supplied; dv_scan matches dv. dht is the final-state upstream gradient.
    """
    return _platform.custom_ops.npu_chunk_kda_bwd_dhu(
        q, k, w, d_o, dv, scale, chunk_size, gk, h0, dht,
        cu_seqlens, chunk_indices, use_exp2, state_v_first,
    )


def npu_chunk_kda_bwd_finalize(
        q: Tensor, k: Tensor, v: Tensor, gk: Tensor, raw_g: Tensor, beta: Tensor,
        a_log: Tensor, dt_bias: Tensor, akk: Tensor, v_new: Tensor, h: Tensor, dh: Tensor,
        dv_scan: Tensor, d_aqk: Tensor, dq_raw: Tensor, *, scale: float, lower_bound: float = -5.0,
        chunk_size: int = 64, safe_gate: bool = True, use_gate_in_kernel: bool = True, use_exp2: bool = True,
        state_v_first: bool = False, q_rstd: Optional[Tensor] = None, k_rstd: Optional[Tensor] = None,
        cu_seqlens: _Metadata = None, chunk_indices: _Metadata = None,
        use_beta_sigmoid_in_kernel: bool = False, allow_neg_eigval: bool = False,
        output_layout: str = "BNSD") -> _StageOutputs:
    """Return ``(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)``.

    Token inputs are head-major; output_layout selects all five token
    gradient layouts. q/k are q_hat/k_hat when in-kernel L2 norm was used;
    pass the matching FP32 rstd caches. With sigmoid beta, pass saved FP32
    beta_eff, not raw logits. dq/dk/dv match input dtypes, dbeta/dg match
    beta/raw_g dtypes; parameter gradients are FP32. The paired FLA ABI
    requires a_log and dt_bias, safe_gate/use_gate_in_kernel/use_exp2=True,
    chunk_size=64 and K-first states for this stage. FLA validates these
    limits, supported SoCs and any GQA limitations; other stage APIs have
    their own capability ranges.
    """
    return _platform.custom_ops.npu_chunk_kda_bwd_finalize(
        q, k, v, gk, raw_g, beta, a_log, dt_bias, akk, v_new, h, dh,
        dv_scan, d_aqk, dq_raw, scale, lower_bound, chunk_size,
        safe_gate, use_gate_in_kernel, use_exp2, state_v_first, q_rstd,
        k_rstd, cu_seqlens, chunk_indices, use_beta_sigmoid_in_kernel,
        allow_neg_eigval, output_layout,
    )


def npu_chunk_kda_merge_fwd_bwd(
        state: Tensor, summaries: Tensor, pre_or_post_num_ranks: int, rank: int, *, forward: bool = True,
        state_v_first: bool = False) -> Tensor:
    """Update and return state in place using already gathered summaries.

    state is contiguous FP32/BF16 [HV,K,V] (or [HV,V,K]); summaries has the
    same dtype and shape [S,HV,K,V+K]. Count, rank and order follow FLA's MergeFwdBwdKernel
    contract. The caller owns the state buffer and communication. Passing
    a noncontiguous state raises rather than silently updating a copy.
    """
    return _platform.custom_ops.npu_chunk_kda_merge_fwd_bwd(
        state, summaries, pre_or_post_num_ranks, rank, forward, state_v_first,
    )
