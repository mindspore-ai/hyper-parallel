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
"""Optional fla-npu Ascend C local KDA backend for dense and packed Ulysses training."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, NamedTuple

import torch
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=1)
def _load_ops():
    """Keep the optional native runtime out of ordinary framework imports."""
    try:
        ops = importlib.import_module("fla_npu.ops.ascendc")
    except ImportError as error:
        raise RuntimeError("backend='ascendc' requires a compatible fla-npu installation.") from error
    required = {
        "chunk_kda_fwd_prepare": ("backward_mode", "use_qk_l2norm_in_kernel", "use_beta_sigmoid_in_kernel"),
        "chunk_fwd_h": ("output_final_state", "cu_seqlens", "chunk_indices"),
        "chunk_kda_fwd_finalize": ("output_layout", "cu_seqlens", "chunk_indices"),
        "chunk_kda_bwd": ("q_rstd", "k_rstd", "implementation", "disable_recompute",
                          "layout", "beta_eff", "use_beta_sigmoid_in_kernel"),
    }
    for name, parameters in required.items():
        operation = getattr(ops, name, None)
        try:
            signature = inspect.signature(operation)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"fla-npu lacks a compatible public {name} API.") from error
        missing = set(parameters).difference(signature.parameters)
        if missing:
            raise RuntimeError(
                f"fla-npu {name} lacks required parameters {sorted(missing)}; "
                "install the paired native-norm KDA build."
            )
    return ops


@lru_cache(maxsize=16)
def _use_native_beta_backward(device):
    """Select the implemented sigmoid epilogue, without rejecting other devices.

    The public native API reserves fused beta backward on the A5 V2 path.
    Native operators remain responsible for validating device support.
    """
    if device.type != "npu":
        raise ValueError("Ascend C KDA requires NPU tensors before querying native capabilities.")
    return str(torch.npu.get_device_name(device)).startswith(("Ascend910B", "Ascend910C", "Ascend910_93"))


class _KDAPrepared(NamedTuple):
    """Public Prepare save-mode slots, in the native ABI's fixed order."""

    gk: torch.Tensor              # FP32 chunk-local cumulative log2 decay.
    aqk: torch.Tensor             # Intra-chunk QK coefficients.
    akk: torch.Tensor             # Triangular KK coefficients for backward.
    w: torch.Tensor               # WY state-feedback coefficients.
    u: torch.Tensor | None        # Forward-only residual input to FwdH.
    qg: torch.Tensor              # Gated queries consumed by backward.
    kg: torch.Tensor              # Gated keys consumed by FwdH and backward.
    qg_scaled: torch.Tensor | None  # Forward-only queries including attention scale.
    q_hat: torch.Tensor           # Normalized head-major queries.
    k_hat: torch.Tensor           # Normalized head-major keys.
    q_rstd: torch.Tensor          # FP32 Q normalization auxiliary.
    k_rstd: torch.Tensor          # FP32 K normalization auxiliary.
    beta_eff: torch.Tensor        # FP32 head-major sigmoid(beta) cache.


def _forward_kda(inputs, scale, lower_bound, metadata):
    """Run native stages, returning only the intermediates needed by autograd."""
    ops = _load_ops()
    query, key, value, gate, beta, a_log, dt_bias = inputs
    result = ops.chunk_kda_fwd_prepare(
        query, key, value, gate.transpose(1, 2).contiguous(), beta.transpose(1, 2).contiguous(),
        scale, layout="BNSD", chunk_size=64,
        safe_gate=True, lower_bound=lower_bound, use_gate_in_kernel=True,
        A_log=a_log, dt_bias=dt_bias.flatten(), epsilon=1e-6,
        use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        use_exp2=True, backward_mode="save", **metadata,
    )
    if len(result) != 13 or any(not isinstance(tensor, torch.Tensor) for tensor in result):
        raise RuntimeError("fla-npu Prepare must return all 13 tensors for native-norm save mode.")
    prepared = _KDAPrepared(*result)
    hidden, v_new, _ = ops.chunk_fwd_h(
        prepared.kg, prepared.w, prepared.u, gk=prepared.gk,
        output_final_state=False, use_exp2=True, **metadata,
    )
    output = ops.chunk_kda_fwd_finalize(
        prepared.qg_scaled, prepared.aqk, v_new, hidden, output_layout="BSND", **metadata)
    # These buffers have no backward consumer and must not live across layers.
    return output, prepared._replace(u=None, qg_scaled=None), hidden, v_new


@lru_cache(maxsize=1)
def _load_beta_backward():
    """Load optional FLA helpers without requiring its staged KDA backend."""
    try:
        importlib.import_module("triton.backends.ascend")
        sigmoid = importlib.import_module("fla.ops.common.gate").fused_beta_sigmoid_bwd
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "backend='ascendc' requires FLA with the Triton-Ascend beta-sigmoid backward helper."
        ) from error
    if not callable(sigmoid):
        raise RuntimeError("The installed FLA beta-sigmoid backward API is incompatible.")
    return sigmoid


class _AscendCKDA(torch.autograd.Function):
    """Own all native activations per invocation, including checkpoint replay."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                gate: torch.Tensor, beta: torch.Tensor, a_log: torch.Tensor,
                dt_bias: torch.Tensor, scale: float, lower_bound: float, metadata: dict) -> torch.Tensor:
        """Save native intermediates without retaining forward-only workspaces."""
        ctx.native_beta_backward = _use_native_beta_backward(query.device)
        if not ctx.native_beta_backward:
            _load_beta_backward()
        inputs = (query, key, value, gate, beta, a_log, dt_bias)
        output, prepared, hidden, v_new = _forward_kda(inputs, scale, lower_bound, metadata)
        # Native norm backward consumes normalized Q/K and rstd, not raw Q/K.
        # Raw gate/beta stay token-major, as required by the backward API.
        ctx.save_for_backward(value, gate, beta, a_log, dt_bias, *prepared, hidden, v_new)
        ctx.dtypes = tuple(tensor.dtype for tensor in inputs)
        ctx.scale, ctx.lower_bound, ctx.native_metadata = scale, lower_bound, metadata
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Apply native local backward and the explicit raw-input chain rule."""
        value, gate, beta_raw, a_log, dt_bias, *saved = ctx.saved_tensors
        (gk, aqk, akk, weights, _, qg, kg, _, query, key,
         query_rstd, key_rstd, beta, hidden, v_new) = saved
        packed = bool(ctx.native_metadata)
        # Native whole backward uses NTD/chunk-major views for varlen,
        # while Prepare/FwdH keep the B=1 BNSD views of the same storage.
        def backward_view(tensor: torch.Tensor) -> torch.Tensor:
            """Expose the rank required by the native packed backward ABI."""
            return tensor.squeeze(0) if packed else tensor

        native_beta = beta_raw if ctx.native_beta_backward else beta.transpose(1, 2).contiguous()
        operands = (query, key, value, native_beta, gk, aqk, akk, weights, qg, kg, v_new, hidden,
                    grad_output.contiguous())
        dq, dk, dv, dbeta, dgate, _, da_log, ddt_bias = _load_ops().chunk_kda_bwd(
            *(backward_view(tensor) for tensor in operands), ctx.scale,
            raw_g=backward_view(gate), A_log=a_log, dt_bias=dt_bias,
            initial_state=None, dht=None, chunk_size=64,
            safe_gate=True, lower_bound=ctx.lower_bound, use_gate_in_kernel=True,
            disable_recompute=True, use_exp2=True, state_v_first=False,
            q_rstd=backward_view(query_rstd), k_rstd=backward_view(key_rstd),
            layout="NTD" if packed else "BNSD", implementation="auto",
            use_beta_sigmoid_in_kernel=ctx.native_beta_backward,
            beta_eff=backward_view(beta) if ctx.native_beta_backward else None, **ctx.native_metadata,
        )
        if packed:
            dq, dk, dv, dbeta, dgate = (tensor.unsqueeze(0) for tensor in (dq, dk, dv, dbeta, dgate))
        # Native norm returns raw Q/K gradients; fused sigmoid already returns
        # raw beta gradients. Apply the external sigmoid chain rule only once.
        if not ctx.native_beta_backward:
            dbeta = _load_beta_backward()(beta_raw, dbeta)

        gradients = (dq, dk, dv, dgate, dbeta, da_log, ddt_bias.reshape_as(dt_bias))
        return (*(grad.to(dtype) for grad, dtype in zip(gradients, ctx.dtypes)), None, None, None)


def _packed_offsets(cu_seqlens, tokens):
    """Snapshot host offsets before deriving native chunk metadata."""
    if isinstance(cu_seqlens, torch.Tensor):
        if (cu_seqlens.device.type != "cpu" or cu_seqlens.ndim != 1
                or cu_seqlens.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Ascend C cu_seqlens must be a one-dimensional CPU integer tensor or sequence.")
        cu_seqlens = cu_seqlens.tolist()
    offsets = tuple(cu_seqlens)
    if (len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != tokens
            or any(not isinstance(x, int) or isinstance(x, bool) for x in offsets)
            or any(begin >= end for begin, end in zip(offsets, offsets[1:]))):
        raise ValueError("Ascend C cu_seqlens must strictly increase from zero to the packed token count.")
    return offsets


def _packed_metadata(cu_seqlens, tokens):
    """Snapshot validated host offsets and derive the canonical chunk order."""
    if cu_seqlens is None:
        return {}
    offsets = _packed_offsets(cu_seqlens, tokens)
    indices = tuple(value for seq, (begin, end) in enumerate(zip(offsets, offsets[1:]))
                    for chunk in range((end - begin + 63) // 64) for value in (seq, chunk))
    return {"cu_seqlens": offsets, "chunk_indices": indices}


def ascendc_chunk_kda(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    gate: torch.Tensor, beta: torch.Tensor, *, a_log: torch.Tensor,
    dt_bias: torch.Tensor, scale: float | None = None, lower_bound: float = -5.0,
    chunk_size: int = 64, cu_seqlens: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Run dense or packed KDA with native attention and paired raw-input gradients.

    Args:
        query: Raw BF16 queries [B,T,H,128] or packed [T,H,128].
        key: Raw BF16 keys with the same shape.
        value: BF16 values with the same shape.
        gate: BF16 or FP32 gate logits with the same shape.
        beta: BF16 or FP32 beta logits [B,T,H] or packed [T,H].
        a_log: FP32 decay parameters with H elements.
        dt_bias: FP32 gate bias with H*128 elements.
        scale: Positive attention scale, default 1/sqrt(128).
        lower_bound: Safe gate lower bound in [-5,0).
        chunk_size: Must be 64 for this native kernel family.
        cu_seqlens: Strictly increasing host offsets for packed inputs, including
            zero and T. Packed rank-four inputs require B=1. Documents need not
            be multiples of 64; zero-length documents are rejected.

    Returns:
        BF16 output with the query shape. External recurrent states and double
        backward are not exposed. Native Q/K norm backward uses save mode;
        native capability checks select the supported device implementation.
    """
    if query.ndim == 3:
        if cu_seqlens is None:
            raise ValueError("Packed TND inputs require cu_seqlens.")
        return ascendc_chunk_kda(
            *(tensor.unsqueeze(0) for tensor in (query, key, value, gate, beta)),
            a_log=a_log, dt_bias=dt_bias, scale=scale, lower_bound=lower_bound,
            chunk_size=chunk_size, cu_seqlens=cu_seqlens).squeeze(0)
    if query.ndim != 4:
        raise ValueError("KDA inputs must have shape [B,T,H,K] or packed [T,H,K].")
    if chunk_size != 64:
        raise ValueError("Ascend C KDA uses chunk_size=64.")
    metadata = _packed_metadata(cu_seqlens, query.shape[1])
    if metadata and query.shape[0] != 1:
        raise ValueError("Packed Ascend C KDA requires B=1.")
    scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    tokens = tuple(tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
    # Gate/beta stay token-major for backward. Native operators validate their
    # dtype, shape and device; only adapter-owned layout/metadata checks live here.
    # Clone sliced parameters to give native tiling this shard's storage descriptor.
    return _AscendCKDA.apply(*tokens, gate.contiguous(), beta.contiguous(), a_log.reshape(-1).clone(),
                            dt_bias.reshape(-1, query.shape[-1]).clone(), scale, lower_bound, metadata)
