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
"""Optional native local KDA phases and autograd for dense/packed inputs."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, NamedTuple

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.components.modules.attention.linear_attention.sequence_layout import local_chunk_metadata


@lru_cache(maxsize=1)
def load_kda_native_ops() -> Any:
    """Keep the optional native runtime out of ordinary framework imports."""
    try:
        ops = importlib.import_module("fla_npu.ops.ascendc")
    except ImportError as error:
        raise RuntimeError("backend='ascendc' requires a compatible fla-npu installation.") from error
    required = {
        "chunk_kda_fwd_prepare": ("backward_mode", "use_qk_l2norm_in_kernel", "use_beta_sigmoid_in_kernel"),
        "chunk_fwd_h": ("output_final_state", "cu_seqlens", "chunk_indices"),
        "chunk_kda_fwd_finalize": ("output_layout", "cu_seqlens", "chunk_indices"),
        "chunk_kda_bwd_prepare": ("scale", "layout", "cu_seqlens", "chunk_indices"),
        "chunk_gated_delta_rule_bwd_dhu": ("gK", "dht", "cu_seqlens", "chunk_indices"),
        "chunk_kda_bwd_finalize": ("q_rstd", "k_rstd", "use_beta_sigmoid_in_kernel", "output_layout"),
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
                "install a fla-npu version providing these public KDA parameters."
            )
    return ops


class KDANativePrepared(NamedTuple):
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


def prepare_kda_native(inputs: tuple[torch.Tensor, ...], scale: float, lower_bound: float,
                       metadata: dict) -> KDANativePrepared:
    """Prepare once for native summaries and the local state recurrence.

    Args:
        inputs: Head-major Q/K/V, token-major raw gate/beta and FP32 A_log/dt_bias.
        scale: Attention output scale.
        lower_bound: Safe gate lower bound.
        metadata: Local cu_seqlens and chunk_indices, or empty for dense input.

    Returns:
        Native save-mode tensors in the public Prepare ABI order.
    """
    ops = load_kda_native_ops()
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
    return KDANativePrepared(*result)


def _native_backward_view(tensor: torch.Tensor, packed: bool) -> torch.Tensor:
    """Expose packed NTD views without copying the B=1 Prepare/FwdH storage."""
    return tensor.squeeze(0) if packed else tensor


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
    metadata = local_chunk_metadata(cu_seqlens, query.shape[1])
    if metadata and query.shape[0] != 1:
        raise ValueError("Packed Ascend C KDA requires B=1.")
    scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    tokens = tuple(tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
    # Gate/beta stay token-major for backward. Native operators validate their
    # dtype, shape and device; only adapter-owned layout/metadata checks live here.
    # Clone sliced parameters to give native tiling this shard's storage descriptor.
    return _AscendCKDAStages.apply(
        *tokens, gate.contiguous(), beta.contiguous(), a_log.reshape(-1).clone(),
        dt_bias.reshape(-1, query.shape[-1]).clone(), scale, lower_bound, metadata)


def finish_kda_native_forward(ctx: Any, inputs: tuple[torch.Tensor, ...], prepared: KDANativePrepared,
                              initial: torch.Tensor | None, scale: float, lower_bound: float,
                              metadata: dict) -> torch.Tensor:
    """Apply the incoming state, finalize outputs and retain native save-mode tensors.

    Args:
        ctx: Invocation-owned autograd context receiving saved tensors.
        inputs: Raw tensors previously passed to prepare_kda_native.
        prepared: Native Prepare results for these inputs.
        initial: Incoming recurrent state, or None for zero entry.
        scale: Attention output scale.
        lower_bound: Safe gate lower bound retained for backward.
        metadata: Local packed indices, or an empty dense descriptor.
    """
    ops = load_kda_native_ops()
    _, _, value, gate, _, a_log, dt_bias = inputs
    hidden, v_new, _ = ops.chunk_fwd_h(
        prepared.kg, prepared.w, prepared.u, gk=prepared.gk, initial_state=initial,
        output_final_state=False, use_exp2=True, **metadata)
    output = ops.chunk_kda_fwd_finalize(
        prepared.qg_scaled, prepared.aqk, v_new, hidden, output_layout="BSND", **metadata)
    prepared = prepared._replace(u=None, qg_scaled=None)
    # Entry states are retained for the native local backward.
    ctx.save_for_backward(value, gate, a_log, dt_bias, *prepared, hidden, v_new)
    ctx.dtypes = tuple(tensor.dtype for tensor in inputs)
    ctx.scale, ctx.lower_bound, ctx.native_metadata = scale, lower_bound, metadata
    return output


def prepare_kda_native_backward(ctx: Any, saved_tensors: tuple, grad_output: torch.Tensor) -> tuple:
    """Prepare local gradients independently of the terminal state gradient.

    Args:
        ctx: Forward context containing scale and local document metadata.
        saved_tensors: The single unpack of ctx.saved_tensors, shared with finalize.
        grad_output: Token-major output gradient.

    Returns:
        Head-major dO and the native dAqk, local dV and raw dQ intermediates.
    """
    ops = load_kda_native_ops()
    prepared = KDANativePrepared(*saved_tensors[4:-2])
    hidden, v_new = saved_tensors[-2:]
    packed = bool(ctx.native_metadata)

    grad = grad_output.transpose(1, 2).contiguous()
    d_aqk, dv_local, dq_raw = ops.chunk_kda_bwd_prepare(
        *(_native_backward_view(tensor, packed) for tensor in (prepared.aqk, v_new, grad, hidden)), scale=ctx.scale,
        layout="NTD" if packed else "BNSD", chunk_size=64, **ctx.native_metadata)
    dv_local_head = dv_local.unsqueeze(0) if packed else dv_local
    return grad, d_aqk, dv_local_head, dq_raw


def finish_kda_native_backward(ctx: Any, saved_tensors: tuple, grad: torch.Tensor, d_aqk: torch.Tensor,
                               dv_local_head: torch.Tensor, dq_raw: torch.Tensor,
                               terminal: torch.Tensor | None) -> tuple:
    """Apply dHT and finish native Q/K/V, raw gate and sigmoid gradients.

    Args:
        ctx: Forward context containing metadata, scale and original input dtypes.
        saved_tensors: The same saved tuple used by backward prepare.
        grad: Head-major output gradient.
        d_aqk: Native intra-chunk coefficient gradient.
        dv_local_head: Head-major local value gradient before state feedback.
        dq_raw: Native raw query-gradient intermediate.
        terminal: Terminal recurrent-state gradient, or None for zero exit.

    Returns:
        Seven input gradients, cast to their original dtypes.
    """
    ops = load_kda_native_ops()
    value, gate, a_log, dt_bias, *saved = saved_tensors
    (gk, _, akk, weights, _, qg, kg, _, query, key,
     query_rstd, key_rstd, beta, hidden, v_new) = saved
    packed = bool(ctx.native_metadata)

    hidden_grad, _, dv_scan = ops.chunk_gated_delta_rule_bwd_dhu(
        qg, kg, weights, grad, dv_local_head, ctx.scale, 64,
        gK=gk, h0=None, dht=terminal, use_exp2=True, **ctx.native_metadata)
    dq, dk, dv, dbeta, dgate, da_log, ddt_bias = ops.chunk_kda_bwd_finalize(
        *(_native_backward_view(tensor, packed)
          for tensor in (query, key, value, gk, gate.transpose(1, 2).contiguous(), beta)),
        a_log, dt_bias,
        *(_native_backward_view(tensor, packed) for tensor in (akk, v_new, hidden, hidden_grad, dv_scan)),
        d_aqk, dq_raw,
        scale=ctx.scale, lower_bound=ctx.lower_bound,
        q_rstd=_native_backward_view(query_rstd, packed),
        k_rstd=_native_backward_view(key_rstd, packed),
        use_beta_sigmoid_in_kernel=False,
        output_layout="NTD" if packed else "BNSD", **ctx.native_metadata)
    if packed:
        dq, dk, dv, dbeta, dgate = (tensor.unsqueeze(0) for tensor in (dq, dk, dv, dbeta, dgate))
    # Prepare already saves FP32 sigmoid(beta); reuse it without device-specific
    # native epilogue selection, recomputing sigmoid, or retaining raw beta.
    dbeta = torch.ops.aten.sigmoid_backward.default(dbeta, beta)
    dbeta, dgate = dbeta.transpose(1, 2), dgate.transpose(1, 2)
    gradients = (dq, dk, dv, dgate, dbeta, da_log, ddt_bias.reshape_as(dt_bias))
    return tuple(gradient.to(dtype) for gradient, dtype in zip(gradients, ctx.dtypes))


class _AscendCKDAStages(torch.autograd.Function):
    """Native local KDA autograd, without communication resources or protocol callbacks."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                gate: torch.Tensor, beta: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
                scale: float, lower_bound: float, metadata: dict) -> torch.Tensor:
        """Prepare and execute a local sequence with a zero entry state.

        Args:
            ctx: Autograd context owned by this invocation.
            query: Head-major raw queries.
            key: Head-major raw keys.
            value: Head-major values.
            gate: Raw gate logits or output-gate values for this stage.
            beta: Activated delta-rule update weights.
            a_log: FP32 logarithmic decay-rate parameters.
            dt_bias: FP32 per-channel decay bias.
            scale: Attention output scaling factor.
            lower_bound: Lower bound of the safe decay gate.
            metadata: Native packed-document offsets and chunk indices.
        """
        inputs = (query, key, value, gate, beta, a_log, dt_bias)
        prepared = prepare_kda_native(inputs, scale, lower_bound, metadata)
        return finish_kda_native_forward(ctx, inputs, prepared, None, scale, lower_bound, metadata)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Compute local gradients with a zero terminal state gradient.

        Args:
            ctx: Autograd context owned by this invocation.
            grad_output: Gradient of the local output.
        """
        # Unpack once: non-reentrant checkpoint hooks consume saved tensors.
        saved = ctx.saved_tensors
        prepared = prepare_kda_native_backward(ctx, saved, grad_output)
        gradients = finish_kda_native_backward(ctx, saved, *prepared, None)
        return (*gradients, None, None, None)
