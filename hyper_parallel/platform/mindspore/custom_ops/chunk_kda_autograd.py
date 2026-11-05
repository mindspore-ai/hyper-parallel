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
"""Local state-aware autograd assembled from the explicit KDA stages.

No communication lives here. An optional caller-owned scan supplies boundary
states and replicated-gradient reduction (used by MF's experimental CP path).
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from hyper_parallel.core.shard.dfunction import DFunction
from hyper_parallel.custom_ops.experimental.chunk_kda_stages import (
    npu_chunk_kda_fwd_prepare, npu_chunk_kda_fwd_h, npu_chunk_kda_fwd_finalize,
    npu_chunk_kda_bwd_prepare, npu_chunk_kda_bwd_dhu, npu_chunk_kda_bwd_finalize,
)

if TYPE_CHECKING:
    import mindspore as ms


def _prepare(inputs, options, mode):
    q, k, v, g, beta, a_log, bias = inputs
    return npu_chunk_kda_fwd_prepare(
        q, k, v, g, beta, options["scale"], options["chunk_size"],
        a_log=a_log, dt_bias=bias.reshape(-1).contiguous(), cu_seqlens=options["cu_seqlens"],
        chunk_indices=options["chunk_indices"], epsilon=options["epsilon"],
        use_qk_l2norm_in_kernel=options["use_qk_l2norm_in_kernel"],
        use_gate_in_kernel=True, safe_gate=True, lower_bound=options["lower_bound"],
        use_beta_sigmoid_in_kernel=options["use_beta_sigmoid_in_kernel"],
        allow_neg_eigval=options["allow_neg_eigval"], backward_mode=mode,
    )


def _scan_forward(cache, h0, options):
    return npu_chunk_kda_fwd_h(
        cache[6], cache[3], cache[4], gk=cache[0], initial_state=h0,
        output_final_state=True, chunk_size=options["chunk_size"],
        cu_seqlens=options["cu_seqlens"], chunk_indices=options["chunk_indices"],
    )


class _ChunkKdaStageFunction(DFunction):
    """Save/recompute relays with h0/dht/dh0 and K-first internal states."""

    @staticmethod
    def forward(ctx: Any, q: ms.Tensor, k: ms.Tensor, v: ms.Tensor, g: ms.Tensor,
                beta: ms.Tensor, a_log: ms.Tensor, bias: ms.Tensor,
                initial_state: ms.Tensor | None, options: dict, scan: Any) -> tuple:
        """Prepare local relays and optionally obtain the CP boundary state."""
        inputs = (q, k, v, g, beta, a_log, bias)
        cache = _prepare(inputs, options, "save" if options["disable_recompute"] else "none")
        h0 = initial_state if scan is None else scan.forward_boundary(cache, initial_state)
        h, vn, final = _scan_forward(cache, h0, options)
        out = npu_chunk_kda_fwd_finalize(
            cache[7], cache[1], vn, h, output_layout="BNSD",
            cu_seqlens=options["cu_seqlens"], chunk_indices=options["chunk_indices"],
        )
        saved = dict(zip(("q", "k", "v", "g", "beta", "a", "bias"), inputs))
        if h0 is not None:
            saved["h0"] = h0
        if options["disable_recompute"]:
            saved.update({f"cache{i}": value for i, value in enumerate(cache) if value is not None})
            saved.update(h=h, vn=vn)
        ctx.names = tuple(saved)
        ctx.save_for_backward(*saved.values())
        ctx.options, ctx.scan = options, scan
        ctx.has_initial = initial_state is not None
        ctx.final_shape = final.shape
        return out, final if options["output_final_state"] else None

    @staticmethod
    def backward(ctx: Any, grad_out: ms.Tensor | None, grad_final: ms.Tensor | None = None) -> tuple:
        """Reconstruct or load relays, reverse the state scan, then finalize."""
        from mindspore import mint  # pylint: disable=C0415
        saved = dict(zip(ctx.names, ctx.saved_tensors))
        options, scan = ctx.options, ctx.scan
        inputs = tuple(saved[key] for key in ("q", "k", "v", "g", "beta", "a", "bias"))
        q, k, v, g, beta, a_log, bias = inputs
        h0 = saved.get("h0")
        if options["disable_recompute"]:
            cache = tuple(saved.get(f"cache{i}") for i in range(13))
            h, vn = saved["h"], saved["vn"]
        else:
            # This really reconstructs prepare + state relays in backward;
            # none of the forward's per-token caches/h/v_new are retained.
            cache = _prepare(inputs, options, "save")
            h, vn, _ = _scan_forward(cache, h0, options)
        d_o = mint.zeros_like(v) if grad_out is None else grad_out.contiguous()
        metadata = {key: options[key] for key in ("cu_seqlens", "chunk_indices", "chunk_size")}
        da, dv_local, dq_raw = npu_chunk_kda_bwd_prepare(
            cache[1], vn, d_o, h, scale=options["scale"], **metadata,
        )
        if scan is not None:
            grad_final = scan.backward_boundary(cache, d_o, dv_local, grad_final, options["scale"])
        relay_h0 = (mint.zeros(ctx.final_shape, dtype=q.dtype) if h0 is None else h0.astype(q.dtype))
        dh, dh0, dv_scan = npu_chunk_kda_bwd_dhu(
            cache[5], cache[6], cache[3], d_o, dv_local, options["scale"],
            gk=cache[0], h0=relay_h0,
            dht=None if grad_final is None else grad_final.astype(q.dtype).contiguous(), **metadata,
        )
        norm = options["use_qk_l2norm_in_kernel"]
        sigmoid = options["use_beta_sigmoid_in_kernel"]
        dq, dk, dv, dbeta, dg, d_a, d_bias = npu_chunk_kda_bwd_finalize(
            cache[8] if norm else q, cache[9] if norm else k, v, cache[0], g,
            cache[12] if sigmoid else beta, a_log, bias.reshape(v.shape[1], q.shape[-1]).contiguous(), cache[2], vn, h,
            dh, dv_scan, da, dq_raw, scale=options["scale"], lower_bound=options["lower_bound"],
            q_rstd=cache[10] if norm else None, k_rstd=cache[11] if norm else None,
            use_beta_sigmoid_in_kernel=sigmoid, allow_neg_eigval=options["allow_neg_eigval"], **metadata,
        )
        d_bias = d_bias.reshape(bias.shape)
        dh0 = dh0.astype(h0.dtype) if ctx.has_initial else None
        if scan is not None:
            d_a, d_bias, dh0 = scan.finish_gradients(d_a, d_bias, dh0)
        return dq, dk, dv, dg, dbeta, d_a, d_bias, dh0, None, None


def _head_major(tensor, layout, scalar=False):
    if layout == "BSND":
        return tensor.transpose((0, 2, 1) if scalar else (0, 2, 1, 3)).contiguous()
    if layout == "TND":
        tensor = tensor.transpose((1, 0) if scalar else (1, 0, 2))
    if layout in ("TND", "NTD"):
        tensor = tensor.expand_dims(0)
    return tensor.contiguous()


def run_chunk_kda_stages(
        q: ms.Tensor, k: ms.Tensor, v: ms.Tensor, g: ms.Tensor, beta: ms.Tensor,
        scale: float, chunk_size: int = 64, *, layout: str = "BNSD",
        output_final_state: bool = False, safe_gate: bool = False, lower_bound: float = -5.,
        use_gate_in_kernel: bool = True, a_log: ms.Tensor | None = None,
        dt_bias: ms.Tensor | None = None, cu_seqlens: list | None = None,
        chunk_indices: list | None = None, initial_state: ms.Tensor | None = None,
        disable_recompute: bool = True, state_v_first: bool = False, epsilon: float = 1e-6,
        use_qk_l2norm_in_kernel: bool = False, use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False, use_exp2: bool = True, scan: Any = None) -> tuple:
    """Internal local training adapter; MF may supply its caller-owned scan.

    The current paired FLA finalize ABI requires safe raw gates, BF16,
    chunk64/K=V128 and exp2. Layout conversion and grouped-head expansion
    happen outside the custom Function so ordinary autograd reverses them.
    """
    import mindspore as ms  # pylint: disable=C0415
    from mindspore import mint  # pylint: disable=C0415
    if not safe_gate or not use_gate_in_kernel or not use_exp2 or a_log is None:
        raise ValueError("State-aware KDA stages require safe_gate/use_gate_in_kernel/use_exp2=True and a_log.")
    if layout not in ("BNSD", "BSND", "NTD", "TND"):
        raise ValueError("layout must be BNSD, BSND, NTD or TND.")
    packed = len(q.shape) == 3
    if packed != (layout in ("NTD", "TND")):
        raise ValueError("rank-3 uses NTD/TND; rank-4 uses BNSD/BSND.")
    q, k, v, g = (_head_major(tensor, layout) for tensor in (q, k, v, g))
    beta = _head_major(beta, layout, scalar=True)
    if q.shape[1] != v.shape[1]:
        if v.shape[1] % q.shape[1]:
            raise ValueError("value heads must be a multiple of query/key heads.")
        ratio = v.shape[1] // q.shape[1]
        q, k = (mint.repeat_interleave(tensor, ratio, dim=1) for tensor in (q, k))
    if dt_bias is None:
        dt_bias = mint.zeros((v.shape[1], q.shape[-1]), dtype=ms.float32)
    if initial_state is not None and state_v_first:
        initial_state = initial_state.swapaxes(-1, -2).contiguous()
    options = {"scale": scale, "chunk_size": chunk_size, "output_final_state": output_final_state,
               "lower_bound": lower_bound, "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices,
               "disable_recompute": disable_recompute, "epsilon": epsilon,
               "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
               "use_beta_sigmoid_in_kernel": use_beta_sigmoid_in_kernel, "allow_neg_eigval": allow_neg_eigval}
    out, final = _ChunkKdaStageFunction.apply(q, k, v, g, beta, a_log, dt_bias, initial_state, options, scan)
    out = out.squeeze(0).transpose(1, 0, 2) if packed else out.transpose(0, 2, 1, 3)
    if final is not None and state_v_first:
        final = final.swapaxes(-1, -2)
    return out, final
