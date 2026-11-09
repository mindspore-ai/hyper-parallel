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
"""Optional native KDA stages for dense/packed Ulysses and summary state exchange."""

from __future__ import annotations

import importlib
import inspect
from bisect import bisect_left
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from hyper_parallel.components.functional.kimi_delta_attention_state_summary import apply_kda_state_summary
from hyper_parallel.distributed.context_parallel.kimi_delta_attention_rd import KDAStateRecursiveDoubling


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


def _prepare_kda(inputs, scale, lower_bound, metadata):
    """Prepare once for native state summaries and the local state recurrence."""
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
    return _KDAPrepared(*result)


def _native_backward_view(tensor: torch.Tensor, packed: bool) -> torch.Tensor:
    """Expose packed NTD views without copying the B=1 Prepare/FwdH storage."""
    return tensor.squeeze(0) if packed else tensor


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
    return _AscendCKDAStages.apply(
        *tokens, gate.contiguous(), beta.contiguous(), a_log.reshape(-1).clone(),
        dt_bias.reshape(-1, query.shape[-1]).clone(), scale, lower_bound, metadata, None)


@lru_cache(maxsize=1)
def _load_state_ops():
    """Check every native stage before entering the first state collective."""
    ops = _load_ops()
    required = (
        "pre_process_fwd_kernel_merged", "pre_process_bwd_kernel_merged",
        "merge_fwd_bwd_kernel",
    )
    missing = [name for name in required if not callable(getattr(ops, name, None))]
    if missing:
        raise RuntimeError(f"Native KDA state CP requires these fla-npu APIs: {', '.join(missing)}.")
    return ops


def _native_ag_intervals(info, tokens, rank, size):
    """Return tail/head windows and document-bounded exclusive scan lengths."""
    if info is None:
        return (0, tokens), (0, tokens), rank, size - rank - 1
    first_rank = (info.start - info.preceding_sample_tokens) // tokens
    last_rank = (info.global_offsets[bisect_left(info.global_offsets, info.end)] - 1) // tokens
    return ((info.tail.start, info.tail.stop), (info.head.start, info.head.stop),
            rank - first_rank if info.continues_previous else 0,
            last_rank - rank if info.continues_next else 0)


class _NativeStateExchange:
    """Bind one invocation's sequence plan to precreated communication groups.

    Only metadata survives forward. Receive buffers and send handles are owned
    by the active direction and drained before that invocation returns.
    """

    def __init__(self, mesh: Any, executor: Any, info: Any, local_length: int, heads: int) -> None:
        """Copy communication metadata and retain only this call's sequence plan."""
        self.executor, self.info, self.heads = executor, info, heads
        self.rank = mesh.get_local_rank()
        self.size = mesh.size()
        self.ranks = tuple(mesh.rank_list)
        self.group = mesh.get_group()
        self.width = getattr(executor, "width", self.size) if executor is not None else 1
        self.block, self.local = divmod(self.rank, self.width)
        self.owner = self.block * self.width + self.width - 1
        self.is_rd = isinstance(executor, KDAStateRecursiveDoubling)
        self.plans = (tuple(executor.directional_plan(info, local_length, reverse=reverse) for reverse in (False, True))
                      if self.is_rd else ())

    def peer(self, forward: bool, receiving: bool) -> int | None:
        """Return a chronological peer, pruning disconnected ordinary P2P edges."""
        direction = 1 if forward else -1
        peer = self.owner + (-direction if receiving else direction) * self.width
        if not 0 <= peer < self.size:
            return None
        if self.width == 1 and self.info is not None:
            needs_previous = forward == receiving
            connected = self.info.continues_previous if needs_previous else self.info.continues_next
            if not connected:
                return None
        return peer

    def needs_summary(self, forward: bool) -> bool:
        """Avoid producing a summary without an exit consumer.

        Args:
            forward: Select the forward S/M or backward G/N direction.
        """
        if self.is_rd:
            plan = self.plans[not forward]
            return plan.shift_peers(plan.local_rank)[0] is not None
        return self.width > 1 or self.peer(forward, receiving=False) is not None

    def receive(self, reference: torch.Tensor, forward: bool) -> tuple[torch.Tensor | None, Any]:
        """Post reception before native prepare, independently of summary production.

        Args:
            reference: Head-major input supplying the head count and device.
            forward: Select the chronological communication direction.
        """
        if self.is_rd or self.width == self.size:
            return None, None
        incoming = reference.new_zeros((1, reference.shape[1], 128, 128), dtype=torch.float32)
        peer = self.peer(forward, receiving=True)
        handle = None
        if self.rank == self.owner and peer is not None:
            handle = dist.irecv(incoming, src=self.ranks[peer], group=self.group)
        return incoming, handle

    @staticmethod
    def apply(packed: torch.Tensor, incoming: torch.Tensor | None) -> torch.Tensor:
        """Apply forward M or native backward N directly, without transposing N."""
        return apply_kda_state_summary(packed[..., :128], packed[..., 128:], incoming)

    def exchange(self, ops: Any, packed: torch.Tensor | None,
                 reception: tuple[torch.Tensor | None, Any], forward: bool,
                 count: int) -> tuple[torch.Tensor, Any, torch.Tensor | None]:
        """Produce the local input state and retain an asynchronous outgoing payload.

        Args:
            ops: Native summary-merge operations for AllGather.
            packed: Local S/M or G/N, absent when no exit consumes it.
            reception: Previously posted receive buffer and handle.
            forward: Select the chronological communication direction.
            count: Number of maps in the relevant AllGather prefix or suffix.
        """
        if self.is_rd:
            state_shape = (1, self.heads, 128, 128)
            maps = None if packed is None else packed.reshape(1, self.heads, 128, 256)
            result = self.executor.scan_affine(maps, self.plans[not forward], state_shape, reverse=not forward)
            return result, None, None
        if self.width == self.size:
            maps = self.executor.gather_packed(packed.reshape(1, *packed.shape[-3:]))
            incoming = maps.new_zeros((maps.shape[1], 128, 128))
            if count:
                ops.merge_fwd_bwd_kernel(incoming, maps, count, self.rank, forward=forward)
            return incoming.unsqueeze(0), None, None
        incoming, recv_handle = reception
        if self.width > 1:
            return self._grouped_exchange(packed, incoming, recv_handle, forward)
        transfer = self.info is None or self.info.num_sequences == 1
        if recv_handle is not None and transfer:
            recv_handle.wait()
            recv_handle = None
        send_handle, outgoing = None, None
        peer = self.peer(forward, receiving=False)
        if peer is not None:
            outgoing = self.apply(packed, incoming if transfer else None).contiguous()
            send_handle = dist.isend(outgoing, dst=self.ranks[peer], group=self.group)
        # A reset permits sending the independent tail/head before this wait,
        # but local h/dhu still consumes the received first/last fragment state.
        if recv_handle is not None:
            recv_handle.wait()
        return incoming, send_handle, outgoing

    def _grouped_exchange(self, packed, incoming, recv_handle, forward):
        """Gather native maps within a group and relay only its exit state."""
        if self.info is not None:
            if not self.info.transfer_active or not self.info.continues_next:
                packed[..., 128:].zero_()
            connected = self.info.continues_next if forward else self.info.continues_previous
            if not connected:
                packed[..., :128].zero_()
        maps = packed.new_empty((self.width, *packed.shape[-3:]))
        dist.all_gather_into_tensor(maps, packed.reshape(1, *packed.shape[-3:]), group=self.executor.intra)
        send_handle, outgoing, result = None, None, None
        indices = range(self.width) if forward else range(self.width - 1, -1, -1)
        prefix_count = self.local if forward else self.width - self.local - 1
        if self.rank == self.owner:
            if recv_handle is not None:
                recv_handle.wait()
            peer = self.peer(forward, receiving=False)
            if peer is not None:
                outgoing, result = self._scan(maps, incoming, indices, prefix_count)
                send_handle = dist.isend(outgoing, dst=self.ranks[peer], group=self.group)
        if self.peer(forward, receiving=True) is not None:
            dist.broadcast(incoming, src=self.ranks[self.owner], group=self.executor.intra)
        if result is None:
            local_indices = range(self.local) if forward else range(self.width - 1, self.local, -1)
            result, _ = self._scan(maps, incoming, local_indices)
        return result, send_handle, outgoing

    @staticmethod
    def _scan(maps, incoming, indices, prefix_count=None):
        """Keep native M/N orientation while applying a nonzero group entry."""
        state = incoming
        prefix = incoming if prefix_count == 0 else None
        for step, index in enumerate(indices, start=1):
            state = _NativeStateExchange.apply(maps[index:index + 1], state)
            if step == prefix_count:
                prefix = state
        return state, prefix


def _exchange_native_gradient(ctx, ops, inputs, reception):
    """Own G/N production, reset injection and the stock dHT dtype boundary."""
    if ctx.exchange is None:
        return None, None, None
    gk, qg, kg, weights, grad, dv_local_head = inputs
    maps = None
    if ctx.exchange.needs_summary(False):
        # Only the stock G/N ABI rounds gk; local dhu/finalize retain FP32.
        maps = ops.pre_process_bwd_kernel_merged(
            qg, kg, weights, grad, dv_local_head, ctx.scale, 64,
            gk=gk.to(qg.dtype), cu_seqlens=ctx.head)
    terminal, send_handle, outgoing = ctx.exchange.exchange(ops, maps, reception, False, ctx.suffix)
    if ctx.exchange.info is not None:
        terminal = ctx.exchange.info.last_state(terminal)
    # Communication and summary apply retain FP32; only stock dhu requires BF16.
    terminal = None if terminal is None else terminal.to(qg.dtype)
    return terminal, send_handle, outgoing


class _AscendCKDAStages(torch.autograd.Function):
    """Share native local stages; state protocols optionally inject H and D."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                gate: torch.Tensor, beta: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
                scale: float, lower_bound: float, metadata: dict,
                exchange: _NativeStateExchange | None) -> torch.Tensor:
        """Share prepare with S/M and local FwdH, then release gathered maps.

        Args:
            ctx: Per-invocation autograd context.
            query: Raw head-major queries.
            key: Raw head-major keys.
            value: Head-major values.
            gate: Sequence-major gate logits.
            beta: Sequence-major beta logits.
            a_log: FP32 decay parameters.
            dt_bias: FP32 gate bias.
            scale: Attention scale.
            lower_bound: Safe-gate lower bound.
            metadata: Native packed offsets and chunk indices.
            exchange: Optional state protocol for this invocation.
        """
        ops = _load_state_ops() if exchange is not None else _load_ops()
        inputs = (query, key, value, gate, beta, a_log, dt_bias)
        reception = None if exchange is None else exchange.receive(query, True)
        prepared = _prepare_kda(inputs, scale, lower_bound, metadata)
        initial = send_handle = outgoing = head = suffix = None
        if exchange is not None:
            tail, head, prefix, suffix = _native_ag_intervals(
                exchange.info, query.shape[2], exchange.rank, exchange.size)
            maps = None
            if exchange.needs_summary(True):
                maps = ops.pre_process_fwd_kernel_merged(
                    prepared.kg, prepared.w, prepared.u, gk=prepared.gk, cu_seqlens=tail)
            initial, send_handle, outgoing = exchange.exchange(ops, maps, reception, True, prefix)
            del maps, reception
            if exchange.info is not None:
                initial = exchange.info.first_state(initial)
        hidden, v_new, _ = ops.chunk_fwd_h(
            prepared.kg, prepared.w, prepared.u, gk=prepared.gk, initial_state=initial,
            output_final_state=False, use_exp2=True, **metadata)
        output = ops.chunk_kda_fwd_finalize(
            prepared.qg_scaled, prepared.aqk, v_new, hidden, output_layout="BSND", **metadata)
        if send_handle is not None:
            send_handle.wait()
        del outgoing
        prepared = prepared._replace(u=None, qg_scaled=None)
        # The saved hidden tensor contains entry states. Summary protocols
        # produce the outgoing gradient themselves, so native dh0 is unused.
        ctx.save_for_backward(value, gate, a_log, dt_bias, *prepared, hidden, v_new)
        ctx.dtypes = tuple(tensor.dtype for tensor in inputs)
        ctx.scale, ctx.lower_bound, ctx.native_metadata = scale, lower_bound, metadata
        ctx.exchange, ctx.head, ctx.suffix = exchange, head, suffix
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Collect G/N, inject nonzero dHT and finish raw-input gradients.

        Args:
            ctx: Forward-owned activations and communication metadata.
            grad_output: Sequence-major output gradient.
        """
        ops = _load_state_ops() if ctx.exchange is not None else _load_ops()
        value, gate, a_log, dt_bias, *saved = ctx.saved_tensors
        (gk, aqk, akk, weights, _, qg, kg, _, query, key,
         query_rstd, key_rstd, beta, hidden, v_new) = saved
        packed = bool(ctx.native_metadata)

        reception = None if ctx.exchange is None else ctx.exchange.receive(query, False)
        grad = grad_output.transpose(1, 2).contiguous()
        d_aqk, dv_local, dq_raw = ops.chunk_kda_bwd_prepare(
            *(_native_backward_view(tensor, packed) for tensor in (aqk, v_new, grad, hidden)), scale=ctx.scale,
            layout="NTD" if packed else "BNSD", chunk_size=64, **ctx.native_metadata)
        dv_local_head = dv_local.unsqueeze(0) if packed else dv_local
        terminal, send_handle, outgoing = _exchange_native_gradient(
            ctx, ops, (gk, qg, kg, weights, grad, dv_local_head), reception)
        del reception
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
        if send_handle is not None:
            send_handle.wait()
        del outgoing
        gradients = (dq, dk, dv, dgate, dbeta, da_log, ddt_bias.reshape_as(dt_bias))
        return (*(grad.to(dtype) for grad, dtype in zip(gradients, ctx.dtypes)), None, None, None, None)


def ascendc_state_kda(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    gate: torch.Tensor, beta: torch.Tensor, *, a_log: torch.Tensor,
    dt_bias: torch.Tensor, state_cp: Any, cp_mesh: Any, sequence_info: Any = None,
    scale: float | None = None, lower_bound: float = -5.0,
) -> torch.Tensor:
    """Run native KDA with P2P, AllGather, grouped AllGather/P2P, or RD.

    Args:
        query: Raw BF16 queries [1,T,H,128].
        key: Matching raw keys.
        value: BF16 values with the query shape.
        gate: Raw gate logits with the query shape.
        beta: Raw beta logits [1,T,H].
        a_log: Per-head FP32 decay parameters.
        dt_bias: FP32 gate bias [H,128].
        state_cp: Existing gather/RD executor, or None for ordinary P2P.
        cp_mesh: Ordered state mesh after any Ulysses redistribution.
        sequence_info: Packed interval metadata after any Ulysses redistribution.
        scale: Attention scale, default 1/sqrt(128).
        lower_bound: Safe gate lower bound.

    Returns:
        Output with the query shape. Uses the unmodified native stage APIs:
        backward summary gk and local dhu dHT are converted to model dtype.
        Prepared gates, communicated states and summary apply remain FP32.
        These ABI conversions introduce rounding relative to FP32-input kernels.
    """
    if query.device.type != "npu" or query.dtype != torch.bfloat16:
        raise ValueError("Native KDA state CP requires BF16 NPU inputs.")
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[-1] != 128:
        raise ValueError("Native KDA state CP requires [1,T,H,128] inputs.")
    if any(tensor.shape != query.shape for tensor in (key, value, gate)) or beta.shape != query.shape[:-1]:
        raise ValueError("Native KDA state CP requires matching Q/K/V/gate heads and beta [1,T,H].")
    metadata = _packed_metadata(None if sequence_info is None else sequence_info.local_offsets, query.shape[1])
    scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    tokens = tuple(tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
    exchange = _NativeStateExchange(cp_mesh, state_cp, sequence_info, query.shape[1], query.shape[2])
    return _AscendCKDAStages.apply(
        *tokens, gate.contiguous(), beta.contiguous(), a_log.reshape(-1).clone(),
        dt_bias.reshape(-1, query.shape[-1]).clone(), scale, lower_bound, metadata, exchange)
