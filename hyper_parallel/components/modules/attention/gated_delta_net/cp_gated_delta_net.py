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
"""GDN Ulysses and state-P2P scheduling over the complete local algorithm."""
from __future__ import annotations
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.nn import functional as F

from hyper_parallel.components.functional.gated_delta_net import (
    chunk_gated_delta_rule, is_triton_gdn_available, apply_gdn_state_summary, apply_gdn_state_gradient_summary,
)
from hyper_parallel.components.functional.linear_attention_convolution import fused_causal_conv1d, packed_causal_conv1d
from hyper_parallel.components.modules.attention.linear_attention.layer_adapter import prepare_linear_attention_input
from hyper_parallel.components.modules.attention.linear_attention.cp_convolution import causal_convolutions_with_halo
from hyper_parallel.components.modules.attention.linear_attention.cp_sharding import validate_linear_attention_layout
from hyper_parallel.components.modules.attention.linear_attention.sequence_layout import PackedSequenceInfo
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.distributed.context_parallel.collectives import exchange_sequence_heads
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from .gated_delta_net import GatedDeltaNetModule


class CPGatedDeltaNetModule(GatedDeltaNetModule):
    """Existing GDN pure-U and state-P2P execution without copying the layer forward."""

    def __init__(self, *, config: Any = None, module: torch.nn.Module | None = None,
                 module_fqn: str = "", context: Any = None, backend: str = "eager",
                 conv_backend: str | None = None, chunk_size: int = 64,
                 input_layout: str = "BSH", ulysses_degree: int | None = 1) -> None:
        """Retain the explicitly requested algorithm and deferred CP resources."""
        del module_fqn, context
        self.ulysses_degree = ulysses_degree
        if conv_backend is None and ulysses_degree == 1:
            conv_backend = "eager"
        super().__init__(config=config, module=module,
                         backend=backend, conv_backend=conv_backend, chunk_size=chunk_size, input_layout=input_layout)
        self.cp_rank, self.cp_size = 0, 1
        self.ulysses_size, self.state_size = 1, 1
        self.cp_context = None
        self.convolve_after_exchange = ulysses_degree is None

    def prepare_input(self, hidden_states: torch.Tensor, options: dict) -> tuple:
        """Resolve real document boundaries against this CP rank's physical shard.

        Args:
            hidden_states: Local dense or flat hidden activations.
            options: Explicit backend options or invocation document metadata.
        """
        return prepare_linear_attention_input(hidden_states, options, self.cp_rank, self.cp_size)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Expose the complete pipeline with internal state exchange owned by autograd."""
        return ParallelModuleSpec("attention.gdn.cp", "cp",
                                  {"execute_attention_pipeline": ("operands", "info")},
                                  validate=validate_linear_attention_layout)

    def bind_parallel(self, binding: Any) -> None:
        """Borrow a full CP context once, preserving GDN's currently qualified modes.

        Args:
            binding: Resolved parallel binding containing the CP context.
        """
        if binding.cp_context is None:
            raise ValueError("GDN CP requires CPContext")
        self.cp_context = binding.cp_context
        self.cp_rank, self.cp_size = self.cp_context.rank, self.cp_context.size
        degree = self.cp_size if self.ulysses_degree is None else self.ulysses_degree
        if degree not in (1, self.cp_size):
            raise NotImplementedError("GDN supports pure Ulysses or P2P, not a mixed degree")
        self.ulysses_size, self.state_size = degree, self.cp_size // degree
        self.convolve_after_exchange = self.ulysses_degree is None or (1 < self.cp_size == degree)
        if self.convolve_after_exchange and (self.num_k_heads % degree or self.num_v_heads % degree):
            raise ValueError("GDN Ulysses degree must divide query and value heads")
        if not self.convolve_after_exchange and self.cp_size > 1 and self.backend != "triton":
            raise NotImplementedError("GDN state CP requires Triton; use local/pure Ulysses for other backends")

    def _exchange_raw_inputs(self, operands: tuple) -> tuple:
        """Exchange packed Q/K/V and gates before pure-Ulysses convolution."""
        mixed, beta, decay = operands
        if self.cp_size == 1:
            return mixed, beta, decay
        query, key, value = mixed.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
        widths = (self.key_dim // self.cp_size, self.key_dim // self.cp_size, self.value_dim // self.cp_size,
                  self.num_v_heads // self.cp_size, self.num_v_heads // self.cp_size)
        chunks = [tensor.split(width, dim=-1) for tensor, width in zip((query, key, value, beta, decay), widths)]
        packed = torch.cat([torch.cat(parts, dim=-1) for parts in zip(*chunks)], dim=-1).contiguous()
        packed = exchange_sequence_heads(packed, self.cp_context.mesh, split_dim=2, concat_dim=1)
        query, key, value, beta, decay = packed.split(widths, dim=-1)
        return torch.cat((query, key, value), dim=-1), beta, decay

    def _local_conv_parameter(self, parameter: torch.Tensor | None) -> torch.Tensor | None:
        if parameter is None:
            return None
        segments = parameter.split((self.key_dim, self.key_dim, self.value_dim), dim=0)
        return torch.cat([part.narrow(0, self.cp_rank * (part.shape[0] // self.cp_size),
                                     part.shape[0] // self.cp_size) for part in segments], dim=0)

    def _convolve_local_heads(self, mixed: torch.Tensor, info: PackedSequenceInfo | None) -> torch.Tensor:
        conv = self.conv1d
        weight, bias = self._local_conv_parameter(conv.weight), self._local_conv_parameter(conv.bias)
        if info is not None:
            return packed_causal_conv1d(mixed, conv, info, weight=weight, bias=bias, backend=self.conv_backend)
        if self.conv_backend != "eager":
            return fused_causal_conv1d(mixed, weight, bias, backend=self.conv_backend, dilation=conv.dilation[0])
        output = F.conv1d(mixed.transpose(1, 2), weight, bias, stride=conv.stride,
                          padding=conv.padding, dilation=conv.dilation, groups=mixed.shape[-1])
        return F.silu(output[:, :, :mixed.shape[1]]).transpose(1, 2)

    def execute_attention_pipeline(self, operands: tuple, info: PackedSequenceInfo | None) -> torch.Tensor:
        """Exchange tokens before pure-U convolution, or exchange states inside P2P autograd.

        Args:
            operands: Projected inputs consumed by this attention stage.
            info: Immutable packed-sequence metadata, or None for dense input.
        """
        if self.cp_context is None:
            raise RuntimeError("Bind GDN CP resources before forward")
        if self.convolve_after_exchange:
            mixed, beta, decay = self._exchange_raw_inputs(operands)
            info = None if info is None else info.ulysses(self.cp_size, self.cp_rank)
            mixed = self._convolve_local_heads(mixed, info)
            heads = self.num_v_heads // self.cp_size
            a_log = self.A_log.narrow(0, self.cp_rank * heads, heads)
            dt_bias = self.dt_bias.narrow(0, self.cp_rank * heads, heads)
            inputs = self.prepare_core_inputs(mixed, beta, decay, self.cp_size, a_log, dt_bias)
            output = self.run_local_core(inputs, info)
            return (output if self.cp_size == 1 else
                    exchange_sequence_heads(output, self.cp_context.mesh, split_dim=1, concat_dim=2))
        mixed, beta, decay = operands
        mixed, = causal_convolutions_with_halo((mixed,), (self.conv1d,), self.cp_context.mesh,
                                               self.cp_rank, self.cp_size, sequence_info=info,
                                               backend=self.conv_backend)
        inputs = self.prepare_core_inputs(mixed, beta, decay)
        if self.backend == "triton":
            if not is_triton_gdn_available(*inputs):
                raise RuntimeError("GDN P2P Triton requires the fixed NPU BF16 K=V=128 contract")
            return triton_state_gdn(*inputs, self.cp_context.mesh, self.cp_rank, self.cp_size, sequence_info=info)
        return self.run_local_core(inputs, info)


def _global_peer_rank(cp_mesh: DeviceMesh, local_rank: int) -> int:
    """Map a CP-local rank index to its global distributed rank."""
    return int(cp_mesh.rank_list[local_rank])


def _l2norm_torch(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Match the pure torch GDN reference l2norm helper."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _prepare_gdn_forward_send(key_norm, w, u, g_cumsum, send_next, sequence_info,
                              recv_buffer, recv_handle, cp_group, next_rank):
    """Forward the tail summary before consuming the prepared local chunks."""
    # These helpers load optional Triton kernels only when the fused path runs.
    from hyper_parallel.components.functional.gated_delta_net_triton import (  # pylint: disable=import-outside-toplevel
        chunk_gated_delta_rule_state_summary_fwd,
    )
    initial_state = None
    state_ext = None
    transition = None
    if send_next:
        tail = slice(None) if sequence_info is None else sequence_info.tail
        state_ext, transition = chunk_gated_delta_rule_state_summary_fwd(
            key_norm[:, tail], w[:, tail], u[:, tail], g_cumsum[:, tail],
            output_transition=sequence_info is None or sequence_info.transfer_active,
        )

    if recv_handle is not None and (sequence_info is None or sequence_info.transfer_active):
        recv_handle.wait()
        initial_state = recv_buffer
        recv_handle = None

    send_buffer = None
    send_handle = None
    if send_next:
        send_buffer = apply_gdn_state_summary(
            state_ext,
            transition,
            initial_state if transition is not None else None,
        ).contiguous()
        send_handle = dist.isend(send_buffer, dst=next_rank, group=cp_group)

    if recv_handle is not None:
        recv_handle.wait()
        initial_state = recv_buffer
    return initial_state, transition, send_buffer, send_handle


def _finish_gdn_local_backward(ctx, query, key, value, g_cumsum, beta, matrix_a,
                               w, h, v_new, dv, dh, grad_output, cu_seqlens):
    """Finish local gradients after the outgoing state gradient is in flight."""
    # These helpers load optional Triton kernels only when the fused path runs.
    from hyper_parallel.components.functional.gated_delta_net_triton import (  # pylint: disable=import-outside-toplevel
        chunk_gated_delta_rule_bwd_finish_saved,
    )
    empty = query.new_empty(0)
    dq, dk, dv, dg, dbeta = chunk_gated_delta_rule_bwd_finish_saved(
        query,
        key,
        query,
        key,
        value,
        g_cumsum,
        beta,
        matrix_a,
        w,
        h,
        v_new,
        dv,
        grad_output,
        dh,
        empty,
        empty,
        ctx.scale,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )
    return dq, dk, dv, dg, dbeta


class _GDNStateP2PTritonFunction(torch.autograd.Function):
    """Pipeline fused affine GDN states over sequence-sharded CP ranks."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ,too-many-locals
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cp_rank: int,
        cp_size: int,
        cp_group: dist.ProcessGroup,
        prev_rank: int,
        next_rank: int,
        sequence_info: Optional[PackedSequenceInfo],
    ) -> torch.Tensor:
        """Run fused local GDN and forward its affine state across CP ranks.

        Args:
            ctx: Autograd context owning tensors and communication metadata for this invocation.
            query: Query tensor in batch, token, head, key-dimension order.
            key: Key tensor with the same token and key-head axes as query.
            value: Value tensor in batch, token, head, value-dimension order.
            g: Log decay values for the recurrence.
            beta: Delta-rule update strengths, or logits when sigmoid fusion is enabled.
            cp_rank: Rank index within the CP mesh.
            cp_size: Number of ranks in the CP mesh.
            cp_group: Process group in sequence-rank order.
            prev_rank: Global rank sending the incoming state.
            next_rank: Global rank receiving the outgoing state.
            sequence_info: Immutable document metadata for this local token interval.
        """
        # Optional Triton kernels must stay unloaded for ordinary CPU imports.
        from hyper_parallel.components.functional.gated_delta_net_triton import (  # pylint: disable=import-outside-toplevel
            chunk_gated_delta_rule_fwd_apply_state_saved, chunk_gated_delta_rule_fwd_output_saved,
            chunk_gated_delta_rule_fwd_prepare_saved,
        )

        receive_previous = cp_rank > 0 and (sequence_info is None or sequence_info.continues_previous)
        send_next = cp_rank < cp_size - 1 and (sequence_info is None or sequence_info.continues_next)
        initial_state = None
        recv_buffer = None
        recv_handle = None
        if receive_previous:
            recv_buffer = torch.empty(
                (query.shape[0], query.shape[2], query.shape[3], value.shape[3]),
                device=query.device,
                dtype=torch.float32,
            )
            recv_handle = dist.irecv(recv_buffer, src=prev_rank, group=cp_group)

        cu_seqlens = None if sequence_info is None else sequence_info.materialize(query.device, 64).cu_seqlens
        (
            query_norm, key_norm, _, _, g_cumsum, matrix_a, w, u, scale,
        ) = chunk_gated_delta_rule_fwd_prepare_saved(
            query, key, value, g, beta,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens,
        )

        initial_state, transition, send_buffer, send_handle = _prepare_gdn_forward_send(
            key_norm, w, u, g_cumsum, send_next, sequence_info,
            recv_buffer, recv_handle, cp_group, next_rank,
        )
        h, v_new, _ = chunk_gated_delta_rule_fwd_apply_state_saved(
            key_norm, g_cumsum, w, u,
            initial_state=initial_state if sequence_info is None else sequence_info.first_state(initial_state),
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        output = chunk_gated_delta_rule_fwd_output_saved(
            query_norm, key_norm, g_cumsum, h, v_new, scale,
            cu_seqlens=cu_seqlens,
        ).to(query.dtype)

        if send_handle is not None:
            send_handle.wait()
        del send_buffer

        empty = query.new_empty(0)
        ctx.save_for_backward(
            query_norm,
            key_norm,
            value,
            g_cumsum,
            beta,
            matrix_a,
            initial_state if initial_state is not None else empty,
            transition if transition is not None else empty,
            cu_seqlens if cu_seqlens is not None else query.new_empty(0, dtype=torch.int64),
        )
        ctx.sequence_info = sequence_info
        ctx.has_initial_state = initial_state is not None
        ctx.has_transition = transition is not None
        ctx.receive_previous = receive_previous
        ctx.send_next = send_next
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.cp_group = cp_group
        ctx.prev_rank = prev_rank
        ctx.next_rank = next_rank
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:  # pylint: disable=too-many-locals
        """Backpropagate local GDN tensors and the state gradient wavefront.

        Args:
            ctx: Autograd context owning tensors and communication metadata for this invocation.
            grad_output: Gradient of the local operator output.
        """
        # Optional Triton kernels must stay unloaded for ordinary CPU imports.
        from hyper_parallel.components.functional.gated_delta_net_triton import (  # pylint: disable=import-outside-toplevel
            chunk_gated_delta_rule_bwd_prepare_saved, chunk_gated_delta_rule_bwd_state_saved,
        )
        from hyper_parallel.components.functional.gated_delta_net_triton import (  # pylint: disable=import-outside-toplevel
            chunk_gated_delta_rule_state_gradient_summary_bwd,
        )

        (
            query,
            key,
            value,
            g_cumsum,
            beta,
            matrix_a,
            initial_state,
            transition,
            saved_cu_seqlens,
        ) = ctx.saved_tensors
        sequence_info = ctx.sequence_info
        cu_seqlens = None if sequence_info is None else saved_cu_seqlens
        if not ctx.has_initial_state:
            initial_state = None

        grad_final_state = None
        recv_handle = None
        if ctx.send_next:
            recv_buffer = torch.empty(
                (query.shape[0], query.shape[2], query.shape[3], value.shape[3]),
                device=grad_output.device,
                dtype=torch.float32,
            )
            recv_handle = dist.irecv(
                recv_buffer,
                src=ctx.next_rank,
                group=ctx.cp_group,
            )

        kernel_initial = initial_state if sequence_info is None else sequence_info.first_state(initial_state)
        w, h, v_new, dv = chunk_gated_delta_rule_bwd_prepare_saved(
            query,
            key,
            value,
            g_cumsum,
            beta,
            matrix_a,
            kernel_initial,
            grad_output,
            ctx.scale,
            cu_seqlens=cu_seqlens,
        )

        grad_state_ext = None
        if ctx.receive_previous:
            head = slice(None) if sequence_info is None else sequence_info.head
            grad_state_ext = chunk_gated_delta_rule_state_gradient_summary_bwd(
                query[:, head], key[:, head], w[:, head], g_cumsum[:, head],
                grad_output[:, head], dv[:, head], ctx.scale)

        if recv_handle is not None and (sequence_info is None or sequence_info.transfer_active):
            recv_handle.wait()
            grad_final_state = recv_buffer
            recv_handle = None

        send_buffer = None
        send_handle = None
        if ctx.receive_previous:
            send_buffer = apply_gdn_state_gradient_summary(
                grad_state_ext,
                transition,
                grad_final_state if ctx.has_transition else None,
            ).contiguous()
            send_handle = dist.isend(
                send_buffer,
                dst=ctx.prev_rank,
                group=ctx.cp_group,
            )

        if recv_handle is not None:
            recv_handle.wait()
            grad_final_state = recv_buffer
        kernel_grad_final = grad_final_state if sequence_info is None else sequence_info.last_state(grad_final_state)
        # Rebind dv here so the prepared local gradient is released before finalize.
        dh, _, dv = chunk_gated_delta_rule_bwd_state_saved(
            query, key, g_cumsum, w, kernel_initial, kernel_grad_final,
            grad_output, dv, ctx.scale, cu_seqlens=cu_seqlens,
        )
        dq, dk, dv, dg, dbeta = _finish_gdn_local_backward(
            ctx, query, key, value, g_cumsum, beta, matrix_a,
            w, h, v_new, dv, dh, grad_output, cu_seqlens,
        )

        if send_handle is not None:
            send_handle.wait()
        return dq, dk, dv, dg, dbeta, None, None, None, None, None, None


def triton_state_gdn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cp_mesh: DeviceMesh,
    cp_rank: int,
    cp_size: int,
    sequence_info: Optional[PackedSequenceInfo] = None,
) -> torch.Tensor:
    """Run local Triton GDN with an affine state wavefront.

    Args:
        query: Queries before paired L2 normalization.
        key: Keys before paired L2 normalization.
        value: Token-major values.
        g: FP32 log-decay gates.
        beta: Activated update weights.
        cp_mesh: Chronological state communication mesh.
        cp_rank: Local rank in that mesh.
        cp_size: Number of state ranks.
        sequence_info: Immutable local document interval, or None for dense input.
    """
    if cp_size == 1:
        output, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            backend="triton",
            cu_seqlens=None if sequence_info is None else sequence_info.materialize(query.device, 64).cu_seqlens,
        )
        return output

    query = _l2norm_torch(query)
    key = _l2norm_torch(key)
    prev_rank = _global_peer_rank(cp_mesh, cp_rank - 1) if cp_rank > 0 else -1
    next_rank = (
        _global_peer_rank(cp_mesh, cp_rank + 1) if cp_rank < cp_size - 1 else -1
    )
    return _GDNStateP2PTritonFunction.apply(
        query,
        key,
        value,
        g,
        beta,
        cp_rank,
        cp_size,
        cp_mesh.get_group(),
        prev_rank,
        next_rank,
        sequence_info,
    )
