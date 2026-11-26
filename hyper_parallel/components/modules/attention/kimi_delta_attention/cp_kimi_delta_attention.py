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
"""KDA layer CP scheduling, state protocols and backend-specific state autograd."""
from __future__ import annotations
from functools import lru_cache
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

from hyper_parallel.components.functional.kimi_delta_attention import (
    apply_kda_state_gradient_summary, apply_kda_state_summary,
)
from hyper_parallel.components.functional.kimi_delta_attention_ascendc import (
    load_kda_native_ops, prepare_kda_native, finish_kda_native_forward, restore_kda_native_backward,
    prepare_kda_native_backward, finish_kda_native_backward,
)
from hyper_parallel.components.functional.kimi_delta_attention_triton import (
    get_fla_kda_staged_ops, validate_triton_kda_inputs, validate_kda_state_summary,
    kda_state_gradient_summary_from_prepared, kda_state_summary_forward_from_prepared,
)
from hyper_parallel.components.functional.linear_attention_convolution import fused_causal_conv1d, packed_causal_conv1d
from hyper_parallel.components.modules.attention.linear_attention_utils.cp_state_exchange import (
    NativeStateAllGather, NativeStateExchange as _NativeStateExchange,
    native_state_intervals as _native_ag_intervals,
)
from hyper_parallel.components.modules.attention.linear_attention_utils.layer_adapter import (
    prepare_linear_attention_input,
)
from hyper_parallel.components.modules.attention.linear_attention_utils.cp_convolution import (
    causal_convolutions_with_halo,
)
from hyper_parallel.components.modules.attention.linear_attention_utils.parallel_layout import (
    validate_linear_attention_layout, split_linear_attention_mesh as split_kda_mesh,
)
from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import (
    PackedSequenceInfo, resolve_packed_sequence, plan_packed_summary_exchange, local_chunk_metadata,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.distributed.context_parallel.collectives import exchange_sequence_heads
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from .kimi_delta_attention import KimiDeltaAttentionModule
from .cp_recursive_doubling import KDAStateRecursiveDoubling


class CPKimiDeltaAttentionModule(KimiDeltaAttentionModule):
    """KDA pure Ulysses or Ulysses plus a state protocol, sharing local layer math."""

    def __init__(self, *, config: Any = None, module: torch.nn.Module | None = None,
                 module_fqn: str = "", context: Any = None, backend: str = "eager",
                 conv_backend: str | None = None, chunk_size: int = 64, input_layout: str = "BSH",
                 ulysses_degree: int | None = 1, state_cp_method: str = "p2p",
                 allgather_group_size: int = 1, p2p_head_chunks: int = 1,
                 reuse_cp_state_on_recompute: bool = False) -> None:
        """Record scheduling choices; do not construct groups or retain an active invocation."""
        del module_fqn, context
        # Preserve the established post-A2A convolution for the degree=None request.
        # Topology is tracked separately as CP = ulysses_size * state_size.
        self.convolve_after_exchange = ulysses_degree is None
        if conv_backend is None and not self.convolve_after_exchange:
            conv_backend = "ascendc" if backend == "ascendc" else "eager"
        super().__init__(config=config, module=module,
                         backend=backend, conv_backend=conv_backend, chunk_size=chunk_size, input_layout=input_layout)
        self.ulysses_degree = ulysses_degree
        self.state_cp_method = state_cp_method
        self.allgather_group_size = allgather_group_size
        self.p2p_head_chunks = p2p_head_chunks
        if reuse_cp_state_on_recompute and backend != "ascendc":
            raise NotImplementedError("Checkpoint entry caching currently requires the Ascend C state backend.")
        self.reuse_cp_state_on_recompute, self._cp_entry_key = reuse_cp_state_on_recompute, object()
        self.cp_rank, self.cp_size = 0, 1
        self.cp_context = None
        self.state_mesh = None
        self.state_cp = None
        self.ulysses_mesh = None
        self.ulysses_rank = 0
        self.ulysses_size, self.state_size = 1, 1

    def prepare_input(self, hidden_states: torch.Tensor, options: dict) -> tuple:
        """Resolve real document boundaries against this CP rank's physical shard.

        Args:
            hidden_states: Local dense or flat hidden activations.
            options: Explicit backend options or invocation document metadata.
        """
        return prepare_linear_attention_input(hidden_states, options, self.cp_rank, self.cp_size)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare the stage containing both token exchange and nested state exchange."""
        return ParallelModuleSpec("attention.kda.cp", "cp",
                                  {"execute_attention_pipeline": ("hidden_states", "info")},
                                  validate=validate_linear_attention_layout)

    def bind_parallel(self, binding: Any) -> None:
        """Borrow the chronological mesh and prebuild the selected U/state groups.

        Args:
            binding: Resolved parallel binding containing the CP context.
        """
        if binding.cp_context is None:
            raise ValueError("KDA CP requires CPContext")
        self.cp_context = binding.cp_context
        self.cp_rank, self.cp_size = self.cp_context.rank, self.cp_context.size
        degree = self.cp_size if self.convolve_after_exchange else self.ulysses_degree
        if (not isinstance(degree, int) or isinstance(degree, bool)) or degree < 1 or self.cp_size % degree:
            raise ValueError("KDA Ulysses degree must divide CP size")
        if self.num_heads % degree or self.num_v_heads % degree:
            raise ValueError("KDA Ulysses degree must divide query and value heads")
        self.ulysses_size, self.state_size = degree, self.cp_size // degree
        if self.state_size == 1:
            # Pure U has no state exchange. Preserve the requested convolution position.
            self.ulysses_mesh = self.cp_context.mesh if degree > 1 else None
            self.ulysses_rank = self.cp_rank
            return
        self.state_mesh = self.cp_context.mesh
        if degree > 1:
            hybrid = split_kda_mesh(self.state_mesh, degree, ("kda_state", "kda_ulysses"))
            self.ulysses_mesh = hybrid["kda_ulysses"]
            self.ulysses_rank = self.ulysses_mesh.get_local_rank()
            self.state_mesh = hybrid["kda_state"]
        self.state_cp = build_kda_state_cp(self.state_mesh, self.state_cp_method, self.allgather_group_size)

    def _run_state_backend(self, operands: tuple, info: PackedSequenceInfo | None,
                           a_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
        """Inject states inside backend autograd, reusing the local path at state degree one."""
        if self.state_size == 1:
            return self.run_local_core(operands, info, a_log, dt_bias)
        size, rank = self.state_size, self.state_mesh.get_local_rank()
        if self.backend == "ascendc":
            # FSDP may expose BF16 compute weights; native gates require FP32, as in the local core.
            return ascendc_state_kda(
                *operands, a_log=a_log.float(), dt_bias=dt_bias.float(), lower_bound=self.lower_bound,
                state_cp=self.state_cp, cp_mesh=self.state_mesh, sequence_info=info,
                p2p_head_chunks=self.p2p_head_chunks,
                reuse_cp_state_on_recompute=self.reuse_cp_state_on_recompute, cache_key=self._cp_entry_key)
        if self.backend != "triton":
            raise NotImplementedError("KDA state CP requires Triton or Ascend C; eager supports local/pure Ulysses")
        ranks = self.state_mesh.rank_list
        return triton_state_kda(
            *operands, a_log=a_log.float(), dt_bias=dt_bias.float(), lower_bound=self.lower_bound,
            chunk_size=self.chunk_size, safe_gate=self.safe_gate,
            cp_group=self.state_mesh.get_group(), cp_rank=rank, cp_size=size,
            prev_rank=int(ranks[max(rank - 1, 0)]), next_rank=int(ranks[min(rank + 1, size - 1)]),
            state_cp=self.state_cp, sequence_info=info)

    def run_state_core(self, operands: tuple, info: PackedSequenceInfo | None,
                       a_log: torch.Tensor | None = None, dt_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Apply optional Ulysses to projected inputs and restore their external token layout.

        Args:
            operands: Projected and convolved Q/K/V plus raw gate and beta.
            info: Real document boundaries for this rank's physical token interval.
            a_log: Optional parameter view used by the projected-stage caller.
            dt_bias: Optional parameter view used by the projected-stage caller.
        """
        a_log = self.A_log if a_log is None else a_log
        dt_bias = self.dt_bias if dt_bias is None else dt_bias
        if self.ulysses_mesh is None:
            return self._run_state_backend(operands, info, a_log, dt_bias)
        inputs = tuple(exchange_sequence_heads(tensor, self.ulysses_mesh, split_dim=2, concat_dim=1)
                       for tensor in operands)
        heads = self.num_v_heads // self.ulysses_size
        a_log = a_log.reshape(self.num_v_heads).narrow(0, self.ulysses_rank * heads, heads).contiguous()
        dt_bias = dt_bias.reshape(self.num_v_heads, self.head_k_dim).narrow(
            0, self.ulysses_rank * heads, heads).contiguous()
        info = None if info is None else info.ulysses(self.ulysses_size, self.ulysses_rank)
        output = self._run_state_backend(inputs, info, a_log, dt_bias)
        return exchange_sequence_heads(output, self.ulysses_mesh, split_dim=1, concat_dim=2)

    def _exchange_raw_inputs(self, operands: tuple) -> tuple:
        if self.cp_size == 1:
            return operands
        widths = (self.key_dim, self.key_dim, self.value_dim,
                  self.num_v_heads * self.head_k_dim, self.num_v_heads)
        widths = tuple(width // self.cp_size for width in widths)
        chunks = [tensor.split(width, dim=-1) for tensor, width in zip(operands, widths)]
        packed = torch.cat([torch.cat(parts, dim=-1) for parts in zip(*chunks)], dim=-1).contiguous()
        packed = exchange_sequence_heads(packed, self.cp_context.mesh, split_dim=2, concat_dim=1)
        return packed.split(widths, dim=-1)

    def _convolve_local_heads(self, operands: tuple, info: PackedSequenceInfo | None) -> tuple:
        """Convolve each local head shard with its matching source weight slice."""
        query, key, value, gate, beta = operands
        if not self.use_short_conv:
            return *(F.silu(tensor) for tensor in (query, key, value)), gate, beta
        convolved = []
        for tensor, conv in zip((query, key, value), (self.q_conv1d, self.k_conv1d, self.v_conv1d)):
            channels = tensor.shape[-1]
            start = self.cp_rank * channels
            weight = conv.weight.narrow(0, start, channels)
            bias = None if conv.bias is None else conv.bias.narrow(0, start, channels)
            if info is not None:
                output = packed_causal_conv1d(tensor, conv, info, weight=weight, bias=bias, backend=self.conv_backend)
            elif self.conv_backend != "eager":
                output = fused_causal_conv1d(tensor, weight, bias, backend=self.conv_backend,
                                             dilation=conv.dilation[0])
            else:
                output = torch.conv1d(tensor.transpose(1, 2), weight, bias, stride=conv.stride,
                                  padding=conv.padding, dilation=conv.dilation, groups=channels)
                output = F.silu(output[:, :, :tensor.shape[1]].transpose(1, 2))
            convolved.append(output)
        return *convolved, gate, beta

    def execute_attention_pipeline(self, hidden_states: torch.Tensor, info: PackedSequenceInfo | None) -> tuple:
        """Preserve pure-U post-exchange convolution and state-CP pre-exchange halos.

        Args:
            hidden_states: Local hidden activations in BSH or flat TH layout.
            info: Immutable packed-sequence metadata, or None for dense input.
        """
        if self.cp_context is None:
            raise RuntimeError("Bind KDA CP resources before forward")
        if not self.training:
            raise NotImplementedError("KDA CP currently supports training only")
        projected = self.project_inputs(hidden_states)
        if self.convolve_after_exchange:
            gate, beta = self.project_state_gates(hidden_states)
            output_gate = self.project_output_gate(hidden_states)
            operands = (*projected, gate, beta)
            exchanged_info = None if info is None else info.ulysses(self.cp_size, self.cp_rank)
            operands = self._exchange_raw_inputs(operands)
            operands = self.shape_inputs(self._convolve_local_heads(operands, exchanged_info), self.cp_size)
            heads = self.num_v_heads // self.cp_size
            a_log = self.A_log.reshape(self.num_v_heads).narrow(0, self.cp_rank * heads, heads)
            dt_bias = self.dt_bias.reshape(self.num_v_heads, self.head_k_dim).narrow(0, self.cp_rank * heads, heads)
            output = self._run_state_backend(operands, exchanged_info, a_log, dt_bias)
            return exchange_sequence_heads(output, self.cp_context.mesh, split_dim=1, concat_dim=2), output_gate
        query, key, value = projected
        if self.use_short_conv:
            query, key, value = causal_convolutions_with_halo(
                (query, key, value), (self.q_conv1d, self.k_conv1d, self.v_conv1d),
                self.cp_context.mesh, self.cp_rank, self.cp_size, sequence_info=info, backend=self.conv_backend)
        else:
            query, key, value = (F.silu(tensor) for tensor in (query, key, value))
        gate, beta = self.project_state_gates(hidden_states)
        output_gate = self.project_output_gate(hidden_states)
        output = self.run_state_core(self.shape_inputs((query, key, value, gate, beta)), info)
        return output, output_gate


def _validate_mesh_order(mesh: DeviceMesh) -> None:
    """Prevent collective slots from being mistaken for chronological positions."""
    group_ranks = tuple(dist.get_process_group_ranks(mesh.get_group()))
    if tuple(mesh.rank_list) != group_ranks:
        raise ValueError("KDA mesh sequence order must match its communication group order.")


def build_kda_state_cp(mesh: DeviceMesh, protocol: str, allgather_group_size: int) -> object | None:
    """Construct a state-CP executor holding only communication metadata.

    Args:
        mesh: One-dimensional mesh ordered along the state sequence.
        protocol: p2p, allgather, alltoallv, grouped_allgather_p2p, or recursive_doubling.
        allgather_group_size: Consecutive ranks per local gather, only for the grouped protocol.

    Returns:
        State executor with an invocation-owned cache, or None for the P2P/local path.
    """
    if protocol not in ("p2p", "allgather", "alltoallv", "grouped_allgather_p2p", "recursive_doubling"):
        raise ValueError(f"Unknown KDA state_cp_method: {protocol!r}.")
    if not isinstance(allgather_group_size, int) or isinstance(allgather_group_size, bool) or allgather_group_size < 1:
        raise ValueError("KDA allgather_group_size must be a positive integer.")
    if protocol != "grouped_allgather_p2p" and allgather_group_size != 1:
        raise ValueError("KDA allgather_group_size is only valid for grouped_allgather_p2p.")
    if protocol == "p2p":
        return None
    ranks = tuple(mesh.rank_list)
    rank, size = mesh.get_local_rank(), mesh.size()
    if protocol == "recursive_doubling":
        return KDAStateRecursiveDoubling(mesh.get_group(), rank, ranks) if size > 1 else None
    if protocol in ("allgather", "alltoallv"):
        _validate_mesh_order(mesh)
        executor = KDAStateAllToAllV if protocol == "alltoallv" else KDAStateAllGather
        return executor(mesh.get_group(), rank, size)
    if size % allgather_group_size:
        raise ValueError("KDA allgather_group_size must divide the state CP size.")
    if allgather_group_size == 1:
        return None
    if allgather_group_size == size:
        _validate_mesh_order(mesh)
        return KDAStateAllGather(mesh.get_group(), rank, size)
    split = split_kda_mesh(mesh, allgather_group_size, ("kda_outer", "kda_group"))
    return KDAStateAllGatherP2P(
        mesh.get_group(), split["kda_group"].get_group(), rank, ranks, allgather_group_size)


def _gather_maps(state: torch.Tensor, matrix: torch.Tensor, group: Any, size: int) -> torch.Tensor:
    """Pack validated local summaries for one invocation; the caller owns the gathered storage."""
    packed = torch.cat((state, matrix), dim=-1)
    output = packed.new_empty((size * packed.shape[0], *packed.shape[1:]))
    dist.all_gather_into_tensor(output, packed, group=group)
    return output.view(size, *packed.shape)


class KDAStateAllGather(NativeStateAllGather):
    """Gather S/M forward and G/M backward; only local M survives forward."""

    def forward(self, state: torch.Tensor, matrix: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return the incoming state and this invocation's original local M.

        Args:
            state: Local FP32 offset S [B,H,128,128].
            matrix: Local forward transfer M of the same shape.

        Returns:
            This rank's incoming recurrent state and a one-tensor local M cache.
        """
        validate_kda_state_summary(state, matrix)
        # The native driver uses transport only and must not import Triton.
        from hyper_parallel.components.functional.kimi_delta_attention_triton import (  # pylint: disable=import-outside-toplevel
            merge_affine_summaries,
        )
        maps = _gather_maps(state, matrix, self.group, self.size)
        return merge_affine_summaries(maps[..., :128], maps[..., 128:], self.rank, True), (matrix,)

    def backward(self, gradient: torch.Tensor, cache: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Collect new G with the saved local M and recover the outgoing gradient.

        Args:
            gradient: Local FP32 offset G [B,H,128,128].
            cache: One-tensor tuple containing this invocation's local forward M.

        Returns:
            This rank's outgoing recurrent-state gradient.
        """
        matrix, = cache
        validate_kda_state_summary(gradient, matrix)
        # Load optional Triton math only for this protocol implementation.
        from hyper_parallel.components.functional.kimi_delta_attention_triton import (  # pylint: disable=import-outside-toplevel
            merge_affine_summaries,
        )
        maps = _gather_maps(gradient, matrix, self.group, self.size)
        return merge_affine_summaries(maps[..., :128], maps[..., 128:], self.rank, False)


class KDAStateAllToAllV(KDAStateAllGather):
    """Collect only packed-document dependencies; dense calls retain AllGather."""

    protocol = "alltoallv"

    def bind_sequence(self, info: PackedSequenceInfo, *, state_shape: tuple) -> object:
        """Bind an immutable sparse plan to this invocation, not the shared executor.

        Args:
            info: Real document offsets and this state rank's local interval.
            state_shape: Local state shape, with physical packed batch size one.
        """
        if state_shape[0] != 1:
            raise ValueError("Sparse packed summaries require physical batch size one.")
        plan = plan_packed_summary_exchange(info, self.rank, self.size)
        return _SparseAllGatherInvocation(self.group, plan)


class _SparseAllGatherInvocation:
    """Own only a plan; transport buffers remain local to each forward/backward."""

    def __init__(self, group: Any, plan: Any) -> None:
        self.group, self.plan = group, plan

    def exchange_maps(self, packed: torch.Tensor, forward: bool) -> tuple[torch.Tensor, int]:
        """Receive compact chronological maps and one unused local-rank slot.

        The sentinel preserves existing merge-kernel rank semantics without a
        full-group restore copy. Raw collectives are inside the attention's
        custom autograd, whose reverse summary exchange owns differentiation.

        Args:
            packed: One rank's [B,H,128,256] S/M or reverse G/M (native G/N).
            forward: Select forward edges; backward uses the exact transposed edges.
        """
        sends, receives = self.plan.send_counts, self.plan.recv_counts
        if not forward:
            sends, receives = receives, sends
        send_count, recv_count = sum(sends), sum(receives)
        maps = packed.new_empty((recv_count + 1, *packed.shape))
        rank = recv_count if forward else 0
        if self.plan.active:
            outgoing = packed.unsqueeze(0).expand(send_count, *packed.shape).contiguous()
            incoming = maps[:recv_count] if forward else maps[1:]
            dist.all_to_all_single(incoming, outgoing, output_split_sizes=list(receives),
                                   input_split_sizes=list(sends), group=self.group)
        return maps, rank

    def forward(self, state: torch.Tensor, matrix: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """Merge only the head document's earlier fragments and cache local M."""
        validate_kda_state_summary(state, matrix)
        # Triton math is optional for native transport-only users.
        from hyper_parallel.components.functional.kimi_delta_attention_triton import (  # pylint: disable=import-outside-toplevel
            merge_affine_summaries,
        )
        maps, rank = self.exchange_maps(torch.cat((state, matrix), dim=-1), True)
        initial = merge_affine_summaries(maps[..., :128], maps[..., 128:], rank, True)
        return initial, (matrix,)

    def backward(self, gradient: torch.Tensor, cache: tuple) -> torch.Tensor:
        """Send head gradients backward and merge only the tail document's suffix."""
        matrix, = cache
        validate_kda_state_summary(gradient, matrix)
        # Keep the native path free of an optional Triton import.
        from hyper_parallel.components.functional.kimi_delta_attention_triton import (  # pylint: disable=import-outside-toplevel
            merge_affine_summaries,
        )
        maps, rank = self.exchange_maps(torch.cat((gradient, matrix), dim=-1), False)
        return merge_affine_summaries(maps[..., :128], maps[..., 128:], rank, False)


class KDAStateAllGatherP2P:
    """Use group-local gathers and an owner chain, retaining only each local M."""

    def __init__(self, group: Any, intra_group: Any, rank: int,
                 ranks: tuple[int, ...], width: int) -> None:
        """Bind ranks already validated against the process-group order by the factory."""
        size = len(ranks)
        if not isinstance(width, int) or isinstance(width, bool) or width < 2 or size % width:
            raise ValueError("Grouped AllGather width must be >= 2 and divide the CP size.")
        if not 0 <= rank < size or len(set(ranks)) != size:
            raise ValueError("Grouped AllGather requires unique global ranks and a valid local rank.")
        self.group, self.intra, self.rank, self.width = group, intra_group, rank, width
        self.ranks = tuple(ranks)
        self.block, self.local = divmod(rank, width)
        self.blocks = size // width
        self.owner = self.block * width + width - 1

    def _receive(self, state: torch.Tensor, forward: bool) -> tuple[torch.Tensor, Any]:
        """Post the owner receive before gathering local summaries."""
        incoming = torch.zeros_like(state)
        recv_handle = None
        peer = self.owner - self.width if forward else self.owner + self.width
        if self.rank == self.owner and 0 <= peer < len(self.ranks):
            recv_handle = dist.irecv(incoming, src=self.ranks[peer], group=self.group)
        return incoming, recv_handle

    def forward(self, state: torch.Tensor, matrix: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Relay ordered group scans and restore this rank's incoming state.

        Args:
            state: Local FP32 offset S [B,H,128,128].
            matrix: Local forward transfer M of the same shape.

        Returns:
            This rank's incoming recurrent state and a one-tensor local M cache.
        """
        validate_kda_state_summary(state, matrix)
        # The native driver borrows groups without loading these Triton scans.
        from hyper_parallel.components.functional.kimi_delta_attention_triton import (  # pylint: disable=import-outside-toplevel
            scan_kda_affine_maps,
        )
        incoming, recv_handle = self._receive(state, True)
        maps = _gather_maps(state, matrix, self.intra, self.width)
        send_handle, send_state = None, None
        result = None
        if self.rank == self.owner:
            if recv_handle is not None:
                recv_handle.wait()
            if self.block + 1 < self.blocks:
                send_state, result = scan_kda_affine_maps(maps, incoming, range(self.width), True,
                                                        prefix_count=self.local)
                send_handle = dist.isend(send_state, dst=self.ranks[self.owner + self.width], group=self.group)
        if self.block > 0:
            dist.broadcast(incoming, src=self.ranks[self.owner], group=self.intra)
        if result is None:
            result, _ = scan_kda_affine_maps(maps, incoming, range(self.local), True)
        if send_handle is not None:
            send_handle.wait()
        return result, (matrix,)

    def backward(self, gradient: torch.Tensor, cache: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Regather G/M and reverse the owner chain with ordered state feedback.

        Args:
            gradient: Local FP32 offset G [B,H,128,128].
            cache: One-tensor tuple containing this invocation's local forward M.

        Returns:
            This rank's outgoing recurrent-state gradient.
        """
        matrix, = cache
        validate_kda_state_summary(gradient, matrix)
        # Keep the optional Triton implementation local to the selected path.
        from hyper_parallel.components.functional.kimi_delta_attention_triton import (  # pylint: disable=import-outside-toplevel
            scan_kda_affine_maps,
        )
        incoming, recv_handle = self._receive(gradient, False)
        maps = _gather_maps(gradient, matrix, self.intra, self.width)
        send_handle, send_state = None, None
        if self.rank == self.owner:
            if recv_handle is not None:
                recv_handle.wait()
            if self.block > 0:
                send_state, _ = scan_kda_affine_maps(maps, incoming, range(self.width - 1, -1, -1), False)
                send_handle = dist.isend(send_state, dst=self.ranks[self.owner - self.width], group=self.group)
        if self.block + 1 < self.blocks:
            dist.broadcast(incoming, src=self.ranks[self.owner], group=self.intra)
        result, _ = scan_kda_affine_maps(maps, incoming, range(self.width - 1, self.local, -1), False)
        if send_handle is not None:
            send_handle.wait()
        return result


def _validate_state_cp_inputs(
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
    """Validate the fixed-shape KDA state CP backend before communication starts."""
    validate_triton_kda_inputs(
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
        raise ValueError(f"Invalid KDA state CP rank {cp_rank} for cp_size {cp_size}.")


def _prepare_packed_kda_summary(kg, w, u, gate, sequence_info, state_cp, send_next, chunk_size):
    """Skip summaries with no consumer when the invocation supports absent tensors."""
    if state_cp is None and not send_next:
        return None, None
    state_shape = (kg.shape[0], u.shape[2], kg.shape[-1], u.shape[-1])
    transition = None
    if getattr(state_cp, "needs_state_summary", True):
        tail = sequence_info.tail
        state, transition = kda_state_summary_forward_from_prepared(
            kg[:, tail], w[:, tail], u[:, tail], gate[:, tail], chunk_size=chunk_size,
            output_transition=(sequence_info.transfer_active
                               and getattr(state_cp, "needs_transition_summary", True)),
        )
    elif getattr(state_cp, "allows_absent_summaries", False):
        return None, None
    else:
        state = kg.new_zeros(state_shape, dtype=torch.float32)
    if state_cp is not None and transition is None:
        # Shape placeholders preserve existing collective validation; no M kernel runs.
        transition = kg.new_zeros((*state_shape[:-1], kg.shape[-1]), dtype=torch.float32)
    return state, transition


def _sequence_metadata(sequence_info, device, chunk_size):
    """Materialize one immutable packed plan for both intra-chunk and recurrent stages."""
    if sequence_info is None:
        return None, {}, {}
    plan = sequence_info.materialize(device, chunk_size)
    varlen = plan.kwargs()
    return plan, varlen, dict(varlen, chunk_offsets=plan.chunk_offsets)


def _exchange_forward(kg, w, u, gate, sequence_info, state_cp, send_next, chunk_size,
                      cp_rank, cp_size, initial_state, recv_handle, next_rank, cp_group):
    """Produce only consumed summaries and keep an outgoing buffer alive until its send completes."""
    state_ext = None
    transition = None
    if sequence_info is not None:
        state_ext, transition = _prepare_packed_kda_summary(
            kg, w, u, gate, sequence_info, state_cp, send_next, chunk_size,
        )
    else:
        if state_cp is None and cp_rank == 0 and cp_size > 1:
            # Zero entry uses S directly, and rank 0 has no reverse-state recipient.
            state_ext, transition = kda_state_summary_forward_from_prepared(
                kg, w, u, gate, chunk_size=chunk_size, output_transition=False,
            )
        elif state_cp is not None or cp_rank < cp_size - 1:
            state_ext, transition = kda_state_summary_forward_from_prepared(kg, w, u, gate, chunk_size=chunk_size)
    # A reset makes the outgoing summary independent of the incoming state.
    # Wait before apply only when it actually reads that state; local h still waits below.
    if recv_handle is not None and (sequence_info is None or sequence_info.transfer_active):
        recv_handle.wait()
        recv_handle = None

    send_handle = None
    send_state = None
    state_cache = (transition,) if transition is not None else ()
    if state_cp is not None:
        initial_state, state_cache = state_cp.forward(state_ext, transition)
        transition = None
    elif send_next:
        summary_initial = initial_state if transition is not None else None
        final_state = apply_kda_state_summary(state_ext, transition, summary_initial)
        send_state = final_state.contiguous()
        send_handle = dist.isend(send_state, dst=next_rank, group=cp_group)

    if recv_handle is not None:
        recv_handle.wait()
    return initial_state, state_cache, send_handle, send_state


def _exchange_backward(ctx, query_gated, key_gated, w, gate, grad_output, grad_value_local,
                       state_shape, state_cache, grad_final_state, recv_handle):
    """Produce G and reverse the state exchange without changing local backward arithmetic."""
    sequence_info = ctx.sequence_info
    grad_state_ext = None
    if ctx.state_cp is not None or ctx.receive_previous:
        if sequence_info is not None and not sequence_info.continues_previous:
            if not getattr(ctx.state_cp, "allows_absent_summaries", False):
                grad_state_ext = query_gated.new_zeros(state_shape, dtype=torch.float32)
        else:
            head = slice(None) if sequence_info is None else sequence_info.head
            grad_state_ext = kda_state_gradient_summary_from_prepared(
                query_gated[:, head], key_gated[:, head], w[:, head], gate[:, head],
                grad_output[:, head], grad_value_local[:, head], ctx.scale, chunk_size=ctx.chunk_size,
            )
    if recv_handle is not None and (sequence_info is None or sequence_info.transfer_active):
        recv_handle.wait()
        recv_handle = None
    send_handle = None
    send_state_gradient = None
    if ctx.state_cp is not None:
        grad_final_state = ctx.state_cp.backward(grad_state_ext, state_cache)
    elif ctx.receive_previous:
        transition = state_cache[0] if state_cache else None
        summary_final_gradient = grad_final_state if transition is not None else None
        grad_initial_state = apply_kda_state_gradient_summary(grad_state_ext, transition, summary_final_gradient)
        send_state_gradient = grad_initial_state.contiguous()
        send_handle = dist.isend(send_state_gradient, dst=ctx.prev_rank, group=ctx.cp_group)

    if recv_handle is not None:
        recv_handle.wait()
    return grad_final_state, send_handle, send_state_gradient


class _KDAStateCPFunction(torch.autograd.Function):
    """Run shared local KDA and autograd with the selected state exchange."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ,too-many-arguments,too-many-locals
        ctx: Any,
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        gate_raw: torch.Tensor, beta_raw: torch.Tensor,
        a_log: torch.Tensor, dt_bias: torch.Tensor,
        scale: float, lower_bound: float, chunk_size: int, safe_gate: bool,
        cp_group: Any, prev_rank: int, next_rank: int, cp_rank: int, cp_size: int,
        state_cp: Any,
        sequence_info: Optional[PackedSequenceInfo],
    ) -> torch.Tensor:
        """Prepare local KDA, propagate its state, and produce token outputs.

        Args:
            ctx: Autograd context holding tensors and communication metadata.
            query: Local query tensor [B,T,H,K].
            key: Matching local key tensor.
            value: Local value tensor [B,T,HV,V].
            gate_raw: Unactivated per-token gate logits.
            beta_raw: Unactivated per-token beta logits.
            a_log: Decay parameter with one value per value head.
            dt_bias: Learned bias with one value per gate channel.
            scale: Optional attention scaling factor.
            lower_bound: Lower bound for the activated gate.
            chunk_size: Tokens per local recurrence chunk.
            safe_gate: Use the staged lower-bounded gate implementation.
            cp_group: Precreated chronological CP process group.
            prev_rank: Global rank of the preceding sequence partition.
            next_rank: Global rank of the following sequence partition.
            cp_rank: Chronological index within the CP group.
            cp_size: Number of state partitions.
            state_cp: Executor returning an invocation cache; None selects P2P.
        """
        ops = get_fla_kda_staged_ops()
        bind_sequence = getattr(state_cp, "bind_sequence", None)
        if sequence_info is not None and bind_sequence is not None:
            state_cp = bind_sequence(
                sequence_info, state_shape=(query.shape[0], value.shape[2], query.shape[-1], value.shape[-1]))
        plan, varlen, recurrent_varlen = _sequence_metadata(sequence_info, query.device, chunk_size)

        # FLA consumes base-2 log gates: exp(x) = exp2(x / ln(2)).
        # Use the same conversion when recomputing gates in backward.
        rcp_ln2 = 1.4426950408889634

        query, key, value, gate_raw, beta_raw = (
            tensor.contiguous() for tensor in (query, key, value, gate_raw, beta_raw))
        query, query_rstd = ops.l2norm_fwd(query)
        key, key_rstd = ops.l2norm_fwd(key)
        beta = ops.fused_beta_sigmoid(beta_raw)
        gate = ops.kda_gate_chunk_cumsum(
            g=gate_raw, A_log=a_log, dt_bias=dt_bias, scale=rcp_ln2, chunk_size=chunk_size,
            lower_bound=lower_bound, **varlen,
        )

        state_shape = (query.shape[0], value.shape[2], query.shape[-1], value.shape[-1])
        receive_previous = cp_rank > 0 and (sequence_info is None or sequence_info.continues_previous)
        send_next = cp_rank < cp_size - 1 and (sequence_info is None or sequence_info.continues_next)
        initial_state = None
        recv_handle = None
        if state_cp is None and receive_previous:
            initial_state = torch.empty(state_shape, device=query.device, dtype=torch.float32)
            recv_handle = dist.irecv(initial_state, src=prev_rank, group=cp_group)

        w, u, _, kg, attention_qk, attention_kk = ops.chunk_kda_fwd_intra(
            q=query, k=key, v=value, gk=gate, beta=beta, scale=scale, chunk_size=chunk_size, safe_gate=safe_gate,
            disable_recompute=True, **varlen,
        )
        # The gated query is unused here and is recomputed in backward.
        del _
        initial_state, state_cache, send_handle, send_state = _exchange_forward(
            kg, w, u, gate, sequence_info, state_cp, send_next, chunk_size,
            cp_rank, cp_size, initial_state, recv_handle, next_rank, cp_group)
        kernel_initial_state = initial_state if sequence_info is None else sequence_info.first_state(initial_state)
        states, value_new, _ = ops.chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=gate, initial_state=kernel_initial_state,
            output_final_state=False, chunk_size=chunk_size, **recurrent_varlen,
        )
        # Local h was their last forward consumer on this stream.
        del w, u, kg, _
        output = ops.chunk_gla_fwd_o_gk(
            q=query, v=value_new, g=gate, A=attention_qk, h=states, scale=scale, chunk_size=chunk_size, **varlen,
        )
        if send_handle is not None:
            send_handle.wait()
        del send_state

        saved_initial_state = initial_state
        if saved_initial_state is None:
            saved_initial_state = query.new_empty(0, dtype=torch.float32)
        packed_tensors = () if plan is None else (
            plan.cu_seqlens, plan.chunk_indices, plan.chunk_offsets)
        ctx.save_for_backward(
            query, query_rstd, key, key_rstd, value, gate_raw, beta_raw, a_log, dt_bias, attention_qk,
            attention_kk, saved_initial_state, *state_cache, *packed_tensors,
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
        ctx.state_cp = state_cp
        ctx.state_cache_count = len(state_cache)
        ctx.receive_previous = receive_previous
        ctx.send_next = send_next
        ctx.sequence_info = sequence_info
        return output.to(value.dtype)

    @staticmethod
    def backward(  # pylint: disable=arguments-differ,too-many-locals
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Reverse the state wavefront, then finish the fused local backward.

        Args:
            ctx: Autograd context holding tensors and communication metadata.
            grad_output: Upstream gradient for the local token output.
        """
        ops = get_fla_kda_staged_ops()

        # FLA consumes base-2 log gates: exp(x) = exp2(x / ln(2)).
        # Use the same conversion when recomputing gates in backward.
        rcp_ln2 = 1.4426950408889634

        (
            query, query_rstd, key, key_rstd, value, gate_raw, beta_raw, a_log, dt_bias,
            attention_qk, attention_kk, saved_initial_state, *saved_tail,
        ) = ctx.saved_tensors
        state_cache = tuple(saved_tail[:ctx.state_cache_count])
        packed_tensors = saved_tail[ctx.state_cache_count:]
        sequence_info = ctx.sequence_info
        cu_seqlens = chunk_indices = chunk_offsets = None
        if sequence_info is not None:
            cu_seqlens, chunk_indices, chunk_offsets = packed_tensors
        varlen = {} if sequence_info is None else {"cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices}
        recurrent_varlen = dict(varlen)
        if sequence_info is not None:
            recurrent_varlen["chunk_offsets"] = chunk_offsets
        initial_state = saved_initial_state if saved_initial_state.numel() else None
        kernel_initial_state = initial_state if sequence_info is None else sequence_info.first_state(initial_state)
        grad_output = grad_output.contiguous()

        grad_final_state = None
        recv_handle = None
        if ctx.state_cp is None and ctx.send_next:
            grad_final_state = torch.empty(
                (query.shape[0], value.shape[2], query.shape[-1], value.shape[-1]), device=query.device,
                dtype=torch.float32,
            )
            recv_handle = dist.irecv(grad_final_state, src=ctx.next_rank, group=ctx.cp_group)

        beta = ops.fused_beta_sigmoid(beta_raw)
        gate = ops.kda_gate_chunk_cumsum(
            g=gate_raw, A_log=a_log, dt_bias=dt_bias, scale=rcp_ln2, chunk_size=ctx.chunk_size,
            lower_bound=ctx.lower_bound, **varlen,
        )
        w, u, query_gated, key_gated = ops.recompute_w_u_fwd(
            q=query, k=key, v=value, beta=beta, A=attention_kk, gk=gate, **varlen,
        )
        states, value_new, _ = ops.chunk_gated_delta_rule_fwd_h(
            k=key_gated, w=w, u=u, gk=gate, initial_state=kernel_initial_state, output_final_state=False,
            chunk_size=ctx.chunk_size, **recurrent_varlen,
        )
        grad_attention_qk, grad_value_local = ops.chunk_kda_bwd_dav(
            q=query, k=key, v=value_new, do=grad_output, A=attention_qk, scale=ctx.scale,
            chunk_size=ctx.chunk_size, **varlen,
        )
        grad_final_state, send_handle, send_state_gradient = _exchange_backward(
            ctx, query_gated, key_gated, w, gate, grad_output, grad_value_local,
            (query.shape[0], value.shape[2], query.shape[-1], value.shape[-1]),
            state_cache, grad_final_state, recv_handle)
        kernel_grad_final = grad_final_state if sequence_info is None else sequence_info.last_state(grad_final_state)
        grad_states, _, grad_value = ops.chunk_gated_delta_rule_bwd_dhu(
            q=query_gated, k=key_gated, w=w, gk=gate, h0=kernel_initial_state, dht=kernel_grad_final, do=grad_output,
            dv=grad_value_local, scale=ctx.scale, chunk_size=ctx.chunk_size, **recurrent_varlen,
        )
        # These recomputed tensors have no consumers after dhu. Dropping local
        # references lets the allocator reuse their storage on this stream.
        del query_gated, key_gated, w, u, grad_value_local, _
        (
            grad_query, grad_key, grad_value, grad_beta, grad_gate, grad_attention_kk,
        ) = ops.chunk_kda_bwd_wy_dqkg_fused(
            q=query, k=key, v=value, v_new=value_new, g=gate, beta=beta, A=attention_kk, h=states, do=grad_output,
            dh=grad_states, dv=grad_value, scale=ctx.scale, chunk_size=ctx.chunk_size,
            **varlen,
        )
        # Release recomputed states before intra backward allocates its outputs.
        del states, grad_states, value_new
        grad_query, grad_key, grad_beta, grad_gate = ops.chunk_kda_bwd_intra(
            q=query, k=key, g=gate, beta=beta, dAqk=grad_attention_qk, dAkk=grad_attention_kk, dq=grad_query,
            dk=grad_key, db=grad_beta, dg=grad_gate, chunk_size=ctx.chunk_size, safe_gate=ctx.safe_gate,
            **varlen,
        )

        num_query_heads = query.shape[2]
        num_value_heads = value.shape[2]
        if num_value_heads > num_query_heads:
            groups = num_value_heads // num_query_heads
            grad_query = grad_query.view(*grad_query.shape[:2], num_query_heads, groups, grad_query.shape[-1]).sum(
                dim=3,
            )
            grad_key = grad_key.view(*grad_key.shape[:2], num_query_heads, groups, grad_key.shape[-1]).sum(dim=3)

        grad_gate = ops.chunk_local_cumsum(grad_gate, chunk_size=ctx.chunk_size, reverse=True, **varlen)
        grad_gate, grad_a_log, grad_dt_bias = ops.kda_gate_bwd(
            g=gate_raw, A_log=a_log, dt_bias=dt_bias, dyg=grad_gate, lower_bound=ctx.lower_bound,
        )
        grad_beta = ops.fused_beta_sigmoid_bwd(beta_raw, grad_beta)
        grad_query = ops.l2norm_bwd(query, query_rstd, grad_query)
        grad_key = ops.l2norm_bwd(key, key_rstd, grad_key)
        if send_handle is not None:
            send_handle.wait()
        del send_state_gradient

        return (
            grad_query, grad_key, grad_value, grad_gate, grad_beta, grad_a_log, grad_dt_bias, None, None, None,
            None, None, None, None, None, None, None, None,
        )


@torch.compiler.disable
def triton_state_kda(
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
    state_cp: Any = None,
    sequence_info: Optional[PackedSequenceInfo] = None,
) -> torch.Tensor:
    """Run one sequence-sharded KDA segment with a selected state CP executor.

    The historical P2P entry name is retained for existing callers; passing
    state_cp selects gathered or recursive state exchange with the same local ops.

    Args:
        query: Query input [B,T,H,K].
        key: Matching key input.
        value: Value input [B,T,HV,V].
        gate: Per-token gate logits.
        beta: Per-token beta logits.
        a_log: FP32 decay parameter per value head.
        dt_bias: FP32 gate-channel bias.
        cp_group: Existing chronological state process group.
        prev_rank: Global predecessor rank for P2P.
        next_rank: Global successor rank for P2P.
        cp_rank: Chronological index within the state group.
        cp_size: Number of state partitions.
        scale: Optional attention scale.
        lower_bound: Lower bound for the activated gate.
        chunk_size: Tokens per local chunk.
        safe_gate: Enable the staged safe-gate path.
        state_cp: State executor returning an invocation cache; None selects existing P2P.

    Returns:
        Output tokens in the local sequence layout.
    """
    _validate_state_cp_inputs(
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
    if sequence_info is not None:
        if query.shape[0] != 1:
            raise ValueError("Packed state CP requires batch dimension one.")
        resolve_packed_sequence({"packed_sequence_info": sequence_info}, query.shape[1],
                                cp_rank=cp_rank, cp_size=cp_size)
    effective_scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    return _KDAStateCPFunction.apply(
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
        state_cp,
        sequence_info,
    )


@lru_cache(maxsize=1)
def _load_state_ops():
    """Check every native stage before entering the first state collective."""
    ops = load_kda_native_ops()
    required = (
        "pre_process_fwd_kernel_merged", "pre_process_bwd_kernel_merged",
        "merge_fwd_bwd_kernel",
    )
    missing = [name for name in required if not callable(getattr(ops, name, None))]
    if missing:
        raise RuntimeError(f"Native KDA state CP requires these fla-npu APIs: {', '.join(missing)}.")
    return ops


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


class _AscendCKDAState(torch.autograd.Function):
    """Share native local stages; state protocols optionally inject H and D."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                gate: torch.Tensor, beta: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
                scale: float, lower_bound: float, metadata: dict,
                exchange: _NativeStateExchange | None) -> torch.Tensor:
        """Share prepare with S/M and local FwdH, then release gathered maps.

        Args:
            ctx: Per-invocation autograd context.
            query: Raw token-major queries.
            key: Raw token-major keys.
            value: Token-major values.
            gate: Sequence-major gate logits.
            beta: Sequence-major beta logits.
            a_log: FP32 decay parameters.
            dt_bias: FP32 gate bias.
            scale: Attention scale.
            lower_bound: Safe-gate lower bound.
            metadata: Native packed offsets and chunk indices.
            exchange: Optional state protocol for this invocation.
        """
        ops = _load_state_ops()
        inputs = (query, key, value, gate, beta, a_log, dt_bias)
        reception = None if exchange is None else exchange.receive(query, True)
        prepared = prepare_kda_native(inputs, scale, lower_bound, metadata)
        initial = send_handle = outgoing = head = suffix = None
        if exchange is not None:
            tail, head, prefix, suffix = _native_ag_intervals(
                exchange.info, query.shape[1], exchange.rank, exchange.size)
            maps = None
            if exchange.needs_summary(True):
                maps = ops.pre_process_fwd_kernel_merged(
                    prepared.kg, prepared.w, prepared.u, gk=prepared.gk, cu_seqlens=tail)
            initial, send_handle, outgoing = exchange.exchange(ops, maps, reception, True, prefix)
            del maps, reception
            if exchange.info is not None:
                initial = exchange.info.first_state(initial)
        output = finish_kda_native_forward(
            ctx, inputs, prepared, initial, scale, lower_bound, metadata)
        if send_handle is not None:
            send_handle.wait()
        del outgoing
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
        ops = _load_state_ops()
        # Share one unpack across phases, including checkpoint recomputation.
        saved = ctx.saved_tensors
        reception = ctx.exchange.receive(saved[12], False)
        saved = restore_kda_native_backward(ctx, saved)
        gk, weights, qg, kg = saved[4], saved[7], saved[9], saved[10]
        grad, d_aqk, dv_local_head, dq_raw = prepare_kda_native_backward(ctx, saved, grad_output)
        terminal, send_handle, outgoing = _exchange_native_gradient(
            ctx, ops, (gk, qg, kg, weights, grad, dv_local_head), reception)
        del reception
        gradients = finish_kda_native_backward(ctx, saved, grad, d_aqk, dv_local_head, dq_raw, terminal)
        if send_handle is not None:
            send_handle.wait()
        del outgoing
        return (*gradients, None, None, None, None)


def ascendc_state_kda(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    gate: torch.Tensor, beta: torch.Tensor, *, a_log: torch.Tensor,
    dt_bias: torch.Tensor, state_cp: Any, cp_mesh: Any, sequence_info: Any = None,
    scale: float | None = None, lower_bound: float = -5.0,
    p2p_head_chunks: int = 1, reuse_cp_state_on_recompute: bool = False, cache_key: object = None,
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
        p2p_head_chunks: Number of state packets for ordinary P2P; keeps Prepare and summaries unsplit.
        reuse_cp_state_on_recompute: Reuse this forward entry during a HyperParallel checkpoint replay.
        cache_key: Stable caller identity; direct callers otherwise follow invocation order.

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
    metadata = local_chunk_metadata(None if sequence_info is None else sequence_info.local_offsets, query.shape[1])
    scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    tokens = tuple(tensor.contiguous() for tensor in (query, key, value))
    exchange = _NativeStateExchange(cp_mesh, state_cp, sequence_info, query.shape[1], query.shape[2],
                                    p2p_head_chunks, reuse_cp_state_on_recompute=reuse_cp_state_on_recompute,
                                    cache_key=cache_key, algorithm="kda")
    return _AscendCKDAState.apply(
        *tokens, gate.contiguous(), beta.contiguous(), a_log.reshape(-1).clone(),
        dt_bias.reshape(-1, query.shape[-1]).clone(), scale, lower_bound, metadata, exchange)
