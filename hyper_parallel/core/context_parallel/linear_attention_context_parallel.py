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
"""Context parallel execution for Qwen3.5-style Gated DeltaNet layers."""

# This module is the Torch implementation of the public CP style.
# pylint: disable=forbidden-backend-import,missing-public-type-hints
# pylint: disable=missing-public-docstring,not-callable
# PyTorch autograd.Function intentionally defines framework-specific signatures.
# pylint: disable=abstract-method,arguments-differ
from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.context_parallel.context_parallel import (
    _ensure_1d,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.tensor_parallel.style import ParallelStyle
from hyper_parallel.models.modules.linear_attention import (
    chunk_gated_delta_rule,
    is_triton_gdn_available,
    torch_chunk_gated_delta_rule,
)
from hyper_parallel.platform import get_platform


platform = get_platform()


def _global_peer_rank(cp_mesh: DeviceMesh, local_rank: int) -> int:
    """Map a CP-local rank index to its global distributed rank."""
    return int(cp_mesh.rank_list[local_rank])


def _slice_local_cp(
    tensor: torch.Tensor,
    dim: int,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Return this CP rank's contiguous slice along ``dim``."""
    dim_size = tensor.shape[dim]
    if dim_size % cp_size != 0:
        raise ValueError(
            f"linear attention CP expects dim size {dim_size} "
            f"to be divisible by cp_size {cp_size}."
        )
    chunk = dim_size // cp_size
    return tensor.narrow(dim, cp_rank * chunk, chunk)


def _slice_qkv_local_cp(
    tensor: torch.Tensor,
    *,
    key_dim: int,
    value_dim: int,
    dim: int,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Slice a fused ``[Q, K, V]`` tensor on the Q/K/V channel dimension."""
    q, k, v = torch.split(tensor, [key_dim, key_dim, value_dim], dim=dim)
    return torch.cat(
        (
            _slice_local_cp(q, dim, cp_rank, cp_size),
            _slice_local_cp(k, dim, cp_rank, cp_size),
            _slice_local_cp(v, dim, cp_rank, cp_size),
        ),
        dim=dim,
    )


def _local_tensor_at_cp_boundary(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local tensor carried by a CP-boundary input.

    The first supported Qwen3.5 linear-attention CP path keeps decoder-layer
    activations as local sequence shards. If an upstream wrapper passes that
    shard as a DTensor, use its local tensor and continue with the same
    ``[B, S_local, H]`` boundary contract.
    """
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _all_to_all_previous_rank_halo(
    tail: torch.Tensor,
    cp_mesh: DeviceMesh,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Send a convolution halo only to the next rank using differentiable A2AV."""
    if cp_size == 1:
        return torch.zeros_like(tail)

    cp_group = cp_mesh.get_group()
    group_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(cp_group))
    rank_list = tuple(int(rank) for rank in cp_mesh.rank_list)
    rank_to_group_index = {rank: index for index, rank in enumerate(group_ranks)}
    halo_width = tail.shape[1]

    input_splits = [0] * cp_size
    exchange_input = tail.permute(1, 0, 2).contiguous()
    if cp_rank < cp_size - 1:
        input_splits[rank_to_group_index[rank_list[cp_rank + 1]]] = halo_width
    else:
        exchange_input = exchange_input[:0]

    output_splits = [0] * cp_size
    if cp_rank > 0:
        output_splits[rank_to_group_index[rank_list[cp_rank - 1]]] = halo_width

    exchange_output = platform.differentiable_all_to_all_single(
        exchange_input,
        input_splits,
        output_splits,
        group=cp_group,
    )
    if cp_rank == 0:
        return torch.zeros_like(tail) + exchange_output.sum().to(tail.dtype) * 0
    return exchange_output.permute(1, 0, 2).contiguous()


def _causal_conv1d_with_cp_halo(
    mixed_qkv: torch.Tensor,
    conv1d: nn.Conv1d,
    cp_mesh: DeviceMesh,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Run causal depthwise Conv1d with only the previous rank's boundary."""
    kernel_size = conv1d.kernel_size[0]
    dilation = conv1d.dilation[0]
    halo_width = (kernel_size - 1) * dilation
    if halo_width == 0 or cp_size == 1:
        conv_out = conv1d(mixed_qkv.transpose(1, 2))
        return F.silu(conv_out[:, :, : mixed_qkv.shape[1]]).transpose(1, 2)

    if mixed_qkv.shape[1] < halo_width:
        raise ValueError(
            "linear attention CP conv halo requires local_seq_len >= "
            f"{halo_width}, got {mixed_qkv.shape[1]}."
        )

    halo = _all_to_all_previous_rank_halo(
        mixed_qkv[:, -halo_width:, :].contiguous(),
        cp_mesh,
        cp_rank,
        cp_size,
    )
    conv_input = torch.cat((halo, mixed_qkv), dim=1).transpose(1, 2)
    conv_out = F.conv1d(
        input=conv_input,
        weight=conv1d.weight,
        bias=conv1d.bias,
        stride=conv1d.stride,
        padding=0,
        dilation=conv1d.dilation,
        groups=conv1d.groups,
    )
    return F.silu(conv_out).transpose(1, 2)


def _all_gather_stack(
    tensor: torch.Tensor,
    cp_mesh: DeviceMesh,
    cp_size: int,
) -> torch.Tensor:
    """All-gather equal-shaped tensors and stack them on a leading rank dim."""
    if cp_size == 1:
        return tensor.unsqueeze(0)
    return platform.differentiable_all_gather_concat(
        tensor.unsqueeze(0),
        cp_mesh.get_group(),
        cp_size,
        0,
        tuple(int(rank) for rank in cp_mesh.rank_list),
    )


def _l2norm_torch(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Match the pure torch GDN reference l2norm helper."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class _GDNPreparedChunks(NamedTuple):
    """Reusable chunk intermediates shared by state-summary CP modes."""

    initial_dtype: torch.dtype
    query: torch.Tensor
    key: torch.Tensor
    chunk_value: torch.Tensor
    g: torch.Tensor
    decay_mask: torch.Tensor
    k_cumdecay: torch.Tensor
    sequence_length: int
    total_sequence_length: int
    chunk_size: int


def _prepare_gdn_chunks_for_summary(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = False,
) -> _GDNPreparedChunks:
    """Prepare GDN chunk intermediates shared by summary and local output."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm_torch(query, dim=-1, eps=1e-6)
        key = _l2norm_torch(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    sequence_length = key.shape[2]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size

    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for row_idx in range(1, chunk_size):
        row = attn[..., row_idx, :row_idx].clone()
        sub = attn[..., :row_idx, :row_idx].clone()
        attn[..., row_idx, :row_idx] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    chunk_value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    return _GDNPreparedChunks(
        initial_dtype=initial_dtype,
        query=query,
        key=key,
        chunk_value=chunk_value,
        g=g,
        decay_mask=decay_mask,
        k_cumdecay=k_cumdecay,
        sequence_length=sequence_length,
        total_sequence_length=total_sequence_length,
        chunk_size=chunk_size,
    )


def _compute_gdn_state_summary_from_prepared(
    prepared: _GDNPreparedChunks,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``state_out = M @ state_in + S`` from prepared GDN chunks."""
    key = prepared.key
    batch_size, num_heads, _, _, k_head_dim = key.shape
    v_head_dim = prepared.chunk_value.shape[-1]
    eye = torch.eye(k_head_dim, device=key.device, dtype=torch.float32).reshape(
        1, 1, k_head_dim, k_head_dim
    )
    state_ext = torch.zeros(
        batch_size,
        num_heads,
        k_head_dim,
        v_head_dim,
        device=key.device,
        dtype=torch.float32,
    )
    transition = eye.expand(batch_size, num_heads, -1, -1).clone()

    for chunk_idx in range(key.shape[2]):
        key_i = key[:, :, chunk_idx]
        value_i = prepared.chunk_value[:, :, chunk_idx]
        w_i = prepared.k_cumdecay[:, :, chunk_idx]
        g_i = prepared.g[:, :, chunk_idx]
        decay = g_i[:, :, -1].exp()
        key_decay = key_i * (g_i[:, :, -1, None] - g_i).exp()[..., None]

        transition_i = (
            decay[:, :, None, None] * eye
            - key_decay.transpose(-1, -2) @ w_i
        )
        state_ext_i = key_decay.transpose(-1, -2) @ value_i
        state_ext = transition_i @ state_ext + state_ext_i
        transition = transition_i @ transition

    return state_ext, transition


def _checkpoint_gdn_state_summary(
    prepared: _GDNPreparedChunks,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a state summary without retaining its per-chunk autograd graph."""
    if not torch.is_grad_enabled():
        return _compute_gdn_state_summary_from_prepared(prepared)

    def recompute(
        key: torch.Tensor,
        chunk_value: torch.Tensor,
        g: torch.Tensor,
        k_cumdecay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuild a prepared view from explicit checkpoint inputs."""
        checkpoint_prepared = prepared._replace(
            key=key,
            chunk_value=chunk_value,
            g=g,
            k_cumdecay=k_cumdecay,
        )
        return _compute_gdn_state_summary_from_prepared(checkpoint_prepared)

    return checkpoint(
        recompute,
        prepared.key,
        prepared.chunk_value,
        prepared.g,
        prepared.k_cumdecay,
        use_reentrant=False,
        preserve_rng_state=False,
    )


def _run_prepared_gdn_chunks(
    prepared: _GDNPreparedChunks,
    initial_state: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run local GDN output using already prepared chunk intermediates."""
    query = prepared.query
    key = prepared.key
    chunk_value = prepared.chunk_value
    batch_size, num_heads, _, _, k_head_dim = key.shape
    v_head_dim = chunk_value.shape[-1]
    recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            device=chunk_value.device,
            dtype=chunk_value.dtype,
        )
        if initial_state is None
        else initial_state.to(chunk_value)
    )
    core_attn_out = torch.zeros_like(chunk_value)

    for chunk_idx in range(0, prepared.total_sequence_length // prepared.chunk_size):
        q_i = query[:, :, chunk_idx]
        k_i = key[:, :, chunk_idx]
        v_i = chunk_value[:, :, chunk_idx]
        attn = q_i @ k_i.transpose(-1, -2) * prepared.decay_mask[:, :, chunk_idx]
        v_prime = prepared.k_cumdecay[:, :, chunk_idx] @ recurrent_state
        v_new = v_i - v_prime
        attn_inter = (
            q_i * prepared.g[:, :, chunk_idx, :, None].exp()
        ) @ recurrent_state
        core_attn_out[:, :, chunk_idx] = attn_inter + attn @ v_new
        recurrent_state = (
            recurrent_state * prepared.g[:, :, chunk_idx, -1, None, None].exp()
            + (
                k_i
                * (
                    prepared.g[:, :, chunk_idx, -1, None]
                    - prepared.g[:, :, chunk_idx]
                ).exp()[..., None]
            ).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0],
        core_attn_out.shape[1],
        -1,
        core_attn_out.shape[-1],
    )
    core_attn_out = core_attn_out[:, :, :prepared.sequence_length]
    return core_attn_out.transpose(1, 2).contiguous().to(prepared.initial_dtype)


def _pack_gdn_state_summary(
    state_ext: torch.Tensor,
    transition: torch.Tensor,
) -> torch.Tensor:
    """Pack ``S`` and ``M`` summaries into one all-gather payload."""
    if state_ext.shape[:-1] != transition.shape[:-1]:
        raise ValueError(
            "state_ext and transition must share [B,H,K] dimensions, got "
            f"{tuple(state_ext.shape)} and {tuple(transition.shape)}."
        )
    return torch.cat((state_ext, transition), dim=-1)


def _unpack_gdn_state_summary(
    packed: torch.Tensor,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a gathered ``[S, M]`` payload."""
    if packed.shape[-1] <= v_head_dim:
        raise ValueError(
            f"packed state summary last dim must be > v_head_dim={v_head_dim}, "
            f"got {packed.shape[-1]}."
        )
    state_ext = packed[..., :v_head_dim]
    transition = packed[..., v_head_dim:]
    return state_ext, transition


def _merge_gdn_prefix_state_summaries_torch(
    state_ext: torch.Tensor,
    transition: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Merge gathered GDN summaries before ``rank`` into its initial state."""
    if state_ext.dim() != 5 or transition.dim() != 5:
        raise ValueError(
            "state summary merge expects state_ext [R,B,H,K,V] and "
            "transition [R,B,H,K,K]."
        )
    if state_ext.shape[0] != transition.shape[0]:
        raise ValueError("state_ext and transition must have the same rank dimension.")
    if rank < 0 or rank > state_ext.shape[0]:
        raise ValueError(f"rank must be in [0, {state_ext.shape[0]}], got {rank}.")

    state = torch.zeros_like(state_ext[0])
    for prev_rank in range(rank):
        state = transition[prev_rank] @ state + state_ext[prev_rank]
    return state


def _gdn_state_all_gather(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cp_mesh: DeviceMesh,
    cp_rank: int,
    cp_size: int,
    *,
    use_qk_l2norm_in_kernel: bool,
) -> torch.Tensor:
    """Apply local GDN with all-gathered recurrent-state summaries."""
    if cp_size == 1:
        core_attn_out, _ = torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        return core_attn_out

    prepared = _prepare_gdn_chunks_for_summary(
        query,
        key,
        value,
        g,
        beta,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    state_ext, transition = _checkpoint_gdn_state_summary(prepared)
    packed_summary = _pack_gdn_state_summary(state_ext, transition)
    gathered_summary = _all_gather_stack(packed_summary, cp_mesh, cp_size)
    gathered_state_ext, gathered_transition = _unpack_gdn_state_summary(
        gathered_summary,
        state_ext.shape[-1],
    )

    initial_state = _merge_gdn_prefix_state_summaries_torch(
        gathered_state_ext,
        gathered_transition,
        cp_rank,
    )
    all_gather_tie = gathered_summary.sum()
    initial_state = initial_state + all_gather_tie.to(initial_state.dtype) * 0
    return _run_prepared_gdn_chunks(prepared, initial_state)


class _RecvInitialStateP2PFunction(torch.autograd.Function):
    """Receive the recurrent initial state; send its gradient in backward."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        ctx,
        anchor: torch.Tensor,
        cp_group,
        prev_rank: int,
        state_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Receive the initial state from the preceding CP rank."""
        state = torch.empty(state_shape, device=anchor.device, dtype=torch.float32)
        dist.recv(state, src=prev_rank, group=cp_group)
        ctx.cp_group = cp_group
        ctx.prev_rank = prev_rank
        return state

    @staticmethod
    def backward(ctx, grad_state: Optional[torch.Tensor]):
        if grad_state is None:
            raise RuntimeError("linear attention P2P backward missing initial-state grad.")
        dist.send(grad_state.contiguous(), dst=ctx.prev_rank, group=ctx.cp_group)
        return None, None, None, None


class _SendFinalStateP2PFunction(torch.autograd.Function):
    """Send the recurrent final state; receive its gradient in backward."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        ctx,
        final_state: torch.Tensor,
        cp_group,
        next_rank: int,
    ) -> torch.Tensor:
        """Send the final state to the succeeding CP rank."""
        dist.send(final_state.contiguous(), dst=next_rank, group=cp_group)
        ctx.cp_group = cp_group
        ctx.next_rank = next_rank
        ctx.state_shape = tuple(final_state.shape)
        ctx.state_dtype = final_state.dtype
        return final_state.new_zeros(())

    @staticmethod
    def backward(ctx, grad_token: torch.Tensor):
        grad_state = torch.empty(
            ctx.state_shape,
            device=grad_token.device,
            dtype=ctx.state_dtype,
        )
        dist.recv(grad_state, src=ctx.next_rank, group=ctx.cp_group)
        return grad_state, None, None


def _apply_gdn_state_summary(
    state_ext: torch.Tensor,
    transition: torch.Tensor,
    initial_state: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply ``state_out = M @ state_in + S`` to an incoming GDN state."""
    if initial_state is None:
        return state_ext
    return transition @ initial_state.to(transition) + state_ext


def _gdn_state_p2p_summary(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cp_mesh: DeviceMesh,
    cp_rank: int,
    cp_size: int,
    *,
    use_qk_l2norm_in_kernel: bool,
) -> torch.Tensor:
    """Run local GDN with an affine-summary state wavefront.

    Every rank prepares its local chunks and state transition in parallel.
    The rank-ordered critical path then contains only ``M @ state + S`` and
    the small state transfer. Token outputs retain the ordinary PyTorch graph,
    while the two custom autograd boundaries reverse the state communication.
    """
    if cp_size == 1:
        core_attn_out, _ = torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        return core_attn_out

    cp_group = cp_mesh.get_group()
    prev_rank = _global_peer_rank(cp_mesh, cp_rank - 1) if cp_rank > 0 else -1
    next_rank = _global_peer_rank(cp_mesh, cp_rank + 1) if cp_rank < cp_size - 1 else -1
    state_shape = (query.shape[0], value.shape[2], query.shape[3], value.shape[3])

    prepared = _prepare_gdn_chunks_for_summary(
        query,
        key,
        value,
        g,
        beta,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    state_ext = None
    transition = None
    if cp_rank < cp_size - 1:
        state_ext, transition = _checkpoint_gdn_state_summary(prepared)

    initial_state = None
    if cp_rank > 0:
        initial_state = _RecvInitialStateP2PFunction.apply(
            query,
            cp_group,
            prev_rank,
            state_shape,
        )

    send_token = None
    if cp_rank < cp_size - 1:
        final_state = _apply_gdn_state_summary(
            state_ext,
            transition,
            initial_state,
        )
        send_token = _SendFinalStateP2PFunction.apply(final_state, cp_group, next_rank)

    core_attn_out = _run_prepared_gdn_chunks(prepared, initial_state)
    if send_token is not None:
        core_attn_out = core_attn_out + send_token.to(core_attn_out.dtype) * 0

    return core_attn_out


class _GDNStateP2PTritonFunction(torch.autograd.Function):
    """Pipeline fused affine GDN states over sequence-sharded CP ranks."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ,too-many-locals
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cp_rank: int,
        cp_size: int,
        cp_group,
        prev_rank: int,
        next_rank: int,
    ) -> torch.Tensor:
        """Run fused local GDN and forward its affine state across CP ranks."""
        from hyper_parallel.platform.torch.custom_ops.gdn.chunk_gated_delta_rule import (  # pylint: disable=import-outside-toplevel
            chunk_gated_delta_rule_fwd_apply_state_saved,
            chunk_gated_delta_rule_fwd_output_saved,
            chunk_gated_delta_rule_fwd_prepare_saved,
        )
        from hyper_parallel.platform.torch.custom_ops.gdn.state_summary import (  # pylint: disable=import-outside-toplevel
            apply_gdn_state_summary,
            chunk_gated_delta_rule_state_summary_fwd,
        )

        (
            query_norm,
            key_norm,
            _,
            _,
            g_cumsum,
            matrix_a,
            w,
            u,
            scale,
        ) = chunk_gated_delta_rule_fwd_prepare_saved(
            query,
            key,
            value,
            g,
            beta,
            use_qk_l2norm_in_kernel=False,
        )
        initial_state = None
        recv_buffer = None
        recv_work = None
        if cp_rank > 0:
            recv_buffer = torch.empty(
                (query.shape[0], query.shape[2], query.shape[3], value.shape[3]),
                device=query.device,
                dtype=torch.float32,
            )
            recv_work = dist.irecv(recv_buffer, src=prev_rank, group=cp_group)

        state_ext = None
        transition = None
        if cp_rank < cp_size - 1:
            state_ext, transition = chunk_gated_delta_rule_state_summary_fwd(
                key_norm,
                w,
                u,
                g_cumsum,
            )

        if recv_work is not None:
            recv_work.wait()
            initial_state = recv_buffer

        send_buffer = None
        send_work = None
        if cp_rank < cp_size - 1:
            send_buffer = apply_gdn_state_summary(
                state_ext,
                transition,
                initial_state,
            ).contiguous()
            send_work = dist.isend(send_buffer, dst=next_rank, group=cp_group)

        h, v_new, _ = chunk_gated_delta_rule_fwd_apply_state_saved(
            key_norm,
            g_cumsum,
            w,
            u,
            initial_state=initial_state,
            output_final_state=False,
        )
        output = chunk_gated_delta_rule_fwd_output_saved(
            query_norm,
            key_norm,
            g_cumsum,
            h,
            v_new,
            scale,
        ).to(query.dtype)

        if send_work is not None:
            send_work.wait()

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
        )
        ctx.has_initial_state = initial_state is not None
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.cp_group = cp_group
        ctx.prev_rank = prev_rank
        ctx.next_rank = next_rank
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pylint: disable=too-many-locals
        """Backpropagate local GDN tensors and the state gradient wavefront."""
        from hyper_parallel.platform.torch.custom_ops.gdn.chunk_gated_delta_rule import (  # pylint: disable=import-outside-toplevel
            chunk_gated_delta_rule_bwd_finish_saved,
            chunk_gated_delta_rule_bwd_prepare_saved,
            chunk_gated_delta_rule_bwd_state_saved,
        )
        from hyper_parallel.platform.torch.custom_ops.gdn.state_summary import (  # pylint: disable=import-outside-toplevel
            apply_gdn_state_gradient_summary,
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
        ) = ctx.saved_tensors
        if not ctx.has_initial_state:
            initial_state = None

        w, h, v_new, dv = chunk_gated_delta_rule_bwd_prepare_saved(
            query,
            key,
            value,
            g_cumsum,
            beta,
            matrix_a,
            initial_state,
            grad_output,
            ctx.scale,
        )

        grad_state_ext = None
        if ctx.cp_rank > 0:
            grad_state_ext = chunk_gated_delta_rule_state_gradient_summary_bwd(
                query,
                key,
                w,
                g_cumsum,
                grad_output,
                dv,
                ctx.scale,
            )

        grad_final_state = None
        recv_work = None
        if ctx.cp_rank < ctx.cp_size - 1:
            recv_buffer = torch.empty(
                (query.shape[0], query.shape[2], query.shape[3], value.shape[3]),
                device=grad_output.device,
                dtype=torch.float32,
            )
            recv_work = dist.irecv(
                recv_buffer,
                src=ctx.next_rank,
                group=ctx.cp_group,
            )
        if recv_work is not None:
            recv_work.wait()
            grad_final_state = recv_buffer

        send_buffer = None
        send_work = None
        if ctx.cp_rank > 0:
            send_buffer = apply_gdn_state_gradient_summary(
                grad_state_ext,
                transition,
                grad_final_state,
            ).contiguous()
            send_work = dist.isend(
                send_buffer,
                dst=ctx.prev_rank,
                group=ctx.cp_group,
            )

        dh, _, dv = chunk_gated_delta_rule_bwd_state_saved(
            query,
            key,
            g_cumsum,
            w,
            initial_state,
            grad_final_state,
            grad_output,
            dv,
            ctx.scale,
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
        )

        if send_work is not None:
            send_work.wait()
        return dq, dk, dv, dg, dbeta, None, None, None, None, None


def _gdn_state_p2p_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cp_mesh: DeviceMesh,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Run fused local GDN with an affine state wavefront."""
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
    )


def _differentiable_all_to_all_shard(
    tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    *,
    split_dim: int,
    concat_dim: int,
) -> torch.Tensor:
    """Split local data on ``split_dim`` and concatenate peers on ``concat_dim``.

    This is the local-tensor equivalent of DTensor ``Shard(concat_dim) ->
    Shard(split_dim)`` redistribution for a 1-D mesh. It uses platform-level
    differentiable all-to-all directly to avoid wrapping each activation in a
    temporary DTensor.
    """
    split_count = device_mesh.size()
    if split_count == 1:
        return tensor

    original_shape = tuple(tensor.shape)
    dim_size = original_shape[split_dim]
    if dim_size % split_count != 0:
        raise ValueError(
            f"linear attention all-to-all split dim {split_dim} with size "
            f"{dim_size} must be divisible by cp_size {split_count}."
        )

    split_size = dim_size // split_count
    final_shape = list(original_shape)
    if split_dim != concat_dim:
        final_shape[split_dim] = split_size
        final_shape[concat_dim] = final_shape[concat_dim] * split_count
    final_shape = tuple(final_shape)

    reshape_dims = list(original_shape)
    reshape_dims[split_dim] = split_count
    reshape_dims.insert(split_dim + 1, split_size)

    trans_dims = list(range(len(reshape_dims)))
    trans_dims.remove(split_dim)
    trans_dims.insert(0, split_dim)

    a2a_input = tensor.reshape(reshape_dims).permute(trans_dims).contiguous()
    reshape_shape = list(a2a_input.shape)
    reshape_shape[0] = reshape_shape[0] * reshape_shape[1]
    reshape_shape.pop(1)
    a2a_input = a2a_input.reshape(reshape_shape)

    a2a_input = a2a_input.contiguous()
    split_len = a2a_input.shape[0] // split_count
    input_splits = [split_len] * split_count
    output_splits = [split_len] * split_count
    output = platform.differentiable_all_to_all_single(
        a2a_input,
        input_splits,
        output_splits,
        group=device_mesh.get_group(),
    )

    output_reshape = list(output.shape)
    output_reshape[0] = split_count
    output_reshape.insert(1, output.shape[0] // split_count)

    out_trans_dims = list(range(len(output_reshape)))
    first_dim = out_trans_dims.pop(0)
    if concat_dim >= len(out_trans_dims):
        out_trans_dims.append(first_dim)
    else:
        out_trans_dims.insert(concat_dim, first_dim)

    final_output = output.reshape(output_reshape).permute(out_trans_dims).contiguous()
    final_reshape = list(final_output.shape)
    if concat_dim < len(final_reshape) - 1:
        final_reshape[concat_dim] = (
            final_reshape[concat_dim] * final_reshape[concat_dim + 1]
        )
        final_reshape.pop(concat_dim + 1)

    return final_output.reshape(final_reshape).view(final_shape)


class LinearAttentionUlyssesCPWrapper(nn.Module):
    """Pure-Ulysses CP execution wrapper for a Qwen3.5 Gated DeltaNet module.

    Parameters stay owned by the original module. The wrapper only changes the
    execution layout:

    ``[B, S_local, full_heads] -> [B, S_full, local_heads] ->
    [B, S_local, full_heads]``.
    """

    def __init__(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
        *,
        backend: str = "eager",
    ):
        super().__init__()
        self.module = module
        self.gdn_backend = backend
        self.cp_mesh = _ensure_1d(device_mesh)
        self.cp_size = self.cp_mesh.size()
        self.cp_rank = self.cp_mesh.get_local_rank()
        self.seq_dim = 1
        self.head_dim = 2
        self._validate_module()

    def _validate_module(self) -> None:
        if self.cp_size <= 1:
            return
        if self.module.num_k_heads % self.cp_size != 0:
            raise ValueError(
                f"linear attention num_k_heads ({self.module.num_k_heads}) must be "
                f"divisible by cp_size ({self.cp_size}) for Ulysses CP."
            )
        if self.module.num_v_heads % self.cp_size != 0:
            raise ValueError(
                f"linear attention num_v_heads ({self.module.num_v_heads}) must be "
                f"divisible by cp_size ({self.cp_size}) for Ulysses CP."
            )

    def _seq_to_head(self, tensor: torch.Tensor) -> torch.Tensor:
        return _differentiable_all_to_all_shard(
            tensor,
            self.cp_mesh,
            split_dim=self.head_dim,
            concat_dim=self.seq_dim,
        )

    def _head_to_seq(self, tensor: torch.Tensor) -> torch.Tensor:
        return _differentiable_all_to_all_shard(
            tensor,
            self.cp_mesh,
            split_dim=self.seq_dim,
            concat_dim=self.head_dim,
        )

    def _seq_to_head_qkvba(
        self,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack Q/K/V/B/A by CP rank and run a single seq-to-head all-to-all."""
        if self.cp_size == 1:
            return q_proj, k_proj, v_proj, b, a

        base = self.module
        local_key_dim = base.key_dim // self.cp_size
        local_value_dim = base.value_dim // self.cp_size
        local_num_v_heads = base.num_v_heads // self.cp_size

        q_chunks = torch.split(q_proj, local_key_dim, dim=-1)
        k_chunks = torch.split(k_proj, local_key_dim, dim=-1)
        v_chunks = torch.split(v_proj, local_value_dim, dim=-1)
        b_chunks = torch.split(b, local_num_v_heads, dim=-1)
        a_chunks = torch.split(a, local_num_v_heads, dim=-1)
        rank_major_chunks = [
            torch.cat(chunks, dim=-1)
            for chunks in zip(q_chunks, k_chunks, v_chunks, b_chunks, a_chunks)
        ]
        packed = torch.cat(rank_major_chunks, dim=-1).contiguous()
        packed = self._seq_to_head(packed)
        return torch.split(
            packed,
            [
                local_key_dim,
                local_key_dim,
                local_value_dim,
                local_num_v_heads,
                local_num_v_heads,
            ],
            dim=-1,
        )

    def _local_conv_weight(self) -> torch.Tensor:
        return _slice_qkv_local_cp(
            self.module.conv1d.weight,
            key_dim=self.module.key_dim,
            value_dim=self.module.value_dim,
            dim=0,
            cp_rank=self.cp_rank,
            cp_size=self.cp_size,
        )

    def _local_conv_bias(self) -> Optional[torch.Tensor]:
        bias = self.module.conv1d.bias
        if bias is None:
            return None
        return _slice_qkv_local_cp(
            bias,
            key_dim=self.module.key_dim,
            value_dim=self.module.value_dim,
            dim=0,
            cp_rank=self.cp_rank,
            cp_size=self.cp_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run Gated DeltaNet with pure Ulysses context parallel."""
        del kwargs
        hidden_states = _local_tensor_at_cp_boundary(hidden_states)

        base = self.module
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(
                hidden_states.dtype
            )

        bsz, local_seq_len, _ = hidden_states.shape
        mixed_qkv = base.in_proj_qkv(hidden_states)
        z = base.in_proj_z(hidden_states).reshape(
            bsz,
            local_seq_len,
            base.num_v_heads,
            base.head_v_dim,
        )
        b = base.in_proj_b(hidden_states)
        a = base.in_proj_a(hidden_states)

        q_proj, k_proj, v_proj = torch.split(
            mixed_qkv,
            [base.key_dim, base.key_dim, base.value_dim],
            dim=-1,
        )
        q_proj, k_proj, v_proj, b, a = self._seq_to_head_qkvba(
            q_proj, k_proj, v_proj, b, a
        )

        full_seq_len = q_proj.shape[1]
        local_key_dim = base.key_dim // self.cp_size
        local_value_dim = base.value_dim // self.cp_size
        local_num_k_heads = base.num_k_heads // self.cp_size
        local_num_v_heads = base.num_v_heads // self.cp_size
        local_conv_dim = local_key_dim * 2 + local_value_dim

        mixed_qkv = torch.cat((q_proj, k_proj, v_proj), dim=-1).transpose(1, 2)
        conv_out = F.conv1d(
            input=mixed_qkv,
            weight=self._local_conv_weight(),
            bias=self._local_conv_bias(),
            stride=base.conv1d.stride,
            padding=base.conv1d.padding,
            dilation=base.conv1d.dilation,
            groups=local_conv_dim,
        )
        mixed_qkv = F.silu(conv_out[:, :, :full_seq_len]).transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv,
            [local_key_dim, local_key_dim, local_value_dim],
            dim=-1,
        )
        query = query.reshape(bsz, full_seq_len, local_num_k_heads, base.head_k_dim)
        key = key.reshape(bsz, full_seq_len, local_num_k_heads, base.head_k_dim)
        value = value.reshape(bsz, full_seq_len, local_num_v_heads, base.head_v_dim)

        a_log = _slice_local_cp(base.A_log, 0, self.cp_rank, self.cp_size)
        dt_bias = _slice_local_cp(base.dt_bias, 0, self.cp_rank, self.cp_size)
        beta = b.sigmoid()
        g = -a_log.float().exp() * F.softplus(a.float() + dt_bias)

        if base.kv_groups > 1:
            query = query.repeat_interleave(base.kv_groups, dim=2)
            key = key.repeat_interleave(base.kv_groups, dim=2)

        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            backend=self.gdn_backend,
        )

        core_attn_out = self._head_to_seq(core_attn_out)
        core_attn_out = core_attn_out.reshape(-1, base.head_v_dim)
        z_flat = z.reshape(-1, base.head_v_dim)
        core_attn_out = base.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(bsz, local_seq_len, base.value_dim)
        if hasattr(base, "out_proj_input"):
            core_attn_out = base.out_proj_input(core_attn_out)
        return base.out_proj(core_attn_out)


class LinearAttentionP2PCPWrapper(nn.Module):
    """Sequence-sharded GDN CP with an affine-summary state wavefront."""

    def __init__(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
        *,
        backend: str = "eager",
    ):
        super().__init__()
        self.module = module
        self.gdn_backend = backend
        self.cp_mesh = _ensure_1d(device_mesh)
        self.cp_size = self.cp_mesh.size()
        self.cp_rank = self.cp_mesh.get_local_rank()
        self._validate_module()

    def _validate_module(self) -> None:
        """Validate the Conv1d requirements of the P2P CP path."""
        if self.gdn_backend == "triton" and (
            self.module.head_k_dim != 128 or self.module.head_v_dim != 128
        ):
            raise NotImplementedError(
                "linear attention P2P Triton backend requires "
                "head_k_dim=head_v_dim=128."
            )
        conv = self.module.conv1d
        if conv.stride != (1,):
            raise ValueError(
                "linear attention P2P CP currently supports only conv1d stride=1."
            )
        if conv.groups != self.module.conv_dim:
            raise ValueError(
                "linear attention P2P CP expects depthwise conv1d groups=conv_dim."
            )
        if (
            conv.in_channels != self.module.conv_dim
            or conv.out_channels != self.module.conv_dim
        ):
            raise ValueError(
                "linear attention P2P CP expects conv1d channels to match conv_dim."
            )

    def _conv1d_with_halo(self, mixed_qkv: torch.Tensor) -> torch.Tensor:
        """Run local Conv1d after exchanging only the previous-rank halo."""
        return _causal_conv1d_with_cp_halo(
            mixed_qkv,
            self.module.conv1d,
            self.cp_mesh,
            self.cp_rank,
            self.cp_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run Gated DeltaNet on local sequence shards with recurrent-state P2P."""
        del kwargs
        hidden_states = _local_tensor_at_cp_boundary(hidden_states)

        base = self.module
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(
                hidden_states.dtype
            )

        bsz, local_seq_len, _ = hidden_states.shape
        mixed_qkv = base.in_proj_qkv(hidden_states)
        z = base.in_proj_z(hidden_states).reshape(
            bsz,
            local_seq_len,
            base.num_v_heads,
            base.head_v_dim,
        )
        b = base.in_proj_b(hidden_states)
        a = base.in_proj_a(hidden_states)

        mixed_qkv = self._conv1d_with_halo(mixed_qkv)
        query, key, value = torch.split(
            mixed_qkv,
            [base.key_dim, base.key_dim, base.value_dim],
            dim=-1,
        )
        query = query.reshape(bsz, local_seq_len, base.num_k_heads, base.head_k_dim)
        key = key.reshape(bsz, local_seq_len, base.num_k_heads, base.head_k_dim)
        value = value.reshape(bsz, local_seq_len, base.num_v_heads, base.head_v_dim)

        beta = b.sigmoid()
        g = -base.A_log.float().exp() * F.softplus(a.float() + base.dt_bias)

        if base.kv_groups > 1:
            query = query.repeat_interleave(base.kv_groups, dim=2)
            key = key.repeat_interleave(base.kv_groups, dim=2)

        if self.gdn_backend == "triton":
            if local_seq_len % 64 != 0:
                raise NotImplementedError(
                    "linear attention P2P Triton backend requires each CP "
                    f"rank's local sequence length ({local_seq_len}) to be "
                    "divisible by 64."
                )
            if not is_triton_gdn_available(query, key, value, g, beta):
                raise RuntimeError(
                    "linear attention P2P Triton backend requires an NPU "
                    "input satisfying the fixed GDN contract and a validated "
                    "triton-ascend 3.2.x installation."
                )
            core_attn_out = _gdn_state_p2p_triton(
                query,
                key,
                value,
                g,
                beta,
                self.cp_mesh,
                self.cp_rank,
                self.cp_size,
            )
        else:
            core_attn_out = _gdn_state_p2p_summary(
                query,
                key,
                value,
                g,
                beta,
                self.cp_mesh,
                self.cp_rank,
                self.cp_size,
                use_qk_l2norm_in_kernel=True,
            )

        core_attn_out = core_attn_out.reshape(-1, base.head_v_dim)
        z_flat = z.reshape(-1, base.head_v_dim)
        core_attn_out = base.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(bsz, local_seq_len, base.value_dim)
        if hasattr(base, "out_proj_input"):
            core_attn_out = base.out_proj_input(core_attn_out)
        return base.out_proj(core_attn_out)


class LinearAttentionAllGatherCPWrapper(nn.Module):
    """Sequence-sharded GDN CP using all-gathered recurrent-state summaries."""

    def __init__(self, module: nn.Module, device_mesh: DeviceMesh):
        super().__init__()
        self.module = module
        self.cp_mesh = _ensure_1d(device_mesh)
        self.cp_size = self.cp_mesh.size()
        self.cp_rank = self.cp_mesh.get_local_rank()
        self._validate_module()

    def _validate_module(self) -> None:
        """Validate the Conv1d requirements of the all-gather CP path."""
        conv = self.module.conv1d
        if conv.stride != (1,):
            raise ValueError(
                "linear attention all-gather CP currently supports only "
                "conv1d stride=1."
            )
        if conv.groups != self.module.conv_dim:
            raise ValueError(
                "linear attention all-gather CP expects depthwise conv1d "
                "groups=conv_dim."
            )
        if (
            conv.in_channels != self.module.conv_dim
            or conv.out_channels != self.module.conv_dim
        ):
            raise ValueError(
                "linear attention all-gather CP expects conv1d channels to "
                "match conv_dim."
            )

    def _conv1d_with_halo(self, mixed_qkv: torch.Tensor) -> torch.Tensor:
        """Run local Conv1d after exchanging only the previous-rank halo."""
        return _causal_conv1d_with_cp_halo(
            mixed_qkv,
            self.module.conv1d,
            self.cp_mesh,
            self.cp_rank,
            self.cp_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run Gated DeltaNet on local sequence shards with all-gather state summaries."""
        del kwargs
        hidden_states = _local_tensor_at_cp_boundary(hidden_states)

        base = self.module
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(
                hidden_states.dtype
            )

        bsz, local_seq_len, _ = hidden_states.shape
        mixed_qkv = base.in_proj_qkv(hidden_states)
        z = base.in_proj_z(hidden_states).reshape(
            bsz,
            local_seq_len,
            base.num_v_heads,
            base.head_v_dim,
        )
        b = base.in_proj_b(hidden_states)
        a = base.in_proj_a(hidden_states)

        mixed_qkv = self._conv1d_with_halo(mixed_qkv)
        query, key, value = torch.split(
            mixed_qkv,
            [base.key_dim, base.key_dim, base.value_dim],
            dim=-1,
        )
        query = query.reshape(bsz, local_seq_len, base.num_k_heads, base.head_k_dim)
        key = key.reshape(bsz, local_seq_len, base.num_k_heads, base.head_k_dim)
        value = value.reshape(bsz, local_seq_len, base.num_v_heads, base.head_v_dim)

        beta = b.sigmoid()
        g = -base.A_log.float().exp() * F.softplus(a.float() + base.dt_bias)

        if base.kv_groups > 1:
            query = query.repeat_interleave(base.kv_groups, dim=2)
            key = key.repeat_interleave(base.kv_groups, dim=2)

        core_attn_out = _gdn_state_all_gather(
            query,
            key,
            value,
            g,
            beta,
            self.cp_mesh,
            self.cp_rank,
            self.cp_size,
            use_qk_l2norm_in_kernel=True,
        )

        core_attn_out = core_attn_out.reshape(-1, base.head_v_dim)
        z_flat = z.reshape(-1, base.head_v_dim)
        core_attn_out = base.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(bsz, local_seq_len, base.value_dim)
        if hasattr(base, "out_proj_input"):
            core_attn_out = base.out_proj_input(core_attn_out)
        return base.out_proj(core_attn_out)


class LinearAttentionContextParallel(ParallelStyle):
    """Apply context parallel execution to a Gated DeltaNet module."""

    def __init__(self, *, mode: str = "ulysses", backend: str = "eager") -> None:
        if mode not in {"ulysses", "p2p", "all_gather"}:
            raise NotImplementedError(
                "LinearAttentionContextParallel currently supports mode='ulysses', "
                "mode='p2p', and mode='all_gather'."
            )
        if backend not in {"eager", "triton"}:
            raise ValueError(
                "LinearAttentionContextParallel backend must be 'eager' or "
                f"'triton', got {backend!r}."
            )
        if mode == "all_gather" and backend == "triton":
            raise NotImplementedError(
                "linear attention all-gather CP does not yet support the "
                "Triton backend."
            )
        self.mode = mode
        self.backend = backend

    def apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Patch ``module.forward`` with a linear-attention CP executor."""
        if self.mode == "ulysses":
            executor = LinearAttentionUlyssesCPWrapper(
                module,
                device_mesh,
                backend=self.backend,
            )
        elif self.mode == "all_gather":
            executor = LinearAttentionAllGatherCPWrapper(module, device_mesh)
        else:
            executor = LinearAttentionP2PCPWrapper(
                module,
                device_mesh,
                backend=self.backend,
            )
        object.__setattr__(module, "_hp_linear_attention_cp_executor", executor)
        object.__setattr__(module, "_hp_linear_attention_original_forward", module.forward)

        def _forward(*args, **kwargs):
            return executor(*args, **kwargs)

        object.__setattr__(module, "forward", _forward)
        return module
