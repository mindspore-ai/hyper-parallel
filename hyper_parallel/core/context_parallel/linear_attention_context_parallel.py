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
"""Context-parallel execution for Qwen3.5 Gated DeltaNet layers."""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from hyper_parallel.core.context_parallel.context_parallel import _ensure_1d
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.tensor_parallel.style import ParallelStyle
from hyper_parallel.models.modules.linear_attention import (
    causal_depthwise_conv1d,
    chunk_gated_delta_rule,
    is_triton_gdn_available,
)
from hyper_parallel.platform import get_platform


platform = get_platform()
_SUPPORTED_MODES = frozenset({"ulysses", "p2p"})


def _global_peer_rank(cp_mesh: DeviceMesh, local_rank: int) -> int:
    """Map a rank index in the CP mesh to its global distributed rank."""
    return int(cp_mesh.rank_list[local_rank])


def _local_tensor_at_cp_boundary(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local sequence shard passed into a CP-wrapped module."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


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
            f"linear attention CP expects dim size {dim_size} to be divisible "
            f"by cp_size {cp_size}."
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
    """Slice a fused ``[Q, K, V]`` tensor on its channel dimension."""
    query, key, value = torch.split(tensor, [key_dim, key_dim, value_dim], dim=dim)
    return torch.cat(
        (
            _slice_local_cp(query, dim, cp_rank, cp_size),
            _slice_local_cp(key, dim, cp_rank, cp_size),
            _slice_local_cp(value, dim, cp_rank, cp_size),
        ),
        dim=dim,
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
    Shard(split_dim)`` redistribution for a 1-D mesh.
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
        final_shape[concat_dim] *= split_count
    final_shape = tuple(final_shape)

    pre_special = all(original_shape[index] == 1 for index in range(split_dim))
    if pre_special:
        reshape_shape = (split_count * split_size,) + original_shape[split_dim + 1:]
        a2a_input = tensor.view(reshape_shape)
    else:
        reshape_dims = list(original_shape)
        reshape_dims[split_dim] = split_count
        reshape_dims.insert(split_dim + 1, split_size)
        transpose_dims = list(range(len(reshape_dims)))
        transpose_dims.remove(split_dim)
        transpose_dims.insert(0, split_dim)
        a2a_input = tensor.reshape(reshape_dims).permute(transpose_dims).contiguous()
        reshape_shape = list(a2a_input.shape)
        reshape_shape[0] *= reshape_shape[1]
        reshape_shape.pop(1)
        a2a_input = a2a_input.reshape(reshape_shape)

    a2a_input = a2a_input.contiguous()
    split_len = a2a_input.shape[0] // split_count
    splits = [split_len] * split_count
    output = platform.differentiable_all_to_all_single(
        a2a_input,
        splits,
        splits,
        group=device_mesh.get_group(),
    )

    post_special = all(final_shape[index] == 1 for index in range(concat_dim))
    if post_special:
        return output.view(final_shape)

    reconstructed_concat_dim = concat_dim - split_dim if pre_special else concat_dim
    output_shape = list(output.shape)
    output_shape[0] = split_count
    output_shape.insert(1, output.shape[0] // split_count)
    transpose_dims = list(range(len(output_shape)))
    rank_dim = transpose_dims.pop(0)
    transpose_dims.insert(reconstructed_concat_dim, rank_dim)
    final_output = output.reshape(output_shape).permute(transpose_dims).contiguous()

    reshape_shape = list(final_output.shape)
    if reconstructed_concat_dim < len(reshape_shape) - 1:
        reshape_shape[reconstructed_concat_dim] *= reshape_shape[
            reconstructed_concat_dim + 1
        ]
        reshape_shape.pop(reconstructed_concat_dim + 1)
    result = final_output.reshape(reshape_shape)
    return result.view(final_shape) if pre_special else result


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


def _causal_conv1d_with_left_halo(
    value: torch.Tensor,
    halo: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    dilation: int,
    num_heads: int,
    backend: str,
) -> torch.Tensor:
    """Prepend the previous rank's halo and discard its convolution outputs."""
    halo_width = halo.shape[1]
    extended_output = causal_depthwise_conv1d(
        torch.cat((halo, value), dim=1),
        weight,
        bias,
        dilation=dilation,
        num_heads=num_heads,
        backend=backend,
    )
    return extended_output[:, halo_width:, :]


class _MemoryEfficientL2NormFunction(torch.autograd.Function):
    """L2-normalize without retaining the unnormalized activation."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, eps: float) -> torch.Tensor:
        inv_norm = torch.rsqrt((tensor * tensor).sum(dim=-1, keepdim=True) + eps)
        normalized = (tensor * inv_norm).to(tensor.dtype)
        ctx.save_for_backward(normalized, inv_norm)
        return normalized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        normalized, inv_norm = ctx.saved_tensors
        projection = (normalized * grad_output).sum(dim=-1, keepdim=True)
        grad_input = inv_norm * (grad_output - normalized * projection)
        return grad_input.to(grad_output.dtype), None


def _memory_efficient_l2norm(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _MemoryEfficientL2NormFunction.apply(tensor, eps)


class _GDNStateP2PFunction(torch.autograd.Function):
    """Pipeline affine GDN state summaries over sequence-sharded CP ranks."""

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
        from hyper_parallel.platform.torch.custom_ops.gdn import (  # pylint: disable=import-outside-toplevel
            apply_gdn_state_summary,
            chunk_gated_delta_rule_fwd_apply_state_saved,
            chunk_gated_delta_rule_fwd_output_saved,
            chunk_gated_delta_rule_fwd_prepare_saved,
            chunk_gated_delta_rule_state_summary_fwd,
        )

        (
            query_norm,
            key_norm,
            query_inv_norm,
            key_inv_norm,
            g_cumsum,
            A,
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
            A,
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
        from hyper_parallel.platform.torch.custom_ops.gdn import (  # pylint: disable=import-outside-toplevel
            apply_gdn_state_gradient_summary,
            chunk_gated_delta_rule_bwd_finish_saved,
            chunk_gated_delta_rule_bwd_prepare_saved,
            chunk_gated_delta_rule_bwd_state_saved,
            chunk_gated_delta_rule_state_gradient_summary_bwd,
        )

        (
            query,
            key,
            value,
            g_cumsum,
            beta,
            A,
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
            A,
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
        recv_buffer = None
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
            A,
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


def _gdn_state_p2p(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cp_mesh: DeviceMesh,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Run local fused GDN while passing affine state summaries between ranks."""
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

    query = _memory_efficient_l2norm(query)
    key = _memory_efficient_l2norm(key)
    prev_rank = _global_peer_rank(cp_mesh, cp_rank - 1) if cp_rank > 0 else -1
    next_rank = _global_peer_rank(cp_mesh, cp_rank + 1) if cp_rank < cp_size - 1 else -1
    return _GDNStateP2PFunction.apply(
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


class LinearAttentionUlyssesCPWrapper(nn.Module):
    """Pure-Ulysses execution for a Qwen3.5 Gated DeltaNet module."""

    def __init__(self, module: nn.Module, device_mesh: DeviceMesh):
        super().__init__()
        self.module = module
        self.cp_mesh = _ensure_1d(device_mesh)
        self.cp_size = self.cp_mesh.size()
        self.cp_rank = self.cp_mesh.get_local_rank()
        self._validate_module()

    def _validate_module(self) -> None:
        for name in ("num_k_heads", "num_v_heads"):
            heads = getattr(self.module, name)
            if heads % self.cp_size != 0:
                raise ValueError(
                    f"linear attention {name} ({heads}) must be divisible by "
                    f"cp_size ({self.cp_size}) for Ulysses CP."
                )

    def _seq_to_head(self, tensor: torch.Tensor) -> torch.Tensor:
        return _differentiable_all_to_all_shard(
            tensor,
            self.cp_mesh,
            split_dim=2,
            concat_dim=1,
        )

    def _head_to_seq(self, tensor: torch.Tensor) -> torch.Tensor:
        return _differentiable_all_to_all_shard(
            tensor,
            self.cp_mesh,
            split_dim=1,
            concat_dim=2,
        )

    def _seq_to_head_qkvba(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack Q/K/V/B/A into one sequence-to-head all-to-all."""
        if self.cp_size == 1:
            return query, key, value, beta, decay

        base = self.module
        local_key_dim = base.key_dim // self.cp_size
        local_value_dim = base.value_dim // self.cp_size
        local_value_heads = base.num_v_heads // self.cp_size
        rank_chunks = zip(
            torch.split(query, local_key_dim, dim=-1),
            torch.split(key, local_key_dim, dim=-1),
            torch.split(value, local_value_dim, dim=-1),
            torch.split(beta, local_value_heads, dim=-1),
            torch.split(decay, local_value_heads, dim=-1),
        )
        packed = torch.cat(
            [torch.cat(chunks, dim=-1) for chunks in rank_chunks],
            dim=-1,
        ).contiguous()
        packed = self._seq_to_head(packed)
        return torch.split(
            packed,
            (
                local_key_dim,
                local_key_dim,
                local_value_dim,
                local_value_heads,
                local_value_heads,
            ),
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
        if self.module.conv1d.bias is None:
            return None
        return _slice_qkv_local_cp(
            self.module.conv1d.bias,
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
        del kwargs
        hidden_states = _local_tensor_at_cp_boundary(hidden_states)
        base = self.module
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(
                hidden_states.dtype
            )

        batch, local_seq_len, _ = hidden_states.shape
        mixed_qkv = base.in_proj_qkv(hidden_states)
        z = base.in_proj_z(hidden_states).reshape(
            batch,
            local_seq_len,
            base.num_v_heads,
            base.head_v_dim,
        )
        beta = base.in_proj_b(hidden_states)
        decay = base.in_proj_a(hidden_states)
        query, key, value = torch.split(
            mixed_qkv,
            [base.key_dim, base.key_dim, base.value_dim],
            dim=-1,
        )
        query, key, value, beta, decay = self._seq_to_head_qkvba(
            query,
            key,
            value,
            beta,
            decay,
        )

        full_seq_len = query.shape[1]
        local_key_dim = base.key_dim // self.cp_size
        local_value_dim = base.value_dim // self.cp_size
        local_key_heads = base.num_k_heads // self.cp_size
        local_value_heads = base.num_v_heads // self.cp_size
        mixed_qkv = causal_depthwise_conv1d(
            torch.cat((query, key, value), dim=-1),
            self._local_conv_weight(),
            self._local_conv_bias(),
            dilation=base.conv1d.dilation[0],
            num_heads=2 * local_key_heads + local_value_heads,
            backend=base.conv_backend,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [local_key_dim, local_key_dim, local_value_dim],
            dim=-1,
        )
        query = query.reshape(
            batch, full_seq_len, local_key_heads, base.head_k_dim
        )
        key = key.reshape(batch, full_seq_len, local_key_heads, base.head_k_dim)
        value = value.reshape(
            batch, full_seq_len, local_value_heads, base.head_v_dim
        )

        A_log = _slice_local_cp(base.A_log, 0, self.cp_rank, self.cp_size)
        dt_bias = _slice_local_cp(base.dt_bias, 0, self.cp_rank, self.cp_size)
        g = -A_log.float().exp() * F.softplus(decay.float() + dt_bias)
        beta = beta.sigmoid()
        if base.kv_groups > 1:
            query = query.repeat_interleave(base.kv_groups, dim=2)
            key = key.repeat_interleave(base.kv_groups, dim=2)

        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            backend=base.gdn_backend,
        )
        core_attn_out = self._head_to_seq(core_attn_out)
        core_attn_out = core_attn_out.reshape(-1, base.head_v_dim)
        core_attn_out = base.norm(core_attn_out, z.reshape(-1, base.head_v_dim))
        core_attn_out = core_attn_out.reshape(
            batch, local_seq_len, base.value_dim
        )
        if hasattr(base, "out_proj_input"):
            core_attn_out = base.out_proj_input(core_attn_out)
        return base.out_proj(core_attn_out)


class LinearAttentionP2PCPWrapper(nn.Module):
    """Sequence-sharded GDN with A2AV conv halo and affine-state P2P."""

    def __init__(self, module: nn.Module, device_mesh: DeviceMesh):
        super().__init__()
        self.module = module
        self.cp_mesh = _ensure_1d(device_mesh)
        self.cp_size = self.cp_mesh.size()
        self.cp_rank = self.cp_mesh.get_local_rank()
        self._validate_module()

    def _validate_module(self) -> None:
        base = self.module
        conv = base.conv1d
        if conv.stride != (1,) or conv.groups != base.conv_dim:
            raise ValueError(
                "linear attention P2P requires a stride-1 depthwise Conv1d."
            )
        if base.head_k_dim != 128 or base.head_v_dim != 128:
            raise NotImplementedError(
                "linear attention P2P state-summary kernels currently require "
                "head_k_dim=head_v_dim=128."
            )
        if base.gdn_backend == "eager":
            raise ValueError(
                "linear attention mode='p2p' requires gdn_backend='auto' or 'triton'."
            )

    def _conv1d_with_halo(self, mixed_qkv: torch.Tensor) -> torch.Tensor:
        base = self.module
        dilation = base.conv1d.dilation[0]
        halo_width = (base.conv1d.kernel_size[0] - 1) * dilation
        if halo_width == 0 or self.cp_size == 1:
            return causal_depthwise_conv1d(
                mixed_qkv,
                base.conv1d.weight,
                base.conv1d.bias,
                dilation=dilation,
                num_heads=2 * base.num_k_heads + base.num_v_heads,
                backend=base.conv_backend,
            )
        if mixed_qkv.shape[1] < halo_width:
            raise ValueError(
                "linear attention P2P convolution requires local_seq_len >= "
                f"{halo_width}, got {mixed_qkv.shape[1]}."
            )
        halo = _all_to_all_previous_rank_halo(
            mixed_qkv[:, -halo_width:, :].contiguous(),
            self.cp_mesh,
            self.cp_rank,
            self.cp_size,
        )
        return _causal_conv1d_with_left_halo(
            mixed_qkv,
            halo,
            base.conv1d.weight,
            base.conv1d.bias,
            dilation=dilation,
            num_heads=2 * base.num_k_heads + base.num_v_heads,
            backend=base.conv_backend,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        hidden_states = _local_tensor_at_cp_boundary(hidden_states)
        base = self.module
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(
                hidden_states.dtype
            )

        batch, local_seq_len, _ = hidden_states.shape
        if local_seq_len % 64 != 0:
            raise ValueError(
                "linear attention P2P requires local sequence length divisible "
                f"by 64, got {local_seq_len}."
            )
        mixed_qkv = base.in_proj_qkv(hidden_states)
        z = base.in_proj_z(hidden_states).reshape(
            batch,
            local_seq_len,
            base.num_v_heads,
            base.head_v_dim,
        )
        beta = base.in_proj_b(hidden_states).sigmoid()
        decay = base.in_proj_a(hidden_states)
        mixed_qkv = self._conv1d_with_halo(mixed_qkv)
        query, key, value = torch.split(
            mixed_qkv,
            [base.key_dim, base.key_dim, base.value_dim],
            dim=-1,
        )
        query = query.reshape(
            batch, local_seq_len, base.num_k_heads, base.head_k_dim
        )
        key = key.reshape(batch, local_seq_len, base.num_k_heads, base.head_k_dim)
        value = value.reshape(
            batch, local_seq_len, base.num_v_heads, base.head_v_dim
        )
        g = -base.A_log.float().exp() * F.softplus(
            decay.float() + base.dt_bias
        )
        if base.kv_groups > 1:
            query = query.repeat_interleave(base.kv_groups, dim=2)
            key = key.repeat_interleave(base.kv_groups, dim=2)
        if not is_triton_gdn_available(query):
            raise RuntimeError(
                "linear attention mode='p2p' requires triton-ascend>=3.2.1 "
                "and NPU bfloat16 GDN inputs."
            )

        core_attn_out = _gdn_state_p2p(
            query,
            key,
            value,
            g,
            beta,
            self.cp_mesh,
            self.cp_rank,
            self.cp_size,
        )
        core_attn_out = core_attn_out.reshape(-1, base.head_v_dim)
        core_attn_out = base.norm(core_attn_out, z.reshape(-1, base.head_v_dim))
        core_attn_out = core_attn_out.reshape(
            batch, local_seq_len, base.value_dim
        )
        if hasattr(base, "out_proj_input"):
            core_attn_out = base.out_proj_input(core_attn_out)
        return base.out_proj(core_attn_out)


class LinearAttentionContextParallel(ParallelStyle):
    """Apply Ulysses or optimized state-P2P CP to a Gated DeltaNet module."""

    def __init__(self, *, mode: str = "ulysses") -> None:
        mode = mode.lower()
        if mode not in _SUPPORTED_MODES:
            raise NotImplementedError(
                "LinearAttentionContextParallel supports only mode='ulysses' "
                "and mode='p2p'."
            )
        self.mode = mode

    def apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Patch ``module.forward`` with the selected CP executor."""
        if hasattr(module, "_hp_linear_attention_cp_executor"):
            raise RuntimeError("linear attention context parallelism is already applied.")
        if self.mode == "ulysses":
            executor = LinearAttentionUlyssesCPWrapper(module, device_mesh)
        else:
            module.gdn_backend = "triton"
            module.conv_backend = "triton"
            executor = LinearAttentionP2PCPWrapper(module, device_mesh)
        object.__setattr__(module, "_hp_linear_attention_cp_executor", executor)
        object.__setattr__(module, "_hp_linear_attention_original_forward", module.forward)

        def _forward(*args, **kwargs):
            return executor(*args, **kwargs)

        object.__setattr__(module, "forward", _forward)
        return module


__all__ = ["LinearAttentionContextParallel"]
