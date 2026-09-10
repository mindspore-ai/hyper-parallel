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
"""Ulysses and state-P2P context-parallel execution for KDA."""
# pylint: disable=forbidden-backend-import
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from hyper_parallel.core.context_parallel.context_parallel import _ensure_1d
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.components.modules.kimi_delta_attention import (
    chunk_kda,
    torch_apply_kda_state_summary,
    torch_kda_state_summary,
)
from hyper_parallel.platform import get_platform


platform = get_platform()
_KDA_BACKENDS = frozenset({"eager", "triton"})


def _run_local_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    lower_bound: float,
    chunk_size: int,
    safe_gate: bool,
    backend: str,
) -> torch.Tensor:
    """Run the explicitly selected local KDA implementation."""
    if backend == "triton" and not safe_gate:
        raise RuntimeError("The fused KDA backend requires the lower-bounded gate.")
    output, _ = chunk_kda(
        query,
        key,
        value,
        gate,
        beta,
        a_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        backend=backend,
    )
    return output


def _is_official_kimi_layer(module: nn.Module) -> bool:
    """Whether ``module`` follows the official Kimi K3 projection contract."""
    return hasattr(module, "f_a_proj") and hasattr(module, "f_b_proj")


def _num_value_heads(module: nn.Module) -> int:
    """Return the number of KDA value heads for either supported layer API."""
    if hasattr(module, "num_v_heads"):
        return int(module.num_v_heads)
    return int(module.num_heads)


def _value_head_dim(module: nn.Module) -> int:
    """Return the KDA value head width for either supported layer API."""
    if hasattr(module, "head_v_dim"):
        return int(module.head_v_dim)
    return int(module.head_dim)


def _key_projection_dim(module: nn.Module) -> int:
    """Return the flattened Q/K projection width."""
    if hasattr(module, "key_dim"):
        return int(module.key_dim)
    return int(module.num_heads * module.head_k_dim)


def _value_projection_dim(module: nn.Module) -> int:
    """Return the flattened value projection width."""
    if hasattr(module, "value_dim"):
        return int(module.value_dim)
    return int(_num_value_heads(module) * _value_head_dim(module))


def _gate_lower_bound(module: nn.Module) -> Optional[float]:
    """Return the lower bound used by the safe KDA gate parameterization."""
    return getattr(module, "lower_bound", getattr(module, "gate_lower_bound", None))


def _uses_safe_gate(module: nn.Module) -> bool:
    """Return whether the layer enables the safe lower-bounded gate."""
    return bool(getattr(module, "safe_gate", _gate_lower_bound(module) is not None))


def _uses_short_conv(module: nn.Module) -> bool:
    """Return whether Q/K/V pass through the layer's ShortConvolution modules."""
    return bool(
        getattr(
            module,
            "use_short_conv",
            all(hasattr(module, name) for name in ("q_conv1d", "k_conv1d", "v_conv1d")),
        )
    )


def _forget_gate_projection(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run the module-specific state-decay gate projection."""
    if hasattr(module, "f_proj"):
        return module.f_proj(hidden_states)
    return module.f_b_proj(module.f_a_proj(hidden_states))


def _output_gate_projection(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run the module-specific output-gate projection."""
    if hasattr(module, "g_proj"):
        return module.g_proj(hidden_states)
    return module.g_b_proj(module.g_a_proj(hidden_states))


def _format_layer_output(module: nn.Module, output: torch.Tensor):
    """Preserve the original layer's tensor or three-tuple output contract."""
    if _is_official_kimi_layer(module):
        return output
    return output, None, None


def _global_peer_rank(cp_mesh: DeviceMesh, local_rank: int) -> int:
    """Map one CP-local rank index to its global distributed rank."""
    return int(cp_mesh.rank_list[local_rank])


def _local_tensor_at_cp_boundary(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local sequence shard carried by a CP-boundary tensor."""
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _all_to_all_previous_rank_halo(
    tail: torch.Tensor,
    cp_mesh: DeviceMesh,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Send a causal-convolution halo only to the next CP rank."""
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


def _causal_short_convs_with_cp_halo(
    projected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    convolutions: tuple[nn.Conv1d, nn.Conv1d, nn.Conv1d],
    cp_mesh: DeviceMesh,
    cp_rank: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run three causal depthwise ShortConvs with one packed halo exchange."""
    halo_widths = {
        (conv.kernel_size[0] - 1) * conv.dilation[0] for conv in convolutions
    }
    if len(halo_widths) != 1:
        raise ValueError("KDA P2P requires Q/K/V ShortConv halo widths to match.")
    halo_width = halo_widths.pop()
    if halo_width == 0 or cp_size == 1:
        outputs = []
        for tensor, convolution in zip(projected, convolutions):
            output = F.conv1d(  # pylint: disable=not-callable
                input=tensor.transpose(1, 2),
                weight=convolution.weight,
                bias=convolution.bias,
                stride=convolution.stride,
                padding=convolution.padding,
                dilation=convolution.dilation,
                groups=convolution.groups,
            )
            outputs.append(F.silu(output[:, :, : tensor.shape[1]]).transpose(1, 2))
        return tuple(outputs)
    if projected[0].shape[1] < halo_width:
        raise ValueError(
            "KDA P2P ShortConv requires local_seq_len >= halo width, got "
            f"{projected[0].shape[1]} < {halo_width}."
        )

    channel_sizes = [tensor.shape[-1] for tensor in projected]
    packed_tail = torch.cat(
        [tensor[:, -halo_width:, :] for tensor in projected],
        dim=-1,
    ).contiguous()
    packed_halo = _all_to_all_previous_rank_halo(
        packed_tail,
        cp_mesh,
        cp_rank,
        cp_size,
    )
    halos = torch.split(packed_halo, channel_sizes, dim=-1)
    outputs = []
    for tensor, halo, convolution in zip(projected, halos, convolutions):
        conv_input = torch.cat((halo, tensor), dim=1).transpose(1, 2)
        output = F.conv1d(  # pylint: disable=not-callable
            input=conv_input,
            weight=convolution.weight,
            bias=convolution.bias,
            stride=convolution.stride,
            padding=0,
            dilation=convolution.dilation,
            groups=convolution.groups,
        )
        outputs.append(F.silu(output).transpose(1, 2))
    return tuple(outputs)


def _slice_local_heads(
    tensor: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Return the current CP rank's contiguous head slice."""
    if tensor.shape[0] % cp_size != 0:
        raise ValueError(
            f"KDA head dimension {tensor.shape[0]} must be divisible by "
            f"cp_size {cp_size}."
        )
    local_heads = tensor.shape[0] // cp_size
    return tensor.narrow(0, cp_rank * local_heads, local_heads)


def _differentiable_all_to_all_shard(
    tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    *,
    split_dim: int,
    concat_dim: int,
) -> torch.Tensor:
    """Redistribute a local tensor from concat-dim shard to split-dim shard."""
    split_count = device_mesh.size()
    if split_count == 1:
        return tensor

    original_shape = tuple(tensor.shape)
    if original_shape[split_dim] % split_count != 0:
        raise ValueError(
            f"KDA all-to-all split dimension {split_dim} with size "
            f"{original_shape[split_dim]} must be divisible by cp_size "
            f"{split_count}."
        )
    split_size = original_shape[split_dim] // split_count
    final_shape = list(original_shape)
    final_shape[split_dim] = split_size
    final_shape[concat_dim] *= split_count

    reshape_dims = list(original_shape)
    reshape_dims[split_dim] = split_count
    reshape_dims.insert(split_dim + 1, split_size)
    permutation = list(range(len(reshape_dims)))
    permutation.remove(split_dim)
    permutation.insert(0, split_dim)
    all_to_all_input = tensor.reshape(reshape_dims).permute(permutation).contiguous()
    all_to_all_input = all_to_all_input.flatten(0, 1)

    split_length = all_to_all_input.shape[0] // split_count
    splits = [split_length] * split_count
    output = platform.differentiable_all_to_all_single(
        all_to_all_input,
        splits,
        splits,
        group=device_mesh.get_group(),
    )

    output_reshape = list(output.shape)
    output_reshape[0] = split_count
    output_reshape.insert(1, output.shape[0] // split_count)
    permutation = list(range(len(output_reshape)))
    rank_dimension = permutation.pop(0)
    permutation.insert(concat_dim, rank_dimension)
    output = output.reshape(output_reshape).permute(permutation).contiguous()

    output_shape = list(output.shape)
    output_shape[concat_dim] *= output_shape[concat_dim + 1]
    output_shape.pop(concat_dim + 1)
    return output.reshape(output_shape).view(final_shape)


class _RecvKDAInitialState(torch.autograd.Function):
    """Receive a KDA initial state and return its gradient to the sender."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        ctx: Any,
        anchor: torch.Tensor,
        cp_group: Any,
        prev_rank: int,
        state_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Receive the initial recurrent state from the previous CP rank."""
        state = torch.empty(state_shape, device=anchor.device, dtype=torch.float32)
        dist.recv(state, src=prev_rank, group=cp_group)
        ctx.cp_group = cp_group
        ctx.prev_rank = prev_rank
        return state

    @staticmethod
    def backward(
        ctx: Any,
        grad_state: Optional[torch.Tensor],
    ) -> tuple[None, None, None, None]:
        """Send the accumulated initial-state gradient to the previous rank."""
        if grad_state is None:
            raise RuntimeError("KDA P2P backward is missing the initial-state gradient.")
        dist.send(grad_state.contiguous(), dst=ctx.prev_rank, group=ctx.cp_group)
        return None, None, None, None


class _SendKDAFinalState(torch.autograd.Function):
    """Send a KDA final state and receive its gradient during backward."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        ctx: Any,
        final_state: torch.Tensor,
        cp_group: Any,
        next_rank: int,
    ) -> torch.Tensor:
        """Send the final recurrent state to the next CP rank."""
        dist.send(final_state.contiguous(), dst=next_rank, group=cp_group)
        ctx.cp_group = cp_group
        ctx.next_rank = next_rank
        ctx.state_shape = tuple(final_state.shape)
        ctx.state_dtype = final_state.dtype
        return final_state.new_zeros(())

    @staticmethod
    def backward(
        ctx: Any,
        grad_token: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        """Receive the final-state gradient from the next rank."""
        grad_state = torch.empty(
            ctx.state_shape,
            device=grad_token.device,
            dtype=ctx.state_dtype,
        )
        dist.recv(grad_state, src=ctx.next_rank, group=ctx.cp_group)
        return grad_state, None, None


def _eager_kda_state_p2p(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    lower_bound: float,
    chunk_size: int,
    cp_group: Any,
    prev_rank: int,
    next_rank: int,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Run the differentiable Torch reference for affine state-P2P KDA."""
    if cp_size == 1:
        return _run_local_kda(
            query,
            key,
            value,
            gate,
            beta,
            a_log=a_log,
            dt_bias=dt_bias,
            scale=scale,
            lower_bound=lower_bound,
            chunk_size=chunk_size,
            safe_gate=True,
            backend="eager",
        )

    state_ext = None
    transition = None
    if cp_rank < cp_size - 1:
        state_ext, transition = torch_kda_state_summary(
            query,
            key,
            value,
            gate,
            beta,
            a_log=a_log,
            dt_bias=dt_bias,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=lower_bound,
            chunk_size=chunk_size,
        )

    initial_state = None
    if cp_rank > 0:
        state_shape = (
            query.shape[0],
            value.shape[2],
            query.shape[-1],
            value.shape[-1],
        )
        initial_state = _RecvKDAInitialState.apply(
            query,
            cp_group,
            prev_rank,
            state_shape,
        )

    send_token = None
    if cp_rank < cp_size - 1:
        final_state = torch_apply_kda_state_summary(
            state_ext,
            transition,
            initial_state,
        )
        send_token = _SendKDAFinalState.apply(final_state, cp_group, next_rank)

    output, _ = chunk_kda(
        query,
        key,
        value,
        gate,
        beta,
        a_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        backend="eager",
    )
    if send_token is not None:
        output = output + send_token.to(output.dtype) * 0
    return output


class KimiDeltaAttentionUlyssesCP(nn.Module):
    """Run KDA on sequence-sharded projected tensors with Ulysses all-to-all.

    The input and output layout is token-first with a local sequence shard:

    ``[B, S/P, H, D] -> [B, S, H/P, D] -> [B, S/P, H, D]``.

    This first-stage interface intentionally starts after projections and
    ShortConv and ends before the output norm/gate/projection. It is therefore
    independent of one particular Kimi K3 model class. Training uses a zero
    initial recurrent state and does not materialize a final state.
    """

    def __init__(
        self,
        device_mesh: DeviceMesh,
        *,
        chunk_size: int = 64,
        lower_bound: float = -5.0,
        safe_gate: bool = True,
        backend: str = "eager",
    ) -> None:
        """Initialize the KDA Ulysses executor for one 1-D CP mesh."""
        super().__init__()
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        if lower_bound >= 0:
            raise ValueError(f"lower_bound must be negative, got {lower_bound}.")
        backend = backend.lower()
        if backend not in _KDA_BACKENDS:
            raise ValueError(
                f"unsupported KDA backend {backend!r}; "
                f"expected one of {sorted(_KDA_BACKENDS)}."
            )
        self.cp_mesh = _ensure_1d(device_mesh)
        self.cp_size = self.cp_mesh.size()
        self.cp_rank = self.cp_mesh.get_local_rank()
        self.chunk_size = chunk_size
        self.lower_bound = lower_bound
        self.safe_gate = safe_gate
        self.backend = backend
        self.seq_dim = 1
        self.head_dim = 2

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> None:
        """Validate the projected-tensor boundary and Ulysses divisibility."""
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4 or gate.dim() != 4:
            raise ValueError("query, key, value, and gate must be rank-4 tensors.")
        if beta.dim() != 3:
            raise ValueError("beta must be a rank-3 tensor.")
        if query.shape != key.shape:
            raise ValueError("query and key must have identical shapes.")

        batch_size, local_seq_len, num_k_heads, k_head_dim = query.shape
        expected_prefix = (batch_size, local_seq_len)
        if value.shape[:2] != expected_prefix:
            raise ValueError("value batch and sequence dimensions must match query.")
        num_v_heads = value.shape[2]
        if gate.shape != (batch_size, local_seq_len, num_v_heads, k_head_dim):
            raise ValueError(
                "gate must have shape [B, S_local, num_v_heads, k_head_dim]."
            )
        if beta.shape != (batch_size, local_seq_len, num_v_heads):
            raise ValueError("beta must have shape [B, S_local, num_v_heads].")
        if num_v_heads % num_k_heads != 0:
            raise ValueError("num_v_heads must be divisible by num_k_heads.")
        if num_k_heads % self.cp_size != 0:
            raise ValueError(
                f"KDA num_k_heads ({num_k_heads}) must be divisible by "
                f"cp_size ({self.cp_size}) for Ulysses CP."
            )
        if num_v_heads % self.cp_size != 0:
            raise ValueError(
                f"KDA num_v_heads ({num_v_heads}) must be divisible by "
                f"cp_size ({self.cp_size}) for Ulysses CP."
            )
        if a_log.numel() != num_v_heads:
            raise ValueError(
                f"a_log must contain {num_v_heads} elements, got {a_log.numel()}."
            )
        if dt_bias.numel() != num_v_heads * k_head_dim:
            raise ValueError(
                f"dt_bias must contain {num_v_heads * k_head_dim} elements, "
                f"got {dt_bias.numel()}."
            )

    def _seq_to_head(self, tensor: torch.Tensor) -> torch.Tensor:
        """Redistribute a local sequence shard into a local head shard."""
        return _differentiable_all_to_all_shard(
            tensor,
            self.cp_mesh,
            split_dim=self.head_dim,
            concat_dim=self.seq_dim,
        )

    def _head_to_seq(self, tensor: torch.Tensor) -> torch.Tensor:
        """Redistribute a local head shard back into a local sequence shard."""
        return _differentiable_all_to_all_shard(
            tensor,
            self.cp_mesh,
            split_dim=self.seq_dim,
            concat_dim=self.head_dim,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        *,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run chunkwise KDA using full sequence and local heads."""
        self._validate_inputs(query, key, value, gate, beta, a_log, dt_bias)
        num_v_heads = value.shape[2]
        k_head_dim = query.shape[-1]

        query = self._seq_to_head(query)
        key = self._seq_to_head(key)
        value = self._seq_to_head(value)
        gate = self._seq_to_head(gate)
        beta = self._seq_to_head(beta)

        local_a_log = _slice_local_heads(
            a_log.reshape(num_v_heads),
            self.cp_rank,
            self.cp_size,
        )
        local_dt_bias = _slice_local_heads(
            dt_bias.reshape(num_v_heads, k_head_dim),
            self.cp_rank,
            self.cp_size,
        )
        output = _run_local_kda(
            query,
            key,
            value,
            gate,
            beta,
            a_log=local_a_log,
            dt_bias=local_dt_bias,
            scale=scale,
            lower_bound=self.lower_bound,
            chunk_size=self.chunk_size,
            safe_gate=self.safe_gate,
            backend=self.backend,
        )
        return self._head_to_seq(output)


class KimiDeltaAttentionP2PCP(nn.Module):
    """Run fused KDA on local sequence shards with affine state P2P."""

    def __init__(
        self,
        device_mesh: DeviceMesh,
        *,
        chunk_size: int = 64,
        lower_bound: float = -5.0,
        safe_gate: bool = True,
        backend: str = "eager",
    ) -> None:
        """Initialize the KDA state-P2P executor for one 1-D CP mesh."""
        super().__init__()
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        backend = backend.lower()
        if backend not in _KDA_BACKENDS:
            raise ValueError(
                f"unsupported KDA backend {backend!r}; "
                f"expected one of {sorted(_KDA_BACKENDS)}."
            )
        self.cp_mesh = _ensure_1d(device_mesh)
        self.cp_size = self.cp_mesh.size()
        self.cp_rank = self.cp_mesh.get_local_rank()
        self.cp_group = self.cp_mesh.get_group()
        self.prev_rank = _global_peer_rank(
            self.cp_mesh,
            max(self.cp_rank - 1, 0),
        )
        self.next_rank = _global_peer_rank(
            self.cp_mesh,
            min(self.cp_rank + 1, self.cp_size - 1),
        )
        self.chunk_size = chunk_size
        self.lower_bound = lower_bound
        self.safe_gate = safe_gate
        self.backend = backend

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        *,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run one local KDA segment and propagate its recurrent state."""
        if self.cp_size == 1:
            return _run_local_kda(
                query,
                key,
                value,
                gate,
                beta,
                a_log=a_log,
                dt_bias=dt_bias,
                scale=scale,
                lower_bound=self.lower_bound,
                chunk_size=self.chunk_size,
                safe_gate=self.safe_gate,
                backend=self.backend,
            )
        if self.backend == "eager":
            return _eager_kda_state_p2p(
                query,
                key,
                value,
                gate,
                beta,
                a_log=a_log,
                dt_bias=dt_bias,
                scale=scale,
                lower_bound=self.lower_bound,
                chunk_size=self.chunk_size,
                cp_group=self.cp_group,
                prev_rank=self.prev_rank,
                next_rank=self.next_rank,
                cp_rank=self.cp_rank,
                cp_size=self.cp_size,
            )

        from hyper_parallel.components.functional.kimi_delta_attention import (  # pylint: disable=import-outside-toplevel
            fused_chunk_kda_p2p,
        )
        return fused_chunk_kda_p2p(
            query,
            key,
            value,
            gate,
            beta,
            a_log=a_log.float(),
            dt_bias=dt_bias.float(),
            cp_group=self.cp_group,
            prev_rank=self.prev_rank,
            next_rank=self.next_rank,
            cp_rank=self.cp_rank,
            cp_size=self.cp_size,
            scale=scale,
            lower_bound=self.lower_bound,
            chunk_size=self.chunk_size,
            safe_gate=self.safe_gate,
        )


class KimiDeltaAttentionLayerUlyssesCP(KimiDeltaAttentionUlyssesCP):
    """Wrap a training-time Kimi K3 attention layer with Ulysses CP.

    The wrapped module keeps ownership of all model parameters. Its local
    sequence shard is projected first, then Q/K/V, per-dimension gate logits,
    and beta logits are packed into one sequence-to-head all-to-all. Each CP
    rank runs the wrapped layer's ShortConv weights and fused KDA core for its
    local head shard and the full sequence. The reverse all-to-all restores a
    local sequence shard before the original output gate, norm, and projection.

    This first integration targets dense training batches. Inference cache and
    padded or packed variable-length sequences require separate CP metadata
    handling and are rejected explicitly.
    """

    def __init__(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
        *,
        chunk_size: int = 64,
        backend: str = "eager",
    ) -> None:
        """Initialize the full-layer adapter without replacing its parameters."""
        lower_bound = _gate_lower_bound(module)
        if lower_bound is None:
            raise ValueError(
                "Kimi K3 Ulysses CP requires the lower-bounded gate parameterization."
            )
        super().__init__(
            device_mesh,
            chunk_size=chunk_size,
            lower_bound=float(lower_bound),
            safe_gate=_uses_safe_gate(module),
            backend=backend,
        )
        self.module = module
        self._validate_module()

    def _validate_module(self) -> None:
        """Validate the first-stage KimiDeltaAttention module contract."""
        required_attributes = (
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "o_norm",
            "o_proj",
            "A_log",
            "dt_bias",
            "num_heads",
            "head_k_dim",
        )
        missing = [name for name in required_attributes if not hasattr(self.module, name)]
        if missing:
            raise TypeError(
                "Kimi K3 layer is missing required attributes: "
                + ", ".join(missing)
            )
        if self.module.num_heads % self.cp_size != 0:
            raise ValueError(
                f"KDA num_heads ({self.module.num_heads}) must be divisible by "
                f"cp_size ({self.cp_size})."
            )
        num_v_heads = _num_value_heads(self.module)
        if num_v_heads % self.cp_size != 0:
            raise ValueError(
                f"KDA num_v_heads ({num_v_heads}) must be divisible "
                f"by cp_size ({self.cp_size})."
            )
        if num_v_heads % self.module.num_heads != 0:
            raise ValueError("KDA num_v_heads must be divisible by num_heads.")
        if not (hasattr(self.module, "f_proj") or _is_official_kimi_layer(self.module)):
            raise TypeError("Kimi K3 layer is missing its forget-gate projection.")
        if not (
            hasattr(self.module, "g_proj")
            or all(hasattr(self.module, name) for name in ("g_a_proj", "g_b_proj"))
        ):
            raise TypeError("Kimi K3 layer is missing its output-gate projection.")
        if getattr(self.module, "allow_neg_eigval", False):
            raise NotImplementedError(
                "Kimi K3 Ulysses CP does not yet support allow_neg_eigval=True."
            )

        if _uses_short_conv(self.module):
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                if not hasattr(self.module, name):
                    raise TypeError(f"Kimi K3 layer is missing {name}.")
                convolution = getattr(self.module, name)
                if convolution.stride != (1,) or convolution.dilation != (1,):
                    raise ValueError(
                        "Kimi K3 Ulysses CP supports ShortConv stride=1 and "
                        "dilation=1 only."
                    )
                if convolution.groups != convolution.in_channels:
                    raise ValueError("Kimi K3 Ulysses CP expects depthwise ShortConv.")
                if getattr(convolution, "activation", None) not in ("silu", "swish"):
                    raise ValueError(
                        "Kimi K3 Ulysses CP expects ShortConv SiLU activation."
                    )

    def _pack_projected_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack all projected paths into one sequence-to-head all-to-all."""
        if self.cp_size == 1:
            return query, key, value, gate, beta

        base = self.module
        split_sizes = (
            _key_projection_dim(base) // self.cp_size,
            _key_projection_dim(base) // self.cp_size,
            _value_projection_dim(base) // self.cp_size,
            _num_value_heads(base) * base.head_k_dim // self.cp_size,
            _num_value_heads(base) // self.cp_size,
        )
        projected = (query, key, value, gate, beta)
        chunks = [torch.split(tensor, size, dim=-1) for tensor, size in zip(projected, split_sizes)]
        rank_major = [
            torch.cat(parts, dim=-1)
            for parts in zip(*chunks)
        ]
        packed = self._seq_to_head(torch.cat(rank_major, dim=-1).contiguous())
        return torch.split(packed, split_sizes, dim=-1)

    def _local_short_conv(
        self,
        projected: torch.Tensor,
        convolution: nn.Conv1d,
    ) -> torch.Tensor:
        """Run one wrapped ShortConv on this rank's contiguous channel slice."""
        local_channels = projected.shape[-1]
        channel_start = self.cp_rank * local_channels
        weight = convolution.weight.narrow(0, channel_start, local_channels)
        bias = convolution.bias
        if bias is not None:
            bias = bias.narrow(0, channel_start, local_channels)
        output = F.conv1d(  # pylint: disable=not-callable
            projected.transpose(1, 2),
            weight,
            bias,
            stride=convolution.stride,
            padding=convolution.padding,
            dilation=convolution.dilation,
            groups=local_channels,
        )
        output = output[:, :, : projected.shape[1]].transpose(1, 2)
        return F.silu(output)

    def forward(  # pylint: disable=arguments-renamed
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, None]:
        """Run the real Kimi layer boundary using KDA and Ulysses CP."""
        if not self.module.training:
            raise NotImplementedError("Kimi K3 Ulysses CP currently supports training only.")
        if past_key_values is not None or use_cache:
            raise NotImplementedError("Kimi K3 Ulysses CP does not support recurrent cache.")
        if output_attentions:
            raise NotImplementedError("KDA does not materialize attention weights.")
        if kwargs.get("cu_seqlens") is not None:
            raise NotImplementedError("Packed variable-length KDA CP is not implemented.")
        if attention_mask is not None and not bool(attention_mask.bool().all().item()):
            raise NotImplementedError("Padded KDA CP batches are not implemented.")

        hidden_states = _local_tensor_at_cp_boundary(hidden_states)
        base = self.module
        batch_size, local_seq_len, _ = hidden_states.shape
        query = base.q_proj(hidden_states)
        key = base.k_proj(hidden_states)
        value = base.v_proj(hidden_states)
        gate = _forget_gate_projection(base, hidden_states)
        beta = base.b_proj(hidden_states)
        output_gate = _output_gate_projection(base, hidden_states).reshape(
            batch_size,
            local_seq_len,
            _num_value_heads(base),
            _value_head_dim(base),
        )

        query, key, value, gate, beta = self._pack_projected_inputs(
            query,
            key,
            value,
            gate,
            beta,
        )
        if _uses_short_conv(base):
            query = self._local_short_conv(query, base.q_conv1d)
            key = self._local_short_conv(key, base.k_conv1d)
            value = self._local_short_conv(value, base.v_conv1d)
        else:
            query, key, value = (F.silu(tensor) for tensor in (query, key, value))

        full_seq_len = query.shape[1]
        local_k_heads = base.num_heads // self.cp_size
        local_v_heads = _num_value_heads(base) // self.cp_size
        query = query.reshape(
            batch_size, full_seq_len, local_k_heads, base.head_k_dim
        )
        key = key.reshape(batch_size, full_seq_len, local_k_heads, base.head_k_dim)
        value = value.reshape(
            batch_size, full_seq_len, local_v_heads, _value_head_dim(base)
        )
        gate = gate.reshape(
            batch_size, full_seq_len, local_v_heads, base.head_k_dim
        )
        beta = beta.reshape(batch_size, full_seq_len, local_v_heads)

        local_a_log = _slice_local_heads(
            base.A_log.reshape(_num_value_heads(base)),
            self.cp_rank,
            self.cp_size,
        )
        local_dt_bias = _slice_local_heads(
            base.dt_bias.reshape(_num_value_heads(base), base.head_k_dim),
            self.cp_rank,
            self.cp_size,
        )
        output = _run_local_kda(
            query,
            key,
            value,
            gate,
            beta,
            a_log=local_a_log,
            dt_bias=local_dt_bias,
            scale=None,
            lower_bound=self.lower_bound,
            chunk_size=self.chunk_size,
            safe_gate=self.safe_gate,
            backend=self.backend,
        )
        output = self._head_to_seq(output)
        output = base.o_norm(output, output_gate)
        output = output.reshape(
            batch_size, local_seq_len, _value_projection_dim(base)
        )
        output = base.o_proj(output)
        return _format_layer_output(base, output)


class KimiDeltaAttentionLayerP2PCP(KimiDeltaAttentionP2PCP):
    """Wrap a training-time Kimi K3 attention layer with state-P2P CP."""

    def __init__(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
        *,
        chunk_size: int = 64,
        backend: str = "eager",
    ) -> None:
        """Initialize the full-layer state-P2P adapter."""
        lower_bound = _gate_lower_bound(module)
        if lower_bound is None:
            raise ValueError(
                "Kimi K3 P2P CP requires the lower-bounded gate parameterization."
            )
        super().__init__(
            device_mesh,
            chunk_size=chunk_size,
            lower_bound=float(lower_bound),
            safe_gate=_uses_safe_gate(module),
            backend=backend,
        )
        self.module = module
        self._validate_module()

    def _validate_module(self) -> None:
        """Validate the KimiDeltaAttention contract used by state-P2P CP."""
        required_attributes = (
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "o_norm",
            "o_proj",
            "A_log",
            "dt_bias",
            "num_heads",
            "head_k_dim",
        )
        missing = [name for name in required_attributes if not hasattr(self.module, name)]
        if missing:
            raise TypeError(
                "Kimi K3 layer is missing required attributes: "
                + ", ".join(missing)
            )
        num_v_heads = _num_value_heads(self.module)
        if num_v_heads % self.module.num_heads:
            raise ValueError("KDA num_v_heads must be divisible by num_heads.")
        if self.backend == "triton" and (
            self.module.head_k_dim != 128 or _value_head_dim(self.module) != 128
        ):
            raise NotImplementedError(
                "Kimi K3 P2P CP currently requires key/value head dimensions 128."
            )
        if getattr(self.module, "allow_neg_eigval", False):
            raise NotImplementedError(
                "Kimi K3 P2P CP does not support allow_neg_eigval=True."
            )
        if not (hasattr(self.module, "f_proj") or _is_official_kimi_layer(self.module)):
            raise TypeError("Kimi K3 layer is missing its forget-gate projection.")
        if not (
            hasattr(self.module, "g_proj")
            or all(hasattr(self.module, name) for name in ("g_a_proj", "g_b_proj"))
        ):
            raise TypeError("Kimi K3 layer is missing its output-gate projection.")
        if _uses_short_conv(self.module):
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                if not hasattr(self.module, name):
                    raise TypeError(f"Kimi K3 layer is missing {name}.")
                convolution = getattr(self.module, name)
                if convolution.stride != (1,) or convolution.dilation != (1,):
                    raise ValueError(
                        "Kimi K3 P2P CP supports ShortConv stride=1 and "
                        "dilation=1 only."
                    )
                if convolution.groups != convolution.in_channels:
                    raise ValueError("Kimi K3 P2P CP expects depthwise ShortConv.")
                if getattr(convolution, "activation", None) not in ("silu", "swish"):
                    raise ValueError(
                        "Kimi K3 P2P CP expects ShortConv SiLU activation."
                    )

    def forward(  # pylint: disable=arguments-renamed
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, None]:
        """Run the real Kimi layer boundary with fused state-P2P KDA."""
        if not self.module.training:
            raise NotImplementedError("Kimi K3 P2P CP currently supports training only.")
        if past_key_values is not None or use_cache:
            raise NotImplementedError("Kimi K3 P2P CP does not support recurrent cache.")
        if output_attentions:
            raise NotImplementedError("KDA does not materialize attention weights.")
        if kwargs.get("cu_seqlens") is not None:
            raise NotImplementedError("Packed variable-length KDA P2P is not implemented.")
        if attention_mask is not None and not bool(attention_mask.bool().all().item()):
            raise NotImplementedError("Padded KDA P2P batches are not implemented.")

        hidden_states = _local_tensor_at_cp_boundary(hidden_states)
        base = self.module
        batch_size, local_seq_len, _ = hidden_states.shape
        query = base.q_proj(hidden_states)
        key = base.k_proj(hidden_states)
        value = base.v_proj(hidden_states)
        if _uses_short_conv(base):
            query, key, value = _causal_short_convs_with_cp_halo(  # pylint: disable=unbalanced-tuple-unpacking
                (query, key, value),
                (base.q_conv1d, base.k_conv1d, base.v_conv1d),
                self.cp_mesh,
                self.cp_rank,
                self.cp_size,
            )
        else:
            query, key, value = (F.silu(tensor) for tensor in (query, key, value))

        query = query.reshape(
            batch_size,
            local_seq_len,
            base.num_heads,
            base.head_k_dim,
        )
        key = key.reshape(
            batch_size,
            local_seq_len,
            base.num_heads,
            base.head_k_dim,
        )
        value = value.reshape(
            batch_size,
            local_seq_len,
            _num_value_heads(base),
            _value_head_dim(base),
        )
        gate = _forget_gate_projection(base, hidden_states).reshape(
            batch_size,
            local_seq_len,
            _num_value_heads(base),
            base.head_k_dim,
        )
        beta = base.b_proj(hidden_states).reshape(
            batch_size,
            local_seq_len,
            _num_value_heads(base),
        )
        output_gate = _output_gate_projection(base, hidden_states).reshape(
            batch_size,
            local_seq_len,
            _num_value_heads(base),
            _value_head_dim(base),
        )

        output = self._run_local_kda(
            query,
            key,
            value,
            gate,
            beta,
            a_log=base.A_log,
            dt_bias=base.dt_bias,
        )
        output = base.o_norm(output, output_gate)
        output = output.reshape(
            batch_size, local_seq_len, _value_projection_dim(base)
        )
        output = base.o_proj(output)
        return _format_layer_output(base, output)

    def _run_local_kda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        *,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch the projected tensors to the state-P2P core."""
        return KimiDeltaAttentionP2PCP.forward(
            self,
            query,
            key,
            value,
            gate,
            beta,
            a_log=a_log,
            dt_bias=dt_bias,
        )


__all__ = [
    "KimiDeltaAttentionLayerP2PCP",
    "KimiDeltaAttentionLayerUlyssesCP",
    "KimiDeltaAttentionP2PCP",
    "KimiDeltaAttentionUlyssesCP",
]
