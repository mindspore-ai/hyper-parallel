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

"""Synchronous, opt-in Ascend stream-KV attention for the Torch CP interfaces.

This module owns the first-order VJP, including FP32 Ulysses/replica gradient
return. Model projections, norms, positions and the output gate remain outside.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.distributed.context_parallel._stream_kv_layout import (
    _add_partial, _gather, _panel_bounds, _pieces, _redistribute, _reduce_panel,
)
from hyper_parallel.distributed.context_parallel.collectives import ulysses_head_to_seq, ulysses_seq_to_head


@dataclass(frozen=True)
class StreamKVConfig:
    """Bound communication and compute; preserve the caller's contiguous token order.

    Args:
        owner_panel_tokens: Maximum tokens supplied by each KV owner per gather.
        key_block_tokens: Maximum keys passed to one FULL or CAUSAL FA call.
        query_head_chunk: Query heads per call within one compact KV-head group.
            None processes that entire group. This does not chunk Q projection.
        causal_load_balance: Mirror early/late Q halves and pair KV stripes inside
            attention, restoring caller order on return. The per-owner budget
            includes both stripes; an odd budget uses one fewer token.
    """

    owner_panel_tokens: int
    key_block_tokens: int
    query_head_chunk: int | None = None
    causal_load_balance: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.causal_load_balance, bool):
            raise ValueError("causal_load_balance must be a bool")
        for name in ("owner_panel_tokens", "key_block_tokens", "query_head_chunk"):
            value = getattr(self, name)
            if value is None and name == "query_head_chunk":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.causal_load_balance and self.owner_panel_tokens < 2:
            raise ValueError("Balanced panels require at least two owner tokens")


@lru_cache(maxsize=8)
def _causal_mask(device):
    return torch.ones((2048, 2048), device=device, dtype=torch.bool).triu_(1)


def _npu_ops() -> Any:
    # Optional Ascend extension: importing the CP interface must remain possible on CPU/CUDA.
    import torch_npu  # pylint: disable=import-outside-toplevel
    return torch_npu


def _validate_tensors(query, key, value):
    """Validate the local BNSD contract before issuing any collective."""
    if not all(isinstance(tensor, torch.Tensor) for tensor in (query, key, value)):
        raise ValueError("Q/K/V must be local torch.Tensor inputs")
    if any(tensor.ndim != 4 or tensor.shape[0] != 1 or min(tensor.shape) <= 0
           for tensor in (query, key, value)):
        raise ValueError("Stream KV requires nonempty B1 BNSD local tensors")
    if key.shape != value.shape or query.shape[2:] != key.shape[2:]:
        raise ValueError("Q/K/V must share local sequence length and head dimension; K/V shapes must match")
    if len({(tensor.device, tensor.dtype) for tensor in (query, key, value)}) != 1:
        raise ValueError("Q/K/V must share dtype and device")
    if query.device.type != "npu" or query.dtype != torch.bfloat16 or query.shape[-1] != 256:
        raise ValueError("Initial stream KV backend requires Ascend BF16 with head dimension 256")


def _head_stages(query_heads, kv_heads, chunk):
    """Keep every query-head stage inside one compact KV-head group."""
    group = query_heads // kv_heads
    width = group if chunk is None else min(chunk, group)
    for kv_head in range(kv_heads):
        for start in range(kv_head * group, (kv_head + 1) * group, width):
            yield start, min(start + width, (kv_head + 1) * group), kv_head


def _merge(state, partial):
    """Merge block-normalized outputs using FP32 global softmax statistics."""
    output, maximum, total = state
    block_output, block_max, block_sum = partial
    merged_max = torch.maximum(maximum, block_max)
    old_weight = (maximum - merged_max).exp() * total
    new_weight = (block_max - merged_max).exp() * block_sum
    merged_sum = old_weight + new_weight
    output.mul_((old_weight / merged_sum)[..., :1])
    weighted = block_output.float()
    weighted.mul_((new_weight / merged_sum)[..., :1])
    weighted.add_(output)
    output.copy_(weighted)
    maximum.copy_(merged_max)
    total.copy_(merged_sum)


def _forward(query, key, value, mesh, scale, config):
    """Scan KV panels and return the globally normalized output and statistics."""
    length = query.shape[2]
    output = torch.zeros_like(query, dtype=torch.float32)
    maximum = torch.full((*query.shape[:3], 8), -torch.inf, device=query.device, dtype=torch.float32)
    total = torch.zeros_like(maximum)
    mask = _causal_mask(query.device)
    balanced = config.causal_load_balance
    for start, end in _panel_bounds(length, config.owner_panel_tokens, balanced):
        panel_k, panel_v = _gather(key, value, mesh, start, end, balanced)
        for first, last, kv_head in _head_stages(query.shape[1], key.shape[1], config.query_head_chunk):
            for begin, finish, q_start, q_end, causal, next_tokens in _pieces(
                    mesh.get_local_rank(), mesh.size(), end - start, start, length, config.key_block_tokens, balanced):
                selection = (slice(None), slice(first, last), slice(q_start, q_end))
                # torch_npu registers these operators dynamically; its static module has no declarations.
                partial = _npu_ops().npu_fusion_attention(  # pylint: disable=no-member
                    query[selection].contiguous(), panel_k[:, kv_head:kv_head + 1, begin:finish].contiguous(),
                    panel_v[:, kv_head:kv_head + 1, begin:finish].contiguous(), last - first, "BNSD",
                    atten_mask=mask if causal else None, scale=scale, keep_prob=1.0,
                    sparse_mode=4 if causal else 0, pre_tockens=2**31 - 1, next_tockens=next_tokens)[:3]
                _merge((output[selection], maximum[selection], total[selection]), partial)
                del partial
        del panel_k, panel_v
    return output.to(query.dtype), maximum, total


def _backward(query, key, value, output, maximum, total, incoming, mesh, scale, config):
    """Use final global normalization for every partial VJP, then SUM to KV owners.

    ``owned`` stores this owner's dK/dV in token-major order. ``send`` stores
    the current panel in owner-major order so a single ReduceScatter sums all
    query-owner contributions after every head stage has finished.
    """
    length, dim = query.shape[2:]
    dq = torch.zeros_like(query, dtype=torch.float32)
    owned = torch.empty((length, 2, 1, key.shape[1], dim), device=key.device, dtype=torch.float32)
    mask = _causal_mask(query.device)
    balanced = config.causal_load_balance
    for start, end in _panel_bounds(length, config.owner_panel_tokens, balanced):
        panel_k, panel_v = _gather(key, value, mesh, start, end, balanced)
        send = torch.zeros((mesh.size() * (end - start) * (2 if balanced else 1), 2, 1, key.shape[1], dim),
                           device=key.device, dtype=torch.float32)
        for first, last, kv_head in _head_stages(query.shape[1], key.shape[1], config.query_head_chunk):
            for begin, finish, q_start, q_end, causal, next_tokens in _pieces(
                    mesh.get_local_rank(), mesh.size(), end - start, start, length, config.key_block_tokens, balanced):
                selection = (slice(None), slice(first, last), slice(q_start, q_end))
                # The native outputs must be released before the next FA allocation.
                dq_block, dk_block, dv_block = _npu_ops().npu_fusion_attention_grad(  # pylint: disable=no-member
                    query[selection].float().contiguous(), panel_k[:, kv_head:kv_head + 1, begin:finish].float(),
                    panel_v[:, kv_head:kv_head + 1, begin:finish].float(), incoming[selection].float().contiguous(),
                    last - first, "BNSD", atten_mask=mask if causal else None,
                    softmax_max=maximum[selection].contiguous(), softmax_sum=total[selection].contiguous(),
                    attention_in=output[selection].float().contiguous(), scale_value=scale, keep_prob=1.0,
                    sparse_mode=4 if causal else 0, pre_tockens=2**31 - 1, next_tockens=next_tokens)[:3]
                dq[selection].add_(dq_block)
                del dq_block
                _add_partial(send, dk_block, begin, end - start, 0, kv_head, balanced)
                del dk_block
                _add_partial(send, dv_block, begin, end - start, 1, kv_head, balanced)
                del dv_block
        del panel_k, panel_v
        _reduce_panel(owned, send, mesh, start, end, balanced)
        del send
    return dq, *(owned[:, component].permute(1, 2, 0, 3).contiguous() for component in range(2))


class _StreamKV(torch.autograd.Function):
    """Keep all gradient collectives FP32 until the local Q/K/V input boundary."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                ulysses_mesh: DeviceMesh, kv_mesh: DeviceMesh, scale: float, config: StreamKVConfig) -> torch.Tensor:
        """Save post-U owned Q/K/V and final global normalization.

        Args:
            ctx: Per-call autograd context owning the saved state.
            query: Contiguous-owner BNSD query input.
            key: Compact BNSD key input with global RoPE already applied.
            value: Compact BNSD value input.
            ulysses_mesh: Existing head/sequence exchange subgroup.
            kv_mesh: Existing KV-owner subgroup.
            scale: Positive attention score scale.
            config: Immutable communication and compute bounds.

        Returns:
            Attention output in the caller's original local BNSD order.
        """
        replicas = max(1, ulysses_mesh.size() // key.shape[1])
        ctx.replicas, ctx.kv_heads, ctx.dtype = replicas, key.shape[1], query.dtype
        if replicas > 1:
            key, value = (tensor.repeat_interleave(replicas, 1) for tensor in (key, value))
        query, key, value = (ulysses_seq_to_head(tensor, 2, 1, ulysses_mesh) for tensor in (query, key, value))
        if config.causal_load_balance:
            query = _redistribute(query, kv_mesh)
            key = _redistribute(key, kv_mesh)
            value = _redistribute(value, kv_mesh)
        with torch.autocast("npu", enabled=False):
            output, maximum, total = _forward(query, key, value, kv_mesh, scale, config)
        ctx.save_for_backward(query, key, value, output, maximum, total)
        ctx.ulysses_mesh, ctx.kv_mesh, ctx.scale, ctx.config = ulysses_mesh, kv_mesh, scale, config
        if config.causal_load_balance:
            output = _redistribute(output, kv_mesh, inverse=True)
        return ulysses_head_to_seq(output, 2, 1, ulysses_mesh)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, incoming: torch.Tensor) -> tuple:
        """Reduce contributions in FP32 and cast once at the input boundary.

        Args:
            ctx: Context saved by this invocation's forward pass.
            incoming: Output gradient in the original local BNSD order.

        Returns:
            Local Q/K/V gradients and None for non-tensor configuration inputs.
        """
        incoming = ulysses_seq_to_head(incoming.float(), 2, 1, ctx.ulysses_mesh)
        if ctx.config.causal_load_balance:
            incoming = _redistribute(incoming, ctx.kv_mesh)
        with torch.autocast("npu", enabled=False):
            gradients = _backward(*ctx.saved_tensors, incoming, ctx.kv_mesh, ctx.scale, ctx.config)
        if ctx.config.causal_load_balance:
            gradients = [_redistribute(tensor, ctx.kv_mesh, inverse=True) for tensor in gradients]
        gradients = [ulysses_head_to_seq(tensor, 2, 1, ctx.ulysses_mesh) for tensor in gradients]
        if ctx.replicas > 1:
            for index in (1, 2):
                tensor = gradients[index]
                gradients[index] = tensor.view(1, ctx.kv_heads, ctx.replicas, *tensor.shape[2:]).sum(2)
        return *(tensor.to(ctx.dtype) for tensor in gradients), None, None, None, None


class StreamKVGQAAttention:
    """B1 dense-causal Ulysses x stream-KV interface for GQA/GatedGQAAttention.

    Each rank initially owns a contiguous token interval in (KV-owner, Ulysses)
    mesh order. Inputs have global RoPE applied already. The caller owns parameter
    gradient reduction and loss normalization; this object reduces only KV input
    contributions. Local dX must not receive an extra CP SUM.

    This initial interface supports BF16 Ascend, D256, equal shards, TP=1 and
    first derivatives. TND, arbitrary masks, caches and dropout are rejected.
    No global monkey patch or model recipe is installed.
    """

    def __init__(self, ulysses_mesh: DeviceMesh, kv_mesh: DeviceMesh, config: StreamKVConfig) -> None:
        """Bind existing orthogonal submeshes of a (KV-owner, Ulysses-lane) mesh.

        Args:
            ulysses_mesh: One-dimensional head/sequence exchange subgroup.
            kv_mesh: One-dimensional sequence-owner subgroup at a fixed Ulysses lane.
            config: Immutable communication, compute and causal-layout settings.

        Note:
            The caller owns mesh construction and must provide equal contiguous
            token shards in parent-mesh order, with identical settings on all ranks.
        """
        if ulysses_mesh.ndim != 1 or kv_mesh.ndim != 1:
            raise ValueError("Ulysses and KV meshes must be one-dimensional")
        if not isinstance(config, StreamKVConfig):
            raise ValueError("config must be a StreamKVConfig")
        self.ulysses_mesh, self.kv_mesh, self.config = ulysses_mesh, kv_mesh, config

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  scale: float | None = None) -> torch.Tensor:
        """Return local BNSD attention with compact original KV gradients.

        Args:
            query: BF16 Ascend BNSD query for one contiguous local token shard.
            key: Compact key with the same batch, sequence and head dimension.
            value: Compact value with the same shape as key.
            scale: Positive finite score scale; defaults to inverse square-root D.

        Returns:
            Attention output with the input query's shape, dtype and token order.

        Raises:
            ValueError: Inputs or configuration exceed the supported contract.
        """
        _validate_tensors(query, key, value)
        degree, heads, kv_heads = self.ulysses_mesh.size(), query.shape[1], key.shape[1]
        if heads % kv_heads or heads % degree or (kv_heads % degree and degree % kv_heads):
            raise ValueError("Require divisible GQA heads and uniform Ulysses KV splitting or replication")
        if self.config.causal_load_balance and query.shape[2] * degree % 2:
            raise ValueError("Causal load balance requires an even post-Ulysses local sequence")
        if query.shape[2] * degree * self.kv_mesh.size() >= 2**31 - 1:
            raise ValueError("Global sequence exceeds the signed-int causal window")
        scale = query.shape[-1]**-0.5 if scale is None else scale
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be a finite positive number")
        return _StreamKV.apply(query, key, value, self.ulysses_mesh, self.kv_mesh, scale, self.config)

    def __call__(self, module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 attention_mask: torch.Tensor | None = None, *, dropout: float = 0.0,
                 scaling: float | None = None, sliding_window: int | None = None,
                 actual_seq_len: Any = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        """Implement the GQA component's attention interface and return BSND output.

        Args:
            module: Calling attention module, used to check causal mode.
            query: Local BNSD query with global RoPE already applied.
            key: Compact local BNSD key with global RoPE already applied.
            value: Compact local BNSD value.
            attention_mask: Must be None; causal visibility is derived from shard positions.
            dropout: Must be zero.
            scaling: Optional attention score scale.
            sliding_window: Must be None; only full causal attention is supported.
            actual_seq_len: Must be None; packed or variable-length sequences are unsupported.
            **kwargs: Extra component arguments must be None.

        Returns:
            Local BSND attention output and None for attention weights.
        """
        if attention_mask is not None or sliding_window is not None or actual_seq_len is not None:
            raise ValueError("Stream KV supports one unpadded dense causal sequence")
        if dropout != 0 or not getattr(module, "is_causal", True):
            raise ValueError("Stream KV requires causal attention and zero dropout")
        if any(value is not None for value in kwargs.values()):
            raise ValueError(f"Unsupported stream KV arguments: {tuple(kwargs)}")
        return self.attention(query, key, value, scaling).transpose(1, 2), None
