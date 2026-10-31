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
"""Dense MLA Ulysses with expanded QKV or shared latent-KV communication."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache, wraps
import math
from typing import Any, Literal

# Native Torch is required, as in the existing CP adapters; the legacy lint rule predates that migration.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.mla_backward_overlap import (
    MLAInputExchange, _ExpandedInputWait, _InputLaunch, _LatentInputWait, _LatentKVLinear,
)
from hyper_parallel.distributed.context_parallel.collectives import (
    _build_hybrid_cp_submeshes, async_cp_allgather_launch, async_ulysses_seq_to_head_launch,
    ulysses_head_to_seq, ulysses_seq_to_head,
)
from hyper_parallel.components.functional.npu_fusion_attention import (
    PACKED_SEQUENCE_ARGUMENTS, npu_fusion_attention_forward, resolve_packed_sequence_lengths,
)
from hyper_parallel.core.utils.communication import (
    differentiable_all_gather_concat, differentiable_all_to_all_single,
    differentiable_all_to_all_single_async, wait_async_tensor,
)


MLACPStrategy = Literal["expanded_ulysses", "latent_kv_head"]
MLA_CP_STRATEGIES = ("expanded_ulysses", "latent_kv_head")


@dataclass(frozen=True)
class MLADimensions:
    """Global MLA dimensions; the latent ranks are not partitioned across heads."""

    heads: int
    q_rank: int
    kv_rank: int
    nope_dim: int
    rope_dim: int
    value_dim: int

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        if self.rope_dim % 2:
            raise ValueError("rope_dim must be even")

    @property
    def qk_dim(self) -> int:
        """Return the score dimension, including the decoupled RoPE band."""
        return self.nope_dim + self.rope_dim


@dataclass(frozen=True)
class MLAState:
    """Normalized local-token latent tensors in BSR layout, before up projection."""

    query_latent: torch.Tensor
    kv_latent: torch.Tensor
    key_rope: torch.Tensor


@dataclass(frozen=True)
class MLACPPlan:
    """Validated explicit execution strategy; no uncalibrated automatic selection."""

    dimensions: MLADimensions
    cp_degree: int
    strategy: MLACPStrategy = "expanded_ulysses"
    tp_degree: int = 1
    overlap: bool = False
    ulysses_degree: int | None = None
    kv_exchange: Literal["all_gather", "all_to_all_v"] = "all_gather"

    def __post_init__(self) -> None:
        if not isinstance(self.overlap, bool):
            raise ValueError("overlap must be bool")
        if self.ulysses_degree is None:
            object.__setattr__(self, "ulysses_degree", self.cp_degree)
        for name in ("cp_degree", "tp_degree", "ulysses_degree"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.strategy not in MLA_CP_STRATEGIES:
            raise ValueError(f"Unknown MLA CP strategy: {self.strategy}")
        if self.cp_degree % self.ulysses_degree:
            raise ValueError("ulysses_degree must divide cp_degree")
        if self.dimensions.heads % (self.tp_degree * self.ulysses_degree):
            raise ValueError("MLA heads must be divisible by TP degree times Ulysses degree")
        if self.kv_exchange not in ("all_gather", "all_to_all_v"):
            raise ValueError("kv_exchange must be all_gather or all_to_all_v")
        if self.kv_exchange == "all_to_all_v" and self.all_gather_degree == 1:
            raise ValueError("selective KV exchange requires query-partitioned MLA CP")
        if self.tp_degree > 1 and self.all_gather_degree > 1:
            raise ValueError("MLA KV AllGather and hybrid CP currently require TP1")

    @property
    def local_heads(self) -> int:
        """Heads in the persistent TP parameter shard, before CP redistribution."""
        return self.dimensions.heads // self.tp_degree

    @property
    def compute_heads(self) -> int:
        """Heads used by this rank's query-partition attention kernel."""
        return self.local_heads // self.ulysses_degree

    @property
    def all_gather_degree(self) -> int:
        """Number of query owners for each Ulysses head partition."""
        return self.cp_degree // self.ulysses_degree

    def validate_scale(self, scale: float) -> None:
        """Reject invalid scales without replacing an explicit YaRN scale.

        Args:
            scale: The model's attention score scale, including any YaRN factor.

        Raises:
            ValueError: If the scale is nonfinite or nonpositive.
        """
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("MLA attention scaling must be finite and positive")


def mla_all_gather(tensor: torch.Tensor, sequence_dim: int, cp_mesh: Any) -> torch.Tensor:
    """Gather source tokens and sum all consumer gradients back to their owner.

    Args:
        tensor: Equal-sized local sequence shard.
        sequence_dim: Dimension along which source shards are concatenated.
        cp_mesh: CP mesh, or None for communication-free execution.

    Returns:
        Full-sequence tensor in CP group-rank order.
    """
    if cp_mesh is None or cp_mesh.size() == 1:
        return tensor
    return differentiable_all_gather_concat(tensor, cp_mesh.get_group(), cp_mesh.size(), sequence_dim)


def _apply_mla_rope(
    tensor: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Rotate one local BSND tensor so Q and shared K can be scheduled independently."""
    if position_embeddings is None:
        return tensor
    cos, sin = (part.to(tensor.dtype) for part in position_embeddings)
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    # Load the optional NPU dependency only when executing RoPE.
    from hyper_parallel.components.functional.rotary_embedding import (  # pylint: disable=C0415
        apply_rotary_pos_emb_single,
    )

    if interleaved:
        cos, sin = (part[..., :tensor.shape[-1] // 2].repeat_interleave(2, dim=-1)
                    for part in (cos, sin))
    cos, sin = (part.expand(tensor.shape[0], -1, -1).transpose(0, 1) for part in (cos, sin))
    rotated = apply_rotary_pos_emb_single(
        tensor.transpose(0, 1), cos, sin,
        rotary_interleaved=interleaved, use_fused_rotary_pos_emb=True,
    )
    if interleaved:
        # V3 consumes real/imaginary half-bands; the kernel returns adjacent pairs.
        rotated = rotated.permute(1, 2, 0, 3)
        return rotated.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2).transpose(1, 2)
    return rotated.transpose(0, 1)


def _validate_query_partition(causal, attention_mask):
    """Reject query-partition modes whose FA metadata contract is not implemented."""
    if not causal or attention_mask is not None:
        raise ValueError("MLA KV AllGather and hybrid CP require causal attention without an explicit mask")


@dataclass(frozen=True)
class _CausalPartition:
    """One continuous query partition and its document-local causal KV prefixes."""

    key_begin: int
    key_end: int
    query_lengths: tuple[int, ...]
    key_lengths: tuple[int, ...]
    documents: tuple[int, ...]

    def select(self, tensor: torch.Tensor, sequence_dim: int) -> torch.Tensor:
        """Keep a view of the needed prefix before either projection or attention."""
        return tensor.narrow(sequence_dim, self.key_begin, self.key_end - self.key_begin)


def _causal_partition(lengths, query_length, query_rank, batch) -> _CausalPartition:
    """Coalesce adjacent document prefixes without copying their KV tensors."""
    query_begin = query_rank * query_length
    query_end = query_begin + query_length
    query_ends, key_ends = [], []
    query_count = key_count = document_begin = 0
    key_begin = None
    for document_end in lengths:
        begin, end = max(document_begin, query_begin), min(document_end, query_end)
        if begin < end:
            if key_begin is None:
                key_begin = document_begin
            query_count += end - begin
            key_count += end - document_begin
            query_ends.append(query_count)
            key_ends.append(key_count)
        document_begin = document_end
    if query_count != query_length:
        raise ValueError("MLA document boundaries do not cover the query partition")
    # Continuous Q ownership makes all selected document prefixes adjacent.
    return _CausalPartition(
        key_begin, query_end,
        tuple(end + index * query_count for index in range(batch) for end in query_ends),
        tuple(end + index * key_count for index in range(batch) for end in key_ends),
        tuple(lengths),
    )


@lru_cache(maxsize=128)
def _kv_exchange_ranges(documents, source_length, query_length, head_copies, degree, rank):
    """Cache only host intervals shared by layers; never retain activation tensors."""
    destinations = tuple(
        _causal_partition(documents, query_length, peer // head_copies, 1)
        for peer in range(degree)
    )
    start = rank * source_length
    sends, receives = [], []
    local = destinations[rank]
    for peer, destination in enumerate(destinations):
        begin, end = max(start, destination.key_begin), min(start + source_length, destination.key_end)
        sends.append((begin - start, end - begin) if begin < end else (0, 0))
        begin, end = max(peer * source_length, local.key_begin), min((peer + 1) * source_length, local.key_end)
        receives.append(max(0, end - begin))
    local_only = all(peer * source_length <= item.key_begin and item.key_end <= (peer + 1) * source_length
                     for peer, item in enumerate(destinations))
    return tuple(sends), tuple(receives), local_only


@dataclass
class _SelectiveKVExchange:
    """A shared differentiable A2Av and its batch-major receive layout."""

    output: torch.Tensor
    receive_tokens: tuple[int, ...]
    batch: int

    def wait(self) -> torch.Tensor:
        """Finish communication before rebuilding batch-major token order."""
        if not self.receive_tokens:
            return self.output
        output = wait_async_tensor(self.output)
        parts, start = [], 0
        for count in self.receive_tokens:
            rows = self.batch * count
            if count:
                parts.append(output.narrow(0, start, rows).reshape(self.batch, count, *output.shape[1:]))
            start += rows
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def _selective_kv_launch(payload, partition, mesh, head_copies, overlap):
    """Send only each query owner's causal document interval, with SUM gradients through repeated slices."""
    batch, source_length = payload.shape[:2]
    rank, degree = mesh.get_local_rank(), mesh.size()
    sends, receives, local_only = _kv_exchange_ranges(
        partition.documents, source_length, partition.query_lengths[-1] // batch, head_copies, degree, rank,
    )
    if local_only:
        begin, count = sends[rank]
        return _SelectiveKVExchange(payload.narrow(1, begin, count), (), batch)
    send = torch.cat([payload.narrow(1, begin, count).flatten(0, 1) for begin, count in sends], dim=0)
    input_splits = [batch * count for _, count in sends]
    output_splits = [batch * count for count in receives]
    exchange = differentiable_all_to_all_single_async if overlap else differentiable_all_to_all_single
    output = exchange(send, input_splits, output_splits, mesh.get_group())
    return _SelectiveKVExchange(output, receives, batch)


def _positions(
    embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    state: MLAState,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Validate local-token RoPE frequencies and broadcast only the batch axis."""
    if embeddings is None:
        return None
    if not isinstance(embeddings, (tuple, list)) or len(embeddings) != 2:
        raise ValueError("position_embeddings must be a local (cos, sin) pair")
    batch, sequence = state.query_latent.shape[:2]
    parts = []
    for tensor in embeddings:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.shape not in ((1, sequence, rope_dim), (batch, sequence, rope_dim)):
            raise ValueError("RoPE cos/sin must describe local tokens with shape [B, S_local, rope_dim]")
        if tensor.requires_grad:
            raise ValueError("MLA CP currently requires non-trainable RoPE frequencies")
        if tensor.device != state.query_latent.device:
            raise ValueError("RoPE and MLA activations must be on the same device")
        parts.append(tensor.expand(batch, -1, -1))
    return tuple(parts)


def _validate_mask(
    mask: torch.Tensor | None,
    lengths: Sequence[int] | None,
    batch: int,
    sequence: int,
    device: torch.device,
) -> None:
    """Require global boolean masks and reject unsupported packed-mask combinations."""
    if mask is None:
        return
    if lengths is not None:
        raise ValueError("packed MLA CP currently accepts document causal/noncausal masks only")
    if mask.dtype != torch.bool:
        raise ValueError("MLA CP masks must be boolean (True=allowed); additive bias is not supported")
    valid = (
        mask.ndim == 2 and mask.shape == (sequence, sequence)
        or mask.ndim == 4 and mask.shape[0] in (1, batch)
        and mask.shape[1] == 1 and mask.shape[-2:] == (sequence, sequence)
    )
    if not valid or mask.device != device:
        raise ValueError("attention_mask must cover global Q/K: [S, S] or [B|1, 1, S, S], on the input device")


def _query(
    module: nn.Module,
    latent: torch.Tensor,
    positions: tuple[torch.Tensor, torch.Tensor] | None,
    heads: int,
) -> torch.Tensor:
    """Expand and rotate the query through the managed TP-local projection module."""
    projected = module.q_b_proj(latent)
    projected = projected.view(*latent.shape[:2], heads, module.qk_head_dim)
    if positions is None:
        return projected.transpose(1, 2)
    content, rotary = projected.split((module.qk_nope_head_dim, module.qk_rope_head_dim), dim=-1)
    rotary = _apply_mla_rope(rotary, positions, interleaved=module.rotary_interleaved)
    return torch.cat((content, rotary), dim=-1).transpose(1, 2)


def _key_value(
    module: nn.Module,
    latent: torch.Tensor,
    key_rope: torch.Tensor,
    heads: int,
    head_range: tuple[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the requested KV heads through the managed projection module."""
    projected = (module.kv_b_proj(latent) if head_range is None
                 else module.kv_b_proj(latent, mla_head_range=head_range))
    return _assemble_key_value(module, projected, key_rope, heads)


def _assemble_key_value(module, projected, key_rope, heads):
    """Build per-head keys and values from shared projection results."""
    projected = projected.view(*projected.shape[:2], heads, module.qk_nope_head_dim + module.v_head_dim)
    content, value = projected.split((module.qk_nope_head_dim, module.v_head_dim), dim=-1)
    key = torch.cat((content, key_rope.unsqueeze(2).expand(-1, -1, heads, -1)), dim=-1)
    return key.transpose(1, 2), value.transpose(1, 2)


def _validate_forward_kwargs(kwargs, position_embeddings):
    """Validate forwarded model options before routing any latent."""
    position_ids = kwargs.pop("position_ids", None)
    if position_ids is not None and position_embeddings is None:
        raise ValueError("position_ids require precomputed local position_embeddings in MLA CP")
    kwargs.pop("cache_position", None)
    if kwargs.pop("use_cache", False) or kwargs.pop("output_attentions", False):
        raise ValueError("MLA CP does not return KV caches or attention weights")
    if kwargs:
        raise ValueError(f"Unsupported MLA CP forward arguments: {sorted(kwargs)}")


def _validate_state(module: nn.Module, plan: MLACPPlan, state: MLAState) -> None:
    """Check latent shape and NPU kernel requirements before collectives."""
    dimensions = plan.dimensions
    query_latent, kv_latent, key_rope = state.query_latent, state.kv_latent, state.key_rope
    if query_latent.ndim != 3:
        raise ValueError("MLA CP latents must have BSR layout")
    batch, sequence = query_latent.shape[:2]
    if batch < 1 or sequence < 1:
        raise ValueError("MLA CP requires nonempty batches and sequence shards")
    for tensor, width in ((query_latent, dimensions.q_rank), (kv_latent, dimensions.kv_rank),
                          (key_rope, dimensions.rope_dim)):
        if (tensor.shape != (batch, sequence, width) or tensor.device != query_latent.device
                or tensor.dtype != query_latent.dtype):
            raise ValueError("MLA CP latent shapes/devices/dtypes do not match the plan")
    if module.training and module.attention_dropout != 0:
        raise ValueError("MLA CP requires attention dropout=0")
    plan.validate_scale(module.scaling)
    if query_latent.device.type != "npu" or query_latent.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("NPU MLA CP requires NPU BF16/FP16 activations")
    if dimensions.value_dim > dimensions.qk_dim:
        raise ValueError("NPU MLA CP requires value_dim <= qk_dim")


class MLAContextParallel:
    """Run CP over normalized latents; parameter-gradient reduction belongs to the trainer."""

    def __init__(self, plan: MLACPPlan, cp_mesh: Any) -> None:
        """Bind a validated execution plan to the framework-owned CP mesh."""
        if (1 if cp_mesh is None else cp_mesh.size()) != plan.cp_degree:
            raise ValueError("MLACPPlan degree must match the CP mesh")
        self.plan = plan
        self.cp_mesh = cp_mesh
        rank = 0 if cp_mesh is None else cp_mesh.get_local_rank()
        self.query_rank, self.head_rank = divmod(rank, plan.ulysses_degree)
        self.ulysses_mesh = self.all_gather_mesh = cp_mesh
        if 1 < plan.ulysses_degree < plan.cp_degree:
            self.ulysses_mesh, self.all_gather_mesh = _build_hybrid_cp_submeshes(cp_mesh, plan.ulysses_degree)

    def __call__(
        self,
        module: nn.Module,
        query_latent: torch.Tensor,
        kv_latent: torch.Tensor,
        key_rope: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        is_causal: bool | None = None,
        restore_output: bool = True,
        backward_overlap: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run attention and optionally restore its output to token-local BSND.

        Args:
            backward_overlap: Schedule pure-Ulysses input adjoints when forward overlap is enabled.
            restore_output: False exposes the head-sharded output so model-owned
                independent work can run before its return exchange completes.
        """
        packed_kwargs = {name: kwargs.pop(name) for name in PACKED_SEQUENCE_ARGUMENTS if name in kwargs}
        packed_kwargs["actual_seq_len"] = actual_seq_len
        _validate_forward_kwargs(kwargs, position_embeddings)
        state = MLAState(query_latent, kv_latent, key_rope)
        dimensions, degree = self.plan.dimensions, self.plan.cp_degree
        batch, sequence = query_latent.shape[:2]
        lengths, key_lengths = resolve_packed_sequence_lengths(
            packed_kwargs, batch * sequence * degree, batch * sequence * degree,
        )
        if lengths is not None and batch != 1:
            raise ValueError("packed MLA CP currently requires batch_size=1")
        if lengths != key_lengths:
            raise ValueError("MLA CP requires matching global query and key document boundaries")
        causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
        if not isinstance(causal, bool):
            raise ValueError("is_causal must be bool")
        _validate_mask(attention_mask, lengths, batch, sequence * degree, query_latent.device)
        partition = None
        if self.plan.all_gather_degree > 1:
            _validate_query_partition(causal, attention_mask)
            partition = _causal_partition(
                lengths or (sequence * degree,), sequence * self.plan.ulysses_degree, self.query_rank, batch,
            )
            lengths, key_lengths = partition.query_lengths, partition.key_lengths
        positions = _positions(position_embeddings, state, dimensions.rope_dim)
        _validate_state(module, self.plan, state)
        rotated_key = _apply_mla_rope(
            state.key_rope.unsqueeze(2), positions, interleaved=module.rotary_interleaved,
        ).squeeze(2)
        if backward_overlap and self.plan.overlap and degree > 1 and self.plan.all_gather_degree == 1:
            prepare = (self._expanded_backward_overlap if self.plan.strategy == "expanded_ulysses"
                       else self._latent_backward_overlap)
            query, key, value = prepare(module, state, positions, rotated_key)
        elif self.plan.overlap and degree > 1:
            prepare = self._expanded_overlap if self.plan.strategy == "expanded_ulysses" else self._latent_overlap
            query, key, value = prepare(module, state, positions, rotated_key, partition)
        else:
            query = _query(module, state.query_latent, positions, self.plan.local_heads)
            if self.plan.strategy == "expanded_ulysses" or degree == 1:
                query, key, value = self._expanded(module, state, query, rotated_key, partition)
            else:
                query, key, value = self._latent(module, state, query, rotated_key, partition)
        output = npu_fusion_attention_forward(
            module, query, key, value, attention_mask, scaling=module.scaling,
            is_causal=causal, actual_seq_qlen=lengths, actual_seq_kvlen=key_lengths,
            pre_tokens=2147483647, next_tokens=0 if causal else 2147483647,
        )[0]
        if restore_output:
            output = self.restore_output(output)
        return output, None

    def restore_output(self, output: torch.Tensor) -> torch.Tensor:
        """Restore token ownership after attention, before the model output projection.

        Args:
            output: Attention result in BSND head-sharded layout.

        Returns:
            Original local-token, all-local-head BSND tensor.
        """
        batch, sequence = output.shape[:2]
        sequence //= self.plan.ulysses_degree
        if self.plan.ulysses_degree > 1:
            output = ulysses_head_to_seq(output, 1, 2, self.ulysses_mesh)
        expected = (batch, sequence, self.plan.local_heads, self.plan.dimensions.value_dim)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"MLA CP output has shape {tuple(output.shape)}, expected {expected}")
        return output

    def _expanded(self, module, state, query, key_rope, partition=None):
        """Pack unequal-width QKV into one sequence-to-head exchange."""
        key, value = _key_value(module, state.kv_latent, key_rope, self.plan.local_heads, None)
        if self.plan.ulysses_degree > 1:
            widths = (query.shape[-1], key.shape[-1], value.shape[-1])
            payload = torch.cat(tuple(tensor.transpose(1, 2) for tensor in (query, key, value)), dim=-1)
            payload = ulysses_seq_to_head(payload, 1, 2, self.ulysses_mesh)
            query, key, value = (tensor.transpose(1, 2) for tensor in payload.split(widths, dim=-1))
        if self.plan.all_gather_degree > 1:
            widths = (key.shape[-1], value.shape[-1])
            payload = torch.cat((key.transpose(1, 2), value.transpose(1, 2)), dim=-1)
            if self.plan.kv_exchange == "all_to_all_v":
                payload = _selective_kv_launch(payload, partition, self.all_gather_mesh, 1, False).wait()
            else:
                payload = partition.select(mla_all_gather(payload, 1, self.all_gather_mesh), 1)
            key, value = (tensor.transpose(1, 2) for tensor in payload.split(widths, dim=-1))
        return query, key, value

    def _latent(self, module, state, query, key_rope, partition=None):
        """Exchange Q heads, gather shared KV latents, then expand only owned heads."""
        if self.plan.ulysses_degree > 1:
            query = ulysses_seq_to_head(query, 2, 1, self.ulysses_mesh)
        payload = torch.cat((state.kv_latent, key_rope), dim=-1)
        if self.plan.kv_exchange == "all_to_all_v":
            payload = _selective_kv_launch(payload, partition, self.cp_mesh, self.plan.ulysses_degree, False).wait()
            partition = None
        else:
            payload = mla_all_gather(payload, 1, self.cp_mesh)
        return query, *self._expand_latent(module, payload, partition)

    def _expand_latent(self, module, payload, partition=None):
        """Keep the same managed head-selective projection for both communication schedules."""
        if partition is not None:
            payload = partition.select(payload, 1)
        dimensions = self.plan.dimensions
        heads = self.plan.compute_heads
        start = self.head_rank * heads
        kv_latent, key_rope = payload.split((dimensions.kv_rank, dimensions.rope_dim), dim=-1)
        # A contiguous latent preserves the native Linear bias-fusion path after payload splitting.
        return _key_value(module, kv_latent.contiguous(), key_rope, heads, (start, start + heads))

    def _expanded_overlap(self, module, state, positions, key_rope, partition=None):
        """Overlap KV transfer with Q preparation, respecting the two-stage hybrid dependency."""
        key, value = _key_value(module, state.kv_latent, key_rope, self.plan.local_heads, None)
        payload = torch.cat((key.transpose(1, 2), value.transpose(1, 2)), dim=-1)
        use_ulysses = self.plan.ulysses_degree > 1
        if use_ulysses:
            pending_kv = async_ulysses_seq_to_head_launch(payload, 1, 2, self.ulysses_mesh)
        else:
            pending_kv = self._expanded_kv_launch(payload, partition)
        query = _query(module, state.query_latent, positions, self.plan.local_heads)
        pending_query = (async_ulysses_seq_to_head_launch(query.transpose(1, 2), 1, 2, self.ulysses_mesh)
                         if use_ulysses else None)
        payload = pending_kv.wait()
        if use_ulysses and self.plan.all_gather_degree > 1:
            payload = self._expanded_kv_launch(payload, partition).wait()
        if pending_query is not None:
            query = pending_query.wait().transpose(1, 2)
        if partition is not None and self.plan.kv_exchange == "all_gather":
            payload = partition.select(payload, 1)
        key, value = payload.split((module.qk_head_dim, module.v_head_dim), dim=-1)
        return query, key.transpose(1, 2), value.transpose(1, 2)

    def _expanded_kv_launch(self, payload, partition):
        """Complete the sequence dimension after Ulysses has selected this head shard."""
        if self.plan.kv_exchange == "all_to_all_v":
            return _selective_kv_launch(payload, partition, self.all_gather_mesh, 1, True)
        return async_cp_allgather_launch(payload, 1, self.all_gather_mesh)

    def _latent_overlap(self, module, state, positions, key_rope, partition=None):
        """Cover latent transfer with Q preparation, then Q A2A with KV expansion."""
        payload = torch.cat((state.kv_latent, key_rope), dim=-1)
        if self.plan.kv_exchange == "all_to_all_v":
            pending_kv = _selective_kv_launch(payload, partition, self.cp_mesh, self.plan.ulysses_degree, True)
            partition = None
        else:
            pending_kv = async_cp_allgather_launch(payload, 1, self.cp_mesh)
        query = _query(module, state.query_latent, positions, self.plan.local_heads)
        payload = pending_kv.wait()
        pending_query = (async_ulysses_seq_to_head_launch(query.transpose(1, 2), 1, 2, self.ulysses_mesh)
                         if self.plan.ulysses_degree > 1 else None)
        key, value = self._expand_latent(module, payload, partition)
        if pending_query is not None:
            query = pending_query.wait().transpose(1, 2)
        return query, key, value

    def _expanded_backward_overlap(self, module, state, positions, key_rope):
        """Keep Q projection backward inside the inverse KV exchange window."""
        key, value = _key_value(module, state.kv_latent, key_rope, self.plan.local_heads, None)
        payload = torch.cat((key.transpose(1, 2), value.transpose(1, 2)), dim=-1)
        exchange = MLAInputExchange(self.ulysses_mesh)
        query_latent = _InputLaunch.apply(payload, state.query_latent, exchange)
        query = _query(module, query_latent, positions, self.plan.local_heads).transpose(1, 2)
        payload, query = _ExpandedInputWait.apply(query, exchange)
        key, value = payload.split((module.qk_head_dim, module.v_head_dim), dim=-1)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def _latent_backward_overlap(self, module, state, positions, key_rope):
        """Cover inverse Q with KV backward, then latent reduction with KV wgrad."""
        payload = torch.cat((state.kv_latent, key_rope), dim=-1)
        pending = async_cp_allgather_launch(payload, 1, self.cp_mesh)
        query = _query(module, state.query_latent, positions, self.plan.local_heads).transpose(1, 2)
        # The managed projection below owns the gather's single gradient edge.
        with torch.no_grad():
            gathered = pending.wait()
        exchange = MLAInputExchange(self.ulysses_mesh)
        bridge = _InputLaunch.apply(query, payload, exchange)
        heads = self.plan.compute_heads
        start = self.head_rank * heads
        projected, shared_key = module.kv_b_proj(
            bridge, mla_head_range=(start, start + heads),
            mla_gathered=gathered, mla_cp_mesh=self.cp_mesh,
        )
        key, value = _assemble_key_value(module, projected, shared_key, heads)
        query, key, value = _LatentInputWait.apply(key, value, exchange)
        return query.transpose(1, 2), key, value


def _head_projection_request(projection: nn.Linear, head_dim: int) -> _ForwardRewriteRequest:
    """Select head rows inside the original Linear call to preserve parameter hooks."""
    original = projection.forward

    # Keep nn.Linear.forward(input=...) compatible with keyword callers.
    @wraps(original)
    def head_projection(  # pylint: disable=redefined-builtin
        input: torch.Tensor, *, mla_head_range: tuple[int, int] | None = None,
        mla_gathered: torch.Tensor | None = None, mla_cp_mesh: Any = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Project selected contiguous heads inside the original module call.

        Args:
            input: Latent activations, or the local packed payload when mla_gathered is provided.
            mla_head_range: Optional half-open range of TP-local heads.
            mla_gathered: Completed latent payload for scheduled gather backward.
            mla_cp_mesh: CP mesh owning the gathered payload.

        Returns:
            Projected activations, or their pair with the shared key for scheduled gather backward.
        """
        if mla_head_range is None:
            return original(input)
        begin, end = mla_head_range
        if not 0 <= begin < end <= projection.weight.shape[0] // head_dim:
            raise ValueError("MLA projection head range is outside the original parameter")
        rows = slice(begin * head_dim, end * head_dim)
        bias = None if projection.bias is None else projection.bias[rows]
        # Read parameters inside the managed module's forward, after pre-hooks.
        # Slicing preserves the Parameter identity and scatters gradients back.
        if mla_gathered is not None:
            if mla_cp_mesh is None:
                raise ValueError("Scheduled MLA latent projection requires its CP mesh")
            return _LatentKVLinear.apply(input, projection.weight[rows], bias, mla_gathered, mla_cp_mesh)
        return F.linear(input, projection.weight[rows], bias)  # pylint: disable=not-callable

    return _ForwardRewriteRequest(projection, head_projection)


def _validate_up_projections(target_module: nn.Module, plan: MLACPPlan) -> None:
    """Check TP-local Linear weights without replacing managed parameters."""
    dimensions = plan.dimensions
    projections = (
        (target_module.q_b_proj, dimensions.q_rank, dimensions.qk_dim),
        (target_module.kv_b_proj, dimensions.kv_rank, dimensions.nope_dim + dimensions.value_dim),
    )
    for projection, in_features, head_dim in projections:
        # Quantized/LoRA subclasses do not obey contiguous Linear head slicing.
        if type(projection) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("MLA CP requires plain nn.Linear up projections")
        weight = projection.weight.to_local() if isinstance(projection.weight, DTensor) else projection.weight
        if weight.shape != (plan.local_heads * head_dim, in_features):
            raise ValueError("MLA up-projection shape must match TP-local heads; apply TP parameter sharding first")
        if getattr(projection.forward, "__func__", None) is not nn.Linear.forward:
            raise ValueError("MLA CP must be installed before other up-projection forward adapters")


def prepare_mla_cp(
    target_module: nn.Module,
    cp_mesh: Any,
    tp_mesh: Any,
    *,
    strategy: MLACPStrategy,
    overlap: bool,
    ulysses_degree: int | None,
    kv_exchange: str,
) -> tuple[MLAContextParallel, list[_ForwardRewriteRequest]]:
    """Prepare a shared MLA runtime and managed head-projection rewrites.

    Args:
        target_module: Module exposing normalized MLA latents and plain projections.
        cp_mesh: Context-parallel mesh.
        tp_mesh: Optional tensor-parallel mesh.
        strategy: Expanded or latent KV representation.
        overlap: Whether to overlap input communication with projections.
        ulysses_degree: Head-parallel degree within CP.
        kv_exchange: Full gather or selective KV exchange.

    Returns:
        Runtime and child rewrites; the model adapter owns its forward contract.
    """
    for name, axis in (("TP", tp_mesh), ("CP", cp_mesh)):
        if axis is not None and (axis.size() < 1 or getattr(axis, "ndim", 1) != 1):
            raise ValueError(f"mla_cp_wrapper requires a one-dimensional {name} mesh")
    cp_degree = 1 if cp_mesh is None else cp_mesh.size()
    tp_degree = 1 if tp_mesh is None else tp_mesh.size()
    if getattr(target_module, "mla_cp_runtime", None) is not None:
        raise ValueError("MLA CP is already configured on this module")
    if target_module.attention_dropout != 0 or target_module.sliding_window is not None:
        raise ValueError("MLA CP requires dropout=0 and no implicit sliding_window; use a supported boolean mask")
    dimensions = MLADimensions(
        heads=target_module.config.num_attention_heads, q_rank=target_module.q_lora_rank,
        kv_rank=target_module.kv_lora_rank, nope_dim=target_module.qk_nope_head_dim,
        rope_dim=target_module.qk_rope_head_dim, value_dim=target_module.v_head_dim,
    )
    plan = MLACPPlan(
        dimensions=dimensions, cp_degree=cp_degree, strategy=strategy,
        tp_degree=tp_degree, overlap=overlap, ulysses_degree=ulysses_degree, kv_exchange=kv_exchange,
    )
    plan.validate_scale(target_module.scaling)
    _validate_up_projections(target_module, plan)
    runtime = MLAContextParallel(plan, cp_mesh)

    requests = []
    if strategy == "latent_kv_head":
        requests.append(_head_projection_request(target_module.kv_b_proj, dimensions.nope_dim + dimensions.value_dim))
    return runtime, requests
