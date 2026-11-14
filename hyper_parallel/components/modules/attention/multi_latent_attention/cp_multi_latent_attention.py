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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Literal

# Native Torch is required, as in the existing CP adapters; the legacy lint rule predates that migration.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.attention.multi_latent_attention.cp_mla_backward_overlap import (
    CPMLAKVLinear, CPMLAOutputLinear, MLAOutputProjectionState, MLAExpandedInputA2AState, MLAInputA2AState,
    _InputA2ALaunch, _LatentInputA2AWait,
    finish_expanded_inputs, launch_input_a2a,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention import (
    MultiLatentAttentionModule, apply_mla_rope as _apply_mla_rope,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention import (
    prepare_mla_attention_mask,
)
from hyper_parallel.distributed.context_parallel.cp_context import CPContext
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention_family import (
    validate_mla_layout,
)
from hyper_parallel.distributed.context_parallel.collectives import (
    async_cp_allgather_launch,
)
from hyper_parallel.components.functional.npu_fusion_attention import (
    PACKED_SEQUENCE_ARGUMENTS, npu_fusion_attention_forward, resolve_packed_sequence_lengths,
)
from hyper_parallel.core.utils.communication import (
    differentiable_all_to_all_single,
    differentiable_all_to_all_single_async, wait_async_tensor,
)


MLACPStrategy = Literal["expanded_ulysses", "latent_kv_head"]
# Preserve the existing helper import path while the mathematical implementation moves into its family.
apply_mla_rope = _apply_mla_rope

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
    # Share the overlap path's SUM ReduceScatter adjoint, but wait immediately.
    # PyTorch's generic autograd AllGather falls back to AllToAll + local sum on
    # HCCL, changing BF16 reduction order when only the overlap flag changes.
    return async_cp_allgather_launch(tensor, sequence_dim, cp_mesh).wait()


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
        """Keep a view of the needed prefix before either projection or attention.

        Args:
            tensor: Gathered KV features in global sequence order.
            sequence_dim: Axis containing the gathered tokens.

        Returns:
            A view of the KV interval needed by this query partition.
        """
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

    def __init__(
        self,
        plan: MLACPPlan,
        cp_mesh: Any,
        *,
        rotary_fn: Callable[..., torch.Tensor] | None = None,
        attention_interface: Callable[..., tuple[torch.Tensor, Any]] | None = None,
    ) -> None:
        """Bind communication to optional local rotary and attention implementations.

        Args:
            plan: Validated MLA communication and head plan.
            cp_mesh: Framework-owned CP mesh, or None for degree one.
            rotary_fn: Local RoPE callback receiving tensor, positions and interleaved.
                It preserves the tensor's shape, dtype and device. None uses generic RoPE.
            attention_interface: Local attention callback with the generic attention contract.
                It returns BSND output and optional weights. Extra keywords mla_head_offset
                and mla_local_heads describe its head slice within the TP-local heads.
                None uses the existing NPU attention implementation.

        Note:
            Callbacks can run again during recomputation; their side effects belong to the caller.
            Communication, projections and output restoration remain owned by this runtime.
        """
        if (1 if cp_mesh is None else cp_mesh.size()) != plan.cp_degree:
            raise ValueError("MLACPPlan degree must match the CP mesh")
        if rotary_fn is not None and not callable(rotary_fn):
            raise TypeError("rotary_fn must be callable")
        if attention_interface is not None and not callable(attention_interface):
            raise TypeError("attention_interface must be callable")
        self.plan = plan
        self.rotary_fn = rotary_fn
        self.attention_interface = attention_interface
        self.cp_mesh = cp_mesh
        rank = 0 if cp_mesh is None else cp_mesh.get_local_rank()
        self.query_rank, self.head_rank = divmod(rank, plan.ulysses_degree)
        self.ulysses_mesh = self.all_gather_mesh = cp_mesh
        self.context = self.ulysses_context = self.query_context = None
        if cp_mesh is not None:
            self.context = CPContext(cp_mesh)
            heads, queries = self.context.split(plan.ulysses_degree)
            self.ulysses_context, self.query_context = heads, queries
            self.ulysses_mesh, self.all_gather_mesh = heads.mesh, queries.mesh

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
        overlap_input_backward: bool = False,
        input_gate: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run attention and optionally restore its output to token-local BSND.

        Args:
            overlap_input_backward: Pair input autograd nodes only for pure Ulysses.
                AG/hybrid input gradients remain native; output backward is selected
                independently by the model adapter.
            input_gate: Model-owned gate to cover expanded pure-Ulysses input exchanges.
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
        attention_mask = prepare_mla_attention_mask(attention_mask, sequence * degree, sequence * degree)
        partition = None
        if self.plan.all_gather_degree > 1:
            _validate_query_partition(causal, attention_mask)
            partition = _causal_partition(
                lengths or (sequence * degree,), sequence * self.plan.ulysses_degree, self.query_rank, batch,
            )
            lengths, key_lengths = partition.query_lengths, partition.key_lengths
        positions = _positions(position_embeddings, state, dimensions.rope_dim)
        _validate_state(module, self.plan, state)
        rotary_fn = module.apply_position if self.rotary_fn is None else self.rotary_fn
        rotated_key = rotary_fn(
            state.key_rope.unsqueeze(2), positions, interleaved=module.rotary_interleaved,
        ).squeeze(2)
        query, key, value = self._prepare_attention_inputs(
            module, state, positions, rotated_key, partition, overlap_input_backward, input_gate, rotary_fn,
        )
        attention_interface = (
            npu_fusion_attention_forward if self.attention_interface is None else self.attention_interface
        )
        output = module.local_attention(
            query, key, value, attention_mask, attention_interface=attention_interface,
            dropout=0.0, sliding_window=None,
            is_causal=causal, actual_seq_qlen=lengths, actual_seq_kvlen=key_lengths,
            pre_tokens=2147483647, next_tokens=0 if causal else 2147483647,
            mla_head_offset=self.head_rank * self.plan.compute_heads,
            mla_local_heads=self.plan.local_heads,
        )[0]
        if restore_output:
            output = self.restore_output(output)
        return output, None

    def _prepare_attention_inputs(self, module, state, positions, key_rope, partition,
                                  overlap_input_backward, input_gate, rotary_fn):
        """Select input preparation; paired input backward requires pure Ulysses.

        Forward overlap also supports KV AllGather and hybrid CP. Those paths
        retain native input collective adjoints; their output dO overlap is
        selected independently by the model adapter.
        """
        use_forward_overlap = self.plan.overlap and self.plan.cp_degree > 1
        pure_ulysses = self.plan.all_gather_degree == 1
        expanded = self.plan.strategy == "expanded_ulysses"
        supports_input_gate = use_forward_overlap and pure_ulysses and expanded
        if input_gate is not None and not supports_input_gate:
            raise ValueError("Input gate overlap requires expanded pure Ulysses with overlap enabled")
        if overlap_input_backward and not pure_ulysses:
            raise ValueError("Paired MLA input backward currently requires pure Ulysses")
        if not use_forward_overlap:
            query = module.project_query(
                state.query_latent, positions, heads=self.plan.local_heads, rotary_fn=rotary_fn,
            )
            prepare = self._expanded if expanded or self.plan.cp_degree == 1 else self._latent
            return prepare(module, state, query, key_rope, partition)
        # Pure Ulysses has no outer KV AllGather. Expanded Q/K/V use the same
        # V/K/Q forward order with or without a gate; overlap_input_backward
        # selects paired input gradients versus the native collective adjoints.
        if pure_ulysses and expanded:
            return self._prepare_expanded_ulysses_inputs(
                module, state, positions, key_rope, input_gate, overlap_input_backward, rotary_fn,
            )
        if overlap_input_backward:
            # Expanded has returned above and AG/hybrid were rejected earlier:
            # this is latent pure Ulysses, pairing Q A2A with KV projection/RS.
            return self._prepare_latent_ulysses_inputs(module, state, positions, key_rope, rotary_fn)
        # AG/hybrid inputs and forward-only latent Ulysses use native adjoints.
        prepare = self._expanded_overlap if expanded else self._latent_overlap
        return prepare(module, state, positions, key_rope, partition, rotary_fn=rotary_fn)

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
            output = self.ulysses_context.head_to_sequence(output)
        expected = (batch, sequence, self.plan.local_heads, self.plan.dimensions.value_dim)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"MLA CP output has shape {tuple(output.shape)}, expected {expected}")
        return output


    def _expanded(self, module, state, query, key_rope, partition=None):
        """Pack unequal-width QKV into one sequence-to-head exchange."""
        key, value = module.project_key_value(state.kv_latent, key_rope, heads=self.plan.local_heads)
        if self.plan.ulysses_degree > 1:
            widths = (query.shape[-1], key.shape[-1], value.shape[-1])
            payload = torch.cat(tuple(tensor.transpose(1, 2) for tensor in (query, key, value)), dim=-1)
            payload = self.ulysses_context.sequence_to_head(payload)
            query, key, value = (tensor.transpose(1, 2) for tensor in payload.split(widths, dim=-1))
        if self.plan.all_gather_degree > 1:
            widths = (key.shape[-1], value.shape[-1])
            payload = torch.cat((key.transpose(1, 2), value.transpose(1, 2)), dim=-1)
            if self.plan.kv_exchange == "all_to_all_v":
                payload = _selective_kv_launch(payload, partition, self.all_gather_mesh, 1, False).wait()
            else:
                payload = partition.select(self.query_context.gather_sequence(payload)[0], 1)
            key, value = (tensor.transpose(1, 2) for tensor in payload.split(widths, dim=-1))
        return query, key, value

    def _latent(self, module, state, query, key_rope, partition=None):
        """Exchange Q heads, gather shared KV latents, then expand only owned heads."""
        if self.plan.ulysses_degree > 1:
            query = self.ulysses_context.sequence_to_head(query, seq_dim=2, head_dim=1)
        payload = torch.cat((state.kv_latent, key_rope), dim=-1)
        if self.plan.kv_exchange == "all_to_all_v":
            payload = _selective_kv_launch(payload, partition, self.cp_mesh, self.plan.ulysses_degree, False).wait()
            partition = None
        else:
            payload = self.context.gather_sequence(payload)[0]
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
        return module.project_key_value(
            kv_latent.contiguous(), key_rope, heads=heads, head_range=(start, start + heads),
        )

    def _expanded_overlap(self, module, state, positions, key_rope, partition=None, *, rotary_fn):
        """Overlap KV transfer with Q preparation, respecting the two-stage hybrid dependency."""
        key, value = module.project_key_value(state.kv_latent, key_rope, heads=self.plan.local_heads)
        payload = torch.cat((key.transpose(1, 2), value.transpose(1, 2)), dim=-1)
        use_ulysses = self.plan.ulysses_degree > 1
        if use_ulysses:
            kv_transfer_handle = self.ulysses_context.launch_sequence_to_head(payload)
        else:
            kv_transfer_handle = self._expanded_kv_launch(payload, partition)
        query = module.project_query(state.query_latent, positions, heads=self.plan.local_heads, rotary_fn=rotary_fn)
        query_a2a_handle = (self.ulysses_context.launch_sequence_to_head(query.transpose(1, 2))
                            if use_ulysses else None)
        payload = kv_transfer_handle.wait()
        if use_ulysses and self.plan.all_gather_degree > 1:
            payload = self._expanded_kv_launch(payload, partition).wait()
        if query_a2a_handle is not None:
            query = query_a2a_handle.wait().transpose(1, 2)
        if partition is not None and self.plan.kv_exchange == "all_gather":
            payload = partition.select(payload, 1)
        key, value = payload.split((module.qk_head_dim, module.v_head_dim), dim=-1)
        return query, key.transpose(1, 2), value.transpose(1, 2)

    def _expanded_kv_launch(self, payload, partition):
        """Complete the sequence dimension after Ulysses has selected this head shard."""
        if self.plan.kv_exchange == "all_to_all_v":
            return _selective_kv_launch(payload, partition, self.all_gather_mesh, 1, True)
        return self.query_context.launch_gather_sequence(payload).collective

    def _latent_overlap(self, module, state, positions, key_rope, partition=None, *, rotary_fn):
        """Cover latent transfer with Q preparation, then Q A2A with KV expansion."""
        payload = torch.cat((state.kv_latent, key_rope), dim=-1)
        if self.plan.kv_exchange == "all_to_all_v":
            kv_transfer_handle = _selective_kv_launch(payload, partition, self.cp_mesh, self.plan.ulysses_degree, True)
            partition = None
        else:
            kv_transfer_handle = self.context.launch_gather_sequence(payload).collective
        query = module.project_query(state.query_latent, positions, heads=self.plan.local_heads, rotary_fn=rotary_fn)
        payload = kv_transfer_handle.wait()
        query_a2a_handle = (self.ulysses_context.launch_sequence_to_head(query.transpose(1, 2))
                            if self.plan.ulysses_degree > 1 else None)
        key, value = self._expand_latent(module, payload, partition)
        if query_a2a_handle is not None:
            query = query_a2a_handle.wait().transpose(1, 2)
        return query, key, value

    def _prepare_expanded_ulysses_inputs(self, module, state, positions, key_rope, input_gate,
                                         overlap_input_backward, rotary_fn):
        """Prepare expanded Q/K/V in V/K/Q order and attach the selected adjoints."""
        launch_input = _InputA2ALaunch.apply if overlap_input_backward else launch_input_a2a
        input_a2a = MLAExpandedInputA2AState(
            query=MLAInputA2AState(self.ulysses_mesh),
            key=MLAInputA2AState(self.ulysses_mesh),
            value=MLAInputA2AState(self.ulysses_mesh),
        )
        # V needs no positional transform: it is ready as soon as KV up-projection
        # is split. Shared-K positional math was already applied by the caller.
        key_nope, value = module.project_key_value_parts(state.kv_latent, heads=self.plan.local_heads)
        # Keep K assembly inside V's communication window.
        key_nope = launch_input(value, key_nope, input_a2a.value)
        key = module.assemble_key(key_nope, key_rope)
        # K is complete; Q projection and its model-defined rotation can cover it.
        query_latent = launch_input(key, state.query_latent, input_a2a.key)
        query = module.project_query(
            query_latent, positions, heads=self.plan.local_heads, rotary_fn=rotary_fn,
        ).transpose(1, 2)
        query, key, value = self.finish_input_exchange(query, input_a2a, overlap_input_backward, input_gate)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def finish_input_exchange(self, query: torch.Tensor, transfers: Any, overlap_backward: bool,
                              input_gate: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Finish ordinary Q/K/V; gated MLA overrides this model-specific boundary."""
        if input_gate is not None:
            raise ValueError("Ordinary MLA cannot consume a gated input dependency")
        return finish_expanded_inputs(query, transfers, overlap_backward)

    def _prepare_latent_ulysses_inputs(self, module, state, positions, key_rope, rotary_fn):
        """Prepare latent Ulysses inputs and pair their Q/KV backward dependencies."""
        payload = torch.cat((state.kv_latent, key_rope), dim=-1)
        kv_all_gather_handle = self.context.launch_gather_sequence(payload).collective
        query = module.project_query(
            state.query_latent, positions, heads=self.plan.local_heads, rotary_fn=rotary_fn,
        ).transpose(1, 2)
        # wait() normally adds the default AllGather autograd node. Instead,
        # _LatentKVLinear launches SUM RS before KV wgrad and returns dPayload
        # via kv_input. Keep this numeric buffer off the default gradient path.
        with torch.no_grad():
            gathered = kv_all_gather_handle.wait()
        query_a2a = MLAInputA2AState(self.ulysses_mesh)
        # Same payload value, with Q's inverse A2A bracketing its KV backward.
        kv_input = _InputA2ALaunch.apply(query, payload, query_a2a)
        heads = self.plan.compute_heads
        start = self.head_rank * heads
        projected, shared_key = module.kv_b_proj(
            kv_input, mla_head_range=(start, start + heads),
            mla_gathered=gathered, mla_cp_mesh=self.cp_mesh,
        )
        key, value = module.assemble_key_value(projected, shared_key, heads=heads)
        query, key, value = _LatentInputA2AWait.apply(key, value, query_a2a)
        return query.transpose(1, 2), key, value



class CPMultiLatentAttentionModule(MultiLatentAttentionModule):
    """Complete MLA CP member; its class owns forward and managed projection stages."""

    def __init__(self, *, module: nn.Module | None = None, config: Any = None,
                 module_fqn: str = "", context: Any = None,
                 strategy: str = "expanded_ulysses", overlap: bool = False,
                 ulysses_degree: int | None = None, kv_exchange: str = "all_gather") -> None:
        """Choose the logical CP plan before parameter sharding and FSDP construction."""
        super().__init__(module=module, config=config, module_fqn=module_fqn, context=context,
                         strategy=strategy, overlap=overlap, ulysses_degree=ulysses_degree, kv_exchange=kv_exchange)
        if self.attention_dropout != 0 or self.sliding_window is not None:
            raise ValueError("MLA CP requires zero dropout and no implicit sliding window")
        # Head ownership assumes the original dense up-projection layout. A
        # quantized/adapter projection needs its own explicit slicing contract.
        if type(self.q_b_proj) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("MLA CP requires a plain Linear query up-projection")
        dimensions = MLADimensions(self.config.num_attention_heads, self.q_lora_rank, self.kv_lora_rank,
                                   self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim)
        cp_degree = (context or {}).get("cp_size", 1)
        self.cp_plan = MLACPPlan(dimensions, cp_degree, strategy=strategy, tp_degree=self.tp_degree,
                                 overlap=overlap, ulysses_degree=ulysses_degree, kv_exchange=kv_exchange)
        self.cp_plan.validate_scale(self.scaling)
        self.cp_runtime = None
        self.kv_b_proj = CPMLAKVLinear(self.kv_b_proj, self.qk_nope_head_dim + self.v_head_dim)
        self.prepare_output_projection()

    def prepare_output_projection(self) -> None:
        """Install the ordinary CP derivative boundary before sharding or FSDP hooks."""
        if self.cp_plan.overlap and self.cp_plan.tp_degree == 1 and self.cp_plan.ulysses_degree > 1:
            self.o_proj = CPMLAOutputLinear(self.o_proj)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare shared TP input preparation and the complete class-owned CP forward."""
        return ParallelModuleSpec(
            "attention.mla.cp", "cp",
            {"prepare_input": ("hidden_states",), "project_output": ("attention_output", "hidden_states")},
            validate=validate_mla_layout,
        )

    def bind_parallel(self, binding: Any) -> None:
        """Borrow CP resources and bind TP execution from the same final layout."""
        super().bind_parallel(binding)
        if binding.cp_context is None:
            raise ValueError("MLA CP requires a bound CPContext")
        self.cp_runtime = self.make_cp_execution(binding.cp_context)

    def make_cp_execution(self, context: CPContext) -> MLAContextParallel:
        """Create ordinary MLA scheduling over job-owned communication resources."""
        return MLAContextParallel(self.cp_plan, context.mesh)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
                actual_seq_len: torch.Tensor | Sequence[int] | None = None,
                **kwargs: Any) -> tuple[torch.Tensor, None]:
        """Run the established CP schedule without replacing any Module.forward at runtime."""
        if past_key_values is not None:
            raise ValueError("MLA CP training does not support a KV cache")
        if self.cp_runtime is None:
            raise RuntimeError("MLA CP resources must be bound before forward")
        hidden_states = self.prepare_input(hidden_states)
        batch, sequence = hidden_states.shape[:2]
        plan, runtime = self.cp_plan, self.cp_runtime
        overlap_output = plan.overlap and plan.tp_degree == 1 and plan.ulysses_degree > 1 and torch.is_grad_enabled()
        output, weights = runtime(
            self, *self.project_latent_inputs(hidden_states), position_embeddings=position_embeddings,
            attention_mask=attention_mask, actual_seq_len=actual_seq_len,
            restore_output=not overlap_output,
            overlap_input_backward=overlap_output and plan.all_gather_degree == 1, **kwargs,
        )
        if overlap_output:
            state = MLAOutputProjectionState(runtime.ulysses_mesh, self.v_head_dim)
            return self.o_proj(output, mla_output_state=state), weights
        return self.project_output(output.reshape(batch, sequence, -1).contiguous(), hidden_states), weights
