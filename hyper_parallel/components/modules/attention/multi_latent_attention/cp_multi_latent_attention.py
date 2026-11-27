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

import torch
from torch import nn

from hyper_parallel.components.modules.attention.multi_latent_attention.cp_mla_backward_overlap import (
    CPMLAKVLinear, CPMLAOutputLinear, MLAOutputProjectionState, MLAExpandedInputA2AState, MLAInputA2AState,
    _InputA2ALaunch, _LatentInputA2AWait,
    finish_expanded_inputs, launch_input_a2a,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention import (
    MLAInputs, MLALatentInputs, MultiLatentAttentionModule, apply_mla_rope as _apply_mla_rope,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention import (
    prepare_mla_attention_mask,
)
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention_family import (
    validate_mla_layout,
)
from hyper_parallel.distributed.context_parallel.collectives import (
    async_cp_allgather_launch,
)
from hyper_parallel.components.functional.npu_fusion_attention import (
    PACKED_SEQUENCE_ARGUMENTS, resolve_packed_sequence_lengths,
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
    state: MLALatentInputs,
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


def _validate_state(module: nn.Module, plan: MLACPPlan, state: MLALatentInputs) -> None:
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


class CPMultiLatentAttentionModule(MultiLatentAttentionModule):
    """Exchange MLA operands at class-defined stages of the shared algorithm.

    CP controls token/head ownership and projection readiness. All projection,
    position and attention mathematics belong to the inherited algorithm.
    """

    def __init__(self, *, module: nn.Module | None = None, config: Any = None,
                 module_fqn: str = "", context: Any = None,
                 strategy: str = "expanded_ulysses", overlap: bool = False,
                 ulysses_degree: int | None = None, kv_exchange: str = "all_gather",
                 rotary_fn: Callable | None = None, attention_interface: Callable | None = None) -> None:
        """Choose CP ownership before parameter sharding and FSDP construction.

        Args:
            module: Optional source attention with parameters to retain.
            config: Native mathematical configuration, exclusive with module.
            module_fqn: Construction context supplied by the family framework.
            context: Logical CP and TP sizes before resources are bound.
            strategy: Expanded QKV or latent KV representation for communication.
            overlap: Enable the existing supported forward/backward schedules.
            ulysses_degree: CP ranks assigned to attention heads.
            kv_exchange: AllGather or selective variable-size KV exchange.
            rotary_fn: Optional BSND position callback; must preserve shape and dtype.
            attention_interface: Optional FA callback returning BSND output and weights.
        """
        del module_fqn
        for name, callback in (("rotary_fn", rotary_fn), ("attention_interface", attention_interface)):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable")
        super().__init__(module=module, config=config)
        self.rotary_fn = rotary_fn
        if attention_interface is not None:
            self.attention_interface = attention_interface
        if self.attention_dropout != 0 or self.sliding_window is not None:
            raise ValueError("MLA CP requires zero dropout and no implicit sliding window")
        if type(self.q_b_proj) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("MLA CP requires a plain Linear query up-projection")
        dimensions = MLADimensions(self.config.num_attention_heads, self.q_lora_rank, self.kv_lora_rank,
                                   self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim)
        cp_degree = (context or {}).get("cp_size", 1)
        self.cp_plan = MLACPPlan(dimensions, cp_degree, strategy=strategy, tp_degree=(context or {}).get("tp_size", 1),
                                 overlap=overlap, ulysses_degree=ulysses_degree, kv_exchange=kv_exchange)
        self.cp_plan.validate_scale(self.scaling)
        self.cp_context = self.ulysses_context = self.query_context = None
        self.cp_mesh = self.ulysses_mesh = self.all_gather_mesh = None
        self.query_rank = self.head_rank = 0
        self.kv_b_proj = CPMLAKVLinear(self.kv_b_proj, self.qk_nope_head_dim + self.v_head_dim)
        self.prepare_output_projection()

    def prepare_output_projection(self) -> None:
        """Keep the existing managed dO/wgrad scheduling boundary."""
        if self.cp_plan.overlap and self.cp_plan.tp_degree == 1 and self.cp_plan.ulysses_degree > 1:
            self.o_proj = CPMLAOutputLinear(self.o_proj)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare operand/output stages; framework TP still owns its outer ports."""
        return ParallelModuleSpec(
            "attention.mla.cp", "cp",
            {"exchange_attention": ("state",),
             "restore_attention": ("output", "hidden_states", "output_state"),
             "project_output": ("attention_output", "hidden_states", "output_state")},
            validate=validate_mla_layout,
        )

    def bind_parallel(self, binding: Any) -> None:
        """Borrow CP transport directly, without installing a whole-layer executor.

        Args:
            binding: Framework-owned CP context, including existing process groups.
        """
        context = binding.cp_context
        if context is None:
            raise ValueError("MLA CP requires a bound CPContext")
        if context.size != self.cp_plan.cp_degree:
            raise ValueError("MLACPPlan degree must match the CP mesh")
        self.cp_context, self.cp_mesh = context, context.mesh
        self.query_rank, self.head_rank = divmod(context.rank, self.cp_plan.ulysses_degree)
        self.ulysses_context, self.query_context = context.split(self.cp_plan.ulysses_degree)
        self.ulysses_mesh, self.all_gather_mesh = self.ulysses_context.mesh, self.query_context.mesh

    def _overlap_output_backward(self) -> bool:
        """Keep the existing TP1 output adjoint; input pairing also requires pure Ulysses."""
        plan = self.cp_plan
        return plan.overlap and plan.tp_degree == 1 and plan.ulysses_degree > 1 and torch.is_grad_enabled()

    def project_output(self, attention_output: torch.Tensor, hidden_states: torch.Tensor,
                       output_state: Any) -> torch.Tensor:
        """Consume deferred restoration inside the existing managed output Function.

        Args:
            attention_output: Local BSND output, or raw head-partitioned FA output when deferred.
            hidden_states: Original local tokens defining the output shape.
            output_state: MLAOutputProjectionState for joint O exchange/Wo, or algorithm state.
        """
        if isinstance(output_state, MLAOutputProjectionState):
            return self.o_proj(attention_output, mla_output_state=output_state)
        return super().project_output(attention_output, hidden_states, output_state)

    def validate_inputs(self, past_key_values: Any) -> None:
        """Reject unsupported calls before projection or collective submission.

        Args:
            past_key_values: Must be None for CP training with fixed token ownership.
        """
        if past_key_values is not None:
            raise ValueError("MLA CP training does not support a KV cache")
        if self.cp_context is None:
            raise RuntimeError("MLA CP resources must be bound before forward")

    def exchange_attention(self, state: MLALatentInputs,
                           position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                           attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
                           actual_seq_len: torch.Tensor | Sequence[int] | None = None, *,
                           output_state: Any = None, is_causal: bool | None = None,
                           **kwargs: Any) -> MLAInputs:
        """Schedule algorithm projections around CP transfers and return ready FA operands.

        Expanded exchange follows projection; latent KV exchange precedes the
        head-selective KV projection. This stage schedules the shared math methods,
        not just collectives. The inherited forward owns the subsequent FA call.

        Args:
            state: Already-computed algorithm latents and shared-key features.
            position_embeddings: Local RoPE tables, already discarded by NoPE algorithms.
            attention_mask: Optional global mask.
            past_key_values: Unsupported by CP training.
            actual_seq_len: Global cumulative document endpoints.
            output_state: Optional input-side gate dependency for this invocation.
            is_causal: Optional causal override.
            **kwargs: Supported packed and model attention options.
        """
        self.validate_inputs(past_key_values)
        mask, positions, partition, options = self._attention_layout(
            state, position_embeddings, attention_mask, actual_seq_len, is_causal, kwargs)
        rotary_fn = self.apply_position if self.rotary_fn is None else self.rotary_fn
        rotated_key = rotary_fn(state.key_rope.unsqueeze(2), positions,
                                interleaved=self.rotary_interleaved).squeeze(2)
        overlap_input = self._overlap_output_backward() and self.cp_plan.all_gather_degree == 1
        if output_state is not None:
            overlap_input = output_state.defer_gate_gradient
        query, key, value = self._prepare_distributed_inputs(
            state, positions, rotated_key, partition, overlap_input, output_state, rotary_fn)
        return MLAInputs(query, key, value, mask, options, output_state)

    def _attention_layout(self, state, positions, attention_mask, actual_seq_len, is_causal, kwargs):
        """Resolve global document ownership before submitting any collective."""
        packed_kwargs = {name: kwargs.pop(name) for name in PACKED_SEQUENCE_ARGUMENTS if name in kwargs}
        packed_kwargs["actual_seq_len"] = actual_seq_len
        _validate_forward_kwargs(kwargs, positions)
        dimensions, degree = self.cp_plan.dimensions, self.cp_plan.cp_degree
        batch, sequence = state.query_latent.shape[:2]
        lengths, key_lengths = resolve_packed_sequence_lengths(
            packed_kwargs, batch * sequence * degree, batch * sequence * degree)
        if lengths is not None and batch != 1:
            raise ValueError("packed MLA CP currently requires batch_size=1")
        if lengths != key_lengths:
            raise ValueError("MLA CP requires matching global query and key document boundaries")
        causal = self.is_causal if is_causal is None else is_causal
        if not isinstance(causal, bool):
            raise ValueError("is_causal must be bool")
        _validate_mask(attention_mask, lengths, batch, sequence * degree, state.query_latent.device)
        attention_mask = prepare_mla_attention_mask(attention_mask, sequence * degree, sequence * degree)
        partition = None
        if self.cp_plan.all_gather_degree > 1:
            _validate_query_partition(causal, attention_mask)
            partition = _causal_partition(
                lengths or (sequence * degree,), sequence * self.cp_plan.ulysses_degree, self.query_rank, batch)
            lengths, key_lengths = partition.query_lengths, partition.key_lengths
        positions = _positions(positions, state, dimensions.rope_dim)
        _validate_state(self, self.cp_plan, state)
        options = {"dropout": 0.0, "sliding_window": None, "is_causal": causal,
                   "actual_seq_qlen": lengths, "actual_seq_kvlen": key_lengths,
                   "pre_tokens": 2147483647, "next_tokens": 0 if causal else 2147483647,
                   "mla_head_offset": self.head_rank * self.cp_plan.compute_heads,
                   "mla_local_heads": self.cp_plan.local_heads}
        return attention_mask, positions, partition, options

    def _prepare_distributed_inputs(self, state, positions, key_rope, partition,
                                    overlap_input_backward, input_gate, rotary_fn):
        """Select input preparation; paired input backward requires pure Ulysses.

        Forward overlap also supports KV AllGather and hybrid CP. Those paths
        retain native input collective adjoints; their output dO overlap is
        selected independently by the output stage.
        """
        use_forward_overlap = self.cp_plan.overlap and self.cp_plan.cp_degree > 1
        pure_ulysses = self.cp_plan.all_gather_degree == 1
        expanded = self.cp_plan.strategy == "expanded_ulysses"
        supports_input_gate = use_forward_overlap and pure_ulysses and expanded
        if input_gate is not None and not supports_input_gate:
            raise ValueError("Input gate overlap requires expanded pure Ulysses with overlap enabled")
        if overlap_input_backward and not pure_ulysses:
            raise ValueError("Paired MLA input backward currently requires pure Ulysses")
        if not use_forward_overlap:
            query = self.project_query(
                state.query_latent, positions, heads=self.cp_plan.local_heads, rotary_fn=rotary_fn,
            )
            prepare = self._expanded_inputs if expanded or self.cp_plan.cp_degree == 1 else self._latent_inputs
            return prepare(state, query, key_rope, partition)
        # Pure Ulysses has no outer KV AllGather. Expanded Q/K/V use the same
        # V/K/Q forward order with or without a gate; overlap_input_backward
        # selects paired input gradients versus the native collective adjoints.
        if pure_ulysses and expanded:
            return self._prepare_expanded_ulysses_inputs(
                state, positions, key_rope, input_gate, overlap_input_backward, rotary_fn,
            )
        if overlap_input_backward:
            # Expanded has returned above and AG/hybrid were rejected earlier:
            # this is latent pure Ulysses, pairing Q A2A with KV projection/RS.
            return self._prepare_latent_ulysses_inputs(state, positions, key_rope, rotary_fn)
        # AG/hybrid inputs and forward-only latent Ulysses use native adjoints.
        prepare = self._expanded_overlap_inputs if expanded else self._latent_overlap_inputs
        return prepare(state, positions, key_rope, partition, rotary_fn=rotary_fn)

    def restore_attention(self, output: torch.Tensor, hidden_states: torch.Tensor,
                          output_state: Any) -> tuple[torch.Tensor, Any]:
        """Restore query ownership, or defer O exchange to the joint output Function.

        Args:
            output: Attention result in BSND head-partitioned layout.
            hidden_states: Original local input, available to gated scheduling overrides.
            output_state: Algorithm-owned invocation state.

        Returns:
            Local BSND output and algorithm state, or unchanged FA output with an
            MLAOutputProjectionState. The latter keeps O A2A and Wo in one autograd
            Function, allowing backward dO communication to overlap Wo wgrad.
        """
        del hidden_states
        if self._overlap_output_backward():
            return output, MLAOutputProjectionState(self.ulysses_mesh, self.v_head_dim)
        batch, sequence = output.shape[:2]
        sequence //= self.cp_plan.ulysses_degree
        if self.cp_plan.ulysses_degree > 1:
            output = self.ulysses_context.head_to_sequence(output)
        expected = (batch, sequence, self.cp_plan.local_heads, self.cp_plan.dimensions.value_dim)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"MLA CP output has shape {tuple(output.shape)}, expected {expected}")
        return output, output_state

    def _expanded_inputs(self, state, query, key_rope, partition=None):
        """Pack unequal-width QKV into one sequence-to-head exchange."""
        key, value = self.project_key_value(state.kv_latent, key_rope, heads=self.cp_plan.local_heads)
        return self._exchange_qkv(query, key, value, partition)

    def _exchange_qkv(self, query, key, value, partition):
        """Exchange completed expanded operands without performing projection math."""
        if self.cp_plan.ulysses_degree > 1:
            widths = (query.shape[-1], key.shape[-1], value.shape[-1])
            payload = torch.cat(tuple(tensor.transpose(1, 2) for tensor in (query, key, value)), dim=-1)
            payload = self.ulysses_context.sequence_to_head(payload)
            query, key, value = (tensor.transpose(1, 2) for tensor in payload.split(widths, dim=-1))
        if self.cp_plan.all_gather_degree > 1:
            widths = (key.shape[-1], value.shape[-1])
            payload = torch.cat((key.transpose(1, 2), value.transpose(1, 2)), dim=-1)
            if self.cp_plan.kv_exchange == "all_to_all_v":
                payload = _selective_kv_launch(payload, partition, self.all_gather_mesh, 1, False).wait()
            else:
                payload = partition.select(self.query_context.gather_sequence(payload)[0], 1)
            key, value = (tensor.transpose(1, 2) for tensor in payload.split(widths, dim=-1))
        return query, key, value

    def _latent_inputs(self, state, query, key_rope, partition=None):
        """Exchange Q heads, gather shared KV latents, then expand only owned heads."""
        if self.cp_plan.ulysses_degree > 1:
            query = self.ulysses_context.sequence_to_head(query, seq_dim=2, head_dim=1)
        payload = torch.cat((state.kv_latent, key_rope), dim=-1)
        payload = self._launch_latent_kv(payload, partition, overlap=False).wait()
        return query, *self._expand_latent(payload, partition)

    def _launch_latent_kv(self, payload, partition=None, *, overlap=True):
        """Exchange compressed KV before the algorithm's head-selective expansion."""
        if self.cp_plan.kv_exchange == "all_to_all_v":
            return _selective_kv_launch(payload, partition, self.cp_mesh, self.cp_plan.ulysses_degree, overlap)
        return self.cp_context.launch_gather_sequence(payload).collective

    def _expand_latent(self, payload, partition=None):
        """Keep the same managed head-selective projection for both communication schedules."""
        if partition is not None and self.cp_plan.kv_exchange == "all_gather":
            payload = partition.select(payload, 1)
        dimensions = self.cp_plan.dimensions
        heads = self.cp_plan.compute_heads
        start = self.head_rank * heads
        kv_latent, key_rope = payload.split((dimensions.kv_rank, dimensions.rope_dim), dim=-1)
        # A contiguous latent preserves the native Linear bias-fusion path after payload splitting.
        return self.project_key_value(
            kv_latent.contiguous(), key_rope, heads=heads, head_range=(start, start + heads),
        )

    def _expanded_overlap_inputs(self, state, positions, key_rope, partition=None, *, rotary_fn):
        """Overlap KV transfer with Q preparation, respecting the two-stage hybrid dependency."""
        key, value = self.project_key_value(state.kv_latent, key_rope, heads=self.cp_plan.local_heads)
        payload = torch.cat((key.transpose(1, 2), value.transpose(1, 2)), dim=-1)
        use_ulysses = self.cp_plan.ulysses_degree > 1
        if use_ulysses:
            kv_transfer_handle = self.ulysses_context.launch_sequence_to_head(payload)
        else:
            kv_transfer_handle = self._launch_expanded_kv(payload, partition)
        query = self.project_query(state.query_latent, positions, heads=self.cp_plan.local_heads, rotary_fn=rotary_fn)
        query_a2a_handle = (self.ulysses_context.launch_sequence_to_head(query.transpose(1, 2))
                            if use_ulysses else None)
        payload = kv_transfer_handle.wait()
        if use_ulysses and self.cp_plan.all_gather_degree > 1:
            payload = self._launch_expanded_kv(payload, partition).wait()
        if query_a2a_handle is not None:
            query = query_a2a_handle.wait().transpose(1, 2)
        if partition is not None and self.cp_plan.kv_exchange == "all_gather":
            payload = partition.select(payload, 1)
        key, value = payload.split((self.qk_head_dim, self.v_head_dim), dim=-1)
        return query, key.transpose(1, 2), value.transpose(1, 2)

    def _launch_expanded_kv(self, payload, partition):
        """Complete the sequence dimension after Ulysses has selected this head shard."""
        if self.cp_plan.kv_exchange == "all_to_all_v":
            return _selective_kv_launch(payload, partition, self.all_gather_mesh, 1, True)
        return self.query_context.launch_gather_sequence(payload).collective

    def _latent_overlap_inputs(self, state, positions, key_rope, partition=None, *, rotary_fn):
        """Cover latent transfer with Q preparation, then Q A2A with KV expansion."""
        payload = torch.cat((state.kv_latent, key_rope), dim=-1)
        kv_transfer_handle = self._launch_latent_kv(payload, partition)
        query = self.project_query(state.query_latent, positions, heads=self.cp_plan.local_heads, rotary_fn=rotary_fn)
        payload = kv_transfer_handle.wait()
        query_a2a_handle = (self.ulysses_context.launch_sequence_to_head(query.transpose(1, 2))
                            if self.cp_plan.ulysses_degree > 1 else None)
        key, value = self._expand_latent(payload, partition)
        if query_a2a_handle is not None:
            query = query_a2a_handle.wait().transpose(1, 2)
        return query, key, value

    def _prepare_expanded_ulysses_inputs(self, state, positions, key_rope, input_gate,
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
        key_nope, value = self.project_key_value_parts(state.kv_latent, heads=self.cp_plan.local_heads)
        # Keep K assembly inside V's communication window.
        key_nope = launch_input(value, key_nope, input_a2a.value)
        key = self.assemble_key(key_nope, key_rope)
        # K is complete; Q projection and its model-defined rotation can cover it.
        query_latent = launch_input(key, state.query_latent, input_a2a.key)
        query = self.project_query(
            query_latent, positions, heads=self.cp_plan.local_heads, rotary_fn=rotary_fn,
        ).transpose(1, 2)
        query, key, value = self.finish_input_exchange(query, input_a2a, overlap_input_backward, input_gate)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def finish_input_exchange(self, query: torch.Tensor, transfers: Any, overlap_backward: bool,
                              input_gate: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Finish ordinary Q/K/V; gated MLA overrides this model-specific boundary.

        Args:
            query: Expanded query whose exchange has not yet been launched.
            transfers: Pending key/value exchanges and their backward state.
            overlap_backward: Pair backward communication with independent projections.
            input_gate: Must be None for ordinary MLA.
        """
        if input_gate is not None:
            raise ValueError("Ordinary MLA cannot consume a gated input dependency")
        return finish_expanded_inputs(query, transfers, overlap_backward)

    def _prepare_latent_ulysses_inputs(self, state, positions, key_rope, rotary_fn):
        """Prepare latent Ulysses inputs and pair their Q/KV backward dependencies."""
        payload = torch.cat((state.kv_latent, key_rope), dim=-1)
        kv_all_gather_handle = self._launch_latent_kv(payload)
        query = self.project_query(
            state.query_latent, positions, heads=self.cp_plan.local_heads, rotary_fn=rotary_fn,
        ).transpose(1, 2)
        # wait() normally adds the default AllGather autograd node. Instead,
        # _LatentKVLinear launches SUM RS before KV wgrad and returns dPayload
        # via kv_input. Keep this numeric buffer off the default gradient path.
        with torch.no_grad():
            gathered = kv_all_gather_handle.wait()
        query_a2a = MLAInputA2AState(self.ulysses_mesh)
        # Same payload value, with Q's inverse A2A bracketing its KV backward.
        kv_input = _InputA2ALaunch.apply(query, payload, query_a2a)
        heads = self.cp_plan.compute_heads
        start = self.head_rank * heads
        projected, shared_key = self.kv_b_proj(
            kv_input, mla_head_range=(start, start + heads),
            mla_gathered=gathered, mla_cp_mesh=self.cp_mesh,
        )
        key, value = self.assemble_key_value(projected, shared_key, heads=heads)
        query, key, value = _LatentInputA2AWait.apply(key, value, query_a2a)
        return query.transpose(1, 2), key, value
