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
"""Explicit local-query KV-AllGather scheduling for DeepSeek sparse attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from hyper_parallel.components.functional.npu_fusion_attention import resolve_packed_sequence_lengths
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.deepseek_sparse_attention import (
    DeepseekSparseAttentionModule,
    SparseAttentionInputs, SparseAttentionSequenceMetadata,
)
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.deepseek_sparse_attention_family import (
    sparse_attention_parallel_spec, validate_sparse_attention_layout,
)
from hyper_parallel.distributed.context_parallel.cp_context import CPContext
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec


@dataclass(frozen=True)
class CPSparseAttentionSequenceMetadata(SparseAttentionSequenceMetadata):
    """CP kernel segments and optional ranges into the flattened global key bank."""

    key_ranges: tuple[tuple[int, int], ...] = ()


def _split_packed_prefixes(
    queries: Sequence[int], keys: Sequence[int], *, global_seq_len: int, local_seq_len: int, rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Pair each contiguous local Q segment with its complete document KV prefix."""
    query_ends, key_ends, key_ranges = [], [], []
    query_start = key_start = query_total = key_total = 0
    for query_end, key_end in zip(queries, keys):
        for row in range(query_start // global_seq_len, (query_end - 1) // global_seq_len + 1):
            shard_start = row * global_seq_len + rank * local_seq_len
            begin, end = max(query_start, shard_start), min(query_end, shard_start + local_seq_len)
            if begin >= end:
                continue
            prefix_end = key_end - (query_end - end)
            if prefix_end <= key_start:
                raise ValueError(f"Sparse attention has an empty packed KV segment on CP rank {rank}")
            query_total += end - begin
            key_total += prefix_end - key_start
            query_ends.append(query_total)
            key_ends.append(key_total)
            key_ranges.append((key_start, prefix_end))
        query_start, key_start = query_end, key_end
    return tuple(query_ends), tuple(key_ends), tuple(key_ranges)


def _project_ends(
    boundaries: Sequence[int], *, global_seq_len: int, local_seq_len: int, rank: int, query: bool,
) -> tuple[int, ...]:
    """Intersect every document with the Q shard or causal KV prefix in each row."""
    previous = total = 0
    result = []
    for end in boundaries:
        length = 0
        for row in range(previous // global_seq_len, (end - 1) // global_seq_len + 1):
            row_start = row * global_seq_len
            begin = row_start + rank * local_seq_len if query else row_start
            stop = row_start + (rank + 1) * local_seq_len
            length += max(0, min(end, stop) - max(previous, begin))
        total += length
        result.append(total)
        previous = end
    return tuple(result)


def build_cp_sequence_metadata(
    *, batch_size: int, local_seq_len: int, device: torch.device, cp_size: int, cp_rank: int,
    actual_seq_len: torch.Tensor | Sequence[int] | None = None, kwargs: Mapping[str, Any] | None = None,
) -> CPSparseAttentionSequenceMetadata:
    """Project packed documents after checking every rank before communication.

    Args:
        batch_size: Number of batch-major rows.
        local_seq_len: Equal contiguous token count per row and CP rank.
        device: Kernel metadata device.
        cp_size: Context-parallel degree.
        cp_rank: Rank within the borrowed CP group.
        actual_seq_len: Global cumulative packed endpoints without leading zero.
        kwargs: Supported packed aliases, including packed_seq_params.

    Returns:
        This rank's cumulative Q/K endpoints and the consumed key-bank ranges.
    """
    global_length = local_seq_len * cp_size
    total = batch_size * global_length
    packed = dict(kwargs or {})
    if actual_seq_len is not None:
        packed["actual_seq_len"] = actual_seq_len
    queries, keys = resolve_packed_sequence_lengths(packed, total, total)
    if queries is None:
        queries = keys = [(row + 1) * global_length for row in range(batch_size)]
    if len(queries) != len(keys):
        raise ValueError("Sparse attention requires the same number of packed Q and KV documents")
    cross_row = cp_size > 1 and any(
        begin // global_length != (end - 1) // global_length
        for boundaries in (queries, keys)
        for begin, end in zip((0, *boundaries[:-1]), boundaries)
    )
    projected = []
    for rank in range(cp_size):
        query = _project_ends(
            queries, global_seq_len=global_length, local_seq_len=local_seq_len, rank=rank, query=True,
        )
        key = _project_ends(
            keys, global_seq_len=global_length, local_seq_len=local_seq_len, rank=rank, query=False,
        )
        if any(end == previous for previous, end in zip((0, *query[:-1]), query)):
            raise ValueError(f"Sparse attention does not support zero-query packed documents on CP rank {rank}")
        if any(end == previous for previous, end in zip((0, *key[:-1]), key)):
            raise ValueError(f"Sparse attention has an empty packed KV document on CP rank {rank}")
        key_ranges = ()
        if cross_row:
            # A rank's cross-row queries are not a contiguous document suffix.
            # CANN's right-aligned causal mask needs one segment per intersection.
            query, key, key_ranges = _split_packed_prefixes(
                queries, keys, global_seq_len=global_length, local_seq_len=local_seq_len, rank=rank,
            )
        projected.append((query, key, key_ranges))
    local_query, local_key, key_ranges = projected[cp_rank]
    return CPSparseAttentionSequenceMetadata(
        (cp_rank + 1) * local_seq_len,
        torch.tensor(local_query, dtype=torch.int32, device=device),
        torch.tensor(local_key, dtype=torch.int32, device=device),
        key_ranges,
    )


def _gather_key_banks(
    context: CPContext, overlap: bool,
    index_states: tuple[torch.Tensor, ...], attention_states: tuple[torch.Tensor, ...],
    metadata: CPSparseAttentionSequenceMetadata,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Gather the three key banks, then select complete causal prefixes."""
    index_query, index_key, weights = index_states
    query, latent_kv, query_rope, key_rope = attention_states
    tensors = (index_key, latent_kv, key_rope)
    if overlap:
        pending = tuple(context.launch_gather_sequence(tensor, dim=1) for tensor in tensors)
        gathered = tuple(receipt.wait()[0] for receipt in pending)
    else:
        gathered = tuple(context.gather_sequence(tensor, dim=1)[0] for tensor in tensors)
    if metadata.key_ranges:
        # Repeated prefixes remain in autograd so their key gradients accumulate
        # before the existing gather adjoint reduces them to the source shards.
        index_key, latent_kv, key_rope = (
            torch.cat([tensor.flatten(0, 1)[begin:end] for begin, end in metadata.key_ranges]).unsqueeze(0)
            for tensor in gathered
        )
    else:
        index_key, latent_kv, key_rope = (
            tensor[:, :metadata.kv_prefix_length].contiguous() for tensor in gathered
        )
    return (index_query, index_key, weights), (query, latent_kv, query_rope, key_rope)


class CPDeepseekSparseAttentionModule(DeepseekSparseAttentionModule):
    """Borrow CP transport while reusing the local constructor and kernel stages.

    Binding installs the borrowed context on each instance; construction needs no mesh.
    """

    _cp_context: CPContext | None = None

    @staticmethod

    def parallel_module_spec() -> ParallelModuleSpec:
        """Declare operand exchange while the local algorithm owns mathematical dispatch."""
        local_spec = sparse_attention_parallel_spec(module=None, member=None)
        return ParallelModuleSpec(
            "attention.deepseek_sparse_attention.cp", binding_kind="cp",
            stages=local_spec.stages,
            validate=validate_sparse_attention_layout,
        )

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Borrow an existing CPContext before FSDP installs projection hooks."""
        context = binding.cp_context
        if context is None or context.size <= 1:
            raise ValueError("DeepSeek sparse-attention CP member requires an active CPContext")
        if self.kv_cp_size is not None and self.kv_cp_size != context.size:
            raise ValueError("kv_cp_size must equal the bound CPContext size")
        if self._cp_context is not None and self._cp_context.mesh is not context.mesh:
            raise ValueError("DeepSeek sparse-attention CP member already belongs to another mesh")
        super().bind_parallel(binding)

        self.kv_cp_size = context.size
        self._cp_context = context

    def prepare_key_bank(
        self, index_states: tuple[torch.Tensor, ...], attention_states: tuple[torch.Tensor, ...],
        metadata: CPSparseAttentionSequenceMetadata,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Gather three banks in fixed order and consume their causal prefixes.

        All readiness receipts belong to this invocation. The existing gather
        adjoint sums consumers into each source shard, including the zero
        gradients from unused suffix tokens.
        """
        return _gather_key_banks(self._cp_context, self.cp_comm_overlap, index_states, attention_states, metadata)

    def exchange_attention(self, inputs: SparseAttentionInputs) -> SparseAttentionInputs:
        """Return ready key banks without taking ownership of the attention algorithm."""
        index_states, attention_states = self.prepare_key_bank(
            inputs.index_states, inputs.attention_states, inputs.metadata,
        )
        return SparseAttentionInputs(index_states, attention_states, inputs.metadata, inputs.use_dense)

    def _validate_attention_mask(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> None:
        """Accept existing causal-mask carriers without reading their device contents."""
        if attention_mask is None:
            return
        batch_size, local_length = hidden_states.shape[:2]
        global_length = local_length * self._cp_context.size
        prefix_length = local_length * (self._cp_context.rank + 1)
        shapes = {
            (local_length, local_length), (local_length, global_length),
            (global_length, global_length), (local_length, prefix_length),
        }
        # TextParallelBatch constructs a global square mask after slicing token fields.
        # Packed endpoints, rather than mask values, remain the CANN causality contract.
        if (attention_mask.ndim != 4 or attention_mask.shape[0] not in (1, batch_size)
                or attention_mask.shape[1] != 1 or tuple(attention_mask.shape[-2:]) not in shapes):
            raise ValueError(
                "DeepSeek sparse attention CP requires a standard 4D causal mask "
                "with local/global sequence dimensions; "
                "padding and custom masks are not supported"
            )

    def preflight_attention(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None,
        past_key_values: Any | None, position_ids: torch.Tensor | None,
        actual_seq_len: torch.Tensor | Sequence[int] | None,
        kwargs: dict[str, Any], *, use_dense: bool,
    ) -> SparseAttentionSequenceMetadata:
        """Validate every rank's packed geometry before any operand collective."""
        del use_dense
        if self._cp_context is None:
            raise ValueError("DeepSeek sparse-attention CP member must be bound before execution")
        self._validate_forward_inputs(hidden_states, None, past_key_values, position_ids, kwargs)
        self._validate_attention_mask(hidden_states, attention_mask)
        return build_cp_sequence_metadata(
            batch_size=hidden_states.shape[0], local_seq_len=hidden_states.shape[1],
            device=hidden_states.device, cp_size=self._cp_context.size, cp_rank=self._cp_context.rank,
            actual_seq_len=actual_seq_len, kwargs=kwargs,
        )
