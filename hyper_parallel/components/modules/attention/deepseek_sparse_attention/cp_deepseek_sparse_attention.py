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
"""Packed CP transport and no-gradient selection scheduling for DeepSeek sparse attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch

import hyper_parallel.components.ops as local_ops
from hyper_parallel.components.functional.dsa_indexer import dsa_indexer, dsa_indexer_select
from hyper_parallel.components.functional.npu_fusion_attention import resolve_packed_sequence_lengths
from hyper_parallel.components.functional.sparse_attention_layout import cumulative_length_tensors
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.deepseek_sparse_attention import (
    DeepseekSparseAttentionModule,
    SparseAttentionInputs, SparseAttentionSequenceMetadata,
)
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.deepseek_sparse_attention_family import (
    sparse_attention_parallel_spec, validate_sparse_attention_layout,
)
from hyper_parallel.distributed.context_parallel.batch_plan import (
    CPBatchPlan, CPBatchPlanConfig, PackedDocumentPlan, packed_document_plans, prepare_cp_batch,
)
from hyper_parallel.distributed.context_parallel.indexer_balance import launch_indexer_inputs
from hyper_parallel.distributed.context_parallel.sequence_dispatch import async_sequence_dispatch
from hyper_parallel.distributed.context_parallel.cp_context import CPContext
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec


@dataclass(frozen=True)
class CPSparseAttentionSequenceMetadata(SparseAttentionSequenceMetadata):
    """CP kernel segments and optional ranges into the flattened global key bank."""

    key_ranges: tuple[tuple[int, int], ...] = ()
    document_plan: PackedDocumentPlan | None = field(default=None, kw_only=True)
    aligned_documents: bool = field(default=True, kw_only=True)
    batch_plan: CPBatchPlan | None = field(default=None, kw_only=True)


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
    if not 0 <= cp_rank < cp_size:
        raise ValueError("Packed CP rank must be within the context-parallel group")
    prepared = packed.get("cp_batch_plan")
    if isinstance(prepared, CPBatchPlan) and batch_size == 1 and queries == keys:
        lengths = tuple(end - start for start, end in zip((0, *queries[:-1]), queries))
        prepared.validate(lengths, cp_size, cp_rank)
        plan = prepared.query_documents
    else:
        plan = packed_document_plans(tuple(queries), tuple(keys), batch_size, local_seq_len, cp_size)[cp_rank]
    return CPSparseAttentionSequenceMetadata(
        (cp_rank + 1) * local_seq_len,
        *cumulative_length_tensors(plan.query_ends, plan.key_ends, device),
        plan.key_ranges, document_plan=plan, host_lengths=(plan.query_ends, plan.key_ends),
        aligned_documents=queries == keys,
    )


def _pack_key_prefixes(tensor: torch.Tensor, ranges: tuple) -> torch.Tensor:
    """Pack active prefixes without copying the common single-range case."""
    flat = tensor.flatten(0, 1)
    parts = [flat[begin:end] for begin, end in ranges]
    return (parts[0] if len(parts) == 1 else torch.cat(parts)).unsqueeze(0)


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
            _pack_key_prefixes(tensor, metadata.key_ranges)
            for tensor in gathered
        )
    else:
        index_key, latent_kv, key_rope = (
            tensor[:, :metadata.kv_prefix_length].contiguous() for tensor in gathered
        )
    return (index_query, index_key, weights), (query, latent_kv, query_rope, key_rope)


@dataclass
class _PendingMainBanks:
    """Keep merged storage alive until one invocation consumes both main fields."""

    receipts: tuple[Any, ...] | None
    key_ranges: tuple | None
    widths: tuple[int, ...] | None = None

    def wait(self) -> tuple[torch.Tensor, ...]:
        """Wait before reading transport outputs and return native-compatible layouts."""
        if self.receipts is None:
            raise RuntimeError("Main key-bank exchange was already consumed")
        banks = tuple(receipt.wait() if self.key_ranges is None else
                      _pack_key_prefixes(receipt.wait()[0], self.key_ranges) for receipt in self.receipts)
        self.receipts = None
        if self.widths is not None:
            # Split views have the combined row stride, which native kernels reject.
            banks = tuple(part.contiguous() for part in banks[0].split(self.widths, dim=-1))
        return banks


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
        if self.kv_cp_size != context.size:
            raise ValueError("kv_cp_size must equal the bound CPContext size")
        if self._cp_context is not None and self._cp_context.mesh is not context.mesh:
            raise ValueError("DeepSeek sparse-attention CP member already belongs to another mesh")
        super().bind_parallel(binding)
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

    def _launch_worker_inputs(self, inputs: SparseAttentionInputs) -> Any:
        """Launch detached selection operands; owner learning tensors stay local."""
        plan = inputs.metadata.batch_plan
        route = plan.for_ratio(1).worker_route
        if inputs.use_dense or route is None:
            return None
        qr, weights = inputs.query_residual, inputs.index_states[2]
        embeddings = inputs.position_embeddings
        if embeddings is None:
            shape = (1, qr.shape[1], self.qk_rope_head_dim)
            embeddings = (qr.new_ones(shape), qr.new_zeros(shape))
        else:
            embeddings = tuple(value.unsqueeze(0) if value.ndim == 2 else value for value in embeddings)
        return launch_indexer_inputs(
            qr, weights, embeddings, plan.lengths, 1, plan.config.mode, self._cp_context.mesh.get_group(),
            plan.config.block_size, plan.config.zigzag_min_length, prepared_route=route,
        )

    def _launch_banks(self, inputs: SparseAttentionInputs) -> tuple:
        """Submit Index K followed by separate or merged main-bank transport."""
        plan = inputs.metadata.batch_plan.for_ratio(1)
        context, group = self._cp_context, self._cp_context.mesh.get_group()
        if plan.index_key is None:
            index = context.launch_gather_sequence(inputs.index_states[1], dim=1)
        else:
            index = async_sequence_dispatch(
                inputs.index_states[1], 1, plan.index_key.dispatch, group, gradient=plan.index_key.gradient,
            )
        banks = (inputs.attention_states[1], inputs.attention_states[3])
        widths = None
        if self.main_kv_comm_fusion:
            if banks[0].dtype != banks[1].dtype or banks[0].shape[:-1] != banks[1].shape[:-1]:
                raise ValueError("Merged main KV communication requires matching leading shapes and dtype")
            widths = tuple(bank.shape[-1] for bank in banks)
            banks = (torch.cat(banks, dim=-1),)
        receipts = tuple(
            context.launch_gather_sequence(bank, dim=1) if plan.main_route is None
            else async_sequence_dispatch(bank, 1, plan.main_route, group)
            for bank in banks
        )
        main = _PendingMainBanks(
            receipts, inputs.metadata.key_ranges if plan.main_route is None else None, widths,
        )
        return index, main

    def _select_worker_queries(self, exchange: Any, worker: tuple,
                               selection_key: torch.Tensor, key_plan: Any) -> Any:
        """Produce document-local IDs and launch their return without repeating LI at the owner."""
        query, weights = worker
        if query is None:
            selected = weights.new_empty((0, self.index_topk), dtype=torch.int32)
        else:
            segments = exchange.route.segments
            if key_plan is not None:
                segments = key_plan.map_segments(segments)
            selected = dsa_indexer_select(
                query, selection_key, weights, segments, self.index_topk, paged_key=self.indexer_paged_k,
            )[:, 0]
        return exchange.launch_result(selected, restore_op=local_ops.prepare_indexer_topk_restore(selected))

    def exchange_attention(self, inputs: SparseAttentionInputs) -> SparseAttentionInputs:
        """Exchange banks and sparse selection while retaining the original owner learning layout.

        Args:
            inputs: Invocation-owned projection states and prepared metadata.
        """
        if inputs.metadata.batch_plan is None:
            index, attention = self.prepare_key_bank(inputs.index_states, inputs.attention_states, inputs.metadata)
            return replace(inputs, index_states=index, attention_states=attention, query_residual=None,
                           position_embeddings=None)
        plan = inputs.metadata.batch_plan.for_ratio(1)
        exchange = self._launch_worker_inputs(inputs)
        index_pending, main_pending = self._launch_banks(inputs)
        worker = None
        if exchange is not None:
            qr, weights, cos, sin = exchange.wait()
            query = None
            if exchange.route.query_count:
                query = self.project_worker_index_query(
                    qr, (cos, sin) if inputs.position_embeddings is not None else None, exchange.owner_queries,
                )
            worker = query, weights
        if plan.index_key is None:
            selection_key = index_pending.wait()[0]
            owner_key = _pack_key_prefixes(selection_key, inputs.metadata.key_ranges)
        else:
            selection_key, owner_key = index_pending.wait_with_gradient()
        index_query, _, weights = inputs.index_states
        selected = pending_topk = None
        if exchange is not None:
            pending_topk = self._select_worker_queries(exchange, worker, selection_key.detach(), plan.index_key)
        elif not inputs.use_dense:
            selected = dsa_indexer(
                index_query, owner_key, weights, *inputs.metadata.sequence_lengths, self.index_topk,
                host_lengths=inputs.metadata.host_lengths, paged_key=self.indexer_paged_k,
            )[0]
        query, _, query_rope, _ = inputs.attention_states
        if not inputs.use_dense:
            # These copies do not depend on returned IDs. Keep them before wait
            # so the same layouts used by SFA and KL can cover TopK transport.
            query, query_rope = query.contiguous(), query_rope.contiguous()
        main = main_pending.wait()
        if pending_topk is not None:
            selected = pending_topk.wait().unsqueeze(1)
        return replace(inputs, index_states=(index_query, owner_key, weights),
                       attention_states=(query, main[0], query_rope, main[1]), selected=selected,
                       query_residual=None, position_embeddings=None)

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
        """Validate every rank's packed geometry before any operand collective.

        Args:
            hidden_states: Local hidden states in batch-sequence-hidden layout.
            attention_mask: Optional causal-mask carrier; document ends define isolation.
            past_key_values: Reserved cache argument; training does not accept a KV cache.
            position_ids: Optional original token positions for the model rotary contract.
            actual_seq_len: Global cumulative document ends supplied by the caller.
            kwargs: Additional sequence metadata and optional immutable CP batch plan.
            use_dense: Whether this invocation uses dense warm-up instead of sparse attention.
        """
        if self._cp_context is None:
            raise ValueError("DeepSeek sparse-attention CP member must be bound before execution")
        self._validate_forward_inputs(hidden_states, None, past_key_values, position_ids, kwargs)
        self._validate_attention_mask(hidden_states, attention_mask)
        metadata = build_cp_sequence_metadata(
            batch_size=hidden_states.shape[0], local_seq_len=hidden_states.shape[1],
            device=hidden_states.device, cp_size=self._cp_context.size, cp_rank=self._cp_context.rank,
            actual_seq_len=actual_seq_len, kwargs=kwargs,
        )
        advanced = (self.indexer_load_balance != "none" or self.indexer_paged_k
                    or self.main_kv_comm != "all_gather" or self.index_k_comm != "all_gather"
                    or self.main_kv_comm_fusion)
        if not advanced and not self.cp_comm_overlap and kwargs.get("cp_batch_plan") is None:
            return metadata
        if not metadata.aligned_documents:
            if advanced or kwargs.get("cp_batch_plan") is not None:
                raise ValueError("DSA advanced CP options require matching self-attention document boundaries")
            return metadata
        if hidden_states.shape[0] != 1:
            if advanced or kwargs.get("cp_batch_plan") is not None:
                raise ValueError("DSA advanced CP options and batch prefetch require packed batch one")
            return metadata
        return self._prepare_batch_metadata(metadata, kwargs.get("cp_batch_plan"), use_dense=use_dense)

    def _prepare_batch_metadata(self, metadata: SparseAttentionSequenceMetadata, prepared: CPBatchPlan | None,
                                *, use_dense: bool) -> SparseAttentionSequenceMetadata:
        """Validate and bind an invocation plan, bypassing worker selection during dense warm-up."""
        config = CPBatchPlanConfig(
            self._cp_context.size, self._cp_context.rank, ratios=(1,), windows=(),
            mode=self.indexer_load_balance, block_size=self.indexer_balance_block_size,
            zigzag_min_length=self.indexer_zigzag_min_length,
            main_kv_comm=self.main_kv_comm, index_k_comm=self.index_k_comm,
        )
        if prepared is not None:
            if not isinstance(prepared, CPBatchPlan) or prepared.config != config:
                raise ValueError("Prepared DSA batch options differ from the attention module")
            prepared.validate(metadata.document_plan.lengths, config.size, config.rank)
        if use_dense and config.mode != "none":
            config = replace(config, mode="none", block_size=None, zigzag_min_length=None)
            prepared = None
        if prepared is None:
            prepared = prepare_cp_batch(metadata.document_plan.lengths, config)
        return replace(metadata, batch_plan=prepared)
