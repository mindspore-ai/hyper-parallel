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
"""CSA2 parallel stages: bank exchange, Indexer dispatch and internal head SUM."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol

from hyper_parallel.core.activation_memory.api import checkpoint_exclude_wrapper
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.utils.communication import differentiable_all_to_all_single
from hyper_parallel.distributed.context_parallel.batch_plan import CPBatchPlan
from hyper_parallel.distributed.context_parallel.index_key_dispatch import IndexKeyPlan, index_key_plan
from hyper_parallel.distributed.context_parallel.compressed_document import (
    document_compression_plan, launch_compressed_bank,
)
from hyper_parallel.distributed.context_parallel.indexer_balance import (
    launch_indexer_inputs, resolve_indexer_balance_block_size, validate_indexer_hybrid_threshold,
)
from hyper_parallel.distributed.context_parallel.sequence_dispatch import (
    async_sequence_dispatch, document_prefix_route, sliding_window_route,
)
from .compressed_sparse_attention2 import (
    CSA2AttentionInputs, CSA2DocumentInputs, CompressedSparseAttention2Module,
    SharedCompressedAttentionState, _apply_v41_rope,
)
from .compressor import CSA2Compressor
from .document_layout import SharedCompressedPackedSequence
from .indexer import CSA2Indexer, CSA2IndexerOutput, compressed_causal_topk


def _reduce_scores(tensor: torch.Tensor, *, group: Any) -> torch.Tensor:
    """Sum detached Indexer/teacher head contributions over TP."""
    return funcol.all_reduce(tensor.contiguous(), "sum", group)


def _launch_bank(tensor: torch.Tensor, dim: int, context: Any) -> Any:
    """Borrow the framework's differentiable asynchronous sequence gather."""
    return context.launch_gather_sequence(tensor, dim=dim).collective


def _exchange_boundary(tensor: torch.Tensor, route: Any, context: Any) -> torch.Tensor:
    """Complete projected boundary rows, including reverse participation for empty receives."""
    if not route.needs_exchange:
        return tensor
    local = tensor.movedim(1, 0).contiguous()
    parts = [local.narrow(0, offset, count)
             for offset, count in zip(route.send_offsets, route.input_splits) if count]
    send = torch.cat(parts, dim=0) if parts else local[:0]
    received = differentiable_all_to_all_single(
        send, list(route.input_splits), list(route.output_splits), group=context.mesh.get_group(),
    )
    return torch.cat((received, local), dim=0).movedim(0, 1).contiguous()


class _CPCompressor(CSA2Compressor):
    """Keep projection and boundary exchange inside the managed compressor call."""

    cp_context = None

    @classmethod
    def from_module(cls, module: CSA2Compressor) -> _CPCompressor:
        """Adopt children by identity without initializing another parameter tree."""
        result = object.__new__(cls)
        torch.nn.Module.__init__(result)  # pylint: disable=unnecessary-dunder-call
        for name, child in module.named_children():
            result.add_module(name, child)
        for name, parameter in module._parameters.items():  # pylint: disable=protected-access
            result.register_parameter(name, parameter)
        # PyTorch has no public getter for buffer persistence, including None buffers.
        non_persistent_buffers = module._non_persistent_buffers_set  # pylint: disable=protected-access
        for name, buffer in module._buffers.items():  # pylint: disable=protected-access
            result.register_buffer(name, buffer, persistent=name not in non_persistent_buffers)
        result.compress_ratio = module.compress_ratio
        result.apply_rope = module.apply_rope
        result.training = module.training
        return result

    def exchange_blocks(self, projected: torch.Tensor, block_plan: Any) -> torch.Tensor:
        """Exchange only the missing projected rows for locally owned compression blocks."""
        return _exchange_boundary(projected, block_plan.route, self.cp_context)


@dataclass
class _MainAttentionWork:
    """Main KV submission and query work executed after the Index K submission."""

    query_residual: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    launch_main_bank: Callable[[], Any] | None
    compressed_handle: Any = None
    query: torch.Tensor | None = None


@dataclass
class _PendingAttention:
    """Per-call communication receipts, never stored in cross-layer shared state."""

    query: torch.Tensor
    raw_handle: Any
    compressed_handle: Any
    indexer_output: Any
    raw_offset: int
    main_offset: int


class _CPIndexer(CSA2Indexer):
    """Schedule shared keys and no-grad qr/TopK while Indexer weights remain resident."""

    cp_context = None
    balance_mode = "none"
    index_k_comm = "all_gather"
    balance_block_size = None
    zigzag_min_length = None
    _input_region = None
    _result_region = None
    _wait_region = None

    def prepare_key(self, latent: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor,
                    index_key: torch.Tensor | None, block_plan: Any, key_position_embeddings: tuple | None,
                    key_plan: IndexKeyPlan | None = None) -> Any:
        """Submit Full-layer K; Reindex reuses the already differentiable shared bank."""
        key = self._project_index_key(latent, cos, sin, index_key, key_position_embeddings)
        if not self.owns_key:
            return key
        if key_plan is not None:
            return async_sequence_dispatch(key, 1, key_plan.dispatch, self.cp_context.mesh.get_group(),
                                           gradient=key_plan.gradient)
        launch = partial(_launch_bank, context=self.cp_context)
        return launch(key, 1) if block_plan is None else launch_compressed_bank(key, block_plan.counts, launch)

    def exchange_key(self, prepared: Any) -> torch.Tensor:
        """Wait only when the local/worker selection consumes Index K."""
        return prepared if isinstance(prepared, torch.Tensor) else prepared.wait()

    def _resolve_key_plan(self, lengths: tuple, source_plan: IndexKeyPlan | None,
                          prepared_batch: CPBatchPlan | None) -> IndexKeyPlan | None:
        """Require a single physical bank layout across Full/Reindex consumers."""
        plan = None
        if self.index_k_comm == "all_to_all_v":
            if self.is_candidate_source or self.uses_candidates:
                raise NotImplementedError("Index K prefix dispatch requires ordinary causal selection")
            plan = (prepared_batch.for_ratio(self.compress_ratio).index_key if prepared_batch is not None else
                    index_key_plan(lengths, self.cp_context.size, self.cp_context.rank, self.compress_ratio,
                                   self.balance_mode, self.balance_block_size, self.zigzag_min_length))
        if not self.owns_key and plan != source_plan:
            raise ValueError("Full/Reindex consumers must share compression and Index K dispatch settings")
        return plan

    def forward(
            self, hidden_states: torch.Tensor, query_residual: torch.Tensor, latent: torch.Tensor | None,
            compress_position_embeddings: tuple[torch.Tensor, torch.Tensor], *,
            index_key: torch.Tensor | None = None, candidate_blocks: torch.Tensor | None = None,
            reduce_sum: Callable | None = None, query_offset: int = 0,
            minimum_key_indices: torch.Tensor | None = None, query_segments: tuple | None = None,
            document_lengths: tuple[int, ...] = (), prepare_attention: Callable | None = None,
            block_plan: Any = None, key_position_embeddings: tuple | None = None,
            causal_upper: torch.Tensor | None = None, source_key_plan: IndexKeyPlan | None = None,
            owner_index_key: torch.Tensor | None = None, prepared_batch: CPBatchPlan | None = None,
    ) -> CSA2IndexerOutput:
        """Keep owner learning separate from detached worker selection storage."""
        key_plan = self._resolve_key_plan(document_lengths, source_key_plan, prepared_batch)
        if not self.owns_key and key_plan is not None and owner_index_key is None:
            raise ValueError("Reindex prefix dispatch requires the Full layer's owner KL key")
        if self.balance_mode == "none" and key_plan is None:
            return super().forward(
                hidden_states, query_residual, latent, compress_position_embeddings,
                index_key=index_key, candidate_blocks=candidate_blocks, reduce_sum=reduce_sum,
                query_offset=query_offset, minimum_key_indices=minimum_key_indices,
                query_segments=query_segments, document_lengths=document_lengths,
                prepare_attention=prepare_attention, block_plan=block_plan,
                key_position_embeddings=key_position_embeddings, causal_upper=causal_upper,
            )
        if self.is_candidate_source or self.uses_candidates or (self.balance_mode != "none" and reduce_sum is not None):
            raise NotImplementedError("Indexer overlap requires ordinary causal selection and TP=1")
        cos, sin = compress_position_embeddings
        wave = None
        if self.balance_mode != "none":
            weights = self.weights_proj(hidden_states.detach())
            weights = weights * (self.head_dim**-0.5 * self.global_num_heads**-0.5)
            launch = self._launch_selection if self._input_region is None else self._input_region
            wave = launch(query_residual, weights, (cos, sin), document_lengths, prepared_batch)
        prepared_key = self.prepare_key(latent, cos, sin, index_key, block_plan, key_position_embeddings, key_plan)
        if prepare_attention is not None:
            prepare_attention()
        prepared = self._prepare_dispatched(wave) if wave is not None else None
        if wave is None:
            query = self._project_index_query(query_residual, cos, sin)
            weights = self.weights_proj(hidden_states.detach())
            weights = weights * (self.head_dim**-0.5 * self.global_num_heads**-0.5)
        if key_plan is not None and self.owns_key:
            key, owner_index_key = prepared_key.wait_with_gradient()
        else:
            key = self.exchange_key(prepared_key)
        if wave is None:
            selected, candidates = self._run_selection(
                query, key, weights, candidate_blocks, query_offset, reduce_sum,
                minimum_key_indices, query_segments, causal_upper, key_plan=key_plan,
            )
        else:
            select = self._select_dispatched if self._result_region is None else self._result_region
            result = select(prepared, key, key_plan)
            del prepared
            query = self._project_index_query(query_residual, cos, sin)
            finish = self._finish_selection if self._wait_region is None else self._wait_region
            selected, candidates = finish(result), None
        if weights.shape[-1] != query.shape[2]:
            raise ValueError("Indexer query and merge weights must have matching head counts")
        return CSA2IndexerOutput(selected, key, candidates, query, weights, key_plan, owner_index_key,
                                 0 if key_plan is None else key_plan.owner_range[0])

    def _select_indices(
            self, query: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor,
            candidate_blocks: torch.Tensor | None, query_offset: int,
            reduce_sum: Callable | None, minimum_key_indices: torch.Tensor | None,
            query_segments: tuple | None, causal_upper: torch.Tensor | None = None,
            *, key_plan: IndexKeyPlan | None = None,
    ) -> tuple:
        """Translate compact bank addresses only inside the CP selection stage."""
        if key_plan is None:
            return super()._select_indices(query, key, merge_weight, candidate_blocks, query_offset,
                                           reduce_sum, minimum_key_indices, query_segments, causal_upper)
        selected = compressed_causal_topk(
            query, key, merge_weight, compress_ratio=self.compress_ratio, sparse_count=self.index_topk,
            query_offset=query_offset, query_chunk_size=self.query_chunk_size, reduce_sum=reduce_sum,
            query_segments=key_plan.map_segments(query_segments), use_fused=self.use_fused,
            paged_key=self.paged_key, global_key_length=key_plan.global_length,
            key_global_positions=None if self.use_fused else key_plan.global_positions(key.device),
        )
        if self.use_fused:
            selected = key_plan.global_indices(selected, query_segments)
        return selected, None

    def enable_selection_exclude(self) -> None:
        """Reuse input/selection/result transfers instead of waiting on consumed handles."""
        super().enable_selection_exclude()
        object.__setattr__(self, "_input_region", checkpoint_exclude_wrapper(self._launch_selection))
        object.__setattr__(self, "_result_region", checkpoint_exclude_wrapper(self._select_dispatched))
        object.__setattr__(self, "_wait_region", checkpoint_exclude_wrapper(self._finish_selection))

    @torch.no_grad()
    def _launch_selection(self, qr: torch.Tensor, weights: torch.Tensor, embeddings: tuple, lengths: tuple,
                          prepared_batch: CPBatchPlan | None = None) -> Any:
        """Use the existing task planner and dtype-preserving input transport."""
        return launch_indexer_inputs(qr, weights, embeddings, lengths, self.compress_ratio, self.balance_mode,
                                     self.cp_context.mesh.get_group(), self.balance_block_size, self.zigzag_min_length,
                                     prepared_route=(None if prepared_batch is None else
                                                     prepared_batch.for_ratio(self.compress_ratio).worker_route))

    @torch.no_grad()
    def _prepare_dispatched(self, wave: Any) -> tuple | None:
        """Expand received qr once; cached selection needs no worker Q during replay."""
        if is_recomputing() and self._result_region is not None:
            # The result and final IDs are checkpoint exclusions. Do not retain expanded
            # worker Q or consume an already finished input receipt during recomputation.
            return None
        qr, weights, cos, sin = wave.wait()
        query = self._project_worker_index_query(qr, cos, sin, wave.owner_queries)
        return query, weights, wave

    @torch.no_grad()
    def _select_dispatched(self, prepared: tuple, key: torch.Tensor, key_plan: IndexKeyPlan | None = None) -> Any:
        """Consume worker Q once and launch the inverse ID transfer."""
        query, weights, wave = prepared
        selected = self._select_indices(query, key, weights, None, 0, None, None, wave.route.segments,
                                        key_plan=key_plan)[0]
        return wave.launch_result(selected[0])

    @staticmethod
    @torch.no_grad()
    def _finish_selection(result: Any) -> torch.Tensor:
        """Cache completed IDs; replay never consumes an already finished handle."""
        return result.wait().unsqueeze(0)


class _TPCompressedSparseAttention2(CompressedSparseAttention2Module):
    """Only internal head SUM; framework boundaries own input/output TP and SP."""

    _parallel_bound = False
    _head_sum = None

    def bind_parallel(self, binding: Any) -> None:
        """Borrow the existing TP group without installing another TP executor."""
        if self._parallel_bound:
            raise ValueError("CSA2 parallel resources are already bound")
        if getattr(binding.mesh_binding.mesh_context, "tp_size", 1) > 1:
            mesh = binding.mesh_binding.axis_mesh("tp")
            self._head_sum = partial(_reduce_scores, group=mesh.get_group())
        self._parallel_bound = True

    def head_reducer(self) -> Callable | None:
        """Provide head SUM before TopK and KL normalization."""
        return self._head_sum

    def query_partition(self, sequence_length: int, position_ids: torch.Tensor) -> tuple[int, int]:
        """Require binding for a selected parallel member before any model computation."""
        if not self._parallel_bound:
            raise RuntimeError("parallel CSA2 must be bound through the family mesh before forward")
        return super().query_partition(sequence_length, position_ids)


class CPCompressedSparseAttention2(_TPCompressedSparseAttention2):
    """Own CP preparation/exchange stages while inheriting the complete local forward."""

    cp_context = None

    def __init__(self, module: torch.nn.Module | None = None, *, config: Any = None, layer_idx: int = 0,
                 replace_indexer: bool = True, use_fused_ops: bool = False,
                 indexer_load_balance: str = "none", main_kv_comm: str = "all_gather",
                 indexer_balance_block_size: int | None = None, indexer_zigzag_min_length: int | None = None,
                 index_k_comm: str = "all_gather", indexer_paged_k: bool = False) -> None:
        """Adopt mathematical children and choose CP stages without creating parameters."""
        indexer_balance_block_size = resolve_indexer_balance_block_size(
            indexer_load_balance, indexer_balance_block_size,
        )
        validate_indexer_hybrid_threshold(indexer_load_balance, indexer_zigzag_min_length)
        if main_kv_comm not in ("all_gather", "all_to_all_v"):
            raise ValueError("main_kv_comm must be 'all_gather' or 'all_to_all_v'")
        if index_k_comm not in ("all_gather", "all_to_all_v"):
            raise ValueError("index_k_comm must be all_gather or all_to_all_v")
        super().__init__(module, config=config, layer_idx=layer_idx,
                         replace_indexer=replace_indexer, use_fused_ops=use_fused_ops, indexer_paged_k=indexer_paged_k)
        self.indexer_load_balance = indexer_load_balance
        self.main_kv_comm = main_kv_comm
        self.index_k_comm = index_k_comm
        self.indexer_balance_block_size = indexer_balance_block_size
        self.indexer_zigzag_min_length = indexer_zigzag_min_length
        if self.is_index_source:
            self.indexer = _CPIndexer(self.indexer, apply_rope=_apply_v41_rope)
            self.indexer.balance_mode = indexer_load_balance
            self.indexer.index_k_comm = index_k_comm
            self.indexer.balance_block_size = indexer_balance_block_size
            self.indexer.zigzag_min_length = indexer_zigzag_min_length
        if self.is_kv_source and isinstance(getattr(self, "compressor", None), CSA2Compressor):
            self.compressor = _CPCompressor.from_module(self.compressor)

    def bind_parallel(self, binding: Any) -> None:
        """Bind all stages to the framework-owned CP context and optional TP group."""
        if (binding.cp_context is None or binding.cp_context.size <= 1
                or not 0 <= binding.cp_context.rank < binding.cp_context.size):
            raise ValueError("CP CSA2 requires an existing CP context with size > 1 and 0 <= rank < size")
        super().bind_parallel(binding)
        self.cp_context = binding.cp_context
        if self.is_index_source:
            self.indexer.cp_context = self.cp_context
        if isinstance(getattr(self, "compressor", None), _CPCompressor):
            self.compressor.cp_context = self.cp_context

    def query_partition(self, sequence_length: int, position_ids: torch.Tensor) -> tuple[int, int]:
        """Derive packed row addresses independently of RoPE position IDs."""
        del position_ids  # Position values are already encoded in the supplied RoPE embeddings.
        if self.cp_context is None:
            raise RuntimeError("CP CSA2 must be bound through the family mesh before forward")
        offset = self.cp_context.rank * sequence_length
        return offset, sequence_length * self.cp_context.size

    def attach_batch_plan(self, document: CSA2DocumentInputs, plan: Any) -> CSA2DocumentInputs:
        """Bind an immutable batch snapshot to this call, never to module state."""
        if plan is None:
            return document
        if not isinstance(plan, CPBatchPlan):
            raise TypeError("cp_batch_plan must be CPBatchPlan")
        boundaries = ((0, document.global_length) if document.packed is None
                      else document.packed.document_boundaries())
        plan.validate(tuple(end - start for start, end in zip(boundaries, boundaries[1:])),
                      self.cp_context.size, self.cp_context.rank)
        config = plan.config
        if (config.mode, config.block_size, config.zigzag_min_length, config.main_kv_comm, config.index_k_comm) != (
                self.indexer_load_balance, self.indexer_balance_block_size, self.indexer_zigzag_min_length,
                self.main_kv_comm, self.index_k_comm):
            raise ValueError("Prepared CP batch configuration differs from the active communication context")
        plan.for_ratio(self.compress_ratio)
        plan.raw_route(self.sliding_window)
        return replace(document, batch_plan=plan)

    def indexer_runtime_options(self, document: CSA2DocumentInputs,
                                shared_state: SharedCompressedAttentionState) -> dict:
        """Pass bank ownership and per-call plans through the managed Indexer boundary."""
        return {"source_key_plan": shared_state.index_key_plans.get(self.kv_source_layer_idx),
                "owner_index_key": shared_state.owner_index_keys.get(self.kv_source_layer_idx),
                "prepared_batch": document.batch_plan}

    def prepare_compression(self, document: CSA2DocumentInputs, embeddings: dict,
                            device: torch.device) -> tuple:
        """Plan owner blocks and explicitly complete block-start position metadata."""
        if not self.is_kv_source or self.compress_ratio <= 1:
            return None, None
        size, rank = self.cp_context.size, self.cp_context.rank
        local_length = document.global_length // size
        packed = document.packed
        boundaries = (0, document.global_length) if packed is None else packed.document_boundaries()
        if local_length % self.compress_ratio == 0 and all(value % self.compress_ratio == 0 for value in boundaries):
            return None, None
        if packed is None:
            packed = SharedCompressedPackedSequence(torch.tensor(boundaries), rank * local_length,
                                                     local_length, document.global_length)
        packed = packed.prepare(device, (self.compress_ratio,))
        plan = (document_compression_plan(boundaries, self.compress_ratio, local_length, size, rank)
                if document.batch_plan is None else document.batch_plan.for_ratio(self.compress_ratio).compression)
        source_positions = embeddings["compress"]
        positions = packed.cached_compressed_positions(plan.ratio, source_positions)
        if positions is None:
            exchanged = _exchange_boundary(torch.cat(source_positions, dim=-1), plan.route, self.cp_context)
            rows = plan.indices(device)[1]
            positions = exchanged.index_select(1, rows).chunk(2, dim=-1)
            packed.cache_compressed_positions(plan.ratio, source_positions, positions)
        return plan, positions

    def launch_bank(self, tensor: torch.Tensor, sequence_dim: int) -> Any:
        """Launch a differentiable rank-ordered gather through CPContext."""
        return _launch_bank(tensor, sequence_dim, self.cp_context)

    def exchange_raw(self, tensor: torch.Tensor, sequence_dim: int, route: Any) -> Any:
        """Launch the same-document raw SWA halo."""
        return async_sequence_dispatch(tensor, sequence_dim, route, self.cp_context.mesh.get_group())

    def exchange_main(self, tensor: torch.Tensor, sequence_dim: int, route: Any) -> Any:
        """Launch the main compressed document prefix."""
        return async_sequence_dispatch(tensor, sequence_dim, route, self.cp_context.mesh.get_group())

    def _launch_compressed_bank(self, compressed: torch.Tensor, route: Any, plan: Any) -> Any:
        """Respect the compressed bank's actual owner counts and physical layout."""
        if route is not None:
            return self.exchange_main(compressed, 1, route)
        if plan is not None:
            return launch_compressed_bank(compressed, plan.counts, self.launch_bank)
        return self.launch_bank(compressed, 1)

    def prepare_attention(self, work: _MainAttentionWork) -> None:
        """Submit delayed main KV behind Index K, then compute the main query."""
        if work.launch_main_bank is not None:
            work.compressed_handle = work.launch_main_bank()
            work.launch_main_bank = None
        work.query = self._expand_query(work.query_residual, work.cos, work.sin)

    def prepare_attention_inputs(self, hidden_states: torch.Tensor, embeddings: dict,
                                 shared_state: SharedCompressedAttentionState,
                                 document: CSA2DocumentInputs) -> _PendingAttention:
        """Retain the original Full/Reindex launch order and owner/worker projection windows."""
        plan, key_positions = self.prepare_compression(document, embeddings, hidden_states.device)
        length = hidden_states.shape[1]
        context = self.cp_context
        boundaries = ((0, document.global_length) if document.packed is None
                      else document.packed.document_boundaries())
        if document.batch_plan is None:
            raw_route = sliding_window_route(boundaries, length, context.size, context.rank, self.sliding_window)
            main_route = (document_prefix_route(boundaries, length, context.size, context.rank, self.compress_ratio)
                          if self.compress_ratio and self.main_kv_comm == "all_to_all_v" else None)
        else:
            raw_route = document.batch_plan.raw_route(self.sliding_window)
            main_route = document.batch_plan.for_ratio(self.compress_ratio).main_route
        lengths = tuple(end - start for start, end in zip(boundaries, boundaries[1:]))
        cos, sin = embeddings["compress" if self.compress_ratio else "main"]
        raw = self._project_raw_kv(hidden_states, cos, sin)
        raw_handle = self.exchange_raw(raw, 2, raw_route)
        if self.is_index_source:
            qr = self.q_a_norm(self.q_a_proj(hidden_states))
            query = None
        else:
            qr, query = self._project_query(hidden_states, cos, sin)
        latent, compressed = self.compress_key_value(hidden_states, embeddings, plan, key_positions)
        launch_main = (None if compressed is None
                       else partial(self._launch_compressed_bank, compressed, main_route, plan))
        work = _MainAttentionWork(qr, cos, sin, launch_main) if self.is_index_source else None
        compressed_handle = launch_main() if work is None and launch_main is not None else None
        indexed = self._run_indexer(
            hidden_states, qr, latent, embeddings, shared_state, document, lengths,
            block_plan=plan, key_position_embeddings=key_positions,
            prepare_attention=partial(self.prepare_attention, work) if work is not None else None,
        )
        if work is not None:
            query, compressed_handle = work.query, work.compressed_handle
        return _PendingAttention(query, raw_handle, compressed_handle, indexed, raw_route.global_start,
                                  0 if main_route is None else main_route.global_start)

    def exchange_attention(self, prepared: _PendingAttention,
                           shared_state: SharedCompressedAttentionState) -> CSA2AttentionInputs:
        """Resolve banks only when the owner teacher and sparse attention can consume them."""
        raw = prepared.raw_handle.wait()
        if prepared.compressed_handle is not None:
            compressed = prepared.compressed_handle.wait()
            shared_state.publish_compressed_kv(self.layer_idx, compressed, prepared.main_offset)
        return CSA2AttentionInputs(prepared.query, raw, prepared.indexer_output, prepared.raw_offset)
