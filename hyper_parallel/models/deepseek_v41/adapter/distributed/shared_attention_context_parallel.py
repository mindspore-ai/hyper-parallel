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
"""Build TP/CP contexts for DeepSeek-V4.1 shared attention."""

from __future__ import annotations

from functools import wraps
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.utils.communication import differentiable_all_to_all_single
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedAttentionCPContext,
    SharedCompressedAttentionTPContext,
    SharedCompressedDSAAttentionBase,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.collectives import (
    async_cp_allgather_launch,
)
from hyper_parallel.distributed.context_parallel.compressed_document import CompressionBoundaryRoute
from hyper_parallel.distributed.context_parallel.index_key_dispatch import index_key_plan
from hyper_parallel.distributed.context_parallel.batch_plan import CPBatchPlan
from hyper_parallel.distributed.context_parallel.indexer_balance import (
    balanced_indexer_select, launch_indexer_inputs, resolve_indexer_balance_block_size,
    validate_indexer_hybrid_threshold,
)
from hyper_parallel.distributed.context_parallel.sequence_dispatch import async_sequence_dispatch
from hyper_parallel.distributed.recipe_spec import inner_wrapper


def _build_shared_attention_cp_context(
        cp_mesh: Any, indexer_load_balance: str = "none",
        main_kv_comm: str = "all_gather", indexer_balance_block_size: int | None = None,
        indexer_zigzag_min_length: int | None = None,
        index_k_comm: str = "all_gather",
        prepared_batch: CPBatchPlan | None = None,
) -> SharedCompressedAttentionCPContext:
    """Build independent raw/main/Index K transport with owner-side gradients."""
    if main_kv_comm not in ("all_gather", "all_to_all_v"):
        raise ValueError("main_kv_comm must be 'all_gather' or 'all_to_all_v'")
    if index_k_comm not in ("all_gather", "all_to_all_v"):
        raise ValueError("index_k_comm must be all_gather or all_to_all_v")
    indexer_balance_block_size = resolve_indexer_balance_block_size(
        indexer_load_balance, indexer_balance_block_size,
    )

    validate_indexer_hybrid_threshold(
        indexer_load_balance, indexer_zigzag_min_length,
    )

    if prepared_batch is not None:
        config = prepared_batch.config
        expected = (cp_mesh.size(), cp_mesh.get_local_rank(), indexer_load_balance,
                    indexer_balance_block_size, indexer_zigzag_min_length, main_kv_comm, index_k_comm)
        actual = (config.size, config.rank, config.mode, config.block_size,
                  config.zigzag_min_length, config.main_kv_comm, config.index_k_comm)
        if actual != expected:
            raise ValueError("Prepared CP batch configuration differs from the active communication context")

    def _gather_sequence(tensor: torch.Tensor, sequence_dim: int) -> torch.Tensor:
        """All-gather one sequence dimension in CP-rank order."""
        gather_handle = async_cp_allgather_launch(tensor, sequence_dim, cp_mesh)
        gathered_sequence = gather_handle.wait()
        return gathered_sequence

    def _launch_sequence(tensor: torch.Tensor, sequence_dim: int) -> Any:
        """Launch a differentiable KV all-gather for projection overlap."""
        gather_handle = async_cp_allgather_launch(tensor, sequence_dim, cp_mesh)
        return gather_handle

    def _exchange_boundary(tensor: torch.Tensor, route: CompressionBoundaryRoute) -> torch.Tensor:
        """Complete compressor groups; autograd sends boundary gradients to their owners."""
        local = tensor.movedim(1, 0).contiguous()
        parts = [local.narrow(0, offset, count)
                 for offset, count in zip(route.send_offsets, route.input_splits) if count]
        send = torch.cat(parts, dim=0) if parts else local[:0]
        received = differentiable_all_to_all_single(
            send, list(route.input_splits), list(route.output_splits), group=cp_mesh.get_group(),
        )
        # Even empty receives must remain connected so every rank participates in backward.
        return torch.cat((received, local), dim=0).movedim(0, 1).contiguous()

    def _launch_dispatch(tensor: torch.Tensor, sequence_dim: int, route: Any) -> Any:
        """Exchange planned bank intervals on the existing CP group."""
        return async_sequence_dispatch(tensor, sequence_dim, route, cp_mesh.get_group())

    def _plan_index_key(lengths: tuple, ratio: int) -> Any:
        """Cache the union of worker selection and original-owner learning demand."""
        if prepared_batch is not None:
            prepared_batch.validate(lengths, cp_mesh.size(), cp_mesh.get_local_rank())
            return prepared_batch.for_ratio(ratio).index_key
        return index_key_plan(lengths, cp_mesh.size(), cp_mesh.get_local_rank(), ratio,
                              indexer_load_balance, indexer_balance_block_size, indexer_zigzag_min_length)

    def _launch_index(tensor: torch.Tensor, sequence_dim: int, plan: Any) -> Any:
        """Transmit each requested prefix once; reverse only differentiable owner rows."""
        return async_sequence_dispatch(tensor, sequence_dim, plan.dispatch, cp_mesh.get_group(),
                                       gradient=plan.gradient)

    def _balance_indexer(query_residual, weights, embeddings, selector, lengths, ratio):
        """Move only selection tasks; differentiable banks and KL remain with owners."""
        return balanced_indexer_select(
            query_residual, weights, embeddings, selector, lengths, ratio, indexer_load_balance,
            cp_mesh.get_group(), indexer_balance_block_size,
            indexer_zigzag_min_length,
        )

    def _launch_balance(query_residual, weights, embeddings, lengths, ratio):
        """Launch no-grad selection inputs before independent owner projections."""
        route = None
        if prepared_batch is not None:
            prepared_batch.validate(lengths, cp_mesh.size(), cp_mesh.get_local_rank())
            route = prepared_batch.for_ratio(ratio).worker_route
        return launch_indexer_inputs(
            query_residual, weights, embeddings, lengths, ratio, indexer_load_balance,
            cp_mesh.get_group(), indexer_balance_block_size,
            indexer_zigzag_min_length, prepared_route=route,
        )

    cp_context = SharedCompressedAttentionCPContext(
        size=cp_mesh.size(),
        rank=cp_mesh.get_local_rank(),
        gather_sequence=_gather_sequence,
        launch_sequence=_launch_sequence,
        exchange_boundary=_exchange_boundary,
        launch_raw=_launch_dispatch,
        launch_main=_launch_dispatch if main_kv_comm == "all_to_all_v" else None,
        balance_indexer=_balance_indexer if indexer_load_balance != "none" else None,
        launch_balance=_launch_balance if indexer_load_balance != "none" else None,
        indexer_priority=True,
        plan_index_key=_plan_index_key if index_k_comm == "all_to_all_v" else None,
        launch_index=_launch_index if index_k_comm == "all_to_all_v" else None,
    )
    return cp_context


def _build_shared_attention_tp_context(tp_mesh: Any) -> SharedCompressedAttentionTPContext:
    """Build PanGu-style Indexer head-score reduction context."""
    group = tp_mesh.get_group()

    def _reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
        """Sum local Indexer-head contributions across the TP group."""
        reduced = tensor.contiguous()
        torch.distributed.all_reduce(reduced, group=group)
        return reduced

    return SharedCompressedAttentionTPContext(
        size=tp_mesh.size(),
        rank=tp_mesh.get_local_rank(),
        reduce_sum=_reduce_sum,
    )


@inner_wrapper
def deepseek_v41_shared_attention_parallel_wrapper(
        target_module: nn.Module,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Install V4.1's explicit TP Indexer and Colossal CP collectives.

    Raw KV uses SWA halo all-to-all-v.
    Main KV and Index K independently select prefix dispatch or all-gather.
    Launch/wait separation permits independent projection work. Indexer Q/merge heads remain TP-local and
    reduce their score and KL-target contributions before global Top-K.
    Packed compression grouping is owned by the shared module; this adapter
    supplies communication callbacks without changing data boundaries.
    Native and optimized attention share this context interface.
    """
    del mesh, ep_mesh
    wrapper_name = "deepseek_v41_shared_attention_parallel_wrapper"
    if not isinstance(target_module, SharedCompressedDSAAttentionBase):
        raise TypeError(
            f"{wrapper_name} requires SharedCompressedDSAAttentionBase, "
            f"got {type(target_module).__name__}"
        )

    original_forward = target_module.forward
    cp_context = (
        _build_shared_attention_cp_context(
            cp_mesh, target_module.indexer_load_balance, target_module.main_kv_comm,
            target_module.indexer_balance_block_size,
            target_module.indexer_zigzag_min_length,
            index_k_comm=target_module.index_k_comm,
        )
        if cp_mesh is not None and cp_mesh.size() > 1
        else None
    )
    tp_context = (
        _build_shared_attention_tp_context(tp_mesh)
        if tp_mesh is not None and tp_mesh.size() > 1
        else None
    )

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Run local queries against the configured V4.1 bank layouts."""
        if "shared_attention_cp_context" in kwargs or "shared_attention_tp_context" in kwargs:
            raise ValueError("shared attention parallel contexts are owned by the wrapper")
        call_kwargs = dict(kwargs)
        prepared_batch = call_kwargs.pop("cp_batch_plan", None)
        active_context = cp_context
        if prepared_batch is not None:
            if cp_context is None or not isinstance(prepared_batch, CPBatchPlan):
                raise ValueError("cp_batch_plan requires a prepared CP batch and an active CP mesh")
            active_context = _build_shared_attention_cp_context(
                cp_mesh, target_module.indexer_load_balance, target_module.main_kv_comm,
                target_module.indexer_balance_block_size, target_module.indexer_zigzag_min_length,
                index_k_comm=target_module.index_k_comm, prepared_batch=prepared_batch,
            )
        if active_context is not None:
            call_kwargs["shared_attention_cp_context"] = active_context
        if tp_context is not None:
            call_kwargs["shared_attention_tp_context"] = tp_context
        return original_forward(*args, **call_kwargs)

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={
            "_hyper_v41_shared_attention_cp_context": cp_context,
            "_hyper_v41_shared_attention_tp_context": tp_context,
        },
    )


deepseek_v41_shared_attention_cp_wrapper = deepseek_v41_shared_attention_parallel_wrapper


__all__ = [
    "deepseek_v41_shared_attention_cp_wrapper",
    "deepseek_v41_shared_attention_parallel_wrapper",
]
