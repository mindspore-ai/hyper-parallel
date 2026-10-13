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

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedAttentionCPContext,
    SharedCompressedAttentionTPContext,
    SharedCompressedDSAAttention,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.collectives import (
    async_cp_allgather_launch,
)
from hyper_parallel.distributed.context_parallel.index_key_dispatch import IndexKeyPlan, index_key_plan
from hyper_parallel.distributed.context_parallel.indexer_balance import (
    balanced_indexer_select, resolve_indexer_balance_block_size, validate_indexer_hybrid_thresholds,
)
from hyper_parallel.distributed.context_parallel.sequence_dispatch import async_sequence_dispatch
from hyper_parallel.distributed.recipe_spec import inner_wrapper


def _build_shared_attention_cp_context(
        cp_mesh: Any, swa_comm: str = "all_to_all_v", indexer_load_balance: str = "none",
        main_kv_comm: str = "all_gather", indexer_balance_block_size: int | None = None,
        index_k_comm: str = "all_gather",
        indexer_zigzag_min_length: int | None = None, indexer_zigzag_min_chunk_size: int | None = None,
) -> SharedCompressedAttentionCPContext:
    """Build independently configured raw, main and Index K bank communication."""
    if index_k_comm not in ("all_gather", "all_to_all_v"):
        raise ValueError("index_k_comm must be 'all_gather' or 'all_to_all_v'")
    indexer_balance_block_size = resolve_indexer_balance_block_size(
        indexer_load_balance, indexer_balance_block_size,
    )

    validate_indexer_hybrid_thresholds(
        indexer_load_balance, indexer_zigzag_min_length, indexer_zigzag_min_chunk_size,
    )

    def _gather_sequence(tensor: torch.Tensor, sequence_dim: int) -> torch.Tensor:
        """All-gather one sequence dimension in CP-rank order."""
        return async_cp_allgather_launch(tensor, sequence_dim, cp_mesh).wait()

    def _launch_sequence(tensor: torch.Tensor, sequence_dim: int) -> Any:
        """Launch a differentiable KV all-gather for projection overlap."""
        return async_cp_allgather_launch(tensor, sequence_dim, cp_mesh)

    def _launch_dispatch(tensor: torch.Tensor, sequence_dim: int, route: Any) -> Any:
        """Exchange planned bank intervals on the existing CP group."""
        return async_sequence_dispatch(tensor, sequence_dim, route, cp_mesh.get_group())

    def _launch_index(tensor: torch.Tensor, sequence_dim: int, plan: IndexKeyPlan) -> Any:
        """Send the worker/owner union, but return only the owner's KL gradients."""
        return async_sequence_dispatch(
            tensor, sequence_dim, plan.dispatch, cp_mesh.get_group(), gradient=plan.gradient,
        )

    def _plan_index_key(lengths: tuple, ratio: int) -> Any:
        """Cache the union of balanced-worker and original-owner prefix requirements."""
        return index_key_plan(lengths, cp_mesh.size(), cp_mesh.get_local_rank(), ratio,
                              indexer_load_balance, indexer_balance_block_size,
                              indexer_zigzag_min_length, indexer_zigzag_min_chunk_size)

    def _balance_indexer(query_residual, weights, embeddings, selector, lengths, ratio):
        """Move only selection tasks; differentiable banks and KL remain with owners."""
        return balanced_indexer_select(
            query_residual, weights, embeddings, selector, lengths, ratio, indexer_load_balance,
            cp_mesh.get_group(), indexer_balance_block_size,
            indexer_zigzag_min_length, indexer_zigzag_min_chunk_size,
        )

    return SharedCompressedAttentionCPContext(
        size=cp_mesh.size(),
        rank=cp_mesh.get_local_rank(),
        gather_sequence=_gather_sequence,
        launch_sequence=_launch_sequence,
        launch_boundary=_launch_dispatch,
        launch_raw=_launch_dispatch if swa_comm == "all_to_all_v" else None,
        launch_main=_launch_dispatch if main_kv_comm == "all_to_all_v" else None,
        launch_index=_launch_index if index_k_comm == "all_to_all_v" else None,
        plan_index_key=_plan_index_key if index_k_comm == "all_to_all_v" else None,
        balance_indexer=_balance_indexer if indexer_load_balance != "none" else None,
    )


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

    Raw KV uses SWA halo all-to-all-v, with full all-gather as a comparison.
    Main KV and Index K independently select document-prefix dispatch or all-gather.
    Launch/wait separation permits independent projection work. Indexer Q/merge heads remain TP-local and
    reduce their score and KL-target contributions before global Top-K.
    """
    del mesh, ep_mesh
    wrapper_name = "deepseek_v41_shared_attention_parallel_wrapper"
    if not isinstance(target_module, SharedCompressedDSAAttention):
        raise TypeError(
            f"{wrapper_name} requires SharedCompressedDSAAttention after "
            f"module replacement, got {type(target_module).__name__}"
        )

    original_forward = target_module.forward
    cp_context = (
        _build_shared_attention_cp_context(
            cp_mesh, target_module.swa_comm, target_module.indexer_load_balance, target_module.main_kv_comm,
            target_module.indexer_balance_block_size, target_module.index_k_comm,
            target_module.indexer_zigzag_min_length, target_module.indexer_zigzag_min_chunk_size,
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
        if cp_context is not None:
            call_kwargs["shared_attention_cp_context"] = cp_context
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
