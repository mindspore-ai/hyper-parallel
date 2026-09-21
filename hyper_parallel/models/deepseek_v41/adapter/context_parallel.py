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
"""TP/CP adapter for DeepSeek-V4.1 shared compressed attention."""

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
from hyper_parallel.distributed.recipe_spec import inner_wrapper


def _build_shared_attention_cp_context(cp_mesh: Any) -> SharedCompressedAttentionCPContext:
    """Build V4.1's asynchronous KV-all-gather communication context."""

    def _gather_sequence(tensor: torch.Tensor, sequence_dim: int) -> torch.Tensor:
        """All-gather one sequence dimension in CP-rank order."""
        return async_cp_allgather_launch(tensor, sequence_dim, cp_mesh).wait()

    def _launch_sequence(tensor: torch.Tensor, sequence_dim: int) -> Any:
        """Launch a differentiable KV all-gather for projection overlap."""
        return async_cp_allgather_launch(tensor, sequence_dim, cp_mesh)

    return SharedCompressedAttentionCPContext(
        size=cp_mesh.size(),
        rank=cp_mesh.get_local_rank(),
        gather_sequence=_gather_sequence,
        launch_sequence=_launch_sequence,
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

    Queries and Top-K indices remain sequence-local; raw KV, compressed KV,
    and index keys use asynchronous differentiable all-gather. Projection
    GEMMs overlap communication. Indexer Q/merge heads remain TP-local and
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
        _build_shared_attention_cp_context(cp_mesh)
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
        """Run local queries against CP-replicated V4.1 key/value states."""
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
