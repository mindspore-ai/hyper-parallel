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
"""Planner-injected Context Parallel wrappers for Kimi Delta Attention."""
# pylint: disable=forbidden-backend-import
from __future__ import annotations

__all__ = [
    "kimi_delta_attention_cp_wrapper",
    "kimi_delta_attention_p2p_cp_wrapper",
    "kimi_delta_attention_ulysses_cp_wrapper",
]

from functools import wraps
from typing import Any, Type

from torch import nn

from hyper_parallel.distributed import inner_wrapper
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.kimi_delta_attention import (
    KimiDeltaAttentionLayerP2PCP,
    KimiDeltaAttentionLayerUlyssesCP,
)
from hyper_parallel.distributed.context_parallel.kimi_delta_attention_hybrid import (
    KimiDeltaAttentionLayerHybridCP,
)


def _build_kda_cp_rewrite(
    target_module: nn.Module,
    cp_mesh: Any,
    tp_mesh: Any,
    *,
    mode: str,
    executor_class: Type[nn.Module],
    backend: str,
    chunk_size: int,
    execution_options: dict[str, Any] | None = None,
) -> _ForwardRewriteRequest:
    """Validate one KDA target and return an atomic forward rewrite request."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"KDA {mode} Context Parallel requires an active CP mesh")
    if tp_mesh is not None and tp_mesh.size() > 1:
        raise NotImplementedError(
            "KDA Context Parallel does not yet support simultaneous TP and CP"
        )
    if getattr(target_module, "_hp_kda_cp_config", None) is not None:
        raise RuntimeError(
            "KDA Context Parallel has already been applied to this module; "
            "applying a CP wrapper twice is not supported"
        )

    execution_options = {} if execution_options is None else execution_options
    executor = executor_class(
        target_module,
        cp_mesh,
        chunk_size=chunk_size,
        backend=backend,
        **execution_options,
    )
    original_forward = target_module.forward

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Execute the selected KDA Context Parallel algorithm.

        Args:
            kwargs: Additional model arguments; packed sequences are unsupported.
        """
        return executor(*args, **kwargs)

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={
            "_hp_kda_cp_config": {
                "mode": mode,
                "backend": backend,
                "chunk_size": chunk_size,
                **execution_options,
            },
        },
    )


@inner_wrapper
def kimi_delta_attention_ulysses_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "eager",
    chunk_size: int = 64,
) -> _ForwardRewriteRequest:
    """Install full-layer Ulysses CP on a Kimi Delta Attention module.

    Args:
        target_module: KDA layer whose parameters remain owned by the model.
        mesh: Planner mesh context.
        tp_mesh: Tensor-parallel mesh; simultaneous TP/CP is unsupported.
        cp_mesh: Chronologically ordered context-parallel mesh.
        ep_mesh: Expert-parallel context, unused by KDA.
        backend: Local KDA implementation: eager or triton.
        chunk_size: Tokens per local recurrence chunk.
    """
    del mesh, ep_mesh
    return _build_kda_cp_rewrite(
        target_module,
        cp_mesh,
        tp_mesh,
        mode="ulysses",
        executor_class=KimiDeltaAttentionLayerUlyssesCP,
        backend=backend,
        chunk_size=chunk_size,
    )


@inner_wrapper
def kimi_delta_attention_p2p_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "eager",
    chunk_size: int = 64,
) -> _ForwardRewriteRequest:
    """Install full-layer recurrent-state P2P CP on a KDA module.

    Args:
        target_module: KDA layer whose parameters remain owned by the model.
        mesh: Planner mesh context.
        tp_mesh: Tensor-parallel mesh; simultaneous TP/CP is unsupported.
        cp_mesh: Chronologically ordered context-parallel mesh.
        ep_mesh: Expert-parallel context, unused by KDA.
        backend: Local KDA implementation: eager or triton.
        chunk_size: Tokens per local recurrence chunk.
    """
    del mesh, ep_mesh
    return _build_kda_cp_rewrite(
        target_module,
        cp_mesh,
        tp_mesh,
        mode="p2p",
        executor_class=KimiDeltaAttentionLayerP2PCP,
        backend=backend,
        chunk_size=chunk_size,
    )


@inner_wrapper
def kimi_delta_attention_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "triton",
    chunk_size: int = 64,
    boundary_protocol: str = "p2p",
    ulysses_degree: int = 1,
    group_size: int = 1,
) -> _ForwardRewriteRequest:
    """Install KDA state CP, optionally combined with Ulysses and group-local AG.

    Args:
        target_module: Training-time KDA layer, retaining its parameter ownership.
        mesh: Planner mesh context.
        tp_mesh: Simultaneous TP and CP is currently unsupported.
        cp_mesh: Chronological CP mesh, retaining its root for hybrid splitting.
        ep_mesh: Unused expert-parallel context.
        backend: Local backend; gathers require triton.
        chunk_size: Tokens per local chunk, currently 64 for triton.
        boundary_protocol: p2p, allgather, or grouped_allgather_p2p.
        ulysses_degree: Consecutive ranks exchanging token/head shards.
        group_size: AG width along state CP, only for grouped_allgather_p2p.

    Returns:
        Atomic forward rewrite with the selected configuration recorded.
    """
    del mesh, ep_mesh
    if not isinstance(ulysses_degree, int) or isinstance(ulysses_degree, bool) or ulysses_degree < 1:
        raise ValueError("ulysses_degree must be a positive integer.")
    options = {"boundary_protocol": boundary_protocol, "group_size": group_size}
    executor_class = KimiDeltaAttentionLayerP2PCP
    if ulysses_degree > 1:
        executor_class = KimiDeltaAttentionLayerHybridCP
        options["ulysses_degree"] = ulysses_degree
    return _build_kda_cp_rewrite(
        target_module, cp_mesh, tp_mesh, mode="hybrid" if ulysses_degree > 1 else "state",
        executor_class=executor_class, backend=backend, chunk_size=chunk_size,
        execution_options=options,
    )
