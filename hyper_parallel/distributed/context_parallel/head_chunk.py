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
"""Explicit framework-owned CP mesh injection for shared-KV Head Chunk GQA."""

from functools import wraps
from typing import Any

from hyper_parallel.components.modules._head_chunk_shared_kv import shared_kv_head_schedule
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.recipe_spec import inner_wrapper


@inner_wrapper
def head_chunk_ulysses_cp_wrapper(
    target_module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Attach an active one-dimensional Ulysses mesh to Head Chunk GQA.

    Args:
        target_module: Shared-KV HeadChunkGQAAttention module.
        mesh: Full model mesh supplied by the framework, unused here.
        tp_mesh: Tensor-parallel mesh; active TP is unsupported.
        cp_mesh: Framework-owned CP DeviceMesh; no process group is created.
        ep_mesh: Expert-parallel mesh, unused by attention.

    Returns:
        Atomic rewrite request carrying the CP mesh companion attribute.

    Raises:
        ValueError: If TP is active, CP geometry is invalid or the target is
            not a shared-KV Head Chunk module.
    """
    del mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1 or getattr(cp_mesh, "ndim", 1) != 1:
        raise ValueError("head_chunk_ulysses_cp_wrapper requires an active one-dimensional CP mesh")
    if tp_mesh is not None and tp_mesh.size() > 1:
        raise ValueError("Head Chunk GQA does not support tensor parallelism")
    required = ("head_chunk_size", "head_chunk_cp_mesh", "num_heads", "num_key_value_heads", "_shared_kv_plans")
    if any(not hasattr(target_module, name) for name in required):
        raise ValueError("head_chunk_ulysses_cp_wrapper requires shared-KV HeadChunkGQAAttention")
    shared_kv_head_schedule(
        target_module.num_heads, target_module.num_key_value_heads, target_module.head_chunk_size, cp_mesh.size(),
    )
    original_forward = target_module.forward

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Preserve the packed Head Chunk signature with the CP mesh attached."""
        return original_forward(*args, **kwargs)

    return _ForwardRewriteRequest(target_module, cp_forward, companion_attrs={"head_chunk_cp_mesh": cp_mesh})


__all__ = ["head_chunk_ulysses_cp_wrapper"]
