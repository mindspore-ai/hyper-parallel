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
"""Context-parallel mesh injection for packed Head Chunk GQA."""

from functools import wraps
from typing import Any

from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper


@inner_wrapper
def head_chunk_ulysses_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Attach the framework-owned CP mesh to a Head Chunk GQA module.

    Args:
        target_module: Structurally compatible ``HeadChunkGQAAttention``.
        mesh: Full model mesh supplied by the recipe builder.
        tp_mesh: Tensor-parallel mesh, unsupported by version one.
        cp_mesh: Context-parallel mesh used by ordered Ulysses stages.
        ep_mesh: Expert-parallel mesh, unused by attention.

    Returns:
        Atomic rewrite request carrying the CP mesh companion attribute.

    Raises:
        ValueError: If CP/head geometry cannot execute the same collective
            order on every rank.
    """
    del mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("head_chunk_ulysses_cp_wrapper requires an active CP mesh")
    if tp_mesh is not None and tp_mesh.size() > 1:
        raise ValueError("Head Chunk GQA version one does not support TP")
    required = (
        "head_chunk_size",
        "kv_chunk_size",
        "expand_kv_heads",
        "head_chunk_cp_mesh",
    )
    missing = [name for name in required if not hasattr(target_module, name)]
    if missing:
        raise ValueError(
            "head_chunk_ulysses_cp_wrapper requires HeadChunkGQAAttention; "
            f"missing attributes {missing}"
        )
    cp_size = cp_mesh.size()
    if target_module.head_chunk_size % cp_size:
        raise ValueError("Head Chunk query-head stage must be divisible by CP size")
    if not target_module.expand_kv_heads and target_module.kv_chunk_size % cp_size:
        raise ValueError(
            "Head Chunk KV stage must be divisible by CP size unless "
            "expand_kv_heads=true"
        )

    original_forward = target_module.forward

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Run the original packed head-stage signature with CP attached."""
        return original_forward(*args, **kwargs)

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={"head_chunk_cp_mesh": cp_mesh},
    )


__all__ = ["head_chunk_ulysses_cp_wrapper"]
