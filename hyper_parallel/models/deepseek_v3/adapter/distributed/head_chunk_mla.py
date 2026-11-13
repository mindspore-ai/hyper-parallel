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
"""Bind the CP mesh to the opt-in head-chunk MLA replacement."""

from typing import Any

from torch import nn

from hyper_parallel.components.modules import HeadChunkMLA
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.recipe_spec import inner_wrapper


@inner_wrapper
def head_chunk_mla_cp_wrapper(
    target_module: nn.Module, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> list[_ForwardRewriteRequest]:
    """Bind the CP mesh to staged MLA without installing a second MLA CP runtime.

    Args:
        target_module: The already replaced HeadChunkMLA module.
        mesh: Unused parent mesh supplied by the recipe executor.
        tp_mesh: Must be absent or degree one.
        cp_mesh: Contiguous-sequence Ulysses mesh.
        ep_mesh: Unused expert mesh; attention weights remain decoder-owned.

    Returns:
        One atomic request retaining the module's own staged forward.
    """
    del mesh, ep_mesh
    if not isinstance(target_module, HeadChunkMLA):
        raise ValueError("head_chunk_mla_cp_wrapper requires the staged MLA replacement")
    if tp_mesh is not None and tp_mesh.size() != 1:
        raise ValueError("Head-chunk MLA does not support TP")
    if cp_mesh is not None and cp_mesh.ndim != 1:
        raise ValueError("Head-chunk MLA requires a one-dimensional CP mesh")
    return [_ForwardRewriteRequest(target_module, target_module.forward,
                                   companion_attrs={"head_chunk_cp_mesh": cp_mesh})]
