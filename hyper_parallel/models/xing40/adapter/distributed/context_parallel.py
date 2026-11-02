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
"""Xing40 context-parallel adapters. Opt-in; no effect on non-CP attention or router."""
from __future__ import annotations

from typing import Any, Literal
# Native Torch is required by this model adapter; the legacy backend-import rule predates that migration.
import torch  # pylint: disable=forbidden-backend-import
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from hyper_parallel.distributed.context_parallel.mla_context_parallel import MLACPStrategy
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel import mla_cp_wrapper
from hyper_parallel.models.xing40.adapter.conversion.module_replacement import Xing40MLAAttention

def xing40_rotary(tensor: torch.Tensor, positions: tuple[torch.Tensor, torch.Tensor] | None, *,
                   interleaved: bool = False) -> torch.Tensor:
    """Apply HF interleaved RoPE in FP32 on BSND and round once.

    Args:
        tensor: Local batch, sequence, head, rotary-width tensor.
        positions: Local cosine and sine embeddings.
        interleaved: Must be true for the Xing40 interleaved layout.

    Returns:
        Rotated tensor in the input dtype.
    """
    if positions is None:
        raise ValueError("Xing40 CP requires RoPE position embeddings")
    if not interleaved:
        raise ValueError('Xing40 CP requires interleaved RoPE')
    cos, sin = positions
    rotated, _ = apply_rotary_pos_emb_interleave(tensor.float(), tensor.float(), cos, sin, unsqueeze_dim=2)
    return rotated.to(tensor.dtype)

@inner_wrapper
def xing40_mla_cp_wrapper(target_module: torch.nn.Module, mesh: Any, tp_mesh: Any,
                            cp_mesh: Any, ep_mesh: Any, *, strategy: MLACPStrategy = 'expanded_ulysses',
                            overlap: bool = False, ulysses_degree: int | None = 1,
                            kv_exchange: Literal['all_gather', 'all_to_all_v'] = 'all_gather') -> list:
    """Install the framework MLA CP runtime with the Xing40 FP32 RoPE.

    The runtime calls the module's own attention, which records the logit maxima of QK clipping.

    The default is the KV AllGather schedule, which the framework plan restricts to TP1; with TP
    above one, set ``ulysses_degree`` to the CP size.

    Args:
        target_module: Xing40 MLA module being wrapped.
        mesh: Full mesh, passed to the generic wrapper.
        tp_mesh: Tensor-parallel mesh or None.
        cp_mesh: Context-parallel mesh.
        ep_mesh: Expert-parallel mesh or None.
        strategy: Expanded QKV Ulysses or expanded Q with latent KV AllGather.
        overlap: Overlap forward input communication with independent MLA projections.
        ulysses_degree: Heads split over this many CP ranks; one selects KV AllGather and None the CP size.
        kv_exchange: Full KV collection or selective causal-prefix distribution.

    Returns:
        Forward rewrite requests from the generic MLA wrapper.
    """
    if not isinstance(target_module, Xing40MLAAttention):
        raise TypeError('Xing40 MLA replacement required')
    if cp_mesh is None:
        raise ValueError('Xing40 CP adapter requires a CP mesh')
    return mla_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh, strategy=strategy, overlap=overlap,
                          ulysses_degree=ulysses_degree, kv_exchange=kv_exchange, rotary_fn=xing40_rotary)
