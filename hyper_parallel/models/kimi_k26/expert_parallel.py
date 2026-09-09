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
"""Configured EP computation for the native Transformers Kimi MoE."""

from typing import Any, Callable

# The native Transformers MoE and EP compute recipes are PyTorch-specific.
import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.recipe_spec import local_compute


def kimi_topk_router(module: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use native sigmoid/group selection, normalized scores and scaling exactly once."""
    indices, weights = module.gate(hidden_states)
    return indices, weights


def _combine_shared(module: Any, hidden_states: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
    """Match native MoE's cast before the shared-expert residual addition."""
    return routed.to(hidden_states.dtype) + module.shared_experts(hidden_states)


@local_compute
def kimi_k26_moe_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    use_grouped_gemm: bool = False,
) -> Callable:
    """Build EP MoE computation through an explicit plan override.

    Args:
        module: Planned MLP; routed experts are stacked when EP is active.
        mesh: Complete topology, supplied by the local-compute framework.
        tp_mesh: Tensor parallel mesh; inactive for this staged adapter.
        cp_mesh: Context parallel mesh; inactive for this staged adapter.
        ep_mesh: Expert mesh used by the expert parameter layouts.
        use_grouped_gemm: Whether to use the existing grouped expert kernel.

    Returns:
        EP local forward, or the original forward for a matched dense MLP.
    """
    del mesh, tp_mesh, cp_mesh
    if not hasattr(module, "experts"):
        return type(module).forward
    return build_ep_compute(
        module,
        ep_mesh,
        router_fn=kimi_topk_router,
        archetype_key="kimi_k26_native_topk_shared",
        expected_attrs=["gate", "experts", "shared_experts"],
        combine=_combine_shared,
        use_grouped_gemm=use_grouped_gemm,
        preserve_router_dtype=True,
    )
