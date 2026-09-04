# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""expert_parallel: the Qwen3-MoE EP compute archetype factory.

Migrated from ``distributed/expert_parallel/recipes.py`` in M3 (adjust doc
§5.1/§5.4): only the Qwen-specific router contract (``TopKRouter`` module
returning ``(logits, scores, indices)``, no shared expert) sinks into the
model adapter. The combine skeleton, the router-adapter registry and the
dispatch/collective primitives stay in the generic
``auto_models/distributed/expert_parallel`` layer; this factory composes
them through the now-public ``build_ep_compute``.
"""

from typing import Any, Callable

from hyper_parallel.distributed.expert_parallel.recipes import (
    build_ep_compute,
)
from hyper_parallel.distributed.expert_parallel.routing import (
    MOE_ROUTER_ADAPTERS,
)
from hyper_parallel.distributed.recipe_spec import (
    local_compute,
)


@local_compute
def qwen3moe_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    use_grouped_gemm: bool = False,
) -> Callable:
    """Archetype ``qwen3moe_topk_router``: TopKRouter module returning
    (logits, scores, indices), no shared expert (Qwen3-MoE).

    Expected module interface: ``gate`` (TopKRouter module), ``experts``.
    """
    del mesh, tp_mesh, cp_mesh
    return build_ep_compute(
        module,
        ep_mesh,
        router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
        archetype_key="qwen3moe_topk_router",
        expected_attrs=["gate", "experts"],
        combine=lambda module, hidden_states, routed: routed,
        use_grouped_gemm=use_grouped_gemm,
    )


__all__ = [
    "qwen3moe_ep_compute_fn",
]
