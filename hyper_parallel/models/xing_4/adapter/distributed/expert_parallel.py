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
"""Expert-parallel execution of the Xing4 MoE layers.

Recipes bind :func:`xing4_ep_compute_fn` to the MoE modules with a
``when: ep`` ``plan_overrides`` entry.
"""

from collections.abc import Callable
from typing import Any

from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.distributed.expert_parallel.experts import SHARED_MERGE_MODES
from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.recipe_spec import local_compute

__all__ = ["xing4_ep_compute_fn"]


@local_compute
def xing4_ep_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                         ep_mesh: Any, fix_router: bool = False) -> Callable:
    """Dispatch routed experts while retaining local shared experts and router state.

    ``fix_router`` balances dispatch indices for benchmarks; safe-token padding
    and statistics from the learned router keep their separate semantics.
    """
    del mesh, tp_mesh, cp_mesh
    if not isinstance(module.experts, GroupedExperts):
        raise ValueError("Xing4 EP requires the grouped experts replacement before binding the executor")
    module.ep_executor = build_ep_compute(
        module, ep_mesh, router_fn=type(module).route, archetype_key="xing4",
        expected_attrs=["gate", "experts", "shared_experts"], combine=SHARED_MERGE_MODES["none"],
        use_grouped_gemm=True, fix_router=fix_router)
    return type(module).forward
