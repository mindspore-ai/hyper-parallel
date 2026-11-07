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
"""Expert-parallel execution of the Xing4.0 MoE layers.

Recipes bind :func:`xing40_ep_compute_fn` to the MoE modules with a
``when: ep`` ``plan_overrides`` entry.
"""

from collections.abc import Callable
from typing import Any

from hyper_parallel.distributed.expert_parallel.experts import SHARED_MERGE_MODES
from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.recipe_spec import local_compute
from hyper_parallel.models.xing40.modeling_xing40 import Xing40MoE

__all__ = ["xing40_ep_compute_fn"]


@local_compute
def xing40_ep_compute_fn(*, module: Xing40MoE, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                         ep_mesh: Any) -> Callable:
    """Dispatch the MoE's routed experts over ``ep_mesh``.

    The module keeps its own forward: pad tokens, the shared experts, the router's
    balance loss and token counts are the same with and without expert parallelism.
    Only the routed branch runs through the framework EP executor.

    Args:
        module: MoE layer to parallelize.
        mesh: Unused global mesh.
        tp_mesh: Unused tensor-parallel mesh.
        cp_mesh: Unused context-parallel mesh.
        ep_mesh: Expert-parallel mesh.

    Returns:
        The MoE forward, whose routed branch now runs through the installed EP executor.

    Raises:
        ValueError: From ``build_ep_compute``, if no expert-parallel mesh is given or ``module``
            lacks ``gate``, ``experts`` or ``shared_experts``.
    """
    del mesh, tp_mesh, cp_mesh
    module.ep_executor = build_ep_compute(
        module, ep_mesh, router_fn=Xing40MoE.route, archetype_key="xing40",
        expected_attrs=["gate", "experts", "shared_experts"], combine=SHARED_MERGE_MODES["none"],
        use_grouped_gemm=True)
    return Xing40MoE.forward
