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
"""Expose the native computation independently of live MegaMoE resources."""

from typing import Any, Callable

from hyper_parallel.distributed.recipe_spec import local_compute
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    _build_megamoe_compute,
    _megamoe_compute_context,
    _router_sequence_groups,
    deepseek_v41_megamoe_compute_fn,
)


@local_compute
def _megamoe_source_factory(
        *, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any, fix_router: bool = False,
) -> Callable:
    """Use the same validated compute builder without initializing runtime resources."""
    _, arguments = _megamoe_compute_context(
        module=module, mesh=mesh, tp_mesh=tp_mesh, cp_mesh=cp_mesh, ep_mesh=ep_mesh, fix_router=fix_router,
    )
    return _build_megamoe_compute(sequence_groups=_router_sequence_groups(tp_mesh, cp_mesh), **arguments)


def source_compute_factory(factory: Callable) -> Callable:
    """Select a resource-free source view only for this adapter's native factory."""
    return _megamoe_source_factory if factory is deepseek_v41_megamoe_compute_fn else factory
