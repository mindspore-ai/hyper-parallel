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
"""GroupedExperts Family EP adaptation with model-owned routing semantics."""

from typing import Any, Callable, Optional

from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.recipe_spec import local_compute


@local_compute
def grouped_experts_ep_compute_fn(
    *, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
    router_fn: Callable, combine: Callable, overlap_fn: Optional[Callable] = None,
    capacity_factor: Optional[float] = None,
) -> Callable:
    """Build EP execution for a MoE owner whose Experts is a Family replacement.

    Like model EP factories, this adapter targets the current MoE boundary.
    The model supplies router/combine semantics and optional shared-branch
    computation. Dispatch, chunking, weights and expert hooks remain owned by
    the existing EP recipe; the selected Experts owns local computation.
    """
    del mesh, tp_mesh, cp_mesh
    if not callable(getattr(module.experts, "forward_expert_major", None)):
        raise ValueError("GroupedExperts EP adapter requires forward_expert_major")
    # Enter the existing expert-major branch; the module method still owns
    # the selected computation. The callable check above excludes native fallback.
    return build_ep_compute(
        module, ep_mesh, router_fn=router_fn, combine=combine,
        archetype_key="grouped_experts", expected_attrs=["experts"],
        use_grouped_gemm=True, overlap_fn=overlap_fn, capacity_factor=capacity_factor,
    )
