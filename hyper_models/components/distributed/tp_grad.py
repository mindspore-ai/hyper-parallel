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
"""tp_grad: build_tp_grad_info (05 §6.7.1).

tp_grad_info is read from the ShardingPlan (rather than from DTensors — under
production the parameters have already been unwrapped by
_local_params_context, and only the plan retains the complete placement
information).
"""

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def build_tp_grad_info(plan, dense_tp_mesh, *, expert_source_mesh=None, tied_pairs=None):
    """Build source metadata for dense and routed-expert parameters.

    tied_pairs: parameter pairs with shared storage (defaults to
    plan.tied_pairs). Both ends of a tied pair must map to the same
    tp_placement — when the placements disagree, take the finer sharding
    (Shard takes precedence over Replicate), guaranteeing consistent TP
    all-reduce / reduce-scatter semantics on both ends.
    """
    info = {}
    for fqn, spec in plan.modules.items():
        for param_name, named_placement in spec.params.items():
            full_fqn = f"{fqn}.{param_name}"
            if spec._ep_size > 0 and param_name.startswith("experts."):  # pylint: disable=protected-access
                if expert_source_mesh is None:
                    raise ValueError(
                        "Routed expert metadata requires an expert EP source mesh"
                    )
                info[full_fqn] = (Shard(0), expert_source_mesh)
            else:
                tp_placement = named_placement.get("tp", Replicate())
                info[full_fqn] = (tp_placement, dense_tp_mesh)

    pairs = tied_pairs if tied_pairs is not None else plan.tied_pairs
    if pairs:
        for a, b in pairs:
            if a in info and b in info:
                pa, _ = info[a]
                pb, _ = info[b]
                if pa != pb:
                    norm = pa if isinstance(pa, Shard) else pb
                    info[a] = (norm, info[a][1])
                    info[b] = (norm, info[b][1])
    return info
