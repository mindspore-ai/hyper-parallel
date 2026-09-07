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
"""Lower public MoE plans to inference-owned local expert storage."""

from copy import deepcopy
from functools import wraps
from typing import Any

# vLLM's local MoE boundary is Torch-only, like the model adapters it wraps.
from torch.nn import functional  # pylint: disable=forbidden-backend-import

from hyper_parallel.auto_models.components.distributed import ShardingPlanner
from hyper_parallel.auto_models.components.distributed.sharding_config import ShardingPlan


def build_moe_tp_plan(model: Any, mesh: Any, *, tp_size: int, ep_size: int) -> ShardingPlan:
    """Reuse public weight rules and bridge replicated inference tokens to EP input.

    The model must still expose its canonical HF expert parameters. Only
    expert storage is lowered out of the apply pass: the inference leaf
    materializes those public EP slices through its existing loader.
    """
    if tp_size <= 1 or ep_size <= 1:
        raise ValueError("The inference MoE TP bridge requires TP>1 and EP>1")
    planner = ShardingPlanner()
    plan = planner.plan(model, mesh, tp_size=tp_size, ep_size=ep_size, sequence_parallel=False)
    sequence_plan = planner.plan(model, mesh, tp_size=tp_size, ep_size=ep_size, sequence_parallel=True)
    count = 0
    moe_boundaries = []
    for name, spec in plan.modules.items():
        if not spec._ep_size:  # pylint: disable=protected-access
            continue
        expert_params = {key: axes for key, axes in spec.params.items() if key.startswith("experts.")}
        if not expert_params:
            raise ValueError(f"Public EP boundary {name!r} contains no expert parameters")
        for parameter, axes in expert_params.items():
            placement = axes.get("ep")
            if placement is None or not placement.is_shard() or placement.dim != 0 or "tp" in axes:
                raise ValueError(f"Unsupported public local-expert placement: {name}.{parameter}: {axes}")
        spec.params = {key: axes for key, axes in spec.params.items() if key not in expert_params}
        # Expert allocation is already owned by the local-leaf adapter. Keep
        # the public EP token contract, without asking apply to shard it twice.
        spec._ep_size = 0  # pylint: disable=protected-access
        spec.in_dst = deepcopy(sequence_plan.modules[name].in_src)
        spec.out_src = deepcopy(sequence_plan.modules[name].out_dst)
        spec.region_dispatch = False
        moe_boundaries.append(name)
        count += 1
    if not count:
        raise ValueError("Public planner did not find a TP-extended MoE boundary")
    for name, spec in plan.modules.items():
        if any(name.startswith(parent + ".") for parent in moe_boundaries):
            # Shared experts execute inside the MoE's local sequence chunk.
            # Their nested public boundary must gather/scatter that contract.
            for field in ("in_src", "in_dst", "out_src", "out_dst"):
                setattr(spec, field, deepcopy(getattr(sequence_plan.modules[name], field)))
    return plan


def pad_moe_tp_tokens(module: Any, tp_size: int) -> None:
    """Pad only the pointwise MoE region for equal-size public TP collectives.

    Attention/cache metadata never sees these rows. Real token outputs are
    trimmed after the public boundary gathers the EP results back to TP.
    """
    original = module.forward

    @wraps(original)
    def forward(hidden_states: Any) -> Any:
        """Evaluate the token-local region and preserve real token outputs."""
        token_count = hidden_states.shape[1]
        padding = (-token_count) % tp_size if token_count else tp_size
        if padding:
            hidden_states = functional.pad(hidden_states, (0, 0, 0, padding))
        output = original(hidden_states)
        return output[:, :token_count, :]

    module.forward = forward
