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
"""Structural TP template for shared experts in TP-extended MoE modules."""

from typing import Any, Dict

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

from .sharding_config import ModuleShardingSpec


_SHARED_EXPERT_LINEARS = ("linear_fc1", "linear_fc2")


def _matching_shared_experts(model: Any) -> Dict[str, Any]:
    """Find shared experts owned by an MoE dispatcher.

    Requiring the parent to expose both ``experts`` and ``token_dispatcher``
    keeps this template specific to the TP-extends-EP execution contract.  A
    regular shared-expert MLP remains eligible for the standard colwise/
    rowwise TP template.
    """
    matches = {}
    for fqn, module in model.named_modules():
        shared_expert = getattr(module, "shared_expert", None)
        if shared_expert is None:
            continue
        if not all(hasattr(module, name) for name in ("experts", "token_dispatcher")):
            continue
        if not all(hasattr(shared_expert, name) for name in _SHARED_EXPERT_LINEARS):
            continue
        matches[f"{fqn}.shared_expert" if fqn else "shared_expert"] = shared_expert
    return matches


def matches_shared_expert_template(model: Any) -> bool:
    """Return whether *model* contains a TP-extended MoE shared expert."""
    return bool(_matching_shared_experts(model))


def build_shared_expert_specs(model: Any) -> Dict[str, ModuleShardingSpec]:
    """Keep shared-expert linears replicated on each TP sequence shard.

    TP ranks participate in the EP dispatch domain, so these projections must
    not additionally shard their hidden dimension.  Each rank independently
    processes its existing sequence shard and returns the same layout.
    """
    specs = {}
    for shared_fqn, shared_expert in _matching_shared_experts(model).items():
        for linear_name in _SHARED_EXPERT_LINEARS:
            linear = getattr(shared_expert, linear_name)
            params = {
                name: {"tp": Replicate()}
                for name, _ in linear.named_parameters(recurse=False)
            }
            specs[f"{shared_fqn}.{linear_name}"] = ModuleShardingSpec(
                params=params,
                in_src={"hidden_states": {"tp": Shard(1)}},
                in_dst={"hidden_states": {"tp": Shard(1)}},
                out_src={"output": {"tp": Shard(1)}},
                out_dst={"output": {"tp": Shard(1)}},
            )
    return specs
