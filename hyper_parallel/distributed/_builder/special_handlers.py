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
"""special_handlers: special-parameter handler registry (planner Phase 6, 05 §6.4.6).

``SPECIAL_HANDLERS`` maps handler names to ``callable(module, param_name, mesh)``;
``_SPECIAL_HANDLER_PATTERNS`` maps lowercase FQN substrings to handler names;
``_collect_special_handlers`` is the Phase 6 collection pass (SPECIAL-role
parameters → handler name).
"""

from typing import Callable, Dict

from torch import nn

from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed.tensor_parallel.param_role import (
    ParamRole,
    _match_any,
)

def _shard_gated_delta(module, param_name, mesh):
    """Custom TP sharding skeleton for gated_delta modules (SSM/Mamba-style
    modules, 05 §6.4.6).

    Shards along the SSM head structure rather than standard
    colwise/rowwise. Skeleton implementation: structural recognition plus
    a standard Shard(0) fallback; the head-aligned fine-grained sharding is
    left to be completed when a concrete model is onboarded.
    """
    param = getattr(module, param_name, None)
    if param is None:
        return
    sharded = distribute_tensor(param.data, mesh, [Shard(0)])
    module.register_parameter(param_name, nn.Parameter(sharded))


# {handler_name: callable(module, param_name, mesh)} — Phase B special parameter handlers.
SPECIAL_HANDLERS: Dict[str, Callable] = {
    "gated_delta_tp_shard": _shard_gated_delta,
}

# planner-side pattern → handler_name mapping (lowercase fqn substring match).
_SPECIAL_HANDLER_PATTERNS: Dict[str, str] = {
    "gated_delta": "gated_delta_tp_shard",
    "a_log": "gated_delta_tp_shard",
    "dt_bias": "gated_delta_tp_shard",
}


def _collect_special_handlers(
    param_roles: Dict[str, ParamRole],
    special_handler_patterns: Dict[str, str],
) -> Dict[str, str]:
    """SPECIAL-role parameters → handler name (unregistered patterns
    fall back to "default")."""
    result: Dict[str, str] = {}
    for fqn, role in param_roles.items():
        if role != ParamRole.SPECIAL:
            continue
        handler_name = "default"
        for pattern, hname in special_handler_patterns.items():
            if _match_any(fqn.lower(), [pattern.lower()]):
                handler_name = hname
                break
        result[fqn] = handler_name
    return result
