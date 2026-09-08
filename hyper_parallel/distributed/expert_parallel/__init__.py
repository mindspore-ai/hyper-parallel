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

"""expert_parallel: EP collectives, routing, expert execution and recipes.

Public surface: the built-in ``@local_compute`` EP archetype factories
(``EP_ARCHETYPES`` + ``EP_ARCHETYPE_SUGGESTIONS``) and the EP primitives
(``ep_all_to_all`` / ``MOE_ROUTER_ADAPTERS`` / ``ep_routed_forward`` /
``bind_local_expert_forward`` and friends). Split out of
components/distributed/ep_compute.py + ep_utils.py in stage 4e.
"""

from hyper_parallel.distributed.expert_parallel.collectives import (
    ep_all_to_all,
)
from hyper_parallel.distributed.expert_parallel.routing import (
    MOE_ROUTER_ADAPTERS,
)
from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
    describe_moe_module,
    ep_routed_forward,
    require_attrs,
    resolve_swiglu_weights,
)
from hyper_parallel.distributed.expert_parallel.recipes import (
    EP_ARCHETYPE_SUGGESTIONS,
    EP_ARCHETYPES,
    build_ep_compute,
    deepseekv3_ep_compute_fn,
    mixtral_ep_compute_fn,
    qwen2moe_ep_compute_fn,
    routed_only_ep_compute_fn,
)

__all__ = [
    "EP_ARCHETYPES",
    "EP_ARCHETYPE_SUGGESTIONS",
    "MOE_ROUTER_ADAPTERS",
    "bind_local_expert_forward",
    "build_ep_compute",
    "deepseekv3_ep_compute_fn",
    "describe_moe_module",
    "ep_all_to_all",
    "ep_routed_forward",
    "mixtral_ep_compute_fn",
    "qwen2moe_ep_compute_fn",
    "require_attrs",
    "resolve_swiglu_weights",
    "routed_only_ep_compute_fn",
]
