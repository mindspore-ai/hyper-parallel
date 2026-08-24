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
"""ep_factories: reference implementations for writing your OWN EP compute
factory (accuracy_fix_plan.md §3 E2).

The built-in archetypes in ``hyper_parallel.auto_models.components.distributed.ep_compute``
cover the four typical transformers MoE behaviors. If your model matches none
of them (custom router, auxiliary-loss side channels, non-SwiGLU experts,
unusual shared-expert wiring...), copy one of the factories below and adjust
— every factory here is a COMPLETE semantic implementation: router /
dispatch / experts / combine / shared expert / gate / merge are all inside,
composed from the public primitives (``ep_routed_forward`` /
``bind_local_expert_forward`` / ``MOE_ROUTER_ADAPTERS`` / ``require_attrs``).

Hard rules (same as the built-in archetypes):

1. Explicit selection — you name YOUR factory in YAML ``_target_``; never
   probe the module with getattr fallback chains ("half-found" branches are
   silent numeric errors).
2. Apply-time interface assertion — ``require_attrs`` fails the build in
   seconds with the module's actual children listed, instead of an
   AttributeError at training step N.
3. Nested-boundary call contract — calling a planned nested-boundary
   submodule (e.g. ``module.shared_expert(...)``) returns its out_dst
   layout with TP communication already sealed inside. NEVER compensate:

       dist.all_reduce(shared, group=tp_group)   # FORBIDDEN

Usage (YAML):

.. code-block:: yaml

    plan_overrides:
      - match: "*.mlp"
        when: ep
        region_dispatch: false
        local_compute_fn:
          _target_: examples.distributed.ep_factories.my_moe_ep_compute_fn
"""

from typing import Any, Callable

import torch

from hyper_parallel.auto_models.components.distributed.ep_utils import (
    MOE_ROUTER_ADAPTERS,
    bind_local_expert_forward,
    ep_routed_forward,
    require_attrs,
)
from hyper_parallel.auto_models.components.distributed.injection import local_compute


@local_compute
def my_moe_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> Callable:
    """Template factory: softmax top-k routed branch + shared expert merged
    with a learned scalar gate (Qwen2-MoE-shaped). Rename the attribute
    accesses to your model's actual names — ``require_attrs`` will tell you
    the actual children if they don't match.

    The mesh family is framework-filled context; a typical factory only
    needs ``ep_mesh`` (the a2a domain — the SAME object the expert weights
    were sharded on). Note there is intentionally NO tp_group usage: TP
    communication of nested-boundary children is sealed inside their own
    boundaries.
    """
    del mesh, tp_mesh, cp_mesh  # framework-filled context; unused here
    if ep_mesh is None:
        raise ValueError(
            "my_moe_ep_compute_fn was built without an ep_mesh — inject it "
            "on an EP-sharded MoE boundary (ep_size > 1)")
    # Apply-time interface assertion: the names this factory will call.
    require_attrs(module, "gate", "experts", "shared_expert",
                  "shared_expert_gate", owner="my_moe_ep_compute_fn")
    ep_group = ep_mesh.get_group("ep")
    bind_local_expert_forward(module, ep_mesh["ep"].size())
    # Router choice is explicit — pick an adapter BY NAME, or write your own
    # (module, hidden) -> (topk_idx [T,K] int64, topk_w [T,K] float).
    router_fn = MOE_ROUTER_ADAPTERS["default"]

    def compute_fn(module, hidden_states):
        routed = ep_routed_forward(
            module, hidden_states, router_fn=router_fn, ep_group=ep_group)
        # nested-boundary calls: NO compensating collectives on the returns
        shared = module.shared_expert(hidden_states)
        gate = torch.sigmoid(module.shared_expert_gate(hidden_states))
        return routed + gate * shared

    return compute_fn


@local_compute
def my_routed_only_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> Callable:
    """Template factory: routed branch only (no shared expert) with a
    TopKRouter-module gate (Qwen3-MoE-shaped)."""
    del mesh, tp_mesh, cp_mesh
    if ep_mesh is None:
        raise ValueError(
            "my_routed_only_ep_compute_fn was built without an ep_mesh — "
            "inject it on an EP-sharded MoE boundary (ep_size > 1)")
    require_attrs(module, "gate", "experts",
                  owner="my_routed_only_ep_compute_fn")
    ep_group = ep_mesh.get_group("ep")
    bind_local_expert_forward(module, ep_mesh["ep"].size())
    router_fn = MOE_ROUTER_ADAPTERS["qwen3moe"]

    def compute_fn(module, hidden_states):
        return ep_routed_forward(
            module, hidden_states, router_fn=router_fn, ep_group=ep_group)

    return compute_fn
