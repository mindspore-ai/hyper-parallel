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
"""ep_compute: public EP local-region compute factories (explicit injection).

Since the explicit-injection rework the planner only SHARDS the expert
parameters (``{EP: Shard(0)}`` + stacking metadata); the EP compute (router
-> all-to-all dispatch -> local experts -> all-to-all combine) is never
injected automatically. Reference the built-in default explicitly — from
YAML:

.. code-block:: yaml

    plan_overrides:
      - match: "*.mlp"
        when: ep
        local_compute_fn:
          _target_: hyper_models.components.distributed.ep_compute.hf_native_ep_compute_fn

or programmatically (``PlanOverride(match=..., local_compute_fn=Target(
hf_native_ep_compute_fn, target_path="..."))``), or point the Target at your
own factory with the same contract.

Factory contract: the factory must be decorated ``@local_compute``
(injection discipline, see injection.py — undecorated factories fail fast
in the resolution chain). The mesh family ``mesh`` / ``tp_mesh`` /
``cp_mesh`` / ``ep_mesh`` is MANDATORY context, ALL filled by the framework
at apply time (None for inactive axes; ``ep_mesh`` is the same object the
expert parameters were sharded on); ``module`` is optional context.
Behavior choices (routing, layouts, ...) are written INTO the factory —
config keys carry data only, never functions. The factory must RETURN the
compute fn ``fn(module, *local_args, **local_kwargs) -> Tensor`` executed
on local tensors inside the local-region skeleton; the returned fn's
params are validated against the module's forward signature at apply time.
"""

from hyper_models.components.distributed.ep_utils import (
    _hf_native_ep_compute,
    _softmax_topk_router,
)
from hyper_models.components.distributed.injection import (
    local_compute,
)


@local_compute
def hf_native_ep_compute_fn(*, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Factory for the built-in TP-extend-EP compute (D-10, 05 §6.4.8).

    Returns ``compute_fn(module, hidden_states)``: router (local chunk) ->
    all-to-all dispatch over the extended EP group -> local SwiGLU on
    complete expert weights (no internal communication) -> all-to-all
    combine -> weighted aggregation (+ shared_experts). Isomorphic to
    Megatron MoEAlltoAllTokenDispatcher with expert_tensor_parallel_size=1.

    The router is PART of the injected compute (injection discipline: the
    framework never decides the user's router, and function-typed config is
    rejected) — this built-in embeds the default softmax top-k adapter
    (``_softmax_topk_router``). A MoE with different routing (Qwen3
    TopKRouter module / DeepSeek sigmoid-group / ...) injects its OWN
    factory with the routing written inline — reusing MOE_ROUTER_ADAPTERS
    entries by name is fine, the choice lives in your code::

        @local_compute
        def qwen3moe_ep_compute_fn(*, mesh, tp_mesh, cp_mesh, ep_mesh):
            ep_group = ep_mesh.get_group("ep")
            tp_group = tp_mesh.get_group() if tp_mesh is not None else None

            def compute_fn(module, hidden_states):
                return _hf_native_ep_compute(
                    module, hidden_states,
                    router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
                    ep_group=ep_group, tp_group=tp_group)
            return compute_fn

    Context args (ALL framework-filled; the user just uses them):
        mesh / tp_mesh / cp_mesh: the active DTensor mesh and its tp/cp
            submeshes (None when the axis is inactive). ``tp_mesh`` gives
            the TP reduction group for shared_experts
            (``tp_mesh.get_group()``); without a tp axis tp_group is None
            (shared_experts then fail fast in _hf_native_ep_compute, as
            TP-sharded partial output would go unreduced).
        ep_mesh: the framework-derived expert mesh ``(edp, ep)`` — the SAME
            object the expert parameters were sharded on, so the a2a domain
            and the sharding domain agree by construction. None means the
            factory was mounted on a non-EP boundary -> configuration error.
    """
    if ep_mesh is None:
        raise ValueError(
            "hf_native_ep_compute_fn was built without an ep_mesh — this "
            "factory must be injected on an EP-sharded MoE boundary "
            "(ep_size > 1 so the framework-derived expert mesh exists). If "
            "you meant TP-only expert sharding, do not use this factory.")
    ep_group = ep_mesh.get_group("ep")
    tp_group = tp_mesh.get_group() if tp_mesh is not None else None

    def compute_fn(module, hidden_states):
        return _hf_native_ep_compute(
            module, hidden_states, router_fn=_softmax_topk_router,
            ep_group=ep_group, tp_group=tp_group)

    return compute_fn
