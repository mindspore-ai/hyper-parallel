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
"""ep_compute: built-in EP compute factories, organized as EXPLICIT archetypes
(accuracy_fix_plan.md §3 E2).

Since the explicit-injection rework the planner only SHARDS the expert
parameters (``{EP: Shard(0)}`` + stacking metadata); the EP compute is never
injected automatically — a factory must be referenced explicitly from YAML:

.. code-block:: yaml

    plan_overrides:
      - match: "*.mlp"
        when: ep
        local_compute_fn:
          _target_: hyper_models.components.distributed.ep_compute.qwen2moe_ep_compute_fn

Archetype table (each factory is a COMPLETE semantic implementation —
router / dispatch / experts / combine / shared expert / gate / merge are
all inside; the framework never fills in any branch for you):

===============================  ======================  ===========================  =========================
archetype key                    router                  shared expert                representative models
===============================  ======================  ===========================  =========================
routed_only_softmax_topk         softmax top-k           none                         legacy linear routers
mixtral_topk_router              TopKRouter tuple        none                         Mixtral
qwen3moe_topk_router             TopKRouter module       none                         Qwen3-MoE
qwen2moe_shared_expert_gate      TopKRouter tuple        shared_expert + sigmoid gate Qwen2-MoE
deepseekv3_sigmoid_group_shared  sigmoid group routing   shared_experts               DeepSeek-V3 / GLM-4-MoE
===============================  ======================  ===========================  =========================

Three hard constraints on every archetype (accuracy_fix_plan.md §3.2):

1. **Explicitly keyed, zero runtime probing** — the user names the factory
   in YAML; the implementation accesses ``module.experts`` /
   ``module.shared_expert`` etc. directly. There are no getattr fallback
   chains anywhere (a half-found branch is a SILENT numeric error — see
   accuracy_problem.md 10.2).
2. **Apply-time interface assertion** — the factory runs ONCE at apply time
   with the real module; it first asserts every attribute it will call and
   fails the build with a teaching error (listing the module's actual
   children) on mismatch. The assertion proves structural match, not
   semantic correctness (that is vouched by numeric verification).
3. **Complete semantics** — nothing is left for the framework to guess.

Not sure which archetype fits? ``describe_moe_module(module)`` (ep_utils)
prints the module's children/param shapes; ``EP_ARCHETYPE_SUGGESTIONS``
maps the planner's canonical arch name to a likely archetype (a SUGGESTION
shown in error messages — selection stays explicit; a wrong suggestion is
caught by constraint 2). Non-typical MoE: copy
``examples/distributed/ep_factories.py`` and write your own factory with
the same contract.

Nested-boundary call contract (inside every compute_fn): calling a
planned nested-boundary submodule (e.g. ``module.shared_expert(...)``)
returns its out_dst layout with TP communication already sealed inside —
the caller MUST NOT apply compensating collectives to the return value:

.. code-block:: python

    dist.all_reduce(shared, group=tp_group)   # FORBIDDEN — the nested
                                              # boundary already reduced

Factory contract: decorated ``@local_compute`` (injection discipline —
undecorated factories fail fast in the resolution chain). The mesh family
``mesh`` / ``tp_mesh`` / ``cp_mesh`` / ``ep_mesh`` is MANDATORY context,
ALL filled by the framework at apply time (None for inactive axes;
``ep_mesh`` is the same object the expert parameters were sharded on).
The factory must RETURN the compute fn
``fn(module, *local_args, **local_kwargs) -> Tensor`` executed on local
tensors inside the local-region skeleton.
"""

from typing import Any, Callable

import torch

from hyper_models.components.distributed.ep_utils import (
    MOE_ROUTER_ADAPTERS,
    bind_local_expert_forward,
    ep_routed_forward,
    require_attrs,
)
from hyper_models.components.distributed.injection import (
    local_compute,
)


# {canonical arch name (ShardingPlanner._get_architecture): archetype key} —
# a SUGGESTION shown in preflight error messages and describe_moe_module
# output, never an automatic selection (accuracy_fix_plan.md §3 E2).
EP_ARCHETYPE_SUGGESTIONS = {
    "qwen2moe": "qwen2moe_shared_expert_gate",
    "qwen2_moe": "qwen2moe_shared_expert_gate",
    "qwen3moe": "qwen3moe_topk_router",
    "qwen3_moe": "qwen3moe_topk_router",
    "mixtral": "mixtral_topk_router",
    "deepseekv3": "deepseekv3_sigmoid_group_shared",
    "deepseek_v3": "deepseekv3_sigmoid_group_shared",
    "glm4moe": "deepseekv3_sigmoid_group_shared",
    "glm4_moe": "deepseekv3_sigmoid_group_shared",
}


def _require_ep_group(ep_mesh: Any, factory_name: str):
    """Validate the expert mesh context and return the EP process group."""
    if ep_mesh is None:
        raise ValueError(
            f"{factory_name} was built without an ep_mesh — this factory must "
            "be injected on an EP-sharded MoE boundary (ep_size > 1 so the "
            "framework-derived expert mesh exists). If you meant TP-only "
            "expert sharding, do not use an EP compute factory."
        )
    return ep_mesh.get_group("ep")


def _require_moe_interface(module: Any, expected_attrs, archetype_key: str) -> None:
    """Apply-time structural assertion for an archetype (constraint 2)."""
    require_attrs(module, *expected_attrs, owner=f"archetype '{archetype_key}'")


def _build_ep_compute(
    module: Any,
    ep_mesh: Any,
    *,
    router_fn: Callable,
    archetype_key: str,
    expected_attrs,
    combine: Callable,
) -> Callable:
    """Shared skeleton for archetype factories: validate context, assert the
    interface, bind the local expert entry point, and close over the
    archetype-specific ``combine`` (the branch composition — the ONLY part
    that differs between archetypes).

    ``combine(module, hidden_states, routed) -> Tensor`` receives the routed
    branch output and returns the MoE block output.
    """
    ep_group = _require_ep_group(ep_mesh, f"archetype '{archetype_key}'")
    _require_moe_interface(module, expected_attrs, archetype_key)
    bind_local_expert_forward(module, ep_mesh["ep"].size())

    def compute_fn(module, hidden_states):
        routed = ep_routed_forward(
            module, hidden_states, router_fn=router_fn, ep_group=ep_group)
        return combine(module, hidden_states, routed)

    return compute_fn


@local_compute
def routed_only_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> Callable:
    """Archetype ``routed_only_softmax_topk``: softmax top-k router, no
    shared expert (Mixtral-style). Output = routed branch only.

    Expected module interface: ``gate`` (router Linear), ``experts``.
    """
    del mesh, tp_mesh, cp_mesh
    return _build_ep_compute(
        module,
        ep_mesh,
        router_fn=MOE_ROUTER_ADAPTERS["default"],
        archetype_key="routed_only_softmax_topk",
        expected_attrs=["gate", "experts"],
        combine=lambda module, hidden_states, routed: routed,
    )


@local_compute
def mixtral_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> Callable:
    """Archetype ``mixtral_topk_router`` with training-time router jitter.

    Transformers 5.12 Mixtral uses a TopKRouter returning
    ``(logits, scores, indices)``. When configured, the original block also
    perturbs hidden states before both routing and expert computation.
    """
    del mesh, tp_mesh, cp_mesh
    ep_group = _require_ep_group(ep_mesh, "archetype 'mixtral_topk_router'")
    _require_moe_interface(module, ["gate", "experts"], "mixtral_topk_router")
    bind_local_expert_forward(module, ep_mesh["ep"].size())

    def compute_fn(module, hidden_states):
        jitter_noise = float(getattr(module, "jitter_noise", 0.0))
        if module.training and jitter_noise > 0.0:
            noise = torch.empty_like(hidden_states).uniform_(
                1.0 - jitter_noise,
                1.0 + jitter_noise,
            )
            hidden_states = hidden_states * noise
        return ep_routed_forward(
            module,
            hidden_states,
            router_fn=MOE_ROUTER_ADAPTERS["mixtral"],
            ep_group=ep_group,
        )

    return compute_fn


@local_compute
def qwen3moe_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> Callable:
    """Archetype ``qwen3moe_topk_router``: TopKRouter module returning
    (logits, scores, indices), no shared expert (Qwen3-MoE).

    Expected module interface: ``gate`` (TopKRouter module), ``experts``.
    """
    del mesh, tp_mesh, cp_mesh
    return _build_ep_compute(
        module,
        ep_mesh,
        router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
        archetype_key="qwen3moe_topk_router",
        expected_attrs=["gate", "experts"],
        combine=lambda module, hidden_states, routed: routed,
    )


@local_compute
def qwen2moe_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> Callable:
    """Archetype ``qwen2moe_shared_expert_gate``: TopKRouter tuple +
    shared_expert + sigmoid scalar gate (Qwen2-MoE):
    ``routed + sigmoid(shared_expert_gate(x)) * shared_expert(x)``.

    Expected module interface: ``gate``, ``experts``, ``shared_expert``,
    ``shared_expert_gate``.

    ``module.shared_expert(...)`` is a planned nested TP boundary: its exit
    already performs the RowWise Partial TP reduction (nested-boundary call
    contract) — do NOT all-reduce its return value.
    """
    del mesh, tp_mesh, cp_mesh

    def combine(module, hidden_states, routed):
        shared = module.shared_expert(hidden_states)            # nested boundary
        gate = torch.sigmoid(module.shared_expert_gate(hidden_states))
        return routed + gate * shared

    return _build_ep_compute(
        module,
        ep_mesh,
        router_fn=MOE_ROUTER_ADAPTERS["qwen2moe"],
        archetype_key="qwen2moe_shared_expert_gate",
        expected_attrs=["gate", "experts", "shared_expert", "shared_expert_gate"],
        combine=combine,
    )


@local_compute
def deepseekv3_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> Callable:
    """Archetype ``deepseekv3_sigmoid_group_shared``: sigmoid group-limited
    routing (with e_score_correction_bias / routed_scaling_factor, already
    folded into the router adapter's topk weights) + shared_experts
    (DeepSeek-V3 / GLM-4-MoE): ``routed + shared_experts(x)``.

    Expected module interface: ``gate``, ``experts``, ``shared_experts``.

    ``module.shared_experts(...)`` is a planned nested TP boundary — its
    exit already performs the TP reduction (nested-boundary call contract).
    """
    del mesh, tp_mesh, cp_mesh

    def combine(module, hidden_states, routed):
        return routed + module.shared_experts(hidden_states)    # nested boundary

    return _build_ep_compute(
        module,
        ep_mesh,
        router_fn=MOE_ROUTER_ADAPTERS["deepseekv3"],
        archetype_key="deepseekv3_sigmoid_group_shared",
        expected_attrs=["gate", "experts", "shared_experts"],
        combine=combine,
    )


# {archetype key: factory} — introspection registry (the YAML _target_
# points at a factory directly; this registry exists for
# describe/preflight messaging).
EP_ARCHETYPES = {
    "routed_only_softmax_topk": routed_only_ep_compute_fn,
    "mixtral_topk_router": mixtral_ep_compute_fn,
    "qwen3moe_topk_router": qwen3moe_ep_compute_fn,
    "qwen2moe_shared_expert_gate": qwen2moe_ep_compute_fn,
    "deepseekv3_sigmoid_group_shared": deepseekv3_ep_compute_fn,
}
