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
"""expert_parallel.recipes: built-in EP compute factories, organized as EXPLICIT archetypes
(accuracy_fix_plan.md §3 E2).

Since the explicit-injection rework the planner only SHARDS the expert
parameters (``{EP: Shard(0)}`` + stacking metadata); the EP compute is never
injected automatically — a factory must be referenced explicitly from YAML:

.. code-block:: yaml

    plan_overrides:
      - match: "*.mlp"
        when: ep
        local_compute_fn:
          _target_: hyper_parallel.distributed.expert_parallel.recipes.qwen2moe_ep_compute_fn

Archetype table (each factory is a COMPLETE semantic implementation —
router / dispatch / experts / combine / shared expert / gate / merge are
all inside; the framework never fills in any branch for you):

===============================  ======================  ===========================  =========================
archetype key                    router                  shared expert                representative models
===============================  ======================  ===========================  =========================
routed_only_softmax_topk         softmax top-k           none                         legacy linear routers
mixtral_topk_router              TopKRouter tuple        none                         Mixtral
qwen2moe_shared_expert_gate      TopKRouter tuple        shared_expert + sigmoid gate Qwen2-MoE
deepseekv3_sigmoid_group_shared  sigmoid group routing   shared_experts               DeepSeek-V3 / GLM-4-MoE
===============================  ======================  ===========================  =========================

Model-family archetypes live at their model-adapter homes (adjust doc §5.1):
the Qwen3-MoE ``qwen3moe_topk_router`` archetype is
``models/qwen3_moe/adapter/distributed/expert_parallel.qwen3moe_ep_compute_fn``.

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

Not sure which archetype fits? ``describe_moe_module(module)`` (expert_parallel.experts)
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

from typing import Any, Callable, Optional

import torch

from hyper_parallel.distributed.expert_parallel.routing import (
    MOE_ROUTER_ADAPTERS,
)
from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
    ep_routed_dispatch,
    ep_routed_experts_and_combine,
    ep_routed_forward,
    require_attrs,
)
from hyper_parallel.distributed.recipe_spec import (
    local_compute,
)


# {canonical arch name (ShardingPlanner._get_architecture): archetype key} —
# a SUGGESTION shown in preflight error messages and describe_moe_module
# output, never an automatic selection (accuracy_fix_plan.md §3 E2).
EP_ARCHETYPE_SUGGESTIONS = {
    "qwen2moe": "qwen2moe_shared_expert_gate",
    "qwen2_moe": "qwen2moe_shared_expert_gate",
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


def build_ep_compute(
    module: Any,
    ep_mesh: Any,
    *,
    router_fn: Callable,
    archetype_key: str,
    expected_attrs,
    combine: Callable,
    use_grouped_gemm: bool = False,
    capacity_factor: Optional[float] = None,
    overlap_fn: Optional[Callable] = None,
) -> Callable:
    """Shared skeleton for archetype factories: validate context, assert the
    interface, bind the local expert entry point, and close over the
    archetype-specific ``combine`` (the branch composition — the ONLY part
    that differs between archetypes).

    ``combine(module, hidden_states, routed) -> Tensor`` receives the routed
    branch output and returns the MoE block output.

    When ``overlap_fn`` is given the routed branch is split into dispatch and
    experts+combine, and ``overlap_fn(module, hidden_states)`` runs in between:
    the token exchange is issued asynchronously there, so the overlapped work
    (for DeepSeek-V3 style blocks the shared-expert MLP, which depends only on
    ``hidden_states``) proceeds while it is in flight.  ``combine`` then
    receives that value as a fourth argument.

    Public since M3 (adjust doc §5.4): model adapters (e.g.
    ``models/qwen3_moe/adapter/distributed/expert_parallel.py``) compose
    their EP archetype on top of this skeleton while the router adapters and
    dispatch primitives stay in this generic layer.
    """
    ep_group = _require_ep_group(ep_mesh, f"archetype '{archetype_key}'")
    _require_moe_interface(module, expected_attrs, archetype_key)
    if use_grouped_gemm:
        bind_local_expert_forward(
            module,
            ep_mesh["ep"].size(),
            use_grouped_gemm=True,
        )
    else:
        # Keep the established call contract unchanged for every reference
        # archetype and for Qwen3 when grouped GEMM is disabled.
        bind_local_expert_forward(module, ep_mesh["ep"].size())

    if capacity_factor is not None:
        # Read back by ``experts._resolve_capacity_factor``; bounds each expert's routed slots
        # so the busiest rank's buffers cannot blow up under a skewed real router.
        module.experts.capacity_factor = float(capacity_factor)

    if overlap_fn is None:
        def compute_fn(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
            """Run the routed branch and compose the MoE block output."""
            routed = ep_routed_forward(
                module, hidden_states, router_fn=router_fn, ep_group=ep_group)
            return combine(module, hidden_states, routed)
    else:
        def compute_fn(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
            """Dispatch, run ``overlap_fn`` against the in-flight exchange, then finish."""
            state = ep_routed_dispatch(
                module, hidden_states, router_fn=router_fn, ep_group=ep_group)
            overlapped = overlap_fn(module, hidden_states)
            routed = ep_routed_experts_and_combine(module, state, ep_group)
            return combine(module, hidden_states, routed, overlapped)

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
    return build_ep_compute(
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

    def compute_fn(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply router jitter when training, then run the routed branch."""
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

    def combine(
        module: Any,
        hidden_states: torch.Tensor,
        routed: torch.Tensor,
    ) -> torch.Tensor:
        """Merge the routed branch with the gated shared-expert branch."""
        shared = module.shared_expert(hidden_states)            # nested boundary
        gate = torch.sigmoid(module.shared_expert_gate(hidden_states))
        return routed + gate * shared

    return build_ep_compute(
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
    use_grouped_gemm: bool = False,
    overlap_shared_expert: bool = False,
    fix_router: bool = False,
    capacity_factor: Optional[float] = None,
) -> Callable:
    """Archetype ``deepseekv3_sigmoid_group_shared``: sigmoid group-limited
    routing (with e_score_correction_bias / routed_scaling_factor, already
    folded into the router adapter's topk weights) + shared_experts
    (DeepSeek-V3 / GLM-4-MoE): ``routed + shared_experts(x)``.

    Expected module interface: ``gate``, ``experts``, ``shared_experts``.

    ``module.shared_experts(...)`` is a planned nested TP boundary — its
    exit already performs the TP reduction (nested-boundary call contract).

    ``use_grouped_gemm=True`` runs the local experts through the packed
    ``gate_up_proj`` grouped GEMM path (``npu_grouped_swiglu``) instead of
    the eager per-expert loop; the default keeps the eager path.

    ``overlap_shared_expert=True`` issues the token exchange asynchronously
    and evaluates ``shared_experts`` while it is in flight — the shared branch
    reads only ``hidden_states``, so it hides communication instead of
    serializing behind the routed branch. The math is unchanged either way.

    ``fix_router=True`` swaps the adapter for ``deepseekv3_fixed``, a
    BENCHMARK-ONLY variant that hands every rank the same token count
    (round-robin over the K slots by destination rank instead of the gate's
    top-k) so that per-rank step time stops depending on how the data
    happened to route. Loss / grad_norm / expert-load metrics are meaningless
    in that mode and it must not be used to report an efficiency number —
    see ``routing._balanced_router``. Correctness runs keep it off.
    """
    del mesh, tp_mesh, cp_mesh

    def combine(
        module: Any,
        hidden_states: torch.Tensor,
        routed: torch.Tensor,
        overlapped: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Merge the routed branch with the shared-experts branch."""
        shared = module.shared_experts(hidden_states) if overlapped is None else overlapped
        return routed + shared    # nested boundary

    def overlap_shared(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        """Shared-experts branch, evaluated while the dispatch exchange is in flight."""
        return module.shared_experts(hidden_states)    # nested boundary

    return build_ep_compute(
        module,
        ep_mesh,
        router_fn=MOE_ROUTER_ADAPTERS[
            "deepseekv3_fixed" if fix_router else "deepseekv3"],
        archetype_key="deepseekv3_sigmoid_group_shared",
        expected_attrs=["gate", "experts", "shared_experts"],
        combine=combine,
        use_grouped_gemm=use_grouped_gemm,
        capacity_factor=capacity_factor,
        overlap_fn=overlap_shared if overlap_shared_expert else None,
    )


# {archetype key: factory} — introspection registry (the YAML _target_
# points at a factory directly; this registry exists for
# describe/preflight messaging).
EP_ARCHETYPES = {
    "routed_only_softmax_topk": routed_only_ep_compute_fn,
    "mixtral_topk_router": mixtral_ep_compute_fn,
    "qwen2moe_shared_expert_gate": qwen2moe_ep_compute_fn,
    "deepseekv3_sigmoid_group_shared": deepseekv3_ep_compute_fn,
}
