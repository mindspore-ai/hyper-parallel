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
"""apply: public entry point for applying a ShardingPlan (05 §4 canonical).

apply_sharding_plan: Phase 0 normalization -> A parameter sharding -> B special
handlers -> C entry unpack + source_shard_info -> C forward wrapping
(production/validate/moe/cp/vocab_embed, five paths) -> D tied weights.

Dual-mode architecture constraint (05 §1.4): production has zero DTensor
dispatch (build-time unpack + PrecompiledBoundary); the only difference between
validate and production is the boundary stitching method -- for any module whose
DTensor dispatch hides data-dependent logic (embedding mask / attention K/V
gather / MoE all-to-all), both modes explicitly reconstruct it with the same
local-region wrapper (D-01''/D-02/D-03').

This module is a thin public facade: the implementation lives in
``_builder/applier.py`` (preflight/mesh/phase orchestration),
``_builder/parameter_sharding.py`` (parameter localize/stack/placement) and
``_builder/forward_rewriter.py`` (forward mutation).
"""

from typing import Any, Optional, Tuple

from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import (
    _normalize_out_fields,
)
from hyper_parallel.distributed._builder.applier import (
    _apply_phase_c,
    _apply_plan_special_handlers,
    _get_active_mesh,
    _get_tp_submesh,
    _preflight_compute_injection,
)
from hyper_parallel.distributed._builder.parameter_sharding import (
    _build_runtime_source_shard_info,
    _replicate_tied_weights,
    _resolve_parameter_source_meshes,
    _shard_planned_parameters,
    detect_tied_weights,
)


def apply_sharding_plan(
    model: Any,
    plan: ShardingPlan,
    mesh: Any,
    *,
    validate_mode: bool = False,
) -> Tuple[Any, Optional[Any]]:
    """Apply a ShardingPlan using a DeviceMesh or MeshContext.

    Args:
        model: The model to shard (an HF-style ``nn.Module``, or a list of
            per-part models in PP scenarios).
        plan: The :class:`ShardingPlan` produced by the planner.
        mesh: A ``DeviceMesh`` or a :class:`MeshContext` carrying one.
        validate_mode: When ``True``, keep parameters as DTensors and wrap
            forwards for placement-propagation validation instead of the
            production local-tensor path.

    Returns (model, source_shard_info):
    - production: at the Phase C entry, a one-shot `_local_params_context` permanently
      unwraps DTensor parameters into plain local tensors, and builds source_shard_info
      for fully_shard to use. Entries record the complete source layout:
      ``{param_fqn: (placements_tuple, source_sub_mesh)}`` with one placement per
      non-FSDP source axis (dense entries derive from the dense-FSDP mesh, routed
      experts from the expert mesh);
    - validate: no unwrap (parameters remain DTensors); source_shard_info is None.
    """
    # Lazy import: components.distributed.__init__ re-exports this module, so
    # a module-level import of the mesh module would cycle through the
    # package __init__.
    from hyper_parallel.distributed.mesh import (  # pylint: disable=C0415
        MeshContext,
    )
    mesh_context = mesh if isinstance(mesh, MeshContext) else None
    if mesh_context is None:
        device_mesh = mesh
    else:
        device_mesh = mesh_context.device_mesh
    if device_mesh is None:
        raise ValueError("apply_sharding_plan requires a DeviceMesh")

    mesh_dim_names = plan.mesh_dim_names
    # Active sub-mesh: the planner strips size=1 axes (plan.mesh_dim_names), but the
    # passed-in mesh may still contain those axes -- placements are resolved against
    # plan.mesh_dim_names, so the dimensionality must align with the mesh, otherwise
    # distribute_tensor will silently shard along the wrong axis.
    full_mesh = device_mesh
    mesh = _get_active_mesh(device_mesh, mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
    models = model if isinstance(model, list) else [model]

    # Explicit-injection guard: CP/EP sharding without an explicit compute
    # injection fails fast here, BEFORE any parameter is touched
    _preflight_compute_injection(plan, mesh, model=models[0])

    expert_mesh, dense_source_mesh, expert_source_mesh = (
        _resolve_parameter_source_meshes(plan, mesh_context, full_mesh, tp_mesh)
    )

    # ====== Phase 0: normalize out_src/out_dst scalar shorthand (idempotent, covers user-injected paths) ======
    for spec in plan.modules.values():
        _normalize_out_fields(spec)

    # ====== Phase A: parameter sharding ======
    _shard_planned_parameters(models, plan, mesh, expert_mesh, validate_mode)

    # ====== Phase B: special handlers ======
    _apply_plan_special_handlers(models, plan, mesh)

    # ====== Phase C entry: one-shot unpack at build time (production only) ======
    source_shard_info = _build_runtime_source_shard_info(
        models,
        plan,
        dense_source_mesh,
        expert_source_mesh,
        validate_mode,
    )

    # ====== Phase C: wrap forward ======
    for part in models:
        _apply_phase_c(part, plan, mesh, validate_mode, expert_mesh=expert_mesh)

    # ====== Phase D: tied weights ======
    tied_pairs = list(plan.tied_pairs) or detect_tied_weights(models[0])
    for part in models:
        _replicate_tied_weights(part, tied_pairs)

    return model, source_shard_info
