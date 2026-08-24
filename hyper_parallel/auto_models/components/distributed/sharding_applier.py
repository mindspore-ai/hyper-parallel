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
"""sharding_applier: runtime application of ShardingPlan (05 §4 canonical).

apply_sharding_plan: Phase 0 normalization -> A parameter sharding -> B special
handlers -> C entry unpack + source_shard_info -> C forward wrapping
(production/validate/moe/cp/vocab_embed, five paths) -> D tied weights.

Dual-mode architecture constraint (05 §1.4): production has zero DTensor
dispatch (build-time unpack + PrecompiledBoundary); the only difference between
validate and production is the boundary stitching method -- for any module whose
DTensor dispatch hides data-dependent logic (embedding mask / attention K/V
gather / MoE all-to-all), both modes explicitly reconstruct it with the same
local-region wrapper (D-01''/D-02/D-03').
"""

import functools
import importlib.metadata
import inspect
import logging
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import (
    Placement,
    Shard,
)
from hyper_parallel.auto_models.components.distributed.cp_wrappers import (
    INNER_WRAPPER_REGISTRY,
    INNER_WRAPPER_REQUIREMENTS,
)
from hyper_parallel.auto_models.components.distributed.injection import (
    INNER_WRAPPER,
    LOCAL_COMPUTE,
    fill_context_kwargs,
    require_injection_meta,
    validate_local_compute_signature,
    validate_wrapped_forward,
)
from hyper_parallel.auto_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
)
from hyper_parallel.auto_models.components.distributed.tp_collective_lowering import (
    create_tp_collective_lowerer,
)
from hyper_parallel.auto_models.components.distributed.sharding.apply import (
    _get_attr_by_path,
    _local_params_context,
    _resolve_module,
    _set_param_by_path,
    _stack_moe_experts,
    _temp_local_params,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    PlacementMismatchError,
    ShardingPlan,
    _normalize_out_fields,
    resolve_placements,
)
from hyper_parallel.auto_models.components.distributed.sharding_planner import (
    SPECIAL_HANDLERS,
    ShardingPlanner,
)
from hyper_parallel.auto_models.components.distributed.ep_compute import (
    EP_ARCHETYPE_SUGGESTIONS,
)
from hyper_parallel.auto_models.components.distributed.source_shard import build_source_shard_info
from hyper_parallel.auto_models.components.distributed.head_count import (
    maybe_update_head_counts,
)
from hyper_parallel.auto_models.components.distributed.infrastructure import MeshContext
from hyper_parallel.auto_models.components.distributed.local_compute_context import (
    install_local_compute_forward_adapters,
    local_compute_context,
)

logger = logging.getLogger(__name__)


def _is_delayed_target(obj) -> bool:
    """Duck-typed check for a config Target (avoids a components -> trainer
    import): a Target carries ``build()`` and ``_target_`` and is itself NOT
    the compute/wrapper callable — it must be built at apply time."""
    return hasattr(obj, "build") and hasattr(obj, "_target_")


def _check_target_config_keys(target, kind):
    """Fail fast on configured Target kwargs the callable would never bind.

    Target kwargs are bound BY NAME at build time
    (``{**configured, **runtime}`` -> ``fn(**kwargs)``). When the target
    callable accepts VAR_KEYWORD (e.g. ``**_context``, there to tolerate the
    framework-filled generic context), a misspelled configured key is
    swallowed SILENTLY — the user's value never takes effect and the
    framework may even auto-fill the intended parameter instead. Guard: any
    configured key that is not an explicitly declared (keyword-bindable)
    parameter fails fast with the valid parameter names. A callable whose
    signature cannot be introspected skips the check (the call itself will
    surface any mismatch).
    """
    configured = getattr(target, "_kwargs", None)
    if not configured:
        return
    fn = getattr(target, "_target_", None)
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return
    bindable = {
        name for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = sorted(set(configured) - bindable)
    if unknown:
        raise ValueError(
            f"{kind} Target {getattr(target, '_target_path', fn)!r} "
            f"configures undeclared keys {unknown} — Target kwargs are "
            f"bound by name to the keyword parameters of the target "
            f"callable; these keys are not in the explicit parameter list "
            f"of {getattr(fn, '__name__', fn)} and would be silently "
            f"swallowed by **kwargs without taking effect (suspected "
            f"typo). Valid parameters: {sorted(bindable) or '(none)'}")


def _preflight_compute_injection(plan, mesh, model=None):
    """Fail-fast BEFORE any mutation: CP/EP sharding without an explicit
    compute injection is a silent numerical error (no auto-injection since
    the explicit-injection rework).

    - CP: an attention boundary (``_needs_cp_attn`` metadata from the
      template) under an active cp mesh needs ``inner_wrapper``;
    - EP: a TP-extend-EP boundary (``_ep_size > 0``, expert params already
      destined for ``{EP: Shard(0)}``) needs ``local_compute_fn`` — or an
      explicit ``region_dispatch=False`` when the module's own forward is
      EP-aware (a2a inside forward).

    ``model`` (optional) enables the arch-aware EP archetype suggestion in
    the error message (accuracy_fix_plan.md §3 E2).
    """
    transformers_version = importlib.metadata.version("transformers")
    cp_mesh = _get_cp_submesh(mesh, plan.mesh_dim_names)
    for fqn, spec in plan.modules.items():
        wrapper = getattr(spec, "inner_wrapper", None)
        target = getattr(spec, "inner_target", None)
        region_dispatch = getattr(spec, "region_dispatch", None)
        if wrapper is None and target is not None:
            raise ValueError(
                f"Invalid inner-wrapper plan at boundary {fqn!r} "
                f"(transformers={transformers_version}): inner_target={target!r} "
                "locates a module but inner_wrapper is missing. Declare both "
                "fields or remove inner_target."
            )
        if wrapper is not None and target is None:
            raise ValueError(
                f"Invalid inner-wrapper plan at boundary {fqn!r} "
                f"(transformers={transformers_version}): "
                f"inner_wrapper={wrapper!r}, inner_target=None, "
                f"region_dispatch={region_dispatch!r}. Declare "
                "inner_target='self' for the boundary module or an explicit "
                "child attribute name."
            )
        requirements = (
            INNER_WRAPPER_REQUIREMENTS.get(wrapper)
            if isinstance(wrapper, str) else None
        )
        if requirements is not None:
            required_dispatch = requirements["region_dispatch"]
            if region_dispatch is not required_dispatch:
                raise ValueError(
                    f"Invalid built-in CP wrapper plan at boundary {fqn!r} "
                    f"(transformers={transformers_version}): "
                    f"inner_wrapper={wrapper!r} contains CP communication and "
                    f"requires region_dispatch={required_dispatch}, got "
                    f"{region_dispatch!r}. Suggested YAML:\n"
                    f"  - match: {fqn!r}\n"
                    "    when: cp\n"
                    f"    region_dispatch: {str(required_dispatch).lower()}\n"
                    f"    inner_target: {target!r}\n"
                    f"    inner_wrapper: {wrapper}"
                )
            if requirements["requires_cp"] and (
                    cp_mesh is None or cp_mesh.size() <= 1):
                raise ValueError(
                    f"Invalid built-in CP wrapper plan at boundary {fqn!r}: "
                    f"inner_wrapper={wrapper!r} requires an active cp mesh, "
                    "but the plan has no cp axis with size > 1"
                )
        if wrapper is not None and target != "self" and getattr(
                spec, "inner_out_src", None) is None:
            raise ValueError(
                f"Invalid inner-wrapper plan at boundary {fqn!r}: wrapper "
                f"{wrapper!r} targets child {target!r}, but inner_out_src is "
                "missing. Declare inner_out_src='first_input' for a "
                "layout-preserving single output, provide explicit output "
                "placements, or use inner_target='self' to reuse boundary "
                "out_src."
            )
    if cp_mesh is not None and cp_mesh.size() > 1:
        for fqn, spec in plan.modules.items():
            if (spec.is_boundary and getattr(spec, "_needs_cp_attn", False)
                    and getattr(spec, "inner_wrapper", None) is None):
                raise ValueError(
                    f"cp_size={cp_mesh.size()} is active, so attention "
                    f"boundary {fqn!r} needs a CP-aware inner forward "
                    "(K/V all-gather), but no inner_wrapper is declared — "
                    "the framework no longer picks one heuristically. "
                    "Inject explicitly:\n"
                    "  plan_overrides:\n"
                    "    - match: \"*.self_attn\"\n"
                    "      when: cp\n"
                    "      region_dispatch: false   # the wrapper contains "
                    "communication; must not dispatch\n"
                    "      inner_wrapper:\n"
                    "        _target_: hyper_parallel.auto_models.components.distributed."
                    "cp_wrappers.sdpa_hf_cp_wrapper\n"
                    f"(the registry {sorted(INNER_WRAPPER_REGISTRY)} can be "
                    "referenced by str name; use sdpa_qkv for the "
                    "NeMo-style (q,k,v) signature, sdpa_hf for the "
                    "HF-style forward(hidden_states); or provide a custom "
                    "callable/Target implementation)")
    for fqn, spec in plan.modules.items():
        if (spec.is_boundary and getattr(spec, "_ep_size", 0)  # pylint: disable=protected-access
                and getattr(spec, "local_compute_fn", None) is None
                and getattr(spec, "region_dispatch", None) is not False):
            suggestion = ""
            if model is not None:
                arch = ShardingPlanner._get_architecture(model)
                archetype = EP_ARCHETYPE_SUGGESTIONS.get(arch)
                if archetype is not None:
                    suggestion = (
                        f"Detected model architecture {arch!r}; the "
                        f"matching archetype is likely {archetype!r} (a "
                        "suggestion, not an automatic choice — confirm and "
                        "configure it explicitly; a wrong pick fails the "
                        "apply-time interface assertion and lists the "
                        "module's actual submodule names).\n"
                    )
            raise ValueError(
                f"ep_size={spec._ep_size} is active (expert parameters "  # pylint: disable=protected-access
                f"will be sharded as {{EP: Shard(0)}}), but boundary "
                f"{fqn!r} has no local-region compute source — nothing "
                "executes the expert compute and all-to-all, and the "
                "framework no longer injects any implementation "
                "automatically. Choose one of:\n"
                f"{suggestion}"
                "  ① Pick a built-in archetype factory per the model's "
                "behavior (full semantics: router / shared expert / gate / "
                "merge all implemented cohesively; see the ep_compute.py "
                "module docstring for the available archetypes and the "
                "module interface each one expects):\n"
                "     plan_overrides:\n"
                "       - match: \"*.mlp\"\n"
                "         when: ep\n"
                "         region_dispatch: false   # a2a is inside the "
                "region; must not dispatch\n"
                "         local_compute_fn:\n"
                "           _target_: hyper_parallel.auto_models.components.distributed."
                "ep_compute.qwen2moe_ep_compute_fn   # pick per the "
                "archetype table\n"
                "  ② Atypical MoE → write your own factory following "
                "examples/distributed/ep_factories.py (use require_attrs "
                "for the same build-time interface validation)\n"
                "  ③ In-house EP-aware MoE (all-to-all already inside "
                "forward) → declare region_dispatch: false")


def _require_region_dispatch(spec, *, source):
    """Injection discipline: declaring an injection (local_compute_fn /
    inner_wrapper) requires an explicit region_dispatch (no default — pass
    True for a dispatchable pure-ops injection, False for one containing
    communication primitives / custom kernels; the tutorials and examples
    explain the reasons case by case)."""
    rd = getattr(spec, "region_dispatch", None)
    has_injection = (getattr(spec, "local_compute_fn", None) is not None
                     or getattr(spec, "inner_wrapper", None) is not None)
    if rd is None:
        if has_injection:
            raise ValueError(
                f"{source}: an injection is declared but region_dispatch "
                "is not explicitly declared (no default) — if the "
                "injected fn is pure standard ops that validate can "
                "dispatch through (fused-op/scripting-style optimizations) "
                "→ region_dispatch=True (in-region strategy propagation + "
                "true out_src validation); if it contains communication "
                "primitives/custom kernels (CP K/V all-gather, EP "
                "all-to-all, quantized GEMM, etc.) → "
                "region_dispatch=False (skeleton/adapter black-box hosted "
                "local execution + declarative rewrap)")
        return
    if rd is True and not has_injection:
        raise ValueError(
            f"{source}: region_dispatch=True but no injection is declared "
            "— an ordinary boundary's forward dispatches through natively "
            "(the axiomatic default), so this declaration is redundant; "
            "remove it")


def _log_injection_choice(module_fqn, spec):
    """Observability enhancement: the resolution chosen for an injection
    boundary is immediately visible (one INFO line per boundary), closing
    the "declare → see the consequence" feedback loop. Must be called after
    _require_region_dispatch (the injection + region_dispatch combination is
    already validated as legal at this point)."""
    rd = getattr(spec, "region_dispatch", None)
    has_fn = getattr(spec, "local_compute_fn", None) is not None
    has_wrap = getattr(spec, "inner_wrapper", None) is not None
    if not (has_fn or has_wrap or rd is not None):
        return   # ordinary boundary: no injection, axiomatic default dispatch-through — do not spam the log
    what = "+".join(
        [x for x, ok in (("local_compute_fn", has_fn),
                         ("inner_wrapper", has_wrap)) if ok]
    ) or "the module's own forward (no fn injected, region_dispatch=False declared)"
    if rd is True:
        effect = ("validate dispatch-through true validation enabled "
                  "(in-region strategy propagation + true out_src "
                  "validation)")
    else:  # False (None + injection was already intercepted by _require_region_dispatch)
        effect = ("black-box hosting (local execution inside the region, "
                  "propagation checks skipped, declarative rewrap)")
    logger.info(
        "boundary %s: injection[%s], region_dispatch=%s → %s",
        module_fqn, what, rd, effect)


def _resolve_parameter_source_meshes(plan, mesh_context, full_mesh, tp_mesh):
    """Resolve dense TP and routed-expert source meshes for one sharding plan."""
    ep_size = next(
        (
            spec._ep_size  # pylint: disable=protected-access
            for spec in plan.modules.values()
            if spec._ep_size > 0  # pylint: disable=protected-access
        ),
        0,
    )
    if mesh_context is not None:
        expert_mesh = mesh_context.fsdp_moe_mesh
    elif ep_size > 0:
        expert_mesh = _build_expert_mesh(
            full_mesh,
            full_mesh.mesh_dim_names,
            ep_size,
        )
    else:
        expert_mesh = None
    if ep_size > 0 and expert_mesh is None:
        raise ValueError("Routed expert plan requires MeshContext.fsdp_moe_mesh")
    if expert_mesh is not None:
        logger.info(
            "expert mesh: using %s for parameter sharding, explicit compute injection, "
            "and FSDP source metadata",
            dict(zip(tuple(expert_mesh.mesh_dim_names), tuple(expert_mesh.mesh_shape))),
        )
    # Pass the full region meshes: build_source_shard_info strips the FSDP-owned
    # axes itself and records the complete source (non-FSDP) layout.
    dense_source_mesh = (
        mesh_context.fsdp_non_moe_mesh
        if mesh_context is not None and mesh_context.fsdp_non_moe_mesh is not None
        else tp_mesh
    )
    expert_source_mesh = (
        expert_mesh
        if expert_mesh is not None and "ep" in expert_mesh.mesh_dim_names
        else None
    )
    return expert_mesh, dense_source_mesh, expert_source_mesh


def _shard_planned_parameters(models, plan, mesh, expert_mesh, validate_mode):
    """Shard dense and expert parameters, then update local attention metadata."""
    for model in models:
        for module_fqn, spec in plan.modules.items():
            module = _resolve_module(model, module_fqn)
            # Runtime planner metadata is internal to the sharding pipeline.
            if spec._ep_stack:  # pylint: disable=protected-access
                _stack_moe_experts(
                    module,
                    spec._ep_stack,  # pylint: disable=protected-access
                )
            if spec._ep_size > 0:  # pylint: disable=protected-access
                expert_params = {
                    name: placement
                    for name, placement in spec.params.items()
                    if name.startswith("experts.")
                }
                dense_params = {
                    name: placement
                    for name, placement in spec.params.items()
                    if not name.startswith("experts.")
                }
                _shard_module_params(
                    module,
                    expert_params,
                    expert_mesh,
                    expert_mesh.mesh_dim_names,
                )
                _shard_module_params(
                    module,
                    dense_params,
                    mesh,
                    plan.mesh_dim_names,
                )
            else:
                _shard_module_params(
                    module,
                    spec.params,
                    mesh,
                    plan.mesh_dim_names,
                )
            if not validate_mode:
                maybe_update_head_counts(
                    module,
                    spec,
                    module_fqn,
                    mesh,
                    plan.mesh_dim_names,
                )


def _apply_plan_special_handlers(models, plan, mesh):
    """Run parameter handlers declared by the sharding plan."""
    for model in models:
        for param_ref, handler_name in plan.special_handlers.items():
            handler = SPECIAL_HANDLERS.get(handler_name)
            if handler is None:
                logger.warning(
                    "SPECIAL_HANDLERS has no registered handler: %s",
                    handler_name,
                )
                continue
            module_fqn, param_name = param_ref.rsplit(".", 1)
            handler(_resolve_module(model, module_fqn), param_name, mesh)


def _build_runtime_source_shard_info(
    models,
    plan,
    dense_source_mesh,
    expert_source_mesh,
    validate_mode,
):
    """Unwrap production parameters and build FSDP source-layout metadata."""
    if validate_mode:
        return None
    source_shard_records = {}
    for model in models:
        source_shard_records.update(_local_params_context(model))
    if not source_shard_records:
        return None
    if dense_source_mesh is None and expert_source_mesh is None:
        return None
    return build_source_shard_info(
        plan,
        dense_source_mesh,
        expert_source_mesh=expert_source_mesh,
    )


# ────────────────────────────────────────────────────────────────────────────
# Main entry (05 §4.1)
# ────────────────────────────────────────────────────────────────────────────

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


def _get_active_mesh(mesh, mesh_dim_names):
    """Return the active sub-mesh aligned with plan.mesh_dim_names (the dimension set after stripping size=1 axes)."""
    names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
    if names == tuple(mesh_dim_names):
        return mesh
    if mesh_dim_names and names and all(n in names for n in mesh_dim_names):
        return mesh[tuple(mesh_dim_names)]
    return mesh


def _get_tp_submesh(mesh, mesh_dim_names):
    if "tp" not in mesh_dim_names:
        return None
    return mesh["tp"]


def _get_cp_submesh(mesh, mesh_dim_names):
    if "cp" not in mesh_dim_names:
        return None
    return mesh["cp"]


def _get_ep_submesh(mesh, mesh_dim_names):
    if "ep" not in mesh_dim_names:
        return None
    return mesh["ep"]


def _expert_mesh_layout(mesh, mesh_dim_names, ep_size):
    """(shape, dim_names, rank_list) of the derived expert mesh (pure mapping; no process group is created).

    D-10 TP-extend-EP (05 §6.4.8 / 06 §4.5.1): the expert domain = the full
    dense region (all ranks on the non-pp axes of the mesh, i.e.
    dp_replicate x dp_cp x tp). After a row-major flatten in mesh axis order,
    it is re-sliced as (edp = D/ep_size, ep = ep_size):
    - EP groups (inner, the a2a communication domain): ep_size consecutive
      ranks in flatten order -- tp is usually the innermost axis, so an EP
      group first spans the entire TP group and then extends to adjacent
      dp/cp ranks (isomorphic to MindSpeed TP-extend-EP / Megatron etp=1 with
      ep spanning TP).
      Example: mesh (dp=4, tp=2), ep_size=4 -> EP groups {0,1,2,3} / {4,5,6,7};
    - edp groups (outer): expert data-parallel degree = D/ep_size.
    Expert weights are only Shard(0) on the ep axis (the expert dim); there is
    no second-axis sharding.
    """
    if "pp" in mesh_dim_names:
        raise NotImplementedError(
            "D-10 TP-extend-EP v1 does not yet support the pp axis (call after splitting the mesh by stage)")
    arr = np.array(mesh.rank_list).reshape(mesh.mesh_shape)
    domain = int(np.prod(arr.shape))
    if ep_size <= 0 or domain % ep_size != 0:
        raise ValueError(
            f"ep_size ({ep_size}) must divide the dense region ({domain})"
        )
    edp = domain // ep_size
    derived = arr.reshape(edp, ep_size)
    return (edp, ep_size), ("edp", "ep"), tuple(
        int(r) for r in derived.flatten())


def _build_expert_mesh(mesh, mesh_dim_names, ep_size):
    """D-10 (05 §6.4.8 / 06 §4.5.1): repartition the full dense region into the derived expert mesh (edp, ep)."""
    shape, names, rank_list = _expert_mesh_layout(mesh, mesh_dim_names, ep_size)
    # Propagate the no-backend (metadata-only) mode of the source mesh —
    # a meta mesh has no _dim_group_names (no process groups were created).
    return init_device_mesh(mesh.device_type, shape, mesh_dim_names=names,
                            rank_list=rank_list,
                            init_backend=hasattr(mesh, "_dim_group_names"))


def build_expert_mesh(mesh: DeviceMesh, ep_size: int) -> DeviceMesh:
    """Public: derive the D-10 TP-extend-EP expert mesh (edp, ep) from *mesh*
    (the FULL mesh — the dense region must include dp/cp axes).

    Standalone helper for introspection and for custom code that needs the
    expert domain outside the injection path. Injected factories/wrappers do
    NOT need to call this: the framework derives the expert mesh once at
    apply time (shared by parameter sharding and injected compute) and hands
    it to them as the ``ep_mesh`` context.

    Args:
        mesh: The full device mesh (dense region including dp/cp axes).
        ep_size: The expert-parallel group size.

    Returns:
        The derived ``(edp, ep)`` expert mesh.
    """
    return _build_expert_mesh(mesh, tuple(mesh.mesh_dim_names), ep_size)


# ────────────────────────────────────────────────────────────────────────────
# Phase A: parameter sharding (05 §4.2)
# ────────────────────────────────────────────────────────────────────────────

def _shard_module_params(module, param_specs, mesh, mesh_dim_names):
    """distribute_tensor() converts parameters into DTensors.

    - meta tensor -> DTensor: _local_tensor remains meta (zero-memory path);
    - real tensor -> DTensor: physically split; each rank holds a local shard;
    - already a DTensor: skipped if the placement matches, otherwise raises
      PlacementMismatchError.
    """
    for param_path, named in param_specs.items():
        param = _get_attr_by_path(module, param_path)
        placements = tuple(resolve_placements(named, mesh_dim_names))
        if not placements:
            continue  # no active DTensor axes (all size 1) -- no sharding needed

        if isinstance(param, DTensor):
            if tuple(param.placements) != placements:
                raise PlacementMismatchError(
                    f"{type(module).__name__}.{param_path}",
                    placements, tuple(param.placements), "params",
                )
            continue

        src = param.data if hasattr(param, "data") else param
        dt = distribute_tensor(src, mesh, placements)
        requires_grad = getattr(param, "requires_grad", True)
        _set_param_by_path(module, param_path,
                           nn.Parameter(dt, requires_grad=requires_grad))


# ────────────────────────────────────────────────────────────────────────────
# Phase C: forward wrapping (05 §4.4)
# ────────────────────────────────────────────────────────────────────────────

def _install_bias_suppression(module, spec):
    """D-22: make each defer-listed Linear run bias-free inside the region.

    The bias Parameter itself is never touched (same FQN / same object /
    state_dict / optimizer unchanged); the wrapper only hides it from
    ``F.linear`` for the duration of the region forward, so the boundary
    exit's Partial reduction sees a pure matmul contribution and the bias is
    added exactly once afterwards by :func:`_maybe_add_deferred_biases`.
    Both modes install the same wrapper (instruction-for-instruction
    identity, D-01''); nesting the suppression is idempotent.
    """
    for param_path in spec._deferred_bias_params:  # pylint: disable=protected-access
        owner_path = param_path.rpartition(".")[0]
        owner = module.get_submodule(owner_path) if owner_path else module
        original = owner.forward

        @functools.wraps(original)
        def bias_free_forward(
            *args: Any,
            __original: Callable[..., Any] = original,
            __owner: nn.Module = owner,
            **kwargs: Any,
        ) -> Any:
            """Run the owner's forward with its bias temporarily hidden.

            The bias Parameter object is restored on exit (even on error), so
            state_dict/optimizer visibility is unchanged; only ``F.linear``
            inside the region sees a bias-free Linear.
            """
            bias = __owner.bias
            try:
                __owner._parameters["bias"] = None  # pylint: disable=protected-access
                return __original(*args, **kwargs)
            finally:
                __owner._parameters["bias"] = bias  # pylint: disable=protected-access

        owner.forward = bias_free_forward


def _add_bias_to_primary_output(output, bias, module_name):
    """Add a deferred bias to the primary Tensor while preserving output structure."""
    if isinstance(output, torch.Tensor):
        primary_output = output
        rebuild = None
    elif isinstance(output, (tuple, list)):
        if not output or not isinstance(output[0], torch.Tensor):
            primary_type = type(output[0]).__name__ if output else "missing"
            raise TypeError(
                f"{module_name}: deferred bias requires output index 0 to be a Tensor, "
                f"got {primary_type}"
            )
        primary_output = output[0]
        rebuild = list(output)
    else:
        raise TypeError(
            f"{module_name}: deferred bias requires a Tensor, tuple, or list output, "
            f"got {type(output).__name__}"
        )

    if not isinstance(primary_output, DTensor) and isinstance(bias, DTensor):
        bias = bias.to_local()
    biased_output = primary_output + bias
    if rebuild is None:
        return biased_output
    rebuild[0] = biased_output
    return tuple(rebuild) if isinstance(output, tuple) else rebuild


def _maybe_add_deferred_biases(module, spec, output):
    """D-22: add each deferred bias exactly once AFTER the boundary exit.

    The bias is read at forward time (never captured in a closure), so
    production sees the unwrapped local tensor and validate sees the DTensor
    — whichever form the boundary output currently has: nested validate keeps
    the DTensor (dispatch add, Shard(1)+Replicate→Shard(1)); the outermost /
    production / local-region exits are local, and a DTensor bias is
    unwrapped to match. For structured attention outputs, index 0 is the
    primary hidden-state Tensor; metadata/cache entries are preserved.
    """
    if not spec._deferred_bias_params:  # pylint: disable=protected-access
        return output
    for param_path in spec._deferred_bias_params:  # pylint: disable=protected-access
        owner_path = param_path.rpartition(".")[0]
        owner = module.get_submodule(owner_path) if owner_path else module
        bias = owner.bias
        if bias is None:
            continue
        output = _add_bias_to_primary_output(output, bias, type(module).__name__)
    return output


def _apply_phase_c(model, plan, mesh, validate_mode, expert_mesh=None):
    """Phase C: wrap forward (production/validate/moe/cp/vocab_embed, five paths).

    D-14 invariant 2 (05 §13.3): boundaries are wrapped in post-order (deepest
    FQN first) — an outer boundary's local_compute_fn may cache inner forwards,
    and the unpack-scope exclusion (invariant 3) requires inner wrappers to be
    installed first.
    """
    mesh_dim_names = plan.mesh_dim_names
    cp_mesh = _get_cp_submesh(mesh, mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
    boundary_op_lowerer = (
        None
        if validate_mode
        else create_tp_collective_lowerer(mesh, mesh_dim_names)
    )
    for module_fqn, spec in sorted(
            plan.modules.items(), key=lambda kv: -kv[0].count(".")):
        if not spec.is_boundary:
            continue
        module = _resolve_module(model, module_fqn)
        boundary = PrecompiledBoundary(
            spec,
            mesh,
            mesh_dim_names,
            op_lowerer=boundary_op_lowerer,
        )
        keep_output_dtensor = _keep_loss_parallel_output(plan, spec)
        _bind_input_indices(boundary, module)
        # Injection discipline: a declared injection requires an explicit
        # region_dispatch; declaring True without any injection is redundant
        _require_region_dispatch(spec, source=f"boundary {module_fqn!r}")
        # Observability: the injection choice is immediately visible (the
        # "declare → consequence" feedback loop)
        _log_injection_choice(module_fqn, spec)

        # Step 0.5 (D-22): rowwise bias defer — install the bias-free region
        # forward BEFORE inner-wrap/forward wrapping, in BOTH modes (the
        # in-region computation stays instruction-for-instruction identical).
        # The bias Parameter itself is untouched; it is added once after the
        # boundary exit reduction by _maybe_add_deferred_biases.
        if spec._deferred_bias_params:  # pylint: disable=protected-access
            _install_bias_suppression(module, spec)

        # Step 1: inner-wrap — the generic "weave into / replace the inner
        # forward" mechanism
        # (D-01'': production and validate inject the same wrapper, so the
        # in-region computation is instruction-for-instruction identical).
        # NOT gated on cp_mesh since the generalization: the derived gate is
        # the resolution chain itself (explicit inner_wrapper declaration ->
        # applied; nothing declared -> None -> no-op). cp_mesh may be None
        # (no cp axis) — the four shipped CP wrappers self-guard and fail
        # fast then; custom callables/Targets receive None and own their
        # semantics.
        # (_preflight_compute_injection has already failed fast when a CP
        # attention boundary declares nothing)
        _wrap_inner_attention(
            module, cp_mesh, spec=spec, mesh=mesh,
            mesh_dim_names=mesh_dim_names, tp_mesh=tp_mesh,
            ep_mesh=expert_mesh, validate_mode=validate_mode,
        )

        # Step 2: forward wrapping
        # local region path (D-03'): the gate is derived from the compute_fn
        # resolution chain (non-None means take the skeleton) — the remaining
        # sources after the explicit-injection rework are user
        # local_compute_fn (callable or factory Target) and the
        # region_dispatch=False gate; the built-in EP auto-injection link
        # was REMOVED (05 §4.4.3)
        compute_fn = _resolve_local_compute_fn(
            module, spec, mesh, mesh_dim_names, expert_mesh)
        if compute_fn is not None:
            if validate_mode:
                # D-17: inside the local region the module sees local
                # tensors in both modes -- validate rewrites cached head
                # counts only for local-region modules (boundary modules
                # keep global counts for DTensor dispatch)
                maybe_update_head_counts(
                    module, spec, module_fqn, mesh, mesh_dim_names)
            # D-14 invariant 3 (05 §13.3): the region's temp-unwrap scope
            # excludes nested-boundary subtrees — their parameters must stay
            # DTensors for the inner validate islands (dispatch needs
            # __torch_function__)
            nested = _descendant_boundary_fqns(plan, module_fqn)
            if nested:
                # E5 (accuracy_fix_plan.md §3): make the nested-boundary call
                # contract visible at apply time
                logger.info(
                    "local-region boundary %s: nested boundaries "
                    "(communication_owner: nested_boundary): %s — their "
                    "return values are already in the declared out_dst "
                    "layout with TP communication sealed inside; the region "
                    "compute MUST NOT apply compensating collectives "
                    "(all-reduce/reduce-scatter/all-gather) to them",
                    module_fqn, nested)
            _wrap_local_region_forward(
                module, boundary, spec, mesh, mesh_dim_names,
                validate_mode=validate_mode, compute_fn=compute_fn,
                exclude_subtrees=nested)
        elif validate_mode:
            _wrap_validate_forward(
                module,
                boundary,
                spec,
                mesh_dim_names,
                keep_output_dtensor=keep_output_dtensor,
            )
        else:
            # D-02: production vocab-parallel embedding masked wrapper
            if _is_vocab_parallel_embed(module, spec, tp_mesh):
                _wrap_vocab_parallel_embedding(module, tp_mesh)
            _wrap_production_forward(
                module,
                boundary,
                spec,
                keep_output_dtensor=keep_output_dtensor,
            )


def _keep_loss_parallel_output(plan, spec):
    """Return whether a terminal vocab-sharded output must remain a DTensor."""
    if not plan.loss_parallel or not spec._is_terminal:  # pylint: disable=protected-access
        return False
    output_placements = (spec.out_dst or {}).get("output", {})
    return any(
        isinstance(placement, Shard) and placement.dim == -1
        for placement in output_placements.values()
    )


def _descendant_boundary_fqns(plan, module_fqn):
    """Relative FQNs of boundaries nested inside *module_fqn* (D-14).

    Returned relative to module_fqn (matching the name space of
    module.named_parameters(recurse=True)); the root spec (fqn "") treats
    every other boundary as a descendant.
    """
    if module_fqn == "":
        return [f for f in plan.modules if f]
    prefix = module_fqn + "."
    return [f[len(prefix):] for f in plan.modules if f.startswith(prefix)]


def _bind_input_indices(boundary, module):
    """Bind the arg_name of in_plan to the positional index of the forward signature.

    Inter-module calls are mostly positional (self.mlp(x) inside a layer), so
    RedistOp's kwargs lookup by name would miss -- the signature index is bound
    at compile time, and at runtime _get_arg checks kwargs first, then args.
    """
    try:
        sig = inspect.signature(module.forward)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        positional = [
            name for name, p in sig.parameters.items()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        name_to_idx = {name: i for i, name in enumerate(positional)}
        for op in boundary.in_plan:
            if op.arg_index is None and op.arg_name in name_to_idx:
                op.arg_index = name_to_idx[op.arg_name]
    # Positional fallback: a single-input contract (in_plan has only 1 op) is
    # bound to the first positional parameter -- covers cases where the
    # template key (e.g. "hidden_states") differs from the leaf module's
    # signature (nn.Linear.forward(input)).
    if len(boundary.in_plan) == 1 and boundary.in_plan[0].arg_index is None:
        boundary.in_plan[0].arg_index = 0


def _wrap_production_forward(
        module, boundary, spec=None, keep_output_dtensor=False):
    """Production mode: pure local tensor computation + precompiled boundary communication (05 §4.4.1).

    _local_params_context was already invoked at the Phase C entry (parameters
    permanently unpacked).
    """
    original_forward = module.forward

    @functools.wraps(original_forward)
    def production_forward(*args: Any, **kwargs: Any) -> Any:
        """Production boundary forward: precompiled entry communication,
        local computation, exit redistribution, then D-22 deferred biases."""
        args, kwargs = boundary.redistribute_inputs(args, kwargs)
        outputs = original_forward(*args, **kwargs)
        outputs = boundary.redistribute_outputs(
            outputs,
            as_dtensor_input=keep_output_dtensor,
        )
        # D-22: deferred rowwise biases — added once after the exit reduction
        return _maybe_add_deferred_biases(module, spec, outputs) \
            if spec is not None else outputs

    module.forward = production_forward


def _wrap_validate_forward(
        module, boundary, spec, mesh_dim_names, keep_output_dtensor=False):
    """Validate mode: DTensor propagation end to end -> validate out_src (core) + out_dst (terminal modules only).

    The in-house DTensor is forward-only: validation covers only forward
    placement propagation; backward is local autograd in both modes (05 §1.0),
    and gradient equivalence is guaranteed by testing/grad_equiv.py.
    """
    original_forward = module.forward
    module_name = type(module).__name__

    @functools.wraps(original_forward)
    def validate_forward(*args: Any, **kwargs: Any) -> Any:
        """Validate boundary forward: wrap inputs as DTensors, propagate
        placements through the original forward, validate out_src, then
        redistribute to out_dst."""
        # D-14 nesting (05 §13.4): detect whether the call arrives from an
        # outer DTensor-propagating boundary BEFORE Step 1 wraps everything
        # into DTensors (Step 1 would make the check useless).
        nested = any(isinstance(a, DTensor) for a in args) or any(
            isinstance(v, DTensor) for v in kwargs.values())
        # Step 1: inputs -> DTensor
        args, kwargs = boundary.redistribute_inputs(args, kwargs, as_dtensor=True)

        # Step 2: parameters stay DTensors; placement propagates via __torch_function__ dispatch
        outputs = original_forward(*args, **kwargs)

        # Step 3: [core validation] out_src -- native DTensor-propagated output vs declaration
        if spec.out_src is not None:
            _validate_out_src(outputs, spec, mesh_dim_names, module_name)

        # Step 4: redistribute to out_dst
        outputs = boundary.redistribute_outputs(outputs, as_dtensor_input=True)

        # Step 5: [defensive validation] out_dst -- terminal modules only
        if spec._is_terminal and spec.out_dst is not None:  # pylint: disable=protected-access
            _validate_out_dst(outputs, spec, mesh_dim_names, module_name)

        # D-22: deferred biases are added after the exit reduction while the
        # output is still a DTensor (Shard/Replicate + Replicate bias dispatch
        # add — the Partial reduction is already done, so every rank adds the
        # bias exactly once); Step 6 then unwraps as usual.
        outputs = _maybe_add_deferred_biases(module, spec, outputs)

        # Step 6: return local (isomorphic to production boundary outputs) --
        # but under an outer DTensor-propagating boundary (D-14 nesting, 05
        # §13.4) keep the DTensor so the outer forward's dispatch chain is
        # unbroken; the outermost boundary exit unwraps.
        if nested or keep_output_dtensor:
            return outputs
        if isinstance(outputs, DTensor):
            outputs = outputs.to_local()
        elif isinstance(outputs, (tuple, list)):
            outputs = tuple(
                t.to_local() if isinstance(t, DTensor) else t for t in outputs
            )
        return outputs

    module.forward = validate_forward


def _validate_out_src(outputs, spec, mesh_dim_names, module_name):
    _validate_outputs(outputs, spec, mesh_dim_names, module_name, "out_src")


def _validate_out_dst(outputs, spec, mesh_dim_names, module_name):
    _validate_outputs(outputs, spec, mesh_dim_names, module_name, "out_dst")


def _normalize_placements_ndim(placements, ndim):
    """Normalize negative dims like Shard(-1) against the tensor ndim (Shard(-1) == Shard(ndim-1))."""
    out = []
    for p in placements:
        if isinstance(p, Shard) and p.dim < 0:
            out.append(Shard(p.dim + ndim))
        else:
            out.append(p)
    return tuple(out)


def _validate_outputs(outputs, spec, mesh_dim_names, module_name, stage):
    """Placement validation for single/multi outputs (shared by out_src / out_dst).

    Multi outputs are mapped to tuple positions via spec.out_names (falling
    back to declaration key order); outputs that are not returned or are not
    DTensors are skipped. Declared and actual placements are
    negative-dim-normalized before comparison.
    """
    declared = getattr(spec, stage)
    if isinstance(outputs, (tuple, list)):
        out_names = getattr(spec, "out_names", None) or list(declared.keys())
        name_to_idx = {name: i for i, name in enumerate(out_names)}
        items = list(outputs)
    else:
        name_to_idx = {name: 0 for name in declared}
        items = [outputs]
    for out_name, expected_named in declared.items():
        idx = name_to_idx.get(out_name)
        if idx is None or idx >= len(items):
            continue
        tensor = items[idx]
        if not isinstance(tensor, DTensor):
            continue
        ndim = len(tensor.shape)
        expected = _normalize_placements_ndim(
            tuple(resolve_placements(expected_named, mesh_dim_names)), ndim)
        actual = _normalize_placements_ndim(tuple(tensor.placements), ndim)
        if expected != actual:
            suffix = f"[{out_name}]" if len(declared) > 1 else ""
            raise PlacementMismatchError(
                module_name, expected, actual, f"{stage}{suffix}"
            )


# ────────────────────────────────────────────────────────────────────────────
# Phase C: MoE EP local region (05 §4.4.3 + D-03')
# ────────────────────────────────────────────────────────────────────────────

def _build_local_compute_factory(factory, module, mesh, mesh_dim_names,
                                 expert_mesh, *, configured=None, source):
    """Build a ``@local_compute`` factory into the region compute_fn (apply-time).

    The factory is invoked ONCE with the framework context filled BY NAME per
    its declared context (``meta.context``): the mesh family ``mesh`` /
    ``tp_mesh`` / ``cp_mesh`` / ``ep_mesh`` (mandatory declarations; None when
    the axis is inactive), plus the optional anchor ``module``. Context keys
    are RESERVED — a user-configured same-name key fails fast
    (fill_context_kwargs); every fill is logged at INFO. Behavior choices
    (routing, layouts, ...) are written INTO the factory — config keys carry
    data only, never functions.
    The returned compute fn is validated against the module's forward
    signature (validate_local_compute_signature: params must match the
    original forward) and bound to *module* by the caller.
    """
    meta = require_injection_meta(factory, LOCAL_COMPUTE, source=source)
    context = {
        "module": module,
        "mesh": mesh,
        "tp_mesh": _get_tp_submesh(mesh, mesh_dim_names),
        "cp_mesh": _get_cp_submesh(mesh, mesh_dim_names),
        "ep_mesh": expert_mesh,
    }
    build_kwargs = fill_context_kwargs(
        meta, context, configured or {}, source=source)
    # {**configured, **context}: same binding order as Target.build (context
    # keys are reserved names; fill_context_kwargs already rejects same-name
    # keys in configured)
    compute_fn = factory(**{**(configured or {}), **build_kwargs})
    if not callable(compute_fn):
        raise TypeError(
            f"{source} returned {type(compute_fn).__name__}, not callable — "
            "the @local_compute factory must return the region compute fn "
            "fn(module, *local_args) (e.g. hyper_parallel.auto_models.components.distributed."
            "ep_compute.qwen2moe_ep_compute_fn)")
    validate_local_compute_signature(
        compute_fn, module.forward,
        owner=f"{source!r} on {type(module).__name__}")
    return compute_fn


def _resolve_local_compute_fn(module, spec, mesh, mesh_dim_names,
                              expert_mesh):
    """Resolve the compute_fn of the local region (**single resolution chain**, 05 §4.4.3).

    Whether a module takes the local-region skeleton is derived by this chain
    (a non-None return means it does) -- the gate is not a stored bool but the
    resolution result. Since the explicit-injection rework the built-in EP
    auto-injection link is REMOVED; the remaining sources:
    1. spec.local_compute_fn: a user-defined FACTORY — single form since
       2026-08-10 (the direct compute-fn form was retired: every injection
       fn is invoked once at apply time with the mesh family filled by the
       framework, use-it-or-not being the user's choice — same discipline as
       @inner_wrapper). Accepted shapes: a ``@local_compute``-decorated
       factory callable (programmatic direct pass), or a Target wrapping one
       (config keys / YAML ``_target_`` reference); both are built at apply
       time by _build_local_compute_factory (e.g. a shipped EP archetype
       factory such as ep_compute.qwen2moe_ep_compute_fn) — undecorated
       functions fail fast
       (injection discipline); the returned compute fn's params are
       validated against the module's forward (params must match the
       original function). Declaring it REQUIRES an explicit
       ``region_dispatch`` (no default — True: dispatchable pure-ops
       injection, validate dispatches through it and truly validates
       out_src; False: comm/custom-kernel injection, the skeleton runs it
       as a black box on local tensors);
    2. spec.region_dispatch=False without inner_wrapper: the module's own
       forward cannot dispatch (it IS the data-dependent logic, e.g. an
       EP-aware in-house MoE with the a2a already inside forward) — the
       skeleton runs it on local tensors. An explicit inner_wrapper owns the
       local computation instead and therefore does not select this path;
    3. none of the above -> None (ordinary module; takes the
       validate/production path — and an EP-sharded boundary hitting this
       was already failed fast by _preflight_compute_injection).
    """
    custom = getattr(spec, "local_compute_fn", None)
    if custom is not None:
        _require_region_dispatch(spec, source="spec.local_compute_fn")
        if _is_delayed_target(custom):
            _check_target_config_keys(custom, "local_compute_fn")
            factory = getattr(custom, "_target_", None)
            source = (f"local_compute_fn factory "
                      f"{getattr(custom, '_target_path', custom)}")
            configured = getattr(custom, "_kwargs", {})
        else:
            factory = custom
            source = "spec.local_compute_fn"
            configured = None
        compute_fn = _build_local_compute_factory(
            factory, module, mesh, mesh_dim_names, expert_mesh,
            configured=configured, source=source)
        return functools.partial(compute_fn, module)
    if spec.region_dispatch is False and spec.inner_wrapper is None:
        return module.forward
    return None


def _rewrap_local_outputs(output, spec, mesh, mesh_dim_names, module_name):
    """Wrap every declared local Tensor output with its ``out_src`` layout."""
    declared = spec.out_src or {}
    if not declared:
        return output

    is_sequence = isinstance(output, (tuple, list))
    items = list(output) if is_sequence else [output]
    out_names = list(getattr(spec, "out_names", None) or declared.keys())
    name_to_idx = {name: index for index, name in enumerate(out_names)}

    for out_name, named_placement in declared.items():
        index = name_to_idx.get(out_name)
        if index is None:
            raise ValueError(
                f"{module_name}: out_src declares output {out_name!r}, but "
                f"out_names={out_names!r} does not contain it"
            )
        if index >= len(items):
            raise ValueError(
                f"{module_name}: out_src maps output {out_name!r} to index "
                f"{index}, but forward returned only {len(items)} output(s)"
            )
        item = items[index]
        if item is None:
            continue
        if isinstance(item, DTensor):
            continue
        if not isinstance(item, torch.Tensor):
            raise TypeError(
                f"{module_name}: declared output {out_name!r} at index "
                f"{index} must be a Tensor or None, got {type(item).__name__}"
            )
        placements = tuple(resolve_placements(named_placement, mesh_dim_names))
        items[index] = DTensor.from_local(item, mesh, placements)

    if isinstance(output, tuple):
        return tuple(items)
    if isinstance(output, list):
        return items
    if len(items) != 1:
        raise ValueError(
            f"{module_name}: scalar forward output cannot satisfy "
            f"{len(declared)} declared out_src entries"
        )
    return items[0]


def _wrap_local_region_forward(module, boundary, spec, mesh, mesh_dim_names,
                               *, validate_mode=False, compute_fn=None,
                               exclude_subtrees=()):
    """Generic local-region forward wrapper (D-03', formerly the _wrap_moe_forward skeleton).

    Structure: boundary entry -> local region -> re-wrap per the declared
    out_src -> boundary exit. Applies to any module containing data-dependent
    logic that DTensor dispatch cannot express (e.g. MoE all-to-all) --
    injected by _apply_phase_c when _resolve_local_compute_fn resolves to
    non-None (derived gate, 05 §4.4.3).

    production: parameters were permanently unpacked at build time and inputs
    are local (boundary passthrough); validate: inputs are DTensors ->
    to_local -> temporarily unwrap parameters -> local computation -> re-wrap
    the output per the declared out_src via from_local (for data-dependent
    modules out_src is declarative validation -- the data dependence of
    all-to-all makes the placement underivable; this is an inherent
    limitation). Both modes share the same wrapper code (local_region
    tolerant passthrough semantics).

    compute_fn: the function actually executed inside the region; defaults to
    the module's own forward. Resolved uniformly by
    _resolve_local_compute_fn (user local_compute_fn / region_dispatch=False
    gate), independent of the original forward.

    spec.region_dispatch=True (dispatchable pure-ops injection): validate
    feeds the DTensors straight into compute_fn — strategy propagation runs
    THROUGH the injected fn and out_src is TRULY validated (propagated vs
    declared); production is unchanged (always local passthrough).
    """
    original_forward = module.forward
    if compute_fn is None:
        compute_fn = original_forward


    out_src_placements = None
    if spec.out_src:
        out_src_named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(out_src_named, mesh_dim_names))

    dispatch_through = bool(getattr(spec, "region_dispatch", None))
    if validate_mode and not dispatch_through:
        install_local_compute_forward_adapters(module, exclude=exclude_subtrees)

    @functools.wraps(original_forward)
    def local_region_forward(*args: Any, **kwargs: Any) -> Any:
        """Local-region forward: boundary entry → local region computation →
        re-wrap per the declared out_src → boundary exit (both modes share
        this wrapper)."""
        # Step 1: PrecompiledBoundary entry (e.g. TP all-gather; identity passthrough)
        args, kwargs = boundary.redistribute_inputs(
            args, kwargs, as_dtensor=validate_mode)

        # Step 2: local region -- the data-dependent computation (e.g. EP
        # dispatch/combine) executes on local tensors; region_dispatch=True
        # instead dispatches THROUGH the injected fn (pure standard ops) so
        # validate's strategy propagation covers it
        if validate_mode and dispatch_through:
            try:
                output = compute_fn(*args, **kwargs)
            except Exception as exc:
                raise type(exc)(
                    f"{exc}\n[region_dispatch=True] dispatch failed while "
                    "validate was dispatching through the injected "
                    "function — does the injection contain "
                    "non-dispatchable communication primitives/custom "
                    "kernels? Declare region_dispatch=False instead "
                    "(skeleton black-box hosting)"
                ) from exc
            # True validation: propagated result vs the out_src declaration
            # (a wrong declaration fails fast)
            if spec.out_src is not None:
                _validate_out_src(output, spec, mesh_dim_names,
                                  type(module).__name__)
        elif validate_mode:
            local_args = tuple(
                a.to_local() if isinstance(a, DTensor) else a for a in args)
            local_kwargs = {
                k: (v.to_local() if isinstance(v, DTensor) else v)
                for k, v in kwargs.items()
            }
            with local_compute_context():
                with _temp_local_params(module, exclude=exclude_subtrees):
                    output = compute_fn(*local_args, **local_kwargs)
        else:
            output = compute_fn(*args, **kwargs)

        # Step 3: local -> DTensor (re-wrap per the declared out_src, restoring
        # the DTensor metadata broken by all-to-all; under production the
        # boundary exit needs the same contract)
        if not isinstance(output, DTensor):
            output = _rewrap_local_outputs(
                output, spec, mesh, mesh_dim_names, type(module).__name__)

        # Step 4: PrecompiledBoundary exit (e.g. TP reduce-scatter)
        output = boundary.redistribute_outputs(
            output, as_dtensor_input=validate_mode)
        # The final boundary exit is always local (when out_plan is empty, the
        # from_local wrap from Step 3 must also be unwrapped here)
        if isinstance(output, DTensor):
            output = output.to_local()
        # D-22: deferred rowwise biases — added once after the exit reduction
        return _maybe_add_deferred_biases(module, spec, output)

    module.forward = local_region_forward


# ────────────────────────────────────────────────────────────────────────────
# Phase C: CP inner attention wrapper (05 §4.4.2 + D-01'' + D-04)
# ────────────────────────────────────────────────────────────────────────────

def _resolve_inner_target(module, spec=None):
    """Resolve the inner-wrap target -- pure location resolution.

    ``inner_target`` is MANDATORY when ``inner_wrapper`` is declared (the
    pairing is enforced in _resolve_inner_wrapper before this is called):
    "self" means the boundary module itself, otherwise resolved by attribute
    name -- fail-fast if the attribute does not exist or has no forward (a
    typo must not silently degrade). The attention-domain auto-location
    heuristic (inner_attention/attn/attention attributes, class-name
    matching, q/k/v_proj structural fallback) was REMOVED (2026-08-10): the
    inner-wrap mechanism is generic (any module, any submodule), and a
    silently located target is a silent wrong-target hazard.
    """
    explicit = getattr(spec, "inner_target", None) if spec is not None else None
    if explicit is None:
        raise ValueError(
            "inner_target is not declared — declaring inner_wrapper "
            "requires an explicit paired inner_target (\"self\" or a "
            "submodule attribute name; the auto-location heuristic was "
            "removed)")
    if explicit == "self":
        return module
    inner = getattr(module, explicit, None)
    if inner is not None and hasattr(inner, "forward"):
        return inner
    raise ValueError(
        f"spec.inner_target={explicit!r} did not match anything on "
        f"{type(module).__name__} (attribute missing or has no forward) "
        f"-- check the spelling in plan_overrides")


def _apply_custom_inner_wrapper(custom_fn, context):
    """Apply a user-defined inner_wrapper (@inner_wrapper callable).

    Contract: custom_fn must be decorated ``@inner_wrapper``; it is invoked
    with its declared context (anchor target_module + the mandatory mesh
    family, filled by name) and replaces target.forward in place. The
    replacement forward runs in a LOCAL-TENSOR world — the dual-mode
    adapter installed by _wrap_inner_attention owns all DTensor
    conversion (to_local / _temp_local_params / from_local rewrap per the
    declared placements), so custom wrappers never touch DTensor.
    """
    meta = require_injection_meta(
        custom_fn, INNER_WRAPPER, source="spec.inner_wrapper")
    custom_fn(**{k: context[k] for k in sorted(meta.context)})


def _find_dtensor_argument(args, kwargs):
    """Return the first DTensor positional or keyword argument."""
    return next(
        (
            value
            for value in (*args, *kwargs.values())
            if isinstance(value, DTensor)
        ),
        None,
    )


def _validate_inner_dispatch_output(
    output,
    expected_placements,
    boundary_module,
    wrapper_name,
):
    """Validate DTensor outputs produced through a dispatchable inner wrapper."""
    if expected_placements is None:
        return
    outputs = list(output) if isinstance(output, (tuple, list)) else [output]
    tensor_outputs = [tensor for tensor in outputs if isinstance(tensor, torch.Tensor)]
    if len(tensor_outputs) != len(expected_placements):
        raise RuntimeError(
            f"inner_wrapper {wrapper_name!r}: tensor output count "
            f"{len(tensor_outputs)} does not match the "
            f"{len(expected_placements)} declared placements — a "
            "multi-output contract must be declared name by name with "
            "matching counts"
        )
    for tensor, placements in zip(tensor_outputs, expected_placements):
        if not isinstance(tensor, DTensor):
            raise RuntimeError(
                f"inner_wrapper {wrapper_name!r} [region_dispatch=True]: "
                f"the dispatch-propagated output is not a DTensor "
                f"({type(tensor).__name__}) — the injection appears to "
                "have left the dispatch chain, so true validation cannot "
                "complete"
            )
        if tuple(tensor.placements) != tuple(placements):
            raise PlacementMismatchError(
                f"{type(boundary_module).__name__} "
                f"(inner_wrapper {wrapper_name!r})",
                tuple(placements),
                tuple(tensor.placements),
                "inner_out_src",
            )


def _rewrap_inner_tensor(tensor, placements, mesh):
    """Wrap one local tensor with its declared layout and pass other values through."""
    if not isinstance(tensor, torch.Tensor) or isinstance(tensor, DTensor):
        return tensor
    return DTensor.from_local(tensor, mesh, placements)


def _rewrap_inner_outputs(output, out_placements, mesh, wrapper_name):
    """Rewrap tensor outputs while preserving auxiliary non-tensor outputs."""
    if not isinstance(output, (tuple, list)):
        return _rewrap_inner_tensor(output, out_placements[0], mesh)
    tensor_output_count = sum(isinstance(value, torch.Tensor) for value in output)
    if tensor_output_count != len(out_placements):
        raise RuntimeError(
            f"inner_wrapper {wrapper_name!r}: the replacement forward "
            f"returned {tensor_output_count} tensor outputs, which does "
            f"not match the {len(out_placements)} declared placements — "
            "a multi-output contract must be declared name by name with "
            "matching counts"
        )
    placement_iter = iter(out_placements)
    wrapped = [
        _rewrap_inner_tensor(value, next(placement_iter), mesh)
        if isinstance(value, torch.Tensor)
        else value
        for value in output
    ]
    return tuple(wrapped) if isinstance(output, tuple) else wrapped


def _resolve_inner_output_placements(
    spec,
    boundary_module,
    target,
    mesh_dim_names,
    wrapper_name,
):
    """Resolve the declared placement rule for an injected inner forward."""
    if target is boundary_module:
        if spec is None or not spec.out_src:
            return False, None
        out_names = list(spec.out_names or spec.out_src.keys())
        missing = [name for name in out_names if name not in spec.out_src]
        if missing:
            raise ValueError(
                f"inner_wrapper {wrapper_name!r}: out_names {missing} "
                "have no declaration in out_src — a multi-output contract "
                "must be declared name by name"
            )
        return False, [
            tuple(resolve_placements(spec.out_src[name], mesh_dim_names))
            for name in out_names
        ]

    declared = getattr(spec, "inner_out_src", None) if spec is not None else None
    if declared is None:
        raise ValueError(
            f"inner_wrapper {wrapper_name!r} targets an inner submodule "
            f"of {type(boundary_module).__name__}, but inner_out_src is "
            "not declared — the framework infers and guesses nothing "
            "about inner output layouts. Choose one of: "
            "① a layout-preserving wrapper sets "
            "inner_out_src: \"first_input\"; ② declare the placements "
            "explicitly; ③ or use inner_target=\"self\" to reuse the "
            "boundary out_src contract"
        )
    if isinstance(declared, str):
        if declared != "first_input":
            raise ValueError(
                "the string value of inner_out_src only accepts the "
                f"sentinel 'first_input', got {declared!r}"
            )
        return True, None
    if all(isinstance(value, Placement) for value in declared.values()):
        return False, [tuple(resolve_placements(declared, mesh_dim_names))]
    return False, [
        tuple(resolve_placements(named, mesh_dim_names))
        for named in declared.values()
    ]


def _install_inner_adapter(target, user_fwd, boundary_module, spec, mesh,
                           mesh_dim_names, wrapper_name, validate_mode=False):
    """Unified dual-mode adapter: rewrap rules are resolved at install time,
    with zero runtime decisions (05 §4.4.2 + D-01'').

    The replacement forward of a user wrapper only faces local tensors. The
    adapter handles:
    - validate (any input is a DTensor): every DTensor input is to_local'd
      (non-tensors pass through) + ``_temp_local_params(target)`` temporarily
      unwraps the parameters → call the user forward → rewrap the outputs
      back into DTensors per the declaration (the propagation chain is
      re-linked and boundary validation continues);
    - production (no DTensor inputs): straight passthrough, zero conversion
      overhead.

    Source of the rewrap placements (the framework infers and guesses
    nothing — everything is explicitly declared):
    - case A (target is the boundary module itself): the boundary
      ``spec.out_src`` declaration (multiple outputs positionally per
      ``out_names``/declaration key order);
    - case B (an inner submodule): the explicit ``spec.inner_out_src``
      declaration — the sentinel ``"first_input"`` (output layout == the
      runtime layout of the first DTensor input; for layout-preserving
      wrappers, single output only) or NamedPlacement /
      {name: NamedPlacement} (multiple outputs mapped to tuple positions per
      declaration key order); undeclared → fail-fast at install time.
    """
    first_input_rule, out_placements = _resolve_inner_output_placements(
        spec,
        boundary_module,
        target,
        mesh_dim_names,
        wrapper_name,
    )
    if validate_mode and not getattr(spec, "region_dispatch", None):
        install_local_compute_forward_adapters(target)

    @functools.wraps(user_fwd)
    def adapted(*args: Any, **kwargs: Any) -> Any:
        """Dual-mode adapter around the user's replacement forward.

        production (no DTensor input): straight passthrough; validate:
        unwrap DTensor inputs/parameters to local, run the user forward, and
        rewrap outputs per the declaration resolved at install time (or, with
        region_dispatch=True, dispatch through and truly validate instead).
        """
        source_dtensor = _find_dtensor_argument(args, kwargs)
        if source_dtensor is None:
            return user_fwd(*args, **kwargs)          # production: straight passthrough
        if getattr(spec, "region_dispatch", None):
            # Validate dispatch-through (region_dispatch=True): the injection
            # is pure standard ops — DTensors go straight in and dispatch
            # propagation runs through the inner region; the declared rewrap
            # rules are promoted to the true-validation baseline (propagated
            # result vs declaration, a mismatch fails fast).
            try:
                out = user_fwd(*args, **kwargs)
            except Exception as exc:
                raise type(exc)(
                    f"{exc}\n[region_dispatch=True] dispatch failed "
                    f"while validate was dispatching through "
                    f"inner_wrapper {wrapper_name!r} — does the injection "
                    "contain non-dispatchable communication "
                    "primitives/custom kernels? Declare "
                    "region_dispatch=False instead (adapter black-box "
                    "hosting)") from exc
            expected = (
                [tuple(source_dtensor.placements)]
                if first_input_rule
                else out_placements
            )
            _validate_inner_dispatch_output(
                out,
                expected,
                boundary_module,
                wrapper_name,
            )
            return out
        # validate: uniformly convert to local (parameters temporarily
        # unwrapped, restored on exit)
        local_args = tuple(
            a.to_local() if isinstance(a, DTensor) else a for a in args)
        local_kwargs = {k: (v.to_local() if isinstance(v, DTensor) else v)
                        for k, v in kwargs.items()}
        with local_compute_context():
            with _temp_local_params(target):
                out = user_fwd(*local_args, **local_kwargs)

        if first_input_rule:
            if isinstance(out, (tuple, list)):
                raise RuntimeError(
                    f"inner_wrapper {wrapper_name!r}: inner_out_src="
                    "'first_input' only supports a single output — for "
                    "multiple outputs declare inner_out_src explicitly in "
                    "the {name: {axis: placement}} form")
            return _rewrap_inner_tensor(
                out,
                tuple(source_dtensor.placements),
                source_dtensor.device_mesh,
            )
        if out_placements is None:
            return out                   # case A with no out_src declaration: no rewrap
        return _rewrap_inner_outputs(out, out_placements, mesh, wrapper_name)

    target.forward = adapted


def _resolve_inner_wrapper(module, spec, cp_mesh, mesh, tp_mesh=None,
                           ep_mesh=None):
    """Resolve one explicit inner-wrapper declaration without mutating modules.

    Resolution accepts a delayed Target, a decorated callable, or a registry
    name and returns ``(name, target, apply_fn)``. ``inner_target`` and
    ``inner_wrapper`` must be declared together; absent declarations return
    ``None``. The mesh family is framework-filled for every wrapper.
    """
    custom = getattr(spec, "inner_wrapper", None) if spec is not None else None
    inner_target = getattr(spec, "inner_target", None) if spec is not None else None
    if custom is None:
        if inner_target is not None:
            raise ValueError(
                f"spec.inner_target={inner_target!r} only locates "
                "(replace whom) — after the rework the wrapping scheme is "
                "no longer chosen heuristically, so inner_wrapper must be "
                "declared explicitly too: a registry name "
                f"{sorted(INNER_WRAPPER_REGISTRY)}, an @inner_wrapper "
                "decorated callable, or a Target pointing to an in-repo "
                "reference implementation "
                "(hyper_parallel.auto_models.components.distributed.cp_wrappers.*)")
        return None

    _require_region_dispatch(spec, source="spec.inner_wrapper")
    if inner_target is None:
        raise ValueError(
            f"inner_wrapper={custom!r} is declared but inner_target is "
            "not — the attention-domain auto-location heuristic was "
            "removed (inner-wrap is a generic mechanism; silently "
            "locating risks wrapping the wrong target). The two fields "
            "must be declared as an explicit pair: to wrap the boundary "
            "module itself → inner_target=\"self\"; to wrap a submodule "
            "→ inner_target=\"<attribute name>\"")
    target = _resolve_inner_target(module, spec)
    context = {
        "target_module": target,
        "mesh": mesh,
        "tp_mesh": tp_mesh,
        "cp_mesh": cp_mesh,
        "ep_mesh": ep_mesh,
    }
    if _is_delayed_target(custom):
        fn = getattr(custom, "_target_", None)
        source = (f"inner_wrapper Target "
                  f"{getattr(custom, '_target_path', custom)}")
        meta = require_injection_meta(fn, INNER_WRAPPER, source=source)
        _check_target_config_keys(custom, "inner_wrapper")

        def _apply_target():
            build_kwargs = fill_context_kwargs(
                meta, context, getattr(custom, "_kwargs", {}), source=source)
            result = custom.build(**build_kwargs)
            if result is None:
                return   # in-place forward replacement (registry-style fn)
            if callable(result):
                _apply_custom_inner_wrapper(result, context)
                return
            raise TypeError(
                f"inner_wrapper Target "
                f"{getattr(custom, '_target_path', custom)!r} returned "
                f"{type(result).__name__} — expected None (in-place forward "
                "replacement) or an @inner_wrapper decorated callable")

        name = getattr(custom, "_target_path", None) or "custom"
        return (name, target, _apply_target)

    if callable(custom):
        require_injection_meta(
            custom, INNER_WRAPPER, source="spec.inner_wrapper")
        return ("custom", target,
                lambda: _apply_custom_inner_wrapper(custom, context))

    if isinstance(custom, str):
        fn = INNER_WRAPPER_REGISTRY.get(custom)
        if fn is None:
            raise ValueError(
                f"inner_wrapper={custom!r} is not registered in "
                f"INNER_WRAPPER_REGISTRY (available: {sorted(INNER_WRAPPER_REGISTRY)})"
                f" -- check the spelling, or first register "
                f"INNER_WRAPPER_REGISTRY[{custom!r}] = your_fn")
        meta = require_injection_meta(
            fn, INNER_WRAPPER, source=f"INNER_WRAPPER_REGISTRY[{custom!r}]")
        context["target_module"] = target
        return (custom, target,
                lambda: fn(**{k: context[k] for k in sorted(meta.context)}))

    raise TypeError(
        f"inner_wrapper must be a registry name (str), an @inner_wrapper "
        f"decorated callable, or a Target — got {type(custom).__name__}")


def _wrap_inner_attention(module, cp_mesh, *, spec=None, mesh=None,
                          mesh_dim_names=(), tp_mesh=None, ep_mesh=None,
                          validate_mode=False):
    """Inject an inner forward wrapper (one-shot replacement at apply time, 05 §4.4.2).

    General "weave into / replace the inner forward" mechanism — **not gated
    on CP**: whenever ``spec.inner_wrapper`` is declared the wrapper is
    applied (declaration == application; the resolution chain is the derived
    gate). The shipped CP wrappers (INNER_WRAPPER_REGISTRY) are its first-class
    built-in use case and still require an active cp axis (fail-fast in the
    resolution chain otherwise); custom callables/Targets receive
    ``cp_mesh=None`` when no cp axis exists and own their semantics.

    Resolution (_resolve_inner_wrapper, a pure function of the chain) is
    separated from application; when resolution returns None (no explicit
    declaration), this returns None and injects nothing. Returns the resolved
    wrapper name (or None).

    D-01'': production and validate inject **the same** wrapper, so the
    in-region computation is instruction-for-instruction identical
    (kernel-level equivalence).
    D-04: when is_causal and CP is active, replace it with an offset-aware
    explicit mask.
    Nothing is located silently: after injection an INFO log records the
    target/wrapper/source, and spec._resolved_inner_wrapper +
    spec._resolved_inner_target are written back for plan introspection.
    """
    resolved = _resolve_inner_wrapper(
        module, spec, cp_mesh, mesh, tp_mesh=tp_mesh, ep_mesh=ep_mesh)
    if resolved is None:
        return None
    name, target, apply_fn = resolved
    orig_forward = target.forward
    apply_fn()
    # Detect "a replacement really happened": attribute access on a bound
    # method creates a new object each time, so an `is` comparison is always
    # true — the underlying function objects (__func__) must be compared,
    # otherwise a pure-probe wrapper (which does not replace forward) would
    # also get an adapter installed by mistake and be forced to declare
    # inner_out_src
    new_forward = target.forward
    replaced = (getattr(new_forward, "__func__", new_forward)
                is not getattr(orig_forward, "__func__", orig_forward))
    if replaced:
        # Principle 1: the replaced forward must accept all inputs of the
        # original forward
        validate_wrapped_forward(
            orig_forward, new_forward,
            owner=f"inner_wrapper {name!r} on {type(module).__name__}")
        # Uniformly install the dual-mode adapter: the user wrapper only
        # faces local tensors; DTensor conversion and declarative rewrap are
        # managed by the adapter (local_map semantics; validate skips
        # propagation checks for the inner region — the safety net is at the
        # boundary layer)
        _install_inner_adapter(
            target, new_forward, module, spec, mesh, mesh_dim_names, name,
            validate_mode=validate_mode)
    target_name = _inner_target_name(module, target)
    if spec is not None:
        spec._resolved_inner_wrapper = name  # pylint: disable=protected-access
        spec._resolved_inner_target = target_name  # pylint: disable=protected-access
    if name == "custom":
        source = "custom callable"
    elif spec is not None and isinstance(
            getattr(spec, "inner_wrapper", None), str):
        source = "explicitly specified (registry)"
    else:
        source = "explicitly specified (Target)"
    logger.info("inner-wrap: %s target=%s <- wrapper %r (%s)",
                type(module).__name__, target_name, name, source)
    return name


def _inner_target_name(attn_module, target) -> str:
    """Readable name of the located inner-wrap target: child attribute name,
    or "self" when the boundary module itself is the target."""
    if target is attn_module:
        return "self"
    for child_name, child in attn_module.named_children():
        if child is target:
            return child_name
    return type(target).__name__


# ────────────────────────────────────────────────────────────────────────────
# Phase C: D-02 vocab-parallel embedding wrapper
# ────────────────────────────────────────────────────────────────────────────

def _is_vocab_parallel_embed(module, spec, tp_mesh) -> bool:
    """production embed boundary check: nn.Embedding + weight Shard(0) on TP + TP>1."""
    if tp_mesh is None or tp_mesh.size() <= 1:
        return False
    if not isinstance(module, nn.Embedding):
        return False
    weight_named = spec.params.get("weight", {})
    return weight_named.get("tp") == Shard(0)


def _wrap_vocab_parallel_embedding(module, tp_mesh):
    """D-02: Megatron-style masked embedding (injected at the production embed boundary).

    The vocab-range mask logic of DTensor dispatch is lost after the
    parameter unwrap -- HF native F.embedding would index out of range when
    given global token ids. The wrapper: tokens outside the local vocab
    interval [lo, hi) are zeroed and indices are shifted by the offset, so the
    output is naturally a Partial contribution and the boundary exit's
    Partial->Shard(1) reduction is unchanged.
    """
    original_forward = module.forward
    v_local = module.weight.shape[0]
    lo = tp_mesh.get_local_rank() * v_local
    hi = lo + v_local

    @functools.wraps(original_forward)
    def masked_embedding_forward(
        input_ids: torch.Tensor, *args: Any, **kwargs: Any,
    ) -> torch.Tensor:
        """Masked vocab-parallel embedding forward: zero out-of-range token
        ids, shift indices by the local vocab offset, and mask the output so
        it is a pure Partial contribution."""
        mask = (input_ids >= lo) & (input_ids < hi)
        local_ids = torch.where(mask, input_ids - lo, torch.zeros_like(input_ids))
        out = original_forward(local_ids, *args, **kwargs)
        return out * mask.unsqueeze(-1).to(out.dtype)

    module.forward = masked_embedding_forward


# ────────────────────────────────────────────────────────────────────────────
# Phase D: tied weights
# ────────────────────────────────────────────────────────────────────────────

def detect_tied_weights(model: Any) -> List[Tuple[str, str]]:
    """Detect tied-weight pairs (embed_tokens.weight <-> lm_head.weight).

    In PP scenarios cross-stage pairs cannot be detected; the user must
    explicitly declare plan.tied_pairs.

    Args:
        model: The model to inspect (an HF-style ``nn.Module`` with a
            ``config`` carrying ``tie_word_embeddings``).

    Returns:
        A list of ``(embed_fqn, lm_head_fqn)`` tied-parameter FQN pairs.
    """
    tied = []
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        embed_fqn = lm_head_fqn = None
        # remove_duplicate=False: under the default dedup of named_parameters
        # a tied parameter appears only once; duplicates must be explicitly
        # retained to discover the FQNs of both ends.
        for name, _ in model.named_parameters(remove_duplicate=False):
            if name.endswith("embed_tokens.weight"):
                embed_fqn = name
            elif name.endswith("lm_head.weight"):
                lm_head_fqn = name
        if embed_fqn and lm_head_fqn:
            tied.append((embed_fqn, lm_head_fqn))
    return tied


def _broadcast_tied_param(model, tied_pair):
    """A tied-weight pair shares storage within this rank (end A's storage is authoritative; end B shares it).

    Cross-rank broadcast would be **wrong**: a tied pair (embed/lm_head) is
    usually Shard(0)-sharded on both ends, and each rank's local shard carries
    a different vocab interval -- broadcasting rank0's shard to rank1 would
    corrupt rank1's sharding. Tied semantics require that **within the same
    rank** the two ends are the same physical parameter (shared gradients),
    not cross-rank consistency (sharding is naturally consistent: same global
    source, same placement).
    """
    fqn_a, fqn_b = tied_pair
    try:
        param_a = _get_attr_by_path(model, fqn_a)
        param_b = _get_attr_by_path(model, fqn_b)
    except AttributeError:
        return
    if param_a is None or param_b is None:
        return
    tensor_a = param_a.to_local() if isinstance(param_a, DTensor) else param_a.data
    # B shares storage with A (a tied weight is the same physical parameter)
    if isinstance(param_b, DTensor):
        param_b._local_tensor = tensor_a  # pylint: disable=protected-access
    else:
        param_b.data = tensor_a


def _replicate_tied_weights(model, tied_pairs=None):
    """Phase D: replicate tied weights across ranks."""
    for tied_pair in (tied_pairs if tied_pairs is not None
                      else detect_tied_weights(model)):
        _broadcast_tied_param(model, tied_pair)
