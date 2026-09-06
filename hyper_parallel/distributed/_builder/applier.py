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
"""applier: apply-phase orchestration — injection preflight (fail-fast before
any mutation), apply-stage mesh selection, special-handler dispatch, and the
Phase C per-boundary driving loop.

The forward writers themselves live in ``forward_rewriter.py``; Target/spec
resolution lives in ``rule_resolver.py``; parameter sharding lives in
``parameter_sharding.py``.
"""

import importlib.metadata
import logging

import numpy as np

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, init_device_mesh
from hyper_parallel.distributed._builder.forward_rewriter import (
    _bind_input_indices,
    _descendant_boundary_fqns,
    _install_bias_suppression,
    _is_vocab_parallel_embed,
    _keep_loss_parallel_output,
    _wrap_inner_attention,
    _wrap_local_region_forward,
    _wrap_production_forward,
    _wrap_validate_forward,
    _wrap_vocab_parallel_embedding,
)
from hyper_parallel.distributed._builder.parameter_sharding import (
    _resolve_module,
)
from hyper_parallel.distributed._builder.planner import (
    ShardingPlanner,
)
from hyper_parallel.distributed._builder.precompiled_boundary import (
    PrecompiledBoundary,
)
from hyper_parallel.distributed._builder.rule_resolver import (
    _require_region_dispatch,
    _resolve_local_compute_fn,
)
from hyper_parallel.distributed._builder.special_handlers import (
    SPECIAL_HANDLERS,
)
from hyper_parallel.distributed._builder.tp_collective_lowering import (
    create_tp_collective_lowerer,
)
from hyper_parallel.distributed.tensor_parallel.head_count import (
    maybe_update_head_counts,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Injection preflight (any mutation happens only after all validators pass)
# ────────────────────────────────────────────────────────────────────────────

def _validate_inner_wrapper_injections(plan, cp_mesh, transformers_version):
    # Lazy import: wrappers imports forward_rewriter at module level;
    # keeping it lazy decouples the applier's import graph.
    from hyper_parallel.distributed.context_parallel.wrappers import (  # pylint: disable=C0415
        INNER_WRAPPER_REGISTRY,
        INNER_WRAPPER_REQUIREMENTS,
    )
    """Validate the structural contract of every inner wrapper."""
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


def _validate_cp_compute_injections(plan, cp_mesh):
    """Require an explicit CP wrapper for each active attention boundary."""
    # Lazy import: see _validate_inner_wrapper_injections.
    from hyper_parallel.distributed.context_parallel.wrappers import (  # pylint: disable=C0415
        INNER_WRAPPER_REGISTRY,
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
                    "        _target_: hyper_parallel.distributed.context_parallel."
                    "wrappers.sdpa_hf_cp_wrapper\n"
                    f"(the registry {sorted(INNER_WRAPPER_REGISTRY)} can be "
                    "referenced by str name; use sdpa_qkv for the "
                    "NeMo-style (q,k,v) signature, sdpa_hf for the "
                    "HF-style forward(hidden_states); or provide a custom "
                    "callable/Target implementation)")


def _validate_ep_compute_injections(plan, model):
    """Require an explicit compute source for each active EP boundary."""
    # Lazy import: see _validate_inner_wrapper_injections.
    from hyper_parallel.distributed.expert_parallel.recipes import (  # pylint: disable=C0415
        EP_ARCHETYPE_SUGGESTIONS,
    )
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
                "merge all implemented cohesively; see the recipes.py "
                "module docstring for the available archetypes and the "
                "module interface each one expects):\n"
                "     plan_overrides:\n"
                "       - match: \"*.mlp\"\n"
                "         when: ep\n"
                "         region_dispatch: false   # a2a is inside the "
                "region; must not dispatch\n"
                "         local_compute_fn:\n"
                "           _target_: hyper_parallel.distributed.expert_parallel."
                "recipes.qwen2moe_ep_compute_fn   # pick per the "
                "archetype table\n"
                "  ② Atypical MoE → write your own factory following "
                "examples/distributed/ep_factories.py (use require_attrs "
                "for the same build-time interface validation)\n"
                "  ③ In-house EP-aware MoE (all-to-all already inside "
                "forward) → declare region_dispatch: false")


def _preflight_compute_injection(plan, mesh, model=None):
    """Fail fast before mutating a plan with incomplete CP/EP injection."""
    transformers_version = importlib.metadata.version("transformers")
    cp_mesh = _get_cp_submesh(mesh, plan.mesh_dim_names)
    _validate_inner_wrapper_injections(plan, cp_mesh, transformers_version)
    _validate_cp_compute_injections(plan, cp_mesh)
    _validate_ep_compute_injections(plan, model)


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


# ────────────────────────────────────────────────────────────────────────────
# Phase B: special handlers (registry dispatch only; concrete handlers live
# in special_handlers.py)
# ────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────
# Apply-stage mesh selection
# ────────────────────────────────────────────────────────────────────────────

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
# Phase C: forward wrapping driving loop (05 §4.4)
# ────────────────────────────────────────────────────────────────────────────

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
            module_fqn=module_fqn,
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
