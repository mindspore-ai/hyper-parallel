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
handlers -> C entry unpack + tp_grad_info -> C forward wrapping
(production/validate/moe/cp/vocab_embed, five paths) -> D tied weights.

Dual-mode architecture constraint (05 §1.4): production has zero DTensor
dispatch (build-time unpack + PrecompiledBoundary); the only difference between
validate and production is the boundary stitching method -- for any module whose
DTensor dispatch hides data-dependent logic (embedding mask / attention K/V
gather / MoE all-to-all), both modes explicitly reconstruct it with the same
local-region wrapper (D-01''/D-02/D-03').
"""

import functools
import inspect
import logging
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_models.components.distributed.cp_utils import (
    _cp_offset_causal_mask,
    flex_cp_allgather,
)
from hyper_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
)
from hyper_models.components.distributed.ep_utils import (
    MOE_ROUTER_ADAPTERS,
    _hf_native_ep_compute,
)
from hyper_models.components.distributed.sharding.apply import (
    _get_attr_by_path,
    _local_params_context,
    _resolve_module,
    _set_param_by_path,
    _stack_moe_experts,
)
from hyper_models.components.distributed.sharding_config import (
    PlacementMismatchError,
    _normalize_out_fields,
    resolve_placements,
)
from hyper_models.components.distributed.sharding_planner import SPECIAL_HANDLERS
from hyper_models.components.distributed.tp_grad import build_tp_grad_info
from hyper_models.components.distributed.head_count import (
    maybe_update_head_counts,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Main entry (05 §4.1)
# ────────────────────────────────────────────────────────────────────────────

def apply_sharding_plan(model, plan, mesh, *, validate_mode=False):
    """Apply a ShardingPlan to any nn.Module (or a list of PP parts), enabling dual-mode DTensor.

    Returns (model, tp_grad_info):
    - production: at the Phase C entry, a one-shot `_local_params_context` permanently
      unwraps DTensor parameters into plain local tensors, and builds tp_grad_info
      for fully_shard to use;
    - validate: no unwrap (parameters remain DTensors); tp_grad_info is None.
    """
    mesh_dim_names = plan.mesh_dim_names
    # Active sub-mesh: the planner strips size=1 axes (plan.mesh_dim_names), but the
    # passed-in mesh may still contain those axes -- placements are resolved against
    # plan.mesh_dim_names, so the dimensionality must align with the mesh, otherwise
    # distribute_tensor will silently shard along the wrong axis.
    full_mesh = mesh   # D-10: deriving the expert mesh requires the full dense region (including dp/cp axes)
    mesh = _get_active_mesh(mesh, mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
    models = model if isinstance(model, list) else [model]

    # D-10: when any spec enables TP-extend-EP, build the derived expert mesh
    # (repartition of the full dense region; used both for expert parameter
    # sharding and for the communication groups within the region; 05 §6.4.8)
    ep_size = next((getattr(s, "_ep_size", 0) for s in plan.modules.values()
                    if getattr(s, "_ep_size", 0)), 0)
    expert_mesh = (_build_expert_mesh(full_mesh, full_mesh.mesh_dim_names, ep_size)
                   if ep_size else None)

    # ====== Phase 0: normalize out_src/out_dst scalar shorthand (idempotent, covers user-injected paths) ======
    for spec in plan.modules.values():
        _normalize_out_fields(spec)

    # ====== Phase A: parameter sharding ======
    for part in models:
        for module_fqn, spec in plan.modules.items():
            module = _resolve_module(part, module_fqn)
            # D-09b: HF native MoE per-expert parameters are first stacked into
            # [E, ...], then sharded as stacked entries (05 §6.4.7)
            if getattr(spec, "_ep_stack", None):
                _stack_moe_experts(module, spec._ep_stack)
            if getattr(spec, "_ep_size", 0):
                # D-10: expert parameters are sharded on the derived expert mesh
                # ({EP: Shard(0)}, only the expert dim is split); all other
                # parameters go through the main mesh
                expert_params = {k: v for k, v in spec.params.items()
                                 if k.startswith("experts.")}
                dense_params = {k: v for k, v in spec.params.items()
                                if not k.startswith("experts.")}
                _shard_module_params(module, expert_params, expert_mesh,
                                     expert_mesh.mesh_dim_names)
                _shard_module_params(module, dense_params, mesh, mesh_dim_names)
            else:
                _shard_module_params(module, spec.params, mesh, mesh_dim_names)
            # D-17: production forwards run on permanently unwrapped local
            # tensors -- rewrite cached head counts to the TP-local value so
            # modeling code that reshapes with an explicit (global) num_heads
            # keeps working. Validate keeps the global counts here: boundary
            # modules run DTensor dispatch on the global logical shape.
            if not validate_mode:
                maybe_update_head_counts(
                    module, spec, module_fqn, mesh, mesh_dim_names)

    # ====== Phase B: special handlers ======
    for part in models:
        for param_ref, handler_name in plan.special_handlers.items():
            handler = SPECIAL_HANDLERS.get(handler_name)
            if handler is None:
                logger.warning("SPECIAL_HANDLERS has no registered handler: %s", handler_name)
                continue
            module_fqn, param_name = param_ref.rsplit(".", 1)
            handler(_resolve_module(part, module_fqn), param_name, mesh)

    # ====== Phase C entry: one-shot unpack at build time (production only) ======
    tp_grad_info = None
    if not validate_mode:
        tp_grad_records = {}
        for part in models:
            tp_grad_records.update(_local_params_context(part))
        if tp_grad_records and tp_mesh is not None:
            tp_grad_info = build_tp_grad_info(plan, tp_mesh)

    # ====== Phase C: wrap forward ======
    for part in models:
        _apply_phase_c(part, plan, mesh, validate_mode, expert_mesh=expert_mesh)

    # ====== Phase D: tied weights ======
    tied_pairs = list(plan.tied_pairs) or detect_tied_weights(models[0])
    for part in models:
        _replicate_tied_weights(part, mesh, tied_pairs)

    return model, tp_grad_info


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
    import numpy as np

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
    from hyper_parallel.core.dtensor.device_mesh import init_device_mesh

    shape, names, rank_list = _expert_mesh_layout(mesh, mesh_dim_names, ep_size)
    return init_device_mesh(mesh.device_type, shape, mesh_dim_names=names,
                            rank_list=rank_list)


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

def _apply_phase_c(model, plan, mesh, validate_mode, expert_mesh=None):
    """Phase C: wrap forward (production/validate/moe/cp/vocab_embed, five paths)."""
    mesh_dim_names = plan.mesh_dim_names
    cp_mesh = _get_cp_submesh(mesh, mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
    for module_fqn, spec in plan.modules.items():
        if not spec.is_boundary:
            continue
        module = _resolve_module(model, module_fqn)
        boundary = PrecompiledBoundary(spec, mesh, mesh_dim_names)
        _bind_input_indices(boundary, module)

        # Step 1: inner-wrap (D-01'': production and validate inject the same
        # wrapper, so the in-region computation is instruction-for-instruction
        # identical). The gate is derived: cp_mesh is active AND
        # _resolve_inner_wrapper resolves to non-None (any declaration of
        # inner_target/inner_wrapper/_needs_cp_attn) -- direct=False; with no
        # declaration nothing is injected
        if cp_mesh is not None and cp_mesh.size() > 1:
            _wrap_cp_inner_attention(
                module, cp_mesh, spec=spec, mesh=mesh,
                mesh_dim_names=mesh_dim_names, direct=False,
            )

        # Step 2: forward wrapping
        # local region path (D-03'): the gate is derived from the compute_fn
        # resolution chain (non-None means take the skeleton) -- the three
        # sources, user local_compute_fn / planner EP injection intent /
        # use_local_map pure gate, are resolved uniformly and never nested
        # (05 §4.4.3)
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
            _wrap_local_region_forward(
                module, boundary, spec, mesh, mesh_dim_names,
                validate_mode=validate_mode, compute_fn=compute_fn)
        elif validate_mode:
            _wrap_validate_forward(module, boundary, spec, mesh, mesh_dim_names)
        else:
            # D-02: production vocab-parallel embedding masked wrapper
            if _is_vocab_parallel_embed(module, spec, tp_mesh):
                _wrap_vocab_parallel_embedding(module, tp_mesh)
            _wrap_production_forward(module, boundary)


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


def _wrap_production_forward(module, boundary):
    """Production mode: pure local tensor computation + precompiled boundary communication (05 §4.4.1).

    _local_params_context was already invoked at the Phase C entry (parameters
    permanently unpacked).
    """
    original_forward = module.forward

    @functools.wraps(original_forward)
    def production_forward(*args, **kwargs):
        args, kwargs = boundary.redistribute_inputs(args, kwargs)
        outputs = original_forward(*args, **kwargs)
        return boundary.redistribute_outputs(outputs)

    module.forward = production_forward


def _wrap_validate_forward(module, boundary, spec, mesh, mesh_dim_names):
    """Validate mode: DTensor propagation end to end -> validate out_src (core) + out_dst (terminal modules only).

    The in-house DTensor is forward-only: validation covers only forward
    placement propagation; backward is local autograd in both modes (05 §1.0),
    and gradient equivalence is guaranteed by testing/grad_equiv.py.
    """
    original_forward = module.forward
    module_name = type(module).__name__

    @functools.wraps(original_forward)
    def validate_forward(*args, **kwargs):
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
        if spec._is_terminal and spec.out_dst is not None:
            _validate_out_dst(outputs, spec, mesh_dim_names, module_name)

        # Step 6: return local (isomorphic to production boundary outputs)
        if isinstance(outputs, DTensor):
            outputs = outputs.to_local()
        elif isinstance(outputs, (tuple, list)):
            outputs = tuple(
                t.to_local() if isinstance(t, DTensor) else t for t in outputs
            )
        return outputs

    module.forward = validate_forward


def _out_placements_of(value, spec, mesh_dim_names, attr, out_name):
    return tuple(resolve_placements(spec.__dict__[attr][out_name], mesh_dim_names))


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

@contextmanager
def _temp_local_params(module):
    """Temporarily unwrap DTensor parameters inside a validate-mode local region (restored on exit).

    Under production the parameters were already permanently unwrapped at build
    time, so this context is unnecessary. A validate-mode local region (MoE
    all-to-all / HF CP attention) computes on local tensors internally and
    needs local parameters; after restoration the DTensor propagation chain is
    unbroken.
    """
    saved = []
    for name, param in list(module.named_parameters(recurse=True)):
        if isinstance(param, DTensor):
            saved.append((name, param))
            _set_param_by_path(module, name, nn.Parameter(
                param.to_local(), requires_grad=param.requires_grad))
    try:
        yield
    finally:
        for name, param in saved:
            _set_param_by_path(module, name, param)


def _resolve_local_compute_fn(module, spec, mesh, mesh_dim_names,
                              expert_mesh):
    """Resolve the compute_fn of the local region (**single resolution chain**, 05 §4.4.3).

    Whether a module takes the local-region skeleton is derived by this chain
    (a non-None return means it does) -- the gate is not a stored bool but the
    resolution result. Priority (declarations take effect immediately, never
    nested):
    1. spec.local_compute_fn: user-defined compute_fn (an in-house
       data-dependent module reuses the skeleton but injects its own
       computation, e.g. an in-house MoE with a custom router / expert layout /
       DeepEP dispatcher);
    2. spec._ep_size > 0 and expert_mesh available: the TP-extend-EP
       **injection intent** recorded by the planner (an explicit chain link on
       par with a user fn) -> _hf_native_ep_compute (SP-in identity boundary +
       a2a over the extended EP group including TP ranks + full local expert
       computation, with no all_gather/reduce_scatter);
    3. spec.use_local_map: **pure gate** -- the module's own forward IS the
       data-dependent logic (self-sufficient, e.g. an EP-aware in-house MoE
       with the a2a already inside forward);
    4. none of the above -> None (ordinary module; takes the
       validate/production path).
    """
    custom_fn = getattr(spec, "local_compute_fn", None)
    if custom_fn is not None:
        return functools.partial(custom_fn, module)
    if getattr(spec, "_ep_size", 0) and expert_mesh is not None:
        router_fn = MOE_ROUTER_ADAPTERS.get(
            spec._moe_router, MOE_ROUTER_ADAPTERS["default"])
        tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
        tp_group = tp_mesh.get_group() if tp_mesh is not None else None
        return functools.partial(
            _hf_native_ep_compute, module, router_fn=router_fn,
            ep_group=expert_mesh.get_group("ep"),
            tp_group=tp_group)
    if getattr(spec, "use_local_map", False):
        return module.forward
    return None


def _wrap_local_region_forward(module, boundary, spec, mesh, mesh_dim_names,
                               *, validate_mode=False, compute_fn=None):
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
    _resolve_local_compute_fn (user local_compute_fn / EP injection intent /
    use_local_map gate), independent of the original forward.
    """
    original_forward = module.forward
    if compute_fn is None:
        compute_fn = original_forward

    out_src_placements = None
    if spec.out_src:
        _out_src_named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(_out_src_named, mesh_dim_names))

    @functools.wraps(original_forward)
    def local_region_forward(*args, **kwargs):
        # Step 1: PrecompiledBoundary entry (e.g. TP all-gather; identity passthrough)
        args, kwargs = boundary.redistribute_inputs(
            args, kwargs, as_dtensor=validate_mode)

        # Step 2: local region -- the data-dependent computation (e.g. EP
        # dispatch/combine) executes on local tensors
        if validate_mode:
            local_args = tuple(
                a.to_local() if isinstance(a, DTensor) else a for a in args)
            local_kwargs = {
                k: (v.to_local() if isinstance(v, DTensor) else v)
                for k, v in kwargs.items()
            }
            with _temp_local_params(module):
                output = compute_fn(*local_args, **local_kwargs)
        else:
            output = compute_fn(*args, **kwargs)

        # Step 3: local -> DTensor (re-wrap per the declared out_src, restoring
        # the DTensor metadata broken by all-to-all; under production the
        # boundary exit needs the same contract)
        if out_src_placements is not None and not isinstance(output, DTensor):
            output = DTensor.from_local(output, mesh, out_src_placements)

        # Step 4: PrecompiledBoundary exit (e.g. TP reduce-scatter)
        output = boundary.redistribute_outputs(
            output, as_dtensor_input=validate_mode)
        # The final boundary exit is always local (when out_plan is empty, the
        # from_local wrap from Step 3 must also be unwrapped here)
        if isinstance(output, DTensor):
            output = output.to_local()
        return output

    module.forward = local_region_forward


# ────────────────────────────────────────────────────────────────────────────
# Phase C: CP inner attention wrapper (05 §4.4.2 + D-01'' + D-04)
# ────────────────────────────────────────────────────────────────────────────

def _resolve_inner_target(module, spec=None):
    """Locate the target of the inner-wrap (the inner attention submodule) -- pure location resolution.

    0. Explicitly specified by the user (spec.inner_target, the plan_overrides
       entry): "self" means the module itself, otherwise resolved by attribute
       name -- fail-fast if the attribute does not exist or is not callable (a
       typo must not silently degrade);
    1. Explicit attributes inner_attention / attn / attention (NeMo/Megatron style);
    2. HF standard: class name contains "SdpaAttention" or ends with
       "Attention" -- the module itself is the inner;
    3. Structural fallback: directly holds q_proj/k_proj/v_proj.
    """
    explicit = getattr(spec, "inner_target", None) if spec is not None else None
    if explicit:
        if explicit == "self":
            return module
        inner = getattr(module, explicit, None)
        if inner is not None and hasattr(inner, "forward"):
            return inner
        raise ValueError(
            f"spec.inner_target={explicit!r} did not match anything on "
            f"{type(module).__name__} (attribute missing or has no forward) "
            f"-- check the spelling in plan_overrides")
    for name in ("inner_attention", "attn", "attention"):
        inner = getattr(module, name, None)
        if inner is not None and hasattr(inner, "forward"):
            return inner
    cls_name = type(module).__name__
    if "SdpaAttention" in cls_name or cls_name.endswith("Attention"):
        return module
    if (hasattr(module, "q_proj") and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")):
        return module
    return None


def _attn_implementation(module):
    cfg = getattr(module, "config", None)
    impl = getattr(cfg, "_attn_implementation", None)
    if impl is None and isinstance(cfg, dict):
        impl = cfg.get("attn_implementation")
    return impl


def _is_sdpa_attention(module) -> bool:
    impl = _attn_implementation(module)
    return (impl == "sdpa") or ("SdpaAttention" in type(module).__name__)


def _is_flex_attention(module) -> bool:
    impl = _attn_implementation(module)
    return (impl == "flex_attention") or ("FlexAttention" in type(module).__name__)


def _is_hf_style_attention(module) -> bool:
    """HF style (forward(hidden_states,...), projections inside forward) -> the primitive-interception path."""
    has_proj = (hasattr(module, "q_proj") and hasattr(module, "k_proj")
                and hasattr(module, "v_proj"))
    if not has_proj:
        return False
    try:
        sig = inspect.signature(module.forward)
        first_param = next(iter(sig.parameters.values()), None)
        return first_param is not None and first_param.name == "hidden_states"
    except (ValueError, TypeError):
        return type(module).__name__.endswith("Attention")


def _dispatch_builtin_cp_wrapper(target) -> str:
    """Heuristic dispatch (chain link 3 fallback): signature style x SDPA/Flex 2x2 -> registry name."""
    if _is_hf_style_attention(target):
        return "flex_hf" if _is_flex_attention(target) else "sdpa_hf"
    return "flex_qkv" if _is_flex_attention(target) else "sdpa_qkv"


def _apply_custom_inner_wrapper(custom_fn, target, cp_mesh):
    """Apply a user-defined inner_wrapper (callable form) + validate-mode DTensor hint.

    Contract: custom_fn(target_module, cp_mesh) replaces target.forward in
    place; the entry tolerantly accepts both DTensor and local inputs
    (dual-mode, 05 §4.4.2). When the user wrapper receives a DTensor (i.e.
    validate mode), a one-time WARNING is emitted -- reminding the user that
    dual-mode tolerance is their responsibility, or that they can instead
    declare use_local_map on the module and let the skeleton convert
    everything to local at the entry.
    """
    custom_fn(target, cp_mesh)
    user_fwd = target.forward

    @functools.wraps(user_fwd)
    def _guarded(*args, **kwargs):
        if not _guarded._warned and any(isinstance(a, DTensor) for a in args):
            logger.warning(
                "Custom inner_wrapper received DTensor inputs (validate mode) "
                "-- please confirm the wrapper implements dual-mode tolerance "
                "(to_local/from_local), or declare use_local_map on the module "
                "so the skeleton converts to local (05 §4.4.2/§8.6)")
            _guarded._warned = True
        return user_fwd(*args, **kwargs)

    _guarded._warned = False
    target.forward = _guarded


def _resolve_inner_wrapper(module, spec, cp_mesh, mesh, mesh_dim_names,
                           direct=False):
    """Resolve the inner-wrap scheme (**pure function, no side effects**) -- where the dual resolution chains converge.

    Returns (name, target, apply_fn) or None (None = no inner-wrap for this
    module; the gate is derived from exactly this). Chain:
    1. spec.inner_wrapper is Callable -> fully custom ("custom"; the target is
       the result of inner_target/auto-location, degrading to the boundary
       module itself if location fails);
    2. spec.inner_wrapper is str -> CP_WRAPPER_REGISTRY lookup (unknown name
       fail-fast; a None target also fail-fasts -- built-in schemes require a
       target);
    3. direct (a direct call is explicit intent) / spec.inner_target /
       spec._needs_cp_attn -> heuristic 2x2 dispatch (None target ->
       fail-fast, because a missing K/V all-gather is a silent numerical
       error);
    4. none of the above -> None.
    """
    custom = getattr(spec, "inner_wrapper", None) if spec is not None else None
    declared = (direct or custom is not None
                or getattr(spec, "inner_target", None) is not None
                or getattr(spec, "_needs_cp_attn", False))
    if not declared:
        return None

    target = _resolve_inner_target(module, spec)
    if callable(custom):
        if target is None:
            target = module
        return ("custom", target,
                lambda: _apply_custom_inner_wrapper(custom, target, cp_mesh))

    if isinstance(custom, str):
        fn = CP_WRAPPER_REGISTRY.get(custom)
        if fn is None:
            raise ValueError(
                f"inner_wrapper={custom!r} is not registered in "
                f"CP_WRAPPER_REGISTRY (available: {sorted(CP_WRAPPER_REGISTRY)})"
                f" -- check the spelling, or first register "
                f"CP_WRAPPER_REGISTRY[{custom!r}] = your_fn")
        if target is None:
            raise ValueError(
                f"inner_wrapper={custom!r} requires a target, but auto-location "
                f"failed -- please also set inner_target='attr_name' (or 'self')")
        return (custom, target,
                lambda: fn(target, cp_mesh, spec=spec, mesh=mesh,
                           mesh_dim_names=mesh_dim_names))

    if target is None:
        raise ValueError(
            f"{type(module).__name__} needs an inner-wrap but no target was "
            f"found (auto-location: inner_attention/attn/attention attribute, "
            f"class name containing SdpaAttention or ending with Attention, "
            f"holding q/k/v_proj). A missing CP K/V all-gather produces "
            f"silent numerical errors -- please explicitly set "
            f"inner_target='attr_name' (or 'self') on the spec in "
            f"plan_overrides, or provide an inner_wrapper custom wrapper")
    name = _dispatch_builtin_cp_wrapper(target)
    return (name, target,
            lambda: CP_WRAPPER_REGISTRY[name](
                target, cp_mesh, spec=spec, mesh=mesh,
                mesh_dim_names=mesh_dim_names))


def _wrap_cp_inner_attention(attn_module, cp_mesh, *, spec=None, mesh=None,
                             mesh_dim_names=(), direct=True):
    """Inject a CP-aware inner forward (one-shot replacement at compile time, 05 §4.4.2).

    Resolution (_resolve_inner_wrapper, a pure function of the dual chains) is
    separated from application; when resolution returns None (no declaration),
    this returns immediately -- derived gate. direct=True means a direct call
    counts as explicit intent (the test/manual-integration path);
    _apply_phase_c passes direct=False (the gate is derived entirely from spec
    declarations).

    D-01'': production and validate inject **the same** all-gather wrapper --
    K/V all-gather + local Q chunk SDPA, so the in-region computation is
    instruction-for-instruction identical (kernel-level equivalence).
    D-04: when is_causal and CP is active, replace it with an offset-aware
    explicit mask.
    Making the implicit explicit: after injection an INFO log records the
    target/wrapper/source, and spec._resolved_inner_wrapper is written back
    for plan introspection.
    """
    resolved = _resolve_inner_wrapper(
        attn_module, spec, cp_mesh, mesh, mesh_dim_names, direct=direct)
    if resolved is None:
        return
    name, target, apply_fn = resolved
    apply_fn()
    if spec is not None:
        spec._resolved_inner_wrapper = name
    if name == "custom":
        source = "custom callable"
    elif spec is not None and isinstance(
            getattr(spec, "inner_wrapper", None), str):
        source = "explicitly specified"
    else:
        source = "heuristic dispatch (pin with inner_wrapper='%s')" % name
    logger.info("inner-wrap: %s target=%s <- wrapper %r (%s)",
                type(attn_module).__name__,
                "self" if target is attn_module else type(target).__name__,
                name, source)


def _cp_sdpa_call(orig_sdpa, cp_mesh, q, k, v, kwargs):
    """CP-aware SDPA: K/V all-gather + D-04 offset-aware causal mask."""
    cp_dim = 2  # sequence dim of the [B, N, S, H] layout
    global_k, global_v = flex_cp_allgather(
        k.contiguous(), v.contiguous(), cp_dim, cp_mesh)
    if kwargs.get("is_causal") and cp_mesh.size() > 1:
        # D-04: triggered by CP semantics (do NOT use the q_len != kv_len
        # shape comparison as a proxy -- GQA's head-count difference does not
        # affect the sequence dim, but cross-attention/KV-cache q_len != kv_len
        # is unrelated to CP, and shape inference would misapply the lo-offset
        # semantics). With CP active, q is this rank's contiguous chunk and kv
        # is the full sequence; when q_len != kv_len, torch's is_causal aligns
        # top-left (equivalent to assuming Q starts at global 0), so the mask
        # is wrong for rank>0 chunks (G4) -> replace it with an explicit lower-
        # triangular mask offset by this rank's global Q offset lo (on rank0
        # lo=0 degenerates to the standard causal mask, same behavior).
        # Performance note: an explicit attn_mask rules out the SDPA flash
        # backend (falling back to mem_efficient/math); correctness of the
        # CP+causal path takes priority over this (05 §4.4.2).
        cp_rank = cp_mesh.get_local_rank()
        lo = cp_rank * q.shape[cp_dim]
        kwargs = dict(kwargs)
        kwargs.pop("is_causal")
        kwargs["attn_mask"] = _cp_offset_causal_mask(
            q.shape[cp_dim], global_k.shape[cp_dim], lo, q.device)
    return orig_sdpa(q, global_k, global_v, **kwargs)


def _wrap_sdpa_for_cp(inner_attn, cp_mesh, *, spec=None, mesh=None,
                      mesh_dim_names=()):
    """NeMo/Megatron SDPA path (registry "sdpa_qkv"): explicit all-gather K/V.

    Assumes the inner_attention.forward(q,k,v,...) signature convention.
    Shared by both modes: when q/k/v are DTensors, unwrap them (validate) and
    re-wrap the output with q's placements; local inputs pass through
    (production).
    """
    original_forward = inner_attn.forward

    @functools.wraps(original_forward)
    def cp_forward(q, k, v, **kwargs):
        was_dtensor = isinstance(q, DTensor)
        q_placements = tuple(q.placements) if was_dtensor else None
        mesh = q.device_mesh if was_dtensor else None
        ql, kl, vl = (t.to_local() if isinstance(t, DTensor) else t
                      for t in (q, k, v))
        out = _cp_sdpa_call(
            lambda *a, **kw: original_forward(*a, **kw),
            cp_mesh, ql, kl, vl, kwargs)
        if was_dtensor and isinstance(out, torch.Tensor):
            out = DTensor.from_local(out, mesh, q_placements)
        return out

    inner_attn.forward = cp_forward


def _wrap_flex_attn_for_cp(inner_attn, cp_mesh, *, spec=None, mesh=None,
                           mesh_dim_names=()):
    """NeMo/Megatron FlexAttention path (registry "flex_qkv"): explicit all-gather K/V (shared by both modes).

    Same constraint as _wrap_hf_flex_for_cp: the block_mask must be built for
    the global kv length.
    """
    original_forward = inner_attn.forward

    @functools.wraps(original_forward)
    def cp_forward(q, k, v, **kwargs):
        was_dtensor = isinstance(q, DTensor)
        q_placements = tuple(q.placements) if was_dtensor else None
        mesh = q.device_mesh if was_dtensor else None
        ql, kl, vl = (t.to_local() if isinstance(t, DTensor) else t
                      for t in (q, k, v))
        global_k, global_v = flex_cp_allgather(
            kl.contiguous(), vl.contiguous(), 2, cp_mesh)
        out = original_forward(ql, global_k, global_v, **kwargs)
        if was_dtensor and isinstance(out, torch.Tensor):
            out = DTensor.from_local(out, mesh, q_placements)
        return out

    inner_attn.forward = cp_forward


def _wrap_hf_sdpa_for_cp(inner_attn, cp_mesh, *, spec=None, mesh=None,
                         mesh_dim_names=()):
    """HF standard SDPA path: forward(hidden_states,...) -> primitive interception (05 §4.4.2).

    Shared by both modes (D-01''): when hidden_states is a DTensor (validate),
    unwrap it and temporarily unwrap the module parameters; at the exit,
    re-wrap per the spec.out_src declaration; local inputs pass through
    (production). The primitive interception is a temporary global function
    replacement (restored via try/finally) and is not thread-safe; it is safe
    under single-process SPMD training (consistent with the TorchTitan CP
    implementation).
    Misfire detection: the heuristic may misroute a module that does not call
    F.sdpa onto this path -- if the primitive did not intercept a single call,
    the K/V were not gathered (a silent numerical error), so raise a
    RuntimeError immediately.
    """
    original_forward = inner_attn.forward
    orig_sdpa = F.scaled_dot_product_attention

    out_src_placements = None
    if spec is not None and spec.out_src:
        _named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(_named, mesh_dim_names))

    @functools.wraps(original_forward)
    def cp_forward(hidden_states, *args, **kwargs):
        fired = {"hit": False}

        def cp_aware_sdpa(q, k, v, **kw):
            fired["hit"] = True
            return _cp_sdpa_call(orig_sdpa, cp_mesh, q, k, v, kw)

        was_dtensor = isinstance(hidden_states, DTensor)
        hs = hidden_states.to_local() if was_dtensor else hidden_states
        F.scaled_dot_product_attention = cp_aware_sdpa
        try:
            if was_dtensor:
                with _temp_local_params(inner_attn):
                    out = original_forward(hs, *args, **kwargs)
            else:
                out = original_forward(hs, *args, **kwargs)
        finally:
            F.scaled_dot_product_attention = orig_sdpa
        if not fired["hit"]:
            raise RuntimeError(
                f"CP wrapper 'sdpa_hf' did not intercept any "
                f"F.scaled_dot_product_attention call on "
                f"{type(inner_attn).__name__} -- the heuristic judgment does "
                f"not match the module implementation (K/V were not "
                f"all-gathered; continuing would produce silent numerical "
                f"errors). Please explicitly set inner_wrapper='sdpa_qkv' "
                f"(the (q,k,v) convention), or provide a custom inner_wrapper "
                f"callable")
        if (was_dtensor and out_src_placements is not None
                and not isinstance(out, DTensor) and isinstance(out, torch.Tensor)):
            out = DTensor.from_local(out, mesh, out_src_placements)
        return out

    inner_attn.forward = cp_forward


def _wrap_hf_flex_for_cp(inner_attn, cp_mesh, *, spec=None, mesh=None,
                         mesh_dim_names=()):
    """HF standard FlexAttention path: intercept flex_attention (same structure as the SDPA path).

    Constraint: score_mod/block_mask pass through verbatim via kwargs -- under
    CP, kv_len changes from S/cp to S, so the block_mask must be built for the
    **global kv length** (constructed on the full sequence in the data
    pipeline / model side), otherwise shapes and semantics are misaligned. The
    wrapper does not validate this.
    Misfire detection is the same as 'sdpa_hf': if no flex_attention call is
    intercepted, raise a RuntimeError.
    """
    original_forward = inner_attn.forward
    from torch.nn.attention.flex_attention import flex_attention as _orig_flex

    out_src_placements = None
    if spec is not None and spec.out_src:
        _named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(_named, mesh_dim_names))

    @functools.wraps(original_forward)
    def cp_forward(hidden_states, *args, **kwargs):
        import torch.nn.attention.flex_attention as _flex_mod
        fired = {"hit": False}

        def cp_aware_flex(q, k, v, **kw):
            fired["hit"] = True
            global_k, global_v = flex_cp_allgather(
                k.contiguous(), v.contiguous(), 2, cp_mesh)
            return _orig_flex(q, global_k, global_v, **kw)

        was_dtensor = isinstance(hidden_states, DTensor)
        hs = hidden_states.to_local() if was_dtensor else hidden_states
        _flex_mod.flex_attention = cp_aware_flex
        try:
            if was_dtensor:
                with _temp_local_params(inner_attn):
                    out = original_forward(hs, *args, **kwargs)
            else:
                out = original_forward(hs, *args, **kwargs)
        finally:
            _flex_mod.flex_attention = _orig_flex
        if not fired["hit"]:
            raise RuntimeError(
                f"CP wrapper 'flex_hf' did not intercept any flex_attention "
                f"call on {type(inner_attn).__name__} -- the heuristic "
                f"judgment does not match the module implementation (K/V were "
                f"not all-gathered; continuing would produce silent numerical "
                f"errors). Please explicitly set inner_wrapper='flex_qkv' "
                f"(the (q,k,v) convention), or provide a custom inner_wrapper "
                f"callable")
        if (was_dtensor and out_src_placements is not None
                and not isinstance(out, DTensor) and isinstance(out, torch.Tensor)):
            out = DTensor.from_local(out, mesh, out_src_placements)
        return out

    inner_attn.forward = cp_forward


# {registry_name: wrapper_fn} -- built-in CP wrapper registry (05 §4.4.2).
# Contract: fn(target, cp_mesh, *, spec=None, mesh=None, mesh_dim_names=())
# replaces target.forward in place (K/V all-gather + dual-mode tolerance).
# Users may register their own named schemes:
# CP_WRAPPER_REGISTRY["my_flash"] = my_fn, after which
# spec.inner_wrapper="my_flash" references it by name.
CP_WRAPPER_REGISTRY = {
    "sdpa_qkv": _wrap_sdpa_for_cp,    # NeMo convention forward(q,k,v,...) + D-04 mask
    "sdpa_hf": _wrap_hf_sdpa_for_cp,  # HF convention + F.sdpa primitive interception (misfire detection)
    "flex_qkv": _wrap_flex_attn_for_cp,
    "flex_hf": _wrap_hf_flex_for_cp,
}


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
    def masked_embedding_forward(input_ids, *args, **kwargs):
        mask = (input_ids >= lo) & (input_ids < hi)
        local_ids = torch.where(mask, input_ids - lo, torch.zeros_like(input_ids))
        out = original_forward(local_ids, *args, **kwargs)
        return out * mask.unsqueeze(-1).to(out.dtype)

    module.forward = masked_embedding_forward


# ────────────────────────────────────────────────────────────────────────────
# Phase D: tied weights
# ────────────────────────────────────────────────────────────────────────────

def detect_tied_weights(model):
    """Detect tied-weight pairs (embed_tokens.weight <-> lm_head.weight).

    In PP scenarios cross-stage pairs cannot be detected; the user must
    explicitly declare plan.tied_pairs.
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


def _broadcast_tied_param(model, tied_pair, mesh):
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
        param_b._local_tensor = tensor_a
    else:
        param_b.data = tensor_a


def _replicate_tied_weights(model, mesh, tied_pairs=None):
    """Phase D: replicate tied weights across ranks."""
    for tied_pair in (tied_pairs if tied_pairs is not None
                      else detect_tied_weights(model)):
        _broadcast_tied_param(model, tied_pair, mesh)
