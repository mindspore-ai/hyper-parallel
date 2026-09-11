# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Codegen runtime: literal-plan execution helpers for generated modules.

The generated modeling file must not re-derive the sharding plan — it only
executes the frozen literal plan (``_HYPER_PARAM_PLAN`` etc.) that generation
recorded in ``codegen_meta.json``.  This module is the shared runtime for
those literals: every helper mirrors the corresponding behavior in
``components/distributed/sharding_applier.py``, but the input is the frozen
JSON form, never a live ``ShardingPlan``.

The generated path executes these helpers instead of entering
``sharding_applier``.  The runtime preserves the same semantics over frozen
literals and reuses the public plumbing (``PrecompiledBoundary``,
``distribute_tensor``, ``_local_params_context``).

Helpers here are pure-literal: any unknown placement form, missing entrypoint,
or absent meta fails fast — nothing silently degrades to a ``Replicate()``
no-op (a silently dropped placement would corrupt the frozen contract).
"""
from __future__ import annotations

import functools
import inspect
import json
import os
import sys
from typing import Any, Optional

from hyper_parallel.codegen.loader import import_generated_module
from hyper_parallel.codegen.meta import (
    CodegenMeta,
    load_codegen_meta as _load_codegen_meta_from_path,
    meta_to_dict,
    validate_meta_schema,
)

# ``sharding/apply`` imports torch at module level, so it is deliberately not
# imported here: runtime.py must stay importable without torch (generation-time
# hosts and the local smoke tests import this module for its pure helpers).  The
# apply helpers are imported lazily inside the one function that needs each.

def load_codegen_meta(artifact_dir: str) -> Optional[CodegenMeta]:
    """Read ``codegen_meta.json`` from an artifact directory.

    Thin wrapper over ``codegen/meta.load_codegen_meta``: the meta module
    takes a full path, the runtime contract is the artifact dir.  ``None``
    when absent or malformed.
    """
    return _load_codegen_meta_from_path(os.path.join(artifact_dir, "codegen_meta.json"))


def verify_codegen_signature(meta: CodegenMeta, hf_config: Any = None) -> None:
    """Validate the generated bundle metadata before model construction.

    The full recompute (canonical signature over YAML + env) is the trainer
    preflight's job — it runs before ``_build_model`` with the trainer config.
    The gate cannot see ``TrainerConfig`` (infrastructure only holds
    ``distributed_setup`` / ``hf_config``), so this check covers what it can
    and fails fast on anything inconsistent:

    - meta passes the schema (``validate_meta_schema``);
    - ``covered`` / ``covered.sharding_plan`` / ``entrypoints["parallelize"]``
      exist — the gate decision would be meaningless without them;
    - the meta's recorded architecture matches ``hf_config.architectures[0]``,
      when the hf_config is available (catches a bundle generated for a
      different model class).
    """
    validate_meta_schema(meta_to_dict(meta))
    covered = meta.covered or {}
    if not isinstance(covered.get("sharding_plan"), bool):
        raise TypeError(
            "codegen gate: meta.covered.sharding_plan must be a bool; "
            f"got {covered.get('sharding_plan')!r}"
        )
    if not meta.entrypoints.get("parallelize"):
        raise KeyError(
            "codegen gate: meta.entrypoints['parallelize'] is missing — the "
            "generated module cannot be parallelized; regenerate the artifact"
        )
    if hf_config is not None:
        architectures = getattr(hf_config, "architectures", None) or []
        arch_name = architectures[0] if architectures else None
        source_arch = (meta.source or {}).get("architecture")
        if arch_name and source_arch and arch_name != source_arch:
            raise ValueError(
                f"codegen gate: meta records architecture {source_arch!r} but "
                f"hf_config resolves {arch_name!r}; the bundle does not match "
                "the requested model"
            )


def publish_codegen_mesh_context(artifact_dir: str, mesh_context: Any) -> None:
    """Publish ``mesh_context`` into the generated module namespace.

    Lowered boundary forwards reference ``mesh_context`` as a module global.
    A pure data-parallel/FSDP run does not need generated parameter sharding,
    but it still executes those lowered forwards; publishing the context keeps
    identity redistributions well-defined without calling ``hyper_parallelize``.
    """
    module = import_generated_module(artifact_dir)
    module.__dict__["mesh_context"] = mesh_context


# ---------------------------------------------------------------------------
# Mesh resolution (mirrors apply_sharding_plan's source-mesh selection)
# ---------------------------------------------------------------------------

def _active_mesh(mesh: Any, mesh_dim_names):
    """Slice a full mesh down to the plan's active axes (``_get_active_mesh``).

    ``apply_sharding_plan`` aligns its sharding mesh to ``plan.mesh_dim_names``
    (the axes the placements were resolved against) by slicing the device
    mesh with ``mesh[tuple(mesh_dim_names)]``.  The runtime receives the same
    full meshes (a raw ``DeviceMesh`` carries ``dp``/``cp`` axes the plan
    never shards on; the trainer's ``fsdp_non_moe_mesh`` carries
    ``fsdp_replicate``/``fsdp_shard``) — slicing by the same names keeps the
    dense mesh identical to the applier's active mesh, so a placement never
    lands on a wrong axis.

    A mesh without the given axes (e.g. an ``OfflineMesh`` built without
    ``ep``) returns itself, matching ``_get_active_mesh``'s fallback.  A mesh
    that HAS the axes but cannot be sliced raises — never silently returns the
    unsliced mesh (that would shard along a wrong axis).

    The frozen plan records its active axes in *offline-mesh* order
    (``build_offline_mesh`` appends ``tp`` then ``cp``), which is NOT
    guaranteed to be the live ``DeviceMesh``'s declared order (a trainer mesh
    is ``("dp", "cp", "tp")``).  ``DeviceMesh.__getitem__`` only accepts a
    slice whose axis indices are ascending (device_mesh.py), so the names are
    reordered into the mesh's own declared order before slicing.  The reordered
    tuple is returned alongside the mesh
    (``(sub_mesh, active_dim_names)``): the caller must resolve placements
    against the *returned* names, because the sliced sub-mesh carries exactly
    that axis order and a placement tuple must line up with it.
    """
    if not mesh_dim_names:
        return mesh, ()
    names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
    if not names:
        return mesh, tuple(mesh_dim_names)
    # Mesh-major order: keep only the axes the plan shards on, reordered to
    # the mesh's own declared order (so `mesh[tuple(...)]` indices ascend).
    active = tuple(n for n in names if n in mesh_dim_names)
    if not active:
        return mesh, ()
    if active == tuple(mesh_dim_names):
        return mesh, tuple(mesh_dim_names)
    if all(n in names for n in mesh_dim_names):
        return mesh[active], active
    # The plan names an axis the mesh does not carry (e.g. a plan frozen for
    # an ``ep`` axis on a mesh that derives EP from the dense region).  Never
    # silently return the unsliced mesh with a dim name that does not exist on
    # it — that would resolve placements against a phantom axis and corrupt
    # every downstream DTensor layout.  Slice to the overlapping active axes
    # and keep the same mesh-major names.
    return mesh[active], active


def _resolve_runtime_meshes(mesh: Any, mesh_dim_names=None):
    """Resolve the active (dense) mesh and the FSDP source meshes.

    Mirrors ``apply_sharding_plan`` / ``_resolve_parameter_source_meshes``.
    ``dense_mesh`` is the mesh parameters shard across and boundaries
    redistribute against: the **device** mesh sliced down to the caller's
    ``mesh_dim_names`` (the plan's active axes), exactly like
    ``_get_active_mesh(mesh_context.device_mesh, plan.mesh_dim_names)`` in the
    applier.  ``dense_source_mesh`` is a different thing — the FSDP2 source
    metadata mesh: the non-MoE topology's ``tp`` child, with the device mesh
    ``tp`` child as the no-FSDP-context fallback.  ``expert_source_mesh`` is
    the MoE topology's ``ep`` child.  The source meshes stay un-sliced (they
    are the true ``tp`` / ``ep`` child meshes used by ``build_tp_grad_info``).

    Returns ``(dense_mesh, dense_source_mesh, expert_source_mesh,
    active_dim_names)``.  ``active_dim_names`` is the mesh-major reordering of
    the caller's ``mesh_dim_names`` (the order ``dense_mesh`` actually carries
    after slicing; ``_active_mesh`` reorders the frozen plan's offline-mesh
    order into the live mesh's declared order).  Callers that resolve
    placements against ``dense_mesh`` must use this returned order, never the
    raw ``mesh_dim_names`` argument.
    """
    device_mesh = getattr(mesh, "device_mesh", mesh)
    dense_mesh, active_dim_names = _active_mesh(device_mesh, mesh_dim_names)

    fsdp_non_moe_mesh = getattr(mesh, "fsdp_non_moe_mesh", None)
    if fsdp_non_moe_mesh is not None:
        dense_source_mesh = (
            fsdp_non_moe_mesh["tp"]
            if "tp" in tuple(getattr(fsdp_non_moe_mesh, "mesh_dim_names", ()) or ())
            else None
        )
    else:
        dense_source_mesh = (
            device_mesh["tp"]
            if "tp" in tuple(getattr(device_mesh, "mesh_dim_names", ()) or ())
            else None
        )
    fsdp_moe_mesh = getattr(mesh, "fsdp_moe_mesh", None)
    expert_mesh = (
        fsdp_moe_mesh["ep"]
        if fsdp_moe_mesh is not None
        and "ep" in tuple(getattr(fsdp_moe_mesh, "mesh_dim_names", ()) or ())
        else None
    )
    if expert_mesh is not None:
        # The expert mesh is also sliced to the plan's active axes, so it gets
        # the same mesh-major reordering (and must arrive with it).
        expert_mesh, _ = _active_mesh(expert_mesh, mesh_dim_names)
    return dense_mesh, dense_source_mesh, expert_mesh, active_dim_names


# ---------------------------------------------------------------------------
# Parameter sharding (literal form of applier _shard_module_params)
# ---------------------------------------------------------------------------

def _shard_module_params(module, param_specs, mesh, mesh_dim_names):
    """distribute_tensor() converts parameters into DTensors.

    Same semantics as the applier's ``_shard_module_params``: meta tensors
    stay meta (zero-memory path), real tensors are physically split, and an
    already-DTensor parameter is skipped when its placement matches and
    rejected otherwise (``PlacementMismatchError``).
    """
    from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
    from hyper_parallel.distributed.recipe_spec import (
        resolve_placements,
    )
    from hyper_parallel.distributed._builder.parameter_sharding import (
        _get_attr_by_path,
        _set_param_by_path,
    )

    for param_path, named in param_specs.items():
        param = _get_attr_by_path(module, param_path)
        placements = tuple(resolve_placements(named, mesh_dim_names))
        if not placements:
            continue  # no active DTensor axes (all size 1) -- no sharding needed

        if isinstance(param, DTensor):
            if tuple(param.placements) != placements:
                from hyper_parallel.distributed.recipe_spec import (
                    PlacementMismatchError,
                )

                raise PlacementMismatchError(
                    f"{type(module).__name__}.{param_path}",
                    placements, tuple(param.placements), "params",
                )
            continue

        import torch

        src = param.data if hasattr(param, "data") else param
        dt = distribute_tensor(src, mesh, placements)
        requires_grad = getattr(param, "requires_grad", True)
        _set_param_by_path(module, param_path,
                           torch.nn.Parameter(dt, requires_grad=requires_grad))


def hyper_shard_params(
    model,
    param_plan: dict[str, Any],
    mesh_context,
    mesh_dim_names: Optional[tuple[str, ...]] = None,
) -> None:
    """Execute parameter sharding from the frozen literal ``param_plan``.

    ``param_plan`` is the meta's per-boundary literal plan (freeze.py
    ``freeze_param_plan`` shape): ``{boundary_fqn: {"params": {param_name:
    {axis: placement_str}}, ...}}``.  Dense parameters shard across the
    active dense mesh; parameters named ``experts.*`` shard across the MoE
    topology mesh (the same "experts."-prefix signal the applier keys on via
    ``spec._ep_size``).

    ``mesh_dim_names`` is the plan's active axes (``meta.mesh_dim_names``):
    the dense mesh is sliced down to them, exactly like ``_get_active_mesh``
    in the applier, so a ``S(0)`` placement lands on the tp axis — never on
    a dp axis the plan doesn't shard on.  ``None`` keeps the full mesh for
    compatibility with metadata that does not record active mesh axes.

    TODO: Apply each frozen ``ep_stack`` entry before sharding its expert
    parameters, matching ``_stack_moe_experts`` in the native applier.
    """
    dense_mesh, _, expert_mesh, active_dim_names = (
        _resolve_runtime_meshes(mesh_context, mesh_dim_names)
    )
    if dense_mesh is None:
        raise ValueError(
            "hyper_shard_params requires a DeviceMesh / MeshContext to shard against"
        )
    from hyper_parallel.distributed._builder.parameter_sharding import _resolve_module

    dense_dim_names = (
        tuple(getattr(dense_mesh, "mesh_dim_names", ()) or ())
        or active_dim_names
    )
    expert_dim_names = (
        tuple(getattr(expert_mesh, "mesh_dim_names", ()) or ())
        if expert_mesh is not None
        else ()
    )

    for boundary_fqn, entry in param_plan.items():
        module = _resolve_module(model, boundary_fqn)
        params = entry.get("params") or {}
        has_experts = any(name.startswith("experts.") for name in params)
        if has_experts and expert_mesh is not None:
            expert_params = {
                name: _parse_named(params[name])
                for name in params
                if name.startswith("experts.")
            }
            dense_params = {
                name: _parse_named(params[name])
                for name in params
                if not name.startswith("experts.")
            }
            _shard_module_params(module, expert_params, expert_mesh, expert_dim_names)
            if dense_params:
                _shard_module_params(module, dense_params, dense_mesh, dense_dim_names)
        else:
            # When ep_size==1 there is no MoE mesh, so experts.* parameters
            # have nowhere to shard to. Skip them entirely — leave them as
            # plain (non-DTensor) parameters so FSDP2 can shard them on the
            # fsdp_shard axis, exactly like the native applier does.
            # Only shard the non-expert parameters on the dense mesh.
            dense_only = {
                name: _parse_named(params[name])
                for name in params
                if not name.startswith("experts.")
            }
            if dense_only:
                _shard_module_params(module, dense_only, dense_mesh, dense_dim_names)
        maybe_update_head_counts(module, entry, dense_mesh, dense_dim_names)


def maybe_update_head_counts(module, entry: dict[str, Any], mesh, mesh_dim_names) -> None:
    """Literal-form head-count adjustment (applier ``maybe_update_head_counts``).

    The applier reads the plan spec's live placement objects; the frozen entry
    keeps the same placement strings, so the head-shard test is reimplemented
    here against the literal form.

    TODO: Serialize and apply ``_tp_local_attr_plan`` so user-defined
    ``tp_divide_attrs`` are reflected in generated models.
    """
    from hyper_parallel.distributed.tensor_parallel.head_count import (
        _QKV_WEIGHT_SUFFIXES,
        _tp_degree,
        update_module_head_counts,
    )

    if "tp" not in mesh_dim_names:
        return
    tp_idx = tuple(mesh_dim_names).index("tp")
    head_sharded = False
    for name, named in (entry.get("params") or {}).items():
        if not name.endswith(_QKV_WEIGHT_SUFFIXES):
            continue
        placements = _parse_named(named)
        if "tp" not in placements:
            continue
        if _is_shard_zero(placements["tp"]):
            head_sharded = True
            break
    if head_sharded:
        update_module_head_counts(module, _tp_degree(mesh, mesh_dim_names))


def _is_shard_zero(placement: Any) -> bool:
    """True for ``Shard(0)`` — the colwise split that divides the head dim."""
    from hyper_parallel.core.dtensor.placement_types import Shard

    return isinstance(placement, Shard) and placement.dim == 0


def _parse_named(data: dict[str, str]) -> dict[str, Any]:
    """Parse one frozen param entry (``{axis: placement_str}``) to placements."""
    from hyper_parallel.codegen.plan.freeze import parse_placement

    return {axis: parse_placement(text) for axis, text in data.items()}


# ---------------------------------------------------------------------------
# Special parameter handlers (applier _apply_plan_special_handlers)
# ---------------------------------------------------------------------------

def hyper_apply_special_handlers(
    model,
    handlers: dict[str, str],
    mesh_context,
    mesh_dim_names: Optional[tuple[str, ...]] = None,
) -> None:
    """Run parameter handlers declared by the frozen literal plan.

    ``handlers`` is ``meta.special_handlers``: ``{param_ref: handler_name}``.
    The registry is the same ``SPECIAL_HANDLERS`` map the applier uses — a
    handler name without a registration fails fast instead of being skipped.
    ``mesh_dim_names`` slices the dense mesh to the plan's active axes (see
    :func:`hyper_shard_params`).
    """
    from hyper_parallel.distributed._builder.special_handlers import SPECIAL_HANDLERS
    from hyper_parallel.distributed._builder.parameter_sharding import _resolve_module

    if not handlers:
        return
    dense_mesh, _, _, _ = _resolve_runtime_meshes(mesh_context, mesh_dim_names)
    for param_ref, handler_name in handlers.items():
        handler = SPECIAL_HANDLERS.get(handler_name)
        if handler is None:
            raise KeyError(
                f"hyper_apply_special_handlers: no registered handler "
                f"{handler_name!r} (referenced by {param_ref!r})"
            )
        module_fqn, param_name = param_ref.rsplit(".", 1)
        handler(_resolve_module(model, module_fqn), param_name, dense_mesh)


# ---------------------------------------------------------------------------
# Tied weights (applier _replicate_tied_weights)
# ---------------------------------------------------------------------------

def hyper_replicate_tied(model, tied_pairs: Optional[list] = None) -> None:
    """Replicate tied weights across ranks.

    Same semantics as the applier: when ``tied_pairs`` is None the pairs are
    detected from the model config (``tie_word_embeddings``), mirroring
    ``detect_tied_weights``.
    """
    from hyper_parallel.distributed._builder.parameter_sharding import (
        _replicate_tied_weights,
        detect_tied_weights,
    )

    _replicate_tied_weights(model, tied_pairs or detect_tied_weights(model))


# ---------------------------------------------------------------------------
# TP gradient metadata (applier _build_runtime_tp_grad_info + tp_grad)
# ---------------------------------------------------------------------------

def hyper_build_tp_grad_info(
    model,
    param_plan: dict[str, Any],
    mesh_context,
    *,
    tied_pairs: Optional[list] = None,
) -> Optional[dict[str, tuple[Any, Any]]]:
    """Build FSDP2 source metadata from the frozen literal plan.

    Mirrors ``build_tp_grad_info``: dense parameters map to their parsed
    ``tp`` placement on the dense TP source mesh; ``experts.*`` parameters map
    to ``Shard(0)`` on the expert EP source mesh.  Tied pairs are normalized
    to a single placement (``Shard`` takes precedence), guaranteeing
    consistent TP semantics on both ends of a tied pair.

    The frozen plan declares expert parameters by the ``experts.`` prefix
    (``_ep_size`` is not serialized per param entry, see ``freeze_param_plan``).
    When ``ep_size==1`` there is no expert EP source mesh, so experts are
    placed on the dense source mesh as Replicate, matching the native applier
    and the ``hyper_shard_params`` fallback — never silently mis-sharded.

    The one-shot ``_local_params_context`` unwrap happens here, exactly like
    the native applier — after this call the model's DTensor
    parameters are permanently plain locals, so ``hyper_build_tp_grad_info``
    must be the LAST sharding helper the generated ``hyper_parallelize``
    runs (the applier's own ordering invariant).
    """
    from hyper_parallel.core.dtensor.placement_types import Shard
    from hyper_parallel.distributed._builder.parameter_sharding import _local_params_context

    # ``build_source_shard_info`` records the complete source layout: one
    # placement per source (non-FSDP) mesh axis, as a tuple.  FSDP2 iterates
    # that tuple (`any(isinstance(p, Partial) for p in placements)`), so a bare
    # ``Shard(0)`` — the naive literal form of an ``ep``/``tp`` placement on a
    # single-axis source mesh — crashes with "'Shard' object is not iterable".
    # Mirror the applier exactly: resolve placements over the source mesh's
    # non-FSDP dimension names and slice the source sub-mesh to those axes.
    from hyper_parallel.distributed.recipe_spec import resolve_placements
    from hyper_parallel.distributed._builder.source_shard import (
        _source_dim_names,
        _source_sub_mesh,
    )

    _local_params_context(model)
    _, dense_source_mesh, expert_source_mesh, _ = _resolve_runtime_meshes(mesh_context)
    dense_source_dims = _source_dim_names(dense_source_mesh, None)
    expert_source_dims = _source_dim_names(expert_source_mesh, ("ep",))
    dense_source_mesh = _source_sub_mesh(dense_source_mesh, dense_source_dims)
    expert_source_mesh = _source_sub_mesh(expert_source_mesh, expert_source_dims)
    if dense_source_mesh is None and expert_source_mesh is None:
        return None

    info: dict[str, tuple[Any, Any]] = {}
    for boundary_fqn, entry in param_plan.items():
        for param_name, named in (entry.get("params") or {}).items():
            full_fqn = f"{boundary_fqn}.{param_name}"
            if param_name.startswith("experts.") and expert_source_mesh is not None:
                placements = tuple(resolve_placements({"ep": Shard(0)}, expert_source_dims))
                info[full_fqn] = (placements, expert_source_mesh)
            elif not param_name.startswith("experts."):
                # Non-expert params: resolve on the dense source mesh.
                parsed = _parse_named(named)
                placements = tuple(resolve_placements(parsed, dense_source_dims))
                info[full_fqn] = (placements, dense_source_mesh)
            # ep_size==1: skip expert params — FSDP2 owns them (no source
            # shard info needed, matching the hyper_shard_params fallback).

    for pair in tied_pairs or []:
        if len(pair) != 2 or pair[0] not in info or pair[1] not in info:
            continue
        pa, _ = info[pair[0]]
        pb, _ = info[pair[1]]
        if pa != pb and len(pa) == len(pb):
            norm = tuple(
                x if isinstance(x, Shard) else y
                for x, y in zip(pa, pb)
            )
            info[pair[0]] = (norm, info[pair[0]][1])
            info[pair[1]] = (norm, info[pair[1]][1])
    return info or None


# ---------------------------------------------------------------------------
# Boundary redistribution through PrecompiledBoundary
# ---------------------------------------------------------------------------

def _load_target_path(path: str):
    """Import the callable at a dotted ``_target_`` path.

    The frozen injection serializes a ``trainer.config.Target`` to
    ``{"_target_": <path>}``; the runtime re-imports the callable and rebuilds
    the ``Target`` so the applier's resolution helpers (``_is_delayed_target``,
    ``Target.build``) accept it unchanged.  ``_target_`` paths may carry
    ``(obj, qualname)`` for classes, but injection targets are module-level
    functions, so the dotted path splits at the last dot into
    (module, attribute).
    """
    from importlib import import_module

    if not isinstance(path, str) or not path:
        raise ValueError(f"frozen injection target path must be a non-empty string, got {path!r}")
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"frozen injection target path {path!r} is not a dotted path")
    fn = getattr(import_module(module_name), attr, None)
    if fn is None:
        raise ValueError(
            f"frozen injection target path {path!r} does not name a callable "
            f"({module_name!r} has no attribute {attr!r})"
        )
    if not callable(fn):
        raise TypeError(f"frozen injection target path {path!r} resolves to {type(fn).__name__}, not callable")
    return fn


def _injection_target(value: Any) -> Any:
    """Rebuild an injection factory/wrapper from its frozen form.

    ``freeze_injections`` serializes a ``Target`` via ``to_dict()``
    (``{"_target_": path, **kwargs}``) and a plain callable by ``__name__``.
    A dict is reconstructed into a real ``trainer.config.Target`` so the
    applier's ``_is_delayed_target`` / ``Target.build`` path applies; a string
    is left as the registry name the applier resolves the same way it would
    have at plan time.
    """
    if isinstance(value, dict) and "_target_" in value:
        from hyper_parallel.trainer.config import Target

        path = value["_target_"]
        return Target(
            _load_target_path(path),
            target_path=path,
            **{k: v for k, v in value.items() if k != "_target_"},
        )
    return value


class _FrozenBoundarySpec:
    """Minimal spec view over one frozen ``param_plan`` boundary entry.

    ``PrecompiledBoundary`` reads ``in_src`` / ``in_dst`` / ``out_src`` /
    ``out_dst`` / ``out_names`` off the spec object; the frozen entry stores
    them as JSON strings, so the adapter re-parses them into real placements
    once, up front.

    The injection helpers (``hyper_bind_compute`` and
    ``hyper_apply_inner_wrapper``) also hand this to the applier's resolution
    chain (``_resolve_local_compute_fn`` / ``_resolve_inner_wrapper``), so it
    additionally mirrors the injection fields those read.  ``injection`` (a
    frozen ``freeze_injections`` entry) supplies ``inner_wrapper`` /
    ``inner_target`` / ``inner_out_src`` / ``local_compute_fn`` (each
    re-imported from the frozen form); ``entry`` (the frozen ``param_plan``
    entry) supplies ``region_dispatch`` / ``out_src`` / ``out_names``.
    """

    def __init__(self, entry: dict[str, Any], injection: Optional[dict[str, Any]] = None):
        from hyper_parallel.codegen.plan.freeze import parse_named_placement

        self.params = parse_named_placement(entry.get("params") or {}) or {}
        self.in_src = parse_named_placement(entry.get("in_src") or {}) or {}
        self.in_dst = parse_named_placement(entry.get("in_dst") or {}) or {}
        self.out_src = (
            parse_named_placement(entry["out_src"]) if entry.get("out_src") else None
        )
        self.out_dst = (
            parse_named_placement(entry["out_dst"]) if entry.get("out_dst") else None
        )
        self.out_names = list(entry["out_names"]) if entry.get("out_names") else None
        self.region_dispatch = entry.get("region_dispatch")
        for key in ("inner_wrapper", "inner_target", "inner_out_src", "local_compute_fn"):
            if injection is not None and key in injection:
                setattr(self, key, _injection_target(injection[key]))


# One compiled PrecompiledBoundary per (entry, mesh, dim-names) triple — the
# boundary compiles the same literal every forward call, so the compiled op
# plan is cached (the mesh/dim-names stay constant for a run).
_BOUNDARY_CACHE: dict[tuple[Any, tuple[str, ...], str], Any] = {}


def _boundary_for_entry(entry: dict[str, Any], mesh: Any, mesh_dim_names=None):
    from hyper_parallel.distributed._builder.precompiled_boundary import (
        PrecompiledBoundary,
    )
    from hyper_parallel.distributed._builder.tp_collective_lowering import (
        create_tp_collective_lowerer,
    )

    dim_names = (
        mesh_dim_names
        or tuple(getattr(mesh, "mesh_dim_names", ()) or ())
    )
    key = (mesh, dim_names, json.dumps(entry, sort_keys=True))
    boundary = _BOUNDARY_CACHE.get(key)
    if boundary is None:
        boundary = PrecompiledBoundary(
            _FrozenBoundarySpec(entry),
            mesh,
            dim_names,
            op_lowerer=create_tp_collective_lowerer(mesh, dim_names),
        )
        _BOUNDARY_CACHE[key] = boundary
    return boundary


def _entry_has_ep_placement(entry: dict[str, Any]) -> bool:
    """Return whether the frozen boundary entry carries any ``ep`` placement.

    EP-boundary entries (e.g. ``*.mlp`` with ``when: ep``) have ``ep`` keys
    in their ``in_src``/``out_src``/etc. placement dicts.  Non-EP entries
    (attention, norm) only have ``tp``/``cp`` keys.
    """
    for field in ("in_src", "in_dst", "out_src", "out_dst"):
        placements = entry.get(field) or {}
        for named in placements.values():
            if isinstance(named, dict) and "ep" in named:
                return True
    return False


def hyper_redistribute(
    tensor,
    plan_entry: dict[str, Any],
    mesh_context,
    mesh_dim_names: Optional[tuple[str, ...]] = None,
    *,
    module: Any = None,
) -> Any:
    """Execute the frozen boundary layout conversion for one tensor / arg set.

    ``plan_entry`` is one frozen ``param_plan`` boundary entry (its
    ``in_src`` / ``in_dst`` / ``out_src`` / ``out_dst`` fields).  The
    conversion reuses ``PrecompiledBoundary`` — no new redistribution is
    written here.

    The call shape selects the side: a ``(args, kwargs)`` pair runs the input
    plan, anything else runs the output plan.  The mesh is derived from
    ``mesh_context`` (the dense active mesh — the same one the parameters
    sharded across, sliced to ``mesh_dim_names`` when given), so generated
    forward code needs no mesh plumbing.

    ``module`` (optional) binds the input plan's ops to the forward signature's
    positional indices (``sharding_applier._bind_input_indices``).  Generated
    forwarded code passes ``module=self`` on the input side; inter-module calls
    are positional, so without the binding the ``(args, kwargs)`` pair would
    silently skip redistribution on the ``in_src`` boundary.  Omitted on the
    output side, where the callable signature is irrelevant, and for callers
    that omit the optional module argument.
    """
    dense_mesh, _, expert_mesh, active_dim_names = (
        _resolve_runtime_meshes(mesh_context, mesh_dim_names)
    )
    if dense_mesh is None:
        raise ValueError(
            "hyper_redistribute requires a DeviceMesh / MeshContext to redistribute against"
        )
    # When the frozen plan declares no active TP/CP axes
    # (mesh_dim_names is empty — tp=1/cp=1), boundary placements on
    # tp/cp are identity. Two sub-cases:
    #   - No expert mesh (ep=1): all placements are identity → no-op.
    #   - Expert mesh exists (ep>1): EP-boundary entries (carrying ``ep``
    #     placements) must still redistribute on the expert mesh (which
    #     has no FSDP axes, avoiding the uneven-shard NotImplementedError).
    #     Non-EP entries are still identity → no-op.
    if not active_dim_names:
        if expert_mesh is not None and _entry_has_ep_placement(plan_entry):
            boundary = _boundary_for_entry(plan_entry, expert_mesh, None)
        else:
            return tensor
    else:
        boundary = _boundary_for_entry(plan_entry, dense_mesh, active_dim_names)
    if module is not None:
        from hyper_parallel.distributed._builder.forward_rewriter import (
            _bind_input_indices,
        )

        _bind_input_indices(boundary, module)
    if isinstance(tensor, tuple) and len(tensor) == 2 and isinstance(tensor[1], dict):
        args, kwargs = tensor
        return boundary.redistribute_inputs(args, kwargs)
    return boundary.redistribute_outputs(tensor)


#: Output names are not retained on the frozen entry; a local-region forward
#: wraps its local results per ``out_src`` before the boundary exit, so read the
#: declared order off the same place ``_rewrap_local_outputs`` does.
def _declared_out_names(entry: dict[str, Any]) -> list[str]:
    declared = entry.get("out_src") or {}
    out_names = entry.get("out_names") or declared.keys()
    return list(out_names)


def hyper_rewrap_outputs(
    output,
    plan_entry: dict[str, Any],
    mesh_context,
    mesh_dim_names: Optional[tuple[str, ...]] = None,
) -> Any:
    """Wrap a local-region forward's local tensors into DTensors.

    Mirrors ``sharding_applier._rewrap_local_outputs``: every non-DTensor
    Tensor that ``out_src`` names is re-homed to the boundary's exit mesh with
    its declared placement, so ``hyper_redistribute`` (the boundary exit) sees
    a sharded tensor rather than a bare local one.

    Works on a tuple/list/scalar the same way the applier does; a declared
    output that is already a DTensor (or ``None``) is passed through.  The mesh
    is derived from ``mesh_context`` like ``hyper_redistribute``.
    """
    declared = plan_entry.get("out_src") or {}
    if not declared:
        return output
    dense_mesh, _, _, active_dim_names = (
        _resolve_runtime_meshes(mesh_context, mesh_dim_names)
    )
    if dense_mesh is None:
        raise ValueError(
            "hyper_rewrap_outputs requires a DeviceMesh / MeshContext to rewrap against"
        )
    # Placements must line up with the dense sub-mesh's own axis order, exactly
    # like the boundary compile in ``hyper_redistribute`` (which falls back to
    # the mesh's declared axes when the plan names none).
    dim_names = (
        active_dim_names
        or tuple(getattr(dense_mesh, "mesh_dim_names", ()) or ())
    )
    from hyper_parallel.distributed.recipe_spec import (
        resolve_placements,
    )
    # The custom DTensor (``hyper_parallel.core.dtensor``) wraps the custom
    # DeviceMesh and *its* ``from_local`` accepts the custom placement types
    # ``resolve_placements`` yields.  torch's ``DTensor.from_local`` cannot
    # cast those placements ("Unable to cast ... Replicate to C++ type"), so
    # never use ``torch.distributed.tensor.DTensor`` here — mirror the native
    # ``_rewrap_local_outputs`` exactly.
    from hyper_parallel.core.dtensor.dtensor import DTensor
    import torch

    is_sequence = isinstance(output, (tuple, list))
    items = list(output) if is_sequence else [output]
    out_names = _declared_out_names(plan_entry)
    name_to_idx = {name: index for index, name in enumerate(out_names)}

    for out_name, named_placement in declared.items():
        index = name_to_idx.get(out_name)
        if index is None:
            raise ValueError(
                f"hyper_rewrap_outputs: out_src declares output {out_name!r}, "
                f"but out_names={out_names!r} does not contain it"
            )
        # Out_src is a frozen JSON contract: its placement leaves are
        # canonical strings ("S(1)"/"R"/"P" ...).  resolve_placements would
        # hand those strings straight to DTensor.from_local, and a string
        # sequence trips _build_layout's alias branch — which demands the
        # placement count equal the tensor rank. The extended MoE identity
        # contract declares 3 axes ({cp,ep,tp}) but the dense mesh carries only
        # 2, so resolving leaves 2 strings against a 3-D (B,S,H) output and
        # fails.  Parse the leaves back into real Placement objects first (the
        # same step _FrozenBoundarySpec applies for hyper_redistribute / the
        # native _rewrap_local_outputs runs on already-object contracts); the
        # non-alias path then maps the 2-axis MeshPlacement onto the 3-D
        # tensor, padding the unpredicted dim with Replicate().
        from hyper_parallel.codegen.plan.freeze import parse_named_placement
        named_placement = parse_named_placement(named_placement)
        if index >= len(items):
            raise ValueError(
                f"hyper_rewrap_outputs: out_src maps output {out_name!r} to index "
                f"{index}, but forward returned only {len(items)} output(s)"
            )
        item = items[index]
        if item is None:
            continue
        if isinstance(item, DTensor):
            continue
        if not isinstance(item, torch.Tensor):
            raise TypeError(
                f"hyper_rewrap_outputs: declared output {out_name!r} at index "
                f"{index} must be a Tensor or None, got {type(item).__name__}"
            )
        placements = tuple(resolve_placements(named_placement, dim_names))
        items[index] = DTensor.from_local(item, dense_mesh, placements)

    if isinstance(output, tuple):
        return tuple(items)
    if isinstance(output, list):
        return items
    if len(items) != 1:
        raise ValueError(
            f"hyper_rewrap_outputs: scalar forward output cannot satisfy "
            f"{len(declared)} declared out_src entries"
        )
    return items[0]


def hyper_to_local_if_dtensor(output):
    """Convert a local-region boundary output back to a local tensor after exit."""
    to_local = getattr(output, "to_local", None)
    if callable(to_local):
        return to_local()
    return output


# ---------------------------------------------------------------------------
# Compute binding and inner-wrapper installation
# ---------------------------------------------------------------------------

def _injection_spec(injections: list[dict[str, Any]], param_plan: dict[str, Any],
                    fqn: str) -> Optional[dict[str, Any]]:
    """Find one frozen boundary's injection declaration by its FQN."""
    for rule in injections:
        if rule.get("match") == fqn:
            return rule
    return None


def hyper_bind_compute(
    model,
    injections: list[dict[str, Any]],
    mesh_context,
    mesh_dim_names: Optional[tuple[str, ...]] = None,
    *,
    param_plan: Optional[dict[str, Any]] = None,
) -> None:
    """Bind ``module.__hyper_compute__`` for every local-region boundary.

    A generated local-compute forward calls
    ``self.__hyper_compute__(*args, **kwargs)`` on the redistributed inputs.
    The bound callable is the applier's local-compute resolution
    (``_resolve_local_compute_fn`` → ``functools.partial(compute_fn, module)``),
    so the compute function's first argument is the boundary module and the
    remaining arguments are the redistributed forward inputs.

    ``injections`` is the frozen ``meta.injections`` list (``freeze_injections``
    shape); only entries with an explicit ``local_compute_fn`` are bound (the
    ``region_dispatch=False`` gate that would otherwise fall back to
    ``module.forward`` is deliberately NOT bound — the applier runs that path
    by re-entering ``forward``, which the generated file already does as
    ``_forward_impl``).  ``param_plan`` (``meta.param_plan``) supplies
    ``region_dispatch`` / ``ep_size`` for the factory's mesh context, exactly
    as the applier reads them off a live spec.
    """
    from hyper_parallel.distributed._builder.rule_resolver import (
        _resolve_local_compute_fn,
    )
    from hyper_parallel.distributed._builder.parameter_sharding import _resolve_module

    dense_mesh, _, expert_mesh, active_dim_names = (
        _resolve_runtime_meshes(mesh_context, mesh_dim_names)
    )
    if dense_mesh is None:
        raise ValueError(
            "hyper_bind_compute requires a DeviceMesh / MeshContext to bind compute against"
        )
    if not injections:
        return
    for rule in injections:
        if rule.get("local_compute_fn") is None:
            continue
        fqn = rule.get("match")
        if not fqn:
            raise ValueError(
                f"hyper_bind_compute: injection rule {rule!r} has no 'match' FQN"
            )
        entry = (param_plan or {}).get(fqn, {})
        spec = _FrozenBoundarySpec(entry, rule)
        module = _resolve_module(model, fqn)
        compute_fn = _resolve_local_compute_fn(
            module, spec, dense_mesh, active_dim_names, expert_mesh)
        # _resolve_local_compute_fn returns module.forward for the
        # region_dispatch=False gate; that path is not a bound compute (the
        # generated forward already calls _forward_impl), so it must not be
        # bound as __hyper_compute__.
        if compute_fn is None or compute_fn is module.forward:
            continue
        module.__hyper_compute__ = compute_fn


def hyper_apply_inner_wrapper(
    model,
    injections: list[dict[str, Any]],
    mesh_context,
    mesh_dim_names: Optional[tuple[str, ...]] = None,
    *,
    param_plan: Optional[dict[str, Any]] = None,
) -> None:
    """Install the declared inner CP wrapper on its target module.

    A generated redistribution forward delegates computation to
    ``_forward_impl``.  The
    CP K/V all-gather wrapper is applied to ``target.forward`` IN PLACE by the
    applier (``_wrap_inner_attention``), which is not bindable as a method
    without changing its invocation contract.  This helper repeats that
    one-shot installation from the frozen ``meta.injections``, so the
    wrapper's ``compute`` runs inside the ``_forward_impl`` call at train time.

    Uses the applier's own resolution chain — ``_wrap_inner_attention``
    resolves the registry name / Target / callable, resolves the target, and
    installs the dual-mode adapter (production pass-through / validate local
    re-wrap) via ``_install_inner_adapter``.
    """
    from hyper_parallel.distributed._builder.applier import (
        _get_cp_submesh,
        _get_ep_submesh,
        _get_tp_submesh,
    )
    from hyper_parallel.distributed._builder.forward_rewriter import (
        _wrap_inner_attention,
    )
    from hyper_parallel.distributed._builder.parameter_sharding import _resolve_module

    dense_mesh, _, expert_mesh, active_dim_names = (
        _resolve_runtime_meshes(mesh_context, mesh_dim_names)
    )
    if dense_mesh is None:
        raise ValueError(
            "hyper_apply_inner_wrapper requires a DeviceMesh / MeshContext to install against"
        )
    if not injections:
        return
    dim_names = tuple(active_dim_names) if active_dim_names else ()
    cp_mesh = _get_cp_submesh(dense_mesh, dim_names)
    tp_mesh = _get_tp_submesh(dense_mesh, dim_names)
    ep_mesh = _get_ep_submesh(dense_mesh, dim_names) or expert_mesh
    for rule in injections:
        if rule.get("inner_wrapper") is None:
            continue
        fqn = rule.get("match")
        if not fqn:
            raise ValueError(
                f"hyper_apply_inner_wrapper: injection rule {rule!r} has no "
                "'match' FQN"
            )
        entry = (param_plan or {}).get(fqn, {})
        spec = _FrozenBoundarySpec(entry, rule)
        module = _resolve_module(model, fqn)
        # The applier's boundary loop runs _resolve_local_compute_fn even for
        # inner-wrap boundaries, and the preflight fails fast on a missing
        # region_dispatch.  Reuse the exact applier entry here for parity.
        _wrap_inner_attention(
            module,
            cp_mesh,
            spec=spec,
            mesh=dense_mesh,
            mesh_dim_names=dim_names,
            tp_mesh=tp_mesh,
            ep_mesh=ep_mesh,
        )


def _is_boundary_entry(entry: Any) -> bool:
    """Return whether a frozen param-plan entry represents a boundary."""
    if not isinstance(entry, dict):
        return False
    if entry.get("is_boundary") is not None:
        return bool(entry["is_boundary"])
    return any(entry.get(field) for field in ("in_src", "in_dst", "out_src", "out_dst"))


def _has_lowered_forward(module: Any) -> bool:
    """Return whether the generated source already owns this boundary wrapper."""
    return hasattr(module, "_forward_impl")


def _is_tp_sequence_parallel_embed_entry(entry: dict[str, Any]) -> bool:
    """Return whether an embedding output is reduce-scattered over sequence."""
    out_src = (entry.get("out_src") or {}).get("output") or {}
    out_dst = (entry.get("out_dst") or {}).get("output") or {}
    return (
        str(out_src.get("tp", "")).startswith("P(")
        and out_dst.get("tp") == "S(1)"
    )


def _wrap_default_position_ids(module: Any) -> None:
    """Build default position ids before an embedding sequence reduce-scatter.

    Transformers commonly derives missing position ids from ``inputs_embeds``.
    A generated model's vocab-parallel embedding may already have changed that
    tensor from TP-replicated sequence length to a TP sequence shard.  Supplying
    the default from the still-replicated ``input_ids`` keeps rotary embeddings
    in the attention boundary's declared TP-replicated layout.
    """
    if getattr(module, "_hyper_codegen_position_ids_wrapped", False):
        return
    original_forward = module.forward
    signature = inspect.signature(original_forward)
    if (
        "input_ids" not in signature.parameters
        or "position_ids" not in signature.parameters
    ):
        return
    positional_names = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    position_index = (
        positional_names.index("position_ids")
        if "position_ids" in positional_names
        else None
    )

    @functools.wraps(original_forward)
    def codegen_position_ids_forward(*args: Any, **kwargs: Any) -> Any:
        """Supply replicated default position ids to the original model forward."""
        bound = signature.bind_partial(*args, **kwargs)
        if bound.arguments.get("position_ids") is not None:
            return original_forward(*args, **kwargs)
        input_ids = bound.arguments.get("input_ids")
        if input_ids is None:
            return original_forward(*args, **kwargs)

        past_key_values = bound.arguments.get("past_key_values")
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        import torch  # pylint: disable=C0415

        position_ids = torch.arange(
            int(input_ids.shape[-1]),
            dtype=input_ids.dtype,
            device="cpu",
        )
        position_ids = (position_ids + past_seen_tokens).unsqueeze(0).to(input_ids.device)
        if position_index is not None and position_index < len(args):
            call_args = list(args)
            call_args[position_index] = position_ids
            return original_forward(*call_args, **kwargs)
        call_kwargs = kwargs.copy()
        call_kwargs["position_ids"] = position_ids
        return original_forward(*args, **call_kwargs)

    module.forward = codegen_position_ids_forward
    module._hyper_codegen_position_ids_wrapped = True
    module._hyper_codegen_original_position_ids_forward = original_forward


def _local_tensor_shape(tensor: Any) -> tuple[int, ...]:
    """Return local shard shape for Hyper DTensor-backed parameters."""
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return tuple(int(dim) for dim in to_local().shape)
    return tuple(int(dim) for dim in tensor.shape)


def _wrap_codegen_vocab_parallel_embedding(module: Any, tp_mesh: Any) -> None:
    """Install masked vocab-parallel embedding using the local vocab shard."""
    if getattr(module, "_hyper_codegen_vocab_parallel_wrapped", False):
        return
    original_forward = module.forward
    local_shape = _local_tensor_shape(module.weight)
    if not local_shape:
        raise ValueError("codegen vocab-parallel embedding expects a rank >= 1 weight")
    v_local = local_shape[0]
    lo = tp_mesh.get_local_rank() * v_local
    hi = lo + v_local

    @functools.wraps(original_forward)
    def masked_embedding_forward(input_ids: Any, *args: Any, **kwargs: Any) -> Any:
        import torch  # pylint: disable=C0415

        mask = (input_ids >= lo) & (input_ids < hi)
        local_ids = torch.where(mask, input_ids - lo, torch.zeros_like(input_ids))
        out = original_forward(local_ids, *args, **kwargs)
        return out * mask.unsqueeze(-1).to(out.dtype)

    module.forward = masked_embedding_forward
    module._hyper_codegen_vocab_parallel_wrapped = True
    module._hyper_codegen_original_vocab_parallel_forward = original_forward


def _wrap_module_boundary_forward(
    module: Any,
    entry: dict[str, Any],
    mesh_context: Any,
    mesh_dim_names: Optional[tuple[str, ...]],
) -> None:
    """Install native-style boundary entry/exit redistribution on one module."""
    if getattr(module, "_hyper_codegen_boundary_wrapped", False):
        return
    original_forward = module.forward

    @functools.wraps(original_forward)
    def codegen_boundary_forward(*args: Any, **kwargs: Any) -> Any:
        redist_args, redist_kwargs = hyper_redistribute(
            (args, kwargs),
            entry,
            mesh_context,
            mesh_dim_names,
            module=module,
        )
        outputs = original_forward(*redist_args, **redist_kwargs)
        outputs = hyper_redistribute(outputs, entry, mesh_context, mesh_dim_names)
        return outputs

    module.forward = codegen_boundary_forward
    module._hyper_codegen_boundary_wrapped = True
    module._hyper_codegen_original_forward = original_forward


def hyper_wrap_module_boundaries(
    model: Any,
    param_plan: dict[str, Any],
    mesh_context: Any,
    mesh_dim_names: Optional[tuple[str, ...]] = None,
) -> None:
    """Wrap frozen boundaries whose class forward was not sunk into source.

    AST lowering rewrites classes defined in the copied modeling file
    and marks them with ``_forward_impl``.  Imported module classes such as
    ``nn.Embedding`` / ``nn.Linear`` have no source span to rewrite, but their
    frozen entries can still carry real boundary communication, for example
    embedding's TP sequence-parallel ``Partial -> Shard`` output transition.
    This helper installs the same entry/original-forward/exit wrapper on the
    live module instance so those contracts are not dropped in the generated
    path.
    """
    if not param_plan:
        return
    from hyper_parallel.distributed._builder.applier import (
        _get_tp_submesh,
    )
    from hyper_parallel.distributed._builder.forward_rewriter import (
        _is_vocab_parallel_embed,
    )
    from hyper_parallel.distributed._builder.parameter_sharding import _resolve_module
    meshes_resolved = False
    tp_mesh = None

    for fqn, entry in sorted(param_plan.items()):
        if _is_tp_sequence_parallel_embed_entry(entry):
            parent_fqn = fqn.rpartition(".")[0]
            _wrap_default_position_ids(_resolve_module(model, parent_fqn))
        if not _is_boundary_entry(entry):
            continue
        module = _resolve_module(model, fqn)
        if _has_lowered_forward(module):
            continue
        spec = _FrozenBoundarySpec(entry)
        if spec.params:
            if not meshes_resolved:
                dense_mesh, _, _, active_dim_names = _resolve_runtime_meshes(
                    mesh_context, mesh_dim_names
                )
                tp_mesh = _get_tp_submesh(dense_mesh, tuple(active_dim_names))
                meshes_resolved = True
        if _is_vocab_parallel_embed(module, spec, tp_mesh):
            _wrap_codegen_vocab_parallel_embedding(module, tp_mesh)
        _wrap_module_boundary_forward(module, entry, mesh_context, mesh_dim_names)


# ---------------------------------------------------------------------------
# Generated entry-point dispatch
# ---------------------------------------------------------------------------

def parallelize_from_generated(
    model,
    mesh_context,
    artifact_dir: str,
    *,
    hf_config: Any = None,
) -> dict[str, tuple[Any, Any]]:
    """Run the generated module's ``hyper_parallelize`` entry point.

    The entry point is taken from ``meta.entrypoints["parallelize"]`` and must
    return ``tp_grad_info`` with the same semantics as
    ``apply_sharding_plan()``.  Infrastructure validates the metadata before
    dispatch, so this function repeats the check only when called directly
    with ``hf_config``.
    """
    meta = load_codegen_meta(artifact_dir)
    if meta is None:
        raise FileNotFoundError(
            f"codegen: no meta at {os.path.join(artifact_dir, 'codegen_meta.json')}; "
            "cannot parallelize from the generated module"
        )
    if hf_config is not None:
        verify_codegen_signature(meta, hf_config)
    entry_name = meta.entrypoints.get("parallelize")
    if not entry_name:
        raise KeyError(
            "codegen: meta.entrypoints['parallelize'] is missing; the generated "
            "module has no hyper_parallelize entry to call"
        )
    module = import_generated_module(artifact_dir)
    entry = getattr(module, entry_name, None)
    if entry is None:
        raise AttributeError(
            f"codegen: generated module {module.__name__} has no entry point "
            f"{entry_name!r} (recorded in meta.entrypoints['parallelize'])"
        )
    return entry(model, mesh_context)


# ---------------------------------------------------------------------------
# Module replacement
# ---------------------------------------------------------------------------

def _module_type_for_generated(model: Any, path: str) -> Optional[type]:
    """Resolve a replacement ``module_type`` against the generated module.

    The generated artifact *copies* the modeling source, so classes it defines
    locally (``Qwen3MoeRMSNorm`` etc.) are re-created inside the generated
    module's namespace — a different type object from the installed
    ``transformers`` one the source path names.  A ``replace_module`` spec
    whose ``module_type`` still points at the installed class therefore fails
    ``compile_module_replacements``' type check at runtime even though the
    generation-time compile (which builds the meta model from the *installed*
    source) passed.

    Recover the generated module from the live model's class and look the leaf
    symbol up there; return ``None`` when it is not defined in that namespace so
    the caller falls back to the installed path.  ``model`` is always a
    ``PreTrainedModel`` instance of the generated entry class, so
    ``type(model).__module__`` is stable (and present in ``sys.modules`` — the
    loader registers it).
    """
    if not isinstance(path, str) or not path:
        return None
    leaf = path.rsplit(".", 1)[-1]
    module_name = type(model).__module__
    module = sys.modules.get(module_name)
    if module is None:
        return None
    candidate = getattr(module, leaf, None)
    if candidate is None or not isinstance(candidate, type):
        return None
    return candidate


def hyper_apply_replacements(model: Any, overrides: list[dict[str, Any]]) -> Any:
    """Apply the generated ``_HYPER_MODULE_OVERRIDES`` literal to ``model``.

    Rebuilds :class:`ModuleReplacementSpec` objects from the frozen records —
    the ``module_type`` and raw ``factory`` re-imported from their dotted paths
    (the replacement factory is a per-entry closure that cannot be a literal),
    the settled FQN set restored from ``fqns`` (the parent spec's ``match``
    globs are *not* replayed — each record is one already-selected target, so
    its ``match`` is the frozen selection itself), and ``exact_type`` carried
    through — then runs ``compile_module_replacements`` +
    ``apply_module_replacements`` on the live model.  Called from the generated
    entry class's ``__init__`` tail.

    ``module_type`` is resolved against the generated module's own namespace
    first (the artifact copies the modeling source, so a locally-defined class
    is a *different type object* from the installed one its dotted path names);
    only when the leaf class is not defined there does it fall back to the
    installed import path.

    Imports torch-affine machinery lazily: this module stays importable on a
    host without torch, so ``hyper_apply_replacements`` must not pull the torch
    import graph in until it actually runs.
    """
    if not overrides:
        return model
    from hyper_parallel.components.checkpoint.weight_conversion import (  # pylint: disable=import-outside-toplevel
        get_model_conversion_mapping,
    )
    from hyper_parallel.models.replacement import (
        ModuleReplacementSpec,
        apply_module_replacements,
        compile_module_replacements,
    )
    from hyper_parallel.trainer.config.parallelism import _import_module_type

    specs = []
    for record in overrides:
        # ``compile_overrides_for_meta`` expands each YAML spec into one
        # record per matched target.  The parent spec's ``match`` patterns are
        # broad globs (``*.input_layernorm``), so replaying them at runtime
        # would make every record of the same spec claim the same modules and
        # trip ``compile_module_replacements``' one-factory-per-source check.
        # The record's frozen ``fqns`` *is* the settled selection, so rebuild
        # the spec's ``match`` from it. The runtime executes the frozen
        # selection instead of matching the model tree again; parent patterns
        # remain as a compatibility fallback for literals without an FQN set.
        match = record.get("fqns") or record.get("fqn")
        if match is None:
            match = record.get("match") or ()
        if isinstance(match, str):
            match = (match,)
        factory = _load_target_path(record["factory"])
        module_type = _module_type_for_generated(model, record["module_type"])
        if module_type is None:
            module_type = _import_module_type(record["module_type"])
        specs.append(
            ModuleReplacementSpec(
                match=tuple(match),
                factory=factory,
                module_type=module_type,
                exact_type=bool(record.get("exact_type", False)),
            )
        )
    plan = compile_module_replacements(model, specs)
    model, _ = apply_module_replacements(
        model,
        plan,
        weights_mapping=get_model_conversion_mapping(model),
    )
    return model


__all__ = [
    "hyper_apply_inner_wrapper",
    "hyper_apply_replacements",
    "hyper_apply_special_handlers",
    "hyper_bind_compute",
    "hyper_build_tp_grad_info",
    "hyper_redistribute",
    "hyper_replicate_tied",
    "hyper_rewrap_outputs",
    "hyper_shard_params",
    "hyper_to_local_if_dtensor",
    "hyper_wrap_module_boundaries",
    "import_generated_module",
    "load_codegen_meta",
    "parallelize_from_generated",
    "verify_codegen_signature",
]
