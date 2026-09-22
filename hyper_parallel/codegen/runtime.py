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
recorded in ``codegen_meta.json``.  Execution is **convergent with the native
applier**: parameter sharding (Phase A ``_shard_planned_parameters``), special
handlers (Phase B) and tied weights (Phase D) run against a live
``ShardingPlan`` rebuilt from the meta (:func:`_rebuild_live_plan_from_meta`),
so codegen no longer keeps a second literal sharding implementation.  Only the
forward assembly (the codegen Phase C helpers ``hyper_install_boundaries`` /
``hyper_bind_compute`` / ``hyper_apply_inner_wrapper``) stays literal, because
the generated file pre-rewrites forward and the native Phase C would
double-wrap that redistribution.

Helpers fail fast: any unknown placement form, missing entrypoint, or absent
meta errors instead of silently degrading to a ``Replicate()`` no-op.
"""
from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import Any, Optional

from hyper_parallel.codegen.loader import import_generated_module
from hyper_parallel.codegen.meta import (
    CodegenMeta,
    load_codegen_meta as _load_codegen_meta_from_path,
    meta_to_dict,
    validate_meta_schema,
)
# Shared boundary-form classifier (module-level imports are stdlib-only, so
# this stays importable without torch).  The EP-dependence criterion must be
# the same one generation used to pick the boundary's forward form: an entry
# whose ``ep`` placement actually *changes* routes to the expert mesh; an
# identity ``R -> R`` key (the frozen plan's "not EP-dependent" marker) is not
# a dependency, so a degenerate topology (tp=cp=ep=1) prunes such boundaries
# to the identity form while the install path is skipped entirely.
from hyper_parallel.codegen.plan.boundary_forms import (
    _entry_has_ep_placement,
    declared_out_names,
    is_boundary_entry,
)
# Pure placement parsing (freeze.py imports no torch), needed by both the
# meta->plan rebuild and its per-field deserializer below.
from hyper_parallel.codegen.plan.freeze import parse_named_placement

# ``sharding/apply`` imports torch at module level, so it is deliberately not
# imported here: runtime.py must stay importable without torch (generation-time
# hosts and the local smoke tests import this module for its pure helpers).  The
# apply helpers are imported lazily inside the one function that needs each.

logger = logging.getLogger(__name__)


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

    Delegates to the native ``_get_active_mesh_with_names`` — the slicing and
    reordering contract lives in one place; this wrapper keeps no mirrored
    implementation to stay in sync.

    ``apply_sharding_plan`` aligns its sharding mesh to ``plan.mesh_dim_names``
    (the axes the placements were resolved against) by slicing the device
    mesh.  The runtime receives the same
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
    from hyper_parallel.distributed._builder.applier import (  # pylint: disable=C0415
        _get_active_mesh_with_names,
    )
    return _get_active_mesh_with_names(mesh, mesh_dim_names)


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
    are the true ``tp`` / ``ep`` child meshes).

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
# TP gradient metadata (delegated to the native builder:
# parameter_sharding._build_runtime_source_shard_info ->
# source_shard.build_source_shard_info)
# ---------------------------------------------------------------------------

def hyper_build_tp_grad_info(
    model,
    plan,
    dense_source_mesh,
    expert_source_mesh,
) -> Optional[dict[str, tuple[Any, Any]]]:
    """Build FSDP2 source metadata through the native builder.

    Delegates to the SAME entry the applier runs —
    ``parameter_sharding._build_runtime_source_shard_info``, which unwraps the
    production parameters once (``_local_params_context``) and calls
    ``source_shard.build_source_shard_info``.  Codegen holds no second copy of
    that logic, so the two paths cannot drift:

    - every planned parameter maps to its own declared placement, resolved over
      the source mesh's non-FSDP axes, recorded as a tuple (FSDP2 iterates the
      placements, so a bare ``Shard(0)`` would crash with "'Shard' object is
      not iterable");
    - ``experts.*`` parameters of an EP spec (``spec._ep_size > 0``) map to the
      expert EP source mesh;
    - ``build_source_shard_info``'s FSDP-owned-axis rejection applies here too;
    - tied pairs normalize to one placement (``Shard`` takes precedence).

    ``plan`` is the live ``ShardingPlan`` rebuilt from the frozen meta
    (:func:`_rebuild_live_plan_from_meta`); the two source meshes come from
    ``parameter_sharding._resolve_parameter_source_meshes``, exactly as
    ``apply_sharding_plan`` resolves them (``build_source_shard_info`` strips
    the FSDP-owned axes itself).

    The one-shot ``_local_params_context`` unwrap happens inside the native
    builder — after this call the model's DTensor parameters are permanently
    plain locals, so ``hyper_build_tp_grad_info`` must be the LAST sharding
    helper the generated ``hyper_parallelize`` runs (the applier's own ordering
    invariant).
    """
    from hyper_parallel.distributed._builder.parameter_sharding import (
        _build_runtime_source_shard_info,
    )

    return _build_runtime_source_shard_info(
        [model], plan, dense_source_mesh, expert_source_mesh, False
    )


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
        # D-22: the native suppression/restore pair reads this off the spec
        # object it is handed, so the frozen view mirrors it.
        self._deferred_bias_params = tuple(entry.get("deferred_bias_params") or ())
        for key in ("inner_wrapper", "inner_target", "inner_out_src", "local_compute_fn"):
            if injection is not None and key in injection:
                setattr(self, key, _injection_target(injection[key]))


def _build_rewrap_plan(
    plan_entry: dict[str, Any],
    dim_names: tuple[str, ...],
) -> list[tuple[int, str, tuple]]:
    """Pre-parse one frozen entry's ``out_src`` into executable rewrap ops.

    Returns ``[(output_index, out_name, placements), ...]``.  ``InstalledBoundary``
    resolves once at install time so the per-forward path never re-parses.
    """
    from hyper_parallel.codegen.plan.freeze import parse_named_placement
    from hyper_parallel.distributed.recipe_spec import resolve_placements

    declared = plan_entry.get("out_src") or {}
    out_names = declared_out_names(plan_entry)
    name_to_idx = {name: index for index, name in enumerate(out_names)}
    plan: list[tuple[int, str, tuple]] = []
    for out_name, named_placement in declared.items():
        index = name_to_idx.get(out_name)
        if index is None:
            raise ValueError(
                f"InstalledBoundary.rewrap_outputs: out_src declares output {out_name!r}, "
                f"but out_names={list(out_names)!r} does not contain it"
            )
        # Out_src is a frozen JSON contract: its placement leaves are
        # canonical strings ("S(1)"/"R"/"P" ...).  resolve_placements would
        # hand those strings straight to DTensor.from_local, and a string
        # sequence trips _build_layout's alias branch — which demands the
        # placement count equal the tensor rank. The extended MoE identity
        # contract declares 3 axes ({cp,ep,tp}) but the dense mesh carries only
        # 2, so resolving leaves 2 strings against a 3-D (B,S,H) output and
        # fails.  Parse the leaves back into real Placement objects first (the
        # same step _FrozenBoundarySpec applies for the boundary compile / the
        # native _rewrap_local_outputs runs on already-object contracts); the
        # non-alias path then maps the 2-axis MeshPlacement onto the 3-D
        # tensor, padding the unpredicted dim with Replicate().
        named_placement = parse_named_placement(named_placement)
        plan.append(
            (index, out_name, tuple(resolve_placements(named_placement, dim_names)))
        )
    return plan


def hyper_to_local_if_dtensor(output):
    """Convert a local-region boundary output back to a local tensor after exit.

    Named by the emitted local-region forward template (``emit.parallel``), so
    it stays until that template goes away — see the boundary-ownership item in
    the restructuring plan.
    """
    to_local = getattr(output, "to_local", None)
    if callable(to_local):
        return to_local()
    return output


# ---------------------------------------------------------------------------
# Install-time boundary compilation (one-shot; generated forwards bind these)
# ---------------------------------------------------------------------------

def _compile_boundary(entry: dict[str, Any], mesh: Any, mesh_dim_names):
    """Compile one frozen entry into a ``PrecompiledBoundary`` (uncached).

    The generated path compiles per module at install time: ``_bind_input_indices``
    mutates ``op.arg_index`` against one module's forward signature, and two classes
    with identical contracts but different signatures must not share a plan.
    """
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
    return PrecompiledBoundary(
        _FrozenBoundarySpec(entry),
        mesh,
        dim_names,
        op_lowerer=create_tp_collective_lowerer(mesh, dim_names),
    )


class InstalledBoundary:
    """One boundary's compiled redistribute plan, bound to a module instance.

    ``hyper_install_boundaries`` resolves everything once, at install time
    rather than on every forward call: the dense/expert mesh routing (the
    empty-active-axes no-op, the ep-entry expert-mesh compile, the ep-key drop
    when the dense mesh is active), the active-axis slicing, the
    rank-order/backend lowerer validation, and the input-index binding.
    Generated forwards then call
    ``self._hyper_boundary.redistribute_inputs/outputs`` with zero per-call
    resolution; the execution semantics — identity passthrough, DTensor
    unwrap on identity ops, local collectives, ``DTensor.redistribute``
    fallback — are the compiled ``PrecompiledBoundary``'s own.
    """

    def __init__(
        self,
        entry: dict[str, Any],
        mesh_context: Any,
        mesh_dim_names: Optional[tuple[str, ...]] = None,
        *,
        module: Any = None,
    ) -> None:
        dense_mesh, _, expert_mesh, active_dim_names = (
            _resolve_runtime_meshes(mesh_context, mesh_dim_names)
        )
        if dense_mesh is None:
            raise ValueError(
                "hyper_install_boundaries requires a DeviceMesh / MeshContext "
                "to compile boundaries against"
            )
        # Routing facts the tp_collective re-validation reads: the dense mesh
        # and the active axis names this boundary actually compiled against
        # (empty = the no-op / expert-mesh branches below).
        self.dense_mesh = dense_mesh
        self.active_dim_names = tuple(active_dim_names or ())
        self.noop = False
        self._boundary = None
        if not active_dim_names:
            if expert_mesh is not None and _entry_has_ep_placement(entry):
                self._boundary = _compile_boundary(entry, expert_mesh, None)
            else:
                # Boundary-exit semantics for this branch: the payload is
                # returned untouched (no DTensor unwrap either).
                self.noop = True
        else:
            self._boundary = _compile_boundary(entry, dense_mesh, active_dim_names)
        if self._boundary is not None and module is not None:
            from hyper_parallel.distributed._builder.forward_rewriter import (
                _bind_input_indices,
            )

            _bind_input_indices(self._boundary, module)

        self._rewrap_plan: Optional[list[tuple[int, str, tuple]]] = None
        self._rewrap_mesh = None
        if entry.get("out_src"):
            rewrap_dims = (
                tuple(active_dim_names)
                or tuple(getattr(dense_mesh, "mesh_dim_names", ()) or ())
            )
            self._rewrap_plan = _build_rewrap_plan(entry, rewrap_dims)
            self._rewrap_mesh = dense_mesh

    def redistribute_inputs(self, payload):
        """Run the input plan on the ``(args, kwargs)`` pair.

        Accepts the pair the generated forward builds and takes it as one
        positional argument, so the lowered call sites stay uniform across
        boundary forms.
        """
        if self.noop:
            return payload
        args, kwargs = payload
        return self._boundary.redistribute_inputs(args, kwargs)

    def redistribute_outputs(self, outputs):
        """Run the output plan on the forward's tensor output."""
        if self.noop:
            return outputs
        return self._boundary.redistribute_outputs(outputs)

    def rewrap_outputs(self, output):
        """Re-wrap local region outputs into DTensors per ``out_src``.

        ``_build_rewrap_plan`` pre-resolves the placements at install time; the
        execution is the shared native local-region mechanism
        (:func:`~hyper_parallel.distributed._builder.forward_rewriter.rewrap_declared_outputs`),
        so the codegen and native paths cannot drift apart.
        """
        if self._rewrap_plan is None:
            return output
        from hyper_parallel.distributed._builder.forward_rewriter import (  # pylint: disable=import-outside-toplevel
            rewrap_declared_outputs,
        )

        return rewrap_declared_outputs(
            output,
            self._rewrap_plan,
            self._rewrap_mesh,
            label="InstalledBoundary.rewrap_outputs",
        )


class TPOperators:
    """Bare-operator runtime bound to a statically lowered boundary forward.

    A ``tp_collective`` template calls ``self._hyper_tp.<kind>(...)`` directly
    (``emit/parallel``'s static form); ``hyper_install_boundaries`` binds this
    object only after re-validating that form against the live mesh, so the
    baked calls never run unvalidated.  Each method is instruction-equivalent
    to the compiled engine running the same transition (``RedistOp.execute``
    with the TP lowerer attached): a DTensor input unwraps to its local shard
    first, ``None`` passes through untouched (the compiled plan's skip), and
    the collective dispatches through ``TPCollectiveLowerer.execution_op`` —
    so a gloo ``reduce_scatter`` remaps to all_reduce + local chunk exactly
    as the placement-driven path does.
    """

    def __init__(self, lowerer: Any) -> None:
        """Capture the install-time TP lowerer every operator dispatches through."""
        self._lowerer = lowerer

    def to_local(self, tensor: Any) -> Any:
        """Identity-transition semantics: a DTensor unwraps to its local shard."""
        from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415

        if isinstance(tensor, DTensor):
            return tensor.to_local()
        return tensor

    def all_gather(self, tensor: Any, dim: Optional[int] = None) -> Any:
        """Shard -> Replicate on the tp axis: concat-gather along ``dim``."""
        return self._execute("all_gather", tensor, dim)

    def all_reduce(self, tensor: Any) -> Any:
        """Partial -> Replicate on the tp axis: sum-reduce over the tp group."""
        return self._execute("all_reduce", tensor)

    def reduce_scatter(self, tensor: Any, dim: Optional[int] = None) -> Any:
        """Partial -> Shard on the tp axis: sum-reduce, then shard along ``dim``."""
        return self._execute("reduce_scatter", tensor, dim)

    def _execute(self, kind: str, tensor: Any, dim: Optional[int] = None) -> Any:
        """Unwrap, skip ``None``, and run one lowered collective op."""
        from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415

        if tensor is None:
            return None
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        return self._lowerer.execution_op(kind, tensor_dim=dim).execute(tensor)


#: Generated class names whose ``forward`` an inline strategy body replaced
#: (the MoE EP shell): it orchestrates through parallel state bound on the
#: instance at install time rather than through a compiled boundary.  Filled
#: per generated module from ``meta.external_state_classes`` — the injected
#: strategy's own resolved spec is the only declaration site.
_EXTERNAL_STATE_CLASSES_BY_MODULE: dict[str, frozenset[str]] = {}


def _mesh_group(mesh: Any, name: str) -> Any:
    """Return a named mesh group, tolerating one-dimensional mesh APIs."""
    if mesh is None:
        return None
    try:
        return mesh.get_group(name)
    except TypeError:
        return mesh.get_group()


def _install_inline_state_bindings(module: Any, expert_mesh: Any) -> None:
    """Bind the instance-attribute parallel channel an inlined forward reads.

    The inlined EP MoE shell orchestrates through the same instance attributes
    the boundary templates read: ``self._hyper_ep_group`` plus the
    ``self.ep_enable`` guard.  The value comes from the live mesh (expert mesh
    -> EP group) and reproduces the removed module-level state object's verdict
    exactly: ``ep_enable`` iff the resolved group spans more than one rank.

    No other axis is bound here.  A class whose forward is *not* replaced by an
    inline body keeps the emitted boundary form, and its channel comes from the
    shared install paths: the compiled plan plus the TP operators
    (``hyper_install_boundaries``) and the declared inner wrapper
    (``hyper_apply_inner_wrapper``).
    """
    ep_group = _mesh_group(expert_mesh, "ep") if expert_mesh is not None else None
    module._hyper_ep_group = ep_group
    module.ep_enable = ep_group is not None and ep_group.size() > 1


def hyper_bind_inline_state(
    model: Any,
    mesh_context: Any,
    mesh_dim_names: Optional[tuple[str, ...]] = None,
    *,
    generated_module: Any = None,
) -> None:
    """Bind the parallel channel on every inlined component module instance.

    A class declared in ``meta.external_state_classes`` has its ``forward``
    replaced by an inline strategy body, so the boundary / compute /
    inner-wrapper installers all skip it: its orchestration reads instance
    attributes instead.  This walks the live model once and binds them on every
    instance of a declared class — the same classes the installers skip, so an
    inlined forward can never run with a partially bound channel.

    The mesh facts are resolved lazily and only when a declared class is
    actually present, so an artifact without inlined components never touches
    the mesh.
    """
    if model is None or generated_module is None:
        return
    targets = [
        module
        for _, module in model.named_modules()
        if _is_external_state_inline_module(module, generated_module)
    ]
    if not targets:
        return
    _, _, expert_mesh, _ = _resolve_runtime_meshes(mesh_context, mesh_dim_names)
    for module in targets:
        _install_inline_state_bindings(module, expert_mesh)


def register_external_state_classes(generated_module: Any, class_names: Any) -> None:
    """Record which generated classes take their parallel state externally."""
    if generated_module is None:
        return
    _EXTERNAL_STATE_CLASSES_BY_MODULE[generated_module.__name__] = frozenset(
        class_names or ()
    )


def _is_external_state_inline_module(module: Any, generated_module: Any = None) -> bool:
    """Whether this generated class gets its state from instance attributes.

    True for an instance of a class the artifact declared in
    ``meta.external_state_classes``: codegen replaced that class's forward with
    an inline strategy body and it reads the attributes
    ``hyper_bind_inline_state`` binds, so the boundary installers must leave it
    alone.
    """
    if generated_module is None:
        return False
    module_name = getattr(generated_module, "__name__", None)
    class_names = _EXTERNAL_STATE_CLASSES_BY_MODULE.get(module_name)
    if not class_names:
        return False
    cls = type(module)
    return cls.__module__ == module_name and cls.__name__ in class_names


def _install_deferred_bias_pair(module: Any, spec: Any, *, lowered: bool) -> None:
    """Install the D-22 bias-suppression / deferred-re-add pair on one boundary.

    Reuses the native Phase C primitives (``_make_bias_free_forward`` /
    ``_maybe_add_deferred_biases`` from ``forward_rewriter``) so the hiding
    semantics cannot drift from the trainer's path; the exit hook is bound as
    ``module._hyper_deferred_bias``, which the emitted forward templates call
    after the output redistribution.

    The one codegen-specific adjustment vs the native
    ``_install_bias_suppression`` loop: on a source-lowered boundary
    (``lowered=True``) a deferred param path with no dot — the boundary's OWN
    bias — wraps the extracted ``_forward_impl`` (the compute forward the
    rewritten ``forward`` calls), not ``forward`` itself: wrapping
    ``forward`` would keep the bias hidden while the emitted
    ``self._hyper_deferred_bias(outputs)`` reads it at the exit.  Child-owned
    paths and imported-class boundaries keep the native placement exactly.
    """
    from hyper_parallel.distributed._builder.forward_rewriter import (  # pylint: disable=C0415
        _make_bias_free_forward,
        _maybe_add_deferred_biases,
    )

    for param_path in spec._deferred_bias_params:  # pylint: disable=protected-access
        owner_path = param_path.rpartition(".")[0]
        if owner_path:
            owner = module.get_submodule(owner_path)
            owner.forward = _make_bias_free_forward(owner, owner.forward)
        elif lowered:
            module._forward_impl = _make_bias_free_forward(
                module, module._forward_impl
            )
        else:
            module.forward = _make_bias_free_forward(module, module.forward)
    module._hyper_deferred_bias = functools.partial(
        _maybe_add_deferred_biases, module, spec
    )


def _wrap_installed_boundary_forward(
    module: Any, installed: InstalledBoundary, spec: Any = None
) -> None:
    """Install native-style boundary entry/exit redistribution on one module.

    The compiled plan is captured once at install time instead of being
    re-derived on every call.  When ``spec`` carries D-22 deferred bias
    params, the wrapper also re-adds them after the exit redistribution —
    the native Phase C exit order (suppress inside, reduce, add once).
    """
    if getattr(module, "_codegen_boundary_wrapped", False):
        return
    original_forward = module.forward
    deferred_bias = spec is not None and bool(spec._deferred_bias_params)

    @functools.wraps(original_forward)
    def codegen_boundary_forward(*args: Any, **kwargs: Any) -> Any:
        redist_args, redist_kwargs = installed.redistribute_inputs((args, kwargs))
        outputs = original_forward(*redist_args, **redist_kwargs)
        outputs = installed.redistribute_outputs(outputs)
        if deferred_bias:
            outputs = module._hyper_deferred_bias(outputs)
        return outputs

    module.forward = codegen_boundary_forward
    module._codegen_boundary_wrapped = True
    module._codegen_original_forward = original_forward


def _wrap_static_fallback_forward(module: Any, installed: InstalledBoundary) -> None:
    """Replace a rejected static ``tp_collective`` forward with the generic engine.

    The static template's body references ``self._hyper_tp``, which this
    install path refused to bind; replacing ``module.forward`` wholesale
    keeps the class runnable — the compiled plan carries the same
    transitions through the generic redistribute engine.  A D-22 exit hook
    already bound on the module (``_hyper_deferred_bias``, installed before
    this fallback wraps) is re-applied after the exit redistribution, so the
    fallback preserves the deferred-bias semantics the emitted template had.
    """
    impl = module._forward_impl
    deferred_bias = getattr(module, "_hyper_deferred_bias", None)

    def generic_boundary_forward(*args: Any, **kwargs: Any) -> Any:
        redist_args, redist_kwargs = installed.redistribute_inputs((args, kwargs))
        outputs = impl(*redist_args, **redist_kwargs)
        outputs = installed.redistribute_outputs(outputs)
        if deferred_bias is not None:
            outputs = deferred_bias(outputs)
        return outputs

    module.forward = generic_boundary_forward


def _install_static_tp_operators(
    module: Any,
    entry: dict[str, Any],
    installed: InstalledBoundary,
    mesh_dim_names: Optional[tuple[str, ...]],
    injection: Optional[dict[str, Any]],
) -> None:
    """Validate a statically lowered ``tp_collective`` forward against the live mesh.

    The emitter baked bare-operator calls under an optimistic verdict — the
    frozen entry classified as ``tp_collective`` under the plan's own axes
    and the source passed the structural gates (the class marker records
    that).  The live mesh may disagree: active axes the plan did not freeze,
    a tp rank order the lowerer rejects, a backend without the needed
    collective.  Re-derive the form from the install-time routing facts
    (``installed.active_dim_names``, the live mesh's mesh-major order) and
    bind ``module._hyper_tp`` only when the live verdict reproduces the
    emitted ops; on any mismatch replace the forward with the generic
    engine, so the baked calls never run unvalidated.
    """
    from hyper_parallel.codegen.emit.parallel import (  # pylint: disable=C0415
        TP_FORM_ATTRIBUTE,
        TP_FORM_MARKER,
    )
    from hyper_parallel.codegen.plan.boundary_forms import (  # pylint: disable=C0415
        FORM_TP_COLLECTIVE,
        classify_boundary_form,
    )
    from hyper_parallel.distributed._builder.tp_collective_lowering import (  # pylint: disable=C0415
        create_tp_collective_lowerer,
    )

    if getattr(module, TP_FORM_ATTRIBUTE, None) != TP_FORM_MARKER:
        return
    lowerer = create_tp_collective_lowerer(
        installed.dense_mesh, installed.active_dim_names
    )
    emitted = classify_boundary_form(entry, mesh_dim_names, injection)
    live = classify_boundary_form(entry, installed.active_dim_names, injection)
    if (
        lowerer is not None
        and emitted.form == FORM_TP_COLLECTIVE
        and live.form == FORM_TP_COLLECTIVE
        and live.in_ops == emitted.in_ops
        and live.out_ops == emitted.out_ops
    ):
        module._hyper_tp = TPOperators(lowerer)
        return
    logger.warning(
        "hyper_install_boundaries: static tp_collective forward on %s does not "
        "match the live mesh (emitted=%s live=%s lowerer=%s); falling back to "
        "the generic redistribute engine",
        type(module).__name__, emitted.form, live.form, lowerer is not None,
    )
    _wrap_static_fallback_forward(module, installed)


def _install_toggle_bindings(
    module: Any,
    installed: InstalledBoundary,
) -> None:
    """Install the switch/toggle enabler booleans and operators onto one module.

    A ``toggle``-marked forward (see ``emit.parallel.lower_forward_boundaries_toggle``)
    calls ``self._hyper_tp.<kind>(...)`` inside ``if self.{axis}_enable:``
    segments.  This binds that runtime surface from the live mesh facts the
    compiled ``InstalledBoundary`` already carries — one codegen path, no second
    execution engine:

    - ``module.tp_enable`` / ``module.cp_enable`` / ``module.ep_enable`` are
      set from ``installed.active_dim_names`` (the boundary's actual parallel
      dimensions on the live mesh), so a segment only runs when its axis is
      genuinely active.
    - ``module._hyper_tp`` reuses the SAME ``TPOperators`` + native lowerer that
      the ``tp_collective`` static form binds, so the toggle body dispatches
      through the production collectives rather than a duplicated path within
      the raw operator.
    """
    from hyper_parallel.codegen.emit.parallel import (  # pylint: disable=C0415
        TOGGLE_FORM_MARKER,
        TP_FORM_ATTRIBUTE,
    )
    from hyper_parallel.distributed._builder.tp_collective_lowering import (  # pylint: disable=C0415
        create_tp_collective_lowerer,
    )

    if getattr(module, TP_FORM_ATTRIBUTE, None) != TOGGLE_FORM_MARKER:
        return
    active = tuple(installed.active_dim_names or ())
    module.tp_enable = bool(active and "tp" in active)
    module.cp_enable = bool(active and "cp" in active)
    module.ep_enable = bool(active and "ep" in active)
    if (
        (module.tp_enable or module.cp_enable)
        and getattr(installed, "dense_mesh", None) is not None
    ):
        lowerer = create_tp_collective_lowerer(installed.dense_mesh, active)
        if lowerer is not None:
            module._hyper_tp = TPOperators(lowerer)


def hyper_install_boundaries(
    model: Any,
    param_plan: dict[str, Any],
    mesh_context: Any,
    mesh_dim_names: Optional[tuple[str, ...]] = None,
    *,
    injections: Optional[list[dict[str, Any]]] = None,
    generated_module: Any = None,
) -> None:
    """Compile every frozen boundary once and bind it to its module instance.

    Owns the per-boundary install the removed per-forward wrappers used to do:
    each boundary's mesh selection (dense vs
    expert, the empty-active-axes no-op), rank-order/backend lowering checks,
    and input-index binding run once here, and the compiled
    ``InstalledBoundary`` is stored as ``module._hyper_boundary`` — the
    generated forwards reference that attribute instead of module globals.

    Source-lowered boundaries (classes rewritten in the generated file) only
    get the attribute binding; imported-class boundaries (``nn.Embedding``,
    ``nn.Linear``) additionally get the entry/exit forward wrapper installed,
    after the vocab-parallel-embedding wrap, mirroring the previous runtime
    behavior. A statically lowered ``tp_collective`` forward additionally gets
    its bare operators bound (``module._hyper_tp``) only after the live-mesh
    re-validation in :func:`_install_static_tp_operators`; a mismatch replaces
    the forward with the generic engine.

    A boundary whose frozen entry carries ``deferred_bias_params`` (D-22)
    additionally gets the native suppression/restore pair installed
    (:func:`_install_deferred_bias_pair`): child ``Linear`` forwards run
    bias-free inside the region and the exit hook
    ``module._hyper_deferred_bias`` re-adds each bias exactly once after the
    output redistribution — the same order the emitted templates encode. An
    external-state inline boundary cannot run that exit hook and fails fast
    instead of silently dropping the bias.

    Boundaries are installed in post-order (deepest FQN first) via the native
    ``_boundary_post_order_key`` — the same D-14 invariant 2 ordering as the
    applier's Phase C driving loop, so inner wrappers exist before an outer
    boundary's local_compute_fn can cache them.
    """
    if not param_plan:
        return
    from hyper_parallel.distributed._builder.applier import (
        _boundary_post_order_key,
        _get_tp_submesh,
    )
    from hyper_parallel.distributed._builder.forward_rewriter import (
        _is_vocab_parallel_embed,
    )
    from hyper_parallel.distributed._builder.parameter_sharding import (
        _resolve_module,
    )

    meshes_resolved = False
    tp_mesh = None

    for fqn, entry in sorted(param_plan.items(), key=_boundary_post_order_key):
        if _is_tp_sequence_parallel_embed_entry(entry):
            parent_fqn = fqn.rpartition(".")[0]
            _wrap_default_position_ids(_resolve_module(model, parent_fqn))
        if not is_boundary_entry(entry):
            continue
        module = _resolve_module(model, fqn)
        if _has_lowered_forward(module, generated_module):
            injection = _injection_spec(injections or [], fqn)
            installed = InstalledBoundary(
                entry, mesh_context, mesh_dim_names, module=module
            )
            if _is_external_state_inline_module(module, generated_module):
                # D-22: an inline-strategy class body never passes through
                # this boundary exit, so a deferred bias would be suppressed
                # but never re-added — fail fast instead of silently
                # dropping the bias from every forward.
                if entry.get("deferred_bias_params"):
                    raise NotImplementedError(
                        f"hyper_install_boundaries: boundary {fqn!r} defers bias "
                        f"params {entry['deferred_bias_params']}, but its class is "
                        "an external-state inline strategy the D-22 exit hook "
                        "cannot run in"
                    )
                continue
            module._hyper_boundary = installed
            if entry.get("deferred_bias_params"):
                # Before _install_static_tp_operators: the static-fallback
                # wrap captures module._forward_impl at wrap time, so the
                # suppression must already be in place there.
                _install_deferred_bias_pair(
                    module, _FrozenBoundarySpec(entry, injection), lowered=True
                )
            _install_static_tp_operators(
                module,
                entry,
                installed,
                mesh_dim_names,
                injection,
            )
            _install_toggle_bindings(module, installed)
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
            _install_vocab_parallel_embedding(module, tp_mesh)
        installed = InstalledBoundary(
            entry, mesh_context, mesh_dim_names, module=module
        )
        # Every boundary module carries its compiled plan — the imported-class
        # wrapper below closes over it, so introspection and debugging see the
        # same object the generated source-lowered forwards would.
        module._hyper_boundary = installed
        if spec._deferred_bias_params:
            # Native order: suppression first, so the wrapper below captures
            # the suppressed forward as its original (D-22, Phase C identity).
            _install_deferred_bias_pair(module, spec, lowered=False)
        _wrap_installed_boundary_forward(module, installed, spec=spec)


# ---------------------------------------------------------------------------
# Compute binding and inner-wrapper installation
# ---------------------------------------------------------------------------

def _injection_spec(injections: list[dict[str, Any]], fqn: str) -> Optional[dict[str, Any]]:
    """Find one frozen boundary's injection declaration by its FQN."""
    for rule in injections:
        if rule.get("match") == fqn:
            return rule
    return None


def hyper_expand_injections(
    rules: Optional[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Expand grouped injection rules into the per-FQN form helpers consume.

    The generated ``_HYPER_INJECTIONS`` literal groups rules that differ only
    in their ``match`` under one record with a ``match`` **list** (one rule
    for every layer's CP wrapper instead of one per layer).  This expansion —
    called once at the top of the generated ``hyper_parallelize`` — restores
    the per-FQN shape, so ``hyper_install_boundaries`` /
    ``hyper_bind_compute`` / ``hyper_apply_inner_wrapper`` keep their frozen
    single-FQN ``match`` contract unchanged.

    A scalar ``match`` (the frozen ``meta.injections`` shape and every
    pre-grouping artifact) passes through untouched; ``None`` / empty becomes
    ``[]``.
    """
    if not rules:
        return []
    expanded: list[dict[str, Any]] = []
    for rule in rules:
        match = rule.get("match") if isinstance(rule, dict) else None
        if isinstance(match, (list, tuple)):
            for fqn in match:
                one = dict(rule)
                one["match"] = fqn
                expanded.append(one)
        else:
            expanded.append(rule)
    return expanded


def hyper_bind_compute(
    model,
    injections: list[dict[str, Any]],
    mesh_context,
    mesh_dim_names: Optional[tuple[str, ...]] = None,
    *,
    param_plan: Optional[dict[str, Any]] = None,
    generated_module: Any = None,
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
        # External-state inline modules (e.g. Qwen3MoeSparseMoeBlock) carry an
        # inlined forward and must not receive __hyper_compute__, but the local
        # compute factory above still has to run: it installs companion state
        # such as ``experts.local_expert_count`` and the bound experts.forward
        # that the inlined forward relies on.
        if _is_external_state_inline_module(module, generated_module):
            continue
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
    generated_module: Any = None,
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
    re-wrap) via ``_install_inner_adapter``.  This is the one CP path for a
    generated artifact: the wrapper is applied on the module whose forward is
    the emitted boundary form (the rendered fused attention), so the companion
    ``attention_interface`` swap and the boundary redistribution compose
    instead of being re-implemented per component.
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
        if _is_external_state_inline_module(module, generated_module):
            continue
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


def _has_lowered_forward(module: Any, generated_module: Any = None) -> bool:
    """Return whether the generated source already owns this boundary wrapper."""
    if _is_external_state_inline_module(module, generated_module):
        return True
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
    if getattr(module, "_codegen_position_ids_wrapped", False):
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
    module._codegen_position_ids_wrapped = True
    module._codegen_original_position_ids_forward = original_forward


def _install_vocab_parallel_embedding(module: Any, tp_mesh: Any) -> None:
    """Install native's D-02 masked-embedding wrapper, once per module.

    The wrapper itself is native
    (``forward_rewriter._wrap_vocab_parallel_embedding``): codegen installs the
    boundary before its one-shot parameter unwrap, and native derives the
    vocab interval from the local shard, so both paths share one
    implementation. Only the re-install guard is codegen-local —
    ``hyper_install_boundaries`` is documented idempotent, while native Phase C
    runs exactly once.
    """
    if getattr(module, "_codegen_vocab_parallel_wrapped", False):
        return
    from hyper_parallel.distributed._builder.forward_rewriter import (  # pylint: disable=C0415
        _wrap_vocab_parallel_embedding,
    )

    _wrap_vocab_parallel_embedding(module, tp_mesh)
    module._codegen_vocab_parallel_wrapped = True


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
    module = import_generated_module(artifact_dir)
    # The artifact's own record of which generated classes take their parallel
    # state externally — declared once by the adapter render spec and carried
    # here in ``meta``.
    register_external_state_classes(module, meta.external_state_classes)
    entry_name = meta.entrypoints.get("parallelize")
    entry = getattr(module, entry_name, None) if entry_name else None
    if entry is not None:
        return entry(model, mesh_context)
    return _parallelize_inline_from_meta(model, mesh_context, meta, module)


def _parallelize_inline_from_meta(model: Any, mesh_context: Any, meta: CodegenMeta, generated_module: Any) -> dict:
    """Parallelize a clean inline artifact whose model file carries no plan globals.

    Execution is convergent with the native applier (see
    :func:`_parallelize_via_native_apply`): Phase A parameter sharding /
    B special handlers / D tied weights run against a live ``ShardingPlan``
    rebuilt from the frozen meta; Phase C forward assembly stays codegen-owned
    because the generated file pre-rewrites forward.
    """
    return _parallelize_via_native_apply(model, mesh_context, meta, generated_module)


def _rebuild_live_plan_from_meta(meta: CodegenMeta):
    """Rebuild a live ``ShardingPlan`` from the frozen meta (meta -> plan).

    Proof-of-concept reconstruction: re-parses the frozen literal forms back
    into real ``Placement``/``Target`` objects so native builder helpers
    (``_shard_planned_parameters`` / ``_apply_plan_special_handlers`` /
    ``_replicate_tied_weights``) can consume this plan exactly as they consume
    a planner-produced one.

    Which fields cross the meta boundary is declared in
    :mod:`hyper_parallel.codegen.plan.spec_fields` — freeze and this rebuild
    walk the same table, so the two sides cannot drift apart.  ``init=False``
    fields (``_deferred_bias_params``, D-22) are rebuilt via ``setattr``
    after construction (``POST_INIT_SPEC_FIELDS``).  Exempt fields (never
    frozen):
    - ``_is_terminal`` — validate-mode only, not used in production;
    - ``_tp_local_attr_plan`` — planner-derived; the rebuilt spec leaves it
      None and ``maybe_update_head_counts`` takes its backward-compatible
      path (auto attrs via the head-sharded heuristic, user attrs via the
      rebuilt ``tp_divide_attrs``);
    - ``_needs_cp_attn`` / ``_resolved_inner_*`` — apply-time preflight and
      introspection state, not consumed on the rebuild path.

    Injection-side fields (``local_compute_fn`` / ``inner_wrapper`` /
    ``inner_out_src``) are reconstructed with the same :func:`_injection_target`
    the existing ``_FrozenBoundarySpec`` uses, so injection fidelity matches
    the status-quo codegen path.
    """
    from hyper_parallel.distributed.plan import ShardingPlan
    from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
    from hyper_parallel.codegen.plan.spec_fields import (
        INJECTION_SPEC_FIELDS,
        PARAM_PLAN_SPEC_FIELDS,
        POST_INIT_SPEC_FIELDS,
    )

    by_fqn: dict[str, Any] = {}
    for rule in meta.injections or []:
        if isinstance(rule, dict) and rule.get("match"):
            by_fqn[rule["match"]] = rule

    modules: dict[str, Any] = {}
    for fqn, entry in (meta.param_plan or {}).items():
        inj = by_fqn.get(fqn, {})
        kwargs: dict[str, Any] = {
            field.name: _rebuild_spec_field(field, entry)
            for field in PARAM_PLAN_SPEC_FIELDS
            if not field.post_init
        }
        for field in INJECTION_SPEC_FIELDS:
            kwargs[field.name] = _rebuild_spec_field(field, inj)
        spec = ModuleShardingSpec(**kwargs)
        for field in POST_INIT_SPEC_FIELDS:
            value = _rebuild_spec_field(field, entry)
            if value is not None:
                setattr(spec, field.name, value)
        modules[fqn] = spec

    return ShardingPlan(
        modules=modules,
        special_handlers=dict(meta.special_handlers or {}),
        mesh_dim_names=tuple(meta.mesh_dim_names or ()),
        tied_pairs=[tuple(pair) for pair in (meta.tied_pairs or [])],
    )


def _rebuild_spec_field(field: Any, source: dict[str, Any]) -> Any:
    """Deserialize one spec field per its :class:`SpecField` kind contract.

    The inverse of freeze's ``_freeze_spec_field``: reads the field's meta
    key from the frozen entry (param-plan fields) or injection rule
    (injection fields) and restores the live value.  Absent keys restore the
    dataclass defaults — ``None`` for optional declarations, ``True`` for
    ``is_boundary`` (pre-flag metas), ``{}``/``0`` for the EP internals —
    matching what ``_rebuild_live_plan_from_meta`` passed positionally
    before the table drove both sides.
    """
    kind, key = field.rebuild, field.key
    if kind == "placement_map":
        # Key absent/None -> None (not declared); an explicit {} ("shards
        # nothing") must round-trip as {} — Phase A iterates params.items()
        # and a None there is the crash this rebuild must not reintroduce.
        if source.get(key) is None:
            return None
        return parse_named_placement(source[key])
    if kind == "named_placements":
        return parse_named_placement(source[key]) if source.get(key) else None
    if kind in ("name_list", "attr_list"):
        return list(source[key]) if source.get(key) is not None else None
    if kind == "bool_default_true":
        return bool(source[key]) if source.get(key) is not None else True
    if kind == "injection_target":
        return _injection_target(source[key]) if source.get(key) is not None else None
    if kind == "dict_copy":
        return dict(source[key]) if source.get(key) else {}
    if kind == "int_truthy":
        return source.get(key) or 0
    if kind == "tuple_empty":
        # D-22 param paths: absent/empty restores the dataclass default ()
        # — consumers iterate the tuple, so None would crash.
        return tuple(source[key]) if source.get(key) else ()
    # opt_scalar
    return source.get(key)


def _parallelize_via_native_apply(
    model: Any,
    mesh_context: Any,
    meta: CodegenMeta,
    generated_module: Any,
) -> dict:
    """Reuse the native execution body for a codegen artifact.

    Rebuilds a live ``ShardingPlan`` from the frozen meta
    (:func:`_rebuild_live_plan_from_meta`) and runs native Phase A
    (``_shard_planned_parameters``) / B (``_apply_plan_special_handlers``) /
    D (``_replicate_tied_weights``) against it.  This is the converging
    execution path: codegen no longer maintains a second literal
    parameter-sharding replay.

    Phase C forward assembly is deliberately NOT reused: the generated file
    has already rewritten every boundary's ``forward`` into the
    redistribute-in / ``_forward_impl`` / redistribute-out form (resolved at
    runtime against the ``_hyper_boundary`` / ``__hyper_compute__`` bound
    here). Native ``_apply_phase_c`` captures the *current* forward as its
    ``original_forward`` and wraps it in ANOTHER
    ``redistribute_inputs -> original -> redistribute_outputs`` — running it
    on a codegen artifact double-wraps the redistribution (double TP/EP
    communication) because no native branch skips the wrap.  So full
    ``apply_sharding_plan`` reuse is not viable; A/B/D reuse with codegen-owned
    Phase C is the minimum viable convergence.

    MoE/EP artifacts are supported: native Phase A applies the same
    ``_stack_moe_experts`` step here as it does for a planner-produced plan, so
    dense and MoE topologies share this one reuse path.
    """
    from hyper_parallel.distributed.mesh import MeshContext  # pylint: disable=C0415
    from hyper_parallel.distributed._builder.applier import (  # pylint: disable=C0415
        _apply_plan_special_handlers,
        _get_active_mesh,
        _get_tp_submesh,
    )
    from hyper_parallel.distributed._builder.parameter_sharding import (  # pylint: disable=C0415
        _replicate_tied_weights,
        _resolve_parameter_source_meshes,
        _shard_planned_parameters,
        detect_tied_weights,
    )

    plan = _rebuild_live_plan_from_meta(meta)
    mesh_dim_names = plan.mesh_dim_names
    param_plan = meta.param_plan or {}
    injections = hyper_expand_injections(meta.injections or [])

    mc = mesh_context if isinstance(mesh_context, MeshContext) else None
    device_mesh = mc.device_mesh if mc is not None else mesh_context
    if device_mesh is None:
        raise ValueError(
            "native A/B/D reuse requires a DeviceMesh / MeshContext"
        )
    mesh = _get_active_mesh(device_mesh, mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
    expert_mesh, dense_source_mesh, expert_source_mesh = (
        _resolve_parameter_source_meshes(plan, mesh_context, device_mesh, tp_mesh)
    )

    # Native A / B / D are executed against the reconstructed live plan.
    _shard_planned_parameters([model], plan, mesh, expert_mesh, validate_mode=False)
    _apply_plan_special_handlers([model], plan, mesh)
    _replicate_tied_weights(model, list(plan.tied_pairs) or detect_tied_weights(model))

    # Native C is not reusable (see docstring) — keep the codegen forward
    # assembly that matches the pre-rewritten generated file.
    hyper_install_boundaries(
        model,
        param_plan,
        mesh_context,
        mesh_dim_names,
        injections=injections,
        generated_module=generated_module,
    )
    hyper_bind_inline_state(
        model,
        mesh_context,
        mesh_dim_names,
        generated_module=generated_module,
    )
    hyper_bind_compute(
        model,
        injections,
        mesh_context,
        mesh_dim_names,
        param_plan=param_plan,
        generated_module=generated_module,
    )
    hyper_apply_inner_wrapper(
        model,
        injections,
        mesh_context,
        mesh_dim_names,
        param_plan=param_plan,
        generated_module=generated_module,
    )
    return hyper_build_tp_grad_info(
        model, plan, dense_source_mesh, expert_source_mesh
    )


__all__ = [
    "hyper_apply_inner_wrapper",
    "hyper_bind_compute",
    "hyper_bind_inline_state",
    "hyper_build_tp_grad_info",
    "hyper_install_boundaries",
    "hyper_to_local_if_dtensor",
    "import_generated_module",
    "load_codegen_meta",
    "parallelize_from_generated",
    "verify_codegen_signature",
]
