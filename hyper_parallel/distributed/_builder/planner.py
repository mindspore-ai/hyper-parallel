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
"""planner: ShardingPlanner 6-phase derivation pipeline (05 §3.6 canonical).

Phase 1  parameter role classification (ParameterClassifier + the family's
         ModelAdapterSpec.sharding_rules from models/registry.py)
Phase 2  communication boundary grouping (two passes: first group by owning
         module, then merge upward depth-first — fixes the flaw in the
         05 §3.6.6 pseudocode where "single-parameter group inference"
         misjudges a q_proj leaf module as an mlp boundary)
Phase 3  semantic role inference (explicit FQN patterns > structural guards
         > parameter role combinations)
Phase 4  template lookup to generate spec (_build_spec_from_template)
Phase 4.5 user plan_overrides merge (_merge_plan_overrides, 05 §3.6.7)
Phase 5  _is_terminal marking only (D-14, 05 §13: compile-time chain
         propagation/validation removed — specs are fully self-declared and
         each module vouches for its own propagation in validate mode)
Phase 6  special parameter handler collection (SPECIAL_HANDLERS)

Registries:
- family sharding rules: ``ModelAdapterSpec.sharding_rules`` providers in
  ``models/<family>/adapter/registration.py``, resolved lazily through
  ``models/registry.py::get_model_adapter`` (any spelling — model_type,
  HF architecture or canonical arch name);
- ``SPECIAL_HANDLERS``: {handler_name: callable(module, param_name, mesh)} —
  lives in ``_builder/special_handlers.py``.
"""

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from torch import nn

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import (
    DP,
    TP,
    ModuleShardingSpec,
)
from hyper_parallel.distributed.tensor_parallel.param_role import (
    ParameterClassifier,
    ParamRole,
    _match_any,
)
from hyper_parallel.distributed._builder.default_templates import (
    ShardingTemplate,
    TEMPLATES,
    _build_spec_from_template,
    _finalize_tp_local_attr_plans,
    _moe_expert_tp_placement,
    _multi_dim,
)
from hyper_parallel.distributed._builder.function_module import FunctionModule
from hyper_parallel.distributed._builder.rule_resolver import (
    _last_segment,
    _merge_plan_overrides,
    _normalize_contract_fields,
)
from hyper_parallel.distributed._builder.special_handlers import (
    _SPECIAL_HANDLER_PATTERNS,
    _collect_special_handlers,
)

logger = logging.getLogger(__name__)

# Per-family naming-rule overrides no longer live here: each family declares
# them via ``ModelAdapterSpec.sharding_rules`` in
# ``models/<family>/adapter/registration.py`` (DeepSeek MLA, Qwen2-MoE
# shared_expert_gate), discovered through ``models/registry.py``. The
# planner core stays family-agnostic (05 §15.9 step 1).


# Leaf-segment guard for projection/container segment names: these segment
# names are not boundary containers themselves; inference returns unknown
# and continues upward.
# NOTE (accuracy fix F3): "shared_experts" was REMOVED from this guard —
# the shared expert (singular or plural spelling) must surface as its own
# nested "mlp" boundary (its boundary exit owns the RowWise Partial TP
# reduction, accuracy_problem.md 10.3 Option A); keeping it in the guard would
# merge it into the parent MoE boundary and silently leave the TP reduction
# without an owner.
_LEAF_SEGMENT_GUARD = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "qkv_proj", "fused_qkv", "linear_qkv", "gate_up_proj", "query_key_value",
    "experts", "gate", "linear", "proj",
    "fc1", "fc2", "w1", "w2", "w3", "w13", "dense", "dense_h_to_4h", "dense_4h_to_h",
})

_ATTN_PATTERNS = ("attn", "attention")
_MLP_PATTERNS = ("mlp", "ffn", "feed_forward")
_MOE_CONTAINER_PATTERNS = ("mlp", "moe", "moe_block", "moe_layer")


class ShardingPlanner:
    """Automatically derive a ShardingPlan from any HF-style model (05 §3.6.6).

    ``plan_overrides``: {module_fqn | fqn_glob: ModuleShardingSpec} — the
    SINGLE injection/override interface (05 §3.6.7 + the unification rework):

    - **merge mode** (key hits a planner-derived boundary): empty contract
      fields (``params``/``in_src``/``in_dst``/``out_src``/``out_dst``/
      ``out_names``) INHERIT the derived values, non-empty ones replace
      (field granularity); the string sentinels ``"auto"`` (explicit
      inherit — self-documenting "derive per template") and ``"none"``
      (explicit clear) are resolved at merge time and never reach the plan
      output; the injection fields (``local_compute_fn`` /
      ``inner_target`` / ``inner_wrapper`` non-None, ``region_dispatch=False``)
      always win; internal flags (``_ep_size`` / ``_ep_stack`` /
      ``_needs_cp_attn``) always inherit. This is how
      CP/EP compute injection is declared — no need to re-declare contracts;
    - **insert mode** (exact key misses every derived boundary): the spec is
      inserted as-is and must be fully self-declared — an override with
      empty params AND empty contracts fails fast ("no template matched");
    - **glob keys** (containing ``*``/``?``/``[``): merge-applied to every
      matching boundary; a pattern hitting nothing warns loudly. Glob keys
      never insert new boundaries.

    The YAML transport (``hyper_parallel.trainer.config.PlanOverride``) is
    converted to this dict by ``entries_to_plan_overrides()`` on the trainer
    side BEFORE the
    planner is constructed — the planner itself has exactly one override
    interface.

    ``derive=False`` turns the planner into a pure declaration assembler:
    template derivation (Phases 2-4) is skipped and the plan contains ONLY
    the ``plan_overrides`` specs — every key is insert mode and must be
    fully self-declared. Use it for subtrees where automatic derivation is
    semantically wrong, e.g. the multimodal encoder_dp ViT bridge (each
    rank encodes a different image shard, so any derived TP collective
    inside the ViT would silently mix samples). This replaces the error-
    prone post-hoc pruning idiom ``plan.modules = {"": plan.modules[""]}``.
    """

    def __init__(
        self,
        plan_overrides: Optional[Dict[str, ModuleShardingSpec]] = None,
        *,
        derive: bool = True,
        allow_uncovered_params: bool = False,
    ) -> None:
        """Create a ShardingPlanner.

        Args:
            plan_overrides: ``{module_fqn | fqn_glob: ModuleShardingSpec}`` —
                the single injection/override interface (05 §3.6.7); see the
                class docstring for merge/insert/glob semantics.
            derive: When ``False``, template derivation (Phases 2-4) is
                skipped and the plan contains ONLY the ``plan_overrides``
                specs (every key is insert mode, fully self-declared).
            allow_uncovered_params: When ``True``, downgrade the F4b
                "every trainable parameter must be covered by the plan"
                hard error to a warning (exploratory debugging only).
        """
        self._classifier = ParameterClassifier()
        self._templates = TEMPLATES
        self._special_handler_patterns = dict(_SPECIAL_HANDLER_PATTERNS)
        self._plan_overrides = dict(plan_overrides or {})
        # derive=False: skip template derivation entirely — the plan contains
        # ONLY the plan_overrides specs (every key is insert mode and must be
        # fully self-declared). This replaces post-hoc pruning of
        # ``plan.modules`` (e.g. the multimodal encoder_dp ViT bridge, where
        # each rank encodes different images and ANY derived TP collective
        # inside the ViT subtree would be a math error).
        self._derive = derive
        # F4b (accuracy_fix_plan.md §2): by default every trainable parameter
        # must be covered by the plan (fail-fast). Set True ONLY for
        # exploratory debugging to downgrade the hard error to a warning.
        self._allow_uncovered_params = allow_uncovered_params

    # ── Main entry point ────────────────────────────────────────────────

    def _derive_boundary_specs(
        self,
        plan: ShardingPlan,
        boundary_groups,
        *,
        sequence_parallel: bool,
        loss_parallel: bool,
        mesh_dim_names: Tuple[str, ...],
        arch: str,
        ep_extend: int,
        mesh: DeviceMesh,
        model: Any,
        param_ndims: Dict[str, int],
    ) -> None:
        """Materialize template-derived boundary specs into a plan."""
        if not self._derive:
            logger.info(
                "derive=False: template derivation skipped — the plan will contain only plan_overrides "
                "specs (all insert mode, fully self-declared)"
            )
            return
        for boundary_fqn, group in boundary_groups.items():
            boundary_type = self._infer_boundary_type(boundary_fqn, group)
            template = self._templates.get(boundary_type)
            if template is None:
                logger.warning("No template for boundary_type=%s at %s", boundary_type, boundary_fqn)
                continue
            spec = _build_spec_from_template(
                self._templates,
                boundary_fqn,
                group,
                template,
                sequence_parallel,
                loss_parallel,
                mesh_dim_names,
                param_ndims=param_ndims,
            )
            if spec is None:
                continue
            if boundary_type == "moe_mlp":
                self._mark_hf_native_moe(
                    spec, group, boundary_fqn, template, mesh_dim_names, arch,
                    ep_extend=ep_extend, mesh=mesh, model=model, param_ndims=param_ndims,
                )
            plan.modules[boundary_fqn] = spec

    def _finalize_boundary_specs(
        self,
        plan: ShardingPlan,
        model: Any,
        *,
        tp_size: int,
        mesh_dim_names: Tuple[str, ...],
    ) -> None:
        """Normalize overrides and finish placement-dependent boundary metadata."""
        _merge_plan_overrides(self._plan_overrides, plan, model, derive=self._derive)
        _normalize_contract_fields(plan)
        _finalize_tp_local_attr_plans(plan, model, tp_size=tp_size, mesh_dim_names=mesh_dim_names)
        self._finalize_deferred_biases(plan, model, mesh_dim_names)
        self._finalize_fused_expert_tp_guard(plan, tp_size=tp_size)

    @staticmethod
    def _log_explanation(plan: ShardingPlan, enabled: bool) -> None:
        """Log the plan explanation when requested."""
        if enabled:
            logger.info("ShardingPlan explain:\n%s", plan.explain())

    def _classify_boundary_groups(self, model: Any, arch: str):
        """Classify model parameters and group them into candidate boundaries."""
        param_roles = self._classify_all_params(model, arch)
        return param_roles, self._group_by_boundary(param_roles)

    def plan(
        self,
        model: Any,
        mesh: DeviceMesh,
        *,
        tp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        sequence_parallel: bool = True,
        loss_parallel: bool = False,
        explain: bool = False,
    ) -> ShardingPlan:
        """Derive a ShardingPlan from *model* and *mesh* (6-phase pipeline).

        Args:
            model: Any HuggingFace ``PreTrainedModel``.
            mesh: A :class:`~hyper_parallel.core.dtensor.device_mesh.DeviceMesh`
                whose ``mesh_dim_names`` declare the physical topology axes (e.g.
                ``("dp", "tp")`` or ``("dp_replicate", "tp", "cp")``).  The
                ``tp`` / ``cp`` axes are managed by DTensor; ``dp*`` / ``pp``
                axes belong to FSDP2 / pipeline-parallel runtimes and are
                filtered out.
            tp_size: Tensor-parallel group size.  Must be **equal** to
                ``mesh[\"tp\"].size()`` when ``mesh_dim_names`` contains ``"tp"``
                and tp_size > 1; a mismatch raises :class:`ValueError`
                immediately (fail-first).
            cp_size: Context-parallel group size.  Same fail-first contract as
                *tp_size* with respect to ``mesh[\"cp\"]``.
            ep_size: Expert-parallel group size.

                - **D-10 TP-extend-EP** (the default MoE path, 05 §6.4.8): the
                  mesh does **not** have an ``"ep"`` axis; *ep_size* is the
                  extended EP group size, and the expert mesh ``(edp, ep)`` is
                  derived by flattening the full dense region
                  (``dp × tp × cp``).  *ep_size* is **not** validated against
                  the mesh in this case.
                - **Old-style EP** (when ``mesh_dim_names`` contains ``"ep"``):
                  *ep_size* must equal ``mesh[\"ep\"].size()``, validated
                  fail-first.
            sequence_parallel: When ``True`` (default), sequence dimension is
                sharded across TP (Shard(1) on the ``tp`` axis of activations).
            loss_parallel: When ``True``, lm_head output stays Shard(-1);
                otherwise gathered to Replicate (cross-entropy compatibility).
            explain: When ``True``, log the plan introspection report
                (:meth:`ShardingPlan.explain`) at INFO level after planning —
                per-boundary param sharding, compiled communication plans,
                and injection resolutions. The same report is available
                standalone via ``plan.explain(fqn=None)``.

        Returns:
            :class:`ShardingPlan`: module FQN → :class:`ModuleShardingSpec`.

        Raises:
            ValueError: If *tp_size* / *cp_size* / *ep_size* do not match the
                corresponding mesh dimensions (fail-first), if model-level
                constraints are violated (head divisibility, expert count, etc.),
                or if any ``plan_overrides`` spec declares a DP placement
                (05 §3.1.1 coordinate-system convention: the plan is always a
                single dp slice).
        """
        arch = self._get_architecture(model)
        self._check_overrides_no_dp()   # fail-first: the plan's coordinate system = a single dp slice
        mesh_dim_names = self._build_mesh_dim_names(mesh, tp_size, cp_size, ep_size)
        # D-10 TP-extend-EP (05 §6.4.8): ep_size is the extended EP group
        # size (the a2a communication domain, extended from the TP group to
        # neighboring dp/cp ranks; expert weights are sharded only along
        # the expert dim; no separate etp configuration). Validation
        # happens when _mark_hf_native_moe actually matches an HF-native
        # MoE (pre-stacked EP-aware modules use their own dispatcher and
        # are not subject to this constraint)
        ep_extend = ep_size if ep_size > 1 else 0

        # Phase 1: parameter role classification
        param_roles, boundary_groups = self._classify_boundary_groups(model, arch)

        # Phase 3+4: semantic inference + template-fills I/O
        param_ndims = {name: p.ndim for name, p in model.named_parameters()}
        # F4a/F4b: full shapes + requires_grad for the plan-time lints
        # (meta tensors carry shapes too — the zero-memory path is unaffected)
        param_shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
        plan = ShardingPlan(
            mesh_dim_names=mesh_dim_names,
            sequence_parallel=sequence_parallel,
            loss_parallel=loss_parallel,
        )
        self._derive_boundary_specs(
            plan,
            boundary_groups,
            sequence_parallel=sequence_parallel,
            loss_parallel=loss_parallel,
            mesh_dim_names=mesh_dim_names,
            arch=arch,
            ep_extend=ep_extend,
            mesh=mesh,
            model=model,
            param_ndims=param_ndims,
        )

        # Phase 4.5: unified override pass — merge mode (unset fields inherit
        # the derived spec) / insert mode (fully self-declared only) / glob.
        self._finalize_boundary_specs(
            plan, model, tp_size=tp_size, mesh_dim_names=mesh_dim_names
        )

        # D-14 invariants (05 §13.2/§13.3): full self-declaration + param
        # uniqueness (the only nesting check that remains)
        self._check_full_declaration(plan)
        self._check_param_uniqueness(plan)

        # Phase 5: _is_terminal marking (D-14: chain propagation removed)
        plan = self._mark_terminal(plan, model)

        # Phase 6: special parameter handling
        plan.special_handlers = _collect_special_handlers(param_roles, self._special_handler_patterns)

        # F4 plan-time lints (accuracy_fix_plan.md §2 — after Phase 4.5
        # overrides merge and Phase 6, so hand-written specs and special
        # handlers are all accounted for):
        # F4a: every Shard(dim) must divide the parameter shape — an empty
        #      shard at apply time becomes a plan-time teaching error;
        # F4b: every trainable parameter must be covered by the plan.
        self._check_shard_divisibility(
            plan, param_shapes, tp_size=tp_size, cp_size=cp_size,
            ep_size=ep_size)
        self._check_all_trainable_params_covered(plan, model)

        # DX guard: FunctionModule instances not covered by any spec run
        # without any boundary communication — warn instead of silently passing
        self._warn_uncovered_function_modules(model, plan)

        # tied-weight detection (embed <-> lm_head sharing storage)
        plan.tied_pairs = self._detect_tied_pairs(model)

        self._log_explanation(plan, explain)

        return plan

    # ── Architecture detection ──────────────────────────────────────────

    @staticmethod
    def _get_architecture(model) -> str:
        """Detect the canonical architecture name:
        config.architectures[0] > config.model_type > class name;
        lowercased with ForCausalLM-style suffixes stripped."""
        cfg = getattr(model, "config", None)
        arch_str = None
        archs = getattr(cfg, "architectures", None)
        if archs:
            arch_str = archs[0]
        if not arch_str:
            arch_str = getattr(cfg, "model_type", None)
        if not arch_str:
            arch_str = type(model).__name__

        s = arch_str.lower()
        for suffix in ("forcausallm", "forconditionalgeneration",
                       "forsequenceclassification", "forimagetexttotext"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
        return s

    @staticmethod
    def _validate_dtensor_axes(
        mesh, tp_size: int, cp_size: int, ep_size: int,
    ) -> None:
        """Fail-first: validate passed tp/cp/ep sizes against the mesh dimensions.

        The mesh *must* declare ``mesh_dim_names`` (and the corresponding
        ``mesh_shape``) for validation to proceed; a mesh without names skips
        validation (backward-compatible fallback).

        Degenerate single-rank meshes (all dimensions size 1, common in
        compile-time unit tests) also skip validation — on a single rank every
        DTensor placement is a no-op.

        Rules:
        - tp_size > 1 → mesh must contain a "tp" axis whose size equals tp_size.
        - cp_size > 1 → mesh must contain a "cp" axis whose size equals cp_size.
        - ep_size > 1 is validated only when the mesh has an explicit "ep"
          dimension (old-style EP).  In D-10 TP-extend-EP the ep group is
          *derived* from the dense region, not a native mesh axis, so the
          absence of "ep" in mesh_dim_names is expected and not an error.
        """
        mesh_names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
        mesh_shape = tuple(getattr(mesh, "mesh_shape", ()) or ())
        if not mesh_names or not mesh_shape:
            return  # cannot validate without mesh metadata

        # Degenerate single-rank mesh (compile-time test fixtures): skip
        if all(sz == 1 for sz in mesh_shape):
            return

        name_to_size = dict(zip(mesh_names, mesh_shape))

        for ax, size in (("tp", tp_size), ("cp", cp_size)):
            if size > 1:
                if ax not in name_to_size:
                    raise ValueError(
                        f"{ax}_size={size} > 1, but the mesh has no '{ax}' dimension. "
                        f"Mesh dimensions: {list(name_to_size.keys())}. "
                        f"Either add '{ax}' to the mesh's mesh_dim_names, or set "
                        f"{ax}_size=1."
                    )
                mesh_sz = name_to_size[ax]
                if size != mesh_sz:
                    raise ValueError(
                        f"{ax}_size ({size}) does not match mesh['{ax}'] size "
                        f"({mesh_sz}). They must be equal."
                    )

        # ep_size: only validate when "ep" is a declared mesh axis (old-style
        # EP).  D-10 TP-extend-EP does not put "ep" in mesh_dim_names — the
        # ep group is derived from the dense region by _expert_mesh_layout.
        if ep_size > 1 and "ep" in name_to_size:
            mesh_sz = name_to_size["ep"]
            if ep_size != mesh_sz:
                raise ValueError(
                    f"ep_size ({ep_size}) does not match mesh['ep'] size "
                    f"({mesh_sz}). They must be equal."
                )

    def _build_mesh_dim_names(
        self, mesh, tp_size: int, cp_size: int, ep_size: int,
    ) -> Tuple[str, ...]:
        """Filter tp/cp/ep with mesh.mesh_dim_names as the authoritative
        order; fall back to (tp,cp,ep) when undeclared; drop size=1 axes.

        dp* axes are ALWAYS stripped (05 §3 coordinate-system convention): the plan's
        coordinate system is a single dp slice — dp semantics live in the
        data pipeline (data assignment) and FSDP (weight/grad domain),
        never in placements.  Overrides declaring DP placements are
        rejected fail-first by :meth:`_check_overrides_no_dp`.

        Raises :class:`ValueError` when *tp_size* / *cp_size* do not match the
        corresponding mesh dimension sizes (fail-first before any planning work).
        """
        self._validate_dtensor_axes(mesh, tp_size, cp_size, ep_size)
        mesh_names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
        dtensor_axes = ("tp", "cp", "ep")
        active = {ax for ax, sz in (("tp", tp_size), ("cp", cp_size), ("ep", ep_size))
                  if sz and sz > 1}
        if mesh_names:
            return tuple(n for n in mesh_names if n in dtensor_axes and n in active)
        return tuple(ax for ax in dtensor_axes if ax in active)

    def _check_overrides_no_dp(self) -> None:
        """Fail-first: plan_overrides must not declare DP placements.

        The plan's coordinate system is a single dp slice (05 §3 coordinate-system convention):
        dp placements never drive boundary communication, and parameter dp
        layout is owned by FSDP *after* the plan — a DP key here would be
        silently dropped at resolve time, so reject it with a teaching
        message instead of letting the misreading happen.
        """

        def _declares_dp(node) -> bool:
            return isinstance(node, dict) and any(
                k == DP or _declares_dp(v) for k, v in node.items())

        for fqn, spec in self._plan_overrides.items():
            if not isinstance(spec, ModuleShardingSpec):
                continue   # wrong types are reported as TypeError by _merge_plan_overrides
            for field in ("params", "in_src", "in_dst", "out_src", "out_dst"):
                if _declares_dp(getattr(spec, field)):
                    raise ValueError(
                        f'plan_overrides["{fqn}"].{field}: declaring a DP '
                        "placement is not allowed. The plan's coordinate "
                        "system is a single dp slice (tp/cp/ep, 05 §3 "
                        "coordinate-system convention) — dp data sharding is "
                        "expressed by the data pipeline, parameter/gradient "
                        "sharding by FSDP; for scenarios such as multimodal "
                        "encoder_dp the dp semantics live in vit_mesh + data "
                        "assignment + fully_shard, and the I/O contract only "
                        "needs to declare tp/cp/ep."
                    )

    # ── Phase 1 ─────────────────────────────────────────────────────────

    @staticmethod
    def _warn_uncovered_function_modules(model, plan) -> None:
        """DX guard: a FunctionModule without a boundary spec gets NO
        redistribution communication (it is invisible to the planner's
        derivation) — surface it instead of silently passing."""
        uncovered = [fqn for fqn, m in model.named_modules()
                     if isinstance(m, FunctionModule) and fqn not in plan.modules]
        if uncovered:
            logger.warning(
                "FunctionModule %s has no boundary spec — no communication "
                "will be inserted around it. Declare one via "
                "plan_overrides (params={}, region_dispatch=False).",
                uncovered,
            )

    def _classify_all_params(self, model, arch: str) -> Dict[str, ParamRole]:
        """Phase 1 entry: default naming rules plus the family's declared
        sharding rules (``ModelAdapterSpec.sharding_rules``, resolved through
        the models registry — the planner core carries no per-family
        knowledge)."""
        from hyper_parallel.models.registry import (  # pylint: disable=import-outside-toplevel
            get_model_adapter,
        )
        spec = get_model_adapter(arch)
        rules = []
        if spec is not None and spec.sharding_rules is not None:
            rules = list(spec.sharding_rules())
        if not rules:
            return self._classifier.classify(model, arch)
        classifier = ParameterClassifier(arch_overrides={arch: rules})
        return classifier.classify(model, arch)

    # ── Phase 2 ─────────────────────────────────────────────────────────

    def _group_by_boundary(
        self, param_roles: Dict[str, ParamRole],
    ) -> Dict[str, List[Tuple[str, ParamRole]]]:
        """Single deepest-first sweep over the module hierarchy:

        Pass 1: group by owning module FQN (strip the leaf parameter name).
        Pass 2: materialize ALL ancestor modules upfront and sweep them in
                strictly decreasing depth — every shallower ancestor is
                therefore inferred only after ALL its descendants' parameters
                have merged into it. On ``unknown`` the whole group's
                parameters merge upward into the parent module. If still
                unknown after backtracking to the root, attribute the group
                to the parameter's own module (no template will match later
                → warning and skip).

        (Fix for the merge-ordering hazard of the former tail-enqueue work
        queue: a parent enqueued by an early child — e.g. the MoE gate —
        was inferred before later descendants — e.g. the experts subtree —
        had merged into it, and the late parameters were then silently
        dropped into an already-consumed module. Accuracy fix F3 made this
        observable: a MOE_GATE-only ``mlp`` group must merge upward, which
        stranded the experts params entirely.)
        """
        # Pass 1
        own: Dict[str, List[Tuple[str, ParamRole]]] = {}
        for fqn, role in param_roles.items():
            module_fqn = ".".join(fqn.split(".")[:-1])
            own.setdefault(module_fqn, []).append((fqn, role))

        # Pass 2
        all_modules = set(own)
        for mfqn in own:
            parts = mfqn.split(".")
            for i in range(1, len(parts)):
                all_modules.add(".".join(parts[:i]))
        merged: Dict[str, List[Tuple[str, ParamRole]]] = {
            mfqn: list(own.get(mfqn, [])) for mfqn in all_modules
        }
        groups: Dict[str, List[Tuple[str, ParamRole]]] = {}
        for mfqn in sorted(all_modules, key=lambda f: f.count("."),
                           reverse=True):
            params = merged[mfqn]
            if not params:
                continue
            if self._infer_boundary_type(mfqn, params) != "unknown":
                groups[mfqn] = params
            else:
                parent = mfqn.rsplit(".", 1)[0] if "." in mfqn else ""
                if parent:
                    merged[parent].extend(params)
                else:
                    # Still unknown after backtracking to the root:
                    # attribute to the parameter's own module (no template
                    # will match later → skipped)
                    origin = ".".join(params[0][0].split(".")[:-1]) if params else mfqn
                    groups.setdefault(origin, params)
        return groups

    # ── Phase 3 ─────────────────────────────────────────────────────────

    @staticmethod
    def _explicit_boundary_type(fqn_lower: str, segment: str) -> Optional[str]:
        """Return an explicit FQN-based boundary type when one matches."""
        if _match_any(
                fqn_lower,
                ["embed_tokens", "wte", ".embed.", "tok_embeddings", "embed_in", "word_embeddings"],
        ):
            return "embed"
        if _match_any(fqn_lower, ["lm_head", "embed_out", "output_layer"]):
            return "lm_head"
        if _match_any(fqn_lower, ["norm", "layernorm", "rmsnorm", "ln_"]):
            return "norm"
        if _match_any(segment, ["router"]):
            return "moe_gate"
        return None

    @staticmethod
    def _moe_boundary_type(fqn: str, roles: set[ParamRole]) -> Optional[str]:
        """Return the boundary type implied by MoE-specific parameter roles."""
        if ParamRole.MOE_EXPERT in roles:
            return "moe_mlp" if _match_any(_last_segment(fqn), list(_MOE_CONTAINER_PATTERNS)) else "unknown"
        if ParamRole.SHARED_EXPERT in roles:
            return "mlp"
        if ParamRole.MOE_GATE in roles:
            return "unknown"
        return None

    @staticmethod
    def _dense_boundary_type(fqn_lower: str, group: List[Tuple[str, ParamRole]]) -> str:
        """Infer an attention or MLP boundary from dense parameter roles."""
        has_colwise = any(
            role in (ParamRole.COLWISE, ParamRole.FUSED_QKV, ParamRole.FUSED_GATE_UP)
            for _, role in group
        )
        has_rowwise = any(role == ParamRole.ROWWISE for _, role in group)
        if has_colwise and has_rowwise:
            if _match_any(fqn_lower, list(_ATTN_PATTERNS)):
                return "attention"
            if _match_any(fqn_lower, list(_MLP_PATTERNS)):
                return "mlp"
            return "attention"
        if has_colwise and _match_any(fqn_lower, list(_MLP_PATTERNS)):
            return "mlp"
        return "unknown"

    def _infer_boundary_type(self, fqn: str, group: List[Tuple[str, ParamRole]]) -> str:
        """Identify the semantic role from the module FQN + the group's
        parameter roles.

        Priority: explicit FQN patterns > leaf-segment guard > MoE roles >
        parameter role combinations > default.
        """
        fqn_lower = fqn.lower()
        seg = _last_segment(fqn)

        explicit_type = self._explicit_boundary_type(fqn_lower, seg)
        if explicit_type is not None:
            return explicit_type

        # 2. Leaf-segment guard: projection/expert leaf modules are not
        # boundary containers themselves
        if seg in _LEAF_SEGMENT_GUARD:
            return "unknown"
        # Numeric-segment guard: HF per-expert containers (experts.0..N) are
        # not boundaries; parameters must aggregate upward into the moe
        # container (D-09, 05 §6.4.7)
        if seg.isdigit():
            return "unknown"

        # 3. MoE roles (accuracy fix F3, accuracy_fix_plan.md §2):
        # structural lint — a "moe_mlp" boundary must contain routed experts
        # (MOE_EXPERT); a MOE_GATE-only group (e.g. a scalar gate Linear
        # misclassified as router) must NOT anchor a MoE boundary. Container
        # identity is judged on the module's OWN last segment, not a
        # substring of the whole FQN (a parent path segment like "mlp" must
        # not qualify a leaf Linear such as shared_expert_gate).
        roles = {r for _, r in group}
        moe_type = self._moe_boundary_type(fqn, roles)
        return moe_type if moe_type is not None else self._dense_boundary_type(fqn_lower, group)


    # ── Phase 4 post-processing: MoE EP marking (D-09/D-10, 05 §6.4.7/§6.4.8) ──

    @staticmethod
    def _validate_ep_extend(ep_extend, mesh, model) -> None:
        """D-10 TP-extend-EP validation (05 §6.4.8): ep_size must not
        exceed the dense region and must divide it;
        num_experts % ep_size == 0 (each rank holds num_experts/ep_size
        complete experts).

        The dense region = all ranks of the non-pp mesh axes
        (dp_replicate × dp_cp × tp).
        """
        names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
        shape = tuple(getattr(mesh, "mesh_shape", ()) or ())
        domain = 1
        for name, size in zip(names, shape):
            if name == "pp" and size > 1:
                raise NotImplementedError(
                    "D-10 TP-extend-EP v1 does not support pp>1 "
                    "(split the mesh by stage before calling)"
                )
            if name != "pp":
                domain *= size
        if ep_extend > domain or domain % ep_extend != 0:
            raise ValueError(
                f"ep_size ({ep_extend}) must not exceed and must divide "
                f"the dense region (dp_replicate × dp_cp × tp = {domain})"
            )
        num_experts = (getattr(getattr(model, "config", None), "num_experts", None)
                       or getattr(getattr(model, "config", None), "n_routed_experts", 0))
        if num_experts and num_experts % ep_extend != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size ({ep_extend})"
            )

    # per-expert parameter pattern: experts.<idx>.<proj>.weight (legacy HF /
    # in-house MoE layouts).
    _PER_EXPERT_RE = re.compile(r"^experts\.(\d+)\.([^.]+)\.weight$")
    # batched parameter pattern: experts.<attr> (a single attribute with no
    # numeric segment, the layout after the HF 2025 refactor —
    # gate_up_proj [E, 2I, H] / down_proj [E, H, I], natively stacked with
    # no stacking needed; the automodel names gate_and_up_projs/down_projs
    # are isomorphic).
    _BATCHED_EXPERT_RE = re.compile(
        r"^experts\.(gate_up_proj|gate_and_up_projs|down_proj|down_projs"
        r"|gate_proj|up_proj)$")
    # Fused (gate|up merged into ONE parameter, chunk-split inside forward)
    # expert weight pattern — covers both the batched 3D layout
    # (``experts.gate_up_proj [E, 2I, H]``) and the per-expert 2D layout
    # (``experts.<idx>.gate_up_proj.weight [2I, H]``). Such weights are
    # incompatible with contiguous-block TP Shard on the fused dim; see
    # _finalize_fused_expert_tp_guard.
    _FUSED_EXPERT_WEIGHT_RE = re.compile(
        r"^experts\.(?:\d+\.)?(?:gate_up_proj|gate_and_up_projs"
        r"|fused_gate_up|w13)(?:\.weight)?$")

    def _detect_expert_layout(self, group, boundary_fqn: str, param_ndims):
        """Return expert parameters grouped by per-expert and batched layouts."""
        expert_params = [fqn for fqn, role in group if role == ParamRole.MOE_EXPERT]
        if not expert_params:
            return None
        stacks: Dict[str, List[Tuple[int, str]]] = {}
        batched: List[str] = []
        for param_fqn in expert_params:
            rel = param_fqn[len(boundary_fqn) + 1:]
            if "bias" in rel.lower():
                logger.warning(
                    "%s: MoE expert has bias (%s); not supported in v1, skipping EP marking",
                    boundary_fqn, rel,
                )
                return None
            match = self._PER_EXPERT_RE.match(rel)
            if match is not None:
                stacks.setdefault(match.group(2), []).append((int(match.group(1)), rel))
            elif self._BATCHED_EXPERT_RE.match(rel) is not None and (param_ndims or {}).get(param_fqn, 2) >= 3:
                batched.append(rel)
        if stacks and batched:
            logger.warning(
                "%s: mixed per-expert and batched layouts (%s ...); skipping EP marking",
                boundary_fqn, batched[0],
            )
            return None
        return expert_params, stacks, batched

    @staticmethod
    def _mark_old_style_experts(
        spec: ModuleShardingSpec,
        stacks: Dict[str, List[Tuple[int, str]]],
        template: ShardingTemplate,
    ) -> None:
        """Stack per-expert parameters while retaining explicit TP and EP axes."""
        for projection, items in stacks.items():
            items.sort()
            sources = [rel for _, rel in items]
            stacked = f"experts.{projection}"
            tp_placement = _moe_expert_tp_placement(stacked, ndim=3, template=template)
            for rel in sources:
                spec.params.pop(rel, None)
            spec.params[stacked] = _multi_dim(
                tp=tp_placement, cp=Replicate(), ep=template.moe_expert_placement
            )
            spec._ep_stack[stacked] = sources  # pylint: disable=protected-access  # planner owns the spec DSL internals

    @staticmethod
    def _mark_extended_expert_params(
        spec: ModuleShardingSpec,
        expert_params: List[str],
        stacks: Dict[str, List[Tuple[int, str]]],
        batched: List[str],
        boundary_fqn: str,
        template: ShardingTemplate,
    ) -> None:
        """Mark expert parameters for TP-extended EP sharding."""
        if stacks:
            for projection, items in stacks.items():
                items.sort()
                sources = [rel for _, rel in items]
                stacked = f"experts.{projection}"
                for rel in sources:
                    spec.params.pop(rel, None)
                spec.params[stacked] = _multi_dim(
                    tp=None, cp=Replicate(), ep=template.moe_expert_placement
                )
                spec._ep_stack[stacked] = sources  # pylint: disable=protected-access  # planner owns the spec DSL internals
            spec.region_dispatch = None
            return
        if batched:
            for rel in batched:
                spec.params[rel] = _multi_dim(
                    tp=None, cp=Replicate(), ep=template.moe_expert_placement
                )
            spec.region_dispatch = None
            return
        for param_fqn in expert_params:
            rel = param_fqn[len(boundary_fqn) + 1:]
            spec.params[rel] = _multi_dim(
                tp=None, cp=Replicate(), ep=template.moe_expert_placement
            )

    @staticmethod
    def _set_extended_ep_contract(spec: ModuleShardingSpec, ep_extend: int) -> None:
        """Set the identity boundary contract used by TP-extended EP."""
        identity = copy.deepcopy(spec.in_src)
        spec.in_dst = copy.deepcopy(identity)
        out_layout = copy.deepcopy(next(iter(identity.values())))
        spec.out_src = {"output": copy.deepcopy(out_layout)}
        spec.out_dst = {"output": copy.deepcopy(out_layout)}
        spec._ep_size = ep_extend  # pylint: disable=protected-access  # planner owns the spec DSL internals

    def _mark_hf_native_moe(
        self, spec: ModuleShardingSpec, group, boundary_fqn: str,
        template: ShardingTemplate, mesh_dim_names: Tuple[str, ...], arch: str,
        *, ep_extend: int = 0, mesh=None, model=None, param_ndims=None,
    ) -> None:
        """Post-process moe_mlp spec for EP (05 §6.4.7/§6.4.8).

        The EP **mode** is determined solely by whether the mesh includes an
        explicit ``"ep"`` axis — NOT by parameter naming:

        * **Old-style EP** (``"ep" in mesh_dim_names``): the mesh already has
          an ``"ep"`` axis.  :meth:`_build_spec_from_template` produced
          ``{TP: Shard(…), EP: Shard(0)}`` — the correct dual-axis
          sharding.  This method only performs **stacking** when the expert
          layout is per-expert 2D (``experts.<idx>.<proj>.weight``); the
          stacked entry keeps both TP and EP keys.  Batched 3D and custom
          (``w1``/``w2``/``w3``) layouts are already ready — nothing to do.
          No ``_ep_size`` is set; communication is handled by the module's
          own dispatcher or external ``_attach_ep``.

        * **D-10 TP-extend-EP** (``"ep" not in mesh_dim_names``, the
          default for HF-native models): the ep group is *derived* from the
          dense region.  Expert weights are rewritten to
          ``{EP: Shard(0)}`` (no TP key), the boundary contract is changed
          to SP-in identity, and ``_ep_size`` is set.

        The expert **layout** (per-expert / batched / custom) only affects
        the *stacking strategy* — orthogonal to the EP mode:

        - **per-expert**: ``experts.<idx>.<proj>.weight`` → stack into
          ``experts.<proj>`` 3D before sharding;
        - **batched** (D-11): ``experts.gate_up_proj`` etc., natively 3D;
        - **custom**: ``experts.w1`` / ``w2`` / ``w3``, already 3D,
          pre-stacked by the module author.

        In D-10 mode all three layouts are supported; in old-style EP mode
        only per-expert needs stacking (batched and custom are already 3D).
        """
        # ``arch`` is part of the dispatch-interface signature shared with the
        # other boundary post-processors; the EP mode is decided by the mesh
        # and layout alone, so it is intentionally unused here.
        _ = arch
        if not ep_extend:
            return
        layout = self._detect_expert_layout(group, boundary_fqn, param_ndims)
        if layout is None:
            return
        expert_params, stacks, batched = layout
        if "ep" in mesh_dim_names:
            if stacks:
                self._mark_old_style_experts(spec, stacks, template)
            return
        self._validate_ep_extend(ep_extend, mesh, model)
        self._mark_extended_expert_params(
            spec, expert_params, stacks, batched, boundary_fqn, template
        )
        self._set_extended_ep_contract(spec, ep_extend)


    @staticmethod
    def _finalize_fused_expert_tp_guard(plan: ShardingPlan, *, tp_size: int) -> None:
        """Fail fast on fused expert weights TP-sharded without D-10 EP.

        A fused expert weight (gate/up merged into ONE parameter — e.g.
        ``experts.gate_up_proj [E, 2I, H]`` — and chunk-split inside the
        module's forward) is incompatible with the framework's
        contiguous-block TP ``Shard`` on the fused dim: after Phase A each
        rank holds a contiguous slice of the fused dim (tp=2 → rank0 all
        gate, rank1 all up), so the runtime ``chunk`` result is NOT
        equivalent to the non-TP semantics (Megatron solves this with
        chunk-aware sharding, which the framework does not support). The
        error is silent: the MoE boundary is local-region black-box hosted
        in BOTH modes, so production and validate run the same (wrong)
        local forward and the dual-mode comparison cannot expose it.

        Runs AFTER all plan_overrides are merged (next to
        ``_finalize_deferred_biases``) and anchors detection on the FINAL
        spec declarations, so an explicit TP-Replicate override on the
        fused params legitimately clears the hazard. D-10 TP-extend-EP
        (``_ep_size > 0``) removes the TP key from expert weights entirely
        (each rank holds complete experts), so it is exempt.
        """
        if tp_size <= 1:
            return
        offenders = []
        for fqn, spec in plan.modules.items():
            if getattr(spec, "_ep_size", 0):  # pylint: disable=protected-access
                # D-10 TP-extend-EP: expert weights carry no TP shard — the
                # fused dim stays complete per rank, chunk is correct.
                continue
            for param_name, named in (spec.params or {}).items():
                if not isinstance(named, dict):
                    continue
                if ShardingPlanner._FUSED_EXPERT_WEIGHT_RE.match(param_name) is None:
                    continue
                if isinstance(named.get(TP), Shard):
                    offenders.append((fqn, param_name, named[TP]))
        if not offenders:
            return
        listing = "\n".join(
            f"  - {fqn}.{param_name}  →  {{TP: {placement}, ...}}"
            for fqn, param_name, placement in offenders[:8])
        more = (f"\n  ... and {len(offenders) - 8} more"
                if len(offenders) > 8 else "")
        boundary_fqn, example_param, _ = offenders[0]
        raise ValueError(
            f"Detected {len(offenders)} fused MoE expert weight(s) (gate/up "
            "merged into a single parameter, chunk-split inside forward) "
            "declared with TP sharding:\n"
            f"{listing}{more}\n"
            "This pattern is incompatible with the framework's contiguous-"
            "block TP Shard: after Phase A shards the fused dim contiguously, "
            "each rank holds a slice whose gate/up layout is misordered, so "
            "the runtime chunk is NOT equivalent to the non-TP semantics "
            "(Megatron handles this with chunk-aware sharding, which the "
            "framework does not support). Worse, the error is silent: the "
            "MoE boundary is local-region black-box hosted in BOTH modes, so "
            "production and validate run the same (wrong) local forward and "
            "the dual-mode comparison cannot expose the accuracy corruption. "
            "Choose one of:\n"
            "  (1) Enable ep_size (recommended): "
            "ShardingPlanner().plan(..., ep_size=N) with N>1, N dividing the "
            "dense region dp*cp*tp and dividing num_experts. Enabling ep_size "
            "takes the TP-extend-EP path (design revision D-10): the dense "
            "region is re-partitioned into the derived expert mesh "
            "(edp_shard, ep), and expert weights are sharded ONLY as "
            "{EP: Shard(0)} along the expert dim with NO TP key — every rank "
            "holds complete experts, so the gate/up produced by the runtime "
            "chunk matches the non-parallel semantics. Note: the MoE "
            "boundary must then also declare local_compute_fn explicitly "
            "(EP compute injection; the apply-time preflight checks this and "
            "prints a YAML example).\n"
            "  (2) Keep ep=1: override the fused weight to TP Replicate via "
            "plan_overrides (it then opts out of TP sharding and saves no "
            "memory for that weight), e.g.:\n"
            "     ShardingPlanner(plan_overrides={\n"
            f"       {boundary_fqn!r}: ModuleShardingSpec(\n"
            "         params={\n"
            f"           {example_param!r}: "
            "{TP: Replicate(), CP: Replicate()},\n"
            "           # ...declare ALL params of this boundary too "
            "(experts.down_proj / gate.weight etc.) — params merge replaces "
            "the field wholesale (no per-key merge)\n"
            "         },\n"
            "         # out_src TP must ALSO be changed from the template-"
            "derived Partial to Replicate: with replicated weights each "
            "rank's MoE output is already complete (not Partial); keeping "
            "Partial would make the boundary exit reduction count the "
            "output tp_size times\n"
            "         out_src={'output': {TP: Replicate(), CP: Replicate()}},"
            "  # keep the template-derived CP value (Shard(1) when "
            "sequence_parallel is on)\n"
            "       ),\n"
            "     })\n"
            "  (3) Switch to separate w1/w3 expert weights (gate_proj and "
            "up_proj as independent parameters) — natively supported, no "
            "override needed.")

    @staticmethod
    def _should_defer_rowwise_bias(
        module_fqn: str,
        module: nn.Module,
        owner_path: str,
        param_name: str,
        bias_tp,
        out_src,
        partial_outputs: List[str],
    ) -> bool:
        """Validate a rowwise bias and return whether it must be deferred."""
        if not partial_outputs:
            return False
        if len(out_src) != 1:
            raise ValueError(
                f"boundary {module_fqn!r}: rowwise bias deferral (D-22) v1 only supports single-output "
                f"boundaries — this boundary's out_src declares {len(out_src)} outputs with a TP Partial "
                f"reduction, so the framework cannot attribute {param_name!r} to a unique output. Take over "
                "the region with local_compute_fn (add the bias yourself after the reduction)"
            )
        if bias_tp is not None and not isinstance(bias_tp, Replicate):
            raise ValueError(
                f"boundary {module_fqn!r}: the bias of rowwise Linear {owner_path!r} declares a non-Replicate "
                f"TP placement ({bias_tp!r}) — D-22 deferred addition requires the bias to stay Replicate "
                f"(added exactly once as a whole after the TP reduction). Remove {param_name!r} from "
                "spec.params, or change it to {TP: replicate()}"
            )
        owner = module.get_submodule(owner_path) if owner_path else module
        if isinstance(owner, nn.Linear):
            return True
        logger.warning(
            "boundary %s: rowwise Linear %r carries a bias and the boundary out_src is TP Partial — but "
            "the owner type is %s (not nn.Linear), so the framework does not touch its forward semantics: "
            "the bias will be counted multiple times by the Partial reduction (production output = correct "
            "value + tp_size × bias). Move the bias after the boundary communication, switch to nn.Linear, "
            "or take over the region with local_compute_fn",
            module_fqn, owner_path, type(owner).__name__,
        )
        return False

    @staticmethod
    def _validate_colwise_bias(
        module_fqn: str,
        weight_path: str,
        param_name: str,
        param,
        bias_named,
        bias_tp,
        shard_dim: int,
    ) -> None:
        """Require a colwise bias to follow its weight's output shard."""
        bias_dim = None
        if isinstance(bias_tp, Shard):
            bias_dim = bias_tp.dim if bias_tp.dim >= 0 else bias_tp.dim + param.ndim
        if bias_dim == shard_dim:
            return
        declared = repr(bias_tp) if bias_named else "undeclared"
        raise ValueError(
            f"boundary {module_fqn!r}: {weight_path!r} is sharded along the output dim as Shard({shard_dim}), "
            f"but {param_name!r} is not sharded along the output channels the same way ({declared}) — "
            f"template mismatch (typical: lm_head.bias). Declare it explicitly via plan_overrides as "
            f"{{'{param_name}': {{TP: shard({shard_dim})}}}}, or remove the bias"
        )

    @staticmethod
    def _collect_deferred_biases(module_fqn: str, spec: ModuleShardingSpec, module: nn.Module) -> List[str]:
        """Validate one boundary's biases and collect rowwise deferred names."""
        named_params = dict(module.named_parameters())
        out_src = spec.out_src or {}
        partial_outputs = [
            out_name for out_name, named in out_src.items() if isinstance(named.get(TP), Partial)
        ]
        deferred: List[str] = []
        for param_name, param in named_params.items():
            if param_name == "bias":
                owner_path = ""
            elif param_name.endswith(".bias"):
                owner_path = param_name[: -len(".bias")]
            else:
                continue
            weight_path = f"{owner_path}.weight" if owner_path else "weight"
            weight = named_params.get(weight_path)
            weight_named = (spec.params or {}).get(weight_path)
            if weight is None or weight_named is None:
                continue
            tp_placement = weight_named.get(TP)
            if not isinstance(tp_placement, Shard):
                continue
            shard_dim = tp_placement.dim if tp_placement.dim >= 0 else tp_placement.dim + weight.ndim
            bias_named = spec.params.get(param_name)
            bias_tp = bias_named.get(TP) if bias_named else None
            if shard_dim == weight.ndim - 1:
                if ShardingPlanner._should_defer_rowwise_bias(
                    module_fqn, module, owner_path, param_name, bias_tp, out_src, partial_outputs
                ):
                    deferred.append(param_name)
            else:
                ShardingPlanner._validate_colwise_bias(
                    module_fqn, weight_path, param_name, param, bias_named, bias_tp, shard_dim
                )
        return deferred

    @staticmethod
    def _finalize_deferred_biases(
        plan: ShardingPlan, model, mesh_dim_names: Tuple[str, ...],
    ) -> None:
        """D-22: decide which Linear biases are deferred to after the boundary
        exit TP reduction, and validate every bias against its sibling weight's
        sharding direction.

        Runs AFTER all plan_overrides are merged (next to
        ``_finalize_tp_local_attr_plans``) and anchors detection on the FINAL
        spec declarations plus the model structure — never on ParamRole — so
        derived / merge / insert / derive=False specs (and family sharding-rule
        naming like ``wo``/``c_proj``) all share one code path.

        Per boundary, for every physical ``X.bias`` parameter (scanned from
        the module, regardless of whether spec.params declares it — a
        physically present bias is fused by F.linear either way):

        - sibling ``X.weight`` TP placement is ``Shard(weight.ndim - 1)``
          (contraction-dim = rowwise) AND the boundary out_src reduces on TP
          (Partial) → the fused bias would be counted once per TP rank by the
          exit reduction → defer it (bias stays Replicate, added exactly once
          after the reduction). Fail-fast guards: a declared non-Replicate
          bias placement is rejected; a non-nn.Linear owner gets a WARNING
          and is skipped (its forward semantics are not touched); a
          multi-output boundary cannot attribute the bias to one output →
          fail-fast (use local_compute_fn).
        - sibling ``X.weight`` TP placement is an output-dim ``Shard(d)``
          (colwise / lm_head / embed) → the bias must follow the same output
          shard; Replicate / undeclared / wrong-dim → plan-time "template
          mismatch"
          error (typical: lm_head.bias), instead of a remote runtime
          broadcast crash.
        - out_src without a TP Partial (no boundary reduction, or the user
          reduces inside the region) → the bias is already added exactly once
          → no defer, no check.
        """
        has_tp = "tp" in mesh_dim_names
        modules = dict(model.named_modules())
        for module_fqn, spec in plan.modules.items():
            spec._deferred_bias_params = ()  # pylint: disable=protected-access
            if not has_tp or not spec.is_boundary:
                continue
            module = modules.get(module_fqn)
            if module is None:
                continue
            deferred = ShardingPlanner._collect_deferred_biases(module_fqn, spec, module)
            spec._deferred_bias_params = tuple(deferred)  # pylint: disable=protected-access
            if deferred:
                logger.info(
                    "boundary %s: deferred bias (D-22, added exactly once "
                    "after the TP reduction): %s",
                    module_fqn, list(deferred))

    def _check_param_uniqueness(self, plan: ShardingPlan) -> None:
        """D-14 invariant 1 (05 §13.3): every parameter is sharded by exactly
        one boundary. spec.params keys are resolved to full parameter FQNs
        (relative to the boundary module); any parameter declared by ≥2 specs
        fails fast (double sharding corrupts silently under production)."""
        seen: Dict[str, str] = {}  # full param fqn -> first declaring boundary
        for fqn, spec in plan.modules.items():
            prefix = fqn + "." if fqn else ""
            for pname in spec.params:
                full = prefix + pname
                if full in seen:
                    raise ValueError(
                        f"param {full!r} is declared by two boundaries: "
                        f"{seen[full]!r} and {fqn!r} — each parameter must be "
                        f"sharded by exactly one boundary (D-14, 05 §13.3); "
                        f"drop it from one of the specs (an outer boundary "
                        f"may only declare params of its own/intermediate "
                        f"modules, never of a nested boundary's subtree)"
                    )
                seen[full] = fqn

    def _check_full_declaration(self, plan: ShardingPlan) -> None:
        """D-14 (05 §13.2): chain fill is removed — every boundary spec must
        fully declare its I/O contract. A non-empty in_dst with an empty
        in_src fails fast (the previous Scenario-1 fill no longer exists)."""
        for fqn, spec in plan.modules.items():
            if spec.is_boundary and spec.in_dst and not spec.in_src:
                raise ValueError(
                    f"boundary {fqn!r} declares in_dst but an empty in_src — "
                    f"chain fill was removed (D-14, 05 §13.2); declare in_src "
                    f"explicitly (keys must mirror in_dst)"
                )

    # ── F4 plan-time lints (accuracy_fix_plan.md §2) ─────────────────────

    def _check_shard_divisibility(
        self,
        plan: ShardingPlan,
        param_shapes: Dict[str, Tuple[int, ...]],
        *,
        tp_size: int,
        cp_size: int,
        ep_size: int,
    ) -> None:
        """F4a: every Shard(dim) in every spec.params must divide the
        parameter's shape along that dim — fail at plan time with a teaching
        error instead of producing an empty shard at apply time
        (accuracy_problem.md 10.1: a (1, 64) scalar-gate weight misclassified
        SHARED_EXPERT was Shard(0)'d over tp=2 into a (0, 64) empty shard,
        surfacing only much later at weight reconstruction).

        Runs after the Phase 4.5 override merge, so hand-written specs are
        checked too. Stacked expert params (D-09/D-10 ``_ep_stack``) resolve
        their shape as ``(num_experts, *source_shape)``; D-10 experts use
        ``spec._ep_size`` as the EP axis size. Params not resolvable at plan
        time (e.g. created later by an inner_target factory) are skipped.
        """
        for fqn, spec in plan.modules.items():
            prefix = fqn + "." if fqn else ""
            ep_stack = getattr(spec, "_ep_stack", None) or {}
            # D-10: the EP axis lives on the derived expert mesh whose size
            # is spec._ep_size; old-style EP uses the mesh "ep" axis (ep_size).
            axis_sizes = {
                "tp": tp_size,
                "cp": cp_size,
                "ep": getattr(spec, "_ep_size", 0) or ep_size,
            }
            for pname, placement in (spec.params or {}).items():
                full = prefix + pname
                shape = param_shapes.get(full)
                if shape is None and pname in ep_stack:
                    sources = ep_stack[pname]
                    src_shape = (param_shapes.get(prefix + sources[0])
                                 if sources else None)
                    if src_shape is not None:
                        shape = (len(sources), *src_shape)
                if shape is None:
                    continue  # created later (inner_target factory) — nothing to check
                ndim = len(shape)
                for axis, p in (placement or {}).items():
                    if not isinstance(p, Shard):
                        continue
                    size = axis_sizes.get(axis, 1)
                    if size <= 1:
                        continue
                    axis_name = getattr(axis, "value", axis)  # MeshAxisName → "tp"
                    dim = p.dim + ndim if p.dim < 0 else p.dim
                    if dim >= ndim:
                        raise ValueError(
                            f"plan-time shard check failed: {full!r} has shape "
                            f"{tuple(shape)} but boundary {fqn!r} declares "
                            f"{{{axis_name}: Shard({p.dim})}} — dim {p.dim} is out "
                            f"of range for a {ndim}D parameter; fix the "
                            f"plan_overrides declaration"
                        )
                    if shape[dim] % size != 0:
                        raise ValueError(
                            f"plan-time shard check failed: {full!r} has shape "
                            f"{tuple(shape)} but boundary {fqn!r} declares "
                            f"{{{axis_name}: Shard({p.dim})}} — shape[{dim}]={shape[dim]} "
                            f"is not divisible by {axis_name} size {size} (it would "
                            f"produce empty shards at apply time). This is most "
                            f"often a parameter-classification error (e.g. a "
                            f"replicated/gate parameter misclassified into a "
                            f"sharded role — see accuracy_fix_plan.md §2): fix "
                            f"the naming rule / family sharding_rules entry, or correct "
                            f"the plan_overrides declaration"
                        )

    def _check_all_trainable_params_covered(self, plan: ShardingPlan, model) -> None:
        """F4b: every ``requires_grad=True`` parameter must appear in some
        spec.params (resolved to a full FQN) or in special_handlers —
        otherwise its gradient-sync semantics would be decided SILENTLY by
        the consumer-side default (the same class of shape-legal/silent-wrong
        hazard as accuracy_problem.md 10.1/10.2), so the plan must be a
        complete declaration of every trainable parameter's layout.

        Covered = spec.params ∪ _ep_stack sources ∪ special_handlers (a
        param explicitly declared via plan_overrides therefore counts as
        covered). Remediation paths, listed per parameter in the error:
        ① classify it explicitly (naming rule / the family's sharding_rules, e.g.
          shared_expert_gate → REPLICATED);
        ② declare it in a spec via plan_overrides (e.g. all-Replicate);
        ③ mark it SPECIAL (special_handlers) or freeze it
          (requires_grad=False).
        Escape hatch for exploratory debugging:
        ``ShardingPlanner(allow_uncovered_params=True)`` downgrades to a
        warning. Skipped entirely under derive=False (pure declaration
        assembler for subtrees — coverage is the caller's responsibility).
        """
        if not self._derive:
            return
        covered = set()
        for fqn, spec in plan.modules.items():
            prefix = fqn + "." if fqn else ""
            for pname in (spec.params or {}):
                covered.add(prefix + pname)
            for sources in (getattr(spec, "_ep_stack", None) or {}).values():
                covered.update(prefix + s for s in sources)
        covered.update(plan.special_handlers or {})
        uncovered = [
            name for name, p in model.named_parameters()
            if p.requires_grad and name not in covered
        ]
        if not uncovered:
            return
        lines = "\n".join(f"  - {n}" for n in uncovered[:20])
        more = (f"  ... and {len(uncovered) - 20} more\n"
                if len(uncovered) > 20 else "")
        msg = (
            f"plan-time coverage check failed: {len(uncovered)} trainable "
            f"parameter(s) are not covered by any spec.params / "
            f"special_handlers:\n{lines}\n{more}"
            "An uncovered parameter is never sharded (kept replicated) AND is "
            "absent from source_shard_info — its gradient-sync semantics would be "
            "decided silently by the consumer-side default. Resolve each "
            "parameter explicitly: ① add a naming rule / a family sharding_rules "
            "entry "
            "(e.g. shared_expert_gate → REPLICATED); ② declare it in a spec "
            "via plan_overrides; ③ mark it SPECIAL or freeze it "
            "(requires_grad=False). For exploratory debugging only, construct "
            "ShardingPlanner(allow_uncovered_params=True) to downgrade this "
            "error to a warning (accuracy_fix_plan.md §2 F4b)"
        )
        if self._allow_uncovered_params:
            logger.warning("%s", msg)
        else:
            raise ValueError(msg)

    # ── Phase 5 ─────────────────────────────────────────────────────────

    def _mark_terminal(self, plan: ShardingPlan, model) -> ShardingPlan:
        """D-14 (05 §13.2): compile-time chain propagation/validation is
        removed; only ``_is_terminal`` marking is kept — the last boundary
        in forward order (each module vouches for its own propagation in
        validate mode; adjacent-contract checks no longer exist)."""
        sorted_fqns = self._topological_sort_by_forward_order(
            list(plan.modules.keys()), model
        )
        terminal = sorted_fqns[-1] if sorted_fqns else None
        for fqn, spec in plan.modules.items():
            spec._is_terminal = fqn == terminal  # pylint: disable=protected-access  # planner owns the spec DSL internals
        return plan

    def _topological_sort_by_forward_order(self, fqns: List[str], model) -> List[str]:
        """Sort by named_modules registration order; unmatched FQNs are
        appended at the end with a warning."""
        fqn_set = set(fqns)
        ordered: List[str] = []
        seen: set = set()
        for name, _ in model.named_modules():
            if name in fqn_set and name not in seen:
                ordered.append(name)
                seen.add(name)
        missing = fqn_set - seen
        if missing:
            logger.warning(
                "_topological_sort_by_forward_order: %d FQNs not found in "
                "named_modules; appended at the end: %s",
                len(missing), sorted(missing)[:5],
            )
            ordered.extend(sorted(missing))
        return ordered


    # ── tied weights ────────────────────────────────────────────────────

    @staticmethod
    def _detect_tied_pairs(model) -> List[Tuple[str, str]]:
        """Detect the embed_tokens.weight <-> lm_head.weight tied pair.

        With HF tie_word_embeddings both ends share storage; in PP
        scenarios the two ends cannot be detected across stages, so the
        user must declare plan.tied_pairs explicitly (05
        detect_tied_weights comment).
        """
        if not getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            return []
        embed_fqn = lm_head_fqn = None
        # remove_duplicate=False: under the default deduplication of
        # named_parameters a tied parameter appears only once.
        for name, _ in model.named_parameters(remove_duplicate=False):
            if name.endswith("embed_tokens.weight"):
                embed_fqn = name
            elif name.endswith("lm_head.weight"):
                lm_head_fqn = name
        if embed_fqn and lm_head_fqn:
            return [(embed_fqn, lm_head_fqn)]
        return []


def validate_model_compatibility(
    model: Any, *, tp_size: int = 1, cp_size: int = 1, ep_size: int = 1,
    seq_len: Optional[int] = None,
) -> None:
    """Model-side compatibility validation (05 §6.5; division of labor
    with 06's topology validation — this only inspects the model
    config)."""
    config = getattr(model, "config", None)
    if config is None:
        return

    if tp_size > 1:
        heads = getattr(config, "num_attention_heads", None)
        if heads is not None and heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({heads}) must be divisible by TP ({tp_size})"
            )
        kv_heads = getattr(config, "num_key_value_heads", None)
        if kv_heads is not None and kv_heads % tp_size != 0:
            raise ValueError(
                f"num_key_value_heads ({kv_heads}) must be divisible by TP ({tp_size})"
            )
        moe_inter = getattr(config, "moe_intermediate_size", None)
        if ep_size > 1 and moe_inter is not None and moe_inter % tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size ({moe_inter}) must be divisible by TP ({tp_size})"
            )

    if cp_size > 1 and seq_len is not None and seq_len % (cp_size * 2) != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by 2*cp ({2 * cp_size})"
        )

    if ep_size > 1:
        num_experts = (getattr(config, "num_experts", None)
                       or getattr(config, "n_routed_experts", None) or 0)
        if num_experts <= 0:
            raise ValueError("EP>1 requires MoE model (num_experts > 0)")
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by EP ({ep_size})"
            )
