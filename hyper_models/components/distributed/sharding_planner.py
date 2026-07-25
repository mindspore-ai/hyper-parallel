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
"""sharding_planner: ShardingPlanner 6-phase derivation pipeline (05 §3.6 canonical).

Phase 1  parameter role classification (ParameterClassifier + ARCH_OVERRIDES)
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
- ``ARCH_OVERRIDES``: {arch_name: [(pattern | [patterns], ParamRole)]}
- ``SPECIAL_HANDLERS``: {handler_name: callable(module, param_name, mesh)}
"""

import copy
import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_models.components.distributed.ep_utils import MOE_ROUTER_ADAPTERS
from hyper_models.components.distributed.param_role import (
    ParameterClassifier,
    ParamRole,
    _match_any,
)
from hyper_models.components.distributed.sharding_config import (
    EP,
    TP,
    ModuleShardingSpec,
    NamedPlacement,
    ShardingPlan,
    ShardingTemplate,
    TEMPLATES,
    _multi_dim,
    _normalize_out_fields,
    resolve_placements,
)

logger = logging.getLogger(__name__)

# {arch_name: [(pattern | [patterns], ParamRole)]} — arch-level naming
# overrides (Option B: replicate down-projections, colwise up-projections).
# A pattern is a lowercase substring (or list of substrings); a hit forces
# the parameter into that role.
# DeepSeek MLA (deepseek_v2/v3 share the same structure): the q_a/kv_a
# down-projections are forced to replicated (the LoRA rank dim is not
# sharded); the q_b/kv_b up-projections are colwise along the head dim —
# isomorphic to the standard attention template (the o_proj rowwise
# contract over the head dim is unchanged). Keys are registered under both
# the architectures spelling ("deepseekv3") and the model_type spelling
# ("deepseek_v3").
_DEEPSEEK_MLA_OVERRIDES = [
    (["q_a_proj", "kv_a_proj_with_mqa"], ParamRole.REPLICATED),
    (["q_b_proj", "kv_b_proj"], ParamRole.COLWISE),
]
ARCH_OVERRIDES: Dict[str, list] = {
    "llama": [],
    "qwen2": [],
    "qwen3": [],
    "mixtral": [],
    "deepseekv2": _DEEPSEEK_MLA_OVERRIDES,
    "deepseekv3": _DEEPSEEK_MLA_OVERRIDES,
    "deepseek_v2": _DEEPSEEK_MLA_OVERRIDES,
    "deepseek_v3": _DEEPSEEK_MLA_OVERRIDES,
}


def _shard_gated_delta(module, param_name, mesh):
    """Custom TP sharding skeleton for gated_delta modules (SSM/Mamba-style
    modules, 05 §6.4.6).

    Shards along the SSM head structure rather than standard
    colwise/rowwise. Skeleton implementation: structural recognition plus
    a standard Shard(0) fallback; the head-aligned fine-grained sharding is
    left to be completed when a concrete model is onboarded.
    """
    import torch.nn as nn

    param = getattr(module, param_name, None)
    if param is None:
        return
    sharded = distribute_tensor(param.data, mesh, [Shard(0)])
    module.register_parameter(param_name, nn.Parameter(sharded))


# {handler_name: callable(module, param_name, mesh)} — Phase B special parameter handlers.
SPECIAL_HANDLERS: Dict[str, Callable] = {
    "gated_delta_tp_shard": _shard_gated_delta,
}

# planner-side pattern → handler_name mapping (lowercase fqn substring match).
_SPECIAL_HANDLER_PATTERNS: Dict[str, str] = {
    "gated_delta": "gated_delta_tp_shard",
    "a_log": "gated_delta_tp_shard",
    "dt_bias": "gated_delta_tp_shard",
}

# Leaf-segment guard for projection/container segment names: these segment
# names are not boundary containers themselves; inference returns unknown
# and continues upward.
_LEAF_SEGMENT_GUARD = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "qkv_proj", "fused_qkv", "gate_up_proj", "query_key_value",
    "experts", "shared_experts", "gate", "linear", "proj",
    "fc1", "fc2", "w1", "w2", "w3", "w13", "dense", "dense_h_to_4h", "dense_4h_to_h",
})

_ATTN_PATTERNS = ("attn", "attention")
_MLP_PATTERNS = ("mlp", "ffn", "feed_forward")
_MOE_CONTAINER_PATTERNS = ("mlp", "moe", "moe_block", "moe_layer")


def _last_segment(fqn: str) -> str:
    return fqn.rsplit(".", 1)[-1].lower() if fqn else ""


def _infer_colwise_vs_rowwise(param_path: str, template: ShardingTemplate):
    """Infer the TP placement from the parameter name suffix: w2/down → rowwise, everything else → colwise."""
    name = param_path.lower()
    if any(k in name for k in ("w2", "down_proj", "down.")):
        return template.rowwise_placement
    return template.colwise_placement


def _moe_expert_tp_placement(param_path: str, ndim: int,
                             template: ShardingTemplate):
    """TP placement for MOE_EXPERT (revision D-08, ndim-aware per parameter).

    When expert weights use a batched 3D layout [E, H_out, H_in] (ndim>=3),
    tensor dim 0 is the expert dim (owned by EP Shard(0)), so the TP
    colwise/rowwise sharding must apply at dim +1:
    colwise (shard H_out) → Shard(1); rowwise (shard the contraction dim
    H_in) → Shard(2).
    The per-expert 2D layout (experts.N.w1 [H_out, H_in]) keeps the standard
    Shard(0)/Shard(1) — but then EP Shard(0) would shard H_out, which is
    semantically invalid: EP must be implemented as "each rank holds a
    subset of experts" (module level), requiring ARCH_OVERRIDES /
    SpecialHandler; it is outside template coverage.
    """
    name = param_path.lower()
    is_rowwise = any(k in name for k in ("w2", "down_proj", "down."))
    if ndim >= 3:
        return Shard(2) if is_rowwise else Shard(1)
    return template.rowwise_placement if is_rowwise else template.colwise_placement


class ShardingPlanner:
    """Automatically derive a ShardingPlan from any HF-style model (05 §3.6.6).

    ``plan_overrides``: {module_fqn: ModuleShardingSpec} — hand-written user
    specs, which wholesale replace/insert entries before the Phase 5 chain
    propagation (05 §3.6.7). Override specs still participate in adjacent
    contract validation and terminal marking, which is safer than patching
    after plan() returns.
    """

    def __init__(
        self,
        plan_overrides: Optional[Dict[str, ModuleShardingSpec]] = None,
    ):
        self._classifier = ParameterClassifier(arch_overrides=ARCH_OVERRIDES)
        self._templates = TEMPLATES
        self._special_handler_patterns = dict(_SPECIAL_HANDLER_PATTERNS)
        self._plan_overrides = dict(plan_overrides or {})

    # ── Main entry point ────────────────────────────────────────────────

    def plan(
        self,
        model,
        mesh,
        *,
        tp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        sequence_parallel: bool = True,
        loss_parallel: bool = False,
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

        Returns:
            :class:`ShardingPlan`: module FQN → :class:`ModuleShardingSpec`.

        Raises:
            ValueError: If *tp_size* / *cp_size* / *ep_size* do not match the
                corresponding mesh dimensions (fail-first), or if model-level
                constraints are violated (head divisibility, expert count, etc.).
        """
        arch = self._get_architecture(model)
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
        param_roles = self._classify_all_params(model, arch)

        # Phase 2: communication boundary grouping
        boundary_groups = self._group_by_boundary(param_roles)

        # Phase 3+4: semantic inference + template-fills I/O
        param_ndims = {name: p.ndim for name, p in model.named_parameters()}
        plan = ShardingPlan(
            mesh_dim_names=mesh_dim_names,
            sequence_parallel=sequence_parallel,
            loss_parallel=loss_parallel,
        )
        inferred_templates: Dict[str, ShardingTemplate] = {}
        for boundary_fqn, group in boundary_groups.items():
            boundary_type = self._infer_boundary_type(boundary_fqn, group)
            template = self._templates.get(boundary_type)
            if template is None:
                logger.warning(
                    "No template for boundary_type=%s at %s", boundary_type, boundary_fqn
                )
                continue
            spec = self._build_spec_from_template(
                boundary_fqn, group, template,
                sequence_parallel, loss_parallel, mesh_dim_names,
                param_ndims=param_ndims,
            )
            if spec is not None:
                if boundary_type == "moe_mlp":
                    self._mark_hf_native_moe(
                        spec, group, boundary_fqn, template, mesh_dim_names, arch,
                        ep_extend=ep_extend, mesh=mesh, model=model,
                        param_ndims=param_ndims)
                plan.modules[boundary_fqn] = spec
                inferred_templates[boundary_fqn] = template

        # Phase 4.5: merge user plan_overrides (05 §3.6.7; must run before
        # the D-14 checks — nested overrides are legal since D-14, subject to
        # the param-uniqueness invariant)
        self._merge_plan_overrides(plan, model, inferred_templates)

        # D-14 invariants (05 §13.2/§13.3): full self-declaration + param
        # uniqueness (the only nesting check that remains)
        self._check_full_declaration(plan)
        self._check_param_uniqueness(plan)

        # Phase 5: _is_terminal marking (D-14: chain propagation removed)
        plan = self._mark_terminal(plan, model)

        # Phase 6: special parameter handling
        plan.special_handlers = self._collect_special_handlers(param_roles)

        # tied-weight detection (embed <-> lm_head sharing storage)
        plan.tied_pairs = self._detect_tied_pairs(model)

        return plan

    # ── Architecture detection ──────────────────────────────────────────

    def _get_architecture(self, model) -> str:
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

    # ── Phase 1 ─────────────────────────────────────────────────────────

    def _classify_all_params(self, model, arch: str) -> Dict[str, ParamRole]:
        return self._classifier.classify(model, arch)

    # ── Phase 2 ─────────────────────────────────────────────────────────

    def _group_by_boundary(
        self, param_roles: Dict[str, ParamRole],
    ) -> Dict[str, List[Tuple[str, ParamRole]]]:
        """Two-pass grouping (fixes the single-parameter group flaw in the
        05 §3.6.6 pseudocode):

        Pass 1: group by owning module FQN (strip the leaf parameter name).
        Pass 2: depth-first work queue — when the group's roles are
                complete, run boundary inference; on unknown, merge the
                whole group's parameters upward into the parent module and
                enqueue it (the parent is shallower and is therefore
                processed later; sibling modules' parameters are merged
                completely before inference, avoiding q_proj being
                misjudged on its own). If still unknown after backtracking
                to the root, attribute the group to the parameter's own
                module (no template will match later → warning and skip).
        """
        # Pass 1
        own: Dict[str, List[Tuple[str, ParamRole]]] = {}
        for fqn, role in param_roles.items():
            module_fqn = ".".join(fqn.split(".")[:-1])
            own.setdefault(module_fqn, []).append((fqn, role))

        # Pass 2
        merged: Dict[str, List[Tuple[str, ParamRole]]] = {
            mfqn: list(params) for mfqn, params in own.items()
        }
        pending = sorted(merged.keys(), key=lambda f: f.count("."), reverse=True)
        consumed: set = set()
        groups: Dict[str, List[Tuple[str, ParamRole]]] = {}
        i = 0
        while i < len(pending):
            mfqn = pending[i]
            i += 1
            if mfqn in consumed:
                continue
            params = merged.get(mfqn, [])
            if self._infer_boundary_type(mfqn, params) != "unknown":
                groups[mfqn] = params
            else:
                parent = mfqn.rsplit(".", 1)[0] if "." in mfqn else ""
                if parent:
                    if parent not in merged:
                        merged[parent] = []
                        pending.append(parent)  # parent is shallower; tail-enqueue suffices
                    merged[parent].extend(params)
                else:
                    # Still unknown after backtracking to the root:
                    # attribute to the parameter's own module (no template
                    # will match later → skipped)
                    origin = ".".join(params[0][0].split(".")[:-1]) if params else mfqn
                    groups.setdefault(origin, params)
            consumed.add(mfqn)
        return groups

    # ── Phase 3 ─────────────────────────────────────────────────────────

    def _infer_boundary_type(self, fqn: str, group: List[Tuple[str, ParamRole]]) -> str:
        """Identify the semantic role from the module FQN + the group's
        parameter roles.

        Priority: explicit FQN patterns > leaf-segment guard > MoE roles >
        parameter role combinations > default.
        """
        fqn_lower = fqn.lower()
        seg = _last_segment(fqn)

        # 1. Explicit rules (highest priority; the leaf module itself is the boundary)
        if _match_any(fqn_lower, ["embed_tokens", "wte", ".embed.", "tok_embeddings",
                                  "embed_in", "word_embeddings"]):
            return "embed"
        if _match_any(fqn_lower, ["lm_head", "embed_out", "output_layer"]):
            return "lm_head"
        if _match_any(fqn_lower, ["norm", "layernorm", "rmsnorm", "ln_"]):
            return "norm"
        if _match_any(seg, ["router"]):
            return "moe_gate"

        # 2. Leaf-segment guard: projection/expert leaf modules are not
        # boundary containers themselves
        if seg in _LEAF_SEGMENT_GUARD:
            return "unknown"
        # Numeric-segment guard: HF per-expert containers (experts.0..N) are
        # not boundaries; parameters must aggregate upward into the moe
        # container (D-09, 05 §6.4.7)
        if seg.isdigit():
            return "unknown"

        # 3. MoE roles: groups containing MOE_* roles aggregate upward into
        # the moe container boundary
        roles = {r for _, r in group}
        moe_roles = {ParamRole.MOE_EXPERT, ParamRole.SHARED_EXPERT, ParamRole.MOE_GATE}
        if roles & moe_roles:
            if _match_any(fqn_lower, list(_MOE_CONTAINER_PATTERNS)):
                return "moe_mlp"
            return "unknown"

        # 4. Parameter role combinations
        has_colwise = any(r in (ParamRole.COLWISE, ParamRole.FUSED_QKV,
                                ParamRole.FUSED_GATE_UP) for _, r in group)
        has_rowwise = any(r == ParamRole.ROWWISE for _, r in group)
        if has_colwise and has_rowwise:
            if _match_any(fqn_lower, list(_ATTN_PATTERNS)):
                return "attention"
            if _match_any(fqn_lower, list(_MLP_PATTERNS)):
                return "mlp"
            return "attention"  # default to attention (more conservative SP communication)
        if has_colwise and not has_rowwise:
            if _match_any(fqn_lower, list(_MLP_PATTERNS)):
                return "mlp"
            return "unknown"

        return "unknown"

    # ── Phase 4 ─────────────────────────────────────────────────────────

    def _build_spec_from_template(
        self, boundary_fqn: str, group: List[Tuple[str, ParamRole]],
        template: ShardingTemplate, sequence_parallel: bool, loss_parallel: bool,
        mesh_dim_names: Tuple[str, ...], param_ndims: Optional[Dict[str, int]] = None,
    ) -> Optional[ModuleShardingSpec]:
        """Template + ParamRole → ModuleShardingSpec (05 §3.5 Template Mapping)."""
        has_tp = "tp" in mesh_dim_names
        has_ep = "ep" in mesh_dim_names
        spec = ModuleShardingSpec()

        # Step 1: fill spec.params per ParamRole
        for param_fqn, role in group:
            param_path = param_fqn[len(boundary_fqn) + 1:]
            ndim = (param_ndims or {}).get(param_fqn, 2)
            placement = self._placement_for_role(param_path, role, template,
                                                 has_tp, has_ep, ndim=ndim)
            if placement is not None:
                spec.params[param_path] = placement

        # Step 2: select the I/O contract per the SP switch (deep copy, so
        # chain propagation cannot dirty the shared templates)
        if sequence_parallel:
            spec.in_src = copy.deepcopy(template.sp_in_src)
            spec.in_dst = copy.deepcopy(template.sp_in_dst)
            spec.out_src = copy.deepcopy(template.sp_out_src)
            spec.out_dst = copy.deepcopy(template.sp_out_dst)
        else:
            spec.in_src = copy.deepcopy(template.nosp_in_src)
            spec.in_dst = copy.deepcopy(template.nosp_in_dst)
            spec.out_src = copy.deepcopy(template.nosp_out_src)
            spec.out_dst = copy.deepcopy(template.nosp_out_dst)

        # Step 2.5: lm_head's out_dst depends on loss_parallel (a runtime
        # decision).
        # The CP dim is always Shard(1) (D-07/R8): under CP the loss is
        # computed on the local chunk; no gather is performed.
        if template is self._templates.get("lm_head"):
            spec.out_dst = _multi_dim(
                tp=Shard(-1) if loss_parallel else Replicate(),
                cp=Shard(1), ep=Replicate(),
            )

        # Step 2.6: embed's CP contract (revision D-05): the CP data
        # pipeline (shard_batch_for_cp, 05 §6.3.4) has already sharded
        # input_ids along CP — the CP dim of in/out is Shard(1) rather than
        # the template's default Replicate, otherwise the boundary would
        # scatter the already-sharded chunk a second time (the sequence
        # would be sharded twice).
        has_cp = "cp" in mesh_dim_names
        if template is self._templates.get("embed") and has_cp and sequence_parallel:
            spec.in_src = {"input": _multi_dim(tp=Replicate(), cp=Shard(1),
                                               ep=Replicate())}
            spec.in_dst = {"input": _multi_dim(tp=Replicate(), cp=Shard(1),
                                               ep=Replicate())}
            spec.out_src = _multi_dim(tp=Partial(), cp=Shard(1), ep=Replicate())

        # Step 3: special flags
        spec.use_local_map = template.use_local_map
        if template.needs_cp_attn:
            spec._needs_cp_attn = True

        # Step 4: normalize out_src/out_dst scalar shorthand
        return _normalize_out_fields(spec)

    @staticmethod
    def _placement_for_role(
        param_path: str, role: ParamRole, template: ShardingTemplate,
        has_tp: bool, has_ep: bool, ndim: int = 2,
    ) -> Optional[NamedPlacement]:
        """13 roles → placement mapping (05 §3.5 mapping table + D-08
        ndim-aware)."""
        if role in (ParamRole.COLWISE, ParamRole.EMBED, ParamRole.LM_HEAD,
                    ParamRole.FUSED_QKV, ParamRole.FUSED_GATE_UP):
            return _multi_dim(tp=template.colwise_placement if has_tp else None,
                              cp=Replicate(), ep=Replicate())
        if role == ParamRole.ROWWISE:
            return _multi_dim(tp=template.rowwise_placement if has_tp else None,
                              cp=Replicate(), ep=Replicate())
        if role in (ParamRole.NORM, ParamRole.MOE_GATE):
            return _multi_dim(tp=template.norm_placement if has_tp else None,
                              cp=Replicate(), ep=Replicate())
        if role == ParamRole.MOE_EXPERT:
            # 05 §3.5 NOTE: when has_tp=False, use an explicit Replicate
            # (rather than omitting the TP key).
            # D-08: the TP dim of a 3D expert weight [E, H_out, H_in] is
            # shifted according to ndim.
            tp_p = (_moe_expert_tp_placement(param_path, ndim, template)
                    if has_tp else Replicate())
            return _multi_dim(tp=tp_p, cp=Replicate(),
                              ep=template.moe_expert_placement if has_ep else None)
        if role == ParamRole.SHARED_EXPERT:
            # replicated along the EP dim; TP per w1/w3(colwise)/w2(rowwise)
            tp_p = _infer_colwise_vs_rowwise(param_path, template)
            return _multi_dim(tp=tp_p if has_tp else None,
                              cp=Replicate(), ep=Replicate())
        if role == ParamRole.BIAS:
            return _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())
        if role == ParamRole.REPLICATED:
            # MLA down-projections etc. (explicitly assigned via
            # ARCH_OVERRIDES): replicate on all dims.
            # The output latent is identical within the TP group, so the
            # input contract of the downstream q_b/kv_b (COLWISE), sharded
            # along the head dim, matches standard attention.
            return _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())
        # SPECIAL → Phase 6; SKIP → not sharded
        return None

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
        if not ep_extend:
            return
        expert_params = [fqn for fqn, r in group if r == ParamRole.MOE_EXPERT]
        if not expert_params:
            return

        has_ep_in_mesh = "ep" in mesh_dim_names

        # ── Detect expert layout (only affects stacking strategy) ──
        stacks: Dict[str, List[Tuple[int, str]]] = {}
        batched: List[str] = []
        for param_fqn in expert_params:
            rel = param_fqn[len(boundary_fqn) + 1:]
            if "bias" in rel.lower():
                logger.warning(
                    "%s: MoE expert has bias (%s); not supported in v1, "
                    "skipping EP marking",
                    boundary_fqn, rel,
                )
                return
            m = self._PER_EXPERT_RE.match(rel)
            if m is not None:
                stacks.setdefault(m.group(2), []).append((int(m.group(1)), rel))
                continue
            if (self._BATCHED_EXPERT_RE.match(rel) is not None
                    and (param_ndims or {}).get(param_fqn, 2) >= 3):
                batched.append(rel)
                continue
            # Custom naming (w1/w2/w3 etc.) — pre-stacked 3D, no-op for
            # both modes; D-10 will mark EP on them below.

        if stacks and batched:
            logger.warning(
                "%s: mixed per-expert and batched layouts (%s ...); "
                "skipping EP marking",
                boundary_fqn, batched[0],
            )
            return

        # ────────────────────────────────────────────────────────────────
        # Old-style EP: mesh has explicit "ep" axis
        # ────────────────────────────────────────────────────────────────
        if has_ep_in_mesh:
            if not stacks:
                # Batched / custom layouts: already 3D, placements from
                # _build_spec_from_template ({TP: Shard(…), EP: Shard(0)})
                # are correct. Nothing to do.
                return
            # Per-expert layout: stack first so the expert dim exists for
            # EP Shard(0).  Keep both TP and EP keys — old-style EP shards
            # on both axes simultaneously on the main mesh.
            for proj, items in stacks.items():
                items.sort()
                sources = [rel for _, rel in items]
                stacked = f"experts.{proj}"
                # Compute the correct TP placement for the 3D stacked
                # tensor (ndim=3 shifts TP axes: colwise Shard(1),
                # rowwise Shard(2))
                tp_p = _moe_expert_tp_placement(stacked, ndim=3, template=template)
                for rel in sources:
                    spec.params.pop(rel, None)
                spec.params[stacked] = _multi_dim(
                    tp=tp_p, cp=Replicate(),
                    ep=template.moe_expert_placement)
                spec._ep_stack[stacked] = sources
            # No _ep_size — old-style EP uses the main mesh's "ep" axis
            return

        # ────────────────────────────────────────────────────────────────
        # D-10 TP-extend-EP: mesh has NO "ep" axis
        # ────────────────────────────────────────────────────────────────
        self._validate_ep_extend(ep_extend, mesh, model)

        if stacks:
            # Per-expert layout: stack + {EP: Shard(0)} (no TP key)
            for proj, items in stacks.items():
                items.sort()
                sources = [rel for _, rel in items]
                stacked = f"experts.{proj}"
                for rel in sources:
                    spec.params.pop(rel, None)
                spec.params[stacked] = _multi_dim(
                    tp=None, cp=Replicate(), ep=template.moe_expert_placement)
                spec._ep_stack[stacked] = sources
        elif batched:
            # Batched layout: already 3D, just mark {EP: Shard(0)}
            for rel in batched:
                spec.params[rel] = _multi_dim(
                    tp=None, cp=Replicate(), ep=template.moe_expert_placement)
        else:
            # Custom naming (w1/w2/w3): pre-stacked 3D, mark {EP: Shard(0)}
            for param_fqn in expert_params:
                rel = param_fqn[len(boundary_fqn) + 1:]
                spec.params[rel] = _multi_dim(
                    tp=None, cp=Replicate(), ep=template.moe_expert_placement)

        spec._moe_router = arch if arch in MOE_ROUTER_ADAPTERS else "default"

        # D-10: change the MoE boundary contract to SP-in identity
        # (Megatron MoE never gathers anyway; all communication is
        # cohesive inside the region, 05 §6.4.8). The layout follows the
        # template in_src (SP → TP Shard(1); non-SP → Replicate); MoE is
        # per-token computation, so the output layout always equals the
        # input layout.
        identity = copy.deepcopy(spec.in_src)
        spec.in_dst = copy.deepcopy(identity)
        out_layout = copy.deepcopy(next(iter(identity.values())))
        spec.out_src = {"output": copy.deepcopy(out_layout)}
        spec.out_dst = {"output": copy.deepcopy(out_layout)}
        spec._ep_size = ep_extend

    # ── Phase 4.5: user spec overrides (05 §3.6.7) ──────────────────────

    def _merge_plan_overrides(
        self, plan: ShardingPlan, model,
        inferred_templates: Dict[str, ShardingTemplate],
    ) -> None:
        """Merge hand-written user specs (plan_overrides), executed before
        Phase 5.

        Semantics:
        - fqn already matches a planner-generated spec → wholesale
          replacement (the user spec is authoritative);
        - fqn not matched (planner missed it / no template / module
          without parameters) → insertion;
        - the structural flags ``use_local_map`` / ``_needs_cp_attn`` are
          backfilled from the inferred template (they are module structural
          properties, not I/O contracts: a missing MoE all-to-all or CP
          K/V all-gather causes numerical errors, so they are force-set
          whenever template inference yields True; the user spec neither
          needs to nor should be responsible for them);
        - the CP customization entries ``inner_target`` / ``inner_wrapper``
          are user fields (preserved via deep copy) and no flag is
          rewritten — inner-wrap gating is derived by the applier's
          ``_resolve_inner_wrapper`` resolution chain (05 §4.4.2);
        - ``local_compute_fn`` is a user field (preserved via deep copy)
          and no flag is rewritten — local-region gating is derived by the
          applier's ``_resolve_local_compute_fn`` resolution chain
          (05 §4.4.3);
        - ``out_src``/``out_dst`` scalar shorthand is normalized here;
        - ``_is_terminal`` is uniformly marked by Phase 5; any user-preset
          value is overwritten;
        - the user spec is deep-copied — plan() can be called repeatedly
          and chain propagation mutates in_src in place, so the caller's
          held object must not be polluted.

        Nesting (ancestor/descendant FQNs) is **allowed** since D-14 (05
        §13): the outer boundary may declare I/O contracts and params of its
        own/intermediate modules, inner boundaries keep theirs. The only
        remaining nesting check is the param-uniqueness invariant
        (``_check_param_uniqueness``, fail-fast on double sharding); each
        module's declaration is fully self-contained (chain fill removed),
        and correctness is covered by validate-mode per-module propagation
        assertions + dual-mode numerical equivalence.
        """
        if not self._plan_overrides:
            return
        module_names = {name for name, _ in model.named_modules()}
        for fqn, user_spec in self._plan_overrides.items():
            if not isinstance(user_spec, ModuleShardingSpec):
                raise TypeError(
                    f"plan_overrides[{fqn!r}] must be a ModuleShardingSpec, "
                    f"got {type(user_spec).__name__}"
                )
            if fqn not in module_names:
                raise ValueError(
                    f"plan_overrides FQN not found in the model's "
                    f"named_modules: {fqn!r} (check spelling; in PP "
                    f"scenarios plan each single-part model separately)"
                )
            spec = copy.deepcopy(user_spec)
            template = inferred_templates.get(fqn)
            if template is not None:
                if template.use_local_map:
                    # force-set when the template is True (guards against
                    # numerical errors); modules the user explicitly set to
                    # True (in-house data-dependent modules) are unaffected
                    # by the template and are naturally preserved
                    spec.use_local_map = True
                if template.needs_cp_attn:
                    spec._needs_cp_attn = True
            # inner_target/inner_wrapper/local_compute_fn need no flag
            # set: inner-wrap and local-region gating are derived by the
            # applier's resolution chains (05 §4.4.2/§4.4.3)
            _normalize_out_fields(spec)
            action = "replace" if fqn in plan.modules else "insert"
            logger.info("plan_overrides: %s the spec of module %s", action, fqn)
            plan.modules[fqn] = spec

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
            spec._is_terminal = fqn == terminal
        return plan

    def _topological_sort_by_forward_order(self, fqns: List[str], model) -> List[str]:
        """Sort by named_modules registration order; unmatched FQNs are
        appended at the end with a warning."""
        fqn_set = set(fqns)
        ordered: List[str] = []
        seen: set = set()
        for name, _module in model.named_modules():
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

    # ── Phase 6 ─────────────────────────────────────────────────────────

    def _collect_special_handlers(
        self, param_roles: Dict[str, ParamRole],
    ) -> Dict[str, str]:
        """SPECIAL-role parameters → handler name (unregistered patterns
        fall back to "default")."""
        result: Dict[str, str] = {}
        for fqn, role in param_roles.items():
            if role != ParamRole.SPECIAL:
                continue
            handler_name = "default"
            for pattern, hname in self._special_handler_patterns.items():
                if _match_any(fqn.lower(), [pattern.lower()]):
                    handler_name = hname
                    break
            result[fqn] = handler_name
        return result

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
    model, *, tp_size: int = 1, cp_size: int = 1, ep_size: int = 1,
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
