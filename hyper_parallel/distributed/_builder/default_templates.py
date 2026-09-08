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
"""default_templates: semantic role → placement templates (05 §3.5 canonical).

``ShardingTemplate`` is the public template type (re-exported from
``hyper_parallel.distributed``); ``TEMPLATES`` is the private
default table consumed by the planner. The template-referencing Phase 4
derivation helpers (``_build_spec_from_template`` / ``_placement_for_role``
and friends) also live here — they are pure functions of the templates.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import (
    CP,
    EP,
    TP,
    ModuleShardingSpec,
    NamedPlacement,
    _normalize_out_fields,
)
from hyper_parallel.distributed.tensor_parallel.head_count import (
    build_tp_local_attr_plan,
)
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole


@dataclass
class ShardingTemplate:
    """Semantic role → placement template (05 §3.5).

    Each I/O field declares placements for all active mesh dimensions
    (TP+CP+EP); the ShardingPlanner filters out dimensions not enabled by the
    actual mesh_dim_names (resolve_placements looks up keys by mesh_dim_names,
    so extra keys are naturally dropped).

    Note: sp_out_src / nosp_out_src etc. are scalar NamedPlacement shorthands
    (single-output modules); they are wrapped into {"output": ...} during
    normalization in _build_spec_from_template.
    """
    # Parameter sharding rules
    colwise_placement: Placement = field(default_factory=lambda: Shard(0))
    rowwise_placement: Placement = field(default_factory=lambda: Shard(1))
    norm_placement: Placement = field(default_factory=Replicate)
    moe_expert_placement: Placement = field(default_factory=lambda: Shard(0))

    # SP-mode I/O (full TP+CP+EP, three dims)
    sp_in_src: Dict[str, NamedPlacement] = field(default_factory=dict)
    sp_in_dst: Dict[str, NamedPlacement] = field(default_factory=dict)
    sp_out_src: Optional[NamedPlacement] = None
    sp_out_dst: Optional[NamedPlacement] = None

    # non-SP-mode I/O
    nosp_in_src: Dict[str, NamedPlacement] = field(default_factory=dict)
    nosp_in_dst: Dict[str, NamedPlacement] = field(default_factory=dict)
    nosp_out_src: Optional[NamedPlacement] = None
    nosp_out_dst: Optional[NamedPlacement] = None

    # Special flags
    # region_dispatch: template-level declaration that the matched module's own
    # forward CANNOT dispatch (data-dependent logic inside, e.g. an EP-aware
    # custom MoE's a2a) — the planner copies False into the spec (derived
    # metadata, logged); None = ordinary dispatchable module.
    region_dispatch: Optional[bool] = None
    needs_cp_attn: bool = False   # CP: inner attention needs a CP-aware forward


def _multi_dim(tp=None, cp=None, ep=None) -> NamedPlacement:
    """Build multi-dim placement dict, filtering out None dims."""
    result = {}
    if tp is not None:
        result[TP] = tp
    if cp is not None:
        result[CP] = cp
    if ep is not None:
        result[EP] = ep
    return result


def _hid(tp_p, cp_p, ep_p=None) -> Dict[str, NamedPlacement]:
    """Shorthand for the single-input hidden_states contract."""
    return {"hidden_states": _multi_dim(tp=tp_p, cp=cp_p, ep=ep_p or Replicate())}


def _out(tp_p, cp_p, ep_p=None) -> NamedPlacement:
    """Shorthand for the single-output (scalar shorthand) contract."""
    return _multi_dim(tp=tp_p, cp=cp_p, ep=ep_p or Replicate())


# ── TEMPLATES: complete templates for the 7 semantic roles (05 §3.5, declared over TP+CP+EP) ──
# CP-dim rule: parameters are always Replicate (CP does not shard parameters);
# input sequence activations use Shard(1) independently of TP sequence_parallel.
# EP-dim rule: non-MoE modules Replicate; MoE experts Shard(0).
TEMPLATES: Dict[str, ShardingTemplate] = {
    # ── Attention (q/k/v Colwise + o Rowwise) ──
    # The CP dim keeps Shard(1) in in_dst: the K/V all-gather is done by the
    # inner attention wrapper inside SDPA/FlexAttention (needs_cp_attn=True),
    # not at the boundary layer.
    "attention": ShardingTemplate(
        colwise_placement=Shard(0),          # q/k/v: [H/tp, H]
        rowwise_placement=Shard(1),          # o: [H, H/tp]
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Shard(1)),
        sp_out_src=_out(Partial(), Shard(1)),     # local Q-chunk output → CP Shard(1)
        sp_out_dst=_out(Shard(1), Shard(1)),
        nosp_in_src=_hid(Replicate(), Shard(1)),
        nosp_in_dst=_hid(Replicate(), Shard(1)),
        nosp_out_src=_out(Partial(), Shard(1)),
        nosp_out_dst=_out(Replicate(), Shard(1)),
        needs_cp_attn=True,
    ),

    # ── MLP (gate/up Colwise + down Rowwise) ──
    # The CP dim stays Shard(1) throughout (revision D-06): MLP is pointwise and
    # CP needs no communication; if in_dst had CP=Replicate, the full-sequence
    # reduce-scatter under TP×CP would produce a tp-major sequence layout
    # inconsistent with embed/attention (cp-major).
    "mlp": ShardingTemplate(
        colwise_placement=Shard(0),
        rowwise_placement=Shard(1),
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Shard(1)),
        sp_out_src=_out(Partial(), Shard(1)),
        sp_out_dst=_out(Shard(1), Shard(1)),
        nosp_in_src=_hid(Replicate(), Shard(1)),
        nosp_in_dst=_hid(Replicate(), Shard(1)),
        nosp_out_src=_out(Partial(), Shard(1)),
        nosp_out_dst=_out(Replicate(), Shard(1)),
    ),

    # ── Norm (RMSNorm/LayerNorm: weight replicated, zero communication) ──
    "norm": ShardingTemplate(
        norm_placement=Replicate(),
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Shard(1), Shard(1)),      # identity
        sp_out_src=_out(Shard(1), Shard(1)),
        sp_out_dst=_out(Shard(1), Shard(1)),     # identity
        nosp_in_src=_hid(Replicate(), Shard(1)),
        nosp_in_dst=_hid(Replicate(), Shard(1)),
        nosp_out_src=_out(Replicate(), Shard(1)),
        nosp_out_dst=_out(Replicate(), Shard(1)),
    ),

    # ── Embedding (weight Shard(0) along the vocab dim, output Partial → SP+CP) ──
    "embed": ShardingTemplate(
        colwise_placement=Shard(0),          # weight: [V/tp, H]
        sp_in_src={"input": _multi_dim(tp=Replicate(), cp=Shard(1), ep=Replicate())},
        sp_in_dst={"input": _multi_dim(tp=Replicate(), cp=Shard(1), ep=Replicate())},
        sp_out_src=_out(Partial(), Shard(1)),
        sp_out_dst=_out(Shard(1), Shard(1)),     # reduce-scatter → SP+CP
        nosp_in_src={"input": _multi_dim(tp=Replicate(), cp=Shard(1), ep=Replicate())},
        nosp_in_dst={"input": _multi_dim(tp=Replicate(), cp=Shard(1), ep=Replicate())},
        nosp_out_src=_out(Partial(), Shard(1)),
        nosp_out_dst=_out(Replicate(), Shard(1)),
    ),

    # ── LM Head (weight Shard(0), output Shard(-1); out_dst is overridden according to loss_parallel) ──
    # The CP dim stays Shard(1) throughout (revision D-07): R8 — the CP dim is
    # always identity at the boundary layer (the CP sequence all-gather happens
    # only inside attention for K/V). lm_head computes logits/loss on the local
    # CP chunk (the standard Megatron CP practice), with no CP gather.
    "lm_head": ShardingTemplate(
        colwise_placement=Shard(0),          # weight: [V/tp, H]
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Shard(1)),
        sp_out_src=_out(Shard(-1), Shard(1)),
        sp_out_dst=_out(Shard(-1), Shard(1)),   # loss_parallel=true default;
        # when loss_parallel=false, _build_spec_from_template overrides it to
        # {TP: Replicate, CP: Shard(1)}
        nosp_in_src=_hid(Replicate(), Shard(1)),
        nosp_in_dst=_hid(Replicate(), Shard(1)),
        nosp_out_src=_out(Shard(-1), Shard(1)),
        nosp_out_dst=_out(Replicate(), Shard(1)),
    ),

    # ── MoE Gate (Router: weight replicated, output redistributes → EP) ──
    "moe_gate": ShardingTemplate(
        norm_placement=Replicate(),          # router weight/bias: replicated
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Replicate()),
        sp_out_src=_out(Replicate(), Replicate()),
        sp_out_dst=_out(Replicate(), Replicate(), Shard(0)),
        nosp_in_src=_hid(Replicate(), Shard(1)),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Replicate(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate(), Shard(0)),
    ),

    # ── MoE MLP (gate + routed experts + optional shared experts) ──
    # CP dim same as mlp (D-06): pointwise per-token, CP stays Shard(1) throughout.
    "moe_mlp": ShardingTemplate(
        colwise_placement=Shard(0),          # expert w1/w3: Colwise on TP
        rowwise_placement=Shard(1),          # expert w2: Rowwise on TP
        norm_placement=Replicate(),          # gate/norm: replicated
        moe_expert_placement=Shard(0),       # expert params: Shard(0) on EP
        sp_in_src={"x_BLD": _multi_dim(tp=Shard(1), cp=Shard(1), ep=Replicate())},
        sp_in_dst={"x_BLD": _multi_dim(tp=Replicate(), cp=Shard(1), ep=Replicate())},
        sp_out_src=_out(Partial(), Shard(1)),
        sp_out_dst=_out(Shard(1), Shard(1)),
        nosp_in_src={"x_BLD": _multi_dim(tp=Replicate(), cp=Shard(1), ep=Replicate())},
        nosp_in_dst={"x_BLD": _multi_dim(tp=Replicate(), cp=Shard(1), ep=Replicate())},
        nosp_out_src=_out(Partial(), Shard(1)),
        nosp_out_dst=_out(Replicate(), Shard(1)),
        region_dispatch=False,           # MoE forward has its own a2a; dispatch not allowed
    ),
}


# ────────────────────────────────────────────────────────────────────────────
# Template → spec derivation (planner Phase 4 machinery, 05 §3.5)
# ────────────────────────────────────────────────────────────────────────────

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
    subset of experts" (module level), requiring family sharding_rules /
    SpecialHandler; it is outside template coverage.
    """
    name = param_path.lower()
    is_rowwise = any(k in name for k in ("w2", "down_proj", "down."))
    if ndim >= 3:
        return Shard(2) if is_rowwise else Shard(1)
    return template.rowwise_placement if is_rowwise else template.colwise_placement


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
        # Generic bias whose owning Linear role could not be inferred.
        # Named colwise/rowwise Linear biases are classified with their
        # weight role before reaching this fallback.
        return _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())
    if role == ParamRole.REPLICATED:
        # MLA down-projections etc. (explicitly assigned via
        # family sharding_rules): replicate on all dims.
        # The output latent is identical within the TP group, so the
        # input contract of the downstream q_b/kv_b (COLWISE), sharded
        # along the head dim, matches standard attention.
        return _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())
    # SPECIAL → Phase 6; SKIP → not sharded
    return None


def _build_spec_from_template(
    templates, boundary_fqn: str, group: List[Tuple[str, ParamRole]],
    template: ShardingTemplate, sequence_parallel: bool, loss_parallel: bool,
    mesh_dim_names: Tuple[str, ...], param_ndims: Optional[Dict[str, int]] = None,
) -> Optional[ModuleShardingSpec]:
    """Template + ParamRole → ModuleShardingSpec (05 §3.5 Template Mapping)."""
    has_tp = "tp" in mesh_dim_names
    has_cp = "cp" in mesh_dim_names
    has_ep = "ep" in mesh_dim_names
    # The derived spec is materialized directly with concrete dicts (None
    # is the "undeclared" semantics on the override input side; derived
    # artifacts always hold concrete values).
    spec = ModuleShardingSpec(params={})

    # Step 1: fill spec.params per ParamRole
    for param_fqn, role in group:
        param_path = param_fqn[len(boundary_fqn) + 1:]
        ndim = (param_ndims or {}).get(param_fqn, 2)
        placement = _placement_for_role(param_path, role, template,
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
    if template is templates.get("lm_head"):
        cp_placement = Shard(1) if has_cp else Replicate()
        for input_contract in (spec.in_src, spec.in_dst):
            for placements in input_contract.values():
                placements[CP] = cp_placement
        spec.out_src = _multi_dim(
            tp=Shard(-1),
            cp=cp_placement,
            ep=Replicate(),
        )
        spec.out_dst = _multi_dim(
            tp=Shard(-1) if loss_parallel else Replicate(),
            cp=cp_placement,
            ep=Replicate(),
        )

    # Step 2.6: embed's CP contract (revision D-05): the CP data
    # pipeline (shard_batch_for_cp, 05 §6.3.4) has already sharded
    # input_ids along CP — the CP dim of in/out is Shard(1) rather than
    # the template's default Replicate, otherwise the boundary would
    # scatter the already-sharded chunk a second time (the sequence
    # would be sharded twice).
    if template is templates.get("embed") and has_cp:
        spec.in_src = {"input": _multi_dim(tp=Replicate(), cp=Shard(1),
                                           ep=Replicate())}
        spec.in_dst = {"input": _multi_dim(tp=Replicate(), cp=Shard(1),
                                           ep=Replicate())}
        spec.out_src = _multi_dim(tp=Partial(), cp=Shard(1), ep=Replicate())

    # Step 3: special flags
    spec.region_dispatch = template.region_dispatch
    if template.needs_cp_attn:
        spec._needs_cp_attn = True  # pylint: disable=protected-access  # planner owns the spec DSL internals

    # Step 4: normalize out_src/out_dst scalar shorthand
    return _normalize_out_fields(spec)


def _finalize_tp_local_attr_plans(
    plan: ShardingPlan, model, *, tp_size: int,
    mesh_dim_names: Tuple[str, ...],
) -> None:
    """Build internal auto/user TP-local attribute plans after overrides."""
    modules = dict(model.named_modules())
    for module_fqn, spec in plan.modules.items():
        module = modules.get(module_fqn)
        if module is None:
            raise ValueError(
                f"Cannot finalize TP-local attributes: module "
                f"{module_fqn!r} is not present in model.named_modules()")
        spec._tp_local_attr_plan = build_tp_local_attr_plan(  # pylint: disable=protected-access  # planner owns the spec DSL internals
            module, spec, module_fqn, tp_size, mesh_dim_names,
        )
