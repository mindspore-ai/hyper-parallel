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
"""model_structure: per-model structural facts consumed by planner Phases 1-4.

Some architectures expose module structures that the generic naming rules
cannot classify: DSA attention leaves (head-sharded index/query projections,
sink parameters, a directly-consumed ``linear_kvb``), MHC pre-modules, MTP
previous-state projections, and shared experts owned by a TP-extended MoE
dispatcher.  Those families are matched from *module capabilities* —
deliberately not from ``config.architectures`` / ``config.model_type`` — and
their knowledge is declared as ordinary Phase 1/2/3/4 rules instead of a
second derivation pass over the model:

- Phase 1 (parameter role classification): ``role_rules()`` returns
  ``fn(param_fqn) -> Optional[ParamRole]`` predicates consulted after the
  family's ``ModelAdapterSpec.sharding_rules`` and before the default naming
  rules.  Families reuse the existing :class:`ParamRole` values, so the
  Phase 4 role→placement mapping keeps working unchanged.  Parameter names
  reach these predicates lower-cased (the classifier normalizes them), so a
  family's ``boundary_types`` map is keyed by lower-cased module FQNs.
- Phase 2 (boundary grouping): ``paramless_boundary_fqns`` seeds the
  boundaries that own no parameter at all (DSA ``rotary_emb`` / the sparse
  indexer), which can never arise from the parameter tree.
- Phase 3 (semantic role inference): ``boundary_type()`` returns the
  boundary-type key for an FQN, resolved against the template tables in
  Phase 4.  All discrimination — including choices that depend on a parent
  module (the reduce-scatter axis of a DSA output projection) — happens at
  discovery, so Phase 3 is a pure lookup.
- Phase 4 (template lookup): :data:`STRUCTURAL_TEMPLATES` maps every
  structural boundary type to its template.  It is kept separate from
  ``TEMPLATES`` (the complete table of the 7 semantic roles), because a
  structural entry may be param-only (``is_boundary=False``, no contract at
  all) or declare only the mesh axes its architecture actually uses.

Discovery runs once per ``plan()`` call (:func:`detect_model_structure`) and
the result is frozen; all later queries are pure lookups.  A planner without
a detected structure (``EMPTY_STRUCTURE``) behaves exactly as if this module
did not exist, which is what direct unit calls to the planner's Phase 1-3
helpers get.  Families are consulted most-specific-first: the first one to
claim an FQN decides its boundary type (Phase 3) and contributes its role
rules ahead of the later ones (Phase 1).
"""

import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed._builder.default_templates import (
    ShardingTemplate,
    sequence_identity_fields,
)
from hyper_parallel.distributed.recipe_spec import TP
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole


# ════════════════════════════════════════════════════════════════════════════
# MHC pre-modules
#
# MHC mixes recurrent streams locally inside each sequence-parallel shard, so
# its coefficients and small projections are replicated across TP ranks and
# the modules themselves are not communication boundaries.  Each physical
# owner is registered separately, so an outer module never claims parameters
# from a nested module subtree: with the MHC module FQNs as the fact set, a
# nested owner such as ``...attn_mhc_pre_module.phi`` is a fact in its own
# right and its ``phi.weight`` is declared by that owner.
# ════════════════════════════════════════════════════════════════════════════

_MHC_MODULES = frozenset({
    "attn_mhc_pre_module",
    "mlp_mhc_pre_module",
    "merge_mhc_module",
})

#: Phase 3 boundary type: an MHC subtree module that owns parameters.
MHC_PARAMS = "mhc_params"


def _is_mhc_module_fqn(fqn_lower: str) -> bool:
    """Return whether any path segment of *fqn_lower* names an MHC module."""
    return any(part in _MHC_MODULES for part in fqn_lower.split("."))


def _discover_mhc(named_modules: Mapping[str, Any]) -> Optional[FrozenSet[str]]:
    """Return the MHC subtree modules that own parameters (None when absent)."""
    found = frozenset(
        fqn.lower()
        for fqn, module in named_modules.items()
        if _is_mhc_module_fqn(fqn.lower())
        and bool(list(module.named_parameters(recurse=False)))
    )
    return found or None


def _mhc_role_rules(
    facts: FrozenSet[str],
) -> Sequence[Callable[[str], Optional[ParamRole]]]:
    """Phase 1: every parameter whose owner module sits in an MHC subtree.

    The default rules would otherwise fall back to ``SKIP`` (``branch_alpha``
    / ``phi.weight``) or ``NORM`` (``norm_gamma``, replicated either way).
    """

    def _replicated(param_fqn: str) -> Optional[ParamRole]:
        return (ParamRole.REPLICATED
                if param_fqn.rsplit(".", 1)[0] in facts else None)

    return (_replicated,)


def _mhc_boundary_type(
    fqn: str, facts: FrozenSet[str],
) -> Optional[str]:
    """Phase 3: a discovered MHC module is a param-only ``mhc_params`` spec.

    ``group`` is non-empty by construction: facts only hold modules with
    direct parameters, and every parameter reaches Phase 2's grouping.
    """
    return MHC_PARAMS if fqn.lower() in facts else None


def _mhc_paramless_fqns(facts: FrozenSet[str]) -> Iterable[str]:
    """Phase 2: an MHC module always owns parameters — nothing to seed."""
    _ = facts
    return ()


#: The param-only template: MHC parameters are replicated and the modules are
#: not communication boundaries, so no I/O contract is materialized.
_MHC_TEMPLATES: Dict[str, ShardingTemplate] = {
    MHC_PARAMS: ShardingTemplate(norm_placement=Replicate(), is_boundary=False),
}


# ════════════════════════════════════════════════════════════════════════════
# MTP layers
#
# MTP concatenates the shifted token embedding and the previous hidden state
# inside each sequence-parallel shard, so ``prev_proj`` keeps its weight
# replicated while preserving the sequence shard on its input and output
# activations.  The family is matched from the layer *structure* — a decoder
# layer exposing the full MTP child set with a parameterised ``prev_proj`` —
# never from a bare ``prev_proj`` name, so an unrelated projection carrying
# the same name is not claimed.
# ════════════════════════════════════════════════════════════════════════════

_MTP_LAYER = re.compile(r"(?:^|\.)layers\.\d+$")
_MTP_CHILDREN = frozenset({"mtp_block", "prev_norm", "emb_norm", "prev_proj"})
_PREV_PROJ = "prev_proj"

#: Phase 3 boundary type: the previous-state projection of an MTP layer.
MTP_PREV_PROJ = "mtp_prev_proj"


def _discover_mtp(named_modules: Mapping[str, Any]) -> Optional[FrozenSet[str]]:
    """Return the ``prev_proj`` FQNs of every complete MTP layer (None if none)."""
    found = set()
    for fqn, module in named_modules.items():
        if not _MTP_LAYER.search(fqn):
            continue
        children = dict(module.named_children())
        if not _MTP_CHILDREN.issubset(children):
            continue
        if list(children[_PREV_PROJ].named_parameters(recurse=False)):
            found.add(f"{fqn}.{_PREV_PROJ}")
    return frozenset(found) or None


def _mtp_role_rules(
    facts: FrozenSet[str],
) -> Sequence[Callable[[str], Optional[ParamRole]]]:
    """Phase 1: ``prev_proj`` parameters stay replicated across TP ranks."""

    def _replicated(param_fqn: str) -> Optional[ParamRole]:
        return (ParamRole.REPLICATED
                if param_fqn.rsplit(".", 1)[0] in facts else None)

    return (_replicated,)


def _mtp_boundary_type(fqn: str, facts: FrozenSet[str]) -> Optional[str]:
    """Phase 3: a discovered ``prev_proj`` is a sequence-identity boundary."""
    return MTP_PREV_PROJ if fqn.lower() in facts else None


def _mtp_paramless_fqns(facts: FrozenSet[str]) -> Iterable[str]:
    """Phase 2: MTP discovery requires a parameterised ``prev_proj``."""
    _ = facts
    return ()


#: Phase 4: ``prev_proj`` receives a single ``input`` and preserves its
#: sequence shard (plain replication when the sequence is not TP-sharded).
_MTP_TEMPLATES: Dict[str, ShardingTemplate] = {
    MTP_PREV_PROJ: ShardingTemplate(
        **sequence_identity_fields("input", sp_placement=Shard(1))),
}


# ════════════════════════════════════════════════════════════════════════════
# Shared experts owned by a TP-extended MoE dispatcher
#
# TP ranks participate in the EP dispatch domain, so the shared-expert
# projections must not additionally shard their hidden dimension: each rank
# independently processes its existing sequence shard and returns the same
# layout.  The family is matched from the *dispatcher contract* — the parent
# module exposes both ``experts`` and ``token_dispatcher``, and the shared
# expert exposes the ``linear_fc1``/``linear_fc2`` pair.  A regular
# shared-expert MLP therefore keeps the standard colwise/rowwise TP template
# (its parameters stay ``ParamRole.SHARED_EXPERT`` through the default naming
# rules, and the shared expert surfaces as its own nested ``mlp`` boundary).
# ════════════════════════════════════════════════════════════════════════════

_SHARED_EXPERT_LINEARS = ("linear_fc1", "linear_fc2")
_SHARED_EXPERT_ATTR = "shared_expert"
_DISPATCHER_ATTRS = ("experts", "token_dispatcher")

#: Phase 3 boundary type: a linear of a dispatcher-owned shared expert.
SHARED_EXPERT_LINEAR = "shared_expert_linear"


@dataclass(frozen=True)
class SharedExpertFacts:
    """Dispatcher-owned shared-expert linears found in one model.

    ``linear_fcns`` holds every declared linear; ``parameterless`` holds the
    ones owning no direct parameter, which Phase 2 seeds as contract-only
    boundaries (they can never surface from the parameter tree).
    """

    linear_fcns: FrozenSet[str]
    parameterless: FrozenSet[str]


def _discover_shared_expert(
    named_modules: Mapping[str, Any],
) -> Optional[SharedExpertFacts]:
    """Return the dispatcher-owned shared-expert linears (None when absent)."""
    linears = set()
    parameterless = set()
    for fqn, module in named_modules.items():
        shared_expert = getattr(module, _SHARED_EXPERT_ATTR, None)
        if shared_expert is None:
            continue
        if not all(hasattr(module, name) for name in _DISPATCHER_ATTRS):
            continue
        if not all(hasattr(shared_expert, name) for name in _SHARED_EXPERT_LINEARS):
            continue
        prefix = f"{fqn}.{_SHARED_EXPERT_ATTR}" if fqn else _SHARED_EXPERT_ATTR
        for linear_name in _SHARED_EXPERT_LINEARS:
            linear_fqn = f"{prefix}.{linear_name}"
            linears.add(linear_fqn)
            linear = getattr(shared_expert, linear_name)
            if not list(linear.named_parameters(recurse=False)):
                parameterless.add(linear_fqn)
    if not linears:
        return None
    return SharedExpertFacts(frozenset(linears), frozenset(parameterless))


def _shared_expert_role_rules(
    facts: SharedExpertFacts,
) -> Sequence[Callable[[str], Optional[ParamRole]]]:
    """Phase 1: shared-expert linears of a dispatcher MoE stay replicated.

    The default naming rules classify these parameters ``SHARED_EXPERT`` (TP
    colwise/rowwise along the hidden dim), which is exactly what the
    dispatcher contract forbids.
    """

    def _replicated(param_fqn: str) -> Optional[ParamRole]:
        return (ParamRole.REPLICATED
                if param_fqn.rsplit(".", 1)[0] in facts.linear_fcns else None)

    return (_replicated,)


def _shared_expert_boundary_type(
    fqn: str, facts: SharedExpertFacts,
) -> Optional[str]:
    """Phase 3: a dispatcher-owned shared-expert linear is its own boundary."""
    return SHARED_EXPERT_LINEAR if fqn.lower() in facts.linear_fcns else None


def _shared_expert_paramless_fqns(facts: SharedExpertFacts) -> Iterable[str]:
    """Phase 2: seed a declared linear that owns no parameter."""
    return facts.parameterless


#: Phase 4: each rank keeps its sequence shard and returns the same layout, so
#: the boundary is a pure identity (replicated parameters, no redistribution).
_SHARED_EXPERT_TEMPLATES: Dict[str, ShardingTemplate] = {
    SHARED_EXPERT_LINEAR: ShardingTemplate(
        **sequence_identity_fields("hidden_states", sp_placement=Shard(1))),
}


# ════════════════════════════════════════════════════════════════════════════
# DSA attention
#
# DSA does not follow a single q/k/v/o projection chain: some projections
# preserve sequence parallelism, some shard ``index``/``query`` heads,
# ``linear_kvb`` is consumed directly, and the attention root keeps shared
# "sink" parameters.  The family is therefore matched from the *module
# contract* — ``attention_type`` in {dsa, mla, gqa}, a ``param_sink_*``
# parameter, or one of the distinctive leaves.
#
# Discovery resolves every boundary of a matched attention subtree once and
# freezes it into a ``{module FQN: boundary type}`` map, so:
#
# - Phase 1 derives the parameter role from the owner's boundary type
#   (``COLWISE`` head-sharded projections, ``ROWWISE`` output projection,
#   ``REPLICATED`` sink parameters and sequence-identity linears); the
#   q/k layernorms keep the default ``NORM`` role;
# - Phase 2 seeds the boundaries whose modules own no parameter at all
#   (``rotary_emb`` / the sparse indexer), typed ``*_contract`` below;
# - Phase 3 is a pure lookup of that map, so the parent-dependent decisions
#   (e.g. the reduce-scatter axis of ``linear_proj``, which follows the owning
#   attention's runtime layout) are made in the inference phase only.
# ════════════════════════════════════════════════════════════════════════════

_ATTN_ROOT = r"(?:^|\.)layers\.\d+\.(?:mtp_block\.)?self_attention$"
_ATTN = _ATTN_ROOT[:-1] + r"\."
_ATTENTION_TYPES = frozenset({"dsa", "mla", "gqa"})
_DISTINCTIVE_LEAVES = frozenset({
    "linear_qb",
    "linear_kvb",
    "index_linear_qb",
    "index_linear_k",
    "linear_merge_weight",
    "sparse_lightning_indexer_kllloss",
})

# ── Phase 3 boundary types ──
DSA_SINK_PARAMS = "dsa_sink_params"
DSA_Q_HEAD_PROJ = "dsa_q_head_proj"
DSA_HEAD_PROJ = "dsa_head_proj"
DSA_KV_PROJ = "dsa_kv_proj"
DSA_LINEAR_IDENTITY = "dsa_linear_identity"
DSA_OUT_PROJ_BSH = "dsa_out_proj_bsh"
DSA_OUT_PROJ_SBH = "dsa_out_proj_sbh"
DSA_LAYERNORM = "dsa_layernorm"
DSA_ROTARY_CONTRACT = "dsa_rotary_contract"
DSA_INDEXER_KLL_CONTRACT = "dsa_indexer_kll_contract"
DSA_LM_HEAD = "dsa_lm_head"
DSA_EMBED_REPLICATED = "dsa_embed_replicated"

#: Leaf segment → boundary type for the direct/descendant projections of a
#: matched attention root.  ``linear_qb`` carries the head-count owner tag
#: (see ``ShardingTemplate.head_count_owner_parent``), the other head-sharded
#: projections merely share its contract.
_DSA_LEAF_TYPES = {
    "linear_qb": DSA_Q_HEAD_PROJ,
    "index_linear_qb": DSA_HEAD_PROJ,
    "linear_merge_weight": DSA_HEAD_PROJ,
    "linear_kvb": DSA_KV_PROJ,
    "linear_qkv": DSA_LINEAR_IDENTITY,
    "index_linear_k": DSA_LINEAR_IDENTITY,
    "q_layernorm": DSA_LAYERNORM,
    "k_layernorm": DSA_LAYERNORM,
    "index_k_layernorm": DSA_LAYERNORM,
    "rotary_emb": DSA_ROTARY_CONTRACT,
    "gather_rotary_emb": DSA_ROTARY_CONTRACT,
    "sparse_lightning_indexer_kllloss": DSA_INDEXER_KLL_CONTRACT,
}
_LINEAR_PROJ = "linear_proj"

#: Phase 1: boundary type → parameter role.  Types absent here (the
#: layernorms, the vocab-parallel lm_head / embed_tokens) keep the role their
#: default naming rule assigns — the structural template only reshapes the
#: boundary contract.
_DSA_TYPE_ROLES = {
    DSA_SINK_PARAMS: ParamRole.REPLICATED,
    DSA_Q_HEAD_PROJ: ParamRole.COLWISE,
    DSA_HEAD_PROJ: ParamRole.COLWISE,
    DSA_KV_PROJ: ParamRole.COLWISE,
    DSA_LINEAR_IDENTITY: ParamRole.REPLICATED,
    DSA_OUT_PROJ_BSH: ParamRole.ROWWISE,
    DSA_OUT_PROJ_SBH: ParamRole.ROWWISE,
}


@dataclass(frozen=True)
class DsaFacts:
    """Boundaries discovered in one DSA-shaped model.

    ``boundary_types`` keys are lower-cased module FQNs (Phase 1 classifies
    lower-cased names and Phase 3 lower-cases its FQN); ``parameterless``
    keeps the module's own FQN spelling because Phase 2 writes it into the
    plan as a module key.
    """

    boundary_types: Mapping[str, str]
    parameterless: FrozenSet[str]


def _is_dsa_attention(module: Any) -> bool:
    """Match the DSA/MLA attention contract without using a model name."""
    if getattr(module, "attention_type", None) in _ATTENTION_TYPES:
        return True
    if any(
        name.startswith("param_sink_")
        for name, _ in module.named_parameters(recurse=False)
    ):
        return True
    child_names = {name for name, _ in module.named_children()}
    return bool(_DISTINCTIVE_LEAVES.intersection(child_names))


def _output_projection_type(root_module: Any, fqn_lower: str) -> str:
    """Reduce-scatter axis of an output projection, from the owning attention.

    GQA/DSA feed ``linear_proj`` in BSH layout, while MLA transposes its BSH
    attention result to SBH before the projection and transposes the output
    back afterwards.  The owning attention implementation decides, not the
    FQN: regular decoder layers may mix DSA and MLA attention, so an
    ``mtp_block`` name alone cannot determine the runtime layout.
    """
    attention_type = getattr(root_module, "attention_type", None)
    if attention_type == "mla":
        sequence_dim = 0
    elif attention_type in {"gqa", "dsa"}:
        sequence_dim = 1
    else:
        # Retain the legacy fallback for architecture-compatible test doubles
        # or external modules that do not expose attention_type.
        sequence_dim = 0 if ".mtp_block." in fqn_lower else 1
    return DSA_OUT_PROJ_SBH if sequence_dim == 0 else DSA_OUT_PROJ_BSH


def _dsa_boundary_type_for(
    fqn_lower: str, module: Any, roots: Mapping[str, Any],
) -> Optional[str]:
    """Return the DSA boundary type of one module (None: not a DSA boundary)."""
    if fqn_lower in roots:
        # The attention root only owns sinks; when it does not, it is not a
        # boundary at all.
        return (DSA_SINK_PARAMS
                if list(module.named_parameters(recurse=False)) else None)
    # The integration gathers the language-model SP output before its
    # multi-token prediction vocabulary heads, so the vocab-parallel
    # projection consumes a replicated sequence (not Shard(1)), and the
    # embedding output stays replicated for the multimodal fusion that runs
    # before the language-model sequence scatter.
    if fqn_lower == "lm_head" or fqn_lower.endswith(".lm_head"):
        return DSA_LM_HEAD
    if fqn_lower.endswith(".language_model.embed_tokens"):
        return DSA_EMBED_REPLICATED
    if not re.search(_ATTN, fqn_lower):
        return None
    owner = next((root for root in roots
                  if fqn_lower.startswith(f"{root}.")), None)
    if owner is None:
        return None
    leaf = fqn_lower.rsplit(".", 1)[-1]
    if leaf == _LINEAR_PROJ:
        return _output_projection_type(roots[owner], fqn_lower)
    return _DSA_LEAF_TYPES.get(leaf)


def _discover_dsa(named_modules: Mapping[str, Any]) -> Optional[DsaFacts]:
    """Resolve every DSA boundary of a module tree (None when it exposes none)."""
    # Attention roots: the FQN must sit at the canonical decoder-layer path
    # AND the module must expose the DSA contract.
    roots: Dict[str, Any] = {}
    for fqn, module in named_modules.items():
        if re.search(_ATTN_ROOT, fqn) and _is_dsa_attention(module):
            roots[fqn.lower()] = module
    if not roots:
        return None

    boundary_types: Dict[str, str] = {}
    parameterless: set = set()
    for fqn, module in named_modules.items():
        fqn_lower = fqn.lower()
        module_type = _dsa_boundary_type_for(fqn_lower, module, roots)
        if module_type is None:
            continue
        boundary_types[fqn_lower] = module_type
        # A boundary owning no parameter can never surface from the parameter
        # tree, so Phase 2 seeds it explicitly.
        if not list(module.named_parameters(recurse=False)):
            parameterless.add(fqn)
    return DsaFacts(dict(boundary_types), frozenset(parameterless))


def _dsa_role_rules(
    facts: DsaFacts,
) -> Sequence[Callable[[str], Optional[ParamRole]]]:
    """Phase 1: derive the role from the owning module's boundary type."""

    def _by_boundary_type(param_fqn: str) -> Optional[ParamRole]:
        owner_type = facts.boundary_types.get(param_fqn.rsplit(".", 1)[0])
        return _DSA_TYPE_ROLES.get(owner_type) if owner_type else None

    return (_by_boundary_type,)


def _dsa_boundary_type(fqn: str, facts: DsaFacts) -> Optional[str]:
    """Phase 3: look the module up in the discovered boundary map."""
    return facts.boundary_types.get(fqn.lower())


def _dsa_paramless_fqns(facts: DsaFacts) -> Iterable[str]:
    """Phase 2: seed the boundaries whose modules own no parameter."""
    return facts.parameterless


# ── Phase 4: DSA boundary type → template ──────────────────────────────────
#
# Every structural template declares the TP axis only, exactly like the
# boundaries they replace: the CP axis of a DSA boundary is not part of its
# contract (a missing dim resolves to Replicate), and mixing CP-declaring and
# CP-blind boundaries inside one attention would insert spurious CP
# redistribution between them.  The sequence_parallel switch is carried by the
# sp_*/nosp_* pair instead:
#   - a contract whose TP placement expresses "the sequence is sharded"
#     (Shard(1) on the hidden-state/sequence axis) becomes Replicate without
#     SP — there is no sequence shard to preserve;
#   - the reduce-scatter targets follow the standard attention/mlp templates
#     (``out_dst`` degrades to Replicate without SP);
#   - the two contract-only auxiliaries (rotary_emb / the sparse indexer)
#     declare explicit per-tensor placements of the DSA runtime's own
#     activations, not hidden states, so they are SP-independent.
_DSA_SEQ_IDENTITY_SP = (Shard(1), Shard(1), Shard(1), Shard(1))
_DSA_SEQ_IDENTITY_NOSP = (Replicate(), Replicate(), Replicate(), Replicate())


def _dsa_template(
    sp_placements, nosp_placements, *, in_key: str = "input",
    out_key: str = "output", **flags,
) -> ShardingTemplate:
    """Build a TP-only template: input/output identity on the declared axes."""
    sp_in_src, sp_in_dst, sp_out_src, sp_out_dst = sp_placements
    nosp_in_src, nosp_in_dst, nosp_out_src, nosp_out_dst = nosp_placements
    return ShardingTemplate(
        sp_in_src={in_key: {TP: sp_in_src}},
        sp_in_dst={in_key: {TP: sp_in_dst}},
        sp_out_src={out_key: {TP: sp_out_src}},
        sp_out_dst={out_key: {TP: sp_out_dst}},
        nosp_in_src={in_key: {TP: nosp_in_src}},
        nosp_in_dst={in_key: {TP: nosp_in_dst}},
        nosp_out_src={out_key: {TP: nosp_out_src}},
        nosp_out_dst={out_key: {TP: nosp_out_dst}},
        **flags,
    )


def _dsa_contract_template(inputs: Mapping[str, Any]) -> ShardingTemplate:
    """A parameter-less boundary that only redistributes named inputs.

    The auxiliary tensors of the DSA runtime are not hidden states: their
    placements are declared explicitly and do not follow the SP switch.
    """

    def _named(placement) -> Dict[str, Any]:
        return {name: {TP: placement} for name in inputs}

    in_src = {name: {TP: placement} for name, placement in inputs.items()}
    return ShardingTemplate(
        sp_in_src=in_src,
        sp_in_dst=_named(Replicate()),
        sp_out_src={},
        sp_out_dst={},
        nosp_in_src=dict(in_src),
        nosp_in_dst=_named(Replicate()),
        nosp_out_src={},
        nosp_out_dst={},
    )


#: Phase 4: the boundary types contributed by DSA.
_DSA_TEMPLATES: Dict[str, ShardingTemplate] = {
    # Shared sink parameters: replicated, no boundary contract.
    DSA_SINK_PARAMS: ShardingTemplate(norm_placement=Replicate(),
                                      is_boundary=False),
    # Head-sharded projections: the input is reduce-scattered from the SP
    # sequence shard to a full sequence, the output stays head-sharded.
    DSA_Q_HEAD_PROJ: _dsa_template(
        (Shard(1), Replicate(), Shard(-1), Shard(-1)),
        (Replicate(), Replicate(), Shard(-1), Shard(-1)),
        head_count_owner_parent=True,
    ),
    DSA_HEAD_PROJ: _dsa_template(
        (Shard(1), Replicate(), Shard(-1), Shard(-1)),
        (Replicate(), Replicate(), Shard(-1), Shard(-1)),
    ),
    DSA_KV_PROJ: _dsa_template(
        (Replicate(), Replicate(), Shard(-1), Shard(-1)),
        (Replicate(), Replicate(), Shard(-1), Shard(-1)),
    ),
    # Already-latent linears and the sequence-parallel layernorms: pure
    # identity on the sequence axis.
    DSA_LINEAR_IDENTITY: _dsa_template(_DSA_SEQ_IDENTITY_SP,
                                       _DSA_SEQ_IDENTITY_NOSP),
    DSA_LAYERNORM: _dsa_template(_DSA_SEQ_IDENTITY_SP, _DSA_SEQ_IDENTITY_NOSP,
                                 in_key="hidden_states"),
    # Output projection: the local matmul is partial over the TP-sharded
    # hidden dim and is reduce-scattered onto the sequence layout of the
    # owning attention implementation.
    DSA_OUT_PROJ_BSH: _dsa_template(
        (Shard(-1), Shard(-1), Partial(), Shard(1)),
        (Shard(-1), Shard(-1), Partial(), Replicate()),
    ),
    DSA_OUT_PROJ_SBH: _dsa_template(
        (Shard(-1), Shard(-1), Partial(), Shard(0)),
        (Shard(-1), Shard(-1), Partial(), Replicate()),
    ),
    DSA_ROTARY_CONTRACT: _dsa_contract_template({
        name: Replicate() for name in ("t", "cos", "sin")
    }),
    DSA_INDEXER_KLL_CONTRACT: _dsa_contract_template({
        "index_query": Replicate(),
        "index_key": Replicate(),
        "merge_weight": Replicate(),
        "query": Shard(1),
        "key": Replicate(),
        "topk_indices": Replicate(),
        "softmax_max": Shard(2),
        "softmax_sum": Shard(2),
        "query_rope": Shard(1),
        "key_rope": Replicate(),
        "actual_seq_qlen": Replicate(),
        "actual_seq_klen": Replicate(),
    }),
    # Vocab-parallel heads of a DSA-shaped (VL/MTP) integration: the
    # multi-token-prediction heads consume a replicated sequence, so the
    # embedding output stays replicated for the parent and the language-model
    # head keeps the loss-parallel terminal contract.
    DSA_LM_HEAD: _dsa_template(
        (Replicate(), Replicate(), Shard(-1), Shard(-1)),
        (Replicate(), Replicate(), Shard(-1), Shard(-1)),
        loss_parallel_out_dst=True,
    ),
    DSA_EMBED_REPLICATED: _dsa_template(
        (Replicate(), Replicate(), Partial(), Replicate()),
        (Replicate(), Replicate(), Partial(), Replicate()),
        in_key="hidden_states",
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# Aggregation: the families above, in Phase 3 priority order
# ════════════════════════════════════════════════════════════════════════════

def _merge_templates() -> Dict[str, ShardingTemplate]:
    """Merge every family's boundary-type → template mapping (Phase 4).

    Static data, so the lookup table is complete before any model is
    inspected.  A boundary type declared by two families fails fast here: it
    is the only key Phase 4 looks up.
    """
    merged: Dict[str, ShardingTemplate] = {}
    for templates in (_DSA_TEMPLATES, _MTP_TEMPLATES, _SHARED_EXPERT_TEMPLATES,
                      _MHC_TEMPLATES):
        for boundary_type, template in templates.items():
            if boundary_type in merged:
                raise ValueError(
                    f"boundary type {boundary_type!r} is declared by two "
                    f"structural families — boundary types are the Phase 4 "
                    f"lookup key and must be unique"
                )
            merged[boundary_type] = template
    return merged


#: Phase 4: every boundary type contributed by the structural families.
STRUCTURAL_TEMPLATES: Dict[str, ShardingTemplate] = _merge_templates()


@dataclass(frozen=True)
class ModelStructure:
    """Structural facts discovered once from the instantiated model.

    One optional facts object per family; ``None`` means the family is not
    present in this model.  The families are consulted most-specific-first:
    the first one to claim an FQN decides its boundary type (Phase 3) and
    contributes its role rules ahead of the later ones (Phase 1).
    """

    dsa: Optional[DsaFacts] = None
    mtp: Optional[FrozenSet[str]] = None
    shared_expert: Optional[SharedExpertFacts] = None
    mhc: Optional[FrozenSet[str]] = None

    def _present(self) -> List[Tuple[Any, Any, Any, Any]]:
        """(facts, role_rules, boundary_type, paramless_fqns) per family."""
        return [
            (self.dsa, _dsa_role_rules, _dsa_boundary_type,
             _dsa_paramless_fqns),
            (self.mtp, _mtp_role_rules, _mtp_boundary_type,
             _mtp_paramless_fqns),
            (self.shared_expert, _shared_expert_role_rules,
             _shared_expert_boundary_type, _shared_expert_paramless_fqns),
            (self.mhc, _mhc_role_rules, _mhc_boundary_type,
             _mhc_paramless_fqns),
        ]

    def role_rules(self) -> List[Callable[[str], Optional[ParamRole]]]:
        """Phase 1: structural parameter-role predicates (may be empty)."""
        rules: List[Callable[[str], Optional[ParamRole]]] = []
        for facts, role_rules, _, _ in self._present():
            if facts is not None:
                rules.extend(role_rules(facts))
        return rules

    def boundary_type(self, fqn: str) -> Optional[str]:
        """Phase 3: the structurally implied boundary type for *fqn*, or None.

        Every family resolves its boundaries at discovery time, so this is a
        pure lookup (the planner keeps calling it per module FQN while it
        walks the module tree).
        """
        for facts, _, boundary_type, _ in self._present():
            if facts is None:
                continue
            family_type = boundary_type(fqn, facts)
            if family_type is not None:
                return family_type
        return None

    @property
    def paramless_boundary_fqns(self) -> FrozenSet[str]:
        """Phase 2: boundary FQNs that own no parameter at all."""
        seeded: set = set()
        for facts, _, _, paramless_fqns in self._present():
            if facts is not None:
                seeded.update(paramless_fqns(facts))
        return frozenset(seeded)


#: The structure of a model no structural family claims.  Unit tests that call
#: the planner's Phase 1-3 helpers directly get this, i.e. generic behaviour.
EMPTY_STRUCTURE = ModelStructure()


def detect_model_structure(
    model: Any, named_modules: Optional[Mapping[str, Any]] = None,
) -> ModelStructure:
    """Discover every structural family present in *model* (once per plan)."""
    modules = dict(model.named_modules()) if named_modules is None else named_modules
    return ModelStructure(
        dsa=_discover_dsa(modules),
        mtp=_discover_mtp(modules),
        shared_expert=_discover_shared_expert(modules),
        mhc=_discover_mhc(modules),
    )
