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
"""sharding_config: data model for the dual-mode DTensor parallel strategy (05 §3.1/§3.2/§3.5 canonical).

Contains:
- ``MeshAxisName``: mesh dimension name enum (canonical definition, imported and reused by later docs such as 06);
- ``NamedPlacement``: alias for ``dict[MeshAxisName, Placement]``;
- ``ShardingPlan`` / ``ModuleShardingSpec``: model-level plan and per-module I/O contract;
- ``ShardingTemplate`` / ``TEMPLATES``: semantic role → placement templates (TP+CP+EP, three dims);
- ``PlacementMismatchError``: error for placement declarations inconsistent with DTensor propagation;
- ``resolve_placements`` / ``_multi_dim`` / ``_normalize_out_fields``: placement utilities.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

logger = logging.getLogger(__name__)


class MeshAxisName(str, Enum):
    """Canonical enum of mesh dimension names (a str enum, directly comparable to plain strings like "tp" and usable as a dict key)."""
    TP = "tp"
    CP = "cp"
    EP = "ep"
    PP = "pp"
    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    DP_CP = "dp_cp"
    EP_SHARD = "ep_shard"


# Shorthand aliases for the {TP: ..., CP: ..., EP: ...} literals in templates and examples.
# str-enum keys interoperate with plain string keys like "tp" (hash/eq are identical).
TP = MeshAxisName.TP
CP = MeshAxisName.CP
EP = MeshAxisName.EP

# NamedPlacement = {MeshAxisName: Placement}.
# The key is a mesh dimension name; the N in Shard(N) of a value is a tensor dimension index (05 §3.2.1).
NamedPlacement = Dict[MeshAxisName, Placement]


class PlacementMismatchError(ValueError):
    """DTensor propagation result is inconsistent with the ModuleShardingSpec declaration (05 §5.3)."""

    def __init__(self, module_name: str, expected, actual, stage: str):
        self.module_name = module_name
        self.expected = expected
        self.actual = actual
        self.stage = stage
        super().__init__(
            f"[{module_name}] {stage} placement mismatch:\n"
            f"  Expected (from ShardingConfig.{stage}): {expected}\n"
            f"  Actual   (from DTensor propagation):   {actual}\n"
            f"  → Check the ShardingConfig for this module."
        )


@dataclass
class ModuleShardingSpec:
    """Complete DTensor contract for a single module (05 §3.2).

    The four placement fields form the complete I/O contract — the runtime does no
    inference and executes exactly as declared:

      in_src:  placement of the input when it arrives at the module boundary
               (from the output of an upstream module or the dataloader)
      in_dst:  placement required by the module's internal computation
               (a mismatch triggers communication)
      out_src: placement naturally produced by the module's internal computation
               (used by validate mode)
      out_dst: placement expected by downstream modules (a mismatch triggers
               communication)
    """
    # ── Parameter sharding: submodule path → NamedPlacement ──
    params: Dict[str, NamedPlacement] = field(default_factory=dict)

    # ── Input contract ──
    in_src: Dict[str, NamedPlacement] = field(default_factory=dict)
    in_dst: Dict[str, NamedPlacement] = field(default_factory=dict)

    # ── Output contract ──
    # out_src=None: no src validation; out_dst=None: output needs no redistribution.
    # Single-output modules use {"output": NamedPlacement}; the scalar shorthand
    # {TP: ...} is wrapped into {"output": ...} during the normalization phase
    # (_normalize_out_fields).
    out_src: Optional[Dict[str, NamedPlacement]] = None
    out_dst: Optional[Dict[str, NamedPlacement]] = None
    # out_names: output-name ordering for multi-output modules (returning a tuple),
    # used to map the keys of out_src/out_dst to tuple positions
    # (RedistOp.arg_index). Defaults to the key order of out_src.
    out_names: Optional[List[str]] = None

    # ── Boundary flag ──
    is_boundary: bool = True

    # ── Structural flags (user-configurable) ──
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ The four user extension-point interfaces: use_local_map /        │
    # │ local_compute_fn / inner_target / inner_wrapper                  │
    # │                                                                 │
    # │ They address scenarios where a module's internal computation     │
    # │ cannot be expressed by DTensor dispatch, or is not covered by    │
    # │ the built-in wrappers. They fall into two families:              │
    # │                                                                 │
    # │ [local-region family] (module level: skeleton unchanged,         │
    # │   content swapped)                                               │
    # │   use_local_map / local_compute_fn                               │
    # │   The skeleton = _wrap_local_region_forward: boundary            │
    # │   entry/exit stitching + local compute + validate dual-mode      │
    # │   fault tolerance (to_local/_temp_local_params/from_local        │
    # │   re-wrapping), shared by both modes. Whether a module runs      │
    # │   through the skeleton is **derived from a single resolution     │
    # │   chain** (not a stored bool) by _resolve_local_compute_fn:      │
    # │     chain link 1  local_compute_fn (user-defined computation)    │
    # │     chain link 2  built-in EP wrapper injection intent           │
    # │                   (_ep_size>0, see below)                        │
    # │     chain link 3  use_local_map (pure gate: the module's own     │
    # │                   forward)                                       │
    # │     none present → None (skeleton not used)                      │
    # │                                                                 │
    # │   ★ Built-in EP wrapper: _hf_native_ep_compute                   │
    # │   When the planner recognizes an HF-native MoE (per-expert or    │
    # │   batched experts layout) and ep_size>1, it automatically        │
    # │   records the TP-extend-EP injection intent (_ep_size>0), and    │
    # │   chain link 2 of the resolution chain injects the built-in EP   │
    # │   forward: router adaptation (MOE_ROUTER_ADAPTERS) → extended    │
    # │   EP-group all-to-all dispatch → full local expert SwiGLU →      │
    # │   a2a combine → weighted aggregation — expert weights are only   │
    # │   Shard(0) along the expert dim, with no all_gather/             │
    # │   reduce_scatter. A user-supplied local_compute_fn is its peer   │
    # │   on the chain: a custom MoE (custom router / non-standard       │
    # │   expert layout / DeepEP dispatcher) can inject its own          │
    # │   implementation to replace it.                                  │
    # │                                                                 │
    # │ [inner-wrap family] (submodule level: content and wrapping both  │
    # │   replaced)                                                      │
    # │   inner_target / inner_wrapper                                   │
    # │   The mechanism is a generic "locate an inner submodule +        │
    # │   replace its forward": inner_target answers "replace whom",     │
    # │   inner_wrapper answers "replace with what" (if omitted, one is  │
    # │   chosen heuristically from the built-in registry). The only     │
    # │   built-in domain at present is CP (K/V all-gather), hence it is │
    # │   called the CP wrapper family.                                  │
    # │                                                                 │
    # │   ★ Built-in CP wrappers: the four entries of                    │
    # │     CP_WRAPPER_REGISTRY                                          │
    # │     "sdpa_qkv":  NeMo convention forward(q,k,v,...) → explicit   │
    # │                  all-gather of K/V + D-04 offset causal mask     │
    # │                  (automatically fixes the is_causal alignment    │
    # │                  error under CP)                                 │
    # │     "sdpa_hf" :  HF convention forward(hidden_states,...) →      │
    # │                  primitive interception of                       │
    # │                  F.scaled_dot_product_attention (reuses HF       │
    # │                  projections/RoPE), with misfire detection       │
    # │                  (an error is raised if no call is intercepted)  │
    # │     "flex_qkv"/"flex_hf": the two isomorphic FlexAttention       │
    # │                  entries (block_mask must be built for the       │
    # │                  global kv length)                               │
    # │   Dual resolution chains: _resolve_inner_target (locating) +     │
    # │   _resolve_inner_wrapper (scheme selection); the gate is         │
    # │   likewise derived (resolution is not None and cp_mesh is        │
    # │   active).                                                       │
    # │                                                                 │
    # │ The two families are orthogonal and composable: the same module  │
    # │ may declare both inner_* (CP wrapping of an inner attention) and │
    # │ local_* (module-level skeleton).                                 │
    # └─────────────────────────────────────────────────────────────────┘
    #
    # use_local_map: **pure gate of the local region** (chain link 3) — declares
    # that "the module's own forward is the data-dependent logic" (inexpressible
    # by DTensor dispatch; e.g. the a2a of an EP-aware custom MoE already lives
    # inside forward) → runs through the skeleton, with compute_fn = the module's
    # own forward. When template inference yields True it is force-inherited (a
    # missing MoE a2a would be a numerical error and must not be disabled via an
    # override); the user may explicitly set it to True for a custom module in
    # plan_overrides (05 §3.6.7/§8.6).
    use_local_map: bool = False

    # ── inner-wrap custom entry points (user-configurable, 05 §4.4.2/§8.6) ──
    # inner_target: **pure location** — names the attribute of the inner
    #   submodule whose forward is to be replaced ("self" means the module
    #   itself). Automatic locating (_resolve_inner_target) fails fast, in which
    #   case the user must specify it via plan_overrides. It does not change
    #   behavior selection (decided by inner_wrapper or the heuristic dispatch)
    #   and does not rewrite any flag (the gate is derived).
    # inner_wrapper: **pure behavior** — selects which scheme wraps the target:
    #   - str: a name in the CP_WRAPPER_REGISTRY registry ("sdpa_qkv"/"sdpa_hf"/
    #     "flex_qkv"/"flex_hf", or a user-registered name); explicitly pins a
    #     built-in scheme; an unknown name fails fast;
    #   - Callable: a fully custom wrapper with signature
    #     fn(target_module, cp_mesh), which replaces target.forward in place
    #     (use cp_utils.flex_cp_allgather for K/V all-gather; the entry point is
    #     fault-tolerant across the DTensor/local dual mode — if it only accepts
    #     local tensors, declare use_local_map for this module instead, and the
    #     skeleton will convert everything to local at the module entry).
    #   Default (None): when inner_target is declared or the module is
    #   recognized as attention by the template, dispatch heuristically over a
    #   2×2 grid (signature style × SDPA/Flex); the dispatch result is visible
    #   in the apply-phase logs and in spec._resolved_inner_wrapper, and can be
    #   pinned with the str form.
    inner_target: Optional[str] = None
    inner_wrapper: Optional[Union[str, Callable]] = None

    # ── local-region custom computation (user-configurable, 05 §4.4.3/§8.6) ──
    # local_compute_fn: the custom compute_fn of the local region (chain link 1),
    #   with signature fn(module, *local_args, **local_kwargs) -> Tensor —
    #   executed on local tensors inside the _wrap_local_region_forward skeleton
    #   (the skeleton handles boundary entry/exit stitching + validate dual-mode
    #   to_local/_temp_local_params/from_local re-wrapping). Suitable for custom
    #   modules that want to reuse the skeleton with their own data-dependent
    #   logic: typically a custom MoE (router not in MOE_ROUTER_ADAPTERS /
    #   expert layout not using HF-standard naming / hooked to a DeepEP fused
    #   dispatcher).
    # Priority: local_compute_fn > built-in EP wrapper injection intent
    #   (_ep_size>0) > use_local_map gate (the module's own forward) — a single
    #   resolution chain (_resolve_local_compute_fn); declaring it takes effect
    #   immediately: **there is no need — and it is wrong — to also set
    #   use_local_map**; the skeleton gate is derived from the resolution chain
    #   (non-None means the skeleton is used).
    # Dual-mode convention: inputs are always local tensors and the return value
    #   is a local tensor; validate's DTensor unwrap/re-wrapping is done by the
    #   skeleton, so the compute_fn need not be aware of the mode.
    local_compute_fn: Optional[Callable] = None

    # ── Internal flags (set automatically by ShardingPlanner / applier) ──
    _is_terminal: bool = False    # marked automatically during chained propagation
    _needs_cp_attn: bool = False  # attention module: inner attention needs a CP-aware forward replacement
    # _resolved_inner_wrapper: written back by the applier after resolution (for
    # introspection) — the inner wrapper actually injected:
    # "sdpa_qkv"/"sdpa_hf"/"flex_qkv"/"flex_hf"/"custom"/None.
    _resolved_inner_wrapper: Optional[str] = None
    # D-09 (05 §6.4.7): EP pass-through for HF-native MoE. A non-empty _ep_stack
    # means per-expert parameters must be pre-stacked into [E, ...] in Phase A,
    # and the wrapper injects _hf_native_ep_compute (D-10 TP-extend-EP,
    # 05 §6.4.8).
    _ep_stack: Dict[str, List[str]] = field(default_factory=dict)
    _moe_router: str = "default"  # adapter name in MOE_ROUTER_ADAPTERS
    # D-10 (05 §6.4.8): TP-extend-EP. When >0 this is the extended EP group size
    # (= ep_size; the a2a communication domain includes TP ranks); the MoE uses
    # an SP-in identity boundary + a derived expert mesh (edp, ep); expert
    # weights are only Shard(0) along the expert dim; the wrapper injects
    # _hf_native_ep_compute.
    _ep_size: int = 0


@dataclass
class ShardingPlan:
    """Complete sharding plan for a model (05 §3.1)."""
    # {module_fqn: ModuleShardingSpec} — contains only modules with is_boundary=True
    modules: Dict[str, ModuleShardingSpec] = field(default_factory=dict)

    # Global switches
    sequence_parallel: bool = True
    loss_parallel: bool = False

    # Special parameter handlers: {module_fqn.param_name: handler_name}
    special_handlers: Dict[str, str] = field(default_factory=dict)

    # Mesh dimension names (consistent with DeviceMesh.mesh_dim_names)
    mesh_dim_names: Tuple[str, ...] = ()

    # Tied-weight pairs: [(fqn_a, fqn_b)], parameters sharing storage
    # (embed_tokens <-> lm_head).
    tied_pairs: List[Tuple[str, str]] = field(default_factory=list)


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
    use_local_map: bool = False   # MoE EP: forward needs a local region
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


def resolve_placements(
    named: NamedPlacement,
    mesh_dim_names: Tuple[str, ...],
) -> List[Placement]:
    """Arrange placements in mesh_dim_names order, fill missing axes with Replicate()."""
    return [named.get(axis, Replicate()) for axis in mesh_dim_names]


def _normalize_out_fields(spec: ModuleShardingSpec) -> ModuleShardingSpec:
    """Normalize the scalar shorthand {TP: ...} into {'output': {TP: ...}} (05 §3.5).

    Detection heuristic: if val is a non-None dict and any of its values is not a
    dict, it is judged to be a scalar NamedPlacement shorthand. Idempotent — a
    second call on an already-normalized dict contract changes nothing.
    """
    for attr in ("out_src", "out_dst"):
        val = getattr(spec, attr, None)
        if val and not all(isinstance(v, dict) for v in val.values()):
            setattr(spec, attr, {"output": dict(val)})
    return spec


def _hid(tp_p, cp_p, ep_p=None) -> Dict[str, NamedPlacement]:
    """Shorthand for the single-input hidden_states contract."""
    return {"hidden_states": _multi_dim(tp=tp_p, cp=cp_p, ep=ep_p or Replicate())}


def _out(tp_p, cp_p, ep_p=None) -> NamedPlacement:
    """Shorthand for the single-output (scalar shorthand) contract."""
    return _multi_dim(tp=tp_p, cp=cp_p, ep=ep_p or Replicate())


# ── TEMPLATES: complete templates for the 7 semantic roles (05 §3.5, declared over TP+CP+EP) ──
# CP-dim rule: parameters are always Replicate (CP does not shard parameters);
# activations are Shard(1) (sequence dim) or Replicate.
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
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Partial(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
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
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Partial(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
    ),

    # ── Norm (RMSNorm/LayerNorm: weight replicated, zero communication) ──
    "norm": ShardingTemplate(
        norm_placement=Replicate(),
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Shard(1), Shard(1)),      # identity
        sp_out_src=_out(Shard(1), Shard(1)),
        sp_out_dst=_out(Shard(1), Shard(1)),     # identity
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Replicate(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
    ),

    # ── Embedding (weight Shard(0) along the vocab dim, output Partial → SP+CP) ──
    "embed": ShardingTemplate(
        colwise_placement=Shard(0),          # weight: [V/tp, H]
        sp_in_src={"input": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        sp_in_dst={"input": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        sp_out_src=_out(Partial(), Replicate()),
        sp_out_dst=_out(Shard(1), Shard(1)),     # reduce-scatter → SP+CP
        nosp_in_src={"input": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        nosp_in_dst={"input": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        nosp_out_src=_out(Partial(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
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
        nosp_in_src=_hid(Replicate(), Replicate()),
        nosp_in_dst=_hid(Replicate(), Replicate()),
        nosp_out_src=_out(Shard(-1), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
    ),

    # ── MoE Gate (Router: weight replicated, output redistributes → EP) ──
    "moe_gate": ShardingTemplate(
        norm_placement=Replicate(),          # router weight/bias: replicated
        sp_in_src=_hid(Shard(1), Shard(1)),
        sp_in_dst=_hid(Replicate(), Replicate()),
        sp_out_src=_out(Replicate(), Replicate()),
        sp_out_dst=_out(Replicate(), Replicate(), Shard(0)),
        nosp_in_src=_hid(Replicate(), Replicate()),
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
        nosp_in_src={"x_BLD": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        nosp_in_dst={"x_BLD": _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())},
        nosp_out_src=_out(Partial(), Replicate()),
        nosp_out_dst=_out(Replicate(), Replicate()),
        use_local_map=True,
    ),
}
