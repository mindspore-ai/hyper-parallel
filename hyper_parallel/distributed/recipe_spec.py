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
"""recipe_spec: recipe input contract and placement DSL for the dual-mode DTensor strategy.

Contains the user-visible plan/spec data model (05 §3.1/§3.2 canonical):
- ``MeshAxisName``: mesh dimension name enum (canonical definition, imported and reused by later docs such as 06);
- ``NamedPlacement``: alias for ``dict[MeshAxisName, Placement]``;
- ``ModuleShardingSpec``: per-module I/O contract;
- ``PlacementMismatchError``: error for placement declarations inconsistent with DTensor propagation;
- ``resolve_placements`` / ``parse_placement`` / ``parse_named_placement`` / ``_normalize_out_fields``:
  placement utilities;
- the injection DSL (``local_compute`` / ``inner_wrapper`` decorators,
  ``InjectionMeta``, ``require_injection_meta``): the recipe-side input
  contract of the explicit injection mechanism.

The model-level :class:`ShardingPlan` lives in ``distributed/plan.py``; the
semantic-role templates live in ``distributed/_builder/default_templates.py``.
"""

import inspect
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

if TYPE_CHECKING:
    from hyper_parallel.distributed.tensor_parallel.head_count import TpLocalAttrPlan

logger = logging.getLogger(__name__)


class MeshAxisName(str, Enum):
    """Canonical enum of mesh dimension names.

    A str enum: directly comparable to plain strings like "tp" and usable as
    a dict key.
    """
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
DP = MeshAxisName.DP

# NamedPlacement = {MeshAxisName: Placement}.
# The key is a mesh dimension name; the N in Shard(N) of a value is a tensor dimension index (05 §3.2.1).
NamedPlacement = Dict[MeshAxisName, Placement]


class PlacementMismatchError(ValueError):
    """DTensor propagation result is inconsistent with the ModuleShardingSpec declaration (05 §5.3)."""

    def __init__(self, module_name: str, expected: Any, actual: Any, stage: str) -> None:
        """Initialize the mismatch error.

        Args:
            module_name: FQN of the module whose placement mismatched.
            expected: Placement declared by the ShardingConfig.
            actual: Placement produced by DTensor propagation.
            stage: Which contract stage mismatched (e.g. "in_dst"/"out_src").
        """
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

    plan_overrides input side only: the contract fields (params/in_src/in_dst/
    out_src/out_dst/out_names) follow one rule — **"unset inherits, set wins"**
    (2026-08-05): ``None`` (unset) or the sentinel ``"auto"`` means
    inherit-from-template (merge only); an explicit value IS the final value,
    including the empty dict ``{}`` (explicit "no sharding / no contract"
    — the ViT ``params={}`` pattern); the sentinel ``"none"`` is a readable
    alias for clearing (``{}`` for params/in_*, ``None`` for out_*). The
    planner resolves sentinels and normalizes ``None`` at merge/insert time;
    a spec inside a finished ShardingPlan always carries concrete dicts
    (params/in_src/in_dst) or Optional dicts (out_*).
    """
    # ── Parameter sharding: submodule path → NamedPlacement ──
    # Default None = "not declared" (merge: inherit); an explicit {} means
    # "this boundary shards nothing" (I/O-stitch-only citizen).
    params: Optional[Dict[str, NamedPlacement]] = None

    # ── Input contract ──
    in_src: Optional[Dict[str, NamedPlacement]] = None
    in_dst: Optional[Dict[str, NamedPlacement]] = None

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

    # TP-local module-instance attributes explicitly declared by the user.
    # None means "not declared" during override merge; [] explicitly clears
    # an inherited glob declaration. The planner normalizes this field into
    # _tp_local_attr_plan after all overrides have been merged.
    tp_divide_attrs: Optional[List[str]] = None

    # Internal planner output. init=False keeps it out of both the public
    # programmatic constructor and the YAML transport interface.
    # Annotated as a string: TpLocalAttrPlan lives in
    # tensor_parallel/head_count.py, which imports resolve_placements from
    # this module — a runtime import here would be circular.
    _tp_local_attr_plan: Optional["TpLocalAttrPlan"] = field(
        default=None, init=False, repr=False, compare=False,
    )

    # D-22 (rowwise bias defer): planner-computed tuple of bias param paths
    # (relative to the boundary module, e.g. ("o_proj.bias",)) whose addition
    # is deferred until AFTER the boundary exit TP reduction (Megatron
    # RowParallelLinear semantics: the bias never enters F.linear inside the
    # region, so the Partial reduce counts it exactly once). Internal — built
    # by ShardingPlanner._finalize_deferred_biases after all overrides are
    # merged (detection is anchored on the FINAL spec declarations, so
    # derived / merge / insert / derive=False specs share one code path);
    # user spec objects never carry this field.
    _deferred_bias_params: Tuple[str, ...] = field(
        default=(), init=False, repr=False, compare=False,
    )

    # ── Boundary flag ──
    is_boundary: bool = True

    # ── Structural flags (user-configurable) ──
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ The user extension-point interfaces: region_dispatch /           │
    # │ local_compute_fn / inner_target / inner_wrapper                  │
    # │                                                                 │
    # │ One axiom governs every boundary: region computation is            │
    # │ dispatch-through by default (under validate: DTensors enter        │
    # │ directly, strategy propagation, real out_src validation); what     │
    # │ cannot dispatch explicitly declares region_dispatch=False          │
    # │ (skeleton black-box: to_local -> local execution ->                │
    # │ declarative re-wrap).                                              │
    # │                                                                 │
    # │ [local-region family] (module level: skeleton unchanged,         │
    # │   content swapped)                                               │
    # │   region_dispatch / local_compute_fn                             │
    # │   The skeleton = _wrap_local_region_forward: boundary            │
    # │   entry/exit stitching + local compute + validate dual-mode      │
    # │   fault tolerance (to_local/_temp_local_params/from_local        │
    # │   re-wrapping), shared by both modes. Whether a module runs      │
    # │   through the skeleton is **derived from a single resolution     │
    # │   chain** (not a stored bool) by _resolve_local_compute_fn:      │
    # │     chain link 1  local_compute_fn (user-defined computation:    │
    # │                   callable, or a factory Target built at apply   │
    # │                   time — e.g. a shipped EP archetype factory       │
    # │                   (recipes.qwen2moe_ep_compute_fn))             │
    # │     chain link 2  region_dispatch=False (the module's own        │
    # │                   forward can't dispatch — it IS the compute)    │
    # │     none present → None (skeleton not used)                      │
    # │                                                                 │
    # │   ★ EXPLICIT INJECTION (since the rework): the shipped EP        │
    # │   compute is NEVER auto-injected. The planner only shards the    │
    # │   expert params ({EP: Shard(0)} + _ep_size/_ep_stack metadata);  │
    # │   the compute must be declared via plan_overrides / direct       │
    # │   spec assignment, WITH an explicit region_dispatch (no          │
    # │   default for injections — the tutorial teaches False for        │
    # │   CP/EP-style comm-carrying injections and why). An EP-sharded   │
    # │   boundary whose chain resolves to None fails fast at apply      │
    # │   time (_preflight_compute_injection). HF-native MoE layouts     │
    # │   (per-expert / batched) get the template's region_dispatch      │
    # │   CLEARED by the planner (their forward is not EP-aware);        │
    # │   custom-named pre-stacked modules (w1/w2/w3) keep False         │
    # │   (EP-aware by construction).                                    │
    # │                                                                 │
    # │ [inner-wrap family] (submodule level: content and wrapping both  │
    # │   replaced)                                                      │
    # │   inner_target / inner_wrapper                                   │
    # │   The mechanism is a generic "locate an inner submodule +        │
    # │   replace its forward": inner_target answers "replace whom",     │
    # │   inner_wrapper answers "replace with what". It is NOT CP-gated  │
    # │   (declaration == application); the shipped reference domain is  │
    # │   CP (K/V all-gather), and the four INNER_WRAPPER_REGISTRY       │
    # │   entries still require an active cp axis (fail-fast otherwise)  │
    # │   — custom wrappers receive cp_mesh=None and own their           │
    # │   semantics.                                                     │
    # │                                                                 │
    # │   ★ EXPLICIT INJECTION (since the rework): NO heuristic          │
    # │   dispatch. inner_wrapper must be declared explicitly: a         │
    # │   INNER_WRAPPER_REGISTRY name ("sdpa_qkv"/"sdpa_hf"/"flex_qkv"/  │
    # │   "flex_hf"), a callable fn(target_module, cp_mesh), or a        │
    # │   Target pointing at a wrapper fn (e.g.                          │
    # │   wrappers.sdpa_hf_cp_wrapper). _needs_cp_attn is now pure    │
    # │   METADATA (template recognition) — it no longer triggers any    │
    # │   injection; an attention boundary under an active cp mesh       │
    # │   without inner_wrapper fails fast at apply time.                │
    # │                                                                 │
    # │ The two families are orthogonal and composable: the same module  │
    # │ may declare both inner_* (wrapping of an inner attention) and    │
    # │ local_* (module-level skeleton).                                 │
    # └─────────────────────────────────────────────────────────────────┘
    #
    # region_dispatch: **the validate execution-mode declaration for the
    #   region computation (whatever its source: the module's own forward or
    #   an injected function)** — one axiom across the whole framework: region
    #   computation is dispatch-through by default; what cannot dispatch
    #   explicitly declares False.
    #   - None (no injection): an ordinary boundary — under validate, DTensors
    #     go straight into the original forward, dispatch-through + real
    #     out_src validation (the axiom's default, not this field's "default
    #     value");
    #   - False (no injection): the module's own forward cannot dispatch
    #     (data-dependent logic inside the forward, e.g. the a2a of an
    #     in-house EP-aware MoE) → takes the local-region skeleton, compute =
    #     the module's own forward (derived and filled by the planner when the
    #     template recognizes such a module, visible in the logs);
    #   - With injection (local_compute_fn / inner_wrapper non-None): **must be
    #     declared explicitly** (no default) — True = the injected code is pure
    #     standard ops, dispatchable (validate dispatch-through + real
    #     out_src/inner_out_src validation); False = the injected code contains
    #     communication / custom kernels (validate black-box local execution +
    #     declarative re-wrap). Missing → fail-fast at apply time;
    #   - True (no injection): a redundant declaration → fail-fast (an
    #     ordinary boundary is inherently dispatch-through).
    #   production is not affected by this field (region is always local
    #   passthrough).
    region_dispatch: Optional[bool] = None

    # ── inner-wrap custom entry points (user-configurable, 05 §4.4.2/§8.6) ──
    # inner_target: **pure location** — names the attribute of the inner
    #   submodule whose forward is to be replaced ("self" means the module
    #   itself). Automatic locating (_resolve_inner_target) fails fast, in which
    #   case the user must specify it via plan_overrides/injections. Declaring
    #   inner_target WITHOUT inner_wrapper is an error since the rework
    #   (location alone cannot pick a scheme — no heuristic dispatch).
    # inner_wrapper: **pure behavior** — selects which scheme wraps the target.
    #   NOT CP-gated: declaration == application, whatever the parallel mode.
    #   - str: a name in the INNER_WRAPPER_REGISTRY registry ("sdpa_qkv"/"sdpa_hf"/
    #     "flex_qkv"/"flex_hf", or a user-registered name); explicitly pins a
    #     built-in scheme; an unknown name fails fast; the four SHIPPED names
    #     are CP schemes and require an active cp axis (fail-fast otherwise);
    #   - Callable: a fully custom wrapper, which MUST be decorated
    #     ``@inner_wrapper`` (injection discipline, see injection.py —
    #     undecorated callables fail fast). Context contract:
    #     fn(target_module, mesh, tp_mesh, cp_mesh, ep_mesh) — the anchor plus
    #     the mesh family are MANDATORY context params, ALL filled by the
    #     framework at apply time (None for inactive axes; the user just uses
    #     them); ``spec`` is the only optional context. The wrapper replaces
    #     target.forward in place (use collectives.flex_cp_allgather for K/V
    #     all-gather). The replaced forward must accept the original
    #     forward's params (validated at apply time). region_dispatch must be
    #     declared explicitly (see above): the injected code either targets
    #     local tensors only (False, the adapter manages the DTensor
    #     conversion) or is pure standard ops, dispatchable (True, validate
    #     dispatch-through with real validation);
    #   - Target (hyper_parallel.trainer.config.Target): a delayed wrapper
    #     reference resolved from YAML; the referenced fn must likewise be
    #     @inner_wrapper decorated (the shipped built-ins
    #     wrappers.sdpa_hf_cp_wrapper etc. satisfy this contract
    #     directly). A None return = in-place replacement done; a callable
    #     return (also decorated) = applied as a custom wrapper.
    inner_target: Optional[str] = None
    inner_wrapper: Optional[Union[str, Callable]] = None   # or Target
    # inner_out_src: the **explicit declaration** of the output placement for
    #   inner-wrap case B (inner_target points at a submodule rather than
    #   self) — the framework does zero derivation and zero guessing about the
    #   inner output layout:
    #   - sentinel "first_input": a layout-preserving declaration (the output
    #     layout == the runtime layout of the first DTensor input argument;
    #     used by attention-type wrappers; illegal for multiple outputs);
    #   - NamedPlacement ({axis: Placement}): an explicit single-output
    #     declaration;
    #   - {name: NamedPlacement}: a per-name declaration for multiple outputs
    #     (tuple positions follow the declaration key order).
    #   Case B without a declaration → fail-fast at apply time (case A
    #   target=self uses the boundary out_src and does not need this field).
    #   validate does no propagation check on the inner region: declaration
    #   errors are backstopped by post-rewrap global-shape consistency / the
    #   boundary out_src check / numerical comparison.
    inner_out_src: Optional[Union[str, Dict]] = None

    # ── local-region custom computation (user-configurable, 05 §4.4.3/§8.6) ──
    # local_compute_fn: the custom compute FACTORY of the local region
    # (chain link 1; single form since 2026-08-10 — the direct compute-fn
    # form was retired, mesh family is always filled, use-it-or-not):
    #   - Callable: a ``@local_compute``-decorated factory
    #     fn(mesh, tp_mesh, cp_mesh, ep_mesh, [module], <config keys...>)
    #     -> compute_fn (programmatic direct pass; injection discipline:
    #     mesh family mandatory, no defaults, no *args/**kwargs);
    #   - Target: the same factory as a delayed reference resolved from
    #     YAML (config keys bound by name; typo fail-fast).
    #   The factory is built ONCE at apply time with the framework context
    #   filled by name (mesh / tp_mesh / cp_mesh / ep_mesh mandatory —
    #   ep_mesh is the same object the expert params were sharded on;
    #   module optional anchor) and must RETURN the region compute fn
    #   fn(module, *local_args) -> Tensor, whose params MUST match the
    #   module's forward params (validated at apply time) — executed on
    #   local tensors inside the _wrap_local_region_forward skeleton (the
    #   skeleton handles boundary entry/exit stitching + validate dual-mode
    #   to_local/_temp_local_params/from_local re-wrapping). Config keys
    #   carry DATA only (function-typed config is rejected by discipline —
    #   custom behavior = your own injected function, e.g. a MoE factory
    #   with its router written inline).
    # Suitable for custom modules that want to reuse the skeleton with their
    # own data-dependent logic: typically a custom MoE (router not in
    # MOE_ROUTER_ADAPTERS / expert layout not using HF-standard naming /
    # hooked to a DeepEP fused dispatcher).
    # Priority: local_compute_fn > region_dispatch=False gate (the module's own
    #   forward) — a single resolution chain (_resolve_local_compute_fn);
    #   declaring it takes effect immediately: **there is no need — and it is
    #   wrong — to also set region_dispatch=False** unless the injected fn is
    #   itself non-dispatchable; the skeleton gate is derived from the
    #   resolution chain (non-None means the skeleton is used).
    # Dual-mode convention: with region_dispatch=False the inputs are always
    #   local tensors and the return value is a local tensor; validate's
    #   DTensor unwrap/re-wrapping is done by the skeleton, so the compute_fn
    #   need not be aware of the mode. With region_dispatch=True the compute
    #   fn must be pure dispatchable ops (validate feeds DTensors straight
    #   through and truly validates out_src).
    local_compute_fn: Optional[Callable] = None   # or factory Target

    # ── Internal flags (set automatically by ShardingPlanner / applier) ──
    _is_terminal: bool = False    # marked automatically during chained propagation
    # _needs_cp_attn: template recognition METADATA (attention module: the
    # inner attention needs a CP-aware forward under an active cp mesh). Since
    # the explicit-injection rework it does NOT trigger any injection — it is
    # used only by the apply-time preflight (an attention boundary under cp>1
    # without inner_wrapper fails fast) and for introspection.
    _needs_cp_attn: bool = False
    # _resolved_inner_wrapper: written back by the applier after resolution (for
    # introspection) — the inner wrapper actually injected:
    # "sdpa_qkv"/"sdpa_hf"/"flex_qkv"/"flex_hf"/"custom"/<target_path>/None.
    _resolved_inner_wrapper: Optional[str] = None
    # _resolved_inner_target: written back by the applier after resolution —
    # the inner-wrap target actually located: the child attribute name, or
    # "self" when the boundary module itself is wrapped. Nothing is located
    # silently: the resolved target is always visible here and in the INFO log.
    _resolved_inner_target: Optional[str] = None
    # D-09 (05 §6.4.7): EP pass-through for HF-native MoE. A non-empty _ep_stack
    # means per-expert parameters must be pre-stacked into [E, ...] in Phase A.
    # Since the explicit-injection rework the compute side is NOT auto-injected:
    # declare local_compute_fn explicitly (e.g. Target → an EP archetype
    # factory such as recipes.qwen2moe_ep_compute_fn).
    _ep_stack: Dict[str, List[str]] = field(default_factory=dict)
    # D-10 (05 §6.4.8): TP-extend-EP parameter-sharding marker. When >0 this is
    # the extended EP group size (= ep_size; the a2a communication domain
    # includes TP ranks); the MoE uses an SP-in identity boundary + a derived
    # expert mesh (edp, ep); expert weights are only Shard(0) along the expert
    # dim. It drives ONLY the parameter sharding/expert-mesh derivation — the
    # compute must be injected explicitly (apply-time preflight fails fast
    # otherwise).
    _ep_size: int = 0


def resolve_placements(
    named: NamedPlacement,
    mesh_dim_names: Tuple[str, ...],
) -> List[Placement]:
    """Arrange placements in mesh_dim_names order, fill missing axes with Replicate()."""
    return [named.get(axis, Replicate()) for axis in mesh_dim_names]


_PLACEMENT_DSL_DOC = (
    '"replicate" / "partial" / "shard(N)" (N is the tensor dim index, e.g. "shard(0)")')


def parse_placement(text: Any, *, path: str = "placement") -> Placement:
    """Parse the YAML placement DSL string into a Placement object.

    Grammar (closed set): ``replicate`` / ``partial`` / ``shard(N)`` — case
    insensitive, whitespace tolerant. Anything else fails fast listing the
    grammar (typo self-discovery, same philosophy as
    ``_check_target_config_keys``).
    """
    if not isinstance(text, str):
        raise ValueError(
            f"{path}: placement must be a string ({_PLACEMENT_DSL_DOC}), "
            f"got {type(text).__name__} {text!r}")
    normalized = re.sub(r"\s+", "", text).lower()
    if normalized == "replicate":
        return Replicate()
    if normalized == "partial":
        return Partial()
    match = re.fullmatch(r"shard\((\d+)\)", normalized)
    if match:
        return Shard(int(match.group(1)))
    raise ValueError(
        f"{path}: cannot parse placement {text!r} — valid grammar: {_PLACEMENT_DSL_DOC}")


def parse_named_placement(raw: Any, *, path: str = "named_placement") -> NamedPlacement:
    """Parse the YAML form ``{axis: placement_str}`` into a NamedPlacement.

    Axis names stay plain strings (str-enum interop; custom mesh dims such as
    "encoder_dp" are legal) — validity against the actual mesh is checked at
    plan-merge time (ShardingPlanner._validate_override_axes), because the
    mesh does not exist at config time.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"{path}: expected a non-empty mapping {{axis name: placement string}}, got {raw!r}")
    named: NamedPlacement = {}
    for axis, placement_text in raw.items():
        if not isinstance(axis, str) or not axis:
            raise ValueError(
                f"{path}: axis name must be a non-empty string, got {axis!r}")
        named[axis] = parse_placement(
            placement_text, path=f"{path}.{axis}")
    return named


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


# ────────────────────────────────────────────────────────────────────────────
# Injection DSL: template decorators for injected functions (the entry-point
# discipline of the explicit injection mechanism).
#
# Design principle: **an injected function is called exactly once at apply
# time, must declare the mesh family (filled by the framework by name; axes
# that are not active are filled with None), and may use them or not**. Two
# decorators cover all injection channels, each with a single canonical form:
#
# - ``@local_compute``: a regional-compute **factory**,
#   ``fn(mesh, tp_mesh, cp_mesh, ep_mesh, [module], <config keys...>)
#   -> compute_fn`` -- built once at apply time;
# - ``@inner_wrapper``: an inner-forward wrapper, ``fn(target_module, mesh,
#   tp_mesh, cp_mesh, ep_mesh)`` -- returns the replacement forward (in-repo
#   discipline; the forward rewriter installs it) or replaces
#   ``target.forward`` in place and returns None (the external contract).
#
# Hard rules enforced by the decorators (fail-fast at import time): required
# context must be declared in full; context parameters must not have default
# values; injected functions forbid *args/**kwargs. The full design rationale
# is documented in docs/design/ (05 §3.7) — the runtime-layer signature
# validators live in ``distributed/_builder/forward_rewriter.py``.
# ────────────────────────────────────────────────────────────────────────────

# Injection kinds (meta.kind)
LOCAL_COMPUTE = "local_compute"        # regional-compute factory (the only form of local_compute_fn)
INNER_WRAPPER = "inner_wrapper"        # inner-forward wrapper

# Reserved framework-context names per kind (declaring one means the
# framework fills it by name; configuring a key with the same name errors)
_MESH_FAMILY = frozenset({"mesh", "tp_mesh", "cp_mesh", "ep_mesh"})
FACTORY_CONTEXT = frozenset({"module"}) | _MESH_FAMILY
WRAPPER_CONTEXT = frozenset({"target_module"}) | _MESH_FAMILY

# Required context per kind (the decorators enforce full declaration --
# the mesh family is always passed by the framework and the user merely
# uses it; the framework fills None for axes that are not active)
FACTORY_REQUIRED = _MESH_FAMILY
WRAPPER_REQUIRED = frozenset({"target_module"}) | _MESH_FAMILY

_ALLOWED_CONTEXT = {
    LOCAL_COMPUTE: FACTORY_CONTEXT,    # optional module anchor + mesh family
    INNER_WRAPPER: WRAPPER_CONTEXT,
}
_REQUIRED_CONTEXT = {
    LOCAL_COMPUTE: FACTORY_REQUIRED,   # mesh family is required
    INNER_WRAPPER: WRAPPER_REQUIRED,
}

_DECORATOR_NAMES = {
    LOCAL_COMPUTE: "@local_compute",
    INNER_WRAPPER: "@inner_wrapper",
}


@dataclass(frozen=True)
class InjectionMeta:
    """Metadata written by the decorators onto the injected function object
    (``fn._injection_meta``).

    Attributes:
        kind: The injection kind (LOCAL_COMPUTE / INNER_WRAPPER).
        context: The declared framework-context keys.
    """
    kind: str                 # LOCAL_COMPUTE / INNER_WRAPPER
    context: frozenset        # declared framework-context keys


def _make_decorator(kind: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Build the template decorator for one injection kind.

    Args:
        kind: The injection kind (LOCAL_COMPUTE or INNER_WRAPPER).

    Returns:
        A decorator that validates the injected function's signature at
        import time and stamps it with :class:`InjectionMeta`.
    """
    allowed = _ALLOWED_CONTEXT[kind]
    required = _REQUIRED_CONTEXT[kind]

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Validate ``fn`` against the injection discipline and stamp its meta.

        Args:
            fn: The injected function to decorate.

        Returns:
            The same function, with ``_injection_meta`` attached.

        Raises:
            TypeError: If the signature cannot be introspected, uses
                *args/**kwargs, gives a context parameter a default value,
                or omits required context parameters.
        """
        fname = getattr(fn, "__name__", fn)
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{_DECORATOR_NAMES[kind]}: cannot introspect the signature of "
                f"{fn!r} (an injected function must be an introspectable "
                "plain callable)") from exc
        context = []
        for name, p in sig.parameters.items():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
                raise TypeError(
                    f"{_DECORATOR_NAMES[kind]} injected function {fname}"
                    f" must not use *args/**kwargs (parameter {name!r}) -- "
                    "the signature of an injected function must be an explicit "
                    "parameter list: context is declared by name and filled by "
                    "the framework, config keys are bound by name, and "
                    "**kwargs would silently swallow typos (conflicting with "
                    "the _check_target_config_keys policy)")
            if name in allowed:
                if p.default is not inspect.Parameter.empty:
                    raise TypeError(
                        f"{_DECORATOR_NAMES[kind]} injected function {fname}: "
                        f"context parameter {name!r} must not have a default "
                        "value -- context names are reserved by the framework "
                        "and are always filled by name, so a default would "
                        "never take effect")
                context.append(name)
        missing = sorted(required - set(context))
        if missing:
            raise TypeError(
                f"{_DECORATOR_NAMES[kind]} injected function {fname} is "
                f"missing required context parameters {missing} -- the "
                "injection discipline requires explicitly receiving "
                f"{sorted(required)} (all filled by the framework by name at "
                "apply time; the user merely uses them)")
        fn._injection_meta = InjectionMeta(kind=kind, context=frozenset(context))
        return fn

    return decorator


local_compute = _make_decorator(LOCAL_COMPUTE)
inner_wrapper = _make_decorator(INNER_WRAPPER)


def require_injection_meta(fn: Callable[..., Any], kind: str, *, source: str) -> InjectionMeta:
    """Fetch an injected function's metadata; fail-fast (with an instructive
    error) if it is undecorated or of the wrong kind.

    Args:
        fn: The function expected to carry injection metadata.
        kind: The expected injection kind.
        source: Call-site label prepended to error messages.

    Returns:
        The :class:`InjectionMeta` attached to ``fn``.

    Raises:
        TypeError: If ``fn`` lacks the template decorator or its decorator
            kind does not match ``kind``.
    """
    meta = getattr(fn, "_injection_meta", None)
    name = getattr(fn, "__name__", fn)
    if meta is None:
        raise TypeError(
            f"{source}: injected function {name} is missing the "
            f"{_DECORATOR_NAMES[kind]} decorator -- the explicit-injection "
            "discipline requires every injected function to carry a template "
            "decorator (the decorator declares the framework context it needs "
            "and guarantees the signature is validatable): use @local_compute "
            "for the runtime fn of local_compute_fn (a regional-compute "
            "factory) and @inner_wrapper for inner_wrapper (exported by "
            "hyper_parallel.distributed)")
    if meta.kind != kind:
        raise TypeError(
            f"{source}: injected function {name} has the wrong decorator "
            f"kind -- got {_DECORATOR_NAMES[meta.kind]}, expected "
            f"{_DECORATOR_NAMES[kind]}")
    return meta
