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
"""rule_resolver: plan_overrides merge/insert machinery (05 §3.6.7 + the
unification rework), contract-field normalization, and injection context
filling (``fill_context_kwargs``).

Everything here is a pure (spec, model) → spec transformation: no mesh
objects, no runtime state.
"""

import copy
import fnmatch
import functools
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

from hyper_parallel.core.dtensor.placement_types import Placement
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import (
    INNER_WRAPPER,
    LOCAL_COMPUTE,
    InjectionMeta,
    MeshAxisName,
    ModuleShardingSpec,
    _normalize_out_fields,
    require_injection_meta,
    resolve_placements,
)

logger = logging.getLogger(__name__)


def _last_segment(fqn: str) -> str:
    return fqn.rsplit(".", 1)[-1].lower() if fqn else ""


_GLOB_CHARS = ("*", "?", "[")

def _is_glob_key(key: str) -> bool:
    return any(c in key for c in _GLOB_CHARS)

def _merge_plan_overrides(plan_overrides, plan: ShardingPlan, model, *,
                          derive: bool = True) -> None:
    """Unified override pass, executed before Phase 5 and the D-14 checks.

    Three modes:

    - **merge** (key hits an existing boundary — derived or previously
      inserted): UNSET contract fields (``None``: ``params`` /
      ``in_src`` / ``in_dst`` / ``out_src`` / ``out_dst`` /
      ``out_names``) INHERIT, set ones replace at field granularity —
      **an explicit empty dict {} is a SET value** (explicit "no
      sharding / no contract", 2026-08-05 "unset inherits, written is
      honored"); the
      sentinels ``"auto"``/``"none"`` mean inherit/clear explicitly;
      injection fields
      (``local_compute_fn`` / ``inner_target`` / ``inner_wrapper``
      non-None, ``region_dispatch=False``) always win; internal flags
      (``_ep_size`` / ``_ep_stack`` / ``_needs_cp_attn`` /
      ``_is_terminal``) always inherit — they are
      planner/applier-owned metadata, not user contracts. This is how
      CP/EP compute injection is declared: an injection-fields-only
      spec (``ModuleShardingSpec(local_compute_fn=...)``) inherits the
      whole derived contract;
    - **insert** (exact key misses every boundary): the spec is
      deep-copied and inserted as-is; at least one contract field must
      be declared (an explicit {} counts — the pure I/O-stitch
      boundary) — an override with EVERYTHING unset fails fast
      ("no template/boundary matched"). Sentinels are rejected (nothing
      to inherit/clear). Nesting (ancestor/descendant
      FQNs) is **allowed** since D-14 (05 §13), subject only to the
      param-uniqueness invariant (``_check_param_uniqueness``);
    - **glob keys** (containing ``*``/``?``/``[``): merge-applied to
      every matching boundary (fnmatchcase, ``*`` spans dots); a
      pattern hitting nothing warns loudly. Glob keys never insert.

    Notes:
    - exact keys must exist in the model's ``named_modules`` (typo
      fail-fast; PP scenarios plan each single-part model separately);
    - ``out_src``/``out_dst`` scalar shorthand is normalized;
    - user spec objects are never mutated (merge reads them, insert
      deep-copies them) — plan() can be called repeatedly.
    """
    entries: List[Tuple[str, ModuleShardingSpec, str]] = [
        (fqn, spec, "plan_overrides")
        for fqn, spec in plan_overrides.items()
    ]
    if not entries:
        return

    module_names = {name for name, _ in model.named_modules()}
    for key, user_spec, source in entries:
        if not isinstance(user_spec, ModuleShardingSpec):
            raise TypeError(
                f"{source}[{key!r}] must be a ModuleShardingSpec, "
                f"got {type(user_spec).__name__}"
            )
        _validate_override_axes(key, user_spec, source, plan)
        if not _is_glob_key(key) and key not in module_names:
            raise ValueError(
                f"{source} FQN not found in the model's "
                f"named_modules: {key!r} (check spelling; in PP "
                f"scenarios plan each single-part model separately)"
            )

    for key, user_spec, source in entries:
        if _is_glob_key(key):
            hits = [fqn for fqn in plan.modules
                    if fnmatch.fnmatchcase(fqn, key)]
            if not hits:
                logger.warning(
                    "%s match=%r hit no boundary spec — check the "
                    "spelling (plan boundaries: %s)",
                    source, key, sorted(plan.modules)[:8])
                continue
            for fqn in hits:
                _warn_dropped_params(
                    source, key, fqn, plan.modules[fqn], user_spec)
                _merge_into(plan.modules[fqn], user_spec)
                logger.info("%s: merge into %s (glob %r)",
                            source, fqn, key)
        elif key in plan.modules:
            _warn_dropped_params(
                source, key, key, plan.modules[key], user_spec)
            _merge_into(plan.modules[key], user_spec)
            logger.info("%s: merge into the spec of module %s",
                        source, key)
        else:
            _insert_spec(plan, key, user_spec, source, model,
                              derive=derive)

_CONTRACT_FIELDS = ("params", "in_src", "in_dst",
                    "out_src", "out_dst", "out_names")
# String sentinels accepted on the plan_overrides input side for contract
# fields (resolved at merge time only; they never reach the plan output):
#   "auto" — explicitly inherit the derived value (derive per template;
#            synonymous with the default unset value, self-documenting)
#   "none" — explicitly clear (params/in_src/in_dst → {}, out_* → None)
_CONTRACT_SENTINELS = ("auto", "none")

def _iter_named_placements(spec: ModuleShardingSpec):
    """Yield (attr, name, named) for every concrete NamedPlacement in an
    override spec (skips sentinel strings/None/empty; out_* scalar
    shorthand yields the whole field as one NamedPlacement)."""
    for attr in ("params", "in_src", "in_dst", "out_src", "out_dst"):
        value = getattr(spec, attr)
        if not value or isinstance(value, str):
            continue
        if not all(isinstance(v, dict) for v in value.values()):
            yield attr, "output", value          # out_* scalar shorthand
        else:
            for name, named in value.items():
                yield attr, name, named

def _validate_override_axes(key, user_spec, source, plan) -> None:
    """Fail fast on typo'd placement axes / non-Placement values.

    ``resolve_placements`` fills missing axes with Replicate() — so a
    typo'd axis (``{"tp2": Shard(0)}``) would otherwise be silently
    IGNORED. Allowed axes = the plan's mesh dims ∪ the canonical
    ``MeshAxisName`` values (canonical-but-absent axes, e.g. CP
    placements declared on a tp-only mesh, are tolerated — templates
    declare all canonical dims and resolve_placements picks the mesh's
    subset; "ep" is the virtual TP-extend-EP axis). Anything outside
    both sets is a typo → fail fast. Placement values must already be
    Placement objects (the YAML string DSL is parsed at desugar time).
    """
    allowed = ({str(a) for a in plan.mesh_dim_names}
               | {axis.value for axis in MeshAxisName})
    for attr, name, named in _iter_named_placements(user_spec):
        for axis, placement in named.items():
            if not isinstance(placement, Placement):
                raise TypeError(
                    f"{source}[{key!r}]: the value of axis "
                    f"{axis!r} in contract field {attr}[{name!r}] must "
                    f"be a Placement object (Shard(N)/"
                    f"Replicate()/Partial(); the YAML string DSL is "
                    f"parsed into objects at desugar time), "
                    f"got {type(placement).__name__} "
                    f"{placement!r}")
            # MeshAxisName is a str subclass (hash/eq consistent with a
            # plain str), so use the raw value for the membership test —
            # do NOT str(axis) (the enum __str__ would produce
            # "MeshAxisName.TP")
            if axis not in allowed:
                raise ValueError(
                    f"{source}[{key!r}] contract field "
                    f"{attr}[{name!r}] uses the unknown axis {axis!r} — "
                    f"legal axes = mesh axes "
                    f"{sorted(str(a) for a in plan.mesh_dim_names)} ∪ "
                    f"canonical axes "
                    f"{sorted(axis.value for axis in MeshAxisName)}. "
                    f"An unknown axis is silently ignored by "
                    f"resolve_placements, so fail fast (suspected typo)")

def _merge_contract_field(derived: ModuleShardingSpec,
                          user_spec: ModuleShardingSpec, attr: str) -> None:
    """Merge one contract field: "unset inherits, written is honored" (2026-08-05).

    Precedence: ``None`` (unset) / ``"auto"`` → inherit derived;
    ``"none"`` → explicit clear (a readable alias for the empty value);
    a concrete dict — **including the empty dict {}** — replaces at field
    granularity ({} = explicit "no sharding / no contract", the ViT
    ``params={}`` pattern).
    """
    value = getattr(user_spec, attr)
    if isinstance(value, str):
        if value == "auto":
            return                      # explicit inherit (same as the default unset)
        if value == "none":
            setattr(derived, attr,
                    {} if attr in ("params", "in_src", "in_dst") else None)
            return                      # explicit clear
        raise ValueError(
            f"the string value of plan_overrides contract field {attr} "
            f"only accepts the sentinels "
            f"{_CONTRACT_SENTINELS} ('auto'=inherit the "
            f"template-derived value, 'none'=explicit clear), "
            f"got {value!r}")
    if value is None:                   # undeclared → inherit
        return
    setattr(derived, attr, copy.deepcopy(value))  # including {}: explicit empty (clear)

def _warn_dropped_params(source, key, fqn, derived, user_spec) -> None:
    """Visibility safeguard: a field-granularity replacement of ``params``
    during merge strips the derived sharding from every parameter not
    covered by the override (they stay replicated) — possibly an
    unintended typo, so a WARNING lists the dropped entries."""
    user_params = user_spec.params
    if (isinstance(user_params, dict) and user_params
            and derived.params):
        dropped = sorted(set(derived.params) - set(user_params))
        if dropped:
            logger.warning(
                "%s[%r] merge into %s: the field-granularity params "
                "replacement strips the derived sharding from %d "
                "parameter(s), which will stay replicated: %s — if "
                "unintended, write the derived values in too "
                "(field-granularity replacement, no per-key merge); if "
                "the de-sharding is intentional, ignore this warning "
                "(params={} or 'none' explicitly clears all)",
                source, key, fqn, len(dropped), dropped)

def _merge_into(derived: ModuleShardingSpec,
                user_spec: ModuleShardingSpec) -> None:
    """Merge one user spec into an existing boundary spec (in place).

    Contract fields: None/"auto" → inherit, "none" → clear, concrete
    dict (including {}) → replace (field granularity); injection fields
    win when set; internal flags always inherit from *derived*.
    """
    for attr in _CONTRACT_FIELDS:
        _merge_contract_field(derived, user_spec, attr)
    for attr in ("local_compute_fn", "inner_target", "inner_wrapper",
                 "inner_out_src", "region_dispatch", "tp_divide_attrs"):
        value = getattr(user_spec, attr)
        if value is not None:
            setattr(derived, attr, value)
    _normalize_out_fields(derived)


def _normalize_contract_fields(plan: ShardingPlan) -> None:
    """Plan output normalization: None (undeclared) of
    params/in_src/in_dst → {}.

    "Unset inherits, written is honored" is input-side semantics; specs
    inside the plan always hold concrete values, so downstream consumers
    (the applier, the D-14 checks, etc.) need no None branches.
    """
    for spec in plan.modules.values():
        for attr in ("params", "in_src", "in_dst"):
            if getattr(spec, attr) is None:
                setattr(spec, attr, {})


def _suggest_insert_skeleton(model, fqn: str) -> str:
    """Derive a draft contract skeleton from the module's forward
    signature (input names) and direct parameters — turns "write a
    contract from scratch" into "edit a draft". Best-effort: degrades
    to a generic skeleton when the module/signature is unavailable."""
    try:
        module = dict(model.named_modules()).get(fqn)
    except Exception:  # pylint: disable=broad-exception-caught  # best-effort hint only
        module = None
    in_names = ["hidden_states"]
    param_names: List[str] = []
    if module is not None:
        try:
            sig = inspect.signature(module.forward)
            names = [
                p.name for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.KEYWORD_ONLY)
                and p.default is inspect.Parameter.empty  # required input parameters
            ]
            if names:
                in_names = names
        except (TypeError, ValueError):
            pass
        param_names = [n for n, _ in module.named_parameters(recurse=False)]
    axis = "tp"   # draft placeholder axis — adjust to the actual topology
    lines = [f'  - match: "{fqn}"']
    if param_names:
        lines.append("    params:  # choose a shard dim per parameter "
                     "(explicit {} = this boundary shards no params)")
        lines.extend(f"      {n}: {{{axis}: \"shard(0)\"}}"
                     for n in param_names)
    else:
        lines.append("    params: {}   # no direct parameters / pure "
                     "I/O-stitch boundary")
    in_entries = ", ".join(f'{n}: {{{axis}: "shard(1)"}}'
                           for n in in_names)
    lines.append(f"    in_src:  {{{in_entries}}}   "
                 "# entry status quo = upstream exit layout")
    lines.append(f"    in_dst:  {{{in_entries}}}   "
                 "# differing from in_src inserts communication")
    lines.append(f'    out_src: {{output: {{{axis}: "replicate"}}}}'
                 "   # multi-output modules: use multiple keys and add "
                 "out_names")
    lines.append(f'    out_dst: {{output: {{{axis}: "replicate"}}}}')
    return "\n".join(lines)


def _insert_spec(plan: ShardingPlan, fqn: str,
                 user_spec: ModuleShardingSpec, source: str,
                 model=None, derive: bool = True) -> None:
    """Insert a fully self-declared spec for a non-derived boundary."""
    for attr in _CONTRACT_FIELDS:
        value = getattr(user_spec, attr)
        if isinstance(value, str):
            reason = (
                "derive=False: template derivation is disabled "
                "entirely, so the plan holds no derived values — "
                if not derive else
                "insert (no derived boundary was hit) — ")
            raise ValueError(
                f"{source}[{fqn!r}] contract field {attr}={value!r} is "
                f"meaningless: {reason}"
                "the 'auto' (inherit the derivation) / 'none' (clear "
                "the inherited value) sentinels only apply to derived "
                "boundaries hit by a merge; with no derived value there "
                "is nothing to inherit/clear. Declare explicitly: a "
                "concrete dict for the sharding/contract, an explicit "
                "empty {} for a boundary that shards no params (params "
                "stay replicated) / has no such contract")
    if all(getattr(user_spec, attr) is None for attr in
           ("params", "in_src", "in_dst", "out_src", "out_dst")):
        hint = ""
        if model is not None:
            hint = (
                "\nSuggested draft (derived from the module forward "
                "signature/direct parameters; placements are "
                "placeholders — fix them per the layout semantics):\n"
                + _suggest_insert_skeleton(model, fqn))
        derive_note = (
            "derive=False has disabled template derivation — every "
            "override is an insert and there are no derived values to "
            "inherit; " if not derive else "")
        raise ValueError(
            f"{source}[{fqn!r}] hit no planner-derived boundary and "
            "leaves params and the I/O contract all undeclared — "
            + derive_note +
            "empty-field inheritance (merge) only applies to derived "
            "boundaries; inserting a new boundary requires a fully "
            "self-declared contract (05 §3.6.7 / D-14; an explicit "
            "empty {} is also a valid declaration, e.g. the pure "
            "I/O-stitch boundary with params={}), or check the fqn "
            "spelling"
            + hint)
    spec = copy.deepcopy(user_spec)
    _normalize_out_fields(spec)
    logger.info("%s: insert the spec of module %s", source, fqn)
    plan.modules[fqn] = spec


def fill_context_kwargs(meta: InjectionMeta,
                        context: Dict[str, Any],
                        configured: Optional[Dict[str, Any]] = None,
                        *,
                        source: str = "") -> Dict[str, Any]:
    """Collect the framework-filled kwargs for the context keys declared in
    ``meta`` (minimal, no hidden behavior).

    - The filled set equals the declared set, no more and no less (each fill
      is logged at INFO);
    - Context keys are framework-reserved names: a user configuring a
      same-named key in Target/YAML triggers fail-fast (context parameters
      have no defaults and the YAML resolver never back-fills them, so any
      reserved key appearing in ``configured`` was definitely written
      explicitly by the user).

    Args:
        meta: The injection metadata whose declared context keys are filled.
        context: The framework-provided context values keyed by name.
        configured: User-configured keys, checked against reserved names.
        source: Call-site label prepended to log and error messages.

    Returns:
        The kwargs dict mapping each declared context key to its
        framework-provided value.

    Raises:
        ValueError: If ``configured`` contains framework-reserved context
            keys.
    """
    configured = configured or {}
    reserved = sorted(set(configured) & meta.context)
    if reserved:
        raise ValueError(
            f"{source}: framework-reserved context keys {reserved} were "
            "configured -- context is filled by the framework according to "
            "the declaration and is not configurable; your config keys must "
            "not collide with reserved names")
    kwargs = {}
    for key in sorted(meta.context):
        kwargs[key] = context[key]
        logger.info("%s: context key %s filled by framework (%s)",
                    source, key, type(context[key]).__name__)
    return kwargs


# ────────────────────────────────────────────────────────────────────────────
# Injection resolution: spec/Target/registry → concrete callables and output
# rules (pure resolution — no module mutation)
# ────────────────────────────────────────────────────────────────────────────

def _is_delayed_target(obj) -> bool:
    """Duck-typed check for a config Target (avoids a components -> trainer
    import): a Target carries ``build()`` and ``_target_`` and is itself NOT
    the compute/wrapper callable — it must be built at apply time."""
    return hasattr(obj, "build") and hasattr(obj, "_target_")


def _check_target_config_keys(target, kind):
    """Fail fast on configured Target kwargs the callable would never bind.

    Target kwargs are bound BY NAME at build time
    (``{**configured, **runtime}`` -> ``fn(**kwargs)``). When the target
    callable accepts VAR_KEYWORD (e.g. ``**_context``, there to tolerate the
    framework-filled generic context), a misspelled configured key is
    swallowed SILENTLY — the user's value never takes effect and the
    framework may even auto-fill the intended parameter instead. Guard: any
    configured key that is not an explicitly declared (keyword-bindable)
    parameter fails fast with the valid parameter names. A callable whose
    signature cannot be introspected skips the check (the call itself will
    surface any mismatch).
    """
    configured = getattr(target, "_kwargs", None)
    if not configured:
        return
    fn = getattr(target, "_target_", None)
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return
    bindable = {
        name for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = sorted(set(configured) - bindable)
    if unknown:
        raise ValueError(
            f"{kind} Target {getattr(target, '_target_path', fn)!r} "
            f"configures undeclared keys {unknown} — Target kwargs are "
            f"bound by name to the keyword parameters of the target "
            f"callable; these keys are not in the explicit parameter list "
            f"of {getattr(fn, '__name__', fn)} and would be silently "
            f"swallowed by **kwargs without taking effect (suspected "
            f"typo). Valid parameters: {sorted(bindable) or '(none)'}")


def _require_region_dispatch(spec, *, source):
    """Injection discipline: declaring an injection (local_compute_fn /
    inner_wrapper) requires an explicit region_dispatch (no default — pass
    True for a dispatchable pure-ops injection, False for one containing
    communication primitives / custom kernels; the tutorials and examples
    explain the reasons case by case)."""
    rd = getattr(spec, "region_dispatch", None)
    has_injection = (getattr(spec, "local_compute_fn", None) is not None
                     or getattr(spec, "inner_wrapper", None) is not None)
    if rd is None:
        if has_injection:
            raise ValueError(
                f"{source}: an injection is declared but region_dispatch "
                "is not explicitly declared (no default) — if the "
                "injected fn is pure standard ops that validate can "
                "dispatch through (fused-op/scripting-style optimizations) "
                "→ region_dispatch=True (in-region strategy propagation + "
                "true out_src validation); if it contains communication "
                "primitives/custom kernels (CP K/V all-gather, EP "
                "all-to-all, quantized GEMM, etc.) → "
                "region_dispatch=False (skeleton/adapter black-box hosted "
                "local execution + declarative rewrap)")
        return
    if rd is True and not has_injection:
        raise ValueError(
            f"{source}: region_dispatch=True but no injection is declared "
            "— an ordinary boundary's forward dispatches through natively "
            "(the axiomatic default), so this declaration is redundant; "
            "remove it")


def _build_local_compute_factory(factory, module, mesh, mesh_dim_names,
                                 expert_mesh, *, configured=None, source):
    """Build a ``@local_compute`` factory into the region compute_fn (apply-time).

    The factory is invoked ONCE with the framework context filled BY NAME per
    its declared context (``meta.context``): the mesh family ``mesh`` /
    ``tp_mesh`` / ``cp_mesh`` / ``ep_mesh`` (mandatory declarations; None when
    the axis is inactive), plus the optional anchor ``module``. Context keys
    are RESERVED — a user-configured same-name key fails fast
    (fill_context_kwargs); every fill is logged at INFO. Behavior choices
    (routing, layouts, ...) are written INTO the factory — config keys carry
    data only, never functions.
    The returned compute fn is validated against the module's forward
    signature (validate_local_compute_signature: params must match the
    original forward) and bound to *module* by the caller.
    """
    # Lazy imports: applier and forward_rewriter both import rule_resolver
    # at module level — importing them here at module level would cycle.
    from hyper_parallel.distributed._builder.applier import (  # pylint: disable=C0415
        _get_cp_submesh,
        _get_tp_submesh,
    )
    from hyper_parallel.distributed._builder.forward_rewriter import (  # pylint: disable=C0415
        validate_local_compute_signature,
    )
    meta = require_injection_meta(factory, LOCAL_COMPUTE, source=source)
    context = {
        "module": module,
        "mesh": mesh,
        "tp_mesh": _get_tp_submesh(mesh, mesh_dim_names),
        "cp_mesh": _get_cp_submesh(mesh, mesh_dim_names),
        "ep_mesh": expert_mesh,
    }
    build_kwargs = fill_context_kwargs(
        meta, context, configured or {}, source=source)
    # {**configured, **context}: same binding order as Target.build (context
    # keys are reserved names; fill_context_kwargs already rejects same-name
    # keys in configured)
    compute_fn = factory(**{**(configured or {}), **build_kwargs})
    if not callable(compute_fn):
        raise TypeError(
            f"{source} returned {type(compute_fn).__name__}, not callable — "
            "the @local_compute factory must return the region compute fn "
            "fn(module, *local_args) (e.g. hyper_parallel.distributed.expert_parallel."
            "recipes.qwen2moe_ep_compute_fn)")
    validate_local_compute_signature(
        compute_fn, module.forward,
        owner=f"{source!r} on {type(module).__name__}")
    return compute_fn


def _resolve_local_compute_fn(module, spec, mesh, mesh_dim_names,
                              expert_mesh):
    """Resolve the compute_fn of the local region (**single resolution chain**, 05 §4.4.3).

    Whether a module takes the local-region skeleton is derived by this chain
    (a non-None return means it does) -- the gate is not a stored bool but the
    resolution result. Since the explicit-injection rework the built-in EP
    auto-injection link is REMOVED; the remaining sources:
    1. spec.local_compute_fn: a user-defined FACTORY — single form since
       2026-08-10 (the direct compute-fn form was retired: every injection
       fn is invoked once at apply time with the mesh family filled by the
       framework, use-it-or-not being the user's choice — same discipline as
       @inner_wrapper). Accepted shapes: a ``@local_compute``-decorated
       factory callable (programmatic direct pass), or a Target wrapping one
       (config keys / YAML ``_target_`` reference); both are built at apply
       time by _build_local_compute_factory (e.g. a shipped EP archetype
       factory such as recipes.qwen2moe_ep_compute_fn) — undecorated
       functions fail fast
       (injection discipline); the returned compute fn's params are
       validated against the module's forward (params must match the
       original function). Declaring it REQUIRES an explicit
       ``region_dispatch`` (no default — True: dispatchable pure-ops
       injection, validate dispatches through it and truly validates
       out_src; False: comm/custom-kernel injection, the skeleton runs it
       as a black box on local tensors);
    2. spec.region_dispatch=False without inner_wrapper: the module's own
       forward cannot dispatch (it IS the data-dependent logic, e.g. an
       EP-aware in-house MoE with the a2a already inside forward) — the
       skeleton runs it on local tensors. An explicit inner_wrapper owns the
       local computation instead and therefore does not select this path;
    3. none of the above -> None (ordinary module; takes the
       validate/production path — and an EP-sharded boundary hitting this
       was already failed fast by _preflight_compute_injection).
    """
    custom = getattr(spec, "local_compute_fn", None)
    if custom is not None:
        _require_region_dispatch(spec, source="spec.local_compute_fn")
        if _is_delayed_target(custom):
            _check_target_config_keys(custom, "local_compute_fn")
            factory = getattr(custom, "_target_", None)
            source = (f"local_compute_fn factory "
                      f"{getattr(custom, '_target_path', custom)}")
            configured = getattr(custom, "_kwargs", {})
        else:
            factory = custom
            source = "spec.local_compute_fn"
            configured = None
        compute_fn = _build_local_compute_factory(
            factory, module, mesh, mesh_dim_names, expert_mesh,
            configured=configured, source=source)
        return functools.partial(compute_fn, module)
    if spec.region_dispatch is False and spec.inner_wrapper is None:
        return module.forward
    return None


def _resolve_inner_target(module, spec=None):
    """Resolve the inner-wrap target -- pure location resolution.

    ``inner_target`` is MANDATORY when ``inner_wrapper`` is declared (the
    pairing is enforced in _resolve_inner_wrapper before this is called):
    "self" means the boundary module itself, otherwise resolved by attribute
    name -- fail-fast if the attribute does not exist or has no forward (a
    typo must not silently degrade). The attention-domain auto-location
    heuristic (inner_attention/attn/attention attributes, class-name
    matching, q/k/v_proj structural fallback) was REMOVED (2026-08-10): the
    inner-wrap mechanism is generic (any module, any submodule), and a
    silently located target is a silent wrong-target hazard.
    """
    explicit = getattr(spec, "inner_target", None) if spec is not None else None
    if explicit is None:
        raise ValueError(
            "inner_target is not declared — declaring inner_wrapper "
            "requires an explicit paired inner_target (\"self\" or a "
            "submodule attribute name; the auto-location heuristic was "
            "removed)")
    if explicit == "self":
        return module
    inner = getattr(module, explicit, None)
    if inner is not None and hasattr(inner, "forward"):
        return inner
    raise ValueError(
        f"spec.inner_target={explicit!r} did not match anything on "
        f"{type(module).__name__} (attribute missing or has no forward) "
        f"-- check the spelling in plan_overrides")


def _resolve_inner_wrapper(module, spec, cp_mesh, mesh, tp_mesh=None,
                           ep_mesh=None):
    """Resolve one explicit inner-wrapper declaration without mutating modules.

    Resolution accepts a delayed Target, a decorated callable, or a registry
    name and returns ``(name, target, apply_fn)``. ``inner_target`` and
    ``inner_wrapper`` must be declared together; absent declarations return
    ``None``. The mesh family is framework-filled for every wrapper.
    """
    # Lazy import: forward_rewriter imports rule_resolver at module level.
    from hyper_parallel.distributed._builder.forward_rewriter import (  # pylint: disable=C0415
        _apply_custom_inner_wrapper,
        _classify_rewrite_result,
    )
    # Lazy import: wrappers imports forward_rewriter, which imports this
    # module at module level (circular import).
    from hyper_parallel.distributed.context_parallel.wrappers import (  # pylint: disable=C0415
        INNER_WRAPPER_REGISTRY,
    )
    custom = getattr(spec, "inner_wrapper", None) if spec is not None else None
    inner_target = getattr(spec, "inner_target", None) if spec is not None else None
    if custom is None:
        if inner_target is not None:
            raise ValueError(
                f"spec.inner_target={inner_target!r} only locates "
                "(replace whom) — after the rework the wrapping scheme is "
                "no longer chosen heuristically, so inner_wrapper must be "
                "declared explicitly too: a registry name "
                f"{sorted(INNER_WRAPPER_REGISTRY)}, an @inner_wrapper "
                "decorated callable, or a Target pointing to an in-repo "
                "reference implementation "
                "(hyper_parallel.distributed.context_parallel.wrappers.*)")
        return None

    _require_region_dispatch(spec, source="spec.inner_wrapper")
    if inner_target is None:
        raise ValueError(
            f"inner_wrapper={custom!r} is declared but inner_target is "
            "not — the attention-domain auto-location heuristic was "
            "removed (inner-wrap is a generic mechanism; silently "
            "locating risks wrapping the wrong target). The two fields "
            "must be declared as an explicit pair: to wrap the boundary "
            "module itself → inner_target=\"self\"; to wrap a submodule "
            "→ inner_target=\"<attribute name>\"")
    target = _resolve_inner_target(module, spec)
    context = {
        "target_module": target,
        "mesh": mesh,
        "tp_mesh": tp_mesh,
        "cp_mesh": cp_mesh,
        "ep_mesh": ep_mesh,
    }
    if _is_delayed_target(custom):
        fn = getattr(custom, "_target_", None)
        source = (f"inner_wrapper Target "
                  f"{getattr(custom, '_target_path', custom)}")
        meta = require_injection_meta(fn, INNER_WRAPPER, source=source)
        _check_target_config_keys(custom, "inner_wrapper")

        def _apply_target():
            build_kwargs = fill_context_kwargs(
                meta, context, getattr(custom, "_kwargs", {}), source=source)
            result = custom.build(**build_kwargs)
            if result is None:
                return None   # in-place forward replacement (registry-style fn)
            if callable(result):
                if getattr(result, "_injection_meta", None) is not None:
                    # An @inner_wrapper-stamped custom wrapper: invoke it with
                    # its declared context (it replaces in place or returns
                    # its own replacement).
                    return _apply_custom_inner_wrapper(result, context)
                # 4d contract: the built-in wrapper already ran and RETURNED
                # the replacement forward — hand it to the rewriter.
            # Validate every supported rewriter result here as well, so a
            # delayed Target accepts atomic _ForwardRewriteRequest objects
            # while retaining the resolver's fail-fast behavior for invalid
            # return values.
            _classify_rewrite_result(result, target, source)
            return result

        name = getattr(custom, "_target_path", None) or "custom"
        return (name, target, _apply_target)

    if callable(custom):
        require_injection_meta(
            custom, INNER_WRAPPER, source="spec.inner_wrapper")
        return ("custom", target,
                lambda: _apply_custom_inner_wrapper(custom, context))

    if isinstance(custom, str):
        fn = INNER_WRAPPER_REGISTRY.get(custom)
        if fn is None:
            raise ValueError(
                f"inner_wrapper={custom!r} is not registered in "
                f"INNER_WRAPPER_REGISTRY (available: {sorted(INNER_WRAPPER_REGISTRY)})"
                f" -- check the spelling, or first register "
                f"INNER_WRAPPER_REGISTRY[{custom!r}] = your_fn")
        meta = require_injection_meta(
            fn, INNER_WRAPPER, source=f"INNER_WRAPPER_REGISTRY[{custom!r}]")
        context["target_module"] = target
        return (custom, target,
                lambda: fn(**{k: context[k] for k in sorted(meta.context)}))

    raise TypeError(
        f"inner_wrapper must be a registry name (str), an @inner_wrapper "
        f"decorated callable, or a Target — got {type(custom).__name__}")


def _resolve_inner_output_placements(
    spec,
    boundary_module,
    target,
    mesh_dim_names,
    wrapper_name,
):
    """Resolve the declared placement rule for an injected inner forward."""
    if target is boundary_module:
        if spec is None or not spec.out_src:
            return False, None
        out_names = list(spec.out_names or spec.out_src.keys())
        missing = [name for name in out_names if name not in spec.out_src]
        if missing:
            raise ValueError(
                f"inner_wrapper {wrapper_name!r}: out_names {missing} "
                "have no declaration in out_src — a multi-output contract "
                "must be declared name by name"
            )
        return False, [
            tuple(resolve_placements(spec.out_src[name], mesh_dim_names))
            for name in out_names
        ]

    declared = getattr(spec, "inner_out_src", None) if spec is not None else None
    if declared is None:
        raise ValueError(
            f"inner_wrapper {wrapper_name!r} targets an inner submodule "
            f"of {type(boundary_module).__name__}, but inner_out_src is "
            "not declared — the framework infers and guesses nothing "
            "about inner output layouts. Choose one of: "
            "① a layout-preserving wrapper sets "
            "inner_out_src: \"first_input\"; ② declare the placements "
            "explicitly; ③ or use inner_target=\"self\" to reuse the "
            "boundary out_src contract"
        )
    if isinstance(declared, str):
        if declared != "first_input":
            raise ValueError(
                "the string value of inner_out_src only accepts the "
                f"sentinel 'first_input', got {declared!r}"
            )
        return True, None
    if all(isinstance(value, Placement) for value in declared.values()):
        return False, [tuple(resolve_placements(declared, mesh_dim_names))]
    return False, [
        tuple(resolve_placements(named, mesh_dim_names))
        for named in declared.values()
    ]


def _inner_target_name(attn_module, target) -> str:
    """Readable name of the located inner-wrap target: child attribute name,
    or "self" when the boundary module itself is the target."""
    if target is attn_module:
        return "self"
    for child_name, child in attn_module.named_children():
        if child is target:
            return child_name
    return type(target).__name__
