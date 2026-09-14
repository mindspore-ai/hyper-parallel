# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Freeze a ``ShardingPlan`` into JSON-safe structures for meta / generated code.

The planner's plan objects are live (``Placement`` instances, ``Target``
references, callables, internal ``init=False`` fields).  The generated
modeling file and ``codegen_meta.json`` must consume the SAME plan without
re-running the planner — and meta must stay JSON-serializable.  The freeze
layer is that projection: placements become stable strings / dicts, targets
become ``to_dict()``, internal EP metadata is made explicit, and the full
sharded-parameter tree is expanded to concrete FQNs.

Everything here is a pure function over planner output; nothing imports
torch or the training stack.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Placement serialization (placement_types.py)
# ---------------------------------------------------------------------------

# Registry of known placement types (imported lazily — freeze.py must stay
# importable without torch) mapped to their string-form serializers.  The
# registry stays empty until the first serialization; ``_register_placement_types``
# is the ONLY writer.  The serializer classes are stateless holders for the
# per-kind string logic — they must NOT self-register (a decorated registry
# keyed by the serializer classes would poison the isinstance dispatch below).
_PLACEMENT_SERIALIZERS: dict[type, type] = {}


class _StridedShardSerializer:
    @staticmethod
    def to_string(p) -> str:
        dim = p.dim if not p.uneven_shard else -p.dim - 1
        return f"SS({dim},{p.split_factor})"


class _ShardSerializer:
    @staticmethod
    def to_string(p) -> str:
        dim = p.dim if not p.uneven_shard else -p.dim - 1
        return f"S({dim})"


class _ReplicateSerializer:
    @staticmethod
    def to_string(p) -> str:
        return "R"


class _PartialSerializer:
    @staticmethod
    def to_string(p) -> str:
        return f"P({p.reduce_op})"


def _register_placement_types() -> None:
    """Register the real placement classes against their serializers.

    ``placement_types`` is imported lazily (torch dependency) — the registry
    stays empty until the first actual serialization, so an environment
    without torch can still import ``freeze.py``.

    Order matters: ``StridedShard`` subclasses ``Shard``, so it must be
    registered FIRST or ``isinstance`` dispatch would serialize every
    strided shard as a plain ``Shard``.
    """
    from hyper_parallel.core.dtensor.placement_types import (
        Partial,
        Replicate,
        Shard,
        StridedShard,
    )

    for cls, serializer in (
        (StridedShard, _StridedShardSerializer),
        (Shard, _ShardSerializer),
        (Replicate, _ReplicateSerializer),
        (Partial, _PartialSerializer),
    ):
        _PLACEMENT_SERIALIZERS[cls] = serializer


def placement_to_string(placement: Any) -> str:
    """Serialize a ``Placement`` instance to its canonical string form.

    Shard(dim) -> "S(dim)"; Replicate() -> "R"; Partial(reduce_op) -> "P";
    StridedShard(dim, split_factor) -> "SS(dim,factor)".  ``uneven_shard`` is
    encoded as a negative dim (``Shard(-dim-1)``) so it survives the
    round-trip without an extra field.  Unknown placement types fail fast —
    a silently dropped placement would corrupt the frozen plan.
    """
    if not _PLACEMENT_SERIALIZERS:
        _register_placement_types()
    for cls, serializer in _PLACEMENT_SERIALIZERS.items():
        if isinstance(placement, cls):
            return serializer.to_string(placement)
    raise TypeError(
        f"cannot serialize placement {placement!r} of unknown type "
        f"{type(placement).__name__}"
    )


# ---------------------------------------------------------------------------
# Named-placement serialization
# ---------------------------------------------------------------------------

def named_placement_to_dict(named: Any) -> dict[str, str]:
    """Serialize a ``NamedPlacement`` / output-contract dict to JSON.

    ``in_*`` entries are keyed by input name (``{"hidden_states": ...}``);
    ``out_*`` entries are normalized by the planner into
    ``{"output": ...}`` — both wrap the same axis -> ``Placement``
    dictionary, so the value at every level may itself be a dict and is
    recursed until a leaf ``Placement`` (or a plain scalar) is reached.
    Axis keys are normalized to strings and sorted for stability; a
    ``None`` value (unset) serializes to an empty dict.
    """
    if not named:
        return {}
    return {
        str(getattr(axis, "value", axis)): _value_to_dict(placement)
        for axis, placement in named.items()
    }


# ---------------------------------------------------------------------------
# Placement deserialization (inverse of the serializers above)
# ---------------------------------------------------------------------------

def parse_placement(text: str) -> Any:
    """Inverse of :func:`placement_to_string`: canonical string -> Placement.

    Compact forms: ``S(dim)`` / ``R`` / ``P(reduce_op)`` / ``SS(dim,factor)``;
    ``uneven_shard`` is recovered from the negative-dim encoding
    (``S(-1)`` -> ``Shard(0, uneven_shard=True)``).  Unknown or malformed
    forms fail fast — a silently downgraded placement would corrupt the
    frozen plan (e.g. quietly becoming ``Replicate()``).
    """
    from hyper_parallel.core.dtensor.placement_types import (
        Partial,
        Replicate,
        Shard,
        StridedShard,
    )

    if not isinstance(text, str) or not text:
        raise ValueError(f"cannot parse placement from {text!r}")

    if text == "R":
        return Replicate()

    match = re.fullmatch(r"([SP]|SS)\((.*)\)", text)
    if match is None:
        raise ValueError(
            f"unknown placement form {text!r} (expected S(dim) / R / "
            f"P(reduce_op) / SS(dim,factor))"
        )
    kind, args = match.groups()

    if kind == "P":
        reduce_op = args
        if not reduce_op:
            raise ValueError(f"malformed partial placement {text!r}: missing reduce_op")
        return Partial(reduce_op)

    parts = [part.strip() for part in args.split(",")]
    try:
        dim = int(parts[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"malformed placement {text!r}: invalid dim") from exc
    uneven = dim < 0
    dim = -dim - 1 if uneven else dim

    if kind == "S":
        if len(parts) != 1:
            raise ValueError(f"malformed shard placement {text!r}: too many args")
        return Shard(dim, uneven_shard=uneven)
    if kind == "SS":
        if len(parts) != 2:
            raise ValueError(f"malformed strided shard {text!r}: expected SS(dim,factor)")
        try:
            split_factor = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"malformed strided shard {text!r}: invalid split_factor"
            ) from exc
        return StridedShard(dim, split_factor, uneven_shard=uneven)
    raise ValueError(f"unknown placement kind {kind!r} in {text!r}")


def _value_from_dict(value: Any) -> Any:
    """Inverse of :func:`_value_to_dict`: JSON-safe value -> planner-ish value.

    Placement strings become ``Placement`` objects; nested dicts / lists
    recurse; plain scalars pass through unchanged.  ``Target`` objects
    serialized via ``to_dict()`` are NOT reconstructed — the frozen boundary
    contract is data, and the generated module resolves runtime targets
    itself.
    """
    if isinstance(value, str):
        return parse_placement(value)
    if isinstance(value, dict):
        return {str(getattr(k, "value", k)): _value_from_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_value_from_dict(v) for v in value]
    return value


def parse_named_placement(data: dict[str, Any]) -> Any:
    """Inverse of :func:`named_placement_to_dict`: JSON dict -> named placement.

    Round-trips ``named_placement_to_dict()`` output back to a
    ``NamedPlacement`` / output-contract dict with real ``Placement`` leaves.
    """
    if not data:
        return {}
    return _value_from_dict(data)


# ---------------------------------------------------------------------------
# FrozenPlan
# ---------------------------------------------------------------------------

@dataclass
class FrozenPlan:
    """JSON-safe projection of a ``ShardingPlan`` for meta / generated code.

    - ``param_plan``: per-boundary parameter placements (the sharding
      contract the generated file executes).
    - ``injections``: per-boundary injection declarations (CP wrappers /
      local compute / inner targets), already serialized to strings/dicts.
    - ``replacements``: reserved module-replacement declarations.
    - ``tied_pairs``: tied-weight FQN pairs.
    - ``special_handlers``: special-handler name per parameter.
    - ``frozen_sharded_params``: the full FQN list of every parameter that
      takes a non-``Replicate`` placement anywhere in the plan — what the
      generated file must create as DTensors at load time.
    - ``boundary_classes``: {边界FQN: 类名} — the class that owns each
      boundary's ``forward``. The frozen plan is keyed by
      boundary FQN, but the generated file rewrites ``class.forward``, which
      one class shares across many instances; the emitter groups FQNs by this
      map (failed to resolve a class = a boundary whose forward is never
      rewritten = silent drop).
    - ``mesh_dim_names``: the plan's active axes (``ShardingPlan.
      mesh_dim_names`` verbatim — ``tp``/``cp``/``ep`` with size-1 axes and
      dp* axes already stripped).  The runtime slices the live mesh down to
      these axes (mirroring ``_get_active_mesh``) so a placement never lands
      on a dp axis the plan never shards on.
    """

    param_plan: dict[str, dict[str, Any]] = field(default_factory=dict)
    injections: list[dict[str, Any]] = field(default_factory=list)
    replacements: list[dict[str, Any]] = field(default_factory=list)
    tied_pairs: list[list[str]] = field(default_factory=list)
    special_handlers: dict[str, str] = field(default_factory=dict)
    frozen_sharded_params: list[str] = field(default_factory=list)
    boundary_classes: dict[str, str] = field(default_factory=dict)
    mesh_dim_names: tuple[str, ...] = ()


def freeze_plan(plan: Any, model: Any) -> FrozenPlan:
    """Freeze a planner ``ShardingPlan`` into a ``FrozenPlan``.

    ``model`` is the (meta-device) model the plan was derived from — used to
    expand the sharded-parameter tree to concrete FQNs and to record each
    boundary FQN's owning class (``boundary_classes``). ``model``
    may be ``None`` when only the plan shape is frozen (class names are then
    left unresolvable — the manager passes a live model).
    """
    if plan is None:
        return FrozenPlan()
    return FrozenPlan(
        param_plan=freeze_param_plan(plan),
        injections=freeze_injections(plan),
        replacements=[],
        tied_pairs=freeze_tied_pairs(plan),
        special_handlers=dict(getattr(plan, "special_handlers", {}) or {}),
        frozen_sharded_params=expand_frozen_sharded_params(plan, model),
        boundary_classes=_freeze_boundary_classes(plan, model),
        mesh_dim_names=tuple(getattr(plan, "mesh_dim_names", ()) or ()),
    )


def _freeze_boundary_classes(plan: Any, model: Any) -> dict[str, str]:
    """Map each boundary FQN to the class that owns its ``forward``.

    The generated file rewrites ``class.forward``, which one class shares
    across many instances — but the frozen plan is keyed by boundary FQN.  For
    every ``plan.modules`` key (``model.layers.0.self_attn``), record the type
    of the submodule at that FQN.  ``get_submodule("")`` returns the root
    module and yields the model class itself.
    """
    if model is None:
        return {}
    modules = getattr(plan, "modules", {}) or {}
    result: dict[str, str] = {}
    for fqn in modules:
        cls = type(model.get_submodule(fqn)).__name__
        result[fqn] = cls
    return result


def freeze_param_plan(plan: Any) -> dict[str, dict[str, Any]]:
    """Freeze the per-boundary parameter sharding plan.

    Output shape (one entry per boundary module)::

        {
          "model.layers.0.self_attn": {
            "params": {          # per-parameter placement
              "q_proj.weight": {"tp": "S(0)"},
              "k_proj.weight": {"tp": "S(0)"},
              ...
            },
            "in_src": {"hidden_states": {"tp": "S(-1)"}},   # optional
            ...
          },
          ...
        }

    Non-``Replicate`` placements are the sharding contract; ``Replicate``
    entries are kept too (an explicit "this stays replicated") so the
    generated file never has to guess.  Placement values are always strings
    via :func:`placement_to_string`; parameter names are kept as declared
    (with the ``.weight`` / ``.bias`` suffix).
    """
    modules = getattr(plan, "modules", {}) or {}
    result: dict[str, dict[str, Any]] = {}
    for fqn, spec in modules.items():
        entry: dict[str, Any] = {}
        params = getattr(spec, "params", None)
        if params:
            entry["params"] = {
                name: {
                    str(getattr(axis, "value", axis)): placement_to_string(placement)
                    for axis, placement in named.items()
                }
                for name, named in params.items()
            }
        for field_name in ("in_src", "in_dst", "out_src", "out_dst"):
            value = getattr(spec, field_name, None)
            if value is not None:
                entry[field_name] = named_placement_to_dict(value)
        for flag in ("is_boundary", "region_dispatch", "needs_cp_attn"):
            value = getattr(spec, flag, None)
            if value is not None:
                entry[flag] = value
        result[fqn] = entry
    return result


def freeze_injections(plan: Any) -> list[dict[str, Any]]:
    """Freeze per-boundary injection declarations.

    CP attention wrappers (``inner_wrapper``), local compute regions
    (``local_compute_fn``), and inner targets — everything the generated
    file must attach at apply time.  ``Target`` values are resolved to their
    ``to_dict()`` form; callables are recorded by name (they cannot cross
    the meta boundary — the generated file re-resolves them from the same
    runtime registry).
    """
    modules = getattr(plan, "modules", {}) or {}
    injections: list[dict[str, Any]] = []
    for fqn, spec in modules.items():
        entry: dict[str, Any] = {}
        if getattr(spec, "inner_wrapper", None) is not None:
            entry["inner_wrapper"] = _value_to_dict(spec.inner_wrapper)
        if getattr(spec, "inner_target", None) is not None:
            entry["inner_target"] = spec.inner_target
        if getattr(spec, "inner_out_src", None) is not None:
            entry["inner_out_src"] = _value_to_dict(spec.inner_out_src)
        if getattr(spec, "local_compute_fn", None) is not None:
            entry["local_compute_fn"] = _value_to_dict(spec.local_compute_fn)
        # Preserve expert pre-stacking and extended expert-parallel metadata
        # required when sharding expert parameters.
        if getattr(spec, "_ep_stack", None):
            entry["ep_stack"] = dict(spec._ep_stack)
        if getattr(spec, "_ep_size", 0):
            entry["ep_size"] = spec._ep_size
        if entry:
            injections.append({"match": fqn, **entry})
    return injections


def freeze_tied_pairs(plan: Any) -> list[list[str]]:
    """Freeze tied-weight pairs as stable string lists."""
    pairs = getattr(plan, "tied_pairs", None) or []
    return [list(pair) for pair in pairs]


def expand_frozen_sharded_params(plan: Any, model: Any) -> list[str]:
    """Expand every sharded parameter in the plan to a concrete FQN.

    A parameter is "sharded" when any axis of its placement is a non-
    ``Replicate`` placement (``Shard`` / ``StridedShard`` / ``Partial``).
    The FQN is the module FQN + ``.weight`` / ``.bias`` suffix — the same
    key the generated runtime uses to look up the parameter.
    """
    modules = getattr(plan, "modules", {}) or {}
    fqns: list[str] = []
    seen: set[str] = set()
    for fqn, spec in modules.items():
        for name, named in (getattr(spec, "params", {}) or {}).items():
            if _is_sharded(named):
                fqn_key = f"{fqn}.{name}" if name else fqn
                if fqn_key not in seen:
                    seen.add(fqn_key)
                    fqns.append(fqn_key)
    return sorted(fqns)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_sharded(named: Any) -> bool:
    """Whether any axis of a named placement is non-``Replicate``."""
    if not named:
        return False
    for placement in named.values():
        if placement is None:
            continue
        kind = type(placement).__name__
        if kind in ("Shard", "StridedShard", "Partial"):
            return True
    return False


def _value_to_dict(value: Any) -> Any:
    """Serialize a spec value: Target -> to_dict(), Placement -> string,
    named placement -> dict, callable -> name."""
    from hyper_parallel.core.dtensor.placement_types import Placement

    if isinstance(value, Placement):
        return placement_to_string(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {
            str(getattr(k, "value", k)): _value_to_dict(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_value_to_dict(v) for v in value]
    if callable(value):
        return getattr(value, "__name__", str(value))
    return value


__all__ = [
    "FrozenPlan",
    "expand_frozen_sharded_params",
    "freeze_injections",
    "freeze_param_plan",
    "freeze_plan",
    "freeze_tied_pairs",
    "named_placement_to_dict",
    "parse_named_placement",
    "parse_placement",
    "placement_to_string",
]
