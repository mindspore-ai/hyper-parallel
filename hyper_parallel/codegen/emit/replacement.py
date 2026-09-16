# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Compile ``replace_module`` overrides against the generation-time meta model.

The trainer's hf backend applies ``plan_overrides.replace_module`` by
desugaring each entry into a :class:`ModuleReplacementSpec` and running
``apply_module_replacements`` on the live model.  Generation must reject the
same bad rules up front: the manager compiles the spec against the
generation-time meta model so conflicts, type mismatches and unmatched
patterns fail here rather than at runtime inside the built model, and records
the resulting FQN set in ``meta.module_overrides``.

The inline pipeline lowers the surviving replacements into the generated
file's module bodies, and ``meta.covered["module_overrides"]`` tells the
trainer to skip its own replacement step.
"""
from __future__ import annotations

import fnmatch
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# generation-time compile (fail-fast)
# ---------------------------------------------------------------------------

def compile_overrides_for_meta(
    meta_model: Any,
    specs: Sequence[Any],
    *,
    factory_paths: Sequence[str | None] = (),
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Match replacement specs against the generation-time meta model.

    Runs :func:`compile_module_replacements` on the meta (empty-weights) model
    so a conflicting or type-mismatched replacement rule fails here, at
    generation time, rather than at runtime inside the built model.  Returns
    ``(records, skipped)``:

    * ``records`` — one JSON-safe record per matched target;
    * ``skipped`` — one ``{"match": [...], "reason": "no_match"}`` dict per
      rule whose patterns matched no module.  ``compile_module_replacements``
      itself raises on an unmatched pattern; this wrapper partitions the rules
      so the manager can record a *skipped* override instead of
      failing the whole generation, matching the runtime path's tolerance.

    A record carries:

    .. code-block:: python

       {
           "match": ["model.layers.0.mlp"],       # spec.match, the patterns
           "fqn":  "model.layers.0.mlp",          # first aliased FQN
           "fqns": ["model.layers.0.mlp"],        # every registered alias
           "module_type": "<dotted path>",        # source module type
           "factory": "<dotted path>",            # the @module_replacement fn
           "exact_type": False,
       }

    The ``factory`` (and ``module_type``) are stored as import paths, not the
    live objects — the replacement factory is a per-entry closure that cannot
    be written as a Python literal, and the runtime re-imports the *raw*
    decoration at those paths to rebuild equal specs.
    """
    if not specs:
        return (), ()
    from hyper_parallel.models.replacement import compile_module_replacements

    # ``compile_module_replacements`` raises when any pattern matches nothing,
    # but a no-match at generation time is a *skippable* override: the rule
    # simply does not apply to this model's topology, so we
    # record it as skipped rather than fail the generation.  Partition the
    # rules up front by testing each pattern against the module-alias tree the
    # compile would use; non-matching specs are dropped before compiling so the
    # type/conflict checks still fail fast on the rules that DO match.
    aliases = {
        fqn for _, fqn in _module_aliases(meta_model)
    }
    paths_by_spec = {
        id(spec): path
        for spec, path in zip(specs, factory_paths)
        if path is not None
    }
    matched_specs, skipped = _partition_specs(specs, aliases)
    if not matched_specs:
        return (), tuple(skipped)

    plan = compile_module_replacements(meta_model, matched_specs)
    records = [_record_for_target(target, paths_by_spec) for target in plan.targets]
    return tuple(records), tuple(skipped)


def _module_aliases(model: Any) -> list[tuple[Any, str]]:
    """``(module, fqn)`` pairs for every registered module alias (no dups removed).

    Mirrors ``replacement._all_module_aliases`` — the set of FQNs a match
    pattern is tested against at compile time.
    """
    return [
        (module, fqn) for fqn, module in model.named_modules(remove_duplicate=False) if fqn
    ]


def _partition_specs(
    specs: Sequence[Any], aliases: set[str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Split ``specs`` into matched and no-match rules by FQN pattern.

    A rule matches when *any* of its patterns matches *any* registered alias —
    the same predicate ``compile_module_replacements`` uses to select targets,
    before its type checks.  Skipped records carry ``{"match": [...], "reason":
    "no_match"}``.
    """
    matched: list[Any] = []
    skipped: list[dict[str, Any]] = []
    for spec in specs:
        if any(
            any(fnmatch.fnmatchcase(fqn, pattern) for fqn in aliases)
            for pattern in spec.match
        ):
            matched.append(spec)
        else:
            skipped.append({"match": list(spec.match), "reason": "no_match"})
    return matched, skipped


def _record_for_target(target: Any, paths_by_spec: dict[int, str]) -> dict[str, Any]:
    spec = target.spec
    factory_path = paths_by_spec.get(id(spec))
    if factory_path is None:
        raise TypeError(
            "codegen: replacement factory for %r has no serializable import "
            "path — the gen backend needs a YAML Target-backed "
            "@module_replacement factory" % (spec.match,)
        )
    return {
        "match": list(spec.match),
        "fqn": target.module_fqns[0],
        "fqns": list(target.module_fqns),
        "module_type": _type_path(spec.module_type),
        "factory": factory_path,
        "exact_type": bool(spec.exact_type),
    }


def _type_path(module_type: type) -> str:
    return f"{module_type.__module__}.{module_type.__qualname__}"


__all__ = ["compile_overrides_for_meta"]
