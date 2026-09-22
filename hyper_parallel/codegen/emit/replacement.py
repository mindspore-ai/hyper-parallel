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

from typing import Any, Sequence


# ---------------------------------------------------------------------------
# generation-time compile (fail-fast)
# ---------------------------------------------------------------------------

def compile_overrides_for_meta(
    meta_model: Any,
    specs: Sequence[Any],
    *,
    factory_paths: Sequence[str | None] = (),
    configs_by_spec: dict[int, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Match replacement specs against the generation-time meta model.

    Runs :func:`compile_module_replacements` on the meta (empty-weights) model
    so a conflicting or type-mismatched replacement rule fails here, at
    generation time, rather than at runtime inside the built model.  Returns
    one JSON-safe record per matched target.

    Fail-fast behavior (a pattern matching no module, a type mismatch, a
    conflict) is owned by the native :func:`compile_module_replacements`
    itself — the same checks the trainer's hf backend runs — so this wrapper
    keeps no mirrored pre-check; it only re-raises the native
    :class:`ValueError` with the codegen context prefix.  Every rule reaching
    this module on the codegen path comes from the user's YAML
    ``plan_overrides.replace_module`` — there is no plan- or internally-derived
    rule source here — so a no-match is a config error (a typo, or a rule
    written for a different architecture) and must fail generation rather than
    silently drop an intended replacement from the artifact.

    A record carries:

    .. code-block:: python

       {
           "match": ["model.layers.0.mlp"],       # spec.match, the patterns
           "fqn":  "model.layers.0.mlp",          # first aliased FQN
           "fqns": ["model.layers.0.mlp"],        # every registered alias
           "module_type": "<dotted path>",        # source module type
           "factory": "<dotted path>",            # the @module_replacement fn
           "exact_type": False,
           "target_config": {...},                # optional YAML Target static args
       }

    The ``factory`` (and ``module_type``) are stored as import paths, not the
    live objects — the replacement factory is a per-entry closure that cannot
    be written as a Python literal, and the runtime re-imports the *raw*
    decoration at those paths to rebuild equal specs.  When the YAML entry
    bound extra static args to its replace-module Target, those must ride
    along in ``configs_by_spec`` (keyed by ``id(spec)``) so the runtime can
    re-pre-bind them onto the rebuilt factory instead of dropping to the
    factory's defaults (or failing a required arg).
    """
    if not specs:
        return ()
    if configs_by_spec is None:
        configs_by_spec = {}
    from hyper_parallel.models.replacement import compile_module_replacements

    paths_by_spec = {
        id(spec): path
        for spec, path in zip(specs, factory_paths)
        if path is not None
    }

    try:
        plan = compile_module_replacements(meta_model, specs)
    except ValueError as exc:
        raise ValueError(
            f"codegen: plan_overrides replace_module: {exc}"
        ) from exc
    return tuple(
        _record_for_target(target, paths_by_spec, configs_by_spec)
        for target in plan.targets
    )


def _record_for_target(
    target: Any,
    paths_by_spec: dict[int, str],
    configs_by_spec: dict[int, Any] | None = None,
) -> dict[str, Any]:
    spec = target.spec
    factory_path = paths_by_spec.get(id(spec))
    if factory_path is None:
        raise TypeError(
            "codegen: replacement factory for %r has no serializable import "
            "path — the gen backend needs a YAML Target-backed "
            "@module_replacement factory" % (spec.match,)
        )
    record = {
        "match": list(spec.match),
        "fqn": target.module_fqns[0],
        "fqns": list(target.module_fqns),
        "module_type": _type_path(spec.module_type),
        "factory": factory_path,
        "exact_type": bool(spec.exact_type),
    }
    config = (configs_by_spec or {}).get(id(spec))
    if config:
        record["target_config"] = config
    return record


def _type_path(module_type: type) -> str:
    return f"{module_type.__module__}.{module_type.__qualname__}"


__all__ = ["compile_overrides_for_meta"]
