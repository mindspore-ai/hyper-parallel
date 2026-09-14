# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Sink ``replace_module`` overrides into the generated artifact.

The trainer's hf backend applies ``plan_overrides.replace_module`` by
desugaring each entry into a :class:`ModuleReplacementSpec` and running
``apply_module_replacements`` on the live model.  The gen backend must carry
the same contract in the *generated file* so a codegen artifact and its HF
counterpart swap the same module:

* the generated module carries a ``_HYPER_MODULE_OVERRIDES`` literal — one
  record per matched target, each holding the ``match`` patterns, the source
  ``module_type``, the raw replacement factory, ``exact_type``, and every FQN
  the target was aliased under;
* the model entry class's ``__init__`` tail calls ``hyper_apply_replacements``
  (the runtime helper in ``codegen.runtime``) so the replacements install as
  soon as a model instance is built.

The manager compiles the spec against the generation-time meta model to expose
conflicts / type errors / unmatched patterns *at generation time* (fail-fast),
and records the resulting FQN set in ``meta.module_overrides``. The literal,
metadata record, and a recompile on the meta model must agree on that set.
"""
from __future__ import annotations

import fnmatch
from typing import Any, Sequence

from hyper_parallel.codegen.astkit.edits import TextEdit, apply_edits
from hyper_parallel.codegen.astkit.index import build_source_index


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


# ---------------------------------------------------------------------------
# literal emission
# ---------------------------------------------------------------------------

def inject_module_override_literals(text: str, module_overrides: Sequence[dict[str, Any]]) -> str:
    """Append the ``_HYPER_MODULE_OVERRIDES`` literal after ``text``.

    The literal is a module-level frozen record list that the runtime rebuilds
    ``ModuleReplacementSpec``\\ s from (``runtime.hyper_apply_replacements``).
    Empty when no replacement rule matched — the runtime treats that as a no-op.
    """
    from hyper_parallel.codegen.emit.modeling import render_python_literal

    rendered = render_python_literal(list(module_overrides))
    block = (
        "# Frozen module replacements applied by\n"
        "# ``hyper_apply_replacements`` when the model ``__init__`` runs.\n"
        "# [HYPER MODULE REPLACEMENT]\n"
        "_HYPER_MODULE_OVERRIDES = " + rendered + "\n"
    )
    return text.rstrip() + "\n\n" + block


# ---------------------------------------------------------------------------
# entry class rewrite
# ---------------------------------------------------------------------------

def rewrite_init_for_overrides(text: str, module_overrides: Sequence[dict[str, Any]], *, model_class: str) -> str:
    """Insert the ``hyper_apply_replacements(self, _HYPER_MODULE_OVERRIDES)`` call.

    ``model_class`` is the top-level model class name (the architecture name,
    e.g. ``AnthropicV3ForCausalLM``) whose ``__init__`` receives the call.  The
    insertion is a zero-width edit just past the ``__init__`` body (or, when the
    body has no replaceable span, past the ``pass``/single statement the class
    already calls at construction).  The literal is passed explicitly (not read
    from frame globals) so the generated module stays self-contained.  A model
    with no replacement records is left untouched — the runtime call is only
    emitted when there is actually a literal to apply.
    """
    if not module_overrides:
        return text

    index = build_source_index(text)
    cls = index.find_class(model_class)
    if cls is None:
        raise ValueError(
            f"codegen: cannot rewrite __init__ for module overrides — generated "
            f"source has no class {model_class!r}"
        )
    init = cls.methods.get("__init__")
    if init is None:
        raise ValueError(
            f"codegen: cannot sink module overrides — {model_class!r} has no __init__"
        )

    anchor, indent = _init_tail_anchor(text, init)
    if anchor is None or indent is None:
        raise ValueError(
            f"codegen: cannot locate the tail of {model_class}.__init__ body"
        )

    edit = TextEdit(anchor, anchor, f"\n{indent}hyper_apply_replacements(self, _HYPER_MODULE_OVERRIDES)\n")
    return apply_edits(text, [edit])


def _init_tail_anchor(text: str, init: Any) -> tuple[int | None, str | None]:
    """``(insert_offset, indent)`` for appending a statement to ``__init__``.

    ``body_end`` is the byte just past the body's trailing newline; the byte
    immediately before it is that newline, so inserting ``\\n<stmt>`` there lands
    the statement on its own line at the method's body indentation.  The indent
    must come from ``body_start`` — ``body_end`` may sit at the *class* indent
    when ``__init__`` is followed by another method (the offset past the last
    statement's newline is the next line's start, not the function's nesting).
    A body without a replaceable span (a single ``pass`` / ``...``) has
    ``body_end`` ``None`` — fall back to anchoring after the ``def`` line's
    ``:`` and synthesize the method indent.
    """
    if init.body_end is not None:
        indent = _leading_indent(text, init.body_start)
        return init.body_end, indent
    if init.def_offset is not None and init.body_start is None:
        # Single-line/pass body: anchor at the end of the ``def`` header line.
        line_end = text.find("\n", init.def_offset)
        anchor = line_end if line_end != -1 else len(text)
        return anchor, _leading_indent(text, anchor)
    return None, None


def _leading_indent(text: str, offset: int) -> str:
    """The whitespace indenting the line containing ``offset`` (or blank)."""
    if offset <= 0 or offset > len(text):
        return ""
    line_start = text.rfind("\n", 0, offset) + 1
    idx = line_start
    while idx < len(text) and text[idx] in " \t":
        idx += 1
    return text[line_start:idx]
