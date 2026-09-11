# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Lower the frozen parallel plan into explicit forward code.

The lowerer sinks forward wrappers into the source: each boundary class's
``forward`` is rewritten to call ``hyper_redistribute`` on the way in/out and
run the region's compute (inner CP wrapper / EP local compute) explicitly.

The lowerer is a pure function over the source text + the frozen plan.  It
returns the patched text.  Structural work (finding a class / its forward /
the import block) is delegated to :mod:`astkit`.

Class vs FQN: the frozen plan is keyed by boundary FQN
(``model.layers.0.self_attn``), but a ``class.forward`` is shared by many
instances.  The lowerer groups boundary FQNs by their class (meta's
``boundary_classes``), enforces that every FQN of a class carries the SAME
boundary contract, and emits one rewritten ``forward`` per class.

The original ``forward`` is kept
verbatim as the class-private ``_forward_impl`` (extracted by a zero-width
insert before the ``def forward`` — a fresh-source operation, so the offsets
stay valid when combined with the body replacement), and the public
``forward`` is rewritten in one of two forms:

- shape 1 — plain redistribute: redistribute in, ``_forward_impl``, out;
- shape 2 — local region: redistribute in, ``__hyper_compute__`` (bound by
  ``hyper_parallelize``), re-wrap locals per ``out_src``, redistribute
  out.

A CP inner-wrapper boundary (shape 3) is *also* shape 1 in the generated
code: the wrapper is installed at runtime by ``hyper_apply_inner_wrapper``
(it mutates ``target.forward`` in place — not a bindable callable), and runs
inside the ``_forward_impl`` call.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from hyper_parallel.codegen.astkit.edits import (
    TextEdit,
    apply_edits,
    insert_after_imports,
    replace_function_body,
)
from hyper_parallel.codegen.astkit.index import (
    FunctionInfo,
    SourceIndex,
    build_source_index,
)

#: Constant name pattern for one boundary class's frozen entry.
_BOUNDARY_CONST_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BOUNDARY_PREFIX = "_HYPER_BOUNDARY_"
#: Meta / entry field names the lowerer reads off a frozen entry.
_ENTRY_META_FIELDS = ("in_src", "in_dst", "out_src", "out_dst", "out_names")
#: Injection keys that describe how a boundary's compute is regioned.
_INJECTION_KEYS = ("inner_wrapper", "inner_target", "inner_out_src", "local_compute_fn")


def lower_forward_boundaries(
    source_text: str,
    frozen_plan: Any,
    *,
    boundary_classes: Optional[dict[str, str]] = None,
    module_name: str = "",
) -> str:
    """Rewrite each boundary class's ``forward`` to call the codegen runtime.

    ``frozen_plan`` is the meta/param-plan dict or a ``CodegenMeta``; the
    literal entries are read via ``_plan_field`` (same shape as the literals
    already appended by ``inject_param_plan_literals``).  ``boundary_classes``
    is the FQN -> class name map (``meta.boundary_classes``); when omitted it
    degrades to the class name looked up from the source index itself, which is
    only correct if every boundary FQN's leaf class name equals the key's tail —
    ``_class_for_fqn``.

    Returns the patched source text.  The frozen per-class constants are NOT
    appended here (that is the literal layer's job) — this function only
    rewrites the ``forward`` bodies and inserts the ``_HYPER_BOUNDARY_<CLASS>``
    references the bodies read.
    """
    index = build_source_index(source_text)
    param_plan = _plan_field(frozen_plan, "param_plan") or {}
    injections = _plan_field(frozen_plan, "injections") or []
    injection_by_class = _index_injections_by_class(injections, boundary_classes, index)

    groups = _group_boundaries_by_class(
        param_plan, boundary_classes, index, module_name=module_name
    )

    edits: list[TextEdit] = []
    const_texts: list[tuple[str, dict[str, Any]]] = []
    # Iterate boundary classes in sorted ``class_name`` order.  ``groups`` is
    # built by walking ``param_plan``, whose insertion order is NOT stable
    # across the write/reload cycle: ``meta.param_plan`` is persisted with
    # ``json.dump(..., sort_keys=True)``, which reorders the keys
    # alphabetically, so reload-ing the meta yields a different class order
    # than the in-memory freeze did.  The per-class ``_HYPER_BOUNDARY_<CLASS>``
    # constants are concatenated in this iteration order, so an order that
    # depends on ``param_plan``'s provenance would make the emitted modeling
    # file differ between "generate once" and "re-emit from reloaded meta" —
    # breaking artifact drift checks (which re-emit the bundle from the on-disk
    # meta and compares bytes).  Sorting here makes the byte stream a pure
    # function of which classes are boundaries, independent of how
    # ``param_plan`` reached us.
    for class_name in sorted(groups):
        group = groups[class_name]
        entry = _verify_class_contract(group, class_name)
        func = _require_forward(index, class_name)
        # The class constant is the params-stripped contract — ``params`` is
        # instance-specific and drives ``hyper_shard_params``, never the forward
        # rewrite, so it is dropped from both the constant and the body.
        contract = _without_params(entry)
        body = _build_forward_body(class_name, contract, injection_by_class.get(class_name),
                                   func)
        # Keep the original forward as ``_forward_impl``. The
        # extracted method is inserted BEFORE the ``def forward`` (zero-width
        # edit at ``def_offset``), and the rewritten body replaces the original
        # body at ``body_start`` — the two spans never overlap, so the
        # incremental-shift validation in ``_apply_edits`` passes.
        impl = _build_forward_impl_edit(source_text, func)
        if impl is not None:
            edits.append(impl)
        edits.extend(replace_function_body(source_text, func, body, indent=""))
        const_texts.append((class_name, contract))
    if not edits:
        return source_text

    # The rewritten forwards reference ``_HYPER_BOUNDARY_<CLASS>`` per-class
    # constants; insert them right after the import block (before the first
    # class), fresh-source offsets so they combine safely with the body edits.
    const_edit = _insert_boundary_constants(index, const_texts)
    if const_edit is not None:
        edits.append(const_edit)

    nested = _validate_no_circular(edits)
    if nested:
        raise ValueError(
            "lower_forward_boundaries: edits overlap (circular class nesting); "
            "cannot safely rewrite: " + ", ".join(nested)
        )
    return apply_edits(source_text, edits)


# ---------------------------------------------------------------------------
# grouping / consistency
# ---------------------------------------------------------------------------

def _group_boundaries_by_class(
    param_plan: dict[str, Any],
    boundary_classes: Optional[dict[str, str]],
    index: SourceIndex,
    *,
    module_name: str,
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Group boundary FQNs by the class that owns their ``forward``.

    Returns ``{class_name: [(fqn, entry), ...]}`` in a stable order (FQN sorted
    for determinism).

    A boundary whose class is NOT defined in this source file — ``index`` is
    the source index, so ``index.find_class(name) is None`` means the class is
    imported (``nn.Embedding`` / ``nn.Linear`` / ``nn.LayerNorm`` for
    ``embed_tokens`` / ``lm_head`` / ``norm``) — has no ``def forward`` span
    to rewrite, so its FQN is dropped from the grouping and the boundary is
    left un-rewritten here.  Its boundary contract is still executed at runtime
    by ``hyper_wrap_module_boundaries``: the in/out
    redistribution for an imported class is installed as a forward wrapper on
    the live module, so the contract is not dropped.  Skipping the rewrite here
    is exactly what lets the real plan (which marks ``embed_tokens`` /
    ``lm_head`` as boundaries) emit without failing on a class that has no
    source ``forward``.

    An FQN whose class CANNOT be resolved fails fast — a boundary whose forward
    is never rewritten would silently drop its redistribution.
    """
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for fqn, entry in param_plan.items():
        if not _is_boundary_entry(entry):
            continue
        class_name = _class_for_fqn(fqn, boundary_classes, index, module_name)
        if class_name is None:
            raise ValueError(
                f"lower_forward_boundaries: cannot resolve the class for boundary "
                f"{fqn!r} — the frozen plan has no class mapping and the source "
                "index does not contain a class whose name matches; a boundary "
                "whose forward is not rewritten would silently drop its "
                "redistribution"
            )
        if index.find_class(class_name) is None:
            # Imported class (Embedding/Linear/LayerNorm): no source ``forward``
            # to rewrite; runtime ``hyper_wrap_module_boundaries`` installs the
            # in/out boundary wrapper. Not added to ``groups``.
            continue
        groups.setdefault(class_name, []).append((fqn, entry))
    for class_name, items in groups.items():
        items.sort(key=lambda item: item[0])
    return groups


def _is_boundary_entry(entry: dict[str, Any]) -> bool:
    """Whether a frozen param-plan entry is a boundary we sink.

    A boundary is any entry carrying in/out redistribution or an injection (a
    pure replay of an explicit ``is_boundary: false`` entry is a no-op).  An
    entry that has neither a layout contract nor an injection is an ordinary
    module and is left untouched.
    """
    if entry.get("is_boundary") is not None:
        return bool(entry["is_boundary"])
    if any(entry.get(field) for field in _ENTRY_META_FIELDS):
        return True
    return False


def _class_for_fqn(
    fqn: str,
    boundary_classes: Optional[dict[str, str]],
    index: SourceIndex,
    module_name: str,
) -> Optional[str]:
    """Resolve a boundary FQN to a top-level class name.

    Resolution order:
    1. the explicit ``boundary_classes`` map (freeze-time truth);
    2. a source-index class whose name matches the FQN's leaf segment;
    3. a source-index class whose name matches after stripping a known module
       prefix (``module_name``) — remote-code modeling files carry the bare
       class name, and the FQN leaf is usually it;
    4. ``None`` (caller fails fast).
    """
    if boundary_classes:
        cls = boundary_classes.get(fqn)
        if cls:
            return cls
    leaf = fqn.rsplit(".", 1)[-1]
    if leaf in index.classes:
        return leaf
    # A remote-code file may name the class without the module-name prefix on
    # the FQN leaf; only accept it when the leaf matches a class AND the leaf
    # is a plausible modeling class (no lambdas / generics).
    if module_name:
        candidate = module_name.rsplit(".", 1)[-1]
        if candidate in index.classes and _name_matches(candidate, leaf):
            return candidate
    return None


def _name_matches(class_name: str, leaf: str) -> bool:
    """Loose match between a module-name class and an FQN leaf.

    The FQN leaf is the module attribute (``self_attn`` / ``mlp``); the class
    is its type (``Qwen3MoeAttention``).  They usually differ, so this only
    accepts the trivial identity.  The strong mapping is ``boundary_classes``;
    this fallback is deliberately conservative.
    """
    return class_name == leaf


def _verify_class_contract(
    group: list[tuple[str, dict[str, Any]]], class_name: str
) -> dict[str, Any]:
    """Fail fast unless every FQN of ``class_name`` carries the same contract.

    One ``class.forward`` is shared by every instance of the class, so all the
    class's boundaries must be describable by the SAME frozen entry (modulo the
    ``params`` tree, which is instance-specific and drives ``hyper_shard_params``
    not the forward rewrite).  Returns the canonical entry.
    """
    canonical: Optional[str] = None
    canonical_entry: dict[str, Any] = {}
    for fqn, entry in group:
        entry = _without_params(entry)
        blob = json.dumps(entry, sort_keys=True)
        if canonical is None:
            canonical = blob
            canonical_entry = entry
        elif blob != canonical:
            raise ValueError(
                f"lower_forward_boundaries: class {class_name!r} has FQNs with "
                f"differing boundary contracts ({fqn!r}); a shared forward "
                "cannot carry two contracts — split the plan or align the "
                "per-rule contracts"
            )
    return canonical_entry


def _without_params(entry: dict[str, Any]) -> dict[str, Any]:
    """Drop the instance-specific ``params`` tree from a contract comparison."""
    if "params" not in entry:
        return entry
    return {k: v for k, v in entry.items() if k != "params"}


def _index_injections_by_class(
    injections: list[dict[str, Any]],
    boundary_classes: Optional[dict[str, str]],
    index: SourceIndex,
) -> dict[str, dict[str, Any]]:
    """Fold a class's injection declarations (inner wrapper / local compute).

    Returns ``{class_name: {key: value}}`` — the union of every injection rule
    whose ``match`` lands on a FQN of that class.  A class with conflicting
    values for one key (two FQNs declared different local compute fns) fails
    fast, for the same reason as ``_verify_class_contract``.
    """
    by_class: dict[str, dict[str, Any]] = {}
    for rule in injections:
        match = rule.get("match")
        if not isinstance(match, str) or not match:
            continue
        class_name = _class_for_fqn(match, boundary_classes, index, "")
        if class_name is None:
            continue
        payload = {k: rule[k] for k in _INJECTION_KEYS if k in rule}
        if not payload:
            continue
        existing = by_class.setdefault(class_name, {})
        for key, value in payload.items():
            if key in existing and _json_stable(existing[key]) != _json_stable(value):
                raise ValueError(
                    f"lower_forward_boundaries: class {class_name!r} has "
                    f"conflicting injection values for {key!r}"
                )
            existing[key] = value
    return by_class


def _json_stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# forward body construction
# ---------------------------------------------------------------------------

def _require_forward(index: SourceIndex, class_name: str) -> FunctionInfo:
    func = index.find_forward(class_name)
    if func is None:
        raise ValueError(
            f"lower_forward_boundaries: class {class_name!r} has no forward "
            "method; a boundary forward must exist to be rewritten"
        )
    if func.body_start is None or func.body_end is None:
        raise ValueError(
            f"lower_forward_boundaries: class {class_name!r}.forward has no "
            "replaceable body (single-line/pass); cannot sink a boundary"
        )
    return func


def _build_forward_impl_edit(
    source_text: str, func: FunctionInfo
) -> Optional[TextEdit]:
    """Build an edit that extracts ``forward`` as ``_forward_impl``.

    The original method (signature + body) is copied out of the source and
    inserted as the private ``_forward_impl`` right before the rewritten
    ``def forward``.  The copy keeps the original signature and body byte-for-
    byte, re-indented to the method's own indent, with the ``def forward``
    renamed to ``def _forward_impl``.

    Returns ``None`` when the forward has no replaceable body (single-line/
    pass) — there is nothing meaningful to extract, and ``forward`` itself was
    already rejected by ``_require_forward`` for the same reason, so this is a
    defensive fallback.
    """
    if func.body_start is None or func.body_end is None:
        return None
    # The extracted method owns the signature + body; DECORATORS stay attached to
    # the original ``forward`` (which is being rewritten) — copying them onto
    # ``_forward_impl`` too would duplicate a side effect (e.g.
    # ``@torch.no_grad()``) on a method that must be pure.  ``func.def_offset``
    # anchors the FIRST line of the method (a decorator when one is present), so
    # scan forward from it past decorator lines and blanks to the ``def`` keyword
    # — this finds the signature line for both the plain and the decorated case
    # without offset arithmetic (a naive backward scan from ``body_start`` finds
    # the newline that *ends* the ``def`` line, landing on the first body line).
    def_line_start = func.def_offset
    for line in source_text[def_line_start:func.body_start].splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("def "):
            break
        def_line_start += len(line)
    span = source_text[def_line_start:func.body_end]
    lines = span.splitlines(keepends=True)
    if not lines:
        return None
    # ``_forward_impl`` is a VERBATIM copy of the original method (signature +
    # body, byte-for-byte, keeping the method indent on the ``def`` and the body
    # nesting) with only the name changed -- so nothing here needs dedenting or
    # re-indenting, and the copy can never drift from the original source.
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def forward("):
            lines[i] = line.replace("def forward(", "def _forward_impl(", 1)
            break
    impl = "".join(lines)
    # The copy ends at ``body_end``, which is clamped to EOF for a final
    # statement with no trailing newline — so the extracted method can lose its
    # terminating newline and land joined onto the ``def forward`` that follows.
    # A method must be separated from the next one by a newline, so pad.
    if not impl.endswith("\n"):
        impl += "\n"
    return TextEdit(func.def_offset, func.def_offset, impl)


def _build_forward_body(
    class_name: str,
    entry: dict[str, Any],
    injection: Optional[dict[str, Any]],
    func: FunctionInfo,
) -> str:
    """Render the rewritten forward body for one boundary class.

    Supports three boundary forms: a
    plain redistribute boundary, a local-region boundary (``local_compute_fn``),
    and an inner-wrapper boundary (''CP'' attention).  ``__hyper_compute__`` is
    bound by ``hyper_bind_compute`` for the local-region shape; an inner-wrapper
    boundary is emitted as plain-redistribute (shape 1) and its CP wrapper is
    installed at runtime by ``hyper_apply_inner_wrapper``.  Either way the class
    stays runtime-importable without a mesh.
    """
    input_redist = _render_input_redistribute(class_name, entry, func)
    out_redist = _render_output_redistribute(class_name, entry)

    local = injection.get("local_compute_fn") if injection else None
    inner = injection.get("inner_wrapper") if injection else None

    if local is not None:
        return _render_local_compute(class_name, entry, local, input_redist, out_redist)
    if inner is not None:
        # The inner CP wrapper is installed at runtime by
        # ``hyper_apply_inner_wrapper`` (it mutates ``target.forward`` in place,
        # not a bindable callable), so the generated forward is SHAPE-1: plain
        # redistribute.  The wrapper runs inside the ``_forward_impl`` call.
        return _render_plain_redistribute(class_name, entry, input_redist, out_redist)
    return _render_plain_redistribute(class_name, entry, input_redist, out_redist)


def _render_input_redistribute(
    class_name: str, entry: dict[str, Any], func: FunctionInfo
) -> str:
    """Render the input-side ``hyper_redistribute`` plumbing.

    The generated forward re-binds args/kwargs against its own signature, so a
    positional ``(args, kwargs)`` pair reaches the runtime as the boundary's
    input side.  ``_bind_input_indices`` binds each ``in_src`` ``arg_name`` to
    the forward signature's positional index, so the pair MUST be built from the
    signature's positional parameter names (not the plan's) — for a
    ``def forward(self, hidden_states)`` that is ``((hidden_states,), {})``.
    """
    params = ", ".join(func.param_names)
    return (
        f"        args, kwargs = hyper_redistribute(\n"
        f"            (({params},), {{}}),\n"
        f"            {_BOUNDARY_PREFIX}{class_name},\n"
        f"            mesh_context,\n"
        f"            _HYPER_MESH_DIM_NAMES,\n"
        f"            module=self,\n"
        f"        )"
    )


def _render_output_redistribute(class_name: str, entry: dict[str, Any]) -> str:
    """Render the output-side ``hyper_redistribute`` call.

    ``module`` is deliberately NOT passed here: the output side has no
    signature to bind against (only the input side needs
    ``_bind_input_indices``), and passing ``self`` would re-bind the boundary's
    input plan against a tensor that isn't an ``(args, kwargs)`` pair.
    """
    return (
        f"        outputs = hyper_redistribute(\n"
        f"            outputs,\n"
        f"            {_BOUNDARY_PREFIX}{class_name},\n"
        f"            mesh_context,\n"
        f"            _HYPER_MESH_DIM_NAMES,\n"
        f"        )"
    )


def _render_rewrap_outputs(class_name: str, entry: dict[str, Any]) -> str:
    """Render local Tensor to DTensor wrapping according to ``out_src``."""
    return (
        f"        outputs = hyper_rewrap_outputs(\n"
        f"            outputs,\n"
        f"            {_BOUNDARY_PREFIX}{class_name},\n"
        f"            mesh_context,\n"
        f"            _HYPER_MESH_DIM_NAMES,\n"
        f"        )"
    )


def _render_plain_redistribute(
    class_name: str, entry: dict[str, Any], input_redist: str, out_redist: str
) -> str:
    """Plain boundary (shape 1): redistribute in, run the original forward, redistribute out."""
    return "\n".join([
        "        # [HYPER BOUNDARY] input redistribution",
        input_redist,
        "",
        "        # [HYPER BOUNDARY] original forward body (now ``_forward_impl``)",
        "        outputs = self._forward_impl(*args, **kwargs)",
        "",
        "        # [HYPER BOUNDARY] output redistribution",
        out_redist,
        "",
        "        return outputs",
    ])


def _render_local_compute(
    class_name: str, entry: dict[str, Any], local: Any, input_redist: str, out_redist: str
) -> str:
    """Local-region boundary (shape 2, EP MoE): compute fn on local tensors, then exit.

    ``__hyper_compute__`` is ``functools.partial(compute_fn, module)`` bound by
    ``hyper_parallelize``. The applier calls it as ``compute_fn(*args,
    **kwargs)`` (``_wrap_local_region_forward``), so the generated forward
    forwards its own re-distributed ``*args, **kwargs`` through.  The local
    Tensors are re-wrapped per ``out_src`` (decision B) before the boundary exit
    so the output side sees a sharded tensor.
    """
    compute_name = _callable_name(local)
    return "\n".join([
        "        # [HYPER LOCAL REGION] boundary entry",
        input_redist,
        "",
        "        # [HYPER LOCAL REGION] run the region compute fn on local tensors",
        f"        # {compute_name} is bound to self by hyper_parallelize.",
        "        outputs = self.__hyper_compute__(*args, **kwargs)",
        "",
        "        # [HYPER LOCAL REGION] re-wrap locals per out_src before the exit",
        _render_rewrap_outputs(class_name, entry),
        "",
        "        # [HYPER LOCAL REGION] boundary exit",
        out_redist,
        "        outputs = hyper_to_local_if_dtensor(outputs)",
        "",
        "        return outputs",
    ])


def _callable_name(value: Any) -> str:
    """Human name for an injection's callable (dict target / string / fn)."""
    if isinstance(value, dict):
        return value.get("name") or value.get("target") or repr(value)
    if isinstance(value, str):
        return value
    return getattr(value, "__name__", repr(value))


def _insert_boundary_constants(
    index: SourceIndex, const_texts: list[tuple[str, dict[str, Any]]]
) -> Optional[TextEdit]:
    """Build the edit that inserts the ``_HYPER_BOUNDARY_<CLASS>`` constants.

    The rewritten forwards reference one module-level constant per boundary
    class.  They are module-level definitions, so they must sit after the
    imports and before the first class body (PEP 8) — the ``insert_after_imports``
    anchor.  Each value is the frozen per-class contract (params stripped; the
    same blob ``_build_forward_body`` renders).
    """
    if not const_texts:
        return None
    # Imported lazily (emit.modeling imports this module lazily for
    # ``lower_forward_boundaries``; a module-level import here would cycle).
    from hyper_parallel.codegen.emit.modeling import render_python_literal

    const_texts = [
        f"{_BOUNDARY_PREFIX}{cls} = {render_python_literal(value)}"
        for cls, value in const_texts
    ]
    block = "\n\n".join(const_texts)
    return insert_after_imports(
        index.source_text,
        index.import_end,
        block,
        default_offset=index.docstring_end or 0,
    )


# ---------------------------------------------------------------------------
# field reads / validation
# ---------------------------------------------------------------------------

def _plan_field(plan: Any, name: str) -> Any:
    """Read a frozen-plan field from a dict or a ``CodegenMeta``."""
    if isinstance(plan, dict):
        return plan.get(name)
    return getattr(plan, name, None)


def _validate_no_circular(edits: list[TextEdit]) -> list[str]:
    """Detect edits authored against different texts (a soft sanity check).

    The lowerer computes all edits against one ``source_text``; the index is
    built once, so a second-pass author (an inner wrapper that itself carries a
    body rewrite) would need to re-index.  We fail fast rather than splice
    stale offsets.
    """
    problems: list[str] = []
    seen: dict[tuple[int, int], str] = {}
    for edit in edits:
        key = (edit.start, edit.end)
        if key in seen:
            problems.append(
                f"[{edit.start},{edit.end}) (was {seen[key]}, now {edit.replacement[:30]!r})"
            )
        else:
            seen[key] = edit.replacement[:30]
    return problems


__all__ = [
    "lower_forward_boundaries",
]
