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
"""Lower the frozen parallel plan into explicit forward code.

The lowerer sinks forward wrappers into the source.  Which form a boundary
class's ``forward`` takes is decided by
:func:`hyper_parallel.codegen.plan.boundary_forms.classify_boundary_form` —
the same pure classifier the runtime re-runs at install time and preflight
re-runs at train time, so the three stages can never disagree about a
boundary's form.

Class vs FQN: the frozen plan is keyed by boundary FQN
(``model.layers.0.self_attn``), but a ``class.forward`` is shared by many
instances.  The lowerer groups boundary FQNs by their class (meta's
``boundary_classes``), enforces that every FQN of a class carries the SAME
boundary contract, and emits one rewritten ``forward`` per class.

The original ``forward`` is kept
verbatim as the class-private ``_forward_impl`` (extracted by a zero-width
insert before the ``def forward`` — a fresh-source operation, so the offsets
stay valid when combined with the body replacement), and the public
``forward`` is rewritten in one of the classified forms:

- ``identity`` (pruned): every declared transition is identity on the plan's
  active axes — the class forward is NOT rewritten at all.  The runtime's
  install path covers the exact no-op / to-local semantics with its generic
  wrapper, so pruning is byte-for-byte equivalent to the previous rewrite.
- ``tp_collective`` (static bare operators): every declared transition is
  identity or TP-lowerable, and the source-side structural gates pass
  (single-value return, no ``*args``/``**kwargs``, declared inputs bind to
  forward parameters).  The forward names its form with the class attribute
  ``_hp_boundary_form = "tp_collective"`` and calls bare operators on
  ``self._hp_tp`` (bound at install time after the runtime re-validates
  the compiled plan against the live mesh; on mismatch the runtime replaces
  the forward with the generic engine — the baked calls never run unvalidated).
- ``generic``: redistribute in, ``_forward_impl``, redistribute out, through
  the instance-bound compiled plan (``self._hp_boundary``, bound once by
  ``hp_install_boundaries``), with a declarative comment block annotating
  what each side's transitions lower to.
- ``region`` (EP MoE local region): redistribute in,
  ``__hp_compute__`` (bound by ``hp_parallelize``), re-wrap locals per
  ``out_src``, redistribute out.

A CP inner-wrapper boundary is ``generic`` in the generated code: the wrapper
is installed at runtime by ``hp_apply_inner_wrapper`` (it mutates
``target.forward`` in place — not a bindable callable), and runs inside the
``_forward_impl`` call.

The lowerer is a pure function over the source text + the frozen plan.  It
returns the patched text.  Structural work (finding a class / its forward /
the import block) is delegated to :mod:`astkit`.
"""
from __future__ import annotations

import json
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
    returns_any_value,
    returns_single_value,
)
from hyper_parallel.codegen.plan.boundary_forms import (
    FORM_GENERIC,
    FORM_IDENTITY,
    FORM_REGION,
    FORM_TP_COLLECTIVE,
    BoundaryForm,
    TransitionOp,
    classify_boundary_form,
    is_boundary_entry,
)
from hyper_parallel.codegen.plan.freeze import parse_placement

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
    plan entries are read via ``_plan_field``.  ``boundary_classes``
    is the FQN -> class name map (``meta.boundary_classes``); when omitted it
    degrades to the class name looked up from the source index itself, which is
    only correct if every boundary FQN's leaf class name equals the key's tail —
    ``_class_for_fqn``.  The plan's active axes (``mesh_dim_names``) drive
    :func:`classify_boundary_form`; the classified form plus the source-side
    structural gates decide which template each class gets (see the module
    docstring).

    Returns the patched source text.  This function only rewrites the
    ``forward`` bodies; the rewritten bodies reference the ``_hp_boundary``
    / ``_hp_tp`` instance attributes that ``hp_install_boundaries`` binds
    at runtime — no per-boundary global constant is emitted.
    """
    edits: list[TextEdit] = []
    needs_runtime_helper_import = False
    for class_name, form, emitted, func, injection, deferred_bias in iter_emitted_forms(
        source_text,
        frozen_plan,
        boundary_classes=boundary_classes,
        module_name=module_name,
    ):
        if emitted == FORM_IDENTITY:
            # Pruned: the runtime's install path covers the exact no-op /
            # to-local semantics with its generic wrapper, so leaving the
            # original forward untouched is behavior-preserving.
            if deferred_bias:
                # D-22: an unrewritten forward never calls the exit hook, so
                # the suppressed bias would silently vanish — fail fast
                # instead (the planner defers only rowwise-bias exits that
                # reduce, which classify as non-identity, so this is a
                # mis-declared plan, not a supported shape).
                raise NotImplementedError(
                    f"lower_forward_boundaries: boundary class {class_name!r} "
                    "prunes to identity but declares deferred bias params; an "
                    "identity forward has no exit hook to re-add them"
                )
            continue
        # Every emitted forward body (generic, local-region) ends with the
        # runtime helper ``hp_to_local_if_dtensor``; that helper is a
        # dependency of the templates, not of any declaration, so the lowerer
        # must import it itself — otherwise the artifact is not self-contained.
        needs_runtime_helper_import = True
        body = _build_forward_body(
            source_text, injection, func, form, emitted, deferred_bias
        )
        # Keep the original forward as ``_forward_impl``. The
        # extracted method is inserted BEFORE the ``def forward`` (zero-width
        # edit at ``def_offset``), and the rewritten body replaces the original
        # body at ``body_start`` — the two spans never overlap, so the
        # incremental-shift validation in ``_apply_edits`` passes.  The static
        # form additionally prefixes the class marker naming the form.
        marker = _TP_FORM_MARKER_LINE if emitted == FORM_TP_COLLECTIVE else None
        impl = _build_forward_impl_edit(source_text, func, prefix=marker)
        if impl is not None:
            edits.append(impl)
        edits.extend(replace_function_body(source_text, func, body, indent=""))
    if not edits:
        return source_text

    nested = _validate_no_circular(edits)
    if nested:
        raise ValueError(
            "lower_forward_boundaries: edits overlap (circular class nesting); "
            "cannot safely rewrite: " + ", ".join(nested)
        )
    patched = apply_edits(source_text, edits)
    if needs_runtime_helper_import:
        patched = _add_runtime_helper_import(patched)
    return patched


#: Runtime helper the emitted forward templates call. The templates (and
#: this constant) must stay in sync — see ``_render_local_compute`` and
#: ``_render_output_redistribute``.
_RUNTIME_HELPER_IMPORT = (
    "from hyper_parallel.codegen.runtime import hp_to_local_if_dtensor"
)


def _add_runtime_helper_import(source_text: str) -> str:
    """Ensure the forward templates' runtime helper is imported.

    The helper is a template dependency, so nothing in the declarations
    contributes it; without this the generated artifact raises ``NameError`` at
    the first forward call. No-op when the source already imports it.
    """
    if "import hp_to_local_if_dtensor" in source_text:
        return source_text
    index = build_source_index(source_text)
    edit = insert_after_imports(
        source_text,
        index.import_end,
        _RUNTIME_HELPER_IMPORT,
        default_offset=index.docstring_end or 0,
    )
    return apply_edits(source_text, [edit]) if edit is not None else source_text


#: Class attribute a ``tp_collective`` forward carries, naming its form so the
#: runtime install path and preflight can recognize the static template.
TP_FORM_ATTRIBUTE = "_hp_boundary_form"
#: The marker value naming the statically lowered TP-collective form.
TP_FORM_MARKER = "tp_collective"
#: The emitted marker statement (method indent, followed by a blank line).
_TP_FORM_MARKER_LINE = f'    {TP_FORM_ATTRIBUTE} = "{TP_FORM_MARKER}"\n\n'


def iter_emitted_forms(
    source_text: str,
    frozen_plan: Any,
    *,
    boundary_classes: Optional[dict[str, str]] = None,
    module_name: str = "",
):
    """Yield every boundary class's emit decision over ``source_text``.

    Single decision path shared by the emitter
    (:func:`lower_forward_boundaries`) and the train-time preflight
    (``check/preflight.verify_boundary_forms``): classification
    (:func:`classify_boundary_form`) plus the structural gates
    (:func:`resolve_emitted_form`), per boundary class.  Yields
    ``(class_name, form, emitted, func, injection, deferred_bias)`` —
    ``form`` the classifier's verdict, ``emitted`` the form actually
    rendered after the gates, ``func`` the class's ``forward``
    :class:`FunctionInfo`, ``injection`` the class's frozen injection rule
    (``None`` when absent), and ``deferred_bias`` whether the class's
    canonical entry carries D-22 deferred bias params (the emitted exit
    then calls ``self._hp_deferred_bias(outputs)`` after the output
    redistribution).

    Boundary classes are iterated in sorted ``class_name`` order.  The
    grouping walks ``param_plan``, whose insertion order is NOT stable
    across the write/reload cycle: ``meta.param_plan`` is persisted with
    ``json.dump(..., sort_keys=True)``, which reorders the keys
    alphabetically, so reload-ing the meta yields a different class order
    than the in-memory freeze did.  Sorting here makes the emitted byte
    stream a pure function of which classes are boundaries, independent of
    how ``param_plan`` reached us.
    """
    index = build_source_index(source_text)
    param_plan = _plan_field(frozen_plan, "param_plan") or {}
    injections = _plan_field(frozen_plan, "injections") or []
    injection_by_class = _index_injections_by_class(injections, boundary_classes, index)
    mesh_dim_names = tuple(_plan_field(frozen_plan, "mesh_dim_names") or ())

    groups = _group_boundaries_by_class(
        param_plan, boundary_classes, index, module_name=module_name
    )
    for class_name in sorted(groups):
        group = groups[class_name]
        # Fail fast when one class's FQNs disagree on the contract: one shared
        # ``forward`` cannot carry two boundary plans.  The canonical entry is
        # the per-class contract every FQN agreed on.
        canonical_entry = _verify_class_contract(group, class_name)
        func = _require_forward(index, class_name)
        injection = injection_by_class.get(class_name)
        form = classify_boundary_form(canonical_entry, mesh_dim_names, injection)
        emitted = resolve_emitted_form(source_text, func, form)
        deferred_bias = bool(canonical_entry.get("deferred_bias_params"))
        yield class_name, form, emitted, func, injection, deferred_bias


def resolve_emitted_form(
    source_text: str,
    func: FunctionInfo,
    form: BoundaryForm,
) -> str:
    """The form the emitter actually renders for one classified boundary.

    The classifier's ``tp_collective`` is an *optimistic* verdict: the static
    template additionally requires the source-side structural gates (no
    ``*args``, value returns, declared inputs binding to forward parameters,
    output ops targeting the first output).  A boundary that fails a gate
    renders as ``generic`` instead.  This resolver is the single decision
    shared by the emitter and preflight, so the emitted file and the train-time
    check can never disagree about which template a class got.
    """
    if form.form in (FORM_IDENTITY, FORM_REGION, FORM_GENERIC):
        return form.form
    if _static_gates_pass(source_text, func, form):
        return FORM_TP_COLLECTIVE
    return FORM_GENERIC


def _static_gates_pass(source_text: str, func: FunctionInfo, form: BoundaryForm) -> bool:
    """Whether the static bare-operator template is admissible for ``func``.

    Gates (each mirrors a concrete way the generic engine is more general
    than a static rewrite):

    1. No ``*args`` — a positional variable tail cannot be enumerated
       statically. ``**kwargs`` is allowed and is re-passed by name.
    2. Every ``return`` carries a value.  Single-value returns receive output
       ops directly; tuple returns receive index-0 output ops and are
       rebuilt, matching the compiled plan's output-index behavior.
    3. Every declared input name binds to a forward parameter — the generic
       engine skips an ``in_src`` name the caller never passed (``_get_arg``
       default ``None``); the static template would reference an unbound
       variable instead, so a declared name outside the signature disqualifies
       the static form.
    4. Every non-identity output op targets the first declared output — with a
       single-value return the compiled plan applies index-0 ops only; an op
       mapped to a later output position cannot be expressed statically.
    """
    if func.vararg_name is not None:
        return False
    if not returns_any_value(source_text, func):
        return False
    signature_names = set(func.param_names) | set(func.kwonly_names)
    if not set(form.in_declared) <= signature_names:
        return False
    out_names = form.out_names
    for name in form.out_ops:
        # An op name missing from the declared order compiles to index 0
        # (``_compile_output_plan``'s ``name_to_idx.get(name, 0)``), which the
        # single-value template covers; a later position does not.
        if name in out_names and out_names.index(name) != 0:
            return False
    return True


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
    by ``hp_install_boundaries``: the in/out
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
        if not is_boundary_entry(entry):
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
            # to rewrite; runtime ``hp_install_boundaries`` installs the
            # in/out boundary wrapper. Not added to ``groups``.
            continue
        groups.setdefault(class_name, []).append((fqn, entry))
    for class_name, items in groups.items():
        items.sort(key=lambda item: item[0])
    return groups


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
    ``params`` tree, which is instance-specific and drives the native Phase A
    parameter sharding, not the forward rewrite).  Returns the canonical entry.
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
# toggle / switch forward template (parallel, non-default generation mode)
# ---------------------------------------------------------------------------
#: Class marker value naming the switch-based (toggle) template.  Distinct from
#: ``TP_FORM_MARKER`` ("tp_collective") so the runtime install path and preflight
#: can tell the two opt-in templates apart.  Like the static marker it names the
#: template a boundary class actually got.
TOGGLE_FORM_MARKER = "toggle"
#: The emitted marker statement (method indent, then a blank line).
_TOGGLE_FORM_MARKER_LINE = f'    {TP_FORM_ATTRIBUTE} = "{TOGGLE_FORM_MARKER}"\n\n'


def lower_forward_boundaries_toggle(
    source_text: str,
    frozen_plan: Any,
    *,
    boundary_classes: Optional[dict[str, str]] = None,
    module_name: str = "",
) -> str:
    """Rewrite each boundary ``forward`` into per-dimension-switched segments.

    A SEPARATE generation mode from :func:`lower_forward_boundaries`: it emits,
    for every source-defined boundary class, a forward that keeps the *original
    HF forward body verbatim* as ``_forward_impl`` and wraps it with independent
    ``if tp_enable`` / ``if cp_enable`` / ``if ep_enable`` segments that slice /
    communicate / merge along each dimension the frozen plan shards on.  Only ONE
    mode is chosen per generation (see ``emit.modeling``), so a generated artifact
    never carries a second parallel execution path.

    What the switch bodies read is bound by the block-1 runtime path:

    - ``mesh_context`` — the codegen mesh context already published as a module
      global by ``runtime.publish_codegen_mesh_context``; the toggle booleans and
      the submesh handles defer to it.  Referencing a module global keeps the
      artifact importable without a mesh (the name is only resolved at call time).
    - ``self._hp_tp`` — the bare TP operator object bound by
      ``hp_install_boundaries``'s collectives step for the ``tp_enable`` body.

    The per-axis decomposition is re-derived here from ``frozen_plan`` (the
    frozen entry's full per-axis placement dicts), independently of the merged
    ``tp_collective`` classification, because the switch template needs one
    guarded segment per *axis* rather than one merged verdict over all axes.
    """
    edits: list[TextEdit] = []
    for class_name, _, emitted, func, injection, deferred_bias in iter_emitted_forms(
        source_text,
        frozen_plan,
        boundary_classes=boundary_classes,
        module_name=module_name,
    ):
        del injection
        if emitted == FORM_IDENTITY:
            # Pruned as in the parallel mode: identity boundaries need no rewrite.
            continue
        if emitted == FORM_REGION:
            raise NotImplementedError(
                "toggle forward: the region (EP MoE local) form is not emitted "
                f"for {_class_for_fqn(class_name, boundary_classes, build_source_index(source_text), module_name)!r}; "
                "the toggle template assumes a dense (non-MoE) boundary. Split the "
                "plan or keep the default parallel mode for MoE topologies."
            )
        if deferred_bias:
            # D-22: the toggle template decomposes the exit into per-axis
            # guarded segments and has no single boundary-exit point to hang
            # the deferred-bias re-add on; emitting nothing would silently
            # drop the suppressed bias. Fail fast and point at the mode that
            # supports deferral.
            raise NotImplementedError(
                f"toggle forward: boundary class {class_name!r} declares "
                "deferred bias params (D-22); the toggle template has no "
                "boundary exit hook to re-add them — use the default "
                "parallel mode for plans that defer rowwise biases"
            )
        body = _render_toggle_forward(
            source_text,
            frozen_plan,
            class_name,
            func,
            boundary_classes=boundary_classes,
            module_name=module_name,
        )
        impl = _build_forward_impl_edit(source_text, func, prefix=_TOGGLE_FORM_MARKER_LINE)
        if impl is not None:
            edits.append(impl)
        edits.extend(replace_function_body(source_text, func, body, indent=""))
    if not edits:
        return source_text

    nested = _validate_no_circular(edits)
    if nested:
        raise ValueError(
            "lower_forward_boundaries_toggle: edits overlap (circular class "
            "nesting); cannot safely rewrite: " + ", ".join(nested)
        )
    return apply_edits(source_text, edits)


def _render_toggle_forward(
    source_text: str,
    frozen_plan: Any,
    class_name: str,
    func: FunctionInfo,
    *,
    boundary_classes: Optional[dict[str, str]] = None,
    module_name: str = "",
) -> str:
    """Render one switch-based forward body for a boundary class.

    The body keeps the original forward as ``_forward_impl`` and inserts one
    guarded segment per enabled dimension.  TP calls the block-1-bound
    ``self._hp_tp``; the ``tp_enable`` / ``cp_enable`` / ``ep_enable``
    booleans are bound on the instance by the install path and defer to
    ``mesh_context``.
    """
    inbound, outbound = _per_axis_toggle_ops(
        frozen_plan, class_name, source_text,
        boundary_classes=boundary_classes, module_name=module_name,
    )

    lines = [
        "        # [HYPER TOGGLE] keep the original HF forward as ``_forward_impl``",
        "        # and insert per-dimension slice / collective / merge segments,",
        "        # each guarded by its own {{tp|cp|ep}}_enable switch.",
        f"        # [HYPER INLINE] {_fusion_note_for(class_name)}",
        "",
    ]
    for axis in ("tp", "cp", "ep"):
        axis_ops = inbound.get(axis)
        if not axis_ops:
            continue
        lines.append(f"        if self.{axis}_enable:")
        for name, op in sorted(axis_ops.items()):
            expr = _render_tp_call_expr(name, op)
            lines.append(f"            {name} = {expr}")
        lines.append("")
    lines.append(
        "        # [HYPER TOGGLE] original forward body (now ``_forward_impl``)"
    )
    lines.append(_render_forward_call(func))
    lines.append("")
    for axis in reversed(("ep", "cp", "tp")):
        axis_ops = outbound.get(axis)
        if not axis_ops:
            continue
        lines.append(f"        if self.{axis}_enable:")
        for name, op in sorted(axis_ops.items()):
            if op.kind == "all_reduce":
                lines.append("            outputs = self._hp_tp.all_reduce(outputs)")
            elif op.kind == "all_gather":
                lines.append(
                    f"            outputs = self._hp_tp.all_gather(outputs, dim={op.tensor_dim})"
                )
            else:
                lines.append(
                    f"            outputs = self._hp_tp.reduce_scatter(outputs, dim={op.tensor_dim})"
                )
        lines.append("")
    lines.append("        return outputs")
    return "\n".join(lines)


def _per_axis_toggle_ops(
    frozen_plan: Any,
    class_name: str,
    source_text: str,
    *,
    boundary_classes: Optional[dict[str, str]] = None,
    module_name: str = "",
) -> tuple[dict[str, dict[str, TransitionOp]], dict[str, dict[str, TransitionOp]]]:
    """Decompose one class's boundary into per-axis switch ops for the toggle body.

    Returns ``(in, out)``, each ``{axis: {name: TransitionOp}}`` for only the
    axes whose placements actually change (op ``kind != identity``).  The
    decomposition re-classifies each declared name on each of the plan's active
    axes independently so a boundary can guard ``tp`` and ``cp`` work in separate
    blocks.  A class's forward is shared by every instance, so only the first
    boundary FQN that resolves to ``class_name`` is decomposed.
    """
    param_plan = _plan_field(frozen_plan, "param_plan") or {}
    axes = tuple(_plan_field(frozen_plan, "mesh_dim_names") or ())
    if not axes or not param_plan:
        return {}, {}
    index = build_source_index(source_text)
    for fqn, entry in param_plan.items():
        if not is_boundary_entry(entry):
            continue
        if _class_for_fqn(fqn, boundary_classes, index, module_name) != class_name:
            continue
        inbound = _per_axis_side(entry, "in_src", "in_dst", axes)
        outbound = _per_axis_side(entry, "out_src", "out_dst", axes)
        return inbound, outbound
    return {}, {}


def _per_axis_side(
    entry: dict[str, Any], src_field: str, dst_field: str, axes: tuple[str, ...]
) -> dict[str, dict[str, TransitionOp]]:
    """Per-axis non-identity ops for one side, from the frozen placement strings.

    Classifies each declared name on each single active axis purely from the
    canonical placement strings (``"R"`` / ``"S(n)"`` / ``"P(sum)"``) that the
    freeze wrote — deliberately NOT via ``classify_tp_transition`` (which pulls
    ``hyper_parallel.distributed`` / ``accelerate`` and trips over the tests'
    ``torch_npu`` stub in isolated runs).  So a boundary can carry an independent
    ``tp`` and ``cp`` segment.  ``ep`` changes (MoE expert routing) are left for
    the caller to reject via the region/dense guard.
    """
    src_raw = entry.get(src_field) or {}
    dst_raw = entry.get(dst_field) or {}
    result: dict[str, dict[str, TransitionOp]] = {}
    for axis in axes:
        axis_ops: dict[str, TransitionOp] = {}
        for name in sorted(set(src_raw) | set(dst_raw)):
            src_pl = src_raw.get(name) or {}
            dst_pl = dst_raw.get(name) or {}
            src = src_pl.get(axis, "R")
            dst = dst_pl.get(axis, "R")
            if not isinstance(src, str) or not isinstance(dst, str) or src == dst:
                continue
            op = _toggle_placement_op(src, dst)
            if op is not None:
                axis_ops[name] = op
        if axis_ops:
            result[axis] = axis_ops
    return result


def _toggle_placement_op(src: str, dst: str) -> Optional[TransitionOp]:
    """One per-axis transition op from a valid frozen placement pair, or None.

    Delegates the placement semantics to native
    ``classify_tp_transition`` (the single source of truth the boundary-form
    classifier already uses), invoked on the single active axis.  Frozen
    strings are parsed with ``freeze.parse_placement``; any transition the
    native path cannot lower (non-``sum`` partial, unsupported pair) or an
    identity yields ``None`` (left to the boundary's compiled-plan/DTensor
    fallback).
    """
    from hyper_parallel.distributed._builder.tp_collective_lowering import (  # pylint: disable=C0415
        classify_tp_transition,
    )

    result = classify_tp_transition(
        [parse_placement(src)], [parse_placement(dst)], ("tp",)
    )
    if result is None or result["kind"] == "identity":
        return None
    return TransitionOp(kind=result["kind"], tensor_dim=result.get("tensor_dim"))


def _fusion_note_for(class_name: str) -> str:
    """One-line replacement note placed at the top of a switch forward."""
    return f"fused module {class_name}: the switch template keeps the original HF " \
           "forward as its backbone and only adds dimension-guarded segments."


# ---------------------------------------------------------------------------
# forward body construction
# ---------------------------------------------------------------------------

def _require_forward(index: SourceIndex, class_name: str) -> FunctionInfo:
    """Return the ``forward`` ``FunctionInfo`` for a boundary class or raise."""
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
    source_text: str,
    func: FunctionInfo,
    prefix: Optional[str] = None,
) -> Optional[TextEdit]:
    """Build an edit that extracts a method as ``_forward_impl``.

    The original method (signature + body) is copied out of the source and
    inserted as the private ``_forward_impl`` right before the rewritten
    ``def forward``.  The copy keeps the original signature and body byte-for-
    byte, re-indented to the method's own indent, with the ``def forward``
    renamed to ``def _forward_impl``.  ``prefix`` (the static form's class
    marker) is prepended to the same insertion — one zero-width edit at
    ``def_offset``, so the edit set never carries two edits at one offset.

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
        if stripped.startswith(f"def {func.name}("):
            lines[i] = line.replace(f"def {func.name}(", "def _forward_impl(", 1)
            break
    impl = "".join(lines)
    # The copy ends at ``body_end``, which is clamped to EOF for a final
    # statement with no trailing newline — so the extracted method can lose its
    # terminating newline and land joined onto the ``def forward`` that follows.
    # A method must be separated from the next one by a newline, so pad.
    if not impl.endswith("\n"):
        impl += "\n"
    if prefix:
        impl = prefix + impl
    return TextEdit(func.def_offset, func.def_offset, impl)


def _build_forward_body(
    source_text: str,
    injection: Optional[dict[str, Any]],
    func: FunctionInfo,
    form: BoundaryForm,
    emitted: str,
    deferred_bias: bool = False,
) -> str:
    """Render the rewritten forward body for one boundary class.

    ``emitted`` is :func:`resolve_emitted_form`'s verdict (the classifier's
    form after the static gates).  The redistribute calls go through the
    instance-bound compiled plan (``self._hp_boundary``, installed by
    ``hp_install_boundaries``); the static ``tp_collective`` form calls bare
    operators on ``self._hp_tp`` instead; ``__hp_compute__`` is bound by
    ``hp_bind_compute`` for the local-region shape. Either way the class
    stays runtime-importable without a mesh.

    ``deferred_bias`` (D-22) appends the exit hook call
    ``self._hp_deferred_bias(outputs)`` after the output redistribution —
    the runtime binds that hook when it installs the native
    suppression/restore pair.
    """
    if emitted == FORM_TP_COLLECTIVE:
        return _render_tp_collective_forward(source_text, func, form, deferred_bias)

    input_redist = _render_input_redistribute(func)
    out_redist = _render_output_redistribute(deferred_bias)

    local = injection.get("local_compute_fn") if injection else None
    if local is not None:
        return _render_local_compute(local, input_redist, out_redist, form)
    # A generic boundary (including an inner-wrapper one — the CP wrapper is
    # installed at runtime by ``hp_apply_inner_wrapper`` and weaves into
    # this forward) redistributes through the compiled plan, annotated with
    # what each side's transitions lower to.
    return _render_plain_redistribute(input_redist, out_redist, form)


def _render_input_redistribute(func: FunctionInfo) -> str:
    """Render the input-side redistribute call on the compiled boundary plan.

    The generated forward re-binds args/kwargs against its own signature, so a
    positional ``(args, kwargs)`` pair reaches the compiled plan as the
    boundary's input side.  Index binding happens once at install time
    (``hp_install_boundaries`` → ``_bind_input_indices``); the pair is still
    built from the signature's own parameter names (not the plan's) — for a
    ``def forward(self, hidden_states)`` that is ``((hidden_states,), {})``.

    The full parameter profile is preserved on the re-pass: positional names go
    in the args tuple (a trailing ``*args`` expands into it), keyword-only
    parameters and a fixed ``**kwargs`` go into the kwargs dict.  The boundary
    plan rewrites only the args/kwargs it owns and forwards the rest through
    unchanged, so ``self._forward_impl(*args, **kwargs)`` receives every
    argument class the original call site used — a keyword-only ``required``
    or an HF ``**kwargs`` pass-through no longer drops out.
    """
    args = list(func.param_names)
    if func.vararg_name:
        args.append(f"*{func.vararg_name}")
    kwargs_parts = [f'"{name}": {name}' for name in func.kwonly_names]
    if func.kwarg_name:
        kwargs_parts.append(f"**{func.kwarg_name}")
    kwargs_literal = "{" + ", ".join(kwargs_parts) + "}" if kwargs_parts else "{}"
    # The args body is spliced into a tuple literal; a single element needs a
    # trailing comma (also for a lone ``*vararg``) so it stays a tuple rather
    # than parenthesized scalars / a generator expression.
    args_body = ", ".join(args) + ("," if len(args) == 1 else "")
    return (
        "        args, kwargs = self._hp_boundary.redistribute_inputs(\n"
        f"            (({args_body}), {kwargs_literal}),\n"
        "        )"
    )


def _render_output_redistribute(deferred_bias: bool = False) -> str:
    """Render the output-side boundary exit (no signature to bind).

    Mirrors the native ``_wrap_local_region_forward`` exit order exactly:
    Step-3 re-wrap first (local -> DTensor per ``out_src``), then the Step-4
    ``out_src -> out_dst`` redistribute, then ``to_local`` so the boundary
    always hands its caller a plain local tensor.

    The order is load-bearing, not stylistic. The compute output is the
    ``out_src`` *local shard* (e.g. a Partial(sum) partial sum). Re-wrapping
    first attaches the matching ``out_src`` metadata, and the redistribute then
    lowers the declared transition (e.g. reduce/all-reduce) to the ``out_dst``
    value. Reversing the two — redistribute first, re-wrap second — would wrap
    the already-reduced value with the stale ``out_src`` metadata (a DTensor
    whose values are reduced but whose placements still say Partial(sum)), and
    any downstream DTensor dispatch would then make placement-based decisions
    on false metadata (e.g. ``AddDistributedOp`` zeroing a replicated operand
    on non-zero coordinates of the partial axis).

    ``deferred_bias`` (D-22) appends the exit hook AFTER ``to_local``: the
    rowwise bias was suppressed inside the region so the exit reduction sees
    a pure matmul contribution, and the hook re-adds it exactly once on the
    reduced local output — Megatron RowParallelLinear semantics.  The hook is
    bound at install time (``module._hp_deferred_bias``); the emitted call
    is conditional on the plan declaring deferred params, so the two sides
    cannot disagree about whether a boundary defers.
    """
    lines = [
        "        outputs = self._hp_boundary.rewrap_outputs(outputs)",
        "        outputs = self._hp_boundary.redistribute_outputs(outputs)",
        "        outputs = hp_to_local_if_dtensor(outputs)",
    ]
    if deferred_bias:
        lines.append("        outputs = self._hp_deferred_bias(outputs)")
    return "\n".join(lines)


def _ordered_notes(notes: dict[str, str]) -> list[str]:
    """Declarative annotation lines for one side, in deterministic name order."""
    return [f"        # {notes[name]}" for name in sorted(notes)]


def _render_plain_redistribute(
    input_redist: str, out_redist: str, form: BoundaryForm
) -> str:
    """Generic boundary: annotated redistribute around the original forward.

    The comment blocks are the plan's own declaration: one line per
    non-identity transition (``name: axis S(1) -> R ==> all_gather(dim=1)`` or
    ``==> dtensor redistribute`` for transitions the compiled plan's DTensor
    fallback owns), so a reader sees what the boundary communicates without
    re-deriving the plan.
    """
    return "\n".join([
        "        # [HYPER BOUNDARY] input redistribution",
        *_ordered_notes(form.in_notes),
        input_redist,
        "",
        "        # [HYPER BOUNDARY] original forward body (now ``_forward_impl``)",
        "        outputs = self._forward_impl(*args, **kwargs)",
        "",
        "        # [HYPER BOUNDARY] output redistribution",
        *_ordered_notes(form.out_notes),
        out_redist,
        "",
        "        return outputs",
    ])


# ---------------------------------------------------------------------------
# static tp_collective template
# ---------------------------------------------------------------------------

def _render_tp_collective_forward(
    source_text: str,
    func: FunctionInfo,
    form: BoundaryForm,
    deferred_bias: bool = False,
) -> str:
    """Static boundary: bare TP operators, semantics of ``RedistOp.execute``.

    The rendered body is instruction-equivalent to the generic engine with the
    TP lowerer attached:

    - identity inputs — ``self._hp_tp.to_local(name)`` (a DTensor unwraps to
      its local shard, anything else passes through);
    - TP-lowerable inputs/outputs — ``self._hp_tp.<kind>(...)`` (the bound
      operator unwraps a DTensor input and runs the collective on the tp
      group; ``None`` passes through, like the compiled plan's ``None`` skip);
    - ``_forward_impl`` is called with every declared parameter in signature
      order (the static forward re-binds names, so no ``*args`` re-pass).

    ``self._hp_tp`` is bound by ``hp_install_boundaries`` only after the
    runtime re-validates the emitted form against the live mesh; on mismatch
    the runtime replaces this forward with the generic engine, so the baked
    calls never run unvalidated.

    ``deferred_bias`` (D-22) appends ``self._hp_deferred_bias(outputs)``
    after the output ops — the Partial(sum) -> R all_reduce above is exactly
    the exit reduction the deferral waits for, so the bias re-add lands once,
    on the reduced output.
    """
    lines = [
        "        # [HYPER TP-COLLECTIVE] statically lowered boundary — the runtime",
        "        # re-validates this form at install time and replaces the forward",
        "        # with the generic engine on any mismatch.",
    ]
    for name in form.in_declared:
        op = form.in_ops.get(name)
        if op is None:
            note = form.in_notes.get(name) or f"{name}: identity ==> to_local"
            lines.append(f"        # {note}")
            lines.append(f"        {name} = self._hp_tp.to_local({name})")
        else:
            lines.append(f"        # {form.in_notes[name]}")
            lines.append(_render_tp_call(name, op))
    lines.append("")
    lines.append(
        "        # [HYPER TP-COLLECTIVE] original forward body (now ``_forward_impl``)"
    )
    lines.append(_render_forward_call(func))
    tuple_output = not returns_single_value(source_text, func)
    for name, op in form.out_ops.items():
        lines.append(f"        # {form.out_notes[name]}")
        lines.extend(_render_output_tp_call(name, op, form, tuple_output))
    if deferred_bias:
        lines.append("")
        lines.append(
            "        # [HYPER D-22] deferred bias re-added once, after the exit reduction"
        )
        lines.append("        outputs = self._hp_deferred_bias(outputs)")
    lines.append("")
    lines.append("        return outputs")
    return "\n".join(lines)


def _render_tp_call_expr(target: str, op: TransitionOp) -> str:
    """One bare-operator call expression (``self._hp_tp.<kind>(...)``)."""
    if op.kind == "all_reduce":
        return f"self._hp_tp.all_reduce({target})"
    return f"self._hp_tp.{op.kind}({target}, dim={op.tensor_dim})"


def _render_tp_call(name: str, op: TransitionOp) -> str:
    """One input-side bare-operator statement (rebinding the named input)."""
    return f"        {name} = {_render_tp_call_expr(name, op)}"


def _render_output_tp_call(
    name: str, op: TransitionOp, form: BoundaryForm, tuple_output: bool
) -> list[str]:
    """Output-side bare-operator statements.

    The common single-tensor case rewrites ``outputs`` directly.  For tuple
    outputs, the frozen plan names the first declared output as index 0;
    rebuild the original container after applying the collective to that item.
    """
    out_names = form.out_names
    if tuple_output and (name not in out_names or out_names.index(name) == 0):
        return [
            f"        _hp_output_0 = {_render_tp_call_expr('outputs[0]', op)}",
            "        outputs = (_hp_output_0, *outputs[1:])",
        ]
    return [f"        outputs = {_render_tp_call_expr('outputs', op)}"]


def _render_forward_call(func: FunctionInfo) -> str:
    """The ``self._forward_impl(...)`` call, every parameter re-passed by name.

    Positional parameters go positionally, keyword-only ones by keyword — the
    call is a pure re-pass of the static forward's own signature, so defaults
    and call-site keyword usage both survive the rewrite.
    """
    args = list(func.param_names) + [f"{name}={name}" for name in func.kwonly_names]
    if func.kwarg_name:
        args.append(f"**{func.kwarg_name}")
    if not args:
        return "        outputs = self._forward_impl()"
    single = f"        outputs = self._forward_impl({', '.join(args)})"
    if len(single) <= 100:
        return single
    lines = ["        outputs = self._forward_impl("]
    lines.extend(f"            {arg}," for arg in args)
    lines.append("        )")
    return "\n".join(lines)


def _render_local_compute(
    local: Any, input_redist: str, out_redist: str, form: BoundaryForm
) -> str:
    """Local-region boundary (shape 2, EP MoE): compute fn on local tensors, then exit.

    ``__hp_compute__`` is ``functools.partial(compute_fn, module)`` bound by
    ``hp_parallelize``. The applier calls it as ``compute_fn(*args,
    **kwargs)`` (``_wrap_local_region_forward``), so the generated forward
    forwards its own re-distributed ``*args, **kwargs`` through.  The local
    Production keeps the compute output local so the lowered boundary
    operations can run on plain tensors. If a transition cannot be lowered, the
    installed boundary owns the necessary fallback wrapping.

    The per-transition declarative notes (``form.in_notes`` / ``out_notes``)
    annotate the redistribution steps exactly like the generic and
    tp-collective templates; identity transitions carry no note, so an
    identity in/out region emits no noise.
    """
    compute_name = _callable_name(local)
    return "\n".join([
        "        # [HYPER LOCAL REGION] boundary entry",
        *_ordered_notes(form.in_notes),
        input_redist,
        "        args = tuple(hp_to_local_if_dtensor(arg) for arg in args)",
        "        kwargs = {",
        "            key: hp_to_local_if_dtensor(value)",
        "            for key, value in kwargs.items()",
        "        }",
        "",
        "        # [HYPER LOCAL REGION] run the region compute fn on local tensors",
        f"        # {compute_name} is bound to self by hp_parallelize.",
        "        outputs = self.__hp_compute__(*args, **kwargs)",
        "",
        "        # [HYPER LOCAL REGION] boundary exit",
        *_ordered_notes(form.out_notes),
        out_redist,
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
    "TP_FORM_ATTRIBUTE",
    "TP_FORM_MARKER",
    "TOGGLE_FORM_MARKER",
    "iter_emitted_forms",
    "lower_forward_boundaries",
    "lower_forward_boundaries_toggle",
    "resolve_emitted_form",
]
