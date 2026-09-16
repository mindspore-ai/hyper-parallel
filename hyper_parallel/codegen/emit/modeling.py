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
"""Assemble the generated HF-style modeling file.

The generated file is the original modeling source, verbatim, with three
appended layers:

* a header banner recording where it came from,
* the codegen runtime import (``hyper_parallel.codegen.runtime``),
* the frozen literal plan (``_HYPER_PARAM_PLAN`` / ``_HYPER_TIED_PAIRS`` /
  ``_HYPER_MESH_DIM_NAMES`` / ``_HYPER_SPECIAL_HANDLERS``) and the
  ``hyper_parallelize(model, mesh_context)`` entry the runtime dispatches on.

Boundary forwards and module replacements are lowered into the copied source.
The preflight import checks that the assembled module remains importable in the
training environment.
"""
from __future__ import annotations

import io
import os
import tokenize
from typing import Any

# ---------------------------------------------------------------------------
# literal rendering
# ---------------------------------------------------------------------------

def render_python_literal(obj: Any) -> str:
    """Serialize ``obj`` to a deterministic, stable Python literal.

    Determinism matters: the generated file is hashed and recorded in meta,
    so the same frozen plan must always render to the same bytes.  Dict keys
    are sorted; container nesting is indented (readable diff), empty
    containers collapse to a single line.  Only JSON-safe values are expected
    (the freeze layer already produced them); anything else falls back to
    ``repr``.
    """
    return _render(obj, 0)


def _render(obj: Any, depth: int) -> str:
    if obj is None:
        return "None"
    if obj is True:
        return "True"
    if obj is False:
        return "False"
    if isinstance(obj, str):
        return repr(obj)
    if isinstance(obj, (int, float)):
        return repr(obj)
    if isinstance(obj, dict):
        return _render_dict(obj, depth)
    if isinstance(obj, (list, tuple)):
        return _render_seq(obj, depth)
    # Unknown object: fall back to repr (the freeze layer should not produce
    # these, but an exotic value must not crash generation).
    return repr(obj)


def _render_dict(obj: dict, depth: int) -> str:
    if not obj:
        return "{}"
    items = sorted(obj.items(), key=lambda kv: _sort_key(kv[0]))
    if all(_is_atomic(v) for _, v in items):
        inner = ", ".join(
            f"{_render(k, depth)}: {_render(v, depth)}" for k, v in items
        )
        return "{" + inner + "}"
    pad = " " * (4 * (depth + 1))
    lines = ["{"]
    for k, v in items:
        lines.append(f"{pad}{_render(k, depth + 1)}: {_render(v, depth + 1)},")
    lines.append(" " * (4 * depth) + "}")
    return "\n".join(lines)


def _render_seq(obj, depth: int) -> str:
    if not obj:
        return "()" if isinstance(obj, tuple) else "[]"
    pad = " " * (4 * (depth + 1))
    lines = ["[" if isinstance(obj, list) else "("]
    for item in obj:
        lines.append(f"{pad}{_render(item, depth + 1)},")
    lines.append(" " * (4 * depth) + ("]" if isinstance(obj, list) else ")"))
    return "\n".join(lines)


def _sort_key(key: Any):
    # Dict keys are expected to be strings; sort strings by value so the
    # output is byte-stable.  Non-string keys sort after strings.
    if isinstance(key, str):
        return (0, key)
    return (1, repr(key))


def _is_leaf(value: Any) -> bool:
    """A scalar or an empty container — carries no visible nesting."""
    if isinstance(value, (dict, list, tuple)):
        return not value
    return True


def _is_atomic(value: Any) -> bool:
    """Whether a dict value renders on one line.

    Scalars and empty containers always do; a non-empty dict does when every
    one of its values is itself a scalar or an empty container — the one-level
    shape of an R-stripped entry (``{'params': {'weight': {}}}``), which keeps
    the slimmed literal one-entry-per-line.  A dict holding a non-empty
    container (``{'weight': {'tp': 'S(0)'}}``) still nests.
    """
    if isinstance(value, dict) and value:
        return all(_is_leaf(v) for v in value.values())
    return _is_leaf(value)


# ---------------------------------------------------------------------------
# source copy
# ---------------------------------------------------------------------------

def copy_original_modeling(source: Any) -> str:
    """Read the original modeling source text from a ``ResolvedSource``-like.

    ``source`` is ``meta.source`` — the ``ResolvedSource.to_dict()`` form with
    a ``file_path`` key.  Accepts either a dict or a ``ResolvedSource`` (the
    dict form is what the manager records, but the helper tolerates the live
    dataclass so it stays usable outside the meta path).
    """
    if isinstance(source, dict):
        path = source.get("file_path")
    else:
        path = getattr(source, "file_path", None)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"codegen emit: original modeling source is not a readable file "
            f"({path!r}); generation must run where the model's modeling file "
            "exists"
        )
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def rewrite_relative_imports(text: str, module_name: str) -> str:
    """Rewrite ``from .xxx`` / ``from ...xxx`` imports to absolute imports.

    HF modeling files commonly import siblings with a relative spec, e.g.
    ``from .configuration_qwen3 import Qwen3Config`` or ``from ...activations
    import ACT2FN``.  The generated bundle is loaded by ``spec_from_file_location``
    under a synthetic package (``hyper_parallel_generated.<digest>.<stem>``), so a
    relative spec resolves against that synthetic package instead of the real
    transformers package and fails with ``attempted relative import beyond
    top-level package``.

    This anchors the relative spec on ``module_name`` (the original module's
    real dotted path, e.g. ``transformers.models.qwen3.modeling_qwen3``): N
    leading dots climb N-1 package levels above the module's parent package, then
    append the rest.  ``from ...activations`` → ``from transformers.activations``
    and ``from .configuration_qwen3`` →
    ``from transformers.models.qwen3.configuration_qwen3``.

    ``module_name`` is ``"<remote>"`` for trusted remote code, which has no real
    package to anchor on — the rewrite is a no-op there (import handling for
    remote code keeps its relative imports). A non-dotted or unanchorable ``module_name``
    is likewise left untouched.

    Only the module-spec part of each relative import is rewritten; the symbol
    list, comments, and whitespace are preserved.  It uses ``tokenize`` rather
    than a regex so relative-looking text inside string literals or comments is
    never touched.
    """
    unless_anchor = _relative_anchor(module_name)
    if unless_anchor is None:
        return text

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except tokenize.TokenError:
        return text
    if not tokens:
        return text

    edits: list[tuple[int, int, str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        tk = tokens[i]
        if tk.type != tokenize.NAME or tk.string != "from":
            i += 1
            continue
        # the spec must start with a relative (dots) operator
        j = i + 1
        while j < n and tokens[j].type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            j += 1
        if j >= n or tokens[j].type != tokenize.OP or not set(tokens[j].string) <= {"."}:
            i += 1
            continue
        dots_tok = tokens[j]
        dots = len(dots_tok.string)

        # collect the relative module path: NAME ('.' NAME)*
        k = j + 1
        path: list[str] = []
        last_path_tok = None
        while k < n:
            t = tokens[k]
            if t.type == tokenize.NAME and t.string != "import":
                path.append(t.string)
                last_path_tok = t
                k += 1
                if k < n and tokens[k].type == tokenize.OP and tokens[k].string == ".":
                    k += 1  # consume the separator; import follows next round
                else:
                    break
            else:
                break

        # the path must be followed by the ``import`` keyword (allow a comment
        # line in between; a multiline symbol list comes *after* ``import``).
        m = k
        while m < n and tokens[m].type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            m += 1
        if m >= n or tokens[m].type != tokenize.NAME or tokens[m].string != "import":
            i += 1
            continue

        abs_spec = _absolute_spec(unless_anchor, dots, path)
        if abs_spec is None:
            i += 1
            continue

        start = dots_tok.start
        end = last_path_tok.end if last_path_tok is not None else dots_tok.end
        edits.append((start, end, abs_spec))
        i = k + 1

    if not edits:
        return text
    return _splice(text, edits)


def _relative_anchor(module_name: str) -> list[str] | None:
    """Return the dotted package segments a relative spec anchors on, or None.

    A real installed-module path (``transformers.models.qwen3.modeling_qwen3``)
    anchors.  The ``"<remote>"`` sentinel (trusted remote code) and any
    non-dotted / placeholder-laden name cannot anchor — the caller leaves such
    imports untouched.
    """
    if not isinstance(module_name, str) or not module_name:
        return None
    if "<" in module_name or ">" in module_name:
        return None
    segs = module_name.split(".")
    if len(segs) < 2 or any(not seg for seg in segs):
        return None
    return segs


def _absolute_spec(anchor: list[str], dots: int, path: list[str]) -> str | None:
    """Compute the absolute import spec for one relative import, or None.

    ``anchor`` is the module's package segments (its path minus the final
    module segment).  ``dots`` leading dots climb ``dots - 1`` package levels
    above the module's parent package, then append ``path``.  Returns None when
    the relative import climbs above the package root (invalid source — leave
    it untouched rather than corrupt it).
    """
    parent = anchor[:-1]
    climb = dots - 1
    if climb > len(parent):
        return None
    base_parts = parent if climb <= 0 else parent[: len(parent) - climb]
    parts = base_parts + path
    return ".".join(part for part in parts if part)


def _splice(text: str, edits: list[tuple[tuple[int, int], tuple[int, int], str]]) -> str:
    """Splice absolute specs back into ``text`` by token (row, col) positions.

    Edits are applied last-first so the earlier offsets stay valid.  A single
    contiguous span is replaced in every case: the relative spec starts at the
    dots token and ends at the last module-path token (or the dots token when
    there is no path, i.e. ``from . import``).
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    line_offsets = []
    acc = 0
    for line in lines:
        line_offsets.append(acc)
        acc += len(line)

    def offset(row: int, col: int) -> int:
        if 1 <= row <= len(line_offsets):
            return line_offsets[row - 1] + col
        return acc

    positioned = sorted(
        ((offset(*start), offset(*end), new) for start, end, new in edits),
        key=lambda item: item[0],
        reverse=True,
    )
    out = text
    for start, end, new in positioned:
        if start < end <= len(out):
            out = out[:start] + new + out[end:]
    return out


# ---------------------------------------------------------------------------
# import injection
# ---------------------------------------------------------------------------

def inject_codegen_imports(text: str, *, has_region: bool = False) -> str:
    """Append the codegen runtime import after the source text.

    The import list is the fixed set the generated entry point and the lowered
    forwards need: the install / literal-plan helpers, plus
    ``hyper_to_local_if_dtensor`` when the artifact lowers a local-region
    boundary.  The lowered forwards bind their compiled plans as instance
    attributes (``hyper_install_boundaries``), so no forward-time runtime
    helper — and no per-boundary global constant — appears in the file.

    The runtime functions are imported lazily-by-module inside the helpers, so
    the generated module only pulls torch/transformers when ``hyper_parallelize``
    actually runs (generation-time hosts import the generated file for the
    preflight gate without materializing a sharding plan).
    """
    names = [
        "hyper_apply_inner_wrapper",
        "hyper_apply_replacements",
        "hyper_apply_special_handlers",
        "hyper_bind_compute",
        "hyper_build_tp_grad_info",
        "hyper_expand_injections",
        "hyper_install_boundaries",
        "hyper_replicate_tied",
        "hyper_shard_params",
    ]
    if has_region:
        names.append("hyper_to_local_if_dtensor")
    joined = ",\n".join(f"    {name}" for name in names)
    block = f"from hyper_parallel.codegen.runtime import (\n{joined},\n)"
    return text.rstrip() + "\n\n" + block + "\n"


# ---------------------------------------------------------------------------
# literal + entry injection
# ---------------------------------------------------------------------------

def inject_param_plan_literals(text: str, frozen_plan: Any) -> str:
    """Append the frozen literal plan constants after ``text``.

    ``frozen_plan`` is ``meta.param_plan`` (the ``freeze_param_plan`` dict) —
    the injected constants mirror the keys the generated ``hyper_parallelize``
    reads: the sharding plan, tied pairs, the plan's active mesh axes, the
    special handlers, and the per-boundary injections (CP inner wrappers / EP
    local compute). An empty plan/axes/handlers/injections still
    renders — the runtime treats empty as no-op, and leaving them out would
    make the generated file silently depend on what happened to be there.

    The ``param_plan`` and ``injections`` literals are slimmed projections of
    the frozen fields (see :mod:`hyper_parallel.codegen.plan.slim` and
    :func:`group_injection_rules`): R-valued axes are stripped, identity-form
    boundaries drop their boundary fields, and same-shape injection rules are
    grouped under one ``match`` list.  ``meta`` keeps the full-fidelity form —
    the emitted projection is a pure function of it, so re-emission (drift
    check) reproduces identical bytes.
    """
    param_plan = _slim_param_plan_for_literal(text, frozen_plan)
    plan = (
        "# Frozen sharding plan: the generated model executes\n"
        "# exactly this literal contract; the planner/applier are NOT re-run\n"
        "# at runtime.\n"
        "# [HYPER PARALLEL ENTRY] hyper_parallelize\n"
        "_HYPER_PARAM_PLAN = " + render_python_literal(param_plan) + "\n"
    )
    plan += "_HYPER_TIED_PAIRS = " + render_python_literal(
        _plan_field(frozen_plan, "tied_pairs")
    ) + "\n"
    plan += "_HYPER_MESH_DIM_NAMES = " + render_python_literal(
        _plan_field(frozen_plan, "mesh_dim_names")
    ) + "\n"
    plan += "_HYPER_SPECIAL_HANDLERS = " + render_python_literal(
        _plan_field(frozen_plan, "special_handlers")
    ) + "\n"
    plan += "_HYPER_INJECTIONS = " + render_python_literal(
        group_injection_rules(_plan_field(frozen_plan, "injections"))
    ) + "\n"
    plan += "_CODEGEN_BOUNDARY_MANIFEST = " + render_python_literal(
        build_boundary_manifest(frozen_plan)
    ) + "\n"
    return text.rstrip() + "\n\n" + plan


def _slim_param_plan_for_literal(text: str, frozen_plan: Any) -> dict[str, Any]:
    """The emitted ``_HYPER_PARAM_PLAN`` literal value (slimmed projection).

    Full plan when no boundary class resolves (no ``boundary_classes`` map —
    a legacy or hand-built meta): slimming identity fields depends on the
    emitter's form decision, which is unavailable, so the literal keeps the
    frozen shape rather than guessing.  R-axis stripping still applies — it
    is decision-free and provably neutral for every reader.
    """
    from hyper_parallel.codegen.plan.slim import (  # pylint: disable=C0415
        collect_identity_boundary_fqns,
        slim_param_plan_for_emission,
    )

    param_plan = _plan_field(frozen_plan, "param_plan")
    if not param_plan:
        return param_plan
    boundary_classes = _plan_attr(frozen_plan, "boundary_classes") or {}
    module_name = _source_module_name(frozen_plan)
    identity_fqns = (
        collect_identity_boundary_fqns(
            text,
            frozen_plan,
            boundary_classes=boundary_classes,
            module_name=module_name,
        )
        if boundary_classes
        else ()
    )
    return slim_param_plan_for_emission(param_plan, identity_fqns)


def _plan_attr(plan: Any, name: str) -> Any:
    """Read one attribute off a dict-or-object frozen plan (no empty default)."""
    if isinstance(plan, dict):
        return plan.get(name)
    return getattr(plan, name, None)


def _source_module_name(frozen_plan: Any) -> str:
    """The modeling source's module_name, from either meta shape."""
    source = _plan_attr(frozen_plan, "source")
    if isinstance(source, dict):
        return source.get("module_name") or ""
    return getattr(source, "module_name", "") or ""


def group_injection_rules(rules: Any) -> list[dict[str, Any]]:
    """Group same-shape frozen injection rules under one ``match`` list.

    The frozen ``injections`` list carries one rule per boundary FQN; rules
    that differ only in ``match`` (every layer's ``self_attn`` CP wrapper,
    every layer's ``mlp`` EP compute) collapse into one record whose
    ``match`` is the sorted FQN list.  The generated entry expands the list
    back to per-FQN rules with ``runtime.hyper_expand_injections`` before any
    helper consumes it, so every runtime reader keeps its per-FQN contract.

    Grouping key is the full rule payload minus ``match`` — two rules with
    equal payloads but different match patterns are still distinct specs and
    stay separate records.
    """
    if not rules:
        return []
    groups: dict[str, dict[str, Any]] = {}
    ungrouped: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        match = rule.get("match")
        if not match:
            # A rule without ``match`` is malformed frozen data — the runtime
            # helpers fail fast on it; pass it through untouched so the error
            # surfaces exactly as it would have ungrouped.
            ungrouped.append(rule)
            continue
        payload = {key: value for key, value in rule.items() if key != "match"}
        key = _render(payload, 0)  # stable, JSON-safe canonical form
        group = groups.setdefault(key, {"payload": payload, "matches": []})
        group["matches"].append(match)
    grouped: list[dict[str, Any]] = ungrouped
    for key in sorted(groups):
        group = groups[key]
        record = dict(group["payload"])
        record["match"] = sorted(group["matches"])
        grouped.append(record)
    return grouped


def build_boundary_manifest(frozen_plan: Any) -> dict[str, Any]:
    """Build the human-readable production communication manifest.

    ``_HYPER_PARAM_PLAN`` remains the machine contract consumed by runtime.
    This projection is deliberately redundant and reader-oriented: generated
    artifacts should tell users which production collectives or EP/CP regions
    a boundary represents without making them reverse-engineer placement dicts.
    """
    from hyper_parallel.codegen.plan.boundary_forms import (  # pylint: disable=C0415
        classify_boundary_form,
    )

    param_plan = _plan_field(frozen_plan, "param_plan")
    if not param_plan:
        return {}
    mesh_dim_names = _plan_field(frozen_plan, "mesh_dim_names")
    boundary_classes = _plan_attr(frozen_plan, "boundary_classes") or {}
    injections = _index_injections_by_fqn(_plan_field(frozen_plan, "injections"))
    manifest: dict[str, Any] = {}
    for fqn, entry in sorted(param_plan.items()):
        if not isinstance(entry, dict) or not _is_boundary_entry(entry):
            continue
        injection = injections.get(fqn)
        form = classify_boundary_form(entry, mesh_dim_names, injection)
        record = {
            "class": boundary_classes.get(fqn),
            "form": _manifest_form(form.form, injection),
            "inputs": _manifest_notes(form.in_notes),
            "outputs": _manifest_notes(form.out_notes),
        }
        production_ops = _manifest_production_ops(form.in_notes, form.out_notes)
        if production_ops:
            record["production_ops"] = production_ops
        injection_note = _manifest_injection(injection)
        if injection_note:
            record["injection"] = injection_note
        manifest[fqn] = {key: value for key, value in record.items() if value}
    return manifest


def _index_injections_by_fqn(rules: Any) -> dict[str, dict[str, Any]]:
    """Return one injection payload per concrete boundary FQN."""
    by_fqn: dict[str, dict[str, Any]] = {}
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        match = rule.get("match")
        matches = match if isinstance(match, list) else [match]
        for fqn in matches:
            if isinstance(fqn, str) and fqn:
                by_fqn[fqn] = rule
    return by_fqn


def _is_boundary_entry(entry: dict[str, Any]) -> bool:
    """Whether a param-plan entry carries boundary communication."""
    if entry.get("is_boundary") is not None:
        return bool(entry["is_boundary"])
    return any(entry.get(field) for field in ("in_src", "in_dst", "out_src", "out_dst"))


def _manifest_form(form: str, injection: Any) -> str:
    """Name the form as a user-facing artifact category."""
    if injection and injection.get("local_compute_fn") is not None:
        return "ep_region"
    if injection and injection.get("inner_wrapper") is not None:
        return "cp_inner_wrapper"
    return form


def _manifest_notes(notes: dict[str, str]) -> list[str]:
    """Deterministic list of communication notes for one boundary side."""
    return [notes[name] for name in sorted(notes)]


def _manifest_production_ops(*note_maps: dict[str, str]) -> list[str]:
    """Extract concise collective names from manifest notes."""
    found: list[str] = []
    for notes in note_maps:
        for note in notes.values():
            for op_name in ("all_gather", "reduce_scatter", "all_reduce"):
                if op_name in note and op_name not in found:
                    found.append(op_name)
    return found


def _manifest_injection(injection: Any) -> dict[str, Any]:
    """Reader-facing description of local compute or inner wrapper injection."""
    if not isinstance(injection, dict):
        return {}
    if injection.get("local_compute_fn") is not None:
        return {
            "kind": "ep_local_compute",
            "target": _callable_target(injection.get("local_compute_fn")),
            "flow": "router -> all_to_all dispatch -> local experts -> all_to_all combine",
        }
    if injection.get("inner_wrapper") is not None:
        return {
            "kind": "cp_inner_wrapper",
            "target": _callable_target(injection.get("inner_wrapper")),
            "flow": "attention K/V all-gather and CP-aware causal mask",
        }
    return {}


def _callable_target(value: Any) -> str:
    """String target from a frozen callable literal."""
    if isinstance(value, dict):
        return value.get("_target_") or value.get("target") or repr(value)
    if isinstance(value, str):
        return value
    return repr(value)


#: Empty value per frozen-plan field.  ``mesh_dim_names`` is deliberately
#: ``()`` and never ``None``: the manager stores ``None`` for "no active axes"
#: (``list(...) or None``), but for the runtime ``None`` means to keep the
#: full mesh unsliced (``hyper_shard_params`` docstring).
#: Rendering ``None`` into the generated file would silently turn "this plan
#: shards on no axis" into "shard against every axis including dp".
_PLAN_FIELD_EMPTY = {
    "param_plan": {},
    "tied_pairs": [],
    "special_handlers": {},
    "mesh_dim_names": (),
    "injections": [],
}


def _plan_field(plan: Any, name: str) -> Any:
    """Read one frozen-plan field from a dict or a ``CodegenMeta``.

    The manager passes ``meta`` around (a ``CodegenMeta``), which already
    carries the frozen fields, so the helper accepts either shape. A missing or ``None``
    value collapses to the field's empty form (see ``_PLAN_FIELD_EMPTY``)
    rather than being rendered as ``None``.
    """
    empty = _PLAN_FIELD_EMPTY.get(name, None)
    if isinstance(plan, dict):
        value = plan.get(name)
    else:
        value = getattr(plan, name, None)
    return empty if value is None else value


def inject_hyper_parallelize(text: str, frozen_plan: Any) -> str:
    """Append the module-level ``hyper_parallelize`` entry point.

    The runtime dispatches on ``meta.entrypoints["parallelize"]``
    (``hyper_parallelize``) and calls it as ``entry(model, mesh_context)``
    (runtime.py ``parallelize_from_generated``).  It calls the runtime helpers
    in semantic order — shard params, special handlers, tied-weight
    replication, one-shot boundary compilation/instance binding
    (``hyper_install_boundaries``), forward binding (``hyper_bind_compute``
    for the shape-2 EP local region, ``hyper_apply_inner_wrapper`` for the
    shape-3 CP inner wrapper), then build ``tp_grad_info`` LAST (that is the
    one-shot ``_local_params_context`` unwrap; after it the model's DTensor
    params are permanently plain locals, so nothing may shard after it).
    """
    body = (
        "\n"
        "def hyper_parallelize(model, mesh_context):\n"
        "    \"\"\"Execute the frozen parallel plan recorded at generation time.\n"
        "\n"
        "    Called by ``parallelize_from_generated`` (codegen runtime) when\n"
        "    the artifact's ``covered.sharding_plan`` gate is True.  Returns\n"
        "    ``tp_grad_info`` with the same semantics as ``apply_sharding_plan``.\n"
        "\n"
        "    Boundary forwards read their compiled plan from the instance\n"
        "    attribute ``_hyper_boundary`` installed here — the generated file\n"
        "    publishes no module globals.\n"
        "    \"\"\"\n"
        "    hyper_shard_params(\n"
        "        model, _HYPER_PARAM_PLAN, mesh_context, _HYPER_MESH_DIM_NAMES\n"
        "    )\n"
        "    hyper_apply_special_handlers(\n"
        "        model, _HYPER_SPECIAL_HANDLERS, mesh_context, _HYPER_MESH_DIM_NAMES\n"
        "    )\n"
        "    hyper_replicate_tied(model, _HYPER_TIED_PAIRS)\n"
        "    injections = hyper_expand_injections(_HYPER_INJECTIONS)\n"
        "    hyper_install_boundaries(\n"
        "        model, _HYPER_PARAM_PLAN, mesh_context, _HYPER_MESH_DIM_NAMES,\n"
        "        injections=injections\n"
        "    )\n"
        "    hyper_bind_compute(\n"
        "        model, injections, mesh_context, _HYPER_MESH_DIM_NAMES,\n"
        "        param_plan=_HYPER_PARAM_PLAN\n"
        "    )\n"
        "    hyper_apply_inner_wrapper(\n"
        "        model, injections, mesh_context, _HYPER_MESH_DIM_NAMES,\n"
        "        param_plan=_HYPER_PARAM_PLAN\n"
        "    )\n"
        "    return hyper_build_tp_grad_info(\n"
        "        model, _HYPER_PARAM_PLAN, mesh_context, tied_pairs=_HYPER_TIED_PAIRS\n"
        "    )\n"
    )
    return text.rstrip() + "\n" + body


# ---------------------------------------------------------------------------
# file assembly
# ---------------------------------------------------------------------------

def emit_modeling_file(meta: Any) -> str:
    """Assemble the full generated modeling file text.

    Returns the text (not written to disk) so the caller can compute a diff
    first. The header banner carries the generation provenance;
    the source is copied verbatim; then the runtime import, literals, and the
    parallel entry are appended.
    """
    text = copy_original_modeling(meta.source)
    # Stale-source sanitize: a trust-remote-code modeling file ships with the
    # checkpoint and reflects the author's transformers, not the installed one.
    # Removing ``is_torch_fx_available`` (a transformers>=5.0 casualty) here
    # keeps the artifact importable without touching the source file. No-op
    # when the source does not import the symbol.
    from hyper_parallel.codegen.source.compat import sanitize_source_compat

    text = sanitize_source_compat(text)
    module_name = (
        meta.source.get("module_name")
        if isinstance(meta.source, dict)
        else getattr(meta.source, "module_name", "")
    )
    text = rewrite_relative_imports(text, module_name)
    inline_text = _try_inline_modeling(text, meta)
    if inline_text is not None:
        inline_text = _lower_forward_boundaries(inline_text, meta, module_name)
        return _banner(meta) + inline_text
    text = _lower_forward_boundaries(text, meta, module_name)
    # The entry class's ``__init__`` tail gets the
    # ``hyper_apply_replacements(self)`` call so replacements install as soon
    # as a model instance is built.  Runs on the source BEFORE the banner /
    # import / literal appends — all of which shift byte offsets.
    text = _rewrite_init_for_overrides(text, meta)
    text = _banner(meta) + text
    text = inject_codegen_imports(text, has_region=_meta_has_region(meta))
    text = inject_param_plan_literals(text, meta)
    text = _inject_module_override_literals(text, meta)
    text = inject_hyper_parallelize(text, meta)
    return text


def _try_inline_modeling(text: str, meta: Any) -> str | None:
    """Run the YAML-driven inline pipeline when it is explicitly enabled."""

    from hyper_parallel.codegen.inline import try_render_inline_modeling

    source = getattr(meta, "source", None)
    architecture = source.get("architecture") if isinstance(source, dict) else getattr(source, "architecture", None)
    model_type = getattr(meta, "model_class", None) or architecture
    return try_render_inline_modeling(text, meta, model_type=model_type)


def _meta_has_region(meta: Any) -> bool:
    """Whether any frozen injection declares a local-region compute fn.

    Drives ``inject_codegen_imports``: only a local-region artifact needs the
    ``hyper_to_local_if_dtensor`` helper import in the generated file.
    """
    return any(
        rule.get("local_compute_fn") is not None
        for rule in (getattr(meta, "injections", None) or [])
    )


def _lower_forward_boundaries(text: str, meta: Any, module_name: str) -> str:
    """Sink boundary wrappers into the model's ``forward`` bodies.

    Each boundary class's ``forward`` is rewritten to redistribute through
    the instance-bound compiled plan and run the region's compute
    explicitly. A boundary whose class cannot be resolved or
    whose per-class contract differs across FQNs fails fast — a silently
    dropped redistribution would change the model's numeric behavior.

    Fresh on the unmodified source text, so offsets stay valid for the whole
    call.  The ``local_compute_fn`` / ``inner_wrapper`` members the rewritten
    forward references are bound by ``hyper_parallelize``, so the
    generated file stays runtime-importable without a mesh.

    ``boundary_classes`` is the FQN -> class-name map frozen at freeze time.  A
    legacy or hand-built plan may have no such map; without
    it the lowerer cannot resolve any boundary FQN to its owning class, so it
    would fail fast on the first boundary instead of degrading to the earlier
    behavior (params sharded by ``hyper_shard_params``, no forward rewrite).
    When the map is absent, the source remains untouched and runtime dispatch
    handles the boundaries.
    """
    from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries

    boundary_classes = getattr(meta, "boundary_classes", None) or {}
    if not boundary_classes:
        return text
    return lower_forward_boundaries(
        text,
        meta,
        boundary_classes=boundary_classes,
        module_name=module_name,
    )


def _rewrite_init_for_overrides(text: str, meta: Any) -> str:
    """Insert the runtime replacement call into the entry class's ``__init__``.

    No-op when no module override was sunk into ``meta``.  The entry class is
    ``meta.model_class`` (the architecture name); ``rewrite_init_for_overrides``
    fails fast when that class or its ``__init__`` cannot be located, so a
    replacement rule never silently disappears from the artifact.
    """
    from hyper_parallel.codegen.emit.replacement import rewrite_init_for_overrides

    overrides = getattr(meta, "module_overrides", None) or []
    if not overrides:
        return text
    model_class = getattr(meta, "model_class", None)
    if not model_class:
        raise RuntimeError(
            "codegen: meta carries module_overrides but no model_class to "
            "rewrite the entry __init__ for"
        )
    return rewrite_init_for_overrides(
        text, overrides, model_class=model_class,
    )


def _inject_module_override_literals(text: str, meta: Any) -> str:
    """Append the ``_HYPER_MODULE_OVERRIDES`` literal (see ``emit.replacement``).

    Matching the lazy-import pattern of ``_rewrite_init_for_overrides``: the
    literal renderer lives in ``emit.replacement`` and is only pulled in when
    there is actually something to render (a bare no-override meta does not need
    the module at all).
    """
    from hyper_parallel.codegen.emit.replacement import inject_module_override_literals

    overrides = getattr(meta, "module_overrides", None) or []
    if not overrides:
        return text
    return inject_module_override_literals(text, overrides)


def _banner(meta: Any) -> str:
    """Render the provenance banner comment."""
    source = meta.source if isinstance(meta.source, dict) else _source_dict(meta.source)
    pd = meta.parallel_dims or {}
    lines = [
        "# Generated by codegen. DO NOT EDIT MANUALLY.",
        "#",
        f"#   config       : {getattr(meta, 'yaml_path', '')}",
        f"#   signature    : {getattr(meta, 'signature', '')}",
        f"#   source       : {source.get('module_name', '')}",
        f"#   source sha256: {source.get('sha256', '')}",
        f"#   architecture : {source.get('architecture', '')}",
        "#   parallel_dims: tp={tp} cp={cp} ep={ep} pp={pp} sp={sp} loss_parallel={lp}".format(
            tp=pd.get("tp_size", 1),
            cp=pd.get("cp_size", 1),
            ep=pd.get("ep_size", 1),
            pp=pd.get("pp_size", 1),
            sp=pd.get("sequence_parallel", False),
            lp=pd.get("loss_parallel", False),
        ),
        "#   injections   : {n} rule(s)".format(n=len(getattr(meta, "injections", []) or [])),
        "#   runtime      : mesh / FSDP2 / PP / AC / compile still handled by runtime",
        "#",
        "# [HYPER PARAM PLAN] the following literals are the frozen sharding",
        "# contract.  Do not hand-edit; regenerate to change.",
        "#",
        "",
    ]
    return "\n".join(lines)


def _source_dict(source: Any) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, dict):
        return source
    return {
        "module_name": getattr(source, "module_name", ""),
        "file_path": getattr(source, "file_path", ""),
        "architecture": getattr(source, "architecture", ""),
        "sha256": getattr(source, "sha256", ""),
    }


__all__ = [
    "build_boundary_manifest",
    "copy_original_modeling",
    "emit_modeling_file",
    "inject_codegen_imports",
    "inject_hyper_parallelize",
    "inject_param_plan_literals",
    "render_python_literal",
    "rewrite_relative_imports",
]
