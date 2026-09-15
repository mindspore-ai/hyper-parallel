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
"""Emission-time slimming of the frozen ``param_plan`` literal.

The frozen plan persisted in ``meta.param_plan`` stays full-fidelity (the
preflight, drift, and verification layers all read it); the *emitted*
``_HYPER_PARAM_PLAN`` literal is a pure, semantics-preserving projection:

* **R-axis stripping** — every per-name placement dict drops axis keys whose
  value is the canonical ``"R"`` string.  Every runtime reader resolves a
  named dict through two defaults that meet at Replicate: the parse layer
  (``parse_named_placement`` / ``_parse_named``) turns an explicit ``"R"``
  leaf into ``Replicate()``, and ``resolve_placements`` fills a *missing*
  axis with ``Replicate()`` (``boundary_forms`` readers default to ``"R"``
  either way).  A stripped literal therefore resolves to exactly the same
  placements.
* **Identity field pruning** — a boundary whose class was pruned to the
  identity form (forward untouched, no ``_hyper_boundary`` reference) drops
  its boundary fields (``in_src`` / ``in_dst`` / ``out_src`` / ``out_dst`` /
  ``is_boundary``).  ``_is_boundary_entry`` then reads the entry as a plain
  params-only record and ``hyper_install_boundaries`` skips it — correct,
  because nothing in the generated source references the unbound attribute.

Names and entries themselves are never dropped: a declared param name (even
an all-``R`` one) participates in ``distribute_tensor`` at install time, and
deleting it would silently change ``explicit Replicate DTensor`` into a plain
tensor.
"""
from __future__ import annotations

from typing import Any, Iterable

#: Entry fields that hold per-name placement dicts (``{name: {axis: "S(0)"}}``).
_PLACEMENT_FIELDS = ("params", "in_src", "in_dst", "out_src", "out_dst")

#: Boundary-identity fields pruned for identity-form classes (A2).
_BOUNDARY_FIELDS = ("in_src", "in_dst", "out_src", "out_dst", "is_boundary")

#: Boolean flags kept only when ``True`` (``False`` is the readers' default).
_FLAG_FIELDS = ("region_dispatch", "needs_cp_attn")


def strip_r_axes(named: Any) -> Any:
    """Drop ``"R"``-valued axis keys from one per-name placement dict.

    ``{'cp': 'R', 'ep': 'R', 'tp': 'S(0)'}`` becomes ``{'tp': 'S(0)'}``; an
    all-``R`` dict becomes ``{}`` (the name stays declared).  Non-dict values
    pass through untouched — the freeze layer always emits dicts here, but a
    malformed entry must not crash generation.
    """
    if not isinstance(named, dict):
        return named
    return {axis: value for axis, value in named.items() if value != "R"}


def slim_param_plan_for_emission(
    param_plan: dict[str, Any],
    identity_fqns: Iterable[str],
) -> dict[str, Any]:
    """Project the frozen ``param_plan`` into its slimmed literal form.

    Args:
        param_plan: ``meta.param_plan`` (full-fidelity, never modified).
        identity_fqns: FQNs whose boundary class was emitted as the identity
            form (forward not rewritten).  Collected from
            ``emit.parallel.iter_emitted_forms`` — the same decision path the
            emitter and preflight use — so the literal can never prune a field
            a rewritten forward still references.

    Returns:
        A new dict safe to render as the ``_HYPER_PARAM_PLAN`` literal; the
        input plan is left untouched.
    """
    identity = set(identity_fqns)
    slimmed: dict[str, Any] = {}
    for fqn, entry in param_plan.items():
        if not isinstance(entry, dict):
            slimmed[fqn] = entry
            continue
        new_entry: dict[str, Any] = {}
        for key, value in entry.items():
            if key in _PLACEMENT_FIELDS and isinstance(value, dict):
                new_entry[key] = {
                    name: strip_r_axes(named) for name, named in value.items()
                }
            else:
                new_entry[key] = value
        if fqn in identity:
            for field in _BOUNDARY_FIELDS:
                new_entry.pop(field, None)
            for flag in _FLAG_FIELDS:
                if new_entry.get(flag) is False:
                    new_entry.pop(flag, None)
        slimmed[fqn] = new_entry
    return slimmed


def collect_identity_boundary_fqns(
    source_text: str,
    frozen_plan: Any,
    *,
    boundary_classes: dict[str, str] | None = None,
    module_name: str = "",
) -> set[str]:
    """FQNs of every boundary whose emitted form is the pruned identity form.

    Reuses :func:`emit.parallel.iter_emitted_forms` — the single decision path
    shared by the emitter and the train-time preflight — so "forward was not
    rewritten" and "boundary fields were pruned from the literal" can never
    disagree.  Imported-class boundaries (``nn.Embedding`` etc.) never resolve
    to a source class and are therefore never pruned, keeping their runtime
    forward-wrapper path intact.
    """
    # Lazy imports: plan modules stay importable without the emit package (and
    # whatever it transitively requires) until slimming actually runs.
    from hyper_parallel.codegen.emit.parallel import (
        _group_boundaries_by_class,
        _plan_field,
        iter_emitted_forms,
    )
    from hyper_parallel.codegen.plan.boundary_forms import FORM_IDENTITY
    from hyper_parallel.codegen.astkit.index import build_source_index

    param_plan = _plan_field(frozen_plan, "param_plan") or {}
    if not param_plan:
        return set()
    index = build_source_index(source_text)
    groups = _group_boundaries_by_class(
        param_plan, boundary_classes, index, module_name=module_name
    )
    identity_classes = {
        class_name
        for class_name, _form, emitted, _func, _injection in iter_emitted_forms(
            source_text,
            frozen_plan,
            boundary_classes=boundary_classes,
            module_name=module_name,
        )
        if emitted == FORM_IDENTITY
    }
    return {
        fqn
        for class_name in identity_classes
        for fqn, _entry in groups.get(class_name, ())
    }
