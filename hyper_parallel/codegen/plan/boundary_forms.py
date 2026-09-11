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
"""Classify each frozen boundary into the forward form the emitter renders.

Single source of truth shared by generation (``emit/parallel``), install-time
validation (``runtime``), and preflight (``check/preflight``).  The per-tensor
transition judgment itself lives in
``distributed._builder.tp_collective_lowering.classify_tp_transition`` — this
module only folds a whole frozen boundary entry (its ``in_src``/``in_dst``/
``out_src``/``out_dst`` named placements) into one of four forms:

- ``region``        — the boundary has a ``local_compute_fn`` injection (EP
  MoE local region): the forward delegates to ``self.__hyper_compute__`` and
  re-wraps locals per ``out_src`` before the boundary exit.
- ``identity``      — every declared transition is identity on the plan's
  active axes AND no inner-wrapper injection is declared: the class forward
  is NOT rewritten (pruned); the runtime install path covers the exact no-op
  / to-local semantics with its generic wrapper.
- ``generic`` (forced) — a boundary with an ``inner_wrapper`` injection is
  generic even when every transition is identity: the CP wrapper is applied
  to the module at runtime by ``hyper_apply_inner_wrapper``, so the class
  must keep the rewritten redistribute forward that P1 always emitted.
- ``tp_collective`` — every declared transition is identity or TP-lowerable
  (``all_gather`` / ``all_reduce`` / ``reduce_scatter`` on the tp axis): the
  emitter may sink bare operator calls (``self._hyper_tp.all_gather(...)``),
  subject to source-side structural gates (see ``emit/parallel``); the
  runtime re-validates against the live mesh at install time and falls back
  to the generic engine when the compiled plan disagrees.
- ``generic``       — anything else (non-tp axis differences, unsupported
  placement pairs, expert-mesh routing): the forward is rewritten to call
  the instance-bound compiled plan (``self._hyper_boundary``) with a
  declarative comment annotation of what each side does.

The classification is a pure function of the frozen entry, the plan's active
axis names, and the (optional) injection rule — the same inputs the drift
check re-emits from, so generation, install, and preflight always agree.

The runtime-conditional facts (live mesh axes, rank order, backend, expert
mesh) are deliberately NOT classified here: the emit-time decision is
optimistic and the runtime's install-time validation is the safety net.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: Region form: local-compute injection owns the forward (EP MoE).
FORM_REGION = "region"
#: Identity form: prune the rewrite entirely; runtime wrapper covers it.
FORM_IDENTITY = "identity"
#: TP-collective form: bare operator calls (source gates permitting).
FORM_TP_COLLECTIVE = "tp_collective"
#: Generic form: annotated redistribute through the instance-bound plan.
FORM_GENERIC = "generic"

#: Placement fields of a frozen boundary entry, per side.
_SIDE_FIELDS = (("in_src", "in_dst"), ("out_src", "out_dst"))


@dataclass(frozen=True)
class TransitionOp:
    """One TP-lowerable placement transition, as the emitter renders it."""

    #: ``all_gather`` / ``all_reduce`` / ``reduce_scatter``.
    kind: str
    #: Tensor dim the collective gathers / scatters on (``None`` for all_reduce).
    tensor_dim: Optional[int] = None


@dataclass(frozen=True)
class BoundaryForm:
    """The classified form of one frozen boundary entry.

    ``in_ops`` / ``out_ops`` carry the non-identity TP-lowerable transitions
    (name -> op); ``in_identity`` lists declared input names whose transition
    is identity (their compiled semantics is an unwrap-to-local, which the
    static form renders as ``self._hyper_tp.to_local(name)``); the ``*_notes``
    dicts carry one declarative annotation line per non-identity transition,
    keyed by name, for the generic form's comment block.
    """

    form: str
    in_ops: dict[str, TransitionOp] = field(default_factory=dict)
    out_ops: dict[str, TransitionOp] = field(default_factory=dict)
    in_identity: tuple[str, ...] = ()
    in_notes: dict[str, str] = field(default_factory=dict)
    out_notes: dict[str, str] = field(default_factory=dict)
    #: Every declared name on each side (identity or not).
    in_declared: tuple[str, ...] = ()
    out_declared: tuple[str, ...] = ()
    #: The entry's declared output order (``out_names`` or ``out_src`` keys).
    out_names: tuple[str, ...] = ()


def classify_boundary_form(
    entry: dict[str, Any],
    mesh_dim_names: Optional[Any],
    injection: Optional[dict[str, Any]] = None,
) -> BoundaryForm:
    """Classify one frozen ``param_plan`` boundary entry into its emit form.

    Args:
        entry: The frozen boundary entry (placements as canonical strings,
            e.g. ``{"tp": "S(1)"}``).
        mesh_dim_names: The plan's active axis names (frozen offline-mesh
            order).  ``None`` / empty means the plan shards on no axis.
        injection: The boundary's frozen injection rule, if any — a
            ``local_compute_fn`` makes the form ``region``; an ``inner_wrapper``
            forces ``generic`` (the CP wrapper mutates the forward at runtime,
            so the class keeps the rewritten redistribute form).

    Returns:
        The :class:`BoundaryForm`; ``form`` is one of ``region`` /
        ``identity`` / ``tp_collective`` / ``generic``.
    """
    if injection is not None and injection.get("local_compute_fn") is not None:
        return BoundaryForm(form=FORM_REGION)
    # An inner-wrapper boundary keeps the generic rewrite even when every
    # transition is identity: ``hyper_apply_inner_wrapper`` mutates the target
    # forward at runtime, and the P1 contract (wrapper weaves into the
    # rewritten redistribute forward) must not change under pruning.
    inner_wrap = injection is not None and injection.get("inner_wrapper") is not None

    axes = tuple(mesh_dim_names or ())
    if not axes:
        # No active sharding axes: the runtime routes ep-keyed entries to the
        # expert mesh (a runtime object — not statically decidable) and treats
        # everything else as an exact no-op passthrough.
        if inner_wrap or _entry_has_ep_placement(entry):
            return BoundaryForm(form=FORM_GENERIC)
        return BoundaryForm(form=FORM_IDENTITY)

    in_ops: dict[str, TransitionOp] = {}
    in_notes: dict[str, str] = {}
    in_identity: list[str] = []
    in_declared: list[str] = []
    in_fallback = False
    for name, result, note in _classify_side(entry, "in_src", "in_dst", axes):
        in_declared.append(name)
        if note is not None:
            in_notes[name] = note
        if result is None:
            # An unlowerable transition anywhere on the side forces the
            # generic form (the compiled plan's DTensor fallback owns it).
            in_fallback = True
        elif result["kind"] == "identity":
            in_identity.append(name)
        else:
            in_ops[name] = TransitionOp(
                kind=result["kind"], tensor_dim=result.get("tensor_dim")
            )

    out_ops: dict[str, TransitionOp] = {}
    out_notes: dict[str, str] = {}
    out_declared: list[str] = []
    out_fallback = False
    for name, result, note in _classify_side(entry, "out_src", "out_dst", axes):
        out_declared.append(name)
        if note is not None:
            out_notes[name] = note
        if result is None:
            out_fallback = True
        elif result["kind"] != "identity":
            out_ops[name] = TransitionOp(
                kind=result["kind"], tensor_dim=result.get("tensor_dim")
            )

    declared_out_names = _declared_out_names(entry)
    if in_fallback or out_fallback or inner_wrap:
        return BoundaryForm(
            form=FORM_GENERIC,
            in_ops=in_ops,
            out_ops=out_ops,
            in_identity=tuple(in_identity),
            in_notes=in_notes,
            out_notes=out_notes,
            in_declared=tuple(in_declared),
            out_declared=tuple(out_declared),
            out_names=declared_out_names,
        )
    if not in_ops and not out_ops:
        return BoundaryForm(
            form=FORM_IDENTITY,
            in_ops=in_ops,
            out_ops=out_ops,
            in_identity=tuple(in_identity),
            in_notes=in_notes,
            out_notes=out_notes,
            in_declared=tuple(in_declared),
            out_declared=tuple(out_declared),
            out_names=declared_out_names,
        )
    return BoundaryForm(
        form=FORM_TP_COLLECTIVE,
        in_ops=in_ops,
        out_ops=out_ops,
        in_identity=tuple(in_identity),
        in_notes=in_notes,
        out_notes=out_notes,
        in_declared=tuple(in_declared),
        out_declared=tuple(out_declared),
        out_names=declared_out_names,
    )


def _classify_side(
    entry: dict[str, Any],
    src_field: str,
    dst_field: str,
    axes: tuple[str, ...],
):
    """Yield ``(name, classified, note)`` for every declared name on one side.

    ``classified`` is ``classify_tp_transition``'s result (``None`` when the
    transition is not TP-lowerable); ``note`` is the declarative annotation
    line for non-identity transitions (``None`` for identity / no-op names).
    """
    # Lazy imports: plan modules must not pull the distributed package (and
    # its torch-dependent __init__) at codegen import time.
    from hyper_parallel.codegen.plan.freeze import parse_named_placement
    from hyper_parallel.distributed.recipe_spec import resolve_placements
    from hyper_parallel.distributed._builder.tp_collective_lowering import (
        classify_tp_transition,
    )

    src_named = parse_named_placement(entry.get(src_field) or {})
    dst_named = parse_named_placement(entry.get(dst_field) or {})
    src_raw = entry.get(src_field) or {}
    dst_raw = entry.get(dst_field) or {}
    for name in sorted(set(src_named) | set(dst_named)):
        src_p = tuple(resolve_placements(src_named.get(name) or {}, axes))
        dst_p = tuple(resolve_placements(dst_named.get(name) or {}, axes))
        result = classify_tp_transition(src_p, dst_p, axes)
        note = _transition_note(
            name, src_raw.get(name) or {}, dst_raw.get(name) or {}, axes, result
        )
        yield name, result, note


def _transition_note(
    name: str,
    src_raw: dict[str, Any],
    dst_raw: dict[str, Any],
    axes: tuple[str, ...],
    result: Optional[dict],
) -> Optional[str]:
    """Render one transition's declarative annotation, or ``None`` for identity.

    Only axes the plan actually shards on participate (an ``ep`` key is
    dropped by the dense-mesh compile, so claiming it here would lie); the
    raw frozen strings are shown verbatim — the annotation is the plan's own
    declaration, not a re-interpretation.
    """
    diffs = [
        f"{axis} {src_raw.get(axis, 'R')} -> {dst_raw.get(axis, 'R')}"
        for axis in axes
        if src_raw.get(axis, "R") != dst_raw.get(axis, "R")
    ]
    if not diffs:
        return None
    head = f"{name}: " + ", ".join(diffs)
    if result is None:
        return f"{head} ==> dtensor redistribute"
    return f"{head} ==> {_op_text(result)}"


def _op_text(result: dict) -> str:
    """Compact text of one classified op, as it appears in annotations/calls."""
    kind = result["kind"]
    if kind == "all_reduce":
        return "all_reduce(sum)"
    return f"{kind}(dim={result['tensor_dim']})"


def _entry_has_ep_placement(entry: dict[str, Any]) -> bool:
    """Whether the entry carries any ``ep`` placement key on any side."""
    for src_field, dst_field in _SIDE_FIELDS:
        for named in (entry.get(src_field) or {}, entry.get(dst_field) or {}):
            for placement in named.values():
                if isinstance(placement, dict) and "ep" in placement:
                    return True
    return False


def _declared_out_names(entry: dict[str, Any]) -> tuple[str, ...]:
    """The entry's declared output order (``out_names`` or ``out_src`` keys)."""
    declared = entry.get("out_src") or {}
    out_names = entry.get("out_names") or declared.keys()
    return tuple(out_names)


__all__ = [
    "BoundaryForm",
    "FORM_GENERIC",
    "FORM_IDENTITY",
    "FORM_REGION",
    "FORM_TP_COLLECTIVE",
    "TransitionOp",
    "classify_boundary_form",
]
