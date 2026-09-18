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
"""Precision-validation case matrix generation from manifest topology specs."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional

from hyper_parallel.models.adapter_spec import RecomputePolicy
from hyper_parallel.models.validation_spec import TopologyConstraint

_TOPOLOGY_DEFAULTS = {
    "tp": 1,
    "cp": 1,
    "ep": 1,
    "fsdp": 1,
    "sequence_parallel": False,
    "recompute": {"layer_count": 0},
}
_PARALLEL_TOPOLOGY_FIELDS = (
    "tp",
    "cp",
    "ep",
    "fsdp",
    "sequence_parallel",
)


@dataclass(frozen=True)
class ValidationCase:
    """One resolved precision-validation topology and execution role."""

    name: str
    topology: dict[str, Any]
    kind: str = "topology"
    compare_to: str = "baseline"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _case_name(topology: Mapping[str, Any], prefix: str = "") -> str:
    ordered = (*_PARALLEL_TOPOLOGY_FIELDS, "recompute")
    pieces = []
    for name in ordered:
        if name not in topology:
            continue
        value = topology[name]
        if name == "recompute" and isinstance(value, Mapping):
            if "layer_count" in value:
                value = f"count{value['layer_count']}"
            elif "layer_indices" in value:
                value = "indices" + "-".join(str(index) for index in value["layer_indices"])
        pieces.append(f"{name}{value}")
    name = "_".join(str(piece).replace("/", "-") for piece in pieces)
    return f"{prefix}_{name}" if prefix else name


def _validate_recompute_selection(selection: Any, path: str) -> dict[str, Any]:
    """Validate one exact layer-count or layer-index recompute selection."""
    if not isinstance(selection, Mapping):
        raise ValueError(f"{path} must be a mapping")
    unsupported = tuple(
        name for name in selection if name not in ("layer_count", "layer_indices")
    )
    if unsupported:
        raise ValueError(
            f"{path} contains unsupported fields {unsupported}; supported fields: "
            "layer_count, layer_indices"
        )
    if ("layer_count" in selection) == ("layer_indices" in selection):
        raise ValueError(
            f"{path} must contain exactly one of layer_count or layer_indices"
        )
    if "layer_count" in selection:
        layer_count = selection["layer_count"]
        if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count < 0:
            raise ValueError(f"{path}.layer_count must be a non-negative integer")
        return {"layer_count": layer_count}

    layer_indices = selection["layer_indices"]
    if not isinstance(layer_indices, list) or not layer_indices:
        raise ValueError(
            f"{path}.layer_indices must be a non-empty list of non-negative integers"
        )
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in layer_indices
    ):
        raise ValueError(
            f"{path}.layer_indices must contain only non-negative integers"
        )
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError(f"{path}.layer_indices must not contain duplicates")
    return {"layer_indices": list(layer_indices)}


def _validate_topology(topology: Mapping[str, Any], path: str) -> dict[str, Any]:
    """Reject fields that the standard Trainer launcher cannot consume."""
    if not isinstance(topology, Mapping):
        raise ValueError(f"{path} must be a mapping")
    resolved = dict(topology)
    unsupported = tuple(name for name in resolved if name not in _TOPOLOGY_DEFAULTS)
    if unsupported:
        supported = ", ".join(_TOPOLOGY_DEFAULTS)
        raise ValueError(
            f"{path} contains unsupported fields {unsupported}; supported fields: {supported}"
        )
    if "recompute" in resolved:
        resolved["recompute"] = _validate_recompute_selection(
            resolved["recompute"],
            f"{path}.recompute",
        )
    return resolved


def _acceptance_class(
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
) -> str:
    """Classify tolerances from the parallel topology actually launched."""
    for field_name in _PARALLEL_TOPOLOGY_FIELDS:
        default = _TOPOLOGY_DEFAULTS[field_name]
        if baseline.get(field_name, default) != candidate.get(field_name, default):
            return "cross_topology"
    return "same_topology"


def generate_validation_cases(
    matrix: Mapping[str, Any],
    recompute_policy: Optional[RecomputePolicy] = None,
    topology_constraints: Iterable[TopologyConstraint] = (),
) -> tuple[ValidationCase, ...]:
    """Generate a minimum single-axis-plus-combined validation matrix."""
    baseline = dict(matrix.get("baseline") or {})
    if not baseline:
        raise ValueError("matrix.baseline is required")
    baseline = _validate_topology(baseline, "matrix.baseline")
    cases = [ValidationCase("baseline", baseline, compare_to="baseline")]
    seen = {json.dumps(baseline, sort_keys=True)}
    axes = matrix.get("axes") or {}
    if not isinstance(axes, Mapping):
        raise ValueError("matrix.axes must be a mapping")
    for axis, values in axes.items():
        if axis not in _TOPOLOGY_DEFAULTS:
            supported = ", ".join(_TOPOLOGY_DEFAULTS)
            raise ValueError(
                f"matrix.axes contains unsupported field {axis!r}; "
                f"supported fields: {supported}"
            )
        if not isinstance(values, list) or not values:
            raise ValueError(f"matrix.axes.{axis} must be a non-empty list")
        for value in values:
            topology = dict(baseline)
            topology[axis] = value
            key = json.dumps(topology, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                ValidationCase(_case_name(topology, f"axis-{axis}"), topology)
            )
    for index, combined in enumerate(matrix.get("combined") or ()):
        if not isinstance(combined, Mapping):
            raise ValueError("matrix.combined entries must be mappings")
        topology = dict(baseline)
        topology.update(_validate_topology(combined, f"matrix.combined[{index}]"))
        key = json.dumps(topology, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        cases.append(ValidationCase(_case_name(topology, f"combined-{index}"), topology))

    if recompute_policy is not None:
        requested_selections = matrix.get(
            "recompute_selections",
            ({"layer_count": 0},),
        )
        if not isinstance(requested_selections, (list, tuple)) or not requested_selections:
            raise ValueError("matrix.recompute_selections must be a non-empty list")
        for index, requested_selection in enumerate(requested_selections):
            selection = _validate_recompute_selection(
                requested_selection,
                f"matrix.recompute_selections[{index}]",
            )
            topology = dict(baseline)
            topology.update(
                _validate_topology(
                    matrix.get("recompute_topology") or {},
                    "matrix.recompute_topology",
                )
            )
            topology["recompute"] = selection
            key = json.dumps(topology, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                ValidationCase(_case_name(topology, "recompute"), topology)
            )

    if int(baseline.get("ep", 1)) > 1 and not any(
        int(case.topology.get("ep", 1)) == 1 for case in cases
    ):
        topology = dict(baseline)
        topology["ep"] = 1
        cases.insert(1, ValidationCase(_case_name(topology, "axis-ep"), topology))

    production_validate = matrix.get("production_validate_pair")
    if production_validate:
        production_validate = _validate_topology(
            production_validate,
            "matrix.production_validate_pair",
        )
        for mode in ("production", "validate"):
            topology = dict(baseline)
            topology.update(production_validate)
            topology["validate_placement"] = mode == "validate"
            cases.append(
                ValidationCase(
                    _case_name(topology, mode),
                    topology,
                    kind="production_validate",
                )
            )
    if matrix.get("same_topology_resume"):
        cases.append(
            ValidationCase(
                "same_topology_resume",
                dict(baseline),
                kind="resume",
                metadata={"resume": "same_topology"},
            )
        )
    if matrix.get("cross_topology_resume"):
        topology = dict(baseline)
        topology.update(
            _validate_topology(
                matrix["cross_topology_resume"],
                "matrix.cross_topology_resume",
            )
        )
        if _acceptance_class(baseline, topology) == "same_topology":
            raise ValueError(
                "matrix.cross_topology_resume must change at least one of "
                f"{_PARALLEL_TOPOLOGY_FIELDS}. A recompute-only change is not a "
                "cross-topology restore; same_topology_resume always uses the "
                "baseline recompute policy, and a recompute-only resume matrix "
                "is not supported"
            )
        cases.append(
            ValidationCase(
                "cross_topology_resume",
                topology,
                kind="resume",
                metadata={
                    "resume": "cross_topology",
                    "prepare_topology": dict(baseline),
                },
            )
        )

    resolved_cases = []
    for case in cases:
        for constraint in topology_constraints:
            if not constraint.predicate(case.topology):
                raise ValueError(
                    f"case {case.name!r} violates topology constraint "
                    f"{constraint.name!r}: {constraint.remediation}"
                )
        metadata = dict(case.metadata)
        metadata["acceptance"] = _acceptance_class(baseline, case.topology)
        resolved_cases.append(dataclasses.replace(case, metadata=metadata))
    return tuple(resolved_cases)


__all__ = [
    "ValidationCase",
    "generate_validation_cases",
]
