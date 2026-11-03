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

from hyper_parallel.models.validation_spec import TopologyConstraint
from hyper_parallel.trainer.config.parallelism import FullCheckpointConfig

_TOPOLOGY_DEFAULTS = {
    "tp": 1,
    "cp": 1,
    "ep": 1,
    "fsdp": 1,
    "sequence_parallel": False,
    "recompute_layer": [],
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
    ordered = (*_PARALLEL_TOPOLOGY_FIELDS, "recompute_layer")
    pieces = []
    for name in ordered:
        if name not in topology:
            continue
        value = topology[name]
        if name == "recompute_layer":
            value = "off" if not value else "layers" + "-".join(str(index) for index in value)
        pieces.append(f"{name}{value}")
    name = "_".join(str(piece).replace("/", "-") for piece in pieces)
    return f"{prefix}_{name}" if prefix else name


def _validate_recompute_layers(value: Any, path: str) -> list[int | str]:
    """Validate one explicit full-checkpoint layer set."""
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list of layer numbers or ranges")
    if not value:
        return []
    try:
        return list(FullCheckpointConfig(recompute_layer=value).recompute_layers)
    except ValueError as error:
        raise ValueError(str(error).replace("activation_checkpoint.full.recompute_layer", path)) from error


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
    if "recompute_layer" in resolved:
        resolved["recompute_layer"] = _validate_recompute_layers(
            resolved["recompute_layer"],
            f"{path}.recompute_layer",
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


def _axis_cases(
        matrix: Mapping[str, Any],
        baseline: Mapping[str, Any],
        seen: set[str],
) -> list[ValidationCase]:
    """Generate one case per declared single-axis deviation."""
    axes = matrix.get("axes") or {}
    if not isinstance(axes, Mapping):
        raise ValueError("matrix.axes must be a mapping")
    cases = []
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
    return cases


def _combined_cases(
        matrix: Mapping[str, Any],
        baseline: Mapping[str, Any],
        seen: set[str],
) -> list[ValidationCase]:
    """Generate one case per declared multi-axis combination."""
    cases = []
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
    return cases


def _recompute_cases(
        matrix: Mapping[str, Any],
        baseline: Mapping[str, Any],
        seen: set[str],
) -> list[ValidationCase]:
    """Generate one case per requested full-checkpoint layer set."""
    requested_layer_sets = matrix.get("recompute_layers") or ()
    if not isinstance(requested_layer_sets, (list, tuple)):
        raise ValueError("matrix.recompute_layers must be a list")
    cases = []
    for index, requested_layers in enumerate(requested_layer_sets):
        recompute_layers = _validate_recompute_layers(
            requested_layers,
            f"matrix.recompute_layers[{index}]",
        )
        topology = dict(baseline)
        topology.update(
            _validate_topology(
                matrix.get("recompute_topology") or {},
                "matrix.recompute_topology",
            )
        )
        topology["recompute_layer"] = recompute_layers
        key = json.dumps(topology, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            ValidationCase(_case_name(topology, "recompute"), topology)
        )
    return cases


def _ep1_fallback_case(
        cases: list[ValidationCase],
        baseline: Mapping[str, Any],
) -> Optional[ValidationCase]:
    """Retain a single-device-EP reference when every case parallelizes experts."""
    if int(baseline.get("ep", 1)) <= 1 or any(
        int(case.topology.get("ep", 1)) == 1 for case in cases
    ):
        return None
    topology = dict(baseline)
    topology["ep"] = 1
    return ValidationCase(_case_name(topology, "axis-ep"), topology)


def _production_validate_cases(
        matrix: Mapping[str, Any],
        baseline: Mapping[str, Any],
) -> list[ValidationCase]:
    """Generate the production/validate placement pair when declared."""
    production_validate = matrix.get("production_validate_pair")
    if not production_validate:
        return []
    production_validate = _validate_topology(
        production_validate,
        "matrix.production_validate_pair",
    )
    cases = []
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
    return cases


def _resume_cases(
        matrix: Mapping[str, Any],
        baseline: Mapping[str, Any],
) -> list[ValidationCase]:
    """Generate the declared same/cross-topology resume cases."""
    cases = []
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
    return cases


def _finalize_cases(
        cases: list[ValidationCase],
        baseline: Mapping[str, Any],
        topology_constraints: Iterable[TopologyConstraint],
) -> tuple[ValidationCase, ...]:
    """Apply topology constraints and classify each case acceptance profile."""
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


def generate_validation_cases(
    matrix: Mapping[str, Any],
    topology_constraints: Iterable[TopologyConstraint] = (),
) -> tuple[ValidationCase, ...]:
    """Generate a minimum single-axis-plus-combined validation matrix."""
    baseline = dict(matrix.get("baseline") or {})
    if not baseline:
        raise ValueError("matrix.baseline is required")
    baseline = _validate_topology(baseline, "matrix.baseline")
    cases = [ValidationCase("baseline", baseline, compare_to="baseline")]
    seen = {json.dumps(baseline, sort_keys=True)}
    cases.extend(_axis_cases(matrix, baseline, seen))
    cases.extend(_combined_cases(matrix, baseline, seen))
    if matrix.get("recompute_layers"):
        cases.extend(_recompute_cases(matrix, baseline, seen))
    ep1_case = _ep1_fallback_case(cases, baseline)
    if ep1_case is not None:
        cases.insert(1, ep1_case)
    cases.extend(_production_validate_cases(matrix, baseline))
    cases.extend(_resume_cases(matrix, baseline))
    return _finalize_cases(cases, baseline, topology_constraints)


__all__ = [
    "ValidationCase",
    "generate_validation_cases",
]
