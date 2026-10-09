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
"""Model-owned declarations consumed by the integration validation tools.

The declarations intentionally contain no execution policy. A family adapter
describes authoritative builders and semantic observations here; manifests
choose devices, topology, data, steps, and acceptance tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional


TensorGetter = Callable[[Any], Any]
ModuleBuilder = Callable[[Any], Any]
InputBuilder = Callable[[Any], Any]
WeightAdapter = Callable[[Any, Any, Any], Any]
Objective = Callable[[Any], Any]
InvariantChecker = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class ObservationSpec:
    """One named tensor or structured value compared by a parity case."""

    name: str
    reference_getter: Optional[TensorGetter] = None
    candidate_getter: Optional[TensorGetter] = None
    comparison: Literal["numeric", "exact"] = "numeric"
    required: bool = True
    atol: Optional[float] = None
    rtol: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observation name must be non-empty")
        if self.comparison not in ("numeric", "exact"):
            raise ValueError(f"unsupported observation comparison: {self.comparison!r}")
        if self.comparison == "exact" and (self.atol is not None or self.rtol is not None):
            raise ValueError("exact observations must not declare atol or rtol")


@dataclass(frozen=True)
class ModuleParityCase:
    """Declare one authoritative-reference versus final-candidate comparison.

    Builders receive the resolved validation context. The candidate builder
    must return the module selected after the production replacement and
    materialization path, not a helper that only resembles that module.
    """

    name: str
    candidate_selector: str
    candidate_builder: Optional[ModuleBuilder] = None
    reference_builder: Optional[ModuleBuilder] = None
    input_builder: Optional[InputBuilder] = None
    weight_adapter: Optional[WeightAdapter] = None
    observations: tuple[ObservationSpec, ...] = ()
    objective: Optional[Objective] = None
    required_dtypes: tuple[str, ...] = ("float32",)
    required_parameter_patterns: tuple[str, ...] = ()
    execution: Literal["in_process", "isolated_process"] = "in_process"
    isolated_runner: Optional[Callable[[Any], dict[str, Any]]] = None

    def __post_init__(self) -> None:
        if not self.name or not self.candidate_selector:
            raise ValueError("module parity case name and candidate_selector must be non-empty")
        if self.execution not in ("in_process", "isolated_process"):
            raise ValueError(f"unsupported parity execution mode: {self.execution!r}")
        if self.execution == "in_process":
            for builder_name in (
                "candidate_builder",
                "reference_builder",
                "input_builder",
                "weight_adapter",
            ):
                if not callable(getattr(self, builder_name)):
                    raise TypeError(
                        f"in_process ModuleParityCase.{builder_name} must be callable"
                    )
        if self.execution == "isolated_process" and not callable(self.isolated_runner):
            raise ValueError("isolated_process parity requires an isolated_runner")


@dataclass(frozen=True)
class ParameterProbeSpec:
    """Select parameters and stages that must emit precision evidence."""

    match: str
    required: bool = True
    stages: tuple[str, ...] = (
        "main_grad_before_clip",
        "main_after_optimizer",
    )
    compare: Literal["numeric_global", "exact_hash"] = "numeric_global"
    check_replica_consistency: bool = True

    def __post_init__(self) -> None:
        if not self.match:
            raise ValueError("parameter probe match must be non-empty")
        if not self.stages:
            raise ValueError("parameter probe stages must not be empty")


@dataclass(frozen=True)
class StateInvariantSpec:
    """Model-owned invariant evaluated on the final built model."""

    name: str
    checker: InvariantChecker
    phase: Literal["structure", "materialization", "checkpoint", "runtime"] = "materialization"
    error_code: str = "HP-MAT-003"

    def __post_init__(self) -> None:
        if not self.name or not callable(self.checker):
            raise ValueError("state invariant requires a name and callable checker")


@dataclass(frozen=True)
class SharedStateValidationSpec:
    """Declare cross-layer state keys and producers that must not be replayed."""

    producer_keys: tuple[str, ...]
    consumer_keys: tuple[str, ...]
    begin_trace: Optional[Callable[[], Any]] = None
    finish_trace: Optional[Callable[[Any], tuple[dict[str, Any], ...]]] = None

    def __post_init__(self) -> None:
        if (self.begin_trace is None) != (self.finish_trace is None):
            raise ValueError("shared-state tracing requires both begin_trace and finish_trace")


@dataclass(frozen=True)
class DataValidationSpec:
    """Declare model-owned training-data and forward-input semantics.

    The declaration is intentionally about observable fields rather than a
    concrete Dataset or DataLoader implementation. The Trainer combines it
    with the actual batch/runtime adapters and first-step inputs.
    """

    required_forward_fields: tuple[str, ...] = ()
    runtime_fields: tuple[str, ...] = ()
    cp_replicated_forward_fields: tuple[str, ...] = ()
    modality_fields: tuple[str, ...] = ()
    modality_parameter_patterns: tuple[str, ...] = ()
    labels_are_shifted: Optional[bool] = None
    deterministic_replay_required: bool = True
    processor_revision_required: bool = False


@dataclass(frozen=True)
class CheckpointValidationSpec:
    """Classify intentional checkpoint omissions by final/source FQN glob."""

    training_only_target_patterns: tuple[str, ...] = ()
    inference_only_source_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not pattern for pattern in self.training_only_target_patterns):
            raise ValueError("training-only checkpoint patterns must be non-empty")
        if any(not pattern for pattern in self.inference_only_source_patterns):
            raise ValueError("inference-only checkpoint patterns must be non-empty")


@dataclass(frozen=True)
class TopologyConstraint:
    """One model-specific topology rule checked before launching a case."""

    name: str
    predicate: Callable[[dict[str, Any]], bool]
    remediation: str

    def __post_init__(self) -> None:
        if not self.name or not callable(self.predicate) or not self.remediation:
            raise ValueError("topology constraint requires name, predicate, and remediation")


@dataclass(frozen=True)
class ModelValidationSpec:
    """All model-owned declarations needed by generic validation tooling."""

    module_cases: tuple[ModuleParityCase, ...] = ()
    parameter_probes: tuple[ParameterProbeSpec, ...] = ()
    state_invariants: tuple[StateInvariantSpec, ...] = ()
    shared_state: Optional[SharedStateValidationSpec] = None
    data: Optional[DataValidationSpec] = None
    checkpoint: Optional[CheckpointValidationSpec] = None
    topology_constraints: tuple[TopologyConstraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CheckpointValidationSpec",
    "DataValidationSpec",
    "ModelValidationSpec",
    "ModuleParityCase",
    "ObservationSpec",
    "ParameterProbeSpec",
    "SharedStateValidationSpec",
    "StateInvariantSpec",
    "TopologyConstraint",
]
