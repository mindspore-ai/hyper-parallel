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
"""Versioned machine-readable schemas for model-integration diagnostics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional


SCHEMA_VERSION = 1
_ERROR_CODE = re.compile(r"^HP-[A-Z]+-[0-9]{3}$")


def tolerance_value(
        tolerance: Mapping[str, Any],
        name: str,
        fallback: str | None = None,
) -> float:
    """Resolve an optional tolerance, treating YAML null as unspecified."""
    value = tolerance.get(name)
    if value is None and fallback is not None:
        value = tolerance.get(fallback)
    return 0.0 if value is None else float(value)


class FindingSeverity(str, Enum):
    """Stable severity levels consumed by CLI and Agent workflows."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class IntegrationState(str, Enum):
    """Evidence-backed validation gates, excluding derived/reporting states."""

    DISCOVERED = "DISCOVERED"
    STRUCTURE_VALIDATED = "STRUCTURE_VALIDATED"
    MODULE_PARITY_PASSED = "MODULE_PARITY_PASSED"
    MATRIX_PASSED = "MATRIX_PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ModelIntegrationFinding:
    """One structured diagnostic with facts and an actionable remediation."""

    code: str
    phase: str
    owner_fqn: str
    severity: FindingSeverity
    message: str
    facts: dict[str, Any] = field(default_factory=dict)
    why_unsafe: str = ""
    remediation: str = ""
    related_config: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _ERROR_CODE.fullmatch(self.code):
            raise ValueError(f"invalid model integration error code: {self.code!r}")
        if not self.phase or not self.message:
            raise ValueError("model integration finding phase and message must be non-empty")
        if self.severity is FindingSeverity.ERROR and (
            not self.why_unsafe or not self.remediation
        ):
            raise ValueError("ERROR findings require why_unsafe and remediation")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload

    def format_human(self) -> str:
        """Format a stable human-readable diagnostic block."""
        owner = f"\nowner: {self.owner_fqn}" if self.owner_fqn else ""
        facts = ""
        if self.facts:
            facts = "\nfacts:" + "".join(
                f"\n  {name}: {value}" for name, value in sorted(self.facts.items())
            )
        unsafe = f"\nwhy unsafe: {self.why_unsafe}" if self.why_unsafe else ""
        remediation = f"\nfix: {self.remediation}" if self.remediation else ""
        return (
            f"[{self.code}] {self.severity.value} phase={self.phase}: {self.message}"
            f"{owner}{facts}{unsafe}{remediation}"
        )


class ModelIntegrationValidationError(RuntimeError):
    """Raised when a model integration report contains one or more ERROR findings."""

    def __init__(self, findings: Iterable[ModelIntegrationFinding]) -> None:
        """Build one exception from ordered structured findings."""
        self.findings = tuple(findings)
        super().__init__("\n\n".join(finding.format_human() for finding in self.findings))


@dataclass
class ModelIntegrationReport:
    """Aggregate findings for one model/build/run while preserving schema version."""

    findings: list[ModelIntegrationFinding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def errors(self) -> tuple[ModelIntegrationFinding, ...]:
        """Return ERROR findings."""
        return tuple(
            finding for finding in self.findings
            if finding.severity is FindingSeverity.ERROR
        )

    def add(self, finding: ModelIntegrationFinding) -> None:
        """Append a finding."""
        self.findings.append(finding)

    def extend(self, findings: Iterable[ModelIntegrationFinding]) -> None:
        """Append findings from another validator."""
        self.findings.extend(findings)

    def raise_for_errors(self) -> None:
        """Raise a structured exception when any ERROR is present."""
        if self.errors:
            raise ModelIntegrationValidationError(self.errors)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON payload."""
        return {
            "schema_version": self.schema_version,
            "status": "FAIL" if self.errors else "PASS",
            "metadata": self.metadata,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class ComparisonMetric:
    """Numerical comparison result for one tensor observation."""

    name: str
    status: str
    max_abs: Optional[float]
    relative_l2: Optional[float]
    finite: bool
    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


__all__ = [
    "ComparisonMetric",
    "ModelIntegrationReport",
    "FindingSeverity",
    "IntegrationState",
    "ModelIntegrationFinding",
    "ModelIntegrationValidationError",
    "SCHEMA_VERSION",
    "tolerance_value",
]
