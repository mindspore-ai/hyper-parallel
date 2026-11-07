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
"""Isolated in-memory observations captured from a native model build."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CodegenStatus(str, Enum):
    """Lifecycle state of one codegen collection session."""

    NOT_REQUIRED = "not_required"
    COLLECTING = "collecting"
    DERIVED = "derived"
    GENERATING = "generating"
    GENERATED = "generated"
    FAILED = "failed"
    SKIPPED = "skipped"


def copy_mapping(value: Mapping[str, Any], path: str) -> dict[str, Any]:
    """Return a deep copy of a string-keyed observation mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} mapping keys must be strings")
    return deepcopy(dict(value))


def _copy_records(
    values: Sequence[Mapping[str, Any]], path: str
) -> tuple[dict[str, Any], ...]:
    """Copy a sequence of observation records."""
    return tuple(
        copy_mapping(item, f"{path}[{index}]")
        for index, item in enumerate(values)
    )


@dataclass(frozen=True)
class ReplacementSnapshot:
    """Declared replacement rules and their observed native applications."""

    declarations: tuple[dict[str, Any], ...] = ()
    applications: tuple[dict[str, Any], ...] = ()
    unmatched: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_data(
        cls,
        *,
        declarations: Sequence[Mapping[str, Any]] = (),
        applications: Sequence[Mapping[str, Any]] = (),
        unmatched: Sequence[Mapping[str, Any]] = (),
    ) -> "ReplacementSnapshot":
        """Copy replacement observations into the session boundary."""
        return cls(
            declarations=_copy_records(
                declarations, "replacements.declarations"
            ),
            applications=_copy_records(
                applications, "replacements.applications"
            ),
            unmatched=_copy_records(unmatched, "replacements.unmatched"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an isolated mutable representation for emission."""
        return deepcopy(
            {
                "declarations": list(self.declarations),
                "applications": list(self.applications),
                "unmatched": list(self.unmatched),
            }
        )


@dataclass(frozen=True)
class FailureSnapshot:
    """Stable failure information without exception or traceback objects."""

    stage: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("failure stage must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("failure reason must be a non-empty string")
        object.__setattr__(
            self,
            "details",
            copy_mapping(self.details, "failure.details"),
        )

    @classmethod
    def from_data(
        cls,
        stage: str,
        reason: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> "FailureSnapshot":
        """Build an isolated failure record."""
        if not stage or not reason:
            raise ValueError("failure stage and reason must be non-empty")
        return cls(stage, reason, details or {})

    def to_dict(self) -> dict[str, Any]:
        """Return an isolated mutable representation for emission."""
        return {
            "stage": self.stage,
            "reason": self.reason,
            "details": deepcopy(self.details),
        }


@dataclass(frozen=True)
class CodegenSnapshot:
    """Isolated observation of one native model build."""

    status: CodegenStatus
    source: dict[str, Any] = field(default_factory=dict)
    replacements: ReplacementSnapshot = field(default_factory=ReplacementSnapshot)
    plan: Optional[dict[str, Any]] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    failure: Optional[FailureSnapshot] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, CodegenStatus):
            raise TypeError("snapshot status must be a CodegenStatus")
        if not isinstance(self.replacements, ReplacementSnapshot):
            raise TypeError("snapshot replacements must be a ReplacementSnapshot")
        if self.failure is not None and not isinstance(self.failure, FailureSnapshot):
            raise TypeError("snapshot failure must be a FailureSnapshot or None")
        object.__setattr__(self, "source", copy_mapping(self.source, "snapshot.source"))
        object.__setattr__(
            self,
            "diagnostics",
            copy_mapping(self.diagnostics, "snapshot.diagnostics"),
        )
        for name in ("plan",):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    copy_mapping(value, f"snapshot.{name}"),
                )

    def to_dict(self) -> dict[str, Any]:
        """Return isolated mutable containers for the emission boundary."""
        return deepcopy(self._emission_view())

    def _emission_view(self) -> dict[str, Any]:
        """Borrow observation data internally; callers must copy before mutation."""
        return {
            "status": self.status.value,
            "source": self.source,
            "replacements": {
                "declarations": list(self.replacements.declarations),
                "applications": list(self.replacements.applications),
                "unmatched": list(self.replacements.unmatched),
            },
            "plan": self.plan,
            "diagnostics": self.diagnostics,
            "reason": self.reason,
            "failure": None if self.failure is None else {
                "stage": self.failure.stage,
                "reason": self.failure.reason,
                "details": self.failure.details,
            },
        }

    @classmethod
    def _from_owned_data(cls, **values: Any) -> "CodegenSnapshot":
        """Adopt data already isolated by a session without copying it again."""
        snapshot = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(snapshot, name, value)
        return snapshot


__all__ = [
    "CodegenSnapshot",
    "CodegenStatus",
    "FailureSnapshot",
    "ReplacementSnapshot",
    "copy_mapping",
]
