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
"""Immutable, JSON-safe records captured from a native model build."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeAlias, Union


SNAPSHOT_SCHEMA_VERSION = 1

JsonPrimitive: TypeAlias = Union[None, bool, int, float, str]
FrozenJson: TypeAlias = Union[JsonPrimitive, "FrozenMapping", tuple["FrozenJson", ...]]


class CodegenStatus(str, Enum):
    """Lifecycle state of one codegen collection session."""

    NOT_REQUIRED = "not_required"
    COLLECTING = "collecting"
    DERIVED = "derived"
    GENERATING = "generating"
    GENERATED = "generated"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class FrozenMapping(Mapping[str, FrozenJson]):
    """An insertion-order-independent immutable mapping of JSON values."""

    _items: tuple[tuple[str, FrozenJson], ...] = ()

    def __post_init__(self) -> None:
        keys = tuple(key for key, _ in self._items)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("FrozenMapping keys must be strings")
        if keys != tuple(sorted(set(keys))):
            raise ValueError("FrozenMapping keys must be unique and sorted")
        if any(not _is_frozen_json(value) for _, value in self._items):
            raise TypeError("FrozenMapping values must be immutable JSON values")

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, key: str) -> FrozenJson:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)


def _is_frozen_json(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, FrozenMapping):
        return True
    if isinstance(value, tuple):
        return all(_is_frozen_json(item) for item in value)
    return False


def freeze_json(value: Any, *, path: str = "snapshot") -> FrozenJson:
    """Deep-freeze a JSON value and reject runtime objects.

    Args:
        value: Value to validate and freeze.
        path: Human-readable field path used in validation errors.

    Returns:
        An immutable representation with mappings sorted by key.

    Raises:
        TypeError: If ``value`` is not representable by JSON.
        ValueError: If a mapping key is not a string or a float is non-finite.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path} mapping keys must be strings")
        items = []
        for key in sorted(value):
            items.append((key, freeze_json(value[key], path=f"{path}.{key}")))
        return FrozenMapping(tuple(items))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains non-JSON value of type {type(value).__name__}")


def thaw_json(value: FrozenJson) -> Any:
    """Convert an immutable snapshot value into ordinary JSON containers."""
    if isinstance(value, FrozenMapping):
        return {key: thaw_json(item) for key, item in value._items}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _freeze_mapping(value: Mapping[str, Any], path: str) -> FrozenMapping:
    frozen = freeze_json(value, path=path)
    if not isinstance(frozen, FrozenMapping):
        raise TypeError(f"{path} must be a mapping")
    return frozen


@dataclass(frozen=True)
class ReplacementSnapshot:
    """Declared replacement rules and their observed native applications."""

    declarations: tuple[FrozenMapping, ...] = ()
    applications: tuple[FrozenMapping, ...] = ()
    unmatched: tuple[FrozenMapping, ...] = ()

    def __post_init__(self) -> None:
        for name in ("declarations", "applications", "unmatched"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(
                isinstance(item, FrozenMapping) for item in value
            ):
                raise TypeError(f"replacement {name} must be a tuple of FrozenMapping")

    @classmethod
    def from_data(
        cls,
        *,
        declarations: Sequence[Mapping[str, Any]] = (),
        applications: Sequence[Mapping[str, Any]] = (),
        unmatched: Sequence[Mapping[str, Any]] = (),
    ) -> "ReplacementSnapshot":
        """Build a replacement snapshot from JSON-safe mappings."""
        return cls(
            declarations=tuple(
                _freeze_mapping(item, f"replacements.declarations[{index}]")
                for index, item in enumerate(declarations)
            ),
            applications=tuple(
                _freeze_mapping(item, f"replacements.applications[{index}]")
                for index, item in enumerate(applications)
            ),
            unmatched=tuple(
                _freeze_mapping(item, f"replacements.unmatched[{index}]")
                for index, item in enumerate(unmatched)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON representation of this replacement record."""
        return {
            "declarations": [thaw_json(item) for item in self.declarations],
            "applications": [thaw_json(item) for item in self.applications],
            "unmatched": [thaw_json(item) for item in self.unmatched],
        }


@dataclass(frozen=True)
class FailureSnapshot:
    """Stable failure information without exception or traceback objects."""

    stage: str
    reason: str
    details: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("failure stage must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("failure reason must be a non-empty string")
        if not isinstance(self.details, FrozenMapping):
            raise TypeError("failure details must be a FrozenMapping")

    @classmethod
    def from_data(
        cls,
        stage: str,
        reason: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> "FailureSnapshot":
        """Build a validated failure record."""
        if not stage or not reason:
            raise ValueError("failure stage and reason must be non-empty")
        return cls(stage, reason, _freeze_mapping(details or {}, "failure.details"))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON representation of this failure record."""
        return {"stage": self.stage, "reason": self.reason, "details": thaw_json(self.details)}


@dataclass(frozen=True)
class CodegenSnapshot:
    """Deeply immutable observation of one native model build."""

    status: CodegenStatus
    source: FrozenMapping = field(default_factory=FrozenMapping)
    replacements: ReplacementSnapshot = field(default_factory=ReplacementSnapshot)
    plan: Optional[FrozenMapping] = None
    plan_application: Optional[FrozenMapping] = None
    diagnostics: FrozenMapping = field(default_factory=FrozenMapping)
    reason: Optional[str] = None
    failure: Optional[FailureSnapshot] = None
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported snapshot schema version {self.schema_version}; "
                f"expected {SNAPSHOT_SCHEMA_VERSION}"
            )
        if not isinstance(self.status, CodegenStatus):
            raise TypeError("snapshot status must be a CodegenStatus")
        if not isinstance(self.replacements, ReplacementSnapshot):
            raise TypeError("snapshot replacements must be a ReplacementSnapshot")
        if self.failure is not None and not isinstance(self.failure, FailureSnapshot):
            raise TypeError("snapshot failure must be a FailureSnapshot or None")
        for name in ("source", "diagnostics"):
            if not isinstance(getattr(self, name), FrozenMapping):
                raise TypeError(f"snapshot {name} must be a FrozenMapping")
        for name in ("plan", "plan_application"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, FrozenMapping):
                raise TypeError(f"snapshot {name} must be a FrozenMapping or None")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CodegenSnapshot":
        """Rebuild and validate a snapshot from its JSON representation."""
        if not isinstance(data, Mapping):
            raise TypeError("snapshot payload must be a mapping")
        version = data.get("schema_version")
        if version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported snapshot schema version {version!r}; "
                f"expected {SNAPSHOT_SCHEMA_VERSION}"
            )
        try:
            status = CodegenStatus(data.get("status"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown snapshot status {data.get('status')!r}") from exc
        source = data.get("source", {})
        replacements = data.get("replacements", {})
        diagnostics = data.get("diagnostics", {})
        if not isinstance(source, Mapping):
            raise TypeError("snapshot source must be a mapping")
        if not isinstance(replacements, Mapping):
            raise TypeError("snapshot replacements must be a mapping")
        if not isinstance(diagnostics, Mapping):
            raise TypeError("snapshot diagnostics must be a mapping")
        plan = data.get("plan")
        plan_application = data.get("plan_application")
        if plan is not None and not isinstance(plan, Mapping):
            raise TypeError("snapshot plan must be a mapping or None")
        if plan_application is not None and not isinstance(plan_application, Mapping):
            raise TypeError("snapshot plan_application must be a mapping or None")
        failure_data = data.get("failure")
        failure = None
        if failure_data is not None:
            if not isinstance(failure_data, Mapping):
                raise TypeError("snapshot failure must be a mapping or None")
            failure = FailureSnapshot.from_data(
                failure_data.get("stage"),
                failure_data.get("reason"),
                failure_data.get("details"),
            )
        reason = data.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise TypeError("snapshot reason must be a string or None")
        return cls(
            status=status,
            source=_freeze_mapping(source, "source"),
            replacements=ReplacementSnapshot.from_data(
                declarations=replacements.get("declarations", ()),
                applications=replacements.get("applications", ()),
                unmatched=replacements.get("unmatched", ()),
            ),
            plan=None if plan is None else _freeze_mapping(plan, "plan"),
            plan_application=(
                None
                if plan_application is None
                else _freeze_mapping(plan_application, "plan_application")
            ),
            diagnostics=_freeze_mapping(diagnostics, "diagnostics"),
            reason=reason,
            failure=failure,
            schema_version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON representation of the snapshot."""
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "source": thaw_json(self.source),
            "replacements": self.replacements.to_dict(),
            "plan": None if self.plan is None else thaw_json(self.plan),
            "plan_application": (
                None if self.plan_application is None else thaw_json(self.plan_application)
            ),
            "diagnostics": thaw_json(self.diagnostics),
            "reason": self.reason,
            "failure": None if self.failure is None else self.failure.to_dict(),
        }

    def signature_payload(self) -> dict[str, Any]:
        """Return deterministic build facts suitable for hashing.

        Diagnostics and failures describe execution rather than artifact
        identity and therefore do not participate in this projection.
        """
        payload = self.to_dict()
        payload.pop("diagnostics")
        payload.pop("failure")
        return payload


__all__ = [
    "CodegenSnapshot",
    "CodegenStatus",
    "FailureSnapshot",
    "FrozenMapping",
    "ReplacementSnapshot",
    "SNAPSHOT_SCHEMA_VERSION",
    "freeze_json",
    "thaw_json",
]
