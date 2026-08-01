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
"""Configuration for quantization-error monitoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


# A framework adapter decides which of these roles it can emit. Keeping the
# schema complete lets one debug program be shared across integrations.
_GEMM_ROLES = frozenset({"fprop", "dgrad", "wgrad"})
_OPERANDS = frozenset({"lhs", "rhs"})


@dataclass(frozen=True)
class DebugOutput:
    """Runtime-owned destination for rank-local diagnostic artifacts."""

    root_dir: str | None = None


@dataclass(frozen=True)
class DebugSchedule:
    every_n_steps: int = 1
    start_step: int = 0
    end_step: int | None = None

    @classmethod
    def from_mapping(
        cls,
        value: "DebugSchedule | Mapping[str, Any] | None",
    ) -> "DebugSchedule":
        """Build a validated schedule from a mapping."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("precision_debug schedule must be a mapping")
        unknown = set(value) - {"every_n_steps", "start_step", "end_step"}
        if unknown:
            raise ValueError(f"Unknown precision_debug schedule fields: {sorted(unknown)}")
        schedule = cls(
            every_n_steps=int(value.get("every_n_steps", 1)),
            start_step=int(value.get("start_step", 0)),
            end_step=int(value["end_step"]) if value.get("end_step") is not None else None,
        )
        if schedule.every_n_steps < 1 or schedule.start_step < 0:
            raise ValueError("schedule requires positive every_n_steps and non-negative start_step")
        if schedule.end_step is not None and schedule.end_step < schedule.start_step:
            raise ValueError("schedule end_step must be no earlier than start_step")
        return schedule

    def active(self, step: int) -> bool:
        """Return whether the given step is selected."""
        return (
            step >= self.start_step
            and (self.end_step is None or step <= self.end_step)
            and (step - self.start_step) % self.every_n_steps == 0
        )


@dataclass(frozen=True)
class Selector:
    module_name_regex: str = ".*"
    gemm_roles: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: "Selector | Mapping[str, Any] | None") -> "Selector":
        """Build a validated observation selector."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("precision_debug select must be a mapping")
        unknown = set(value) - {"module_name_regex", "gemm_roles"}
        if unknown:
            raise ValueError(f"Unknown precision_debug select fields: {sorted(unknown)}")
        regex = value.get("module_name_regex", ".*")
        if not isinstance(regex, str) or not regex:
            raise ValueError("module_name_regex must be a non-empty string")
        try:
            re.compile(regex)
        except re.error as error:
            raise ValueError(f"Invalid module_name_regex '{regex}': {error}") from error
        roles = value.get("gemm_roles", ())
        if isinstance(roles, str):
            roles = (roles,)
        if not isinstance(roles, (list, tuple)):
            raise ValueError("gemm_roles must be a string or list")
        unknown_roles = set(roles) - _GEMM_ROLES
        if unknown_roles:
            raise ValueError(f"Unknown GEMM roles: {sorted(unknown_roles)}")
        return cls(module_name_regex=regex, gemm_roles=tuple(roles))

    def matches(self, module_fqn: str, gemm_role: str) -> bool:
        """Return whether a module and GEMM role match this selector."""
        return (
            re.search(self.module_name_regex, module_fqn) is not None
            and (not self.gemm_roles or gemm_role in self.gemm_roles)
        )


@dataclass(frozen=True)
class ObserveAction:
    operands: tuple[str, ...] = ("lhs", "rhs")
    schedule: DebugSchedule = field(default_factory=DebugSchedule)

    @classmethod
    def from_mapping(
        cls,
        value: "ObserveAction | Mapping[str, Any] | bool",
    ) -> "ObserveAction":
        """Build a validated observation action."""
        if isinstance(value, cls):
            return value
        if value is True:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("observe must be true or a mapping")
        unknown = set(value) - {
            "operands",
            "schedule",
        }
        if unknown:
            raise ValueError(f"Unknown precision_debug observe fields: {sorted(unknown)}")
        operands = value.get("operands", ("lhs", "rhs"))
        if isinstance(operands, str):
            operands = (operands,)
        if not isinstance(operands, (list, tuple)) or not operands:
            raise ValueError("observe.operands must be a non-empty string or list")
        unknown_operands = set(operands) - _OPERANDS
        if unknown_operands:
            raise ValueError(f"Unknown operand roles: {sorted(unknown_operands)}")
        return cls(
            operands=tuple(operands),
            schedule=DebugSchedule.from_mapping(value.get("schedule")),
        )


@dataclass(frozen=True)
class DebugSection:
    name: str
    select: Selector
    observe: ObserveAction

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DebugSection":
        """Build one named debug section."""
        if not isinstance(value, Mapping):
            raise ValueError("precision_debug section must be a mapping")
        unknown = set(value) - {"name", "select", "observe"}
        if unknown:
            raise ValueError(f"Unknown precision_debug section fields: {sorted(unknown)}")
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("precision_debug section name must be a non-empty string")
        if "observe" not in value:
            raise ValueError("precision_debug section requires observe")
        return cls(
            name=name,
            select=Selector.from_mapping(value.get("select")),
            observe=ObserveAction.from_mapping(value["observe"]),
        )


@dataclass(frozen=True)
class PrecisionDebugProgram:
    sections: tuple[DebugSection, ...]
    output: DebugOutput = field(default_factory=DebugOutput)

    @classmethod
    def from_mapping(
        cls,
        value: "PrecisionDebugProgram | Mapping[str, Any]",
    ) -> "PrecisionDebugProgram":
        """Build a validated precision-debug program."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("precision_debug must be a mapping")
        unknown = set(value) - {"sections"}
        if unknown:
            raise ValueError(f"Unknown precision_debug fields: {sorted(unknown)}")
        sections = value.get("sections")
        if not isinstance(sections, (list, tuple)) or not sections:
            raise ValueError("precision_debug.sections must be a non-empty list")
        parsed = tuple(DebugSection.from_mapping(item) for item in sections)
        names = tuple(item.name for item in parsed)
        if len(names) != len(set(names)):
            raise ValueError("precision_debug section names must be unique")
        return cls(sections=parsed)

    @classmethod
    def default(cls) -> "PrecisionDebugProgram":
        """Default monitor used by programmatic installations without YAML."""
        return cls(
            sections=(
                DebugSection(
                    name="observe_all_fprop",
                    select=Selector(gemm_roles=("fprop",)),
                    observe=ObserveAction(),
                ),
            ),
        )


@dataclass(frozen=True)
class DebugOverlay:
    observations: Mapping[tuple[str, str], tuple[ObserveAction, ...]]
    output: DebugOutput = field(default_factory=DebugOutput)

    def active(self, step: int) -> bool:
        """Return whether any configured observation is active."""
        return any(
            action.schedule.active(step)
            for actions in self.observations.values()
            for action in actions
        )

    def quantization_error(
        self,
        module_fqn: str,
        gemm_role: str,
        operand_role: str,
        step: int,
    ) -> bool:
        """Return whether one quantization operand should be measured."""
        return any(
            operand_role in action.operands
            and action.schedule.active(step)
            for action in self.observations.get((module_fqn, gemm_role), ())
        )
