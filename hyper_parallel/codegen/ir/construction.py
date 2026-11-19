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
"""Resource-free construction IR projected from shared model preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional


_SCALARS = (type(None), bool, int, float, str)


def freeze_ir_value(value: Any, *, path: str) -> Any:
    """Copy JSON-compatible construction data into immutable containers."""
    if type(value) in _SCALARS:
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(name, str) for name in value):
            raise TypeError(f"{path} mapping keys must be strings")
        return MappingProxyType({
            name: freeze_ir_value(item, path=f"{path}.{name}")
            for name, item in value.items()
        })
    if isinstance(value, (tuple, list)):
        return tuple(
            freeze_ir_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} contains runtime value "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def ir_value_to_data(value: Any) -> Any:
    """Return ordinary isolated containers for the compatibility emitter."""
    if isinstance(value, Mapping):
        return {name: ir_value_to_data(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [ir_value_to_data(item) for item in value]
    return value


@dataclass(frozen=True)
class ModelSourceIR:
    """Stable identity of the model source selected by Trainer preparation."""

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_ir_value(self.values, path="source"))

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy source mapping."""
        return ir_value_to_data(self.values)


@dataclass(frozen=True)
class ComponentBoundaryIR:
    """One prepared component and its optional source-replacement lowering."""

    boundary_id: str
    parent_boundary_id: Optional[str]
    emission_owner_boundary_id: str
    origin: str
    aliases: tuple[str, ...]
    declaration_id: Optional[str] = None
    values: Mapping[str, Any] = field(default_factory=dict)
    replacement: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "values", freeze_ir_value(self.values, path="component.values"))
        if self.replacement is not None:
            object.__setattr__(
                self,
                "replacement",
                freeze_ir_value(self.replacement, path="component.replacement"),
            )


@dataclass(frozen=True)
class OwnedLayoutIR:
    """One final native layout assigned to a prepared component or model root."""

    module_fqn: str
    owner_boundary_id: Optional[str]
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_ir_value(self.values, path="layout.values"))


@dataclass(frozen=True)
class ParallelBindingIR:
    """One final native parallel binding emitted for a prepared boundary."""

    boundary_id: str
    layout_owner_boundary_id: str
    aliases: tuple[str, ...]
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "values", freeze_ir_value(self.values, path="binding.values"))


@dataclass(frozen=True)
class EmissionPlanIR:
    """Plain native-plan facts required by the source emitter."""

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_ir_value(self.values, path="plan"))

    def to_dict(self) -> dict[str, Any]:
        """Return the existing emitter's plan representation."""
        return ir_value_to_data(self.values)


@dataclass(frozen=True)
class CodegenBuildIR:
    """Complete immutable projection of one shared Trainer model build."""

    source: ModelSourceIR
    declarations: tuple[Mapping[str, Any], ...]
    components: tuple[ComponentBoundaryIR, ...]
    layouts: tuple[OwnedLayoutIR, ...]
    bindings: tuple[ParallelBindingIR, ...]
    plan: Optional[EmissionPlanIR]
    replacement_order: tuple[str, ...]
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "declarations",
            tuple(
                freeze_ir_value(item, path=f"declarations[{index}]")
                for index, item in enumerate(self.declarations)
            ),
        )
        for name in ("components", "layouts", "bindings", "replacement_order"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def replacement_records(self) -> tuple[dict[str, Any], ...]:
        """Return source replacements for the transitional snapshot adapter."""
        records = {
            component.boundary_id: component.replacement
            for component in self.components
            if component.replacement is not None
        }
        return tuple(ir_value_to_data(records[boundary_id]) for boundary_id in self.replacement_order)

    def declaration_records(self) -> tuple[dict[str, Any], ...]:
        """Return isolated replacement declarations for snapshot emission."""
        return tuple(ir_value_to_data(item) for item in self.declarations)


__all__ = [
    "CodegenBuildIR",
    "ComponentBoundaryIR",
    "EmissionPlanIR",
    "ModelSourceIR",
    "OwnedLayoutIR",
    "ParallelBindingIR",
]
