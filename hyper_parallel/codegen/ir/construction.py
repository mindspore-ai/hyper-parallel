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
from dataclasses import dataclass
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
    """Return plain observation data for the snapshot."""
    if isinstance(value, Mapping):
        return {name: ir_value_to_data(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [ir_value_to_data(item) for item in value]
    return value


@dataclass(frozen=True)
class CodegenBuildIR:
    """The native observations consumed by source emission."""

    source: Mapping[str, Any]
    declarations: tuple[Mapping[str, Any], ...]
    replacements: tuple[Mapping[str, Any], ...]
    plan: Optional[Mapping[str, Any]]
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("source", "declarations", "replacements", "plan"):
            object.__setattr__(self, name, freeze_ir_value(getattr(self, name), path=name))


__all__ = ["CodegenBuildIR"]
