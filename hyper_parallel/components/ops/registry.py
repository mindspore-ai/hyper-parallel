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
"""Registry of operator contracts, implementations and automatic selection orders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch

from .spec import DeviceType, Fn, OpSpec


class ImplementationUnavailable(RuntimeError):
    """A selected adapter cannot load a required optional dependency or operator."""


@dataclass(frozen=True)
class _Implementation:
    load: Callable[[], Any]


class OpRegistry:
    """Assemble a catalog and seal it before resolving operator calls."""

    def __init__(self) -> None:
        """Start with an empty catalog open for registration."""
        self._families: dict[str, dict[str, OpSpec[Any]]] = {}
        self._names: dict[str, set[str]] = {}
        self._implementations: dict[tuple[OpSpec[Any], str, str], _Implementation] = {}
        self._implementations_by_name: dict[tuple[str, str, str, str], _Implementation] = {}
        self._auto: dict[tuple[OpSpec[Any], str], tuple[str, ...]] = {}
        self._sealed = False

    def _require_mutable(self) -> None:
        if self._sealed:
            raise RuntimeError("Operator registry is sealed")

    def _require_spec(self, spec: OpSpec[Any]) -> None:
        if self._families.get(spec.config_key, {}).get(spec.variant) is not spec:
            raise ValueError(f"Unregistered operator contract {spec.config_key}.{spec.variant}")

    def register_family(self, config_key: str, *, specs: tuple[OpSpec[Any], ...]) -> None:
        """Register an operator family and its typed backend contracts."""
        self._require_mutable()
        variants = {spec.variant: spec for spec in specs}
        if (
            not isinstance(config_key, str)
            or not config_key.strip()
            or config_key in self._families
            or not variants
            or len(variants) != len(specs)
            or any(spec.config_key != config_key for spec in specs)
        ):
            raise ValueError(f"Invalid or duplicate operator family {config_key!r}")
        self._families[config_key] = variants
        self._names[config_key] = set()

    def register_impl(
        self,
        spec: OpSpec[Fn],
        *,
        name: str,
        devices: tuple[DeviceType, ...],
        load: Callable[[], Fn],
    ) -> None:
        """Register a lazy implementation for the given contract and devices."""
        self._require_mutable()
        self._require_spec(spec)
        if not isinstance(name, str) or not name.strip() or name == "auto":
            raise ValueError(f"Invalid operator implementation name {name!r}")
        if not devices or len(set(devices)) != len(devices) or set(devices) - {"cpu", "cuda", "npu"}:
            raise ValueError(f"Invalid operator devices {devices!r} for {spec.config_key}.{name}")
        if not callable(load):
            raise ValueError(f"Invalid operator loader for {spec.config_key}.{name}")
        keys = [(spec, name, device_type) for device_type in devices]
        if any(key in self._implementations for key in keys):
            raise ValueError(f"Overlapping operator registration for {spec.config_key}.{spec.variant}:{name}")
        implementation = _Implementation(load)
        self._implementations.update({key: implementation for key in keys})
        # String keys let Dynamo resolve implementations without hashing OpSpec dataclasses.
        self._implementations_by_name.update(
            {(spec.config_key, spec.variant, name, device): implementation for device in devices}
        )
        self._names[spec.config_key].add(name)

    def set_auto_order(
        self,
        spec: OpSpec[Any],
        *,
        device: DeviceType,
        candidates: tuple[str, ...],
    ) -> None:
        """Declare an ordered list of registered candidates for one device."""
        self._require_mutable()
        self._require_spec(spec)
        key = (spec, device)
        if (
            device not in {"cpu", "cuda", "npu"}
            or key in self._auto
            or len(set(candidates)) != len(candidates)
            or any((spec, name, device) not in self._implementations for name in candidates)
        ):
            raise ValueError(f"Invalid automatic backend order for {spec.config_key}.{spec.variant} on {device}")
        self._auto[key] = tuple(candidates)

    def seal(self) -> None:
        """Finish catalog assembly and prevent subsequent registrations."""
        self._sealed = True

    def validate_selections(self, selections: Mapping[str, str]) -> None:
        """Validate user-facing family and implementation names."""
        if not self._sealed:
            raise ValueError("Operator backend selection requires a sealed registry")
        if not isinstance(selections, Mapping):
            raise ValueError("ops must be a mapping from family names to implementation names")
        for key, name in selections.items():
            if not isinstance(key, str) or key not in self._names:
                raise ValueError(f"Unknown operator family {key!r} in 'ops'")
            if not isinstance(name, str) or (name != "auto" and name not in self._names[key]):
                allowed = ", ".join(sorted(self._names[key] | {"auto"}))
                raise ValueError(f"Invalid ops.{key} value {name!r}, expected one of {allowed}")

    def resolve(
        self,
        spec: OpSpec[Fn],
        selection: str,
        device: torch.device,
    ) -> tuple[str, Fn]:
        """Return the selected name and callable, or explain candidate rejection."""
        self._require_spec(spec)
        device_type = device.type
        names = self._auto.get((spec, device_type), ()) if selection == "auto" else (selection,)
        reasons = []
        for name in names:
            impl = self._implementations_by_name.get((spec.config_key, spec.variant, name, device_type))
            if impl is None:
                reasons.append(f"{name}: not registered for this contract and device")
                continue
            try:
                selected = impl.load()
            except ImplementationUnavailable as exc:
                reasons.append(f"{name}: {exc}")
                continue
            if not callable(selected):
                raise TypeError(f"Operator loader for {spec.config_key}.{name} did not return a callable")
            return name, cast(Fn, selected)
        detail = ", ".join(reasons) or "no auto candidates registered"
        raise RuntimeError(
            f"Cannot select ops.{spec.config_key}={selection!r} "
            f"for variant {spec.variant!r} on {device}: {detail}"
        )


# The selector and family modules share this instance without importing the catalog.
builtin_registry = OpRegistry()
