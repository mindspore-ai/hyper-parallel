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
"""Process-wide operator backend selection and direct function dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from types import MappingProxyType
from typing import Any, cast

import torch

from .registry import builtin_registry
from .spec import Fn, OpSpec


class _BackendState:
    """Hold one configuration snapshot and its resolution cache and usage."""

    def __init__(self, selections: Mapping[str, str]) -> None:
        self.selections = MappingProxyType(dict(selections))
        self.resolved: dict[
            tuple[str, str, str, torch.device], tuple[str, Any]
        ] = {}
        self.used: dict[tuple[str, str, str], None] = {}
        self.lock = RLock()

    def select(self, spec: OpSpec[Fn], device: torch.device) -> Fn:
        """Resolve only from the contract, configuration and explicit execution device."""
        selection = self.selections.get(spec.config_key, "auto")
        key = (spec.config_key, spec.variant, selection, device)
        selected = self.resolved.get(key)
        if torch.compiler.is_compiling():
            # Dynamo captures the callable. Publishing it in the cache would
            # introduce a Python callable side effect during graph tracing.
            if selected is None:
                selected = builtin_registry.resolve(
                    spec, selection, device,
                )
            self.used[(spec.config_key, spec.variant, selected[0])] = None
        elif selected is None:
            with self.lock:
                selected = self.resolved.get(key)
                if selected is None:
                    selected = self.resolved[key] = builtin_registry.resolve(
                        spec, selection, device,
                    )
                self.used[(spec.config_key, spec.variant, selected[0])] = None
        return cast(Fn, selected[1])


class _OpSelector:
    """Configure and inspect operator backend choices for the current process."""

    def __init__(self) -> None:
        self._state = _BackendState({})

    def set_backends(self, selections: Mapping[str, str]) -> None:
        """Replace the process-wide backend configuration and clear its cache and usage.

        Args:
            selections: Complete mapping from operator family names to backend
                names. Omitted families use auto.
                An empty mapping restores auto for all families.

        Raises:
            ValueError: A family, backend name or mapping value is invalid.
                Validation failures preserve the active configuration.
        """
        if not isinstance(selections, Mapping):
            raise ValueError("ops must be a mapping from family names to implementation names")
        snapshot = dict(selections)
        builtin_registry.validate_selections(snapshot)
        self._state = _BackendState(snapshot)

    def get_backends(self) -> dict[str, str]:
        """Return a copy of the explicitly configured family backend selections."""
        return dict(self._state.selections)

    def used_backends(self) -> tuple[tuple[str, str, str], ...]:
        """Return deduplicated operator, variant and backend triples since the last setting."""
        state = self._state
        with state.lock:
            return tuple(state.used)


ops = _OpSelector()


def _select_for_call(spec: OpSpec[Fn], *, device: torch.device) -> Fn:
    """Select an implementation without inspecting tensors or their argument positions."""
    state = ops._state  # pylint: disable=protected-access
    return state.select(spec, device)
