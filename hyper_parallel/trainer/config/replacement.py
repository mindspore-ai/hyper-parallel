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
"""Replacement-only Target retaining family selectors for Python/CLI overrides."""

from __future__ import annotations

from typing import Any, Callable

from hyper_parallel.components.modules.family import (
    SelectorSpec,
    get_family_selector_specs,
    resolve_module_family_target,
)
from hyper_parallel.trainer.config.target import Target


class ReplacementTarget(Target):
    """Resolve the implementation without discarding the requested baseline."""

    def __init__(self, _target_: Callable, *, target_path: str, **kwargs: Any) -> None:
        """Select a final callable and strictly validate constructor keywords.

        Args:
            _target_: Requested baseline or ordinary replacement callable.
            target_path: Original import path retained in serialized configuration.
            **kwargs: Explicit selectors and constructor arguments.
        """
        self.requested_target = _target_
        selector_specs = get_family_selector_specs(_target_)
        self.selectors = {name: value for name, value in kwargs.items() if name in selector_specs}
        resolved, arguments = resolve_module_family_target(_target_, kwargs)
        super().__init__(resolved, target_path=target_path, **arguments)

    def get_family_selector_spec(self, name: str) -> SelectorSpec | None:
        """Query the requested baseline's schema for Target-level overrides.

        Args:
            name: Selector name supplied by the override path.

        Returns:
            Its declaration, or None when it is not a selector for this target.
        """
        return get_family_selector_specs(self.requested_target).get(name)

    def replace(self, **changes: Any) -> "ReplacementTarget":
        """Re-resolve selectors after an override, without mutating the original.

        Args:
            **changes: Explicit selectors or constructor arguments to replace.
        """
        return type(self)(
            self.requested_target, target_path=self._target_path, **{**self._kwargs, **self.selectors, **changes}
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the baseline path and original selectors for round trips."""
        return {**super().to_dict(), **self.selectors}
