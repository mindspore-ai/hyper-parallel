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
    normalize_family_selectors,
    resolve_module_family_target,
)
from hyper_parallel.components.quantization.config import LowPrecisionDtypeScheme
from hyper_parallel.trainer.config.target import Target


class ReplacementTarget(Target):
    """Resolve the implementation without discarding the requested baseline."""

    def __init__(self, _target_: Callable, *, target_path: str,
                 low_precision_dtype_scheme: str | None = None, **kwargs: Any) -> None:
        """Select a final callable and strictly validate constructor keywords.

        Args:
            _target_: Requested baseline or ordinary replacement callable.
            target_path: Original import path retained in serialized configuration.
            low_precision_dtype_scheme: Catalog name resolved when the replacement plan is built.
            **kwargs: Explicit selectors and constructor arguments.
        """
        self.requested_target = _target_
        self.low_precision_dtype_scheme = low_precision_dtype_scheme
        kwargs = normalize_family_selectors(_target_, kwargs)
        selector_specs = get_family_selector_specs(_target_)
        self.selectors = {name: value for name, value in kwargs.items() if name in selector_specs}
        if low_precision_dtype_scheme is None:
            resolved, arguments = resolve_module_family_target(_target_, kwargs)
        else:
            if not isinstance(low_precision_dtype_scheme, str) or not low_precision_dtype_scheme.strip():
                raise ValueError("low_precision_dtype_scheme must be a non-empty catalog name")
            # The catalog belongs to training configuration and is unavailable while parsing this Target.
            resolved = _target_
            arguments = {name: value for name, value in kwargs.items() if name not in selector_specs}
        super().__init__(resolved, target_path=target_path, **arguments)

    def bind_dtype_scheme(self, scheme: LowPrecisionDtypeScheme) -> Target:
        """Select from scheme contents while retaining its catalog name in the source config.

        Args:
            scheme: Resolved catalog entry providing format selectors.
        """
        resolved, arguments = resolve_module_family_target(self.requested_target, {
            **self._kwargs, **self.selectors,
            "weight_format": scheme.weight_format,
            "act_format": scheme.act_format,
            "is_fake_quantize": scheme.is_fake_quantize,
        })
        return Target(resolved, target_path=self._target_path, **arguments)

    def build(self, **runtime_kwargs: Any) -> Any:
        """Build only after a named scheme has been bound by the replacement planner."""
        if self.low_precision_dtype_scheme is not None:
            raise ValueError("Bind low_precision_dtype_scheme through the replacement plan before building")
        return super().build(**runtime_kwargs)

    def get_family_selector_spec(self, name: str) -> SelectorSpec | None:
        """Query the requested baseline's schema for Target-level overrides.

        Args:
            name: Selector name supplied by the override path.

        Returns:
            Its declaration, or None when it is not a selector for this target.
        """
        if name == "low_precision_dtype_scheme":
            return SelectorSpec(str, "")
        return get_family_selector_specs(self.requested_target).get(name)

    def replace(self, **changes: Any) -> "ReplacementTarget":
        """Re-resolve selectors after an override, without mutating the original.

        Args:
            **changes: Explicit selectors or constructor arguments to replace.
        """
        return type(self)(
            self.requested_target, target_path=self._target_path,
            **{**self._kwargs, **self.selectors,
               "low_precision_dtype_scheme": self.low_precision_dtype_scheme,
               **normalize_family_selectors(self.requested_target, changes)}
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the baseline path and original selectors for round trips."""
        result = {**super().to_dict(), **self.selectors}
        if self.low_precision_dtype_scheme is not None:
            result["low_precision_dtype_scheme"] = self.low_precision_dtype_scheme
        return result
