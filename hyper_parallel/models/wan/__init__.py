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
"""Wan model-family public API with lazy Diffusers runtime imports."""

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "WanConditionModel": ("hyper_parallel.models.wan.condition", "WanConditionModel"),
    "build_wan_condition_model": (
        "hyper_parallel.models.wan.condition",
        "build_wan_condition_model",
    ),
}

__all__ = [
    "WanConditionModel",
    "build_wan_condition_model",
]


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
    """Load Wan runtime symbols only when callers request them."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pylint: disable=invalid-name
    """Include lazily exported Wan symbols in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
