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
"""OpenCompass integration with lazy optional-dependency loading."""

from importlib import import_module
from typing import Any


__all__ = ["HyperOpenCompassModel", "causal_lm_ppl_scores"]

_LAZY_EXPORTS = {
    "HyperOpenCompassModel": ("hyper_parallel.integration.opencompass.model", "HyperOpenCompassModel"),
    "causal_lm_ppl_scores": ("hyper_parallel.integration.opencompass.scoring", "causal_lm_ppl_scores"),
}


def __getattr__(name: str) -> Any:
    """Resolve OpenCompass integration symbols only when requested."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public integration symbols."""
    return sorted(__all__)
