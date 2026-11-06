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
"""Public cross-module contracts for the Hyper-RL runtime."""

# Public names are resolved lazily by __getattr__, not bound at import time.
__all__ = list((
    "ExperienceBatch",
    "Message",
    "PromptRecord",
    "Trajectory",
    "Turn",
))

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ExperienceBatch": ("rl.data.contracts", "ExperienceBatch"),
    "Message": ("rl.data.contracts", "Message"),
    "PromptRecord": ("rl.data.contracts", "PromptRecord"),
    "Trajectory": ("rl.data.contracts", "Trajectory"),
    "Turn": ("rl.data.contracts", "Turn"),
}


# PEP 562 requires these exact names to keep framework imports lazy.
def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
    """Load public contracts only when a caller requests them."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pylint: disable=invalid-name
    """Expose lazy public contracts to interactive callers."""
    return sorted((*globals(), *_EXPORTS))
