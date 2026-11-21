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
"""Prepare and release module-owned MegaKernel runtimes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from torch import nn


def _runtime_groups(model: nn.Module) -> Iterator[list[nn.Module]]:
    """Group unique runtime owners by compatible execution resources."""
    groups: dict[Any, list[nn.Module]] = {}
    seen = set()
    for module in model.modules():
        key = getattr(module, "runtime_group_key", None)
        if callable(key):
            # Transparent wrappers may delegate this bound method to their child.
            owner = getattr(key, "__self__", module)
            if id(owner) not in seen:
                groups.setdefault(key(), []).append(owner)
                seen.add(id(owner))
    yield from groups.values()


def prepare_module_runtimes(model: nn.Module) -> None:
    """Prepare declared module groups after FSDP wrapping and materialization.

    Args:
        model: Final module tree. Ordinary modules declare no runtime group.
    """
    for modules in _runtime_groups(model):
        modules[0].prepare_runtime_group(modules)


def close_module_runtimes(model: nn.Module) -> None:
    """Release declared runtimes after every forward and backward has completed.

    Args:
        model: Final model tree; each holder releases its own runtime reference.
    """
    for modules in _runtime_groups(model):
        for module in modules:
            module.close_runtime()
