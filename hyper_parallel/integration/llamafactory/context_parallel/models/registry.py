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
"""Registry primitives for model-specific context-parallel patches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

MeshGetter = Callable[[], object]


@dataclass(frozen=True)
class ContextParallelModelPatch:
    """Describe the CP hooks needed by one supported model family."""

    name: str
    supports: Callable[[nn.Module], bool]
    prepare: Callable[[nn.Module, object, MeshGetter], None]
