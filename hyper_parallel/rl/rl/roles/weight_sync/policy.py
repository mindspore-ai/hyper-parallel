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
"""Versioned policy publication contract shared by rollout engines."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicySnapshot:
    """One immutable policy publication event.

    ``payload`` is backend-specific: the co-located Hyper engine receives the
    actor object, while a vLLM refitter may receive a state dict or checkpoint
    descriptor.  The monotonically increasing version is backend-neutral.
    """

    version: int
    model_name: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("PolicySnapshot version must be non-negative")
        if not self.model_name:
            raise ValueError("PolicySnapshot model_name must be non-empty")
