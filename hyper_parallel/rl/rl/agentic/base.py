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
"""Backend-neutral agentic environment contract for one episode."""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class Observation:
    content: str
    token_ids: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    content: str
    token_ids: Any
    rollout_log_probs: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    observation: Observation
    reward: float
    done: bool
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Environment(Protocol):

    async def reset(self, prompt: Any) -> Observation:
        """Reset one episode and return its initial observation."""

    async def step(self, action: Action) -> Transition:
        """Apply an agent action and return the environment transition."""

    async def close(self) -> None:
        """Release resources owned by the episode."""
