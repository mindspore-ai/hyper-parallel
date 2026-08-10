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
"""Backend-neutral generation request and result types."""

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from rl.roles.weight_sync import PolicySnapshot


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    pad_token_id: int
    eos_token_id: int
    collect_log_probs: bool = False


@dataclass(frozen=True)
class GenerationRequest:
    input_ids: Any
    attention_mask: Any
    settings: GenerationSettings


@dataclass(frozen=True)
class GenerationResult:
    sequences: Any
    rollout_log_probs: Optional[Any]
    generation_seconds: float
    response_mask: Optional[Any] = None


class GenerationEngine(Protocol):
    """The only generation surface known by rollout orchestration."""

    name: str
    policy_version: int

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate responses for one backend-neutral request."""

    def update_weights(self, snapshot: PolicySnapshot) -> None:
        """Load and acknowledge a strictly newer policy snapshot."""
