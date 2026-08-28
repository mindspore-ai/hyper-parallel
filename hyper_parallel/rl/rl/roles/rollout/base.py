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
    seed: Optional[int] = None
    eos_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        """Normalize the primary and additional terminal token IDs."""
        normalized_ids = tuple(
            dict.fromkeys(
                (int(self.eos_token_id), *(int(token_id) for token_id in self.eos_token_ids))
            )
        )
        if any(token_id < 0 for token_id in normalized_ids):
            raise ValueError(f"EOS token IDs must be non-negative, got {normalized_ids}")
        object.__setattr__(self, "eos_token_ids", normalized_ids)


@dataclass(frozen=True)
class GenerationRequest:
    input_ids: Any
    attention_mask: Any
    settings: GenerationSettings
    row_seeds: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class GenerationResult:
    sequences: Any
    rollout_log_probs: Optional[Any]
    generation_seconds: float
    response_mask: Optional[Any] = None
    worker_policy_version: Optional[int] = None
    worker_policy_fingerprint: Optional[str] = None


class GenerationEngine(Protocol):
    """The only generation surface known by rollout orchestration."""

    name: str
    policy_version: int

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate responses for one backend-neutral request."""

    def synchronize_error(self, local_error: Optional[Exception], operation: str) -> None:
        """Propagate one local rollout orchestration failure to every rank."""

    def prepare_for_training(self) -> None:
        """Release inference residency before the synchronous training phase."""

    def update_weights(self, snapshot: PolicySnapshot) -> None:
        """Load and acknowledge a strictly newer policy snapshot."""

    def prepare_for_rollout(self) -> None:
        """Restore inference residency after publishing the next policy."""
