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
"""Reusable policy objectives hidden behind complete algorithm Recipes."""

from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class PolicyObjectiveOutput:
    """Per-token objective values and clipping indicators."""

    loss: Any
    clipped: Any


class PolicyObjective(Protocol):
    """Internal policy-loss seam selected by a complete Recipe."""

    def compute(
        self,
        current_log_probs: Any,
        old_log_probs: Any,
        advantages: Any,
    ) -> PolicyObjectiveOutput:
        """Compute per-token policy loss and clipping indicators."""


@dataclass(frozen=True)
class ClippedPolicyObjective:
    """Clipped importance-ratio objective with optional dual clipping."""

    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    dual_clip: Optional[float] = None

    def compute(
        self,
        current_log_probs: Any,
        old_log_probs: Any,
        advantages: Any,
    ) -> PolicyObjectiveOutput:
        """Compute the clipped importance-ratio policy objective."""
        log_ratio = current_log_probs - old_log_probs
        ratio = log_ratio.exp()
        unclipped_loss = -advantages * ratio
        clipped_ratio = ratio.clamp(
            min=1.0 - self.clip_ratio_low,
            max=1.0 + self.clip_ratio_high,
        )
        policy_loss = unclipped_loss.maximum(-advantages * clipped_ratio)
        if self.dual_clip is not None:
            dual_clip_loss = (-advantages * self.dual_clip).minimum(policy_loss)
            policy_loss = dual_clip_loss.where(advantages < 0, policy_loss)
        clipped = (
            (ratio < 1.0 - self.clip_ratio_low)
            | (ratio > 1.0 + self.clip_ratio_high)
        ).to(dtype=current_log_probs.dtype)
        return PolicyObjectiveOutput(loss=policy_loss, clipped=clipped)
