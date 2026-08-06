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
"""Reusable policy regularizers hidden behind complete algorithm Recipes."""

from dataclasses import dataclass
from typing import Any, Protocol


class Regularizer(Protocol):
    """Internal regularization seam selected by a complete Recipe."""

    def compute(
        self,
        current_log_probs: Any,
        reference_log_probs: Any,
    ) -> tuple[Any, Any]:
        """Return weighted and raw per-token regularization values."""


def low_variance_kl(
    current_log_probs: Any,
    target_log_probs: Any,
) -> Any:
    """Non-negative k3 KL estimate used by GRPO and PPO Recipes."""
    log_ratio = target_log_probs - current_log_probs
    return (log_ratio.exp() - log_ratio - 1.0).clamp(min=0.0, max=10.0)


@dataclass(frozen=True)
class LowVarianceKLRegularizer:
    """Apply a configurable coefficient to the k3 reference-policy KL."""

    coefficient: float

    def compute(
        self,
        current_log_probs: Any,
        reference_log_probs: Any,
    ) -> tuple[Any, Any]:
        """Compute weighted and raw low-variance KL estimates."""
        raw = low_variance_kl(current_log_probs, reference_log_probs)
        return self.coefficient * raw, raw
