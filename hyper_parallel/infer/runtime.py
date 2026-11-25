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
"""Code-injected generation state and structured results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hyper_parallel.infer.config import GenerationOptions
from hyper_parallel.infer.utils import GenerationConfig


@dataclass
class GenerationRuntime:
    """Runtime-only objects for independent single-device generation lanes.

    Model-parallel handoffs are reserved and rejected until their own acceptance.
    The legacy GenerationConfig API retains its existing parallel extensions.
    """

    model_adapter: Any = None
    logits_layout: Any = None
    logits_group: Any = None
    sampling_owner: int | None = None
    sampling_group: Any = None
    forward_sync_group: Any = None
    device: torch.device | None = None
    rng: torch.Generator | None = None

    def bind(self, options: GenerationOptions) -> GenerationConfig:
        """Adapt resolved parameters to the existing generation engine.

        Args:
            options: Sampling parameters with special-token IDs already resolved.

        Returns:
            Engine configuration combining the parameters with this runtime's RNG.
        """
        if any(value is not None for value in (
                self.logits_layout, self.logits_group, self.sampling_owner,
                self.sampling_group, self.forward_sync_group)):
            raise NotImplementedError("GenerationRuntime currently supports independent DP lanes only")
        if self.model_adapter is not None and not self.model_adapter.supports_generation:
            raise NotImplementedError("model adapter does not declare supports_generation")
        return GenerationConfig(**options.to_dict(), rng=self.rng)


@dataclass
class GenerationResult:
    """Final sequences with left padding removed and per-request stopping metadata.

    Generated lengths include a terminating EOS token, and exclude right padding.
    Completion tokens occupy ``[prompt_length:prompt_length + generated_length]``.
    """

    sequences: torch.Tensor
    prompt_lengths: torch.Tensor
    generated_lengths: torch.Tensor
    finish_reasons: list[str]
