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
"""Cost-model interfaces for multimodal local-batch planning."""

from __future__ import annotations

import math
from typing import Protocol

from hyper_parallel.distributed_data.schema import LocalBatchMeta, WorkloadCost


class CostModel(Protocol):
    """Estimate normalized stage costs from lightweight local-batch metadata."""

    def estimate(self, metadata: LocalBatchMeta) -> WorkloadCost:
        """Estimate one local batch's workload cost."""


class LinearMultimodalCostModel:
    """Small deterministic baseline cost model.

    The model is deliberately replaceable. ``cost_hint`` takes precedence so
    offline profiling can provide calibrated values without changing planner
    code. Otherwise token counts are normalized to 1K-token units and I/O to
    MiB units.
    """

    def __init__(
        self,
        *,
        io_weight: float = 1.0,
        transform_weight: float = 1.0,
        encoder_weight: float = 1.0,
        llm_weight: float = 1.0,
        memory_weight: float = 1.0,
        communication_weight: float = 1.0,
    ) -> None:
        """Initialize weights for normalized workload components."""
        self._weights = (
            io_weight,
            transform_weight,
            encoder_weight,
            llm_weight,
            memory_weight,
            communication_weight,
        )
        if any(
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight < 0
            for weight in self._weights
        ):
            raise ValueError(f"Cost-model weights must be finite and non-negative, but got {self._weights}.")

    def estimate(self, metadata: LocalBatchMeta) -> WorkloadCost:
        """Estimate normalized I/O, encoder, LLM, memory, and communication cost."""
        if metadata.cost_hint is not None:
            return metadata.cost_hint

        io_weight, transform_weight, encoder_weight, llm_weight, memory_weight, comm_weight = self._weights
        media_tokens = metadata.vision_tokens + metadata.audio_tokens
        total_tokens = metadata.text_tokens + media_tokens
        media_units = media_tokens / 1024.0
        total_units = total_tokens / 1024.0
        return WorkloadCost(
            io=(metadata.io_bytes / (1024.0 * 1024.0)) * io_weight,
            transform=media_units * transform_weight,
            encoder=(media_units ** 2) * encoder_weight,
            llm=(total_units ** 2) * llm_weight,
            memory=total_units * memory_weight,
            communication=media_units * comm_weight,
        )
