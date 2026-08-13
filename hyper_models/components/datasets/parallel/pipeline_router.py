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
"""Reserved stage-aware Pipeline batch-routing contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class PipelineBatchRouter(ABC):
    """Route source fields to the PP stage that consumes each field.

    A future Omni implementation can route raw modality inputs to encoder
    ranks, encoded features to LLM PP0, and loss fields to the last LLM stage.
    No full-batch PP broadcast implementation is provided intentionally.
    """

    @abstractmethod
    def route_source_batch(
            self,
            source_batch: Mapping[str, Any] | None,
            *,
            source_exhausted: bool,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Return the fields owned by the current PP stage.

        Args:
            source_batch: Batch read by the configured PP source stage.
            source_exhausted: Whether the source iterator is exhausted.

        Returns:
            Stage-local source batch and synchronized exhaustion state.
        """
        raise NotImplementedError


__all__ = ["PipelineBatchRouter"]
