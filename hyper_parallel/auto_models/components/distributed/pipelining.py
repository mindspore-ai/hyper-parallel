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
"""Pipeline Parallelism — stub implementation for 06 §8.2 / 01 §8.2.

AutoPipeline splits the model into PP stages in-place (model.parts).
"""

import logging
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)


class AutoPipeline:
    """Pipeline Parallelism manager stub."""

    def __init__(self, pipeline_config: Any, mesh_context: Any) -> None:
        """Store the pipeline config and mesh context for a later build."""
        self.config = pipeline_config
        self.mesh_context = mesh_context

    def build(self, model: nn.Module, loss_fn: nn.Module | None = None) -> None:
        """Split model into PP stages in-place.

        Stub: logs a warning and leaves model unchanged. `model` and
        `loss_fn` are accepted for interface compatibility only.
        """
        _ = (model, loss_fn)  # Stub: accepted for interface compatibility.
        logger.warning("AutoPipeline.build is a stub; PP stage splitting not implemented")


def _instantiate_pipeline(pipeline_config: Any, mesh_context: Any) -> AutoPipeline | None:
    """Factory used by instantiate_infrastructure (01 §8.1 / 06 §8.2)."""
    if pipeline_config is None:
        return None
    return AutoPipeline(pipeline_config, mesh_context)
