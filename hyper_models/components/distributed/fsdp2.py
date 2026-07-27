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
"""FSDP2Manager — stub implementation for 06 §4.

The fully_shard integration is intentionally a no-op until the core
fully_shard API (core/fully_shard) is wired in.
"""

import logging
from typing import Any

import torch.nn as nn

from hyper_models.components.distributed.config import FSDP2Config

logger = logging.getLogger(__name__)


class FSDP2Manager:
    """FSDP2 wrapper manager.

    Canonical 2-arg constructor: (config, mesh_context).
    mesh_context carries device_mesh and device (C6 alignment).
    """

    def __init__(self, config: FSDP2Config, mesh_context: Any):
        self.config = config
        self.mesh_context = mesh_context
        self.device_mesh = getattr(mesh_context, "device_mesh", None)
        self.device = getattr(mesh_context, "device", "cuda")

    def parallelize(
        self,
        model: nn.Module,
        tp_shard_plan: Any = None,
        tp_grad_info: dict | None = None,
    ) -> nn.Module:
        """Apply FSDP2 wrapping on top of TP/CP sharding.

        Currently a stub: logs and returns model unchanged.
        """
        logger.warning(
            "FSDP2Manager.parallelize is a stub (tp_grad_info=%s, mesh=%s). "
            "Real fully_shard integration pending.",
            tp_grad_info is not None,
            self.device_mesh,
        )
        return model


def _instantiate_fsdp2(*, config: Any, mesh_context: Any) -> FSDP2Manager | None:
    """Factory used by instantiate_infrastructure (01 §8.1 canonical signature)."""
    if config is None:
        return None
    if isinstance(config, FSDP2Config):
        return FSDP2Manager(config=config, mesh_context=mesh_context)
    logger.warning("Unexpected FSDP2 config type %s; returning None", type(config))
    return None
