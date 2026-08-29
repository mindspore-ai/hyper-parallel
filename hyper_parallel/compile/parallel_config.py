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
"""
Parallel Configuration - FSDP Configuration

Simple configuration for FSDP training.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch.distributed as dist


@dataclass
class ParallelConfig:
    """Parallel configuration for FSDP (+ optional TP) training.

    ``fsdp_degree``: size of the FSDP group. ``None`` (default) falls back to
    ``world_size`` for backward compatibility with the FSDP-only path. In a
    TP+FSDP hybrid the FSDP group is a proper sub-group of the world, so it
    must be set explicitly — the ``GraphTrainer`` fills it from the automodel
    ``MeshContext`` when one is provided.
    """

    enable_overlap: bool = True
    fsdp_degree: Optional[int] = None
    tp_size: int = 1

    @property
    def fsdp_enabled(self) -> bool:
        """Check if FSDP is enabled (always True if distributed)"""
        return dist.is_initialized() and dist.get_world_size() > 1


def parallel_config(**kwargs: Any) -> ParallelConfig:
    """Convenience function to create parallel configuration"""
    return ParallelConfig(**kwargs)


__all__ = ["ParallelConfig", "parallel_config"]
