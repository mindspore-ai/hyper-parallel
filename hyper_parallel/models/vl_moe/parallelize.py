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
"""Parallelization for VL-MoE model.

Stub implementation — provides the required API surface so that the model
registers correctly with hyper-parallel's ``ModelSpec``.  Full TP/EP/PP
parallelization logic should be filled in here when needed.
"""

from __future__ import annotations

__all__ = [
    "parallelize_vl_moe",
    "pipeline_vl_moe_for_trainer",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


def parallelize_vl_moe(model: nn.Module, mesh, cfg) -> nn.Module:
    """Apply TP/EP/CP parallelism to the VL-MoE model.

    Currently a no-op pass-through.  Replace with actual parallelization
    logic (e.g. ``dtensor`` device-mesh sharding, expert-parallel
    dispatch, etc.) when ready.
    """
    return model


def pipeline_vl_moe_for_trainer(model: nn.Module, mesh, cfg):
    """Set up pipeline-parallel stages for the VL-MoE model.

    Currently returns ``None`` to indicate no PP schedule.  Replace with
    actual stage partitioning when ready.
    """
    return None