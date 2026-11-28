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
"""V4.1 Host Engram validation backed by shared lifecycle ownership."""

from __future__ import annotations

from torch import nn

from hyper_parallel.components.modules.engram.host_state import (
    HostEngramState as DeepseekV41HostState,
    build_engram_external_state,
)
from hyper_parallel.components.modules.engram.host_engram_table import HostEngramTable
from hyper_parallel.models.external_state import ExternalBuildContext


def build_deepseek_v41_external_state(
        model: nn.Module, context: ExternalBuildContext,
) -> DeepseekV41HostState | None:
    """Reject legacy device Engram modules before binding shared Host state."""
    # The architecture imports its spec during registration, so resolve its class after construction.
    from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (  # pylint: disable=C0415
        DeepseekV41Engram,
    )

    if getattr(getattr(model, "config", None), "engram_storage_backend", "device") == "host":
        if any(isinstance(module, DeepseekV41Engram) and not isinstance(module.embed, HostEngramTable)
               for module in model.modules()):
            raise ValueError("Host Engram configuration still contains a device replacement")
    return build_engram_external_state(model, context)


__all__ = ["DeepseekV41HostState", "build_deepseek_v41_external_state"]
