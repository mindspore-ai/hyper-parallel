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
"""Device-table Engram replacement entry point and family registration."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from torch import nn
from hyper_parallel.components.modules.family import SelectorSpec, VariantRule
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family


@module_replacement
def Engram(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
    engram_storage_backend: str = "device",
) -> nn.Module:
    """Select host/device table storage independently of operator choice and full-module placement."""
    return build_family(Engram, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, engram_storage_backend=engram_storage_backend)


register_module_family(ModuleFamily(
    name="engram", baseline=Engram,
    variants={"local": "hyper_parallel.components.modules.engram.engram.EngramModule",
              "host": "hyper_parallel.components.modules.engram.host_engram.HostEngramModule"},
    selector_specs={"engram_storage_backend": SelectorSpec(str, "device")},
    default_variant="local",
    variant_rules=(VariantRule({"engram_storage_backend": "host", "parallel": False}, "host"),),
))
