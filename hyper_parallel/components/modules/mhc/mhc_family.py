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
"""mHC replacement entry points and family declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn
from hyper_parallel.components.modules.module_family import (
    ModuleFamily,
    SelectorSpec,
    VariantRule,
    build_family,
    register_module_family,
)
from hyper_parallel.models.replacement import module_replacement


@module_replacement
def MHCPre(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
) -> nn.Module:
    """Construct a tracked mhc_pre member without changing its execution contract."""
    return build_family(MHCPre, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh)


register_module_family(ModuleFamily(
    name="mhc_pre", baseline=MHCPre,
    variants={"local": "hyper_parallel.components.modules.mhc.mhc_pre.MhcPreModule"}, default_variant="local",
))


@module_replacement
def MHCPost(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
) -> nn.Module:
    """Construct a tracked mhc_post member without changing its execution contract."""
    return build_family(MHCPost, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh)


register_module_family(ModuleFamily(
    name="mhc_post", baseline=MHCPost,
    variants={"local": "hyper_parallel.components.modules.mhc.mhc_post.MhcPostModule"}, default_variant="local",
))


@module_replacement
def PipelinedMhc(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
) -> nn.Module:
    """Construct a tracked PipelinedMhc member without changing its execution contract."""
    return build_family(PipelinedMhc, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh)


register_module_family(ModuleFamily(
    name="pipelined_mhc", baseline=PipelinedMhc,
    variants={"local": "hyper_parallel.components.modules.mhc.pipelined_mhc.PipelinedMhcModule"},
    default_variant="local",
))


@module_replacement
def MegaMhc(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Mapping[str, Any] | None = None, mesh: Any = None, backend: str = "torch",
) -> nn.Module:
    """Build a complete shifted boundary with an explicit execution backend."""
    return build_family(MegaMhc, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, backend=backend)


register_module_family(ModuleFamily(
    name="mega_mhc", baseline=MegaMhc,
    variants={"local": "hyper_parallel.components.modules.mhc.mega_mhc.MegaMhcModule",
              "fused": "hyper_parallel.components.modules.mhc.mega_mhc.FusedMegaMhcModule"},
    selector_specs={"backend": SelectorSpec(str, "torch")},
    default_variant="local",
    variant_rules=(VariantRule({"backend": "fused", "parallel": False}, "fused"),),
))
