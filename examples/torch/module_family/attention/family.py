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
"""Public attention family; importing this module installs its declaration."""

from __future__ import annotations

from typing import Any

from torch import nn

from examples.torch.module_family.config import DecoderConfig
from hyper_parallel.components.modules import BackendSpec, ModuleFamily, build_family, register_module_family
from hyper_parallel.models.replacement import module_replacement


@module_replacement
def CausalAttention(*, config: DecoderConfig | None = None, module: nn.Module | None = None,
                    module_fqn: str = "", context: Any = None, mesh: Any = None,
                    backend: str | None = None) -> nn.Module:
    """Construct the attention family using complete config or a source member."""
    return build_family(CausalAttention, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, backend=backend)


register_module_family(ModuleFamily(
    name="example_causal_attention", baseline=CausalAttention,
    variants={"local": "examples.torch.module_family.attention.attention.CausalAttentionModule"},
    default_variant="local", local_variant_of={"local": "local"},
    backend_specs={"torch": BackendSpec(("local",))}, default_backends={"local": "torch"},
))
