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
"""Wan attention backend configuration through module replacement rules."""

from collections.abc import Mapping
from typing import Any

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.replacement import module_replacement


@module_replacement
def replace_wan_attention_backend(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
    backend: str,
) -> nn.Module:
    """Configure an official Diffusers Wan attention module for FA execution.

    The source module, parameters, state-dict keys, and official
    ``WanAttnProcessor`` are retained. Only the processor's Diffusers backend
    selection is changed, so checkpoint and FSDP structure remain identical.

    Args:
        module: Official Diffusers ``WanAttention`` instance.
        module_fqn: Fully qualified module name used in diagnostics.
        context: HyperParallel replacement context.
        backend: Diffusers attention backend, for example ``_native_npu``.

    Returns:
        The same official attention module after backend configuration.

    Raises:
        TypeError: If the selected module does not expose the Diffusers
            attention-backend contract.
    """
    del context
    set_backend = getattr(module, "set_attention_backend", None)
    if not callable(set_backend):
        raise TypeError(
            f"{module_fqn}: Wan attention replacement requires "
            "Diffusers AttentionModuleMixin.set_attention_backend()"
        )
    set_backend(backend)
    return module


__all__ = ["replace_wan_attention_backend"]
