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
"""Small model-neutral module factory used by the NPU replacement matrix."""

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules import RMSNorm
from hyper_parallel.models.replacement import module_replacement


@module_replacement
def replace_rms_norm(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> nn.Module:
    """Reuse one native RMSNorm's weight and epsilon in the generic RMSNorm."""
    return RMSNorm(module=module, module_fqn=module_fqn, context=context)
