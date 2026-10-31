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

"""Structure-preserving replacement for the Triton-Ascend GDN primitive."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from importlib import import_module
from typing import Any

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.replacement import module_replacement


@module_replacement
class TritonGDN:
    """Factory selecting the Triton-Ascend GDN primitive for an existing module.

    The returned module retains its source type so the model's forward, CP
    wrapper, and checkpoint layout remain unchanged.
    """

    def __new__(
        cls,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> nn.Module:
        """Bind the full Triton-Ascend GDN forward/backward without changing weights.

        Args:
            module: GDN module exposing an instance-level chunk primitive.
            module_fqn: Source module's fully qualified name.
            context: Model replacement context.

        Returns:
            A shallow copy with its chunk primitive bound to the Triton implementation.

        Raises:
            TypeError: The source does not expose the expected primitive.
        """
        del cls, context
        if not callable(getattr(module, "chunk_gated_delta_rule", None)):
            raise TypeError(f"{module_fqn}: GDN replacement requires callable chunk_gated_delta_rule")
        # Triton-Ascend is optional and should load only when this replacement is selected.
        triton_rule = import_module(
            "hyper_parallel.components.functional.gated_delta_net"
        ).chunk_gated_delta_rule
        replacement = copy.copy(module)
        replacement.chunk_gated_delta_rule = triton_rule
        return replacement
