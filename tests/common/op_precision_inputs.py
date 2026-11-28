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
"""Already-quantized deterministic inputs shared across operator domains."""

from __future__ import annotations

import torch


class Inputs:
    """Construct already-quantized reproducible CPU leaves without global RNG state."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.generator = torch.Generator().manual_seed(20261120)

    def tensor(
        self, *shape: int, dtype: torch.dtype | None = None, grad: bool = True, scale: float = 0.2,
    ) -> torch.Tensor:
        """Return a leaf with moderate values to avoid accidental unstable TopK ties."""
        values = torch.randn(shape, generator=self.generator) * scale
        return values.to(self.dtype if dtype is None else dtype).requires_grad_(grad)
