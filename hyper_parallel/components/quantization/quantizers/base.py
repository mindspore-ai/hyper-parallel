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
"""Abstract interface for low-precision tensor quantizers."""

from abc import ABC, abstractmethod

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.tensor import QuantizedTensor


class Quantizer(ABC):
    """Convert high-precision tensors into typed quantized representations."""

    @abstractmethod
    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensor:
        """Quantize a tensor in the requested directional representations."""
