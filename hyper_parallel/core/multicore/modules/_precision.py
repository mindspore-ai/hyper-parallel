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
"""Preserve full-precision control parameters during ordinary model dtype conversion."""

from collections.abc import Callable, Iterable

import torch


def preserve_fp32_parameters(
    module: torch.nn.Module, transform: Callable[[torch.Tensor], torch.Tensor], names: Iterable[str],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap Module._apply without rounding protected Parameters or their existing gradients.

    Args:
        module: Module holding the protected control Parameters.
        transform: Requested parameter device or dtype transformation.
        names: Names of control Parameters which must retain FP32 values.
    """
    protected = set()
    for name in names:
        parameter = getattr(module, name, None)
        if parameter is not None:
            protected.add(id(parameter))
            if parameter.grad is not None:
                protected.add(id(parameter.grad))

    def apply(tensor: torch.Tensor) -> torch.Tensor:
        """Retain source FP32 values while following the requested target device.

        Args:
            tensor: Parameter or gradient being transformed.
        """
        converted = transform(tensor)
        if id(tensor) not in protected or converted.dtype == torch.float32:
            return converted
        if tensor.is_meta:
            return converted.to(dtype=torch.float32)
        return tensor.to(device=converted.device, dtype=torch.float32)

    return apply
