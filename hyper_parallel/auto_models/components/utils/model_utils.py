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
"""Generic model utilities.

init_empty_weights: context manager that allocates meta tensors.
This is a lightweight stub that can be replaced with accelerate/accelerate
when available.
"""

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
from torch import nn


@contextmanager
def init_empty_weights(include_buffers: bool = False) -> Iterator[None]:  # pylint: disable=unused-argument
    """Context manager where nn.Parameter/Buffer are allocated on meta device.

    Stub implementation based on the standard "meta" dispatch trick.

    Args:
        include_buffers: Kept for API compatibility with accelerate's
            ``init_empty_weights``; currently unused by this stub.

    Yields:
        None.
    """
    device = torch.device("meta")
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(
        module: nn.Module,
        name: str,
        param: Optional[torch.Tensor],
    ) -> None:
        """Register ``param`` on ``module`` after moving it to the meta device.

        Args:
            module: Target module.
            name: Parameter name.
            param: Parameter to register, or None.
        """
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])

            # Standard nn.Parameter only accepts requires_grad, not arbitrary __dict__ attributes
            # (e.g., TransformerEngine sets tensor_model_parallel on weights)
            if param_cls is nn.Parameter:
                kwargs = {"requires_grad": param.requires_grad}
                is_hf_initialized = None
            else:
                kwargs = module._parameters[name].__dict__.copy()
                kwargs["requires_grad"] = param.requires_grad
                is_hf_initialized = kwargs.pop("_is_hf_initialized", None)

            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            if is_hf_initialized is not None:
                setattr(module._parameters[name], "_is_hf_initialized", is_hf_initialized)

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
