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

import torch


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """Context manager where nn.Parameter/Buffer are allocated on meta device.

    Stub implementation based on the standard "meta" dispatch trick.
    """
    old_init = torch.nn.Module.__init__
    old_reset_parameters = getattr(torch.nn.Module, "reset_parameters", None)

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        for name, param in self.named_parameters(recurse=False):
            if param is not None:
                self._parameters[name] = torch.nn.Parameter(
                    torch.empty_like(param, device="meta"),
                    requires_grad=param.requires_grad,
                )
        if include_buffers:
            for name, buf in self.named_buffers(recurse=False):
                if buf is not None:
                    self._buffers[name] = torch.empty_like(buf, device="meta")

    def new_reset_parameters(self):
        pass

    torch.nn.Module.__init__ = new_init
    if old_reset_parameters is not None:
        torch.nn.Module.reset_parameters = new_reset_parameters
    try:
        yield
    finally:
        torch.nn.Module.__init__ = old_init
        if old_reset_parameters is not None:
            torch.nn.Module.reset_parameters = old_reset_parameters
