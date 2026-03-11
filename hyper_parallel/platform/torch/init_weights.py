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
"""PyTorch implementation of on-device weight initialization."""

from contextlib import contextmanager

from torch import nn


@contextmanager
def init_on_device(device, include_buffers=False):
    """Monkey-patch ``nn.Module`` so that every parameter (and optionally every
    buffer) is placed on *device* at registration time.

    Args:
        device (torch.device): Target device.
        include_buffers (bool): Also redirect buffers to *device*.
    """
    orig_register_parameter = nn.Module.register_parameter
    orig_register_buffer = nn.Module.register_buffer

    # pylint: disable=W0212
    def _register_parameter(module, name, param):
        orig_register_parameter(module, name, param)
        if param is not None:
            orig_param = module._parameters[name]
            param_cls = type(orig_param)
            kwargs = orig_param.__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = (param if param.device == device
                                        else param_cls(orig_param.to(device), **kwargs))

    # pylint: disable=W0212
    def _register_buffer(module, name, buffer, persistent=True):
        orig_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    try:
        nn.Module.register_parameter = _register_parameter
        if include_buffers:
            nn.Module.register_buffer = _register_buffer
        yield
    finally:
        nn.Module.register_parameter = orig_register_parameter
        if include_buffers:
            nn.Module.register_buffer = orig_register_buffer
