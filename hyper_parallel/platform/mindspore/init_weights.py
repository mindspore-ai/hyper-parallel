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
"""MindSpore implementation of on-device weight initialization."""

from contextlib import contextmanager

from mindspore import nn, DeviceCtx


@contextmanager
def init_on_device(device, include_buffers=False):
    """Context manager that initializes model parameters (and optionally
    buffers) on *device*.

    Args:
        device: ``"meta"`` to skip allocation (no real memory used), or a
            real device string (e.g. ``"cpu"``, ``"npu"``) for placement.
        include_buffers (bool): Also redirect buffers to *device*.
    """
    if device == "meta":
        with DeviceCtx(device):
            yield
    else:
        orig_insert_param = nn.Cell.insert_param_to_cell
        orig_register_buffer = nn.Cell.register_buffer

        # pylint: disable=W0212
        def _insert_param_to_cell(module, param_name, param, check_name_contain_dot=True):
            orig_insert_param(module, param_name, param, check_name_contain_dot)
            if param is not None:
                orig_param = module._params[param_name]
                param_cls = type(orig_param)
                attrs = {"name", "requires_grad", "layerwise_parallel", "parallel_optimizer"}
                kwargs = {k: v for k, v in orig_param.__dict__.items() if k in attrs}
                kwargs["requires_grad"] = param.requires_grad
                module._params[param_name] = (param if param.device == device
                                              else param_cls(orig_param.to(device=device), **kwargs))

        # pylint: disable=W0212
        def _register_buffer(module, name, tensor, persistent=True):
            orig_register_buffer(module, name, tensor, persistent=persistent)
            if tensor is not None:
                module._buffers[name] = tensor.to(device=device)

        try:
            nn.Cell.insert_param_to_cell = _insert_param_to_cell
            if include_buffers:
                nn.Cell.register_buffer = _register_buffer
            yield
        finally:
            nn.Cell.insert_param_to_cell = orig_insert_param
            if include_buffers:
                nn.Cell.register_buffer = orig_register_buffer
