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

import mindspore as ms
from mindspore import mint, nn

from hyper_parallel.platform.mindspore.utils import normalize_runtime_device


def _cell_to_empty(self, device=None, recurse=True):
    """Patch for ``nn.Cell.to_empty`` (init.md): ``param.set_data(mint.empty_like(...))``.

    Walks ``parameters_and_names(expand=...)`` — ``recurse`` maps to MindSpore's
    ``expand`` flag. ``device`` defaults from ``device_target`` when omitted so
    ``net.to_empty()`` works without arguments.
    """
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.core.dtensor.dtensor import DTensor

    if device is None:
        device = ms.get_context("device_target")
    for _, param in self.parameters_and_names(expand=recurse):
        if param is None:
            continue
        if isinstance(param, DTensor):
            local = param.to_local()
            new_tensor = mint.empty_like(local, device=device)
            param.set_data(new_tensor)
            continue
        new_tensor = mint.empty_like(param, device=device)
        param.set_data(new_tensor)
    return self


def _install_cell_to_empty_patch():
    if getattr(nn.Cell, "_hyper_parallel_to_empty_installed", False):
        return
    nn.Cell.to_empty = _cell_to_empty
    nn.Cell._hyper_parallel_to_empty_installed = True # pylint: disable=W0212


def _check_valid_init_device(device: str):
    """Validate user-provided init device for MindSpore init_on_device."""
    if device not in {"npu", "cpu", "meta"}:
        raise ValueError(f'Unsupported device "{device}", only "npu", "cpu", and "meta" are allowed.')


@contextmanager
def init_on_device(device, include_buffers=False):
    """Context manager that initializes model parameters (and optionally
    buffers) on *device*.

    Args:
        device: ``"meta"`` to skip allocation (no real memory used), or a
            real device string (e.g. ``"cpu"``, ``"npu"``) for placement.
        include_buffers (bool): Also redirect buffers to *device*.
    """
    if include_buffers:
        raise ValueError("MindSpore platform does not support include_buffers=True.")
    _check_valid_init_device(device)
    orig_insert_param = nn.Cell.insert_param_to_cell

    # pylint: disable=W0212
    def _insert_param_to_cell(module, param_name, param, check_name_contain_dot=True):
        orig_insert_param(module, param_name, param, check_name_contain_dot)
        if param is not None:
            def _custom_kwargs(param_obj):
                ms_graph_attrs = [
                    "init_mode", "is_default_input_init", "_param_info", "is_init", "_inited_param", "_sliced",
                    "requires_aggr", "_cast_type", "_unique", "is_in_parallel", "_pipeline_stage_list", "load",
                ]
                return {k: v for k, v in param_obj.__dict__.items() if k not in ms_graph_attrs}
            orig_param = module._params[param_name]
            # get custom kwargs from orig_param, do not change the original param __dict__
            kwargs = _custom_kwargs(orig_param)
            kwargs["name"] = param.name
            kwargs["requires_grad"] = param.requires_grad
            param_cls = type(orig_param)
            param_device = normalize_runtime_device(param.device)
            module._params[param_name] = (param if param_device == device
                                          else param_cls(orig_param.to(device=device), **kwargs))

    try:
        nn.Cell.insert_param_to_cell = _insert_param_to_cell
        yield
    finally:
        nn.Cell.insert_param_to_cell = orig_insert_param
