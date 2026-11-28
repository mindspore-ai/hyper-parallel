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
"""Empty / on-device weight initialization utilities."""
# pylint: disable=C9006,C9007

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import sys
from typing import Optional

import torch

from hyper_parallel.core.dtensor._utils import init_on_device as _init_on_device


def _get_npu_fill_state() -> bool:
    """Read NPU's independent uninitialized-memory fill flag when ``torch_npu`` is loaded."""
    torch_npu_c = getattr(sys.modules.get("torch_npu"), "_C", None)
    getter = getattr(torch_npu_c, "_npu_get_fill_uninitialized_memory", None)
    return bool(getter()) if callable(getter) else False


def _get_npu_fill_setter() -> Optional[Callable[[bool], None]]:
    """Return ``torch_npu._C._npu_set_fill_uninitialized_memory`` when ``torch_npu`` is loaded."""
    torch_npu_c = getattr(sys.modules.get("torch_npu"), "_C", None)
    setter = getattr(torch_npu_c, "_npu_set_fill_uninitialized_memory", None)
    return setter if callable(setter) else None


@contextmanager
def init_empty_weights(include_buffers: bool = False) -> Iterator[None]:
    """Context manager that initializes model parameters (and optionally
    buffers) on the meta device.

    Models created under this context have no real data, which avoids allocating
    any NPU/GPU/CPU memory.  This is useful when the model is too large to fit in RAM
    and the weights will be loaded from a checkpoint afterwards.

    Args:
        include_buffers (bool): If ``True``, buffers are also placed on the meta
            device.  Defaults to ``False``.

    Example::

        from hyper_parallel import init_empty_weights

        with init_empty_weights():
            model = MyLargeModel()  # no memory allocated

    Note:
        A model initialised under this context has **no weights**.  You cannot
        call ``model.to(some_device)`` directly; load a checkpoint first.

        Deterministic memory filling is suspended only inside this context:
        factories may allocate CPU tensors before registration moves them to
        meta. Filling that discarded storage defeats empty initialization.
        The deterministic algorithm setting is unchanged.
    """
    fill_uninitialized_memory = torch.utils.deterministic.fill_uninitialized_memory
    # torch_npu patches only the fill_uninitialized_memory setter while keeping its
    # NPU-side fill flag False by default; preserve NPU's own flag across exit.
    npu_fill_uninitialized_memory = _get_npu_fill_state()
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
        with _init_on_device(torch.device("meta"), include_buffers=include_buffers):
            yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = fill_uninitialized_memory
        npu_setter = _get_npu_fill_setter()
        if npu_setter is not None:
            npu_setter(npu_fill_uninitialized_memory)


@contextmanager
def init_on_device(device, include_buffers=False):
    """Context manager that initializes model parameters (and optionally buffers) on *device*.

    Args:
        device: Target device for parameter (and buffer) allocation.
        include_buffers (bool): If ``True``, buffers are also placed on *device*.
            Defaults to ``False``.

    Example::

        from hyper_parallel import init_on_device
        import torch

        with init_on_device(torch.device("npu")):
            model = MyModel()  # parameters live on Ascend
    """
    with _init_on_device(device, include_buffers=include_buffers):
        yield
