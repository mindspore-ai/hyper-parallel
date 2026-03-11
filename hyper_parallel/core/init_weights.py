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

from contextlib import contextmanager
from hyper_parallel.platform import get_platform

platform = get_platform()


@contextmanager
def init_empty_weights(include_buffers=False):
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
    """
    with platform.init_on_device(platform.meta_device, include_buffers=include_buffers):
        yield


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
    with platform.init_on_device(device, include_buffers=include_buffers):
        yield
