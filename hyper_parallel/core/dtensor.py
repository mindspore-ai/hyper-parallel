# Copyright 2025 Huawei Technologies Co., Ltd
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
"""dtensor"""
import copy as cp
from hyper_parallel.core.layout import Layout
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType
platform = get_platform()
DTensorBase = platform.DTensorBase
Tensor = platform.Tensor


class SkipDTensorDispatch():
    def __enter__(self):
        # pylint: disable=C0415
        from hyper_parallel.core.shard._op_dispatch import disable_dtensor_dispatch
        disable_dtensor_dispatch()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # pylint: disable=C0415
        from hyper_parallel.core.shard._op_dispatch import enable_dtensor_dispatch
        enable_dtensor_dispatch()


class DTensor(DTensorBase):
    """
    DTensor
    """
    _local_tensor: Tensor
    _layout: Layout

    def __init_data__(self, local_tensor, layout):
        self._local_tensor = local_tensor
        self._layout = layout

    @property
    def layout(self):
        """Sharding state for dtensor"""
        if not hasattr(self, '_layout'):
            return None
        return self._layout

    @staticmethod
    def _infer_cpu_device(local_tensor: Tensor):
        try:
            device_obj = local_tensor.device
        except Exception:
            return None
        try:
            device_str = str(device_obj)
        except Exception:
            return None
        if "cpu" in device_str.lower():
            return "CPU"
        return None

    @staticmethod
    def from_local(local_tensor: Tensor, layout: Layout, device: str = None):
        if device is None and getattr(platform, "platform_type", None) == PlatformType.MINDSPORE:
            device = DTensor._infer_cpu_device(local_tensor)
        if device is None:
            return DTensor(local_tensor, layout)
        if getattr(platform, "platform_type", None) == PlatformType.MINDSPORE:
            return DTensor(local_tensor, layout, device=device)
        return DTensor(local_tensor, layout)

    def to_local(self):
        """covert global_tensor to local_tensor"""
        return self._local_tensor

    @property
    def shape(self):
        return self._layout.get_global_shape(self._local_tensor.shape)

    @property
    def local_shape(self):
        return self._local_tensor.shape

    def redistribute(self, dst_layout):
        """
        Redistribute dtensor to destination layout.
        """
        # pylint: disable=C0415
        from hyper_parallel.core.tensor_redistribution import _tensor_redistribution
        out = _tensor_redistribution.redistribution(self, dst_layout)
        return out

    def reduce_partial(self):
        """
        Reduce partial sharding state for dtensor.

        """
        if not self.layout:
            return self
        to_layout = cp.deepcopy(self.layout)
        to_layout.reset_partial()
        # pylint: disable=C0415
        from hyper_parallel.core.tensor_redistribution import _tensor_redistribution
        out = _tensor_redistribution.reduce_partial(self, to_layout)
        return out
