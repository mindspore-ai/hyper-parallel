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
platform = get_platform()
DTensorBase = platform.DTensorBase
Tensor = platform.Tensor


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
    def from_local(local_tensor: Tensor, layout: Layout):
        d_tensor =  DTensor(local_tensor, layout)
        return d_tensor

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
        from hyper_parallel.core.tensor_redistribution import _tensor_redistribution
        out = _tensor_redistribution.reduce_partial(self, to_layout)
        return out
