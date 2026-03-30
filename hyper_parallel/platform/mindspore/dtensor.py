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
"""mindspore dtensor base"""
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore._c_expression import NoFallbackGuard


class DTensorBase(Tensor):
    """
    DTensorBase - Base class for distributed tensors in MindSpore.

    This class extends Tensor to support distributed tensor operations with
    device mesh and placement specifications.
    """

    def __new__(cls, local_tensor, device_mesh=None, placements=None):
        """
        Create a new DTensorBase instance.

        Args:
            local_tensor: The local tensor shard or another DTensorBase instance.
            device_mesh: The device mesh describing the device topology.
            placements: The placement strategy for each mesh dimension.
            device: The device type (default: "Ascend").
        """
        npu_device = "Ascend"
        if isinstance(local_tensor, DTensorBase):
            device_local_tensor = local_tensor.to_local()
            if device_local_tensor.device != "meta" and not device_local_tensor.has_init:
                device_local_tensor = device_local_tensor.to(npu_device)
            t = Tensor._make_subclass(cls, device_local_tensor)
            copy_placements = local_tensor.layout.alias_placements if local_tensor.layout else local_tensor.placements
            t.__init_data__(device_local_tensor, local_tensor.device_mesh, copy_placements)
            return t

        if local_tensor is None:
            raise ValueError(
                "DTensorBase: local_tensor must not be None when constructing from a raw tensor."
            )
        if device_mesh is None:
            raise ValueError(
                "DTensorBase: device_mesh must be a DeviceMesh instance, got None."
            )
        if placements is None:
            raise ValueError(
                "DTensorBase: placements must be a sequence of Placement objects, got None."
            )
        device_local_tensor = local_tensor
        if device_local_tensor.device != "meta" and not device_local_tensor.has_init:
            device_local_tensor = device_local_tensor.to(npu_device)
        if local_tensor.has_init:
            local_tensor.init_device = npu_device
        t = Tensor._make_subclass(cls, device_local_tensor)
        t.__init_data__(device_local_tensor, device_mesh, placements)
        return t

    def asnumpy(self):
        """
        Numpy value of local tensor.
        """
        return self._local_tensor.asnumpy()

    def __str__(self):
        return str(self._local_tensor)

    def __copy__(self):
        """
        Create a shallow copy of the DTensorBase instance.

        This method ensures that device_mesh and placements are correctly
        propagated when creating a copy (e.g., for optimizer states).
        """
        # Get device_mesh and placements from layout (prefer alias_placements to preserve multi-axis ordering)
        device_mesh = getattr(self, '_device_mesh', None)
        placements = None

        if hasattr(self, '_layout') and self._layout is not None:
            if device_mesh is None:
                device_mesh = self._layout.mesh
            placements = self._layout.alias_placements

        if placements is None:
            placements = getattr(self, '_placements', None)

        if device_mesh is None or placements is None:
            raise ValueError(
                "DTensorBase.__copy__: cannot copy without device_mesh and placements; "
                f"device_mesh={device_mesh!r}, placements={placements!r}. "
                "Ensure the tensor was constructed with a valid layout."
            )

        if self._local_tensor.has_init:
            obj = DTensorBase.__new__(
                type(self),
                initializer(self._local_tensor.init, self._local_tensor.shape, self._local_tensor.dtype),
                device_mesh,
                placements
            )
        else:
            obj = DTensorBase.__new__(
                type(self),
                self._local_tensor.clone(),
                device_mesh,
                placements
            )
        filtered_dict = {k: v for k, v in self.__dict__.items() if k != '_local_tensor'}
        obj.__dict__.update(filtered_dict)
        return obj

    # pylint: disable=W0211
    # pylint: disable=W0102
    # pylint: disable=C0415
    def __fallback__(self, func, args={}, kwargs=None):
        if kwargs is None:
            kwargs = {}
        from hyper_parallel.core.shard._op_dispatch import _OP_DISPATCHER
        with NoFallbackGuard():
            out = _OP_DISPATCHER.dispatch(func, args, kwargs)
        return out

    # pylint: disable=W0212
    def _need_contiguous(self):
        """_need_contiguous"""
        return self._local_tensor._need_contiguous()

    @property
    def device(self):
        """Device info for dtensor"""
        device_info = self._local_tensor.device
        return device_info.split(':', 1)[0]

    # pylint: disable=W0212
    def set_data(self, data):
        """
        Set shape/dtype/storage for dtensor and local tensor.
        """
        if not isinstance(data, Tensor):
            raise ValueError(f"The data type {type(data)} is not Tensor")
        if data.has_init:
            data.init_data()
            data = data.to(self.device)
        if isinstance(data, DTensorBase):
            self._local_tensor._update_data(data.to_local())
            self._device_mesh = data.device_mesh
            self._placements = data.placements
            self._layout = data.layout
            self._update_data(self._local_tensor)
            return

        self._local_tensor._update_data(data)
        self._update_data(data)

    @property
    def has_init(self):
        """
        Property to check if the initialization state is set in the local tensor.

        Returns:
            bool: True if the local tensor has the 'has_init' attribute, False otherwise.
        """
        if not hasattr(self._local_tensor, "has_init"):
            return False
        return self._local_tensor.has_init

    @property
    def init(self):
        """
        Property to get the initialization value from the local tensor.

        Returns:
            Any: The initialization value stored in the local tensor if the 'init' attribute exists;
                 None if the 'init' attribute is not present in the local tensor.
        """
        if not hasattr(self._local_tensor, "init"):
            return None
        return self._local_tensor.init

    @init.setter
    def init(self, init_value):
        """
        Setter for the initialization value, which assigns the value to the local tensor's 'init' attribute.

        Args:
            init_value: The value to be set as the initialization value in the local tensor.
        """
        self._local_tensor.init = init_value

    @property
    def local_param_info(self):
        """
        Property to get the param_info value from the local tensor.

        Returns:
            Any: The param_info value stored in the local tensor if the 'param_info' attribute exists;
                 None if the 'param_info' attribute is not present in the local tensor.
        """
        if not hasattr(self._local_tensor, "param_info"):
            return None
        return self._local_tensor.param_info

    @local_param_info.setter
    def local_param_info(self, local_param_info_value):
        """
        Setter for local_param_info value, which assigns the value to the local tensor's 'param_info' attribute.

        Args:
            local_param_info_value: The value to be set as the param_info value in the local tensor.
        """
        self._local_tensor.param_info = local_param_info_value
