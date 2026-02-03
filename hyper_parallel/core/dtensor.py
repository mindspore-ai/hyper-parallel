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
from typing import Sequence, Tuple
from hyper_parallel.core.layout import Layout, DeviceMesh, _get_slice_tensor_by_layout
from hyper_parallel.core.placement_types import Placement, Replicate
from hyper_parallel.platform import get_platform
from hyper_parallel.core.utils import compute_local_shape_and_global_offset

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


def _build_layout(
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
    tensor_dim: int
) -> Layout:
    """
    Build Layout from device_mesh and placements.

    Args:
        device_mesh: The device mesh describing the device topology.
        placements: Sequence of Placement objects (Shard, Replicate, etc.).
        tensor_dim: Number of dimensions in the tensor.

    Returns:
        Layout: The built layout object.
    """
    layout = Layout.from_device_mesh(device_mesh)
    result = layout(placements)
    result.placement_to_tensor_map(tensor_dim)
    return result


class DTensor(DTensorBase):
    """
    DTensor - Distributed Tensor

    A DTensor represents a tensor that is distributed across multiple devices
    according to a DeviceMesh and placement specifications.

    Args:
        local_tensor (Tensor): The local tensor shard on this device.
        device_mesh (DeviceMesh): The device mesh describing the device topology.
        placements (Sequence[Placement]): The placement strategy for each mesh dimension.
            Each element should be a Placement object (Shard, Replicate, Partial, etc.).

    Example:
        >>> from hyper_parallel.core.placement_types import Shard, Replicate
        >>> mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
        >>> local_tensor = Tensor(np.ones((4, 4)))
        >>> dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0), Replicate()])
    """
    _local_tensor: Tensor
    _device_mesh: DeviceMesh
    _placements: Sequence[Placement]

    def __init_data__(
        self,
        local_tensor: Tensor,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement]
    ):
        self._local_tensor = local_tensor
        self._device_mesh = device_mesh
        self._placements = tuple(placements)
        # Build internal layout for redistribution operations
        self._layout = _build_layout(
            device_mesh, placements, len(local_tensor.shape)
        )

    @property
    def device_mesh(self) -> DeviceMesh:
        """The device mesh of this DTensor."""
        return self._device_mesh

    @property
    def placements(self) -> Sequence[Placement]:
        """The placements of this DTensor."""
        return self._placements

    @property
    def layout(self) -> Layout:
        """Internal layout for redistribution (for backward compatibility)."""
        if not hasattr(self, '_layout'):
            return None
        return self._layout

    @staticmethod
    def from_local(
        local_tensor: Tensor,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement]
    ) -> 'DTensor':
        """
        Create a DTensor from a local tensor with device mesh and placements.

        Args:
            local_tensor (Tensor): The local tensor shard on this device.
            device_mesh (DeviceMesh): The device mesh describing the device topology.
            placements (Sequence[Placement]): The placement strategy. Each element
                should be a Placement object (Shard, Replicate, Partial, etc.).

        Returns:
            DTensor: A new DTensor instance.

        Example:
            >>> from hyper_parallel.core.placement_types import Shard, Replicate
            >>> mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
            >>> local_tensor = Tensor(np.ones((4, 4)))
            >>> dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0), Replicate()])
        """
        return DTensor(local_tensor, device_mesh, placements)

    def to_local(self) -> Tensor:
        """
        Convert DTensor to local tensor.

        Returns:
            Tensor: The local tensor shard on this device.
        """
        return self._local_tensor

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The global shape of this DTensor.

        Returns:
            Tuple[int, ...]: The global tensor shape.
        """
        return self._layout.get_global_shape(self._local_tensor.shape)

    @property
    def local_shape(self) -> Tuple[int, ...]:
        """
        The local shape of this DTensor on this device.

        Returns:
            Tuple[int, ...]: The local tensor shape.
        """
        return self._local_tensor.shape

    def redistribute(
        self,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement]
    ) -> 'DTensor':
        """
        Redistribute this DTensor to a new device mesh and placements.

        Args:
            device_mesh (DeviceMesh): The target device mesh.
            placements (Sequence[Placement]): The target placements. Each element
                should be a Placement object (Shard, Replicate, Partial, etc.).

        Returns:
            DTensor: A new DTensor with the specified distribution.

        Example:
            >>> from hyper_parallel.core.placement_types import Shard, Replicate
            >>> new_dtensor = dtensor.redistribute(mesh, [Replicate(), Shard(1)])
        """
        # Build dst_layout from device_mesh and placements
        dst_layout = _build_layout(
            device_mesh, placements, len(self._local_tensor.shape)
        )

        # pylint: disable=C0415
        from hyper_parallel.core.tensor_redistribution import _tensor_redistribution
        out = _tensor_redistribution.redistribution(self, dst_layout)
        return out

    def reduce_partial(self) -> 'DTensor':
        """
        Reduce partial sharding state for this DTensor.

        Returns:
            DTensor: A new DTensor with partial state reduced.
        """
        if not self._layout:
            return self
        to_layout = cp.deepcopy(self._layout)
        to_layout.reset_partial()
        # pylint: disable=C0415
        from hyper_parallel.core.tensor_redistribution import _tensor_redistribution
        out = _tensor_redistribution.reduce_partial(self, to_layout)
        return out

    def full_tensor(self) -> Tensor:
        """
        Return the full tensor of this DTensor.

        Returns:
            Tensor: A Tensor object that represents the full tensor of this DTensor.
                    The returned tensor contains the complete data gathered from
                    all ranks.

        Note:
            This operation involves communication across all ranks in the DeviceMesh,
            which may be expensive for large tensors. Use with caution in
            performance-critical code paths.

        Example:
            >>> # Assume dtensor is sharded across multiple devices
            >>> local_tensor = dtensor.to_local()  # Returns only the local shard
            >>> full_tensor = dtensor.full_tensor()  # Returns the complete tensor
        """
        if not self._layout:
            return self._local_tensor

        # Create a fully replicated layout
        replicated_layout = cp.deepcopy(self._layout)

        # Set all placements to Replicate and convert to tensor_map
        replicated_placements = [Replicate()] * len(replicated_layout.mesh_shape)
        replicated_layout.set_placements(replicated_placements)
        replicated_layout.placement_to_tensor_map(len(self._local_tensor.shape))

        # Clear partial status from original layout since Replicate has no partial
        replicated_layout.reset_partial()

        # Redistribute to the replicated layout and return local tensor
        # pylint: disable=C0415
        from hyper_parallel.core.tensor_redistribution import _tensor_redistribution
        out = _tensor_redistribution.redistribution(self, replicated_layout)
        return out.to_local()

    @staticmethod
    def distribute_tensor(
        tensor: Tensor,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement]
    ) -> 'DTensor':
        """
        Distribute a global tensor to the device mesh according to the placements.

        Args:
            tensor (Tensor): The global tensor to be distributed. All ranks
                should have the same tensor data.
            device_mesh (DeviceMesh): The device mesh describing the device topology.
            placements (Sequence[Placement]): The placement strategy for distribution.
                Each element should be a Placement object (Shard, Replicate, etc.).

        Returns:
            DTensor: A new DTensor with the local shard on each rank.

        Note:
            This method assumes all ranks have the same global tensor. It slices
            the tensor locally without communication. If ranks have different
            data, use `from_local` instead.

        Example:
            >>> from hyper_parallel.core.placement_types import Shard, Replicate
            >>> mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
            >>> global_tensor = Tensor(np.arange(16).reshape(4, 4))
            >>> dtensor = DTensor.distribute_tensor(global_tensor, mesh, [Shard(0), Replicate()])
            >>> # rank 0 and rank1 gets: [[0,1,2,3], [4,5,6,7]]
            >>> # rank 2 and rank3 gets: [[8,9,10,11], [12,13,14,15]]
        """
        # Build layout from device_mesh and placements
        layout = _build_layout(device_mesh, placements, len(tensor.shape))

        # Slice the global tensor to get local shard based on layout
        local_tensor = _get_slice_tensor_by_layout(tensor, layout)

        return DTensor(local_tensor, device_mesh, placements)


def _dtensor_init_helper(
        init_op,
        size,
        device_mesh,
        placements,
        **kwargs,
) -> DTensor:
    """
        Helper function to create and initialize a distributed tensor.

        Args:
            size: Shape of the tensor.
            dtype: Data type of the tensor.
            device: Target device for the tensor.
            requires_grad: Whether the tensor requires gradient.

        Returns:
            DTensor: The initialized distributed tensor.
    """
    # get local tensor shape
    local_shape = compute_local_shape_and_global_offset(
        size, device_mesh, placements
    )

    # initialize the local tensor
    if init_op is platform.full:
        fill_value = kwargs.pop("fill_value", 0)
        local_tensor = init_op(local_shape, fill_value, **kwargs)
    else:
        local_tensor = init_op(local_shape, **kwargs)

    return DTensor.from_local(
            local_tensor,
            device_mesh,
            placements,
    )

def ones(
    size,
    device_mesh,
    placements,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 1, with the shape defined
    by the variable argument ``size``.

    Args:
        size (Union[tuple[int], list[int], int, Tensor]): The specified shape of output tensor. Only positive integer or
            tuple or Tensor containing positive integers are allowed. If it is a Tensor,
            it must be a 0-D or 1-D Tensor with int32 or int64 dtypes.

    Keyword args:
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    ones_ = platform.ones
    return _dtensor_init_helper(
        ones_,
        size,
        device_mesh=device_mesh,
        placements=placements,
    )

def empty(
    size,
    device_mesh,
    placements,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with uninitialized data. The shape of the :class:`DTensor`
    is defined by the variable argument ``size``.

    Args:
        size (Union[tuple[int], list[int], int]): The specified shape of output tensor. Can be variable numbers of
            positive integers or tuple or list containing positive integers.

    Keyword args:
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    empty_ = platform.empty
    return _dtensor_init_helper(
        empty_,
        size,
        device_mesh=device_mesh,
        placements=placements,
    )


def full(
    size,
    fill_value,
    *,
    device_mesh,
    placements,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with ``fill_value`` according to ``device_mesh`` and
    ``placements``, with the shape defined by the argument ``size``.

    Args:
        size (Union[tuple[int], list[int]]): The specified shape of output tensor.
        fill_value (Union[numbers.Number, Tensor]): Value to fill the returned tensor. It can be a scalar number, a 0-D
            Tensor, or a 1-D Tensor with only one element.

    Keyword args:
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    full_ = platform.full
    return _dtensor_init_helper(
        full_,
        size,
        fill_value=fill_value,
        device_mesh=device_mesh,
        placements=placements,
    )

def zeros(
    size,
    device_mesh,
    placements,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 0.

    Args:
        size (Union[tuple[int], list[int], int, Tensor]): The specified shape of output tensor. Only positive integer or
        tuple or Tensor containing positive integers are allowed. If it is a Tensor,
            it must be a 0-D or 1-D Tensor with int32 or int64 dtypes.
    Keyword args:
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    zeros_ = platform.zeros
    return _dtensor_init_helper(
        zeros_,
        size,
        device_mesh=device_mesh,
        placements=placements,
    )
