# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""framework platform api"""
# Backend platform modules intentionally import this abstraction to register
# their implementations; the resulting import cycle is architectural.
# pylint: disable=cyclic-import
import os
from datetime import timedelta
from enum import auto, Enum
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

# Environment variable name used to specify the AI framework platform to use
HYPER_PARALLEL_PLATFORM = "HYPER_PARALLEL_PLATFORM"

# Identifier for the MindSpore framework
HYPER_PARALLEL_PLATFORM_MINDSPORE = "mindspore"

# Identifier for the PyTorch framework
HYPER_PARALLEL_PLATFORM_TORCH = "torch"


class AsyncHandle:
    """Idempotent wait handle for an async collective operation.

    Wraps the async tensor returned by
    :meth:`Platform.differentiable_all_to_all_single_async` and provides a
    :meth:`wait` method that is safe to call multiple times.
    """

    def __init__(self, async_tensor) -> None:
        self._tensor = async_tensor
        self._waited = False

    def wait(self):
        """Wait for the async collective to complete.

        Idempotent — the first call blocks until the collective finishes;
        subsequent calls are no-ops.

        Returns:
            The now-materialised result tensor.
        """
        if not self._waited:
            get_platform().wait_async_tensor(self._tensor)
            self._waited = True
        return self._tensor


class PlatformType(Enum):
    """Enumeration class for AI framework platform types.

    Used to identify different deep learning framework platform types.
    """
    MINDSPORE = auto()
    PYTORCH = auto()


# Global platform instance, used to cache the created platform object
platform = None


def get_mindspore_platform():
    """Create and return a MindSpore platform instance.

    Returns:
        MindSporePlatform: A MindSpore platform instance.
    """
    # pylint: disable=C0415
    from hyper_parallel.platform.mindspore.platform import MindSporePlatform
    global platform
    platform = MindSporePlatform()
    return platform


def get_torch_platform():
    """Create and return a PyTorch platform instance.

    Returns:
        TorchPlatform: A PyTorch platform instance.
    """
    # pylint: disable=C0415
    from hyper_parallel.platform.torch.platform import TorchPlatform
    global platform
    platform = TorchPlatform()
    return platform


def get_platform():
    """Obtain a framework platform instance.

    Returns the appropriate AI framework platform instance based on environment variables or a default priority order.
    The lookup priority is as follows:
    1. Platform specified by environment variable
    2. MindSpore platform (default preferred choice)
    3. PyTorch platform (fallback option)

    Returns:
        Platform: An instance of the framework platform

    Raises:
        ImportError: Raised when none of the supported frameworks are available
    """
    if platform is not None:
        return platform
    platform_type = os.environ.get(HYPER_PARALLEL_PLATFORM)
    if platform_type is not None and isinstance(platform_type, str):
        platform_type = platform_type.lower()
        if platform_type == HYPER_PARALLEL_PLATFORM_MINDSPORE:
            return get_mindspore_platform()
        if platform_type == HYPER_PARALLEL_PLATFORM_TORCH:
            return get_torch_platform()
    try:
        return get_mindspore_platform()
    except ImportError:
        return get_torch_platform()


EXISTING_COMM_GROUPS = {}


def _build_p2p_edge_rank_lists(pp_rank_list: list[int], include_wrap: bool = False) -> list[tuple[int, int]]:
    """Build normalized two-rank groups for adjacent pipeline ranks."""
    if not isinstance(pp_rank_list, (list, tuple)):
        raise ValueError(
            f"pp_rank_list must be a list or tuple of integer ranks, but got {type(pp_rank_list)}."
        )
    if any(not isinstance(rank, int) or isinstance(rank, bool) for rank in pp_rank_list):
        raise ValueError(f"pp_rank_list must contain only integer ranks, but got {pp_rank_list}.")
    if len(set(pp_rank_list)) != len(pp_rank_list):
        raise ValueError(f"pp_rank_list must not contain duplicate ranks, but got {pp_rank_list}.")
    if len(pp_rank_list) < 2:
        return []

    edge_rank_lists = {
        tuple(sorted((src_rank, dst_rank)))
        for src_rank, dst_rank in zip(pp_rank_list, pp_rank_list[1:])
    }
    if include_wrap and len(pp_rank_list) > 2:
        edge_rank_lists.add(tuple(sorted((pp_rank_list[-1], pp_rank_list[0]))))
    return sorted(edge_rank_lists)


class Platform:
    """Platform api"""
    current_grad_handle = None
    post_grad_handle_process = None
    grad_sync_stream = None

    @property
    def custom_ops(self):
        """Return the platform-specific custom ops interface.

        Subclasses MUST override this property to return an object that
        exposes the platform-specific custom operator implementations.

        Returns:
            object: Platform-specific custom ops class instance.
        """
        raise NotImplementedError(
            "Platform subclasses must implement custom_ops"
        )

    @staticmethod
    def get_swap_optimizer():
        """Return the active backend's optimizer-state swap wrapper class."""
        raise NotImplementedError("Platform subclasses must implement get_swap_optimizer")

    @staticmethod
    def get_rank():
        """Get the rank of the current process in the default process group.

        Returns:
            int: The rank of the current process.
        """
        raise NotImplementedError("Platform subclasses must implement get_rank")

    @staticmethod
    def get_global_rank(group, group_rank):
        """Convert a group rank to its global rank.

        Args:
            group: The process group to query.
            group_rank (int): The rank within the group.

        Returns:
            int: The global rank corresponding to the group rank.
        """
        raise NotImplementedError("Platform subclasses must implement get_global_rank")

    @staticmethod
    def get_group_rank(group):
        """Return this process's rank within *group*."""
        raise NotImplementedError("Platform subclasses must implement get_group_rank")

    @staticmethod
    def get_world_size():
        """Get the total number of processes in the default process group.

        Returns:
            int: The world size (total number of processes).
        """
        raise NotImplementedError("Platform subclasses must implement get_world_size")

    @staticmethod
    def get_op_name(func):
        """Get the canonical name of an operator function.

        Args:
            func: The operator function to query.

        Returns:
            str: The canonical name of the operator.
        """
        raise NotImplementedError("Platform subclasses must implement get_op_name")

    @staticmethod
    def differentiable_all_gather_concat(data, group, concat_size, concat_dim, rank_list=None):
        """Perform differentiable all-gather and concatenate tensors along a dimension.

        Args:
            data: The input tensor to gather.
            group: The process group for collective communication.
            concat_size (int): The size to concatenate along concat_dim.
            concat_dim (int): The dimension along which to concatenate.
            rank_list: Optional rank order expected by the logical layout.

        Returns:
            The concatenated tensor after all-gather operation.
        """
        raise NotImplementedError("Platform subclasses must implement differentiable_all_gather_concat")

    @staticmethod
    def chunk(data, split_dim, split_size, index):
        """Split tensor along a dimension and return the chunk at the given index.

        Args:
            data: The input tensor to split.
            split_dim (int): The dimension along which to split.
            split_size (int): The size of each split chunk.
            index (int): The index of the chunk to return.

        Returns:
            The tensor chunk at the specified index.
        """
        raise NotImplementedError("Platform subclasses must implement chunk")

    @staticmethod
    def differentiable_all_to_all(input_data, output_shape, group):
        """Perform differentiable all-to-all communication.

        Args:
            input_data: The input tensor to redistribute.
            output_shape: The shape of the output tensor.
            group: The process group for collective communication.

        Returns:
            The output tensor after all-to-all operation.
        """
        raise NotImplementedError("Platform subclasses must implement differentiable_all_to_all")

    @staticmethod
    def tensor_type_cast(input_data, cast_type):
        """Cast tensor to a specified dtype.

        Args:
            input_data: The input tensor to cast.
            cast_type: The target dtype to cast to.

        Returns:
            The tensor cast to the specified dtype.
        """
        raise NotImplementedError("Platform subclasses must implement tensor_type_cast")

    @staticmethod
    def is_tensor(obj: Any) -> bool:
        """Return True if ``obj`` is this framework's tensor type."""
        raise NotImplementedError("Platform subclasses must implement is_tensor")

    @staticmethod
    def get_tensor_storage_size(tensor: Any) -> int:
        """Return serialized byte size (numel * element size) for this framework's tensor."""
        raise NotImplementedError("Platform subclasses must implement get_tensor_storage_size")

    @staticmethod
    def differentiable_all_reduce(data, op, group):
        """Perform differentiable all-reduce operation.

        Args:
            data: The input tensor to reduce.
            op: The reduction operation (e.g., sum, max, min).
            group: The process group for collective communication.

        Returns:
            The reduced tensor with gradients supported.
        """
        raise NotImplementedError("Platform subclasses must implement differentiable_all_reduce")

    @staticmethod
    def differentiable_reduce_scatter(data, dev_num, axis, op, group):
        """Perform differentiable reduce-scatter operation.

        Args:
            data: The input tensor to reduce and scatter.
            dev_num (int): The number of devices to scatter across.
            axis (int): The axis along which to scatter.
            op: The reduction operation (e.g., sum, max, min).
            group: The process group for collective communication.

        Returns:
            The scattered tensor chunk with gradients supported.
        """
        raise NotImplementedError("Platform subclasses must implement differentiable_reduce_scatter")

    @staticmethod
    def init_parameters(module, stage_index):
        """Initialize parameters for a module at a specific pipeline stage.

        This method is primarily needed for MindSpore platform which requires
        explicit parameter initialization interface.

        Args:
            module: The module whose parameters need to be initialized.
            stage_index (int): The pipeline stage index for the module.

        Raises:
            ValueError: If module is None or stage_index is negative.
        """
        if module is None:
            raise ValueError("input module must not be none.")
        if stage_index < 0:
            raise ValueError("input stage_index must be positive.")

    @staticmethod
    def get_cell_construct(cell):
        """Get the construct (forward) function of a cell/module.

        Args:
            cell: The cell or module to get the construct function from.

        Returns:
            The construct/forward callable of the cell.
        """
        raise NotImplementedError("Platform subclasses must implement get_cell_construct")

    @staticmethod
    def get_cells_and_names(cell):
        """Get all nested cells/modules and their names.

        Args:
            cell: The root cell or module to traverse.

        Returns:
            list: A list of tuples containing (name, cell) pairs.
        """
        raise NotImplementedError("Platform subclasses must implement get_cells_and_names")

    @staticmethod
    def get_modules(module):
        """Return all sub-modules contained in the given module."""
        raise NotImplementedError("Platform subclasses must implement get_modules")

    @staticmethod
    def search_parameter_by_name(cell, param_name: str):
        """Search for a parameter by name within a cell/module.

        Args:
            cell: The cell or module to search in.
            param_name (str): The name of the parameter to find.

        Returns:
            The parameter if found, otherwise None.
        """
        raise NotImplementedError("Platform subclasses must implement search_parameter_by_name")

    @staticmethod
    def update_parameter_by_name(cell, result: tuple, new_param) -> bool:
        """Update a parameter by name within a cell/module.

        Args:
            cell: The cell or module containing the parameter.
            result (tuple): A tuple containing (param_name, parameter) to update.
            new_param: The new parameter value to set.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        raise NotImplementedError("Platform subclasses must implement update_parameter_by_name")

    @staticmethod
    def set_layout_into_parameter(param, layout):
        """Attach a DTensor layout to a parameter.

        Args:
            param: The parameter to attach the layout to.
            layout: The DTensor layout describing tensor distribution.
        """
        raise NotImplementedError("Platform subclasses must implement set_layout_into_parameter")

    @staticmethod
    def get_param_local_shape(param):
        """Get the local shape of a distributed parameter.

        Args:
            param: The parameter to query.

        Returns:
            tuple: The local shape of the parameter shard.
        """
        raise NotImplementedError("Platform subclasses must implement get_param_local_shape")

    @staticmethod
    def get_param_local_data(param):
        """Get the local data tensor of a distributed parameter.

        Args:
            param: The parameter to query.

        Returns:
            The local tensor data of the parameter shard.
        """
        raise NotImplementedError("Platform subclasses must implement get_param_local_data")

    @staticmethod
    def update_param_data(param, data):
        """Update the data of a parameter with new tensor data.

        Args:
            param: The parameter to update.
            data: The new tensor data to assign.
        """
        raise NotImplementedError("Platform subclasses must implement update_param_data")

    @staticmethod
    def get_param_type_size(param):
        """Get the size in bytes of a parameter's dtype.

        Args:
            param: The parameter to query.

        Returns:
            int: The size in bytes of the parameter's data type.
        """
        raise NotImplementedError("Platform subclasses must implement get_param_type_size")

    @staticmethod
    def new_zero_parameter(param_shape, param_type, requires_grad, device):
        """Create a new parameter initialized with zeros.

        Args:
            param_shape (tuple): The shape of the parameter.
            param_type: The dtype of the parameter.
            requires_grad (bool): Whether the parameter requires gradients.
            device: The device on which to create the parameter.

        Returns:
            A new parameter tensor filled with zeros.
        """
        raise NotImplementedError("Platform subclasses must implement new_zero_parameter")

    @staticmethod
    def new_tensor(tensor_shape, tensor_type, device):
        """Create a new tensor with the specified shape, dtype, and device.

        Args:
            tensor_shape (tuple): The shape of the tensor.
            tensor_type: The dtype of the tensor.
            device: The device on which to create the tensor.

        Returns:
            A new tensor with uninitialized values.
        """
        raise NotImplementedError("Platform subclasses must implement new_tensor")

    @staticmethod
    def full_like(tensor, fill_value, dtype=None):
        """Create a tensor filled with a value, with same shape as input.

        Args:
            tensor: The input tensor to copy shape from.
            fill_value: The value to fill the new tensor with.
            dtype: Optional dtype for the new tensor. If None, uses input tensor's dtype.

        Returns:
            A new tensor filled with the specified value.
        """
        raise NotImplementedError("Platform subclasses must implement full_like")

    @staticmethod
    def set_tensor_requires_grad(input_tensor):
        """Enable gradient tracking for a tensor in-place.

        Args:
            input_tensor: The tensor to enable gradients for.

        Returns:
            The same tensor with requires_grad set to True.
        """
        raise NotImplementedError("Platform subclasses must implement set_tensor_requires_grad")

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        """Gather tensors from all ranks into a single output tensor.

        Args:
            data: The input tensor to gather.
            group_info: The process group for collective communication.
            async_op (bool): If True, returns a work handle for async operation.

        Returns:
            The gathered tensor, or a tuple of (tensor, handle) if async_op is True.
        """
        raise NotImplementedError("Platform subclasses must implement all_gather_into_tensor")

    @staticmethod
    def all_reduce(data, group_info, async_op=False):
        """Reduce tensors across all ranks using specified operation.

        Args:
            data: The input tensor to reduce.
            group_info: The process group for collective communication.
            async_op (bool): If True, returns a work handle for async operation.

        Returns:
            The reduced tensor, or a tuple of (tensor, handle) if async_op is True.
        """
        raise NotImplementedError("Platform subclasses must implement all_reduce")

    @staticmethod
    def broadcast(data, src=None, group=None, async_op=False, group_src=None):
        """Broadcast tensor from source rank to all ranks in group."""
        raise NotImplementedError("Platform subclasses must implement broadcast")

    @staticmethod
    def scatter(output, scatter_list, src=None, group=None, async_op=False, group_src=None):
        """Scatter tensor list from source rank to all ranks in group."""
        raise NotImplementedError("Platform subclasses must implement scatter")

    @staticmethod
    def isend(tensor, dst=None, group=None, tag=0):
        """Send tensor asynchronously to destination rank.

        Args:
            tensor: The tensor to send.
            dst (int, optional): The destination rank. Defaults to None.
            group: The process group for communication. Defaults to None.
            tag (int): A tag to identify the send operation. Defaults to 0.

        Returns:
            A work handle that can be waited on.
        """
        raise NotImplementedError("Platform subclasses must implement isend")

    @staticmethod
    def irecv(tensor, src=None, group=None, tag=0):
        """Receive tensor asynchronously from source rank.

        Args:
            tensor: The tensor buffer to receive data into.
            src (int, optional): The source rank. Defaults to None.
            group: The process group for communication. Defaults to None.
            tag (int): A tag to identify the receive operation. Defaults to 0.

        Returns:
            A work handle that can be waited on.
        """
        raise NotImplementedError("Platform subclasses must implement irecv")

    @staticmethod
    def p2p_op(op_type, tensor, peer, group=None):
        """Build a batched-P2P descriptor (no launch).

        Returns an opaque object understood by :meth:`batch_isend_irecv`.
        Lets callers assemble a mixed send/recv batch that the backend can
        run concurrently (e.g. TX/RX duplex on one link) in a single op.

        Args:
            op_type (str): ``"isend"`` or ``"irecv"``.
            tensor: Tensor to send, or the buffer to receive into.
            peer (int): Global rank of the peer.
            group: Process group. ``None`` uses the default group.

        Returns:
            A backend P2P-op descriptor.
        """
        raise NotImplementedError("Platform subclasses must implement p2p_op")

    @staticmethod
    def batch_isend_irecv(p2p_ops):
        """Launch a batch of :meth:`p2p_op` descriptors as one async op.

        The whole batch shares a single completion handle (the backend runs
        the items concurrently on one comm stream), so a send and a recv to
        the same peer overlap on the duplex link.

        Args:
            p2p_ops (list): Descriptors from :meth:`p2p_op`.

        Returns:
            A single work handle covering the whole batch, or ``None`` when
            ``p2p_ops`` is empty.
        """
        raise NotImplementedError("Platform subclasses must implement batch_isend_irecv")

    @staticmethod
    def prepare_batch_p2p_group(group: Any = None) -> None:
        """Prepare a process group before its first batched P2P operation.

        Backends that require full-group participation before subset batched
        P2P should synchronize the group here. Other backends may implement
        this as a no-op.

        Args:
            group: The process group used by the batched P2P operations.
                ``None`` uses the default group.
        """
        raise NotImplementedError("Platform subclasses must implement prepare_batch_p2p_group")

    @staticmethod
    def p2p_exchange(tensor, peer_rank: int, group=None):
        """Differentiable symmetric P2P exchange (send local tensor, receive peer's tensor).

        Sends ``tensor`` to ``peer_rank`` and simultaneously receives the peer's
        tensor.  The operation is differentiable: the backward pass performs the
        same symmetric exchange on the upstream gradient.

        Args:
            tensor: Local tensor to send.
            peer_rank (int): Global rank of the communication peer.
            group: Process group. ``None`` uses the default group.

        Returns:
            Tensor received from ``peer_rank``, with the same shape and dtype as
            the input ``tensor``.
        """
        raise NotImplementedError("Platform subclasses must implement p2p_exchange")

    @staticmethod
    def send_object_list(obj_list, dst=None, group=None):
        """Send a list of Python objects to destination rank.

        Args:
            obj_list (list): The list of Python objects to send.
            dst (int, optional): The destination rank. Defaults to None.
            group: The process group for communication. Defaults to None.
        """
        raise NotImplementedError("Platform subclasses must implement send_object_list")

    @staticmethod
    def recv_object_list(obj_list, src=None, group=None):
        """Receive a list of Python objects from source rank.

        Args:
            obj_list (list): The list buffer to receive objects into.
            src (int, optional): The source rank. Defaults to None.
            group: The process group for communication. Defaults to None.
        """
        raise NotImplementedError("Platform subclasses must implement recv_object_list")

    @staticmethod
    def reduce_scatter_tensor(data, group_info, async_op=False):
        """Reduce and scatter tensor across all ranks in group.

        Args:
            data: The input tensor to reduce and scatter.
            group_info: The process group for collective communication.
            async_op (bool): If True, returns a work handle for async operation.

        Returns:
            The scattered tensor chunk, or a tuple of (tensor, handle) if async_op is True.
        """
        raise NotImplementedError("Platform subclasses must implement reduce_scatter_tensor")

    @staticmethod
    def all_gather_single(input_tensor, output_shape, group, async_op=False):
        """All-gather tensor shards with optional async execution.

        Args:
            input_tensor: Input tensor whose leading dimension is gathered.
            output_shape: Shape of the gathered output tensor.
            group: Process group (ProcessGroup for torch, group name string for mindspore).
            async_op: If True, returns an async work handle.

        Returns:
            Tuple ``(output, work)`` where *output* is the gathered tensor and
            *work* is the async handle (``None`` when ``async_op=False``).
        """
        raise NotImplementedError("Platform subclasses must implement all_gather_single")

    @staticmethod
    def reduce_scatter_single(input_tensor, output_shape, group, async_op=False):
        """Reduce-scatter a tensor with optional async execution.

        Args:
            input_tensor: Input tensor whose leading dimension is split across ranks.
            output_shape: Shape of the local reduced output tensor.
            group: Process group (ProcessGroup for torch, group name string for mindspore).
            async_op: If True, returns an async work handle.

        Returns:
            Tuple ``(output, work)`` where *output* is the local shard and
            *work* is the async handle (``None`` when ``async_op=False``).
        """
        raise NotImplementedError("Platform subclasses must implement reduce_scatter_single")

    @staticmethod
    def all_to_all_single(input_tensor, output_shape, group, async_op=False):
        """All-to-all single collective with optional async execution.

        Args:
            input_tensor: Input tensor to scatter.
            output_shape: Shape of the pre-allocated output tensor.
            group: Process group (ProcessGroup for torch, group name string for mindspore).
            async_op: If True, returns a work handle; the output tensor is
                      filled only after ``work.wait()`` is called.

        Returns:
            Tuple ``(output, work)`` where *output* is the result tensor and
            *work* is the async handle (``None`` when ``async_op=False``).

        Raises:
            NotImplementedError: Must be implemented by platform subclasses.
        """
        raise NotImplementedError("Platform subclasses must implement all_to_all_single")

    @staticmethod
    def differentiable_variable_all_gather(
            input_tensor: Any, output_splits: Sequence[int], group: Any) -> Any:
        """Gather variable dim-zero shards on every rank with gradient support.

        Args:
            input_tensor: Local input shaped ``[local_rows, *feature_dims]``.
            output_splits: Dim-zero rows contributed by each group rank.
            group: Raw platform process group.

        Returns:
            Tensor concatenated in group-rank order along dim zero.

        Raises:
            NotImplementedError: Must be implemented by platform subclasses.
        """
        raise NotImplementedError(
            "Platform subclasses must implement differentiable_variable_all_gather"
        )

    @staticmethod
    def differentiable_async_allgather_wait(x, work, out_perm, group, world_size, gather_dim,
                                            handle_box=None):
        """Differentiable wrapper that waits for a pre-launched async all-gather.

        Forward waits for the all-gather handle and reconstructs the tensor by
        moving the gathered leading dimension back to ``gather_dim``.

        Backward launches the reverse reduce-scatter. If ``handle_box`` is a
        mutable list, the reduce-scatter handle is appended there and a zero
        gradient is returned to be replaced by the caller's backward pre-hook.
        If ``handle_box`` is ``None``, the reduce-scatter is waited immediately
        and its local result is returned, preserving composability with an
        upstream autograd communication op.

        Args:
            x: Original input tensor; anchors the op in the autograd graph.
            work: Async work handle from all-gather.
            out_perm: Output buffer filled by all-gather.
            group: Communication group for backward reduce-scatter.
            world_size: Group size.
            gather_dim: Dimension gathered in forward.
            handle_box: Optional mutable list for deferred backward wait.

        Returns:
            Gathered tensor connected to the autograd graph through *x*.
        """
        raise NotImplementedError("Platform subclasses must implement differentiable_async_allgather_wait")

    @staticmethod
    def differentiable_async_a2a_wait(x, work, out_perm, group, world_size, concat_dim, split_dim,
                                      handle_box=None):
        """Differentiable wrapper that waits for a pre-launched async A2A.

        Wraps the wait-and-reconstruct step in the platform autograd mechanism
        so gradients flow correctly through the all-to-all communication.

        The A2A direction is seq→head (forward): the output gathers along
        ``concat_dim`` (sequence grows from S/cp to S) and scatters along
        ``split_dim`` (heads shrink from H to H/ws).

        In backward, launches an async head→seq A2A on the incoming gradient
        and appends ``(work, out_perm)`` to ``handle_box`` so the caller can
        wait just before the projection GEMM, achieving GEMM–A2A overlap.

        Args:
            x:          Original projection output tensor; anchors the op
                        in the autograd graph.
            work:       Async work handle from ``all_to_all_single(async_op=True)``.
            out_perm:   Output buffer filled once ``work.wait()`` completes
                        (shape ``[ws, ...]``).
            group:      Process group for the reverse A2A in backward.
            world_size: CP/Ulysses degree.
            concat_dim: Dimension that is gathered (concatenated) in forward;
                        typically the sequence dimension.
            split_dim:  Dimension that is scattered (split) in forward;
                        typically the head dimension.
            handle_box: Optional mutable list ``[]``. In backward, ``(work, out_perm)``
                        for the reverse A2A is appended here so the pre-hook can wait.

        Returns:
            Result tensor with ``concat_dim`` gathered and ``split_dim`` split,
            connected to the autograd graph through *x*.

        Raises:
            NotImplementedError: Must be implemented by platform subclasses.
        """
        raise NotImplementedError("Platform subclasses must implement differentiable_async_a2a_wait")

    @staticmethod
    def differentiable_sync_hook(x, hook_name: str, coordinator):
        """Identity operation that intercepts both forward and backward to call
        coordinator rendezvous, enabling deterministic comm/compute overlap.

        This is the differentiable building block for dual-pipe schedules.
        In the forward pass the coordinator is invoked with the forward-side
        roles for ``hook_name``; in the backward pass it is invoked with the
        backward-side roles.  The tensor value and gradient flow through
        unchanged.

        Args:
            x:           Input tensor.  Returned as-is; gradients flow through.
            hook_name:   One of ``"A"``, ``"B"``, ``"C"``, ``"D"`` identifying
                         the position relative to MoE dispatch/combine.
            coordinator: A :class:`HookCoordinator` instance shared between the
                         forward and backward threads.

        Returns:
            The same tensor *x*, attached to the autograd graph so that the
            backward hook will fire.
        """
        raise NotImplementedError("Platform subclasses must implement differentiable_sync_hook")

    @staticmethod
    def differentiable_all_to_all_single(input_tensor, input_splits, output_splits, group):
        """Variable-split all-to-all single that supports gradient flow.

        Unlike ``all_to_all_single`` (which is not differentiable), this method
        wraps the collective in an autograd function so gradients are correctly
        routed back through the reverse all-to-all in the backward pass.
        Intended for Expert Parallelism token dispatch / combine.

        Args:
            input_tensor: Input tensor to scatter. Shape ``[sum(input_splits), *feature_dims]``.
            input_splits: Per-rank sizes of data sent from this rank (list of ints,
                          length equal to ep_degree).
            output_splits: Per-rank sizes of data received by this rank (list of ints,
                           length equal to ep_degree).
            group: Process group (ProcessGroup for torch, group name str for mindspore).

        Returns:
            Output tensor of shape ``[sum(output_splits), *feature_dims]``.

        Raises:
            NotImplementedError: Must be implemented by platform subclasses.
        """
        raise NotImplementedError("Platform subclasses must implement differentiable_all_to_all_single")

    @staticmethod
    def differentiable_all_to_all_single_async(input_tensor, input_splits, output_splits, group):
        """Async variant of :meth:`differentiable_all_to_all_single`.

        Same semantics but launches the collective with ``async_op=True`` and
        only performs a stream-level ``wait`` — the host returns immediately
        after dispatching the kernel.  Intended for dual-pipe comm/compute
        overlap paths where the paired COMPUTE side's rendezvous notify must
        fire right after kernel launch (not after the collective actually
        completes on device).

        Args:
            input_tensor: Input tensor to scatter. Shape ``[sum(input_splits), *feature_dims]``.
            input_splits: Per-rank sizes of data sent from this rank.
            output_splits: Per-rank sizes of data received by this rank.
            group: Process group.

        Returns:
            Output tensor of shape ``[sum(output_splits), *feature_dims]``.

        Raises:
            NotImplementedError: Must be implemented by platform subclasses.
        """
        raise NotImplementedError(
            "Platform subclasses must implement differentiable_all_to_all_single_async"
        )

    @staticmethod
    def wait_async_tensor(tensor):
        """Wait for an async collective tensor to become materialised.

        Intended for use with :class:`AsyncHandle` so that callers can
        wait on an async all-to-all result without importing framework-specific
        modules directly.  The call is **idempotent** — waiting on an already-
        completed tensor is a no-op.

        Args:
            tensor: An async collective tensor (e.g. PyTorch
                ``AsyncCollectiveTensor``) whose values have not yet been
                fully written by the remote ranks.

        Returns:
            The same *tensor*, now guaranteed to be fully materialised.

        Raises:
            NotImplementedError: Must be implemented by platform subclasses.
        """
        raise NotImplementedError(
            "Platform subclasses must implement wait_async_tensor"
        )

    @staticmethod
    def arange(start, end=None, step=1, dtype=None, device=None):
        """Create a 1-D tensor with evenly spaced values.

        Args:
            start: Start of interval (inclusive).  If *end* is ``None``,
                treated as the stop value and *start* defaults to 0.
            end: End of interval (exclusive).  Defaults to ``None``.
            step: Step size.  Defaults to ``1``.
            dtype: Data type.  ``None`` uses the framework default (int64).
            device: Target device.

        Returns:
            1-D tensor ``[start, start+step, ..., end)``.

        Raises:
            NotImplementedError: Must be implemented by platform subclasses.
        """
        raise NotImplementedError("Platform subclasses must implement arange")

    @staticmethod
    def zeros(size, dtype=None, device=None):
        """Create a zero-filled tensor of the given shape.

        Args:
            size: Shape of the tensor (a single tuple/list).
            dtype: Desired data type.  ``None`` uses the framework default (float32).
            device: Target device.  ``None`` uses the framework default.

        Returns:
            Zero-filled tensor of the specified shape.

        Raises:
            NotImplementedError: Must be implemented by platform subclasses.
        """
        raise NotImplementedError("Platform subclasses must implement zeros")

    @staticmethod
    def parameters_dict(cell):
        """Get the parameters dictionary of a cell/module.

        Args:
            cell: The cell or module to get parameters from.

        Returns:
            dict: A dictionary mapping parameter names to parameters.
        """
        raise NotImplementedError("Platform subclasses must implement parameters_dict")

    @staticmethod
    def buffers_dict(cell: Any) -> Any:
        """Get the named buffers of a cell/module.

        Args:
            cell: The cell or module to get buffers from.

        Returns:
            An iterable of ``(name, buffer)`` pairs, including non-persistent
            buffers and buffers registered by child modules.
        """
        raise NotImplementedError("Platform subclasses must implement buffers_dict")

    @staticmethod
    def get_model_state_dict(model: Any, *, options: Any = None) -> dict[str, Any]:
        """Get the state dictionary of a model.

        Args:
            model: The model to extract state from.
            options: Optional configuration for state dict extraction.

        Returns:
            dict: The state dictionary containing model parameters and buffers.

        Raises:
            NotImplementedError: Platform subclasses must implement this method.
        """
        raise NotImplementedError(
            "Platform subclasses must implement get_model_state_dict"
        )

    @staticmethod
    def set_model_state_dict(model: Any, model_state_dict: dict[str, Any], *, options: Any = None) -> None:
        """Set the state dictionary of a model.

        Args:
            model: The model to load state into.
            model_state_dict: The state dict to load into the model.
            options: Optional configuration for state dict loading.

        Returns:
            None.

        Raises:
            NotImplementedError: Platform subclasses must implement this method.
        """
        raise NotImplementedError(
            "Platform subclasses must implement set_model_state_dict"
        )

    @staticmethod
    def save_checkpoint(cell, file_path: str, ckpt_format: str = "safetensors") -> None:
        """Save a cell/module checkpoint to file.

        Args:
            cell: The cell or module to save.
            file_path (str): The path to save the checkpoint to.
            ckpt_format (str): The file format.
        """
        raise NotImplementedError("Platform subclasses must implement save_checkpoint")

    @staticmethod
    def load_checkpoint(file_path: str, ckpt_format: str = "safetensors") -> dict:
        """Load a checkpoint from file.

        Args:
            file_path (str): The path to load the checkpoint from.
            ckpt_format (str): The file format.

        Returns:
            dict: The loaded checkpoint state dictionary.
        """
        raise NotImplementedError("Platform subclasses must implement load_checkpoint")

    def _create_group(self, rank_list):
        """Create a new process group with the specified ranks.

        Internal method to be implemented by subclasses.

        Args:
            rank_list (list): List of ranks to include in the group.

        Returns:
            The newly created process group.
        """
        raise NotImplementedError("Platform subclasses must implement _create_group")

    def new_stream(self):
        """Create a new compute stream for asynchronous operations.

        Returns:
            A new stream object for the current device.
        """
        raise NotImplementedError("Platform subclasses must implement new_stream")

    def get_stream_context(self):
        """Get a context manager for executing operations on a specific stream.

        Returns:
            A context manager that can be used with 'with' statement to set stream.
        """
        raise NotImplementedError("Platform subclasses must implement get_stream_context")

    @staticmethod
    def get_tensor_transform():
        """Get the tensor transformation utilities for the current framework.

        Returns:
            A module or object containing tensor transformation functions.
        """
        raise NotImplementedError("Platform subclasses must implement get_tensor_transform")

    @staticmethod
    def construct_strided_slice(x, begin, end, stride):
        """Construct a strided slice operation on a tensor.

        Args:
            x: The input tensor to slice.
            begin: The starting indices for each dimension.
            end: The ending indices for each dimension.
            stride: The stride for each dimension.

        Returns:
            The sliced tensor.
        """
        raise NotImplementedError("Platform subclasses must implement construct_strided_slice")

    @staticmethod
    def micro_batch(micro_batch_num, args_batch_dim=None, kwargs_batch_dim=None):
        """Split inputs into micro-batches for pipeline parallelism.

        Args:
            micro_batch_num (int): The number of micro-batches to create.
            args_batch_dim (list, optional): Batch dimension for each positional arg.
            kwargs_batch_dim (dict, optional): Batch dimension for each keyword arg.

        Returns:
            A decorator that splits function inputs into micro-batches.
        """
        raise NotImplementedError("Platform subclasses must implement micro_batch")

    @staticmethod
    def get_symmetric_memory_handler():
        """Return a platform-specific symmetric memory handler instance."""
        raise NotImplementedError("Platform subclasses must implement get_symmetric_memory_handler")

    @staticmethod
    def load_into_param(param, data):
        """Load data into a parameter, handling framework-specific semantics."""
        raise NotImplementedError("Platform subclasses must implement load_into_param")

    def create_group(self, rank_list):
        """Create or retrieve a communication group with the specified ranks.

        If a group with the same rank list already exists, returns the existing
        group instead of creating a new one.

        Args:
            rank_list (list): List of ranks to include in the group.

        Returns:
            The process group for the specified ranks.
        """
        group_key = str(tuple(sorted(rank_list)))
        if group_key in EXISTING_COMM_GROUPS:
            return EXISTING_COMM_GROUPS[group_key]

        group = self._create_group(rank_list)
        EXISTING_COMM_GROUPS[group_key] = group
        return group

    @staticmethod
    def create_p2p_multi_stream_groups(
            pp_rank_list: list[int],
            include_wrap: bool = False,
    ) -> dict[int, Any]:
        """Create P2P groups that enable independent communication streams.

        Backends may use different process-group initialization protocols, but
        must return the same logical mapping from peer global rank to the raw
        process group shared by that two-rank pipeline edge.

        Args:
            pp_rank_list: Ordered global ranks in one pipeline-parallel group.
            include_wrap: Whether the last and first ranks also communicate,
                as required by interleaved virtual pipeline chunks.

        Returns:
            A mapping from adjacent peer global rank to its two-rank process
            group. A rank at a linear pipeline boundary has one entry; a
            middle rank normally has two.
        """
        raise NotImplementedError("Platform subclasses must implement create_p2p_multi_stream_groups")

    @staticmethod
    def _process_current_handle():
        """Wait for the current gradient handle and execute post-process callback.

        Internal method to synchronize pending gradient operations.
        """
        if Platform.current_grad_handle is None:
            return

        Platform.current_grad_handle.wait()
        if Platform.post_grad_handle_process is None:
            return
        # pylint: disable=E1102
        Platform.post_grad_handle_process()

    def set_grad_reduce_handle(self, handle, post_process=None):
        """Set a new gradient reduction handle after waiting for the current one.

        Waits for any pending gradient handle on the grad sync stream, then
        sets the new handle and optional post-process callback.

        Args:
            handle: The async work handle for gradient reduction.
            post_process (callable, optional): Callback to run after handle completes.
        """
        if Platform.grad_sync_stream is None:
            Platform.grad_sync_stream = self.new_stream()
        stream_context = self.get_stream_context()
        with stream_context(Platform.grad_sync_stream):
            Platform._process_current_handle()
        Platform.current_grad_handle = handle
        Platform.post_grad_handle_process = post_process

    def wait_grad_handle(self):
        """Wait for the current gradient handle to complete.

        Blocks until the current gradient reduction handle completes and
        clears the handle state.
        """
        if Platform.current_grad_handle is None:
            return
        if Platform.grad_sync_stream is None:
            Platform.grad_sync_stream = self.new_stream()
        stream_context = self.get_stream_context()
        with stream_context(Platform.grad_sync_stream):
            Platform._process_current_handle()
            sync_event = Platform.grad_sync_stream.record_event()
        sync_event.wait()
        Platform.current_grad_handle = None
        Platform.post_grad_handle_process = None

    @staticmethod
    def all_gather_object(object_list, obj, group=None) -> None:
        """Gather Python objects from all ranks into a list.

        Each rank contributes its object, and all ranks receive the complete list.

        Args:
            object_list (list): List to store gathered objects (output parameter).
            obj: The Python object from this rank to contribute.
            group: The process group for communication. Defaults to None (default group).
        """
        raise NotImplementedError("Platform subclasses must implement all_gather_object")

    @staticmethod
    def barrier(group=None, async_op: bool = False, device_ids=None) -> Any:
        """Synchronize all processes in the given process group.

        Each rank blocks until every rank in the group enters this collective (when ``async_op``
        is False), or returns an async handle that must be completed before proceeding.

        Args:
            group: The process group or communication group. ``None`` uses the default group.
            async_op (bool): If True, returns a backend-specific async work handle. Default: False.
            device_ids: Optional device id list; semantics depend on the backend.

        Returns:
            Async work handle when ``async_op`` is True; otherwise ``None`` (unless the rank
            is not in the group, in which case the backend may return ``None``).
        """
        raise NotImplementedError("Platform subclasses must implement barrier")

    @staticmethod
    def init_process_group(
            backend: Optional[str] = None,
            *,
            init_method: Optional[str] = None,
            timeout: Optional[timedelta] = None,
            world_size: int = -1,
            rank: int = -1,
            store: Any = None,
            pg_options: Any = None,
            device_id: Any = None
    ) -> None:
        """
        Initialize the default distributed process group.

        Args:
            backend: The backend to use for distributed communication
            init_method: URL specifying how to initialize the process group
            timeout: Timeout for operations executed against the process group
            world_size: Number of processes participating in the job
            rank: Rank of the current process
            store: Key/value store for exchanging connection information
            pg_options: Process group options for backend-specific configurations
            device_id: Specific device this process will work on

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Platform subclasses must implement init_process_group")

    @staticmethod
    def destroy_process_group(group=None) -> None:
        """
        Destroy a given process group.

        Args:
            group: The process group to be destroyed. If None, destroys the default group.

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Platform subclasses must implement destroy_process_group")

    @staticmethod
    def get_process_group_ranks(group=None) -> list[int]:
        """
        Get rank list of the given process group.

        Args:
            group: The process group to get ranks from. If None, uses the default group.

        Returns:
            List of ranks in the specified process group.

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Platform subclasses must implement get_process_group_ranks")

    @staticmethod
    def get_backend(group=None):
        """
        Get the backend of the given process group.
        Args:
            group: The process group to get backend from. If None, uses the default group.

        Returns:
            The backend name of the specified process group.

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Platform subclasses must implement get_backend")

    @staticmethod
    def split_group(parent_pg: Any = None,
                    split_ranks: Optional[list] = None,
                    timeout: Optional[timedelta] = None,
                    pg_options: Optional[Any] = None,
                    group_desc: Optional[str] = None,
                    ) -> Any:
        """Create a split group relative to the parent process group.

        Args:
            parent_pg: The parent process group to split from.
            split_ranks (list, optional): Ranks to include in the split group.
            timeout (timedelta, optional): Timeout for operations.
            pg_options: Process group options for backend-specific configurations.
            group_desc (str, optional): Description of the group.

        Returns:
            The new split process group.
        """
        raise NotImplementedError("Platform subclasses must implement split_group")

    @staticmethod
    def get_group_local_rank(group=None) -> int:
        """Get the local rank within the given process group.

        Args:
            group: The process group to query. If None, uses the default group.

        Returns:
            int: The local rank within the group.
        """
        raise NotImplementedError("Platform subclasses must implement get_group_local_rank")

    @staticmethod
    def no_grad():
        """Get a context manager to disable gradient computation.

        Returns:
            A context manager that disables gradient tracking.
        """
        raise NotImplementedError("Platform subclasses must implement no_grad")

    @staticmethod
    def preserve_version_counter(tensor):
        """Get a context manager that preserves version for an internal tensor update."""
        raise NotImplementedError("Platform subclasses must implement preserve_version_counter")

    @staticmethod
    def relu(tensor):
        """Apply ReLU activation element-wise.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor with ReLU applied (max(0, x)).
        """
        raise NotImplementedError("Platform subclasses must implement relu")

    @staticmethod
    def cat(tensors, dim=0):
        """Concatenate tensors along a dimension."""
        raise NotImplementedError("Platform subclasses must implement cat")

    @staticmethod
    def empty_like(tensor, *, dtype=None, device=None, pin_memory=False):
        """Create an uninitialized tensor with the same shape as input.

        Args:
            tensor: The input tensor to copy shape from.
            dtype: Optional dtype for the new tensor. If None, uses input tensor's dtype.
            device: Optional device for the new tensor. If None, uses input tensor's device.
            pin_memory (bool): If True, allocate pinned memory for faster CPU-GPU transfer.

        Returns:
            An uninitialized tensor with the same shape as input.
        """
        raise NotImplementedError("Platform subclasses must implement empty_like")

    def get_current_stream(self):
        """Get the current compute stream for the device.

        Returns:
            The current stream object.
        """
        raise NotImplementedError("Platform subclasses must implement get_current_stream")

    def new_event(self):
        """Create a new event for stream synchronization.

        Returns:
            A new event object.
        """
        raise NotImplementedError("Platform subclasses must implement new_event")

    def tree_map(self, fn, tree):
        """Apply a function to all tensors in a nested structure.

        Args:
            fn (callable): Function to apply to each tensor.
            tree: Nested structure (list, tuple, dict) containing tensors.

        Returns:
            The same nested structure with fn applied to all tensors.
        """
        raise NotImplementedError("Platform subclasses must implement tree_map")

    @staticmethod
    def is_linear_module(module) -> bool:
        """Check whether *module* is a linear/dense layer for the current framework.

        Args:
            module: The module instance to check.

        Returns:
            True if *module* is the framework's linear layer type.
        """
        raise NotImplementedError("Platform subclasses must implement is_linear_module")

    @staticmethod
    def is_embedding_module(module) -> bool:
        """Check whether *module* is an embedding layer for the current framework.

        Args:
            module: The module instance to check.

        Returns:
            True if *module* is the framework's embedding layer type.
        """
        raise NotImplementedError("Platform subclasses must implement is_embedding_module")

    @staticmethod
    def register_forward_pre_hook(module, hook, prepend=False, with_kwargs=False):
        """Register a forward pre-hook on a module.

        Args:
            module: The module to register the hook on.
            hook (callable): The hook function to register.
            prepend (bool): If True, prepend the hook to existing hooks.
            with_kwargs (bool): If True, hook receives both args and kwargs.

        Returns:
            A handle that can be used to remove the hook.
        """
        return module.register_forward_pre_hook(hook, prepend=prepend, with_kwargs=with_kwargs)

    @staticmethod
    def register_full_backward_hook(module, hook, prepend=False):
        """Register a full backward hook on a module.

        Args:
            module: The module to register the hook on.
            hook (callable): The hook function to register.
            prepend (bool): If True, prepend the hook to existing hooks.

        Returns:
            A handle that can be used to remove the hook.
        """
        return module.register_full_backward_hook(hook, prepend)

    @staticmethod
    def register_full_backward_pre_hook(module, hook, prepend=False):
        """Register a full backward pre-hook on a module.

        Args:
            module: The module to register the hook on.
            hook (callable): The hook function to register.
            prepend (bool): If True, prepend the hook to existing hooks.

        Returns:
            A handle that can be used to remove the hook.
        """
        return module.register_full_backward_pre_hook(hook, prepend)

    @property
    def checkpoint(self):
        """Get the checkpoint function for activation checkpointing.

        Returns:
            The checkpoint function for the current framework.
        """
        raise NotImplementedError("Platform subclasses must implement checkpoint")

    @staticmethod
    def checkpoint_wrapper(module, **checkpoint_kwargs):
        """Wrap a module with activation checkpointing functionality.

        Args:
            module: The module or callable to wrap with activation checkpointing.
            **checkpoint_kwargs: Keyword arguments forwarded to the framework
                checkpoint wrapper implementation.

        Returns:
            The wrapped module with activation checkpointing enabled.
        """
        raise NotImplementedError("Platform subclasses must implement checkpoint_wrapper")

    @staticmethod
    def checkpoint_exclude_wrapper(module: Any, *, save_output: bool = True) -> Any:
        """Wrap a callable whose activations should be saved instead of recomputed.

        Args:
            module: The module or callable to exclude from activation recomputation.
            save_output: Whether to retain the excluded region output for replay.

        Returns:
            The wrapped module or callable.
        """
        raise NotImplementedError("Platform subclasses must implement checkpoint_exclude_wrapper")

    @staticmethod
    def swap_wrapper(module, policy_fn=None, group_swap=False):
        """Wrap a module with activation swap functionality.

        Args:
            module: The module to wrap with activation swap.
            policy_fn: Optional per-tensor swap policy function.
            group_swap (bool, optional): Whether tensors participate in group copy fusion. Default: ``False``.

        Returns:
            The wrapped module with activation swap enabled.
        """
        raise NotImplementedError("Platform subclasses must implement swap_wrapper")

    @staticmethod
    def swap_tensor_wrapper(target, tag=None, group_swap=False):
        """Register target tensors into the current swap group.

        Args:
            target: A tensor or nested container of tensors to register.
            tag: Optional debug tag associated with the wrapped tensors.
            group_swap (bool, optional): Whether tensors participate in group copy fusion. Default: ``False``.

        Returns:
            The original target structure, unchanged semantically.
        """
        raise NotImplementedError("Platform subclasses must implement swap_tensor_wrapper")

    @staticmethod
    def get_class_activation_wrapper():
        """Return the platform-specific activation wrapper class."""
        raise NotImplementedError("Platform subclasses must implement get_class_activation_wrapper")

    @property
    def noop_context_fn(self):
        """Get a no-op context function for checkpointing.

        Returns:
            A context function that performs no operation.
        """
        raise NotImplementedError("Platform subclasses must implement noop_context_fn")

    @staticmethod
    def ignore_sac_ops(ignore_ops: list[object | None]) -> None:
        """Exclude backend operators from selective-checkpoint replay accounting.

        Args:
            ops (list[object | None]): Iterable of backend-native operator identifiers. Unavailable
                optional operators may be represented by ``None``.
        """
        raise NotImplementedError("Platform subclasses must implement ignore_sac_ops")

    @staticmethod
    def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False, group_swap=False):
        """Create contexts for selective activation checkpointing.

        Args:
            policy_fn_or_list: A policy function or list of layer names to checkpoint.
            allow_cache_entry_mutation (bool): Whether to allow cache entry mutation.
            group_swap (bool, optional): Whether MUST_SWAP tensors participate in group copy fusion. Default: ``False``.

        Returns:
            Context functions for selective checkpointing.
        """
        raise NotImplementedError("Platform subclasses must implement create_selective_checkpoint_contexts")

    @staticmethod
    def create_native_selective_checkpoint_contexts(policy_fn: Callable) -> Any:
        """Create framework-native selective checkpoint contexts for compile."""
        raise NotImplementedError(
            "Native selective checkpoint compile is not supported by this platform"
        )

    @staticmethod
    def async_save_on_cpu(policy_fn=None, group_swap: bool = False):
        """Create an async CPU offload context for activation checkpointing.

        Args:
            policy_fn: Optional policy function to determine which activations to offload.
            group_swap (bool): Whether swapped tensors participate in group copy fusion.
                Default: ``False``.

        Returns:
            Context manager for async CPU offloading during checkpointing.
        """
        raise NotImplementedError("Platform subclasses must implement async_save_on_cpu")

    @staticmethod
    def recompute_handle_collector_ctx():
        """Context manager that collects recompute handles created in its scope.

        Yields:
            A list populated with one opaque recompute handle per checkpointed
            block executed during the forward pass within the context.  Each
            handle can later be fired via :meth:`recompute_handle`.
        """
        raise NotImplementedError("Platform subclasses must implement recompute_handle_collector_ctx")

    @staticmethod
    def recompute_handle(handle, session_id):
        """Eagerly fire one checkpointed block's forward re-run.

        Materializes and caches the block's activations under ``session_id`` so
        a later backward in the same session reuses them instead of re-running.

        Args:
            handle: An opaque recompute handle from
                :meth:`recompute_handle_collector_ctx`.
            session_id: Stable key shared by the producing re-run and the
                consuming backward.
        """
        raise NotImplementedError("Platform subclasses must implement recompute_handle")

    @staticmethod
    def recompute_session_ctx(session_id, retain_on_unpack=False):
        """Context manager binding recompute unpack to a caller-provided session.

        Args:
            session_id: Required stable session key. Recompute caches are keyed
                by this instead of the transient autodiff engine id, so a re-run
                fired under one engine can be reused by another. Must not be
                ``None``.
            retain_on_unpack (bool): When ``True``, unpack returns recomputed
                tensors without popping them, so a later backward can consume
                them.  Default: ``False``.

        Returns:
            A context manager activating the session for its scope.

        Yields:
            The supplied session id.
        """
        raise NotImplementedError("Platform subclasses must implement recompute_session_ctx")

    @staticmethod
    def clear_recompute_session(session_id):
        """Release retained recompute data for a session.

        Args:
            session_id: The session key whose cached recompute data is cleared.
        """
        raise NotImplementedError("Platform subclasses must implement clear_recompute_session")

    @staticmethod
    def get_element_size(tensor):
        """Get Tensor Element Size"""
        raise NotImplementedError("Platform subclasses must implement get_element_size")

    @staticmethod
    def alloc_tensor_buffer(numel: int, dtype, device, pin_memory: bool = False):
        """Allocate an uninitialized 1-D tensor buffer."""
        raise NotImplementedError("Platform subclasses must implement alloc_tensor_buffer")

    @staticmethod
    def tensor_to_numpy(tensor) -> np.ndarray:
        """Convert a framework tensor to a NumPy array.

        Args:
            tensor: The tensor to convert.

        Returns:
            np.ndarray: The tensor data as a NumPy array.
        """
        raise NotImplementedError("Platform subclasses must implement tensor_to_numpy")

    @staticmethod
    def from_numpy(np_array):
        """Create a host-resident tensor from a NumPy array (inverse of tensor_to_numpy).

        The result stays on the host regardless of the active device context, so it
        remains asnumpy-able even when built under ``ms.DeviceCtx("meta")`` (e.g. while
        ``fully_shard`` lazily constructs a default device mesh). Use it for rank/mesh
        bookkeeping tensors, which are only ever read back via ``tensor_to_numpy``.
        """
        raise NotImplementedError("Platform subclasses must implement from_numpy")

    @staticmethod
    def profiler_record(name):
        """Record a profiler event with the given name.

        Args:
            name (str): The name of the profiler event.

        Returns:
            A context manager or decorator for profiling a code region.
        """
        raise NotImplementedError("Platform subclasses must implement profiler_record")

    def cast_fp_tensor(self, dtype, x):
        """Cast floating-point tensor to target dtype if applicable.

        Args:
            dtype: The target dtype to cast to.
            x: The input tensor.

        Returns:
            The tensor cast to target dtype, or unchanged if not floating-point.
        """
        raise NotImplementedError("Platform subclasses must implement cast_fp_tensor")

    def apply_to_tensors(self, fn, container):
        """Recursively apply a function to all tensors in a container.

        Supports nested structures including lists, tuples, and dicts.

        Args:
            fn (callable): Function to apply to each tensor.
            container: Nested structure containing tensors.

        Returns:
            The same structure with fn applied to all tensors.
        """
        raise NotImplementedError("Platform subclasses must implement apply_to_tensors")

    @staticmethod
    def clip_grad_norm_(
        parameters, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, foreach=None,
    ):
        """Compute and clip gradient norms for distributed models.

        Communication is derived from each parameter's DTensor spec.
        Subclasses must implement this method.

        Args:
            parameters: An ``nn.Module``, a single ``Tensor``, or an
                iterable of ``Tensor`` s whose gradients to clip.
            max_norm: Maximum allowed gradient norm.
            norm_type: Type of the norm (default ``2.0``).
            error_if_nonfinite: If ``True``, raise when total norm is
                non-finite. Default ``False``.
            foreach: Unused, accepted for API compatibility.

        Returns:
            The total (unclipped) gradient norm.
        """
        raise NotImplementedError(
            "Platform subclasses must implement clip_grad_norm_"
        )

    @staticmethod
    def get_created_group(rank_list: Union[list[int], tuple[int]]):
        """Get an existing process group by rank list.

        Args:
            rank_list (Union[list[int], tuple[int]]): Tuple or list of ranks.

        Returns:
            The process group corresponding to the rank list if it exists, else None.
        """
        group_key = str(tuple(sorted(rank_list)))
        if group_key in EXISTING_COMM_GROUPS:
            return EXISTING_COMM_GROUPS[group_key]
        return None

    @classmethod
    def mark_created_groups(cls, process_group: Union[Any, list[Any]]) -> None:
        """Register process groups in the global cache for reuse.

        Args:
            process_group (Union[Any, list[Any]]): A process group or a list of process groups.
        """
        if not isinstance(process_group, list):
            process_group = [process_group]
        for group in process_group:
            rank_list = cls.get_process_group_ranks(group)
            group_key = str(tuple(sorted(rank_list)))
            EXISTING_COMM_GROUPS[group_key] = group

    @property
    def meta_device(self):
        """Get the framework-specific meta device for tensor shape inference.

        The meta device allows creating tensors without allocating actual storage,
        useful for shape inference and model initialization.

        Returns:
            The meta device object for the current framework.
        """
        raise NotImplementedError("Platform subclasses must implement meta_device")

    def init_on_device(self, device, include_buffers=False):
        """Get a context manager for initializing module parameters on a device.

        Args:
            device: The target device for parameter initialization.
            include_buffers (bool): If True, also initialize buffers on the device.

        Returns:
            A context manager for device-specific initialization.
        """
        raise NotImplementedError("Platform subclasses must implement init_on_device")

    def str_to_dtype(self, dtype_str: str) -> Any:
        """
        Map a framework-style dtype string (e.g. ``torch.float32``) to the backend dtype object.

        Args:
            dtype_str (str): Serialized dtype identifier produced by checkpoint metadata.

        Returns:
            Framework dtype object (e.g. ``torch.dtype`` or MindSpore dtype).
        """
        raise NotImplementedError("Platform subclasses must implement str_to_dtype")

    def list_to_size(self, size_list: list[int]) -> Any:
        """
        Convert a shape list from checkpoint metadata to the framework's size type (e.g. ``torch.Size``).

        Args:
            size_list (list[int]): Tensor global shape as a list of ints.

        Returns:
            Framework-specific size object.
        """
        raise NotImplementedError("Platform subclasses must implement list_to_size")
