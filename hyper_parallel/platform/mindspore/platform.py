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
"""MindSpore platform api"""
from datetime import timedelta
from typing import Any, Optional, Union
import dataclasses
from collections import OrderedDict

import contextlib
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.mint.distributed import TCPStore

from mindspore.nn import Cell
from mindspore import mint
from mindspore.common.api import _no_grad
from mindspore.common._grad_function import _Function
from mindspore.common.dtype import type_size_in_bytes
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.recompute import null_context_fn
from mindspore.communication import GlobalComm
from mindspore.communication import get_group_size
from mindspore.communication import create_group as new_group
from mindspore.communication import get_rank as get_rank_id
from mindspore.ops import communication as ops_comm
from mindspore.ops.function import comm_func
from mindspore._c_expression import TensorTransform
import mindspore.mint.distributed as dist

from hyper_parallel.platform.platform import Platform, PlatformType, EXISTING_COMM_GROUPS
from hyper_parallel.platform.mindspore.dtensor import DTensorBase
from hyper_parallel.platform.mindspore.pipeline_parallel.stage import PipelineStageBase
from hyper_parallel.platform.mindspore.parameter_init import init_parameters as _init_parameters
from hyper_parallel.platform.mindspore.init_weights import (
    init_on_device as _init_on_device,
    _install_cell_to_empty_patch,
)

comm_func.set_comm_ops_inplace(False)
_tensor_transform = TensorTransform.get_instance()


# pylint: disable=C0103


def _a2a_reconstruct_ms(out_perm: Tensor, concat_dim: int) -> Tensor:
    """Reconstruct A2A result from raw out_perm buffer."""
    new_ndim = out_perm.dim()
    chunk_in_perm = concat_dim + 1
    recon_perm = list(range(1, chunk_in_perm)) + [0] + list(range(chunk_in_perm, new_ndim))
    x_recon = out_perm.permute(recon_perm).contiguous()
    shape = list(x_recon.shape)
    merged = shape[concat_dim] * shape[concat_dim + 1]
    return x_recon.reshape(shape[:concat_dim] + [merged] + shape[concat_dim + 2:])


def _normalize_all_to_all_single_result(result, output: Tensor) -> tuple[Tensor, object]:
    """Normalize MindSpore all_to_all_single return values to ``(output, handle)``."""
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError(
                "mindspore all_to_all_single returned an unexpected tuple "
                f"with length {len(result)}"
            )
        return result
    return output, result


def _mindspore_all_to_all_single(input_tensor: Tensor, output_shape, group, async_op=False) -> tuple[Tensor, object]:
    """Launch MindSpore all_to_all_single and normalize return values."""
    output = mint.empty(tuple(output_shape), dtype=input_tensor.dtype)
    result = ops_comm.all_to_all_single(output, input_tensor, group=group, async_op=async_op)
    normalized_output, handle = _normalize_all_to_all_single_result(result, output)
    if not async_op:
        return normalized_output, None
    return normalized_output, handle


class _MSAsyncA2AFunction(_Function):
    """Differentiable wrapper for pre-launched async all-to-all."""

    @staticmethod
    def forward(ctx, x, work, out_perm, group, world_size, concat_dim, split_dim, handle_box):  # pylint: disable=arguments-differ
        """Wait for pre-launched async A2A and return reconstructed output."""
        ctx.group = group
        ctx.world_size = world_size
        ctx.concat_dim = concat_dim
        ctx.split_dim = split_dim
        ctx.handle_box = handle_box
        ctx.x_shape = tuple(x.shape)
        work.wait()
        return _a2a_reconstruct_ms(out_perm, concat_dim)

    @staticmethod
    def backward(ctx, grad_output):
        """Launch async head->seq A2A for backward overlap, or return zero grad."""
        if ctx.handle_box is not None:
            g = grad_output.contiguous()
            shape = list(g.shape)
            seq_dim = ctx.concat_dim
            s_full = shape[seq_dim]
            ndim = len(shape) + 1
            x_perm = g.reshape(
                shape[:seq_dim] + [ctx.world_size, s_full // ctx.world_size] + shape[seq_dim + 1:]
            ).permute(
                [seq_dim] + list(range(seq_dim)) + list(range(seq_dim + 1, ndim))
            ).contiguous()
            out_perm, work = _mindspore_all_to_all_single(
                x_perm,
                list(x_perm.shape),
                ctx.group,
                async_op=True,
            )
            ctx.handle_box.append((work, out_perm))
        return mint.zeros(ctx.x_shape, dtype=grad_output.dtype), None, None, None, None, None, None, None


class MindSporePlatform(Platform):
    """MindSpore platform api"""
    Tensor = Tensor
    tensor = Tensor
    Parameter = Parameter
    Module = Cell
    DTensorBase = DTensorBase
    PipelineStageBase = PipelineStageBase
    platform_type = PlatformType.MINDSPORE
    tensor_dtype = mstype
    dtype = ms.Type
    Function = _Function

    def __init__(self):
        # Ensure MindSpore ``nn.Cell.to_empty`` is patched as soon as the
        # MindSpore platform instance is created.
        _install_cell_to_empty_patch()

    @staticmethod
    def apply_runtime_patches_if_needed():
        """Apply MindSpore-specific runtime patches."""
        from hyper_parallel.platform.mindspore.parameter_param_info_patch import (  # pylint: disable=import-outside-toplevel
            patch_mindspore_parameter_param_info_cycle_if_needed,
        )

        patch_mindspore_parameter_param_info_cycle_if_needed()

    @staticmethod
    def is_linear_module(module) -> bool:
        """Check whether *module* is a MindSpore ``Dense`` (linear) or ``mint.nn.Linear`` layer."""
        return isinstance(module, (ms.nn.Dense, mint.nn.Linear))

    @staticmethod
    def is_embedding_module(module) -> bool:
        """Check whether *module* is a MindSpore ``Embedding`` or ``mint.nn.Embedding`` layer."""
        return isinstance(module, (ms.nn.Embedding, mint.nn.Embedding))

    def device_count(self, device_handle):
        """
        Get the number of available devices.

        Args:
            device_handle: The device handle (e.g., ms.device_context).

        Returns:
            int: The number of available devices.
        """
        device_type = self.device_type()
        if device_type == "cpu":
            return device_handle.device_context.cpu.device_count()
        if device_type == "gpu":
            return device_handle.device_context.gpu.device_count()
        return device_handle.device_context.ascend.device_count()

    @staticmethod
    def get_rng_state(device=None, device_handle=None):
        """
        Get the random number generator state.

        Args:
            device (Optional): The device to get RNG state from (not used in MindSpore).
            device_handle (Optional): The device handle (not used in MindSpore).

        Returns:
            Tensor: The RNG state as a tensor.
        """
        _ = device, device_handle
        return ms.get_rng_state()

    @staticmethod
    def set_rng_state(state, device=None, device_handle=None):
        """
        Set the random number generator state.

        Args:
            state (Tensor): The RNG state to set.
            device (Optional): The device to set RNG state for (not used in MindSpore).
            device_handle (Optional): The device handle (not used in MindSpore).
        """
        _ = device, device_handle
        return ms.set_rng_state(state)

    def device_type(self):
        """
        Get the current device type.

        Returns:
            str: The device type string ("npu" for Ascend, "gpu" for GPU, "cpu" for CPU).
        """
        device_type = ms.get_context("device_target")
        if device_type == "Ascend":
            return "npu"
        return device_type.lower()

    def device(self, device_idx=None):
        """
        Get the device type string.

        Args:
            device_idx (Optional[int]): The device index (not used in MindSpore).

        Returns:
            str: The device type string.
        """
        _ = device_idx
        device_type = self.device_type()
        return device_type

    @staticmethod
    def get_device_handle():
        """
        Get the MindSpore module as the device handle.

        Returns:
            module: The mindspore module.
        """
        return ms

    @staticmethod
    def manual_seed(seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The random seed value.

        Returns:
            None
        """
        return ms.manual_seed(seed)

    @staticmethod
    def ones(size, dtype=None):
        """
        Create a tensor filled with ones.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[ms.Type]): The desired data type.

        Returns:
            Tensor: A tensor filled with ones.
        """
        return mint.ones(size, dtype=dtype)

    @staticmethod
    def zeros(size, dtype=None, device=None):
        """
        Create a tensor filled with zeros.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[ms.Type]): The desired data type.
            device (Optional[ms.device]): The device to create the tensor on.

        Returns:
            Tensor: A tensor filled with zeros.
        """
        tensor = mint.zeros(size, dtype=dtype)
        if device in ("GPU", "Ascend"):
            return tensor.to(device)
        return tensor

    @staticmethod
    def full(size, fill_value, dtype=None):
        """
        Create a tensor filled with a scalar value.

        Args:
            size (tuple): The shape of the output tensor.
            fill_value (scalar): The value to fill the tensor with.
            dtype (Optional[ms.Type]): The desired data type.

        Returns:
            Tensor: A tensor filled with the specified value.
        """
        return mint.full(size, fill_value, dtype=dtype)

    @staticmethod
    def empty(size, dtype=None):
        """
        Create an uninitialized tensor.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[ms.Type]): The desired data type.

        Returns:
            Tensor: An uninitialized tensor.
        """
        return mint.empty(size, dtype=dtype)

    @staticmethod
    def get_rank():
        """
        Get the rank of the current process in the distributed group.

        Returns:
            int: The rank of the current process.
        """
        return get_rank_id()

    @staticmethod
    def get_global_rank(group, group_rank):
        """
        Get the global rank from a group rank.

        Args:
            group (str): The process group name.
            group_rank (int): The rank within the group.

        Returns:
            int: The global rank.
        """
        return dist.get_global_rank(group, group_rank)

    @staticmethod
    def get_world_size():
        """
        Get the total number of processes in the distributed group.

        Returns:
            int: The world size.
        """
        return get_group_size()

    @staticmethod
    def get_op_name(func):
        """
        Extract the operation name from a function.

        Args:
            func: The function to extract the name from.

        Returns:
            str: The operation name.
        """
        return func.name

    @staticmethod
    def differentiable_all_gather_concat(data, group, concat_size, concat_dim):
        output, _ = comm_func.all_gather_into_tensor(None, data, group=group)
        if concat_dim == 0:
            return output
        output_tensors = ms.ops.Split(output_num=concat_size)(output)
        return ms.mint.concat(output_tensors, concat_dim)

    @staticmethod
    def chunk(data, split_dim, split_size, index):
        return ms.ops.Split(axis=split_dim, output_num=split_size)(data)[index]

    @staticmethod
    def differentiable_all_to_all(input_data, output_shape, group):
        output_tensor, _ = comm_func.all_to_all_single(
            output_shape,
            input_data,
            group=group,
            async_op=False
        )
        return output_tensor

    @staticmethod
    def tensor_type_cast(input_data, cast_type):
        """Cast tensor to specified data type."""
        type_mapping = {
            'float32': ms.float32,
            'float16': ms.float16,
            'int64': ms.int64,
            'int32': ms.int32
        }
        if cast_type not in type_mapping:
            raise ValueError(f"Unknown cast type: {cast_type}. Supported types: {list(type_mapping.keys())}")
        return input_data.to(type_mapping[cast_type])

    @staticmethod
    def differentiable_all_reduce(data, op, group):
        output, _ = comm_func.all_reduce(data, op, group)
        return output

    @staticmethod
    def differentiable_reduce_scatter(data, dev_num, axis, op, group):
        if axis > 0:
            data = ms.mint.concat(ms.ops.Split(axis=axis, output_num=dev_num)(data), dim=0)
        output_tensor, _ = comm_func.reduce_scatter_tensor(None, data, 'sum', group)
        if op == 'avg':
            output_tensor = output_tensor / dev_num
        return output_tensor

    @staticmethod
    def init_parameters(module, stage_index):
        return _init_parameters(module, stage_index)

    # pylint: disable=W0212
    @staticmethod
    def update_param_data(param, data):
        """update param data"""
        if isinstance(param, DTensorBase):
            param.set_data(data)
        else:
            param._update_data(data)

    @staticmethod
    def load_into_param(param, data):
        copy_tensor = MindSporePlatform.empty_like(data)
        copy_tensor.copy_(data)
        if isinstance(param, DTensorBase):
            param.set_data(copy_tensor)
        else:
            param._update(copy_tensor)

    @staticmethod
    def get_cell_construct(cell):
        return cell.construct

    @staticmethod
    def get_cells_and_names(cell):
        return cell.cells_and_names()

    @staticmethod
    def search_parameter_by_name(cell, param_name: str):
        """
        Find the parent Module of the parameter, the parameter's name in the parent Module, and the parameter.
        Return value: (parent Module instance, parameter's name in parent Module, parameter object).
        Returns None if not found.
        """
        # Remove the "self." prefix from param_name (to maintain compatibility with original logic)
        param_name = param_name.replace("self.", "")
        # Case 1: The parameter is a direct parameter of the current Module (not in any sub-Module)
        if param_name in cell._params:
            return (cell, param_name, cell._params[param_name])

        # Case 2: The parameter is in a sub-Module (supports multi-level nesting, e.g., "net_b.dense1.weight")
        if "." in param_name:
            # Split into: sub-Module path + parameter name (e.g., "net_b.dense1" + "weight")
            cell_path, param_key = param_name.rsplit(".", 1)
            try:
                # Locate the sub-Module where the parameter resides (supports multi-level paths)
                target_cell = cell.get_sub_cell(cell_path)
                # Check if the sub-Module directly contains this parameter
                if param_key in target_cell._params:
                    return target_cell, param_key, target_cell._params[param_key]
            except AttributeError:
                # Sub-Module path does not exist or the parameter is not in that sub-Module
                pass

        # Traverse all sub-Modules (recursively) to search for the parameter
        for _, child_cell in cell._cells.items():
            if isinstance(child_cell, Cell):
                # Recursively search within the sub-Module
                result = MindSporePlatform.search_parameter_by_name(child_cell, param_name)
                if result is not None:
                    return result

        return None

    @staticmethod
    def update_parameter_by_name(cell, result: tuple, new_param) -> bool:
        """
        Modify the original parameter in a Module or sub-Module using the search result
        Args:
            cell: The cell which parameter is to update
            result: A tuple contains parent Module, parameter key and old parameter.
            new_param: New Parameter object (used to replace the original parameter)
        """
        parent_cell, param_key, _ = result
        # Key operation: directly modify the _params dictionary of the parent Module (original storage location)
        parent_cell._params[param_key] = new_param

        if param_key in parent_cell.__dict__:
            parent_cell.__dict__[param_key] = new_param
        parent_cell._params_list[param_key] = new_param
        return True

    @staticmethod
    def set_layout_into_parameter(param, layout):
        """Set layout in to parameter"""
        from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=import-outside-toplevel
        from hyper_parallel.core.dtensor.layout import _infer_slice_shape_by_layout, \
            _get_slice_tensor_by_layout  # pylint: disable=import-outside-toplevel
        if isinstance(param, DTensor):
            raise ValueError(f"Parameter {param.name} has been configured layout, cannot be set repeatedly.")
        param_info = param.param_info
        requires_grad = param.requires_grad
        name = param.name
        slice_shape = _infer_slice_shape_by_layout(param.shape, layout)

        if not param.has_init:
            # has been init, get slice data
            param_dtensor = DTensor.from_local(
                _get_slice_tensor_by_layout(param, layout).value(), layout.mesh, layout.alias_placements
            )
            param = Parameter(param_dtensor, name=name, requires_grad=requires_grad)
            param.param_info = param_info
        else:
            # has not been init, need to modify init shape
            param.init_mode.shape = slice_shape
            param_dtensor = DTensor.from_local(param.init_mode, layout.mesh, layout.alias_placements)
            param = Parameter(param_dtensor, name=name, requires_grad=requires_grad)
            param.param_info = param_info
        return param

    @staticmethod
    def get_param_local_shape(param):
        """get param local shape"""
        if isinstance(param, DTensorBase):
            return param.local_shape
        return param.shape

    @staticmethod
    def get_param_local_data(param):
        """get param local shape"""
        if isinstance(param, DTensorBase):
            return param.to_local()
        return param

    @staticmethod
    def get_param_type_size(param):
        return type_size_in_bytes(param.dtype)

    @staticmethod
    def is_tensor(obj: Any) -> bool:
        """Return True if ``obj`` is a ``mindspore.Tensor``."""
        return isinstance(obj, Tensor)

    @staticmethod
    def get_tensor_storage_size(tensor: Any) -> int:
        """Return serialized byte size (numel * itemsize) for a MindSpore tensor."""
        if not MindSporePlatform.is_tensor(tensor):
            raise TypeError(
                f"MindSporePlatform.get_tensor_storage_size expects mindspore.Tensor, got {type(tensor)!r}"
            )
        return int(tensor.numel()) * int(tensor.itemsize)

    @staticmethod
    def new_zero_parameter(param_shape, param_type, requires_grad, device):
        param = Parameter(initializer("zeros", param_shape, param_type), requires_grad=requires_grad)
        if device in ("GPU", "Ascend"):
            return param.to(device)
        return param

    @staticmethod
    def new_tensor(tensor_shape, tensor_type, device):
        tensor = Tensor(shape=tensor_shape, dtype=tensor_type)
        if device in ("GPU", "Ascend"):
            return tensor.to(device)
        return tensor

    @staticmethod
    def full_like(tensor, fill_value, dtype=None):
        return mint.full_like(tensor, fill_value, dtype=dtype)

    @staticmethod
    def isend(tensor, dst=None, group=None, tag=0):
        return dist.isend(tensor, dst, group, tag)

    @staticmethod
    def irecv(tensor, src=None, group=None, tag=0):
        return dist.irecv(tensor, src, group, tag)

    @staticmethod
    def p2p_exchange(tensor, peer_rank: int, group=None):  # pylint: disable=unused-argument
        raise NotImplementedError(
            "p2p_exchange is not yet supported on the MindSpore platform."
        )

    @staticmethod
    def send_object_list(obj_list, dst=None, group=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.pipeline_parallel._utils import send_object_list
        send_object_list(obj_list, dst, group)

    @staticmethod
    def recv_object_list(obj_list, src=None, group=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.pipeline_parallel._utils import recv_object_list
        recv_object_list(obj_list, src, group)

    @staticmethod
    def set_tensor_requires_grad(input_tensor):
        """
        set requires grad flag for input tensor
        """
        input_tensor.requires_grad_()

    def _create_group(self, rank_list):
        world_group = self._maybe_reuse_world_group(rank_list)
        if world_group is not None:
            return world_group

        group_name = str(tuple(sorted(rank_list)))
        new_group(rank_ids=rank_list, group=group_name)
        EXISTING_COMM_GROUPS[group_name] = group_name
        return group_name

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        return comm_func.all_gather_into_tensor(None, data, group=group_info.group_name, async_op=async_op)

    @staticmethod
    def all_reduce(data, group_info, async_op=False):
        if isinstance(group_info, str):
            handle = dist.all_reduce(data, group=group_info, async_op=async_op)
        else:
            handle = dist.all_reduce(data, group=group_info.group_name, async_op=async_op)
        return data, handle

    @staticmethod
    def broadcast(data, src, group=None, async_op=False):
        handle = dist.broadcast(data, src, group, async_op)
        if async_op:
            handle.wait()
        return data

    @staticmethod
    def reduce_scatter_tensor(data, group_info, async_op=False):
        return comm_func.reduce_scatter_tensor(None, data, group=group_info.group_name, async_op=async_op)

    @staticmethod
    def all_to_all_single(input_tensor, output_shape, group, async_op=False):
        return _mindspore_all_to_all_single(input_tensor, output_shape, group, async_op=async_op)

    @staticmethod
    def differentiable_async_a2a_wait(x, work, out_perm, group, world_size, concat_dim, split_dim,  # pylint: disable=unused-argument
                                      handle_box=None):
        return _MSAsyncA2AFunction.apply(
            x, work, out_perm, group, world_size, concat_dim, split_dim, handle_box
        )

    @staticmethod
    def parameters_dict(cell: Cell):
        return cell.parameters_and_names()

    @staticmethod
    def get_tensor_transform():
        return _tensor_transform

    @staticmethod
    def construct_strided_slice(x, begin, end, stride):
        return ms.ops.strided_slice(x, begin, end, stride)

    @staticmethod
    def micro_batch(micro_batch_num, args_batch_dim=None, kwargs_batch_dim=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.pipeline_parallel._utils import _MicroBatch
        return _MicroBatch(micro_batch_num, args_batch_dim, kwargs_batch_dim)

    @staticmethod
    def get_model_state_dict(model, *, options=None):
        raise NotImplementedError(
            "get_model_state_dict is not yet supported on MindSpore"
        )

    @staticmethod
    def save_checkpoint(cell: Union[Cell, dict], file_path: str, ckpt_format: str = "safetensors") -> None:
        if isinstance(cell, dict):
            save_dict = {}
            for k, v in cell.items():
                if isinstance(v, Parameter):
                    save_dict[k] = v
                elif isinstance(v, Tensor):
                    save_dict[k] = Parameter(v, name=k)
                else:
                    save_dict[k] = v
        else:
            save_dict = cell._params
        ms.save_checkpoint(save_obj=save_dict, ckpt_file_name=file_path, format=ckpt_format)

    @staticmethod
    def load_checkpoint(file_path: str, ckpt_format: str = "safetensors") -> dict:
        return ms.load_checkpoint(ckpt_file_name=file_path, format=ckpt_format)

    @staticmethod
    def get_symmetric_memory_handler():
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.symmetric_memory import MSSymmetricMemoryHandler
        symmetric_memory = MSSymmetricMemoryHandler()
        return symmetric_memory

    @staticmethod
    def get_multicore_handler():
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.multicore import MSMulticoreHandler
        return MSMulticoreHandler()

    def new_stream(self):
        return ms.runtime.Stream()

    def get_stream_context(self):
        return ms.runtime.StreamCtx

    @staticmethod
    def all_gather_object(object_list, obj, group=None) -> None:
        """
        Gathers objects from the given group into object list.

        Args:
            object_list (list[Any]): Define the output list, which size equal to the size of group.
            obj (Any): The object on current rank and in given process group.
            group (ProcessGroup, optional): The process group to gather obj. Default is ``None``, and ``None`` means
                global group.

        Returns:
            None. Objs are gathered into ``object_list``.
        """
        dist.all_gather_object(object_list, obj, group)

    @staticmethod
    def barrier(group=None, async_op: bool = False, device_ids=None) -> Any:
        """
        Synchronize all processes in the given communication group.

        Args:
            group (str, optional): The communication group to work on. Default is ``None``,
                meaning the default world group.
            async_op (bool, optional): Whether this op should be asynchronous. Default: ``False``.
            device_ids (list[int], optional): Reserved parameter on Ascend. Default: ``None``.

        Returns:
            CommHandle if ``async_op`` is True; otherwise ``None``.
        """
        return dist.barrier(group, async_op, device_ids)

    @staticmethod
    def init_process_group(
            backend: str = None,
            *,
            init_method: Optional[str] = None,
            timeout: Optional[timedelta] = None,
            world_size: int = -1,
            rank: int = -1,
            store: TCPStore = None,
            pg_options=None,
            device_id=None
    ) -> None:
        """
        Initialize global process group.

        Args:
            backend (str): The backend used to init process group. Default is ``"hccl"`` and now only support hccl.
            init_method (str, optional): URL specifying how to initialize the process group. Default is ``None``.
            timeout (timedelta, optional): Timeout for API executed. Default is ``None``.
            world_size (int): Number of processes. Default is ``-1``.
            rank (int, optional): Rank of the current process. Default is ``-1``.
            store (Store, optional): An object that stores key/value data, facilitating the exchange of inter-process
                communication addresses and connection information. Default is ``None``. Currently, only the
                ``TCPStore`` type is supported.
            pg_options (ProcessGroupOptions, optional): Reserved parameter. Current not take effect.
            device_id (int, optional): Reserved parameter. Current not take effect.
        """
        if backend is None:
            backend = "hccl"
        try:
            if dist.is_initialized():
                return
        except AttributeError:
            pass
        dist.init_process_group(backend=backend, init_method=init_method, timeout=timeout, world_size=world_size,
                                rank=rank, store=store, pg_options=pg_options, device_id=device_id)

    @staticmethod
    def destroy_process_group(group: Optional[str] = None) -> None:
        """
        Destroy given process group.

        Args:
            group (str, optional): Specify the group to destroy. Default: ``None`` means ``hccl_world_group``. If group
                is None or "hccl_world_group", destroy global process group and all process groups relative to global
                process group.
        """
        if group in EXISTING_COMM_GROUPS.values():
            keys_to_destroy = [k for k, v in EXISTING_COMM_GROUPS.items() if v == group]
            for k in keys_to_destroy:
                del EXISTING_COMM_GROUPS[k]
        dist.destroy_process_group(group)

    @staticmethod
    def get_process_group_ranks(group: Optional[str] = None) -> list[int]:
        """
        Get all ranks in given process group.

        Args:
            group (str, optional): Specify the process group to work on. Default: ``None`` means ``hccl_world_group``.

        Returns:
            List[int]: List of ranks in given process group.
        """
        return dist.get_process_group_ranks(group)

    @staticmethod
    def get_backend(group: Optional[str] = None) -> str:
        """
        Get the backend of given process group.

        Args:
            group (str, optional): Specify the process group to work on. Default: ``None`` means ``hccl_world_group``.

        Returns:
            str: The backend of the group.
        """
        return dist.get_backend(group)

    @staticmethod
    def split_group(parent_pg: Optional[str] = None,
                    split_ranks: Optional[list] = None,
                    timeout: Optional[timedelta] = None,
                    pg_options: Optional[str] = None,
                    group_desc: Optional[str] = None,
                    ) -> str:
        """
        Create split group for a specific group rank in split_ranks, which group contains current rank id.

        Args:
            parent_pg (str, Optional): A process group which the goal group split from.
            split_ranks (Optional[list]): A list like ``list[list[int]]``.
            timeout (Optional[timedelta]): Timeout for API executed. Default is ``None``.
            pg_options (Optional[str]): Reserved parameter. Current not take effect.
            group_desc (Optional[str]): Description of process group.

        Returns:
            str: The split group name.
        """
        if split_ranks is None or len(split_ranks) == 0:
            raise ValueError("split_ranks cannot be None or empty")

        rank_id = MindSporePlatform.get_rank()
        for split_rank in split_ranks:
            if rank_id in split_rank:
                world_group = MindSporePlatform._maybe_reuse_world_group(split_rank)
                if world_group is not None:
                    return world_group
                split_group = MindSporePlatform.get_created_group(split_rank)
                if split_group:
                    return split_group
                group_name = str(tuple(sorted(split_rank)))
                new_group(rank_ids=split_rank, group=group_name)
                EXISTING_COMM_GROUPS[group_name] = group_name
                return group_name
        raise ValueError(f"Split group invalid rank, the Split_ranks {split_ranks} does not contain current rank"
                         f" {rank_id}")

    @staticmethod
    def get_group_local_rank(group=None) -> int:
        """get group local rank id."""
        return dist.get_group_rank(group, MindSporePlatform.get_rank())

    @staticmethod
    def no_grad():
        return _no_grad()

    @staticmethod
    def cat(tensors, dim=0):
        return mint.cat(tensors, dim=dim)

    @staticmethod
    def empty_like(tensor, *, dtype=None, device=None, pin_memory=False):
        return mint.empty_like(tensor, dtype=dtype, device=device, pin_memory=pin_memory)

    def get_current_stream(self):
        return ms.runtime.current_stream()

    def new_event(self):
        return ms.runtime.Event()

    def tree_map(self, fn, tree):
        """
        Apply fn to each leaf in a nested structure (list / tuple / dict),
        preserving the original structure.
        """
        if isinstance(tree, dict):
            return type(tree)(
                (k, self.tree_map(fn, v)) for k, v in tree.items()
            )

        if isinstance(tree, tuple):
            return tuple(self.tree_map(fn, v) for v in tree)

        if isinstance(tree, list):
            return [self.tree_map(fn, v) for v in tree]

        # leaf
        return fn(tree)

    @staticmethod
    def register_forward_pre_hook(module, hook, prepend=False, with_kwargs=False):
        return module.register_forward_pre_hook(hook, with_kwargs=with_kwargs)

    @staticmethod
    def register_full_backward_hook(module, hook, prepend=False):
        return module.register_backward_hook(hook)

    @staticmethod
    def register_full_backward_pre_hook(module, hook, prepend=False):
        return module.register_backward_pre_hook(hook)

    @property
    def checkpoint(self):
        return ms.recompute

    @staticmethod
    def ckpt_wrapper(module, checkpoint_fn=None, **checkpoint_fn_kwargs):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.activation_checkpoint.checkpoint_wrapper import checkpoint_wrapper
        return checkpoint_wrapper(module, checkpoint_fn=checkpoint_fn, **checkpoint_fn_kwargs)

    @staticmethod
    def swap_wrapper(module, policy_fn=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import swap_wrapper
        return swap_wrapper(module, policy_fn=policy_fn)

    @property
    def noop_context_fn(self):
        return null_context_fn

    @staticmethod
    def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.activation_checkpoint.sac import create_selective_checkpoint_contexts
        return create_selective_checkpoint_contexts(policy_fn_or_list,
                                                    allow_cache_entry_mutation=allow_cache_entry_mutation)

    @staticmethod
    def async_save_on_cpu(policy_fn=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import AsyncSaveOnCpu
        return AsyncSaveOnCpu(policy_fn=policy_fn)

    @staticmethod
    def get_element_size(tensor):
        """Get Tensor Element Size"""
        return tensor.itemsize

    @staticmethod
    def tensor_to_numpy(tensor) -> np.ndarray:
        """Convert MindSpore tensor to numpy array."""
        return tensor.asnumpy()

    @staticmethod

    def clip_grad_norm_(
        parameters, max_norm, norm_type=2.0,
        error_if_nonfinite=False, foreach=None,
    ):
        raise NotImplementedError(
            "clip_grad_norm_ is not yet supported on MindSpore"
        )

    @property
    def meta_device(self):
        return "meta"

    def init_on_device(self, device, include_buffers=False):
        return _init_on_device(device, include_buffers=include_buffers)

    def cast_fp_tensor(self, dtype, x):
        """
        Cast floating-point tensor to target dtype if applicable.
        """
        if (
            not isinstance(x, ms.Tensor)
            or not ms.ops.is_floating_point(x)
            or x.dtype == dtype
        ):
            return x
        return x.to(dtype)

    def apply_to_tensors(self, fn, container):
        """Recursively apply to all tensor in different kinds of container types."""

        def apply(x):
            if isinstance(x, ms.Tensor):
                return fn(x)
            if hasattr(x, "__dataclass_fields__"):
                dc = dataclasses.replace(x)
                changes = {
                    f.name: apply(getattr(dc, f.name)) for f in dataclasses.fields(dc)
                }
                return dataclasses.replace(dc, **changes)
            if isinstance(x, OrderedDict):
                od = x.__class__()
                for key, value in x.items():
                    od[key] = apply(value)
                return od
            if isinstance(x, dict):
                return {key: apply(value) for key, value in x.items()}
            if isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields"):
                res = (apply(el) for el in x)
                return type(x)(*res)
            if isinstance(x, (list, tuple, set)):
                return type(x)(apply(el) for el in x)
            return x

        return apply(container)

    @staticmethod
    def profiler_record(name):
        """Profiler context manager for recording operations using mindspore.profiler."""
        return contextlib.nullcontext()

    def str_to_dtype(self, dtype_str: str) -> Any:
        """Resolve checkpoint dtype strings (``mindspore.*`` or short ``str(Tensor.dtype)`` e.g. ``Float32``)."""
        if "." in dtype_str:
            prefix, name = dtype_str.split(".", 1)
            if prefix == "mindspore":
                return getattr(ms, name)
        dtype = getattr(ms, dtype_str.lower(), None)
        if dtype is not None:
            return dtype
        raise ValueError(
            f"Expected dtype string like 'mindspore.float32' or 'Float32', got {dtype_str!r}."
        )

    def list_to_size(self, size_list: list[int]) -> tuple[int, ...]:
        return tuple(size_list)

    @staticmethod
    def _maybe_reuse_world_group(rank_list):
        """Reuse the default world group for full-world rank lists."""
        normalized = tuple(sorted(rank_list))
        world_ranks = tuple(range(MindSporePlatform.get_world_size()))
        if normalized != world_ranks:
            return None

        EXISTING_COMM_GROUPS[str(normalized)] = GlobalComm.WORLD_COMM_GROUP
        return GlobalComm.WORLD_COMM_GROUP
