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
"""MindSpore platform api"""
from datetime import timedelta
from typing import Optional

import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.mint.distributed import TCPStore

from mindspore.nn import Cell
from mindspore import mint
from mindspore.common.api import _no_grad
from mindspore.common.dtype import type_size_in_bytes
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.communication import get_group_size
from mindspore.communication import create_group as new_group
from mindspore.communication import get_rank as get_rank_id
from mindspore.communication import comm_func
from mindspore._c_expression import TensorTransform
import mindspore.mint.distributed as dist

from hyper_parallel.platform.platform import Platform, PlatformType
from hyper_parallel.platform.mindspore.dtensor import DTensorBase
from hyper_parallel.platform.mindspore.pipeline_parallel.stage import PipelineStageBase
from hyper_parallel.platform.mindspore.parameter_init import init_parameters as _init_parameters

_tensor_transform = TensorTransform.get_instance()


# pylint: disable=C0103


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

    def device_count(self, device_handle):
        device_type = self.device_type()
        if device_type == "cpu":
            return device_handle.device_context.cpu.device_count()
        if device_type == "gpu":
            return device_handle.device_context.gpu.device_count()
        return device_handle.device_context.ascend.device_count()

    @staticmethod
    def get_rng_state(device=None, device_handle=None):
        """Get RNG state """
        _ = device, device_handle
        return ms.get_rng_state()

    @staticmethod
    def set_rng_state(state, device=None, device_handle=None):
        _ = device, device_handle
        return ms.set_rng_state(state)

    def device_type(self):
        device_type = ms.get_context("device_target")
        if device_type == "Ascend":
            return "npu"
        return device_type.lower()

    def device(self, device_idx=None):
        _ = device_idx
        device_type = self.device_type()
        return device_type

    @staticmethod
    def get_device_handle():
        return ms

    @staticmethod
    def manual_seed(seed):
        return ms.manual_seed(seed)

    @staticmethod
    def ones(size, dtype=None):
        return mint.ones(size, dtype=dtype)

    @staticmethod
    def zeros(size, dtype=None):
        return mint.zeros(size, dtype=dtype)

    @staticmethod
    def full(size, fill_value, dtype=None):
        return mint.full(size, fill_value, dtype=dtype)

    @staticmethod
    def empty(size, dtype=None):
        return mint.empty(size, dtype=dtype)

    @staticmethod
    def get_rank():
        return get_rank_id()

    @staticmethod
    def get_global_rank(group, group_rank):
        return dist.get_global_rank(group, group_rank)

    @staticmethod
    def get_world_size():
        return get_group_size()

    @staticmethod
    def get_op_name(func):
        return func.name

    @staticmethod
    def differentiable_all_gather_concat(data, group, concat_size, concat_dim):
        output, _ = comm_func.all_gather_into_tensor(data, group=group)
        if concat_dim == 0:
            return output
        output_tensors = ms.ops.Split(output_num=concat_size)(output)
        return ms.mint.concat(output_tensors, concat_dim)

    @staticmethod
    def chunk(data, split_dim, split_size, index):
        return ms.ops.Split(axis=split_dim, output_num=split_size)(data)[index]

    @staticmethod
    def differentiable_all_to_all(input_data, output_shape, group):
        output_tensor, _ = comm_func.all_to_all_single_with_output_shape(
            output_shape=output_shape,
            tensor=input_data,
            group=group,
            async_op=False
        )
        return output_tensor

    @staticmethod
    def differentiable_all_reduce(data, op, group):
        output, _ = comm_func.all_reduce(data, op, group)
        return output

    @staticmethod
    def differentiable_reduce_scatter(data, dev_num, axis, op, group):
        if axis > 0:
            data = ms.mint.concat(ms.ops.Split(axis=axis, output_num=dev_num)(data), dim=0)
        output_tensor, _ = comm_func.reduce_scatter_tensor(data, 'sum', group)
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
        from hyper_parallel.core.dtensor import DTensor  # pylint: disable=import-outside-toplevel
        from hyper_parallel.core.layout import _infer_slice_shape_by_layout, \
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
                _get_slice_tensor_by_layout(param, layout).value(), layout.mesh, layout.placements
                )
            param = Parameter(param_dtensor, name=name, requires_grad=requires_grad)
            param.param_info = param_info
        else:
            # has not been init, need to modify init shape
            param.init_mode.shape = slice_shape
            param_dtensor = DTensor.from_local(param.init_mode, layout.mesh, layout.placements)
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

    def _create_group(self, rank_list, group_name=None):
        if group_name is None:
            hash_str_rank_list = '-'.join([str(rank) for rank in rank_list])
            group_name = f"{len(rank_list)}-{hash_str_rank_list}"
        new_group(rank_ids=rank_list, group=group_name)
        return group_name

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        return comm_func.all_gather_into_tensor(data, group=group_info.group_name, async_op=async_op)

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
        return comm_func.reduce_scatter_tensor(data, group=group_info.group_name, async_op=async_op)

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
    def save_checkpoint(cell: Cell, file_path: str) -> None:
        save_dict = cell._params
        ms.save_checkpoint(save_obj=save_dict, ckpt_file_name=file_path, format="safetensors")

    @staticmethod
    def load_checkpoint(file_path: str) -> dict:
        return ms.load_checkpoint(ckpt_file_name=file_path, format="safetensors")

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
                if pg_options is None:
                    hash_str_rank_list = '-'.join([str(rank) for rank in split_rank])
                    pg_options = f"{len(split_rank)}-{hash_str_rank_list}"
                new_group(rank_ids=split_rank, group=pg_options)
                return pg_options
        raise ValueError(f"The Split_ranks {split_ranks} does not contain current rank {rank_id}")

    @staticmethod
    def no_grad():
        return _no_grad()

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
        return module.register_forward_pre_hook(hook, with_kwargs)

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
        raise NotImplementedError("ckpt_wrapper is not supported on MindSpore platform")

    @property
    def noop_context_fn(self):
        raise NotImplementedError("noop_context_fn is not supported on MindSpore platform")

    @staticmethod
    def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
        raise NotImplementedError("create_selective_checkpoint_contexts is not supported on MindSpore platform")

    @staticmethod
    def async_save_on_cpu(policy_fn=None):
        raise NotImplementedError("async_save_on_cpu is not supported on MindSpore platform")

    @staticmethod
    def tensor_to_numpy(tensor) -> np.ndarray:
        """Convert MindSpore tensor to numpy array."""
        return tensor.asnumpy()
