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
"""Torch platform api"""
from datetime import timedelta
from typing import Optional, Any, Union
import dataclasses
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch._C._distributed_c10d import Store, ProcessGroup
from torch.distributed import Backend
from torch.distributed.distributed_c10d import _get_default_group
from torch.nn import Parameter, Module
from torch.nn.utils.rnn import PackedSequence
from torch._ops import OpOverload, OpOverloadPacket
from torch.utils.checkpoint import noop_context_fn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
import torch.distributed.nn.functional as dist_func
import torch.distributed as dist
from hyper_parallel.platform.torch.dtensor import DTensorBase
from hyper_parallel.platform.torch.pipeline_parallel.stage import PipelineStageBase
from hyper_parallel.platform.torch.group_utils import create_sub_groups
from hyper_parallel.platform.platform import Platform, PlatformType
from hyper_parallel.platform.torch.function_override import override_functions

override_functions()

# Mapping from string op names to torch.distributed.ReduceOp
_OP_MAP = {
    'sum': dist.ReduceOp.SUM,
    'prod': dist.ReduceOp.PRODUCT,
    'max': dist.ReduceOp.MAX,
    'min': dist.ReduceOp.MIN,
    # convert tensor elements to int32 and use MIN
    'all': dist.ReduceOp.MIN,
    # 'avg' is typically handled by SUM followed by division in current implementation logic
    'avg': dist.ReduceOp.SUM,
}

# Try to add AVG for 'mean' if supported by current torch version
if hasattr(dist.ReduceOp, "AVG"):
    _OP_MAP['mean'] = dist.ReduceOp.AVG
else:
    # Fallback for older torch versions if necessary, though this might require manual division upstream
    # Assuming standard behavior where 'mean' implies native AVG support or upstream handling
    _OP_MAP['mean'] = dist.ReduceOp.SUM


# pylint: disable=C0103
class TorchPlatform(Platform):
    """Torch platform api"""
    Tensor = Tensor
    Parameter = Parameter
    Module = Module
    DTensorBase = DTensorBase
    PipelineStageBase = PipelineStageBase
    platform_type = PlatformType.PYTORCH
    tensor_dtype = torch

    @staticmethod
    def ones(size, dtype=None):
        return torch.ones(size, dtype=dtype)

    @staticmethod
    def zeros(size, dtype=None):
        return torch.zeros(size, dtype=dtype)

    @staticmethod
    def full(size, fill_value, dtype=None):
        return torch.full(size, fill_value, dtype=dtype)

    @staticmethod
    def empty(size, dtype=None):
        return torch.empty(size, dtype=dtype)

    @staticmethod
    def get_rank():
        return dist.get_rank()

    @staticmethod
    def get_global_rank(group, group_rank):
        return dist.get_global_rank(group, group_rank)

    @staticmethod
    def get_world_size():
        return dist.get_world_size()

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
    def update_param_data(param, data):
        """update param data"""
        param.data = data

    @staticmethod
    def get_op_name(func):
        if hasattr(func, "__name__"):
            return func.__name__
        if isinstance(func, OpOverload):
            full_name = func.name
            core_name = full_name.split("::")[-1].split(".")[0]
            return core_name
        if isinstance(func, OpOverloadPacket):
            return func.name.split("::")[-1]
        func_str = str(func)
        if "built-in function" in func_str:
            return func_str.split()[-1].strip(">")
        if "function" in func_str:
            return func_str.split()[1]
        return "unknown_op"

    @staticmethod
    def differentiable_all_gather_concat(data, group, concat_size, concat_dim):
        output = dist_func.all_gather(data, group=group)
        return torch.cat(output, dim=concat_dim)

    @staticmethod
    def chunk(data, split_dim, split_size, index):
        return torch.chunk(data, split_size, dim=split_dim)[index]

    @staticmethod
    def differentiable_all_to_all(input_data, output_shape, group):
        output_tensor = torch.empty(output_shape, device=input_data.device, dtype=input_data.dtype)
        output_tensor = dist_func.all_to_all_single(
            output_tensor,
            input_data,
            group=group
        )
        return output_tensor

    @staticmethod
    def tensor_type_cast(input_data, cast_type):
        """Cast tensor to specified data type."""
        type_mapping = {
            'float32': torch.float32,
            'float16': torch.float16,
            'int64': torch.int64,
            'int32': torch.int32
        }
        if cast_type not in type_mapping:
            raise ValueError(f"Unknown cast type: {cast_type}. Supported types: {list(type_mapping.keys())}")
        return input_data.to(type_mapping[cast_type])

    @staticmethod
    def differentiable_all_reduce(data, op, group):
        # Resolve the op from string to ReduceOp enum if necessary
        reduce_op = _OP_MAP.get(op, dist.ReduceOp.SUM) if isinstance(op, str) else op
        return dist_func.all_reduce(data, op=reduce_op, group=group)

    @staticmethod
    def get_cell_construct(cell):
        return cell.forward

    @staticmethod
    def get_cells_and_names(cell):
        return cell.named_modules()

    @staticmethod
    def search_parameter_by_name(cell, param_name: str):
        """
        Find the parent Module of the parameter, the parameter's name in the parent Module, and the parameter.
        Return value: (parent Module instance, parameter's name in parent Module, parameter object).
        Returns None if not found.
        """
        # Remove the "self." prefix from param_name
        param_name = param_name.replace("self.", "")
        # Case 1: The parameter is a direct parameter of the current Module
        if param_name in cell._parameters:  # pylint:disable=protected-access
            return (cell, param_name, cell._parameters[param_name])  # pylint:disable=protected-access

        # Case 2: The parameter is in a sub-Module
        if "." in param_name:
            cell_path, param_key = param_name.rsplit(".", 1)
            try:
                # Locate the sub-Module where the parameter resides (supports multi-level paths)
                target_cell = cell.get_submodule(cell_path)
                # Check if the sub-Module directly contains this parameter
                if param_key in target_cell._parameters:  # pylint:disable=protected-access
                    return target_cell, param_key, target_cell._parameters[param_key]  # pylint:disable=protected-access
            except AttributeError:
                pass

        # Traverse all sub-Modules (recursively) to search for the parameter
        for _, child_cell in cell.named_children():
            if isinstance(child_cell, Module):
                result = TorchPlatform.search_parameter_by_name(child_cell, param_name)
                if result is not None:
                    return result

        return None

    @staticmethod
    def update_parameter_by_name(cell, result: tuple, new_param) -> bool:
        """
        Modify the original parameter in a Module or sub-Module using the search result
        """
        parent_cell, param_key, _ = result
        # Key operation: directly modify the _parameters dictionary.
        if param_key in parent_cell._parameters:  # pylint:disable=protected-access
            parent_cell._parameters[param_key] = new_param  # pylint:disable=protected-access
        else:
            parent_cell.register_parameter(param_key, new_param)
        return True

    @staticmethod
    def set_layout_into_parameter(param, layout):
        """Set layout in to parameter"""
        from hyper_parallel.core.dtensor import DTensor  # pylint: disable=import-outside-toplevel
        from hyper_parallel.core.layout import _get_slice_tensor_by_layout  # pylint: disable=import-outside-toplevel
        if isinstance(param, DTensor):
            raise ValueError(f"Parameter {param} has been configured layout, cannot be set repeatedly.")
        requires_grad = param.requires_grad
        param_dtensor = DTensor.from_local(_get_slice_tensor_by_layout(param, layout), layout.mesh, layout.placements)
        new_param = Parameter(param_dtensor, requires_grad=requires_grad)
        return new_param

    @staticmethod
    def differentiable_reduce_scatter(data, dev_num, axis, op, group):
        input_tuple = torch.chunk(data, dev_num, dim=axis)
        output_tensor = torch.empty(input_tuple[0].shape, device=data.device, dtype=data.dtype)

        # Resolve the op from string to ReduceOp enum
        reduce_op = _OP_MAP.get(op, dist.ReduceOp.SUM) if isinstance(op, str) else op

        output_tensor = dist_func.reduce_scatter(output_tensor, input_tuple, op=reduce_op, group=group)

        # Keep manual handling for 'avg' string as it maps to SUM in _OP_MAP
        if op == 'avg':
            output_tensor = output_tensor / dev_num
        return output_tensor

    @staticmethod
    def get_device_handle():
        if hasattr(torch, "npu"):
            return torch.npu
        return torch.cuda

    @staticmethod
    def get_param_type_size(param):
        # pylint: disable=W0212
        return torch._utils._element_size(param.dtype)

    @staticmethod
    def parameters_dict(cell: Module):
        return cell.named_parameters()

    @staticmethod
    def save_checkpoint(cell: Module, file_path: str) -> None:
        torch.save(obj=cell, f=file_path)

    @staticmethod
    def load_checkpoint(file_path: str) -> dict:
        return torch.load(f=file_path)

    @staticmethod
    def new_zero_parameter(param_shape, param_type, requires_grad, device):
        return nn.Parameter(torch.zeros(param_shape, dtype=param_type, device=device), requires_grad=requires_grad)

    @staticmethod
    def new_tensor(tensor_shape, tensor_type, device):
        return torch.empty(size=tensor_shape, dtype=tensor_type, device=device)

    @staticmethod
    def full_like(tensor, fill_value, dtype=None):
        return torch.full_like(tensor, fill_value, dtype=dtype)

    @staticmethod
    def set_tensor_requires_grad(input_tensor):
        """
        set requires grad flag for input tensor, only effective for leaf node
        """
        if input_tensor.is_leaf:
            input_tensor.requires_grad = True

    def _create_group(self, rank_list, group_name=None):
        group_dict = create_sub_groups(rank_list)
        return group_dict[tuple(rank_list)]

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        output_shape = list(data.shape)
        output_shape[0] = output_shape[0] * group_info.rank_size
        output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
        handle = dist.all_gather_into_tensor(output, data, group=group_info.group, async_op=async_op)
        return output, handle

    @staticmethod
    def all_reduce(data, group_info, async_op=False):
        if not data.is_contiguous():
            data = data.contiguous()
        handle = dist.all_reduce(data, group=group_info.group, async_op=async_op)
        return data, handle

    @staticmethod
    def broadcast(data, src, group, async_op=False):
        handle = dist.broadcast(data, src, group, async_op)
        if async_op:
            handle.wait()

    @staticmethod
    def isend(tensor, dst=None, group=None, tag=0):
        return dist.isend(tensor, dst, group, tag)

    @staticmethod
    def irecv(tensor, src=None, group=None, tag=0):
        return dist.irecv(tensor, src, group, tag)

    @staticmethod
    def send_object_list(obj_list, dst=None, group=None):
        dist.send_object_list(obj_list, dst, group)

    @staticmethod
    def recv_object_list(obj_list, src=None, group=None):
        dist.recv_object_list(obj_list, src, group)

    @staticmethod
    def reduce_scatter_tensor(data, group_info, async_op=False):
        output_shape = list(data.shape)
        output_shape[0] = output_shape[0] // group_info.rank_size
        output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
        handle = dist.reduce_scatter_tensor(output, data, group=group_info.group, async_op=async_op)
        return output, handle

    @staticmethod
    def get_tensor_transform():
        raise NotImplementedError("Unsupported get_tensor_transform for torch platform")

    @staticmethod
    def construct_strided_slice(x, begin, end, stride):
        raise NotImplementedError("Unsupported construct_strided_slice for torch platform")

    @staticmethod
    def micro_batch(micro_batch_num, args_batch_dim=None, kwargs_batch_dim=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.pipeline_parallel._utils import _MicroBatch
        return _MicroBatch(micro_batch_num, args_batch_dim, kwargs_batch_dim)

    def new_stream(self):
        device = self.get_device_handle()
        return device.Stream()

    def get_stream_context(self):
        device = self.get_device_handle()
        return device.stream

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
            backend: Optional[str] = None,
            *,
            init_method: Optional[str] = None,
            timeout: Optional[timedelta] = None,
            world_size: int = -1,
            rank: int = -1,
            store: Optional[Store] = None,
            pg_options: Optional[Any] = None,
            device_id: Optional[Union[torch.device, int]] = None,
    ) -> None:
        """
        Initialize global process group.

        Args:
            backend (str or Backend, optional): The backend to use for distributed communication. Valid values
                are ``nccl``, ``gloo``, ``mpi``, ``ucc``, ``xccl``.
            init_method (str, optional): URL specifying how to initialize the process group. Default is "env://",
                can not be specified at the same time with ``store``.
            timeout (timedelta, optional): Timeout for process group. Default 10 minutes for NCCL and for other
                backends 30 minutes.
            world_size (int, optional): Number of processes. If ``store`` is specified, world_size is required.
            rank (int, optional): Rank of the current process, which value must between 0 and ``world_size``-1. If
                ``store`` is specified, rank is required.
            store (Store, optional): Key/value store accessible to all workers, used to exchange connection/address
                information. Can not be specified at the same time with ``init_method``.
            pg_options (ProcessGroupOptions, optional): Extra options to pass during constructing process groups.
            device_id (torch.device | int, optional): Specific device this process will work on.
        """
        try:
            _get_default_group()
        except ValueError:
            dist.init_process_group(backend=backend, init_method=init_method, timeout=timeout, world_size=world_size,
                                    rank=rank, store=store, pg_options=pg_options, device_id=device_id)

    @staticmethod
    def destroy_process_group(group: Optional[ProcessGroup] = None) -> None:
        """
        Destroy given process group.

        Args:
            group (ProcessGroup, optional): Given process group will be destroyed, if not given, all process groups
                will be destroyed.
        """
        group = group or _get_default_group()
        dist.destroy_process_group(group)

    @staticmethod
    def get_process_group_ranks(group: Optional[ProcessGroup] = None) -> list[int]:
        """
        Get all ranks relative to given process group.

        Args:
            group (Optional[ProcessGroup]): Process group worked on. Default is ``None``, and ``None`` means global
                group.

        Returns:
            Rank list.
        """
        group = group or _get_default_group()
        return dist.get_process_group_ranks(group)

    @staticmethod
    def get_backend(group: Optional[ProcessGroup] = None) -> Backend:
        """
        Get the backend of the given process group.

        Args:
            group (ProcessGroup, optional): Process group worked on. Default is ``None``, and ``None`` means global
                group.

        Returns:
            The backend object of the given process group.
        """
        group = group or _get_default_group()
        return dist.get_backend(group)

    @staticmethod
    def split_group(parent_pg: Optional[ProcessGroup] = None,
                    split_ranks: Optional[list] = None,
                    timeout: Optional[timedelta] = None,
                    pg_options: Optional[Any] = None,
                    group_desc: Optional[str] = None,
                    ) -> Optional[ProcessGroup]:
        """
        Create split groups for every group rank in split_ranks, and return the split process group which relative to
            current rank id.

        Args:
            parent_pg (Optional[ProcessGroup]): A process group which the goal group split from.
            split_ranks (Optional[list]): A list like ``list[list[int]]``.
            timeout (Optional[timedelta]): Timeout for process group. Default 10 minutes for NCCL and for other
                backend 30 minutes.
            pg_options (Optional[Any]): Extra options to pass during constructing process groups.
            group_desc (Optional[str]): Description of process group.

        Return:
            Optional[ProcessGroup]: One of split process group which relative to current rank id
        """
        if split_ranks is None or len(split_ranks) == 0:
            raise ValueError("split_ranks cannot be None or empty")

        split_group = None
        for split_rank in split_ranks:
            dist_group = dist.new_group(split_rank)
            if TorchPlatform.get_rank() in split_rank:
                split_group = dist_group

        return split_group

    @staticmethod
    def no_grad():
        return torch.no_grad()

    @staticmethod
    def empty_like(tensor, *, dtype=None, device=None, pin_memory=False):
        return torch.empty_like(tensor, dtype=dtype, device=device, pin_memory=pin_memory)

    def get_current_stream(self):
        device = self.get_device_handle()
        return device.current_stream()

    def new_event(self):
        device = self.get_device_handle()
        return device.Event()

    def tree_map(self, fn, tree):
        return torch.utils._pytree.tree_map(fn, tree)  # pylint:disable=protected-access

    @property
    def checkpoint(self):
        return torch.utils.checkpoint.checkpoint

    @staticmethod
    def ckpt_wrapper(module, checkpoint_fn=None, **checkpoint_fn_kwargs):
        return checkpoint_wrapper(module, checkpoint_fn=checkpoint_fn, **checkpoint_fn_kwargs)

    @property
    def noop_context_fn(self):
        return noop_context_fn

    @staticmethod
    def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.sac import create_selective_checkpoint_contexts
        return create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation)

    @staticmethod
    def async_save_on_cpu(policy_fn=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.activation_swap import AsyncSaveOnCpu
        return AsyncSaveOnCpu(policy_fn)

    @staticmethod
    def tensor_to_numpy(tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        return tensor.cpu().numpy()

    def cast_fp_tensor(self,dtype, x):
        """
        Cast floating-point tensor to target dtype if applicable.
        """
        if (
            not isinstance(x, torch.Tensor)
            or not torch.is_floating_point(x)
            or x.dtype == dtype
        ):
            return x
        return x.to(dtype)

    def apply_to_tensors(self, fn, container):
        """Recursively apply to all tensor in different kinds of container types."""

        def apply(x):

            if isinstance(x, torch.Tensor):
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
            if isinstance(x, PackedSequence):
                apply(x.data)
                return x
            if isinstance(x, dict):
                return {key: apply(value) for key, value in x.items()}
            if isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields"):
                res = (apply(el) for el in x)
                return type(x)(*res)
            if isinstance(x, (list, tuple, set)):
                return type(x)(apply(el) for el in x)
            return x

        return apply(container)
