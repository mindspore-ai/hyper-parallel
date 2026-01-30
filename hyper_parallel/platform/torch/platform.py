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
import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter, Module
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
        param_dtensor = DTensor.from_local(_get_slice_tensor_by_layout(param, layout), layout)
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
        Gathers picklable objects from the whole group into a list.

        Args:
            object_list (list[Any]): Output list. It should be correctly sized as the
                size of the group for this collective and will contain the output.
            obj (Any): Pickable Python object to be broadcast from current process.
            group (ProcessGroup, optional): The process group to work on. If None,
                the default process group will be used. Default is ``None``.

        Returns:
            None. If the calling rank is part of this group, the output of the
            collective will be populated into the input ``object_list``. If the
            calling rank is not part of the group, the passed in ``object_list`` will
            be unmodified.
        """
        dist.all_gather_object(object_list, obj, group)

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
