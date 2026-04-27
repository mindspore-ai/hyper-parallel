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
"""Torch platform api"""
from datetime import timedelta
from typing import Optional, Any, Union
import dataclasses
from collections import OrderedDict

import numpy as np
from safetensors.torch import save_file, load_file
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
from hyper_parallel.platform.platform import Platform, PlatformType, EXISTING_COMM_GROUPS
from hyper_parallel.platform.torch.function_override import override_functions
from hyper_parallel.platform.torch.init_weights import init_on_device as _init_on_device

override_functions()


# ---------------------------------------------------------------------------
# Module-level A2A reshape helpers
# ---------------------------------------------------------------------------

def _a2a_reconstruct(out_perm: torch.Tensor, concat_dim: int) -> torch.Tensor:
    """Reconstruct A2A result from raw out_perm buffer.

    ``out_perm`` has shape ``[ws, *rest_dims]``, chunk at ``concat_dim + 1``.
    Returns tensor with merged chunk dimension.
    """
    new_ndim = out_perm.dim()
    chunk_in_perm = concat_dim + 1
    recon_perm = list(range(1, chunk_in_perm)) + [0] + list(range(chunk_in_perm, new_ndim))
    x_recon = out_perm.permute(recon_perm).contiguous()
    shape = list(x_recon.shape)
    merged = shape[concat_dim] * shape[concat_dim + 1]
    return x_recon.reshape(shape[:concat_dim] + [merged] + shape[concat_dim + 2:])


class _TorchAsyncA2AFunction(torch.autograd.Function):
    """Differentiable wrapper for pre-launched async all-to-all.

    Forward: wait async handle, reconstruct A2A result.
    Backward: launch async head→seq A2A and store handle in ``handle_box``
    for the projection pre-hook to wait, achieving GEMM–A2A overlap.
    """

    @staticmethod
    def forward(ctx, x, work, out_perm, group, world_size, concat_dim, split_dim,  # pylint: disable=arguments-differ
                handle_box):
        """Wait for pre-launched async A2A and return reconstructed output."""
        ctx.group = group
        ctx.world_size = world_size
        ctx.concat_dim = concat_dim
        ctx.split_dim = split_dim
        ctx.handle_box = handle_box
        ctx.x_shape = x.shape
        work.wait()
        return _a2a_reconstruct(out_perm, concat_dim)

    @staticmethod
    def backward(ctx, grad_output):
        """Launch async head→seq A2A for backward overlap, or return zero grad."""
        if ctx.handle_box is not None:
            # Launch async head→seq A2A (reverse of forward seq→head)
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
            out_perm = torch.empty_like(x_perm)
            work = dist.all_to_all_single(out_perm, x_perm, group=ctx.group, async_op=True)
            ctx.handle_box.append((work, out_perm))
        return grad_output.new_zeros(ctx.x_shape), None, None, None, None, None, None, None


class _TorchP2PExchangeFunction(torch.autograd.Function):
    """Symmetric bidirectional P2P: send local tensor to peer, receive peer's tensor."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, peer_rank: int, group) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Perform symmetric bidirectional P2P exchange with peer_rank."""
        ctx.peer_rank = peer_rank
        ctx.group = group
        send_buf = tensor.contiguous()
        recv_buf = torch.empty_like(send_buf)
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, peer_rank, group),
            dist.P2POp(dist.irecv, recv_buf, peer_rank, group),
        ])
        for req in reqs:
            req.wait()
        return recv_buf

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Perform symmetric P2P exchange for the backward gradient pass."""
        send_buf = grad_output.contiguous()
        recv_buf = torch.empty_like(send_buf)
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, ctx.peer_rank, ctx.group),
            dist.P2POp(dist.irecv, recv_buf, ctx.peer_rank, ctx.group),
        ])
        for req in reqs:
            req.wait()
        return recv_buf, None, None


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
    tensor = torch.tensor
    Parameter = Parameter
    Module = Module
    DTensorBase = DTensorBase
    PipelineStageBase = PipelineStageBase
    platform_type = PlatformType.PYTORCH
    tensor_dtype = torch
    dtype = torch.dtype
    Function = torch.autograd.Function

    @staticmethod
    def apply_runtime_patches_if_needed():
        """Torch currently does not require extra runtime patches."""
        return None

    @staticmethod
    def is_linear_module(module) -> bool:
        """Check whether *module* is a ``torch.nn.Linear`` instance."""
        return isinstance(module, nn.Linear)

    @staticmethod
    def is_embedding_module(module) -> bool:
        """Check whether *module* is a ``torch.nn.Embedding`` instance."""
        return isinstance(module, nn.Embedding)

    @staticmethod
    def device_count(device_handle):
        """
        Get the number of available devices.

        Args:
            device_handle: The device handle (e.g., torch.cuda, torch.npu).

        Returns:
            int: The number of available devices.
        """
        return device_handle.device_count()

    def device_type(self):
        """
        Get the current device type.

        Returns:
            str: The device type string ("npu" for NPU, "cuda" for GPU).
        """
        device_handle = self.get_device_handle()
        if device_handle == torch.npu:
            return "npu"
        return "cuda"

    def device(self, device_idx=None):
        """
        Get a torch.device object for the specified device index.

        Args:
            device_idx (Optional[int]): The device index. If None, returns device without index.

        Returns:
            torch.device: A torch device object.
        """
        device_type = self.device_type()
        if device_idx is None:
            return torch.device(device_type)
        return torch.device(f"{device_type}:{device_idx:d}")

    @staticmethod
    def get_rng_state(device=None, device_handle=None):
        """
        Get the random number generator state.

        Args:
            device (Optional): The device to get RNG state from.
            device_handle (Optional): The device handle (torch.cuda, torch.npu, etc.).

        Returns:
            Tensor: The RNG state as a byte tensor.
        """
        if device_handle is None:
            return torch.get_rng_state()
        if device is None:
            return device_handle.get_rng_state()
        return device_handle.get_rng_state(device)

    @staticmethod
    def set_rng_state(state, device=None, device_handle=None):
        """
        Set the random number generator state.

        Args:
            state (Tensor): The RNG state to set.
            device (Optional): The device to set RNG state for.
            device_handle (Optional): The device handle (torch.cuda, torch.npu, etc.).
        """
        if device_handle is None:
            return torch.set_rng_state(state)
        if device is None:
            return device_handle.set_rng_state(state)
        return device_handle.set_rng_state(state, device)

    @staticmethod
    def manual_seed(seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The random seed value.

        Returns:
            torch.Generator: The random number generator.
        """
        return torch.manual_seed(seed)

    @staticmethod
    def ones(size, dtype=None):
        """
        Create a tensor filled with ones.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[torch.dtype]): The desired data type.

        Returns:
            Tensor: A tensor filled with ones.
        """
        return torch.ones(size, dtype=dtype)

    @staticmethod
    def zeros(size, dtype=None, device=None):
        """
        Create a tensor filled with zeros.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[torch.dtype]): The desired data type.
            device (Optional[torch.device]): The device to create the tensor on.

        Returns:
            Tensor: A tensor filled with zeros.
        """
        return torch.zeros(size, dtype=dtype, device=device)

    @staticmethod
    def full(size, fill_value, dtype=None):
        """
        Create a tensor filled with a scalar value.

        Args:
            size (tuple): The shape of the output tensor.
            fill_value (scalar): The value to fill the tensor with.
            dtype (Optional[torch.dtype]): The desired data type.

        Returns:
            Tensor: A tensor filled with the specified value.
        """
        return torch.full(size, fill_value, dtype=dtype)

    @staticmethod
    def empty(size, dtype=None):
        """
        Create an uninitialized tensor.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[torch.dtype]): The desired data type.

        Returns:
            Tensor: An uninitialized tensor.
        """
        return torch.empty(size, dtype=dtype)

    @staticmethod
    def get_rank():
        """
        Get the rank of the current process in the distributed group.

        Returns:
            int: The rank of the current process.
        """
        return dist.get_rank()

    @staticmethod
    def get_global_rank(group, group_rank):
        """
        Get the global rank from a group rank.

        Args:
            group (ProcessGroup): The process group.
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
        return dist.get_world_size()

    @staticmethod
    def get_param_local_shape(param):
        """
        Get the local shape of a parameter, handling both regular and distributed tensors.

        Args:
            param (Union[Tensor, DTensorBase]): The parameter tensor.

        Returns:
            torch.Size: The local shape of the parameter.
        """
        if isinstance(param, DTensorBase):
            return param.local_shape
        return param.shape

    @staticmethod
    def get_param_local_data(param):
        """
        Get the local data of a parameter, handling both regular and distributed tensors.

        Args:
            param (Union[Tensor, DTensorBase]): The parameter tensor.

        Returns:
            Tensor: The local tensor data.
        """
        if isinstance(param, DTensorBase):
            return param.to_local()
        return param

    @staticmethod
    def update_param_data(param, data):
        """
        Update the data of a parameter.

        Args:
            param (Parameter): The parameter to update.
            data (Tensor): The new data tensor.
        """
        param.data = data

    @staticmethod
    def load_into_param(param, data):
        """Load tensor *data* into *param* (plain tensor or DTensor)."""
        if isinstance(param, DTensorBase):
            local = param._local_tensor # pylint: disable=W0212
            if local.is_meta:
                # Meta tensor materialisation: replace the placeholder.
                orig_requires_grad = param.requires_grad
                param._local_tensor = data # pylint: disable=W0212
                if data.requires_grad != orig_requires_grad:
                    param.requires_grad_(orig_requires_grad)
            else:
                local.copy_(data)
        else:
            param.copy_(data)

    @staticmethod
    def get_op_name(func):
        """
        Extract the operation name from various function types.

        Args:
            func: The function or operation to extract the name from.

        Returns:
            str: The operation name.
        """
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
        if param_name in cell._parameters:  # pylint: disable=protected-access
            return (cell, param_name, cell._parameters[param_name])  # pylint: disable=protected-access

        # Case 2: The parameter is in a sub-Module
        if "." in param_name:
            cell_path, param_key = param_name.rsplit(".", 1)
            try:
                # Locate the sub-Module where the parameter resides (supports multi-level paths)
                target_cell = cell.get_submodule(cell_path)
                # Check if the sub-Module directly contains this parameter
                if param_key in target_cell._parameters:  # pylint: disable=protected-access
                    return target_cell, param_key, target_cell._parameters[param_key]  # pylint: disable=protected-access
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
        if param_key in parent_cell._parameters:  # pylint: disable=protected-access
            parent_cell._parameters[param_key] = new_param  # pylint: disable=protected-access
        else:
            parent_cell.register_parameter(param_key, new_param)
        return True

    @staticmethod
    def set_layout_into_parameter(param, layout):
        """Set layout into parameter"""
        from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=import-outside-toplevel
        from hyper_parallel.core.dtensor.layout import _get_slice_tensor_by_layout  # pylint: disable=import-outside-toplevel
        if isinstance(param, DTensor):
            raise ValueError(f"Parameter {param} has been configured layout, cannot be set repeatedly.")
        requires_grad = param.requires_grad
        param_dtensor = DTensor.from_local(
            _get_slice_tensor_by_layout(param, layout),
            layout.mesh, layout.alias_placements)
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
    def get_device_handle(device_type: str = "npu"):
        try:
            handle =  getattr(torch, device_type)
        except AttributeError as e:
            raise RuntimeError(f"TorchPlatform expect got device handle: 'torch.{device_type}' failed.") from e
        return handle

    @staticmethod
    def get_param_type_size(param):
        # pylint: disable=W0212
        return torch._utils._element_size(param.dtype)

    @staticmethod
    def is_tensor(obj: Any) -> bool:
        """Return True if ``obj`` is a ``torch.Tensor``."""
        return isinstance(obj, Tensor)

    @staticmethod
    def get_tensor_storage_size(tensor: Any) -> int:
        """Return serialized byte size (numel * element size) for a PyTorch tensor."""
        if not TorchPlatform.is_tensor(tensor):
            raise TypeError(
                f"TorchPlatform.get_tensor_storage_size expects torch.Tensor, got {type(tensor)!r}"
            )
        return int(tensor.numel()) * int(tensor.element_size())

    @staticmethod
    def parameters_dict(cell: Module):
        return cell.named_parameters()

    @staticmethod
    def get_model_state_dict(model, *, options=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.fully_shard.state_dict_utils import (
            get_model_state_dict as _get_model_state_dict,
        )
        return _get_model_state_dict(model, options=options)

    @staticmethod
    def save_checkpoint(cell: Module, file_path: str, ckpt_format: str = "safetensors") -> None:
        if ckpt_format == "safetensors":
            save_file(tensors=cell, filename=file_path)
        else:
            torch.save(obj=cell, f=file_path)

    @staticmethod
    def load_checkpoint(file_path: str, ckpt_format: str = "safetensors") -> dict:
        if ckpt_format == "safetensors":
            return load_file(filename=file_path)
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

    def _create_group(self, rank_list):
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
    def broadcast(data, src, group=None, async_op=False):
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
    def p2p_exchange(tensor, peer_rank: int, group=None):
        if peer_rank == dist.get_rank(group):
            return tensor
        return _TorchP2PExchangeFunction.apply(tensor, peer_rank, group)

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
    def all_to_all_single(input_tensor, output_shape, group, async_op=False):
        output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
        work = dist.all_to_all_single(output, input_tensor, group=group, async_op=async_op)
        return output, work

    @staticmethod
    def differentiable_all_to_all_single(input_tensor, input_splits, output_splits, group):
        """Variable-split all-to-all with autograd support for EP token dispatch/combine."""
        out_total = sum(output_splits)
        output = torch.empty(
            out_total, *input_tensor.shape[1:],
            dtype=input_tensor.dtype, device=input_tensor.device,
        )
        output = dist_func.all_to_all_single(
            output, input_tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        return output

    @staticmethod
    def arange(start, end=None, step=1, dtype=None, device=None):
        """Create a 1-D tensor with evenly spaced values."""
        if end is None:
            return torch.arange(start, dtype=dtype, device=device)
        return torch.arange(start, end, step, dtype=dtype, device=device)

    @staticmethod
    def differentiable_async_a2a_wait(x, work, out_perm, group, world_size, concat_dim, split_dim,
                                      handle_box=None):
        """Wait async A2A handle and reconstruct result (differentiable).

        Args:
            x: Input tensor.
            work: Async work handle from all_to_all.
            out_perm: Output buffer from all_to_all.
            group: Process group.
            world_size: World size.
            concat_dim: Dimension for concatenation.
            split_dim: Dimension for split.
            handle_box: Optional mutable list; backward appends (work, out_perm) here.
        """
        return _TorchAsyncA2AFunction.apply(
            x, work, out_perm, group, world_size, concat_dim, split_dim, handle_box
        )

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

    @staticmethod
    def get_symmetric_memory_handler():
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.symmetric_memory import TorchSymmetricMemoryHandler
        symmetric_memory = TorchSymmetricMemoryHandler()
        return symmetric_memory

    @staticmethod
    def get_multicore_handler():
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.multicore import TorchMulticoreHandler
        return TorchMulticoreHandler()

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
    def barrier(group=None, async_op: bool = False, device_ids=None) -> Any:
        """
        Synchronize all processes in the given process group.

        Args:
            group (ProcessGroup, optional): The process group to work on. Default is ``None``,
                meaning the default process group.
            async_op (bool, optional): Whether this op should be asynchronous. Default: ``False``.
            device_ids (list[int], optional): Device ids for backends that require a device for
                barrier (e.g. NCCL). Default: ``None``.

        Returns:
            Async work handle if ``async_op`` is True; otherwise ``None``.
        """
        return dist.barrier(group, async_op, device_ids)

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
            backend (str or Backend, optional): The backend to use for distributed communication.
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
        # except multi version error
        except (ValueError, RuntimeError):
            if backend is None:
                backend = "hccl"
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
        if group in EXISTING_COMM_GROUPS.values():
            keys_to_destroy = [k for k, v in EXISTING_COMM_GROUPS.items() if v == group]
            for k in keys_to_destroy:
                del EXISTING_COMM_GROUPS[k]
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
            dist_group = TorchPlatform.get_created_group(split_rank)
            if dist_group is None:
                dist_group = dist.new_group(ranks=split_rank)
                EXISTING_COMM_GROUPS[str(tuple(sorted(split_rank)))] = dist_group
            if TorchPlatform.get_rank() in split_rank:
                split_group = dist_group

        return split_group

    @staticmethod
    def get_group_local_rank(group: ProcessGroup = None) -> int:
        """get group local rank id."""
        group = group or _get_default_group()
        return group.rank()

    @staticmethod
    def no_grad():
        return torch.no_grad()

    @staticmethod
    def cat(tensors, dim=0):
        return torch.cat(tensors, dim=dim)

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
        return torch.utils._pytree.tree_map(fn, tree)  # pylint: disable=protected-access

    @property
    def checkpoint(self):
        return torch.utils.checkpoint.checkpoint

    @staticmethod
    def ckpt_wrapper(module, checkpoint_fn=None, **checkpoint_fn_kwargs):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.activation_swap import FuncModule
        if callable(module) and not isinstance(module, torch.nn.Module):
            module = FuncModule(module)
        return checkpoint_wrapper(module, checkpoint_fn=checkpoint_fn, **checkpoint_fn_kwargs)

    @staticmethod
    def swap_wrapper(module, policy_fn=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.activation_swap import swap_wrapper
        return swap_wrapper(module, policy_fn=policy_fn)

    @staticmethod
    def swap_tensor_wrapper(target, tag=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.activation_swap import swap_tensor_wrapper
        return swap_tensor_wrapper(target, tag=tag)

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
    def get_element_size(tensor):
        """Get Tensor Element Size"""
        return tensor.element_size()

    @staticmethod
    def tensor_to_numpy(tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        return tensor.cpu().numpy()

    @staticmethod
    def clip_grad_norm_(
        parameters, max_norm, norm_type=2.0,
        error_if_nonfinite=False, foreach=None,
    ):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.clip_grad import (
            clip_grad_norm_ as _clip_grad_norm,
        )
        return _clip_grad_norm(
            parameters, max_norm, norm_type,
            error_if_nonfinite=error_if_nonfinite, foreach=foreach,
        )

    @staticmethod
    def profiler_record(name):
        """Profiler context manager for recording operations using torch.profiler."""
        return torch.profiler.record_function(name)

    def cast_fp_tensor(self, dtype, x):
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


    @property
    def meta_device(self):
        return torch.device("meta")

    def init_on_device(self, device, include_buffers=False):
        return _init_on_device(device, include_buffers=include_buffers)

    def str_to_dtype(self, dtype_str: str) -> torch.dtype:
        """Map ``torch.<type>`` strings from checkpoint metadata to ``torch.dtype``."""
        parts = dtype_str.split(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Expected dtype string like 'torch.float32', got {dtype_str!r}."
            )
        prefix, name = parts
        if prefix != "torch":
            raise ValueError(
                f"Expected PyTorch dtype string with prefix 'torch', got {dtype_str!r}."
            )
        dtype = getattr(torch, name)
        if isinstance(dtype, torch.dtype):
            return dtype
        raise ValueError(f"{dtype_str!r} does not resolve to a torch.dtype.")

    def list_to_size(self, size_list: list[int]) -> torch.Size:
        return torch.Size(size_list)
