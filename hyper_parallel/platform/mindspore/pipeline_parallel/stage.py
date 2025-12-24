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
"""pipeline stage"""
from abc import ABC
import mindspore as ms
from mindspore import ops, mint, Parameter
from mindspore.mint.distributed import isend, irecv, get_global_rank, broadcast, all_reduce
from mindspore.communication.management import create_group, get_group_size, get_rank
from hyper_parallel.core.hsdp import HSDPCell
from hyper_parallel.core.dtensor import DTensor
from ._utils import _RecvInfo, send_object, recv_object  # pylint: disable = E0402


class SharedParameterInfo:
    """
    Used to specify information about shared parameter in pipeline parallel, including the parameter obj
    and the stages between which they are shared.

    Args:
        parameter (Parameter): The shared parameter object.
        shared_stage (list): The shared stage list.
    """
    def __init__(self, parameter, shared_stage):
        if not isinstance(parameter, Parameter):
            raise TypeError(f"Argument 'parameter' must be type of Parameter, \
                             but got type {type(parameter)}.")
        if not isinstance(shared_stage, (list, tuple)):
            raise TypeError(f"Argument 'shared_stage' must be list or tuple, \
                             but got type {type(shared_stage)}.")
        self._shared_stage = shared_stage
        self._parameter = parameter
        self.group = None

    @property
    def parameter(self):
        return self._parameter

    @property
    def shared_stage(self):
        return self._shared_stage

    def __repr__(self):
        return f"Shared parameter name:({self.parameter.name}), shared stage:({self.shared_stage})"

    def __str__(self):
        return f"Shared parameter name:({self.parameter.name}), shared stage:({self.shared_stage})"


class CellWrapper(ms.nn.Cell):
    """CellWrapper"""
    def __init__(self, fn):
        super().__init__(auto_prefix=False)
        self.network = fn

    def construct(self, *args, **kwargs):
        return self.network(*args, **kwargs)

class PipelineStage(ABC):
    """
    PipelineStage represents a pipeline stage in pipeline parallelism.

    PipelineStage requires the input of a segmented model.

    PipelineStage encapsulates the forward and backward functions used in PipelineSchedule,
    as well as P2P communication.

    Args:
        submodule (HSDPCell): Segmented model.
        stage_index (int): Stage index of current stage.
        stage_num (int): Total stage number.
        group (str): Group of p2p communication.
        has_backward (bool, optional): Specify whether this stage has backward. Default ``True``.
        recv_info(P2PInfo, optional): Specify Receive information. Default ``None``.
        send_info(P2PInfo, optional): Specify Send information. Default ``None``.
    """
    def __init__(self, submodule: HSDPCell, stage_index: int, stage_num: int, group=None,
                 src_stage=None, dst_stage=None, dyn_shape=False, has_backward=True, shared_parameters=None):
        super().__init__()
        if not isinstance(submodule, HSDPCell) and has_backward:
            raise TypeError(f"Argument submodule must be of type HSDPCell, but got type {type(submodule)}.")
        self.submodule = CellWrapper(submodule)
        self.stage_index = stage_index
        self.stage_num = stage_num
        self.pp_group = self._check_pp_group(group)
        self._has_backward = has_backward
        self._recv_info = []
        self._send_info = []
        self._src_stage = self._check_src_stage(src_stage)
        self._dst_stage = self._check_dst_stage(dst_stage)
        self._recv_num = len(self._src_stage)
        self._backward_func = None
        self._has_init = False
        if self._has_backward:
            self.submodule.set_grad(True)
            self._construct_backward_func()
        self._layout_been_communicated = False
        self._shape_been_communicated = False
        self.fwd_inputs_cache = {}
        self.fwd_outputs_cache = {}
        self.last_stage_outputs = None
        self.args_recv_info = {}
        self.grad_recv_info = {}
        self.bwd_cache = {}
        self._layout_been_send = False
        self._layout_been_recv = False
        self._shape_been_send = False
        self._shape_been_recv = False
        self._layout_cache = []
        self._dtype_cache = []
        self._shape_cache = []
        self._dyn_shape = dyn_shape
        self._shared_parameters = self._check_shared_parameters(shared_parameters)
        self._virtual_chunk_num = 1

    def init(self, virtual_chunk_num):
        self._virtual_chunk_num = virtual_chunk_num
        self._init_pp_group()
        self._sync_shared_parameters()

    def _init_pp_group(self):
        """init pipeline parallel communication group."""
        if self.pp_group is None:
            rank_id = get_rank()
            device_num = get_group_size()
            real_stage_num = self.stage_num // self._virtual_chunk_num
            device_num_per_stage = device_num // real_stage_num
            index = self.stage_index % real_stage_num
            rank_ids = [rank_id + device_num_per_stage * (i - index) for i in range(real_stage_num)]
            # if the names are the same, an error will be reported
            pp_group = f"pipeline_group_{rank_ids}"
            create_group(pp_group, rank_ids)
            self.pp_group = pp_group

    def clear_cache(self):
        """clear cache."""
        self.fwd_inputs_cache.clear()
        self.fwd_outputs_cache.clear()
        self.bwd_cache.clear()
        self._layout_cache.clear()
        self._dtype_cache.clear()
        self._shape_cache.clear()

    def clear_states(self):
        """clear fwd and bwd recv_info list."""
        self.args_recv_info.clear()
        self.grad_recv_info.clear()

    def _check_pp_group(self, group):
        """check the type of pipeline group, if it is None, perform default initialization."""
        if group is None:
            return None
        if not isinstance(group, str):
            raise TypeError("Argument 'group' must be type of str, but got type of {type(group)}.")
        return group

    def _check_shared_parameters(self, shared_parameters):
        """check type for shared_parameters."""
        if shared_parameters is None:
            return shared_parameters

        if isinstance(shared_parameters, SharedParameterInfo):
            return [shared_parameters]

        if isinstance(shared_parameters, (list, tuple)):
            for shared_param in shared_parameters:
                if not isinstance(shared_param, SharedParameterInfo):
                    raise TypeError(f"The elements in shared_parameters must be of type SharedParameterInfo, but \
                                     got type {type(shared_parameters)}.")
            return shared_parameters

        raise TypeError(f"Argument 'shared_parameters' must be of type None, SharedParameterInfo, \
                         list/tuple of SharedParameterInfo, but got type {type(shared_parameters)}.")

    def _sync_shared_parameters(self):
        """sync shared parameters with Broadcast."""
        if self._shared_parameters is None:
            return
        for shared_param_info in self._shared_parameters:
            param = shared_param_info.parameter
            shared_stage = shared_param_info.shared_stage
            group, group_ranks = self._init_shared_parameter_group(shared_stage)
            shared_param_info.group = group
            broadcast(param, group_ranks[0], group)

    def _global_rank(self, stage_index):
        real_stage_num = self.stage_num // self._virtual_chunk_num
        real_stage_index = stage_index % real_stage_num
        return get_global_rank(self.pp_group, real_stage_index)

    def _init_shared_parameter_group(self, shared_stage):
        """init group of shared parameter."""
        group_ranks = []
        for stage in shared_stage:
            global_rank = self._global_rank(stage)
            group_ranks.append(global_rank)
        group = "group_" + str(group_ranks)
        create_group(group, group_ranks)
        return group, group_ranks

    def sync_shared_parameters_grad(self):
        """sync shared parameters' grad with AllReduce."""
        if self._shared_parameters is None or not self._has_backward:
            return
        for shared_param_info in self._shared_parameters:
            param = shared_param_info.parameter
            if not param.requires_grad:
                continue
            grad = param.grad
            group = shared_param_info.group
            all_reduce(grad, group=group)

    def _check_src_stage(self, src_stage):
        """check type for src_stage."""
        if src_stage is None:
            return [self.stage_index - 1]

        if isinstance(src_stage, int):
            return [src_stage]

        if isinstance(src_stage, (list, tuple)):
            for each_stage in src_stage:
                if not isinstance(each_stage, int):
                    raise TypeError(f"Argument src_stage must be of type None, int, \
                                     list/tuple of int, but got type list/tuple of {type(each_stage)}.")
            return src_stage

        raise TypeError(f"Argument src_stage must be of type None, int, \
                          list/tuple of int, but got {type(src_stage)}.")

    def _check_dst_stage(self, dst_stage):
        """check type for dst_stage."""
        if dst_stage is None:
            return [self.stage_index + 1]

        if isinstance(dst_stage, int):
            return [dst_stage]

        if isinstance(dst_stage, (list, tuple)):
            for each_stage in dst_stage:
                if not isinstance(each_stage, int):
                    raise TypeError(f"Argument dst_stage must be of type None, int, \
                                     list/tuple of int, but got type list/tuple of {type(each_stage)}.")
            return dst_stage

        raise TypeError(f"Argument dst_stage must be of type None, int, \
                          list/tuple of int, but got {type(dst_stage)}.")

    def _communicate_shape(self, global_rank, idx, tensor_send=None):
        """communicate shape obj."""
        if tensor_send is not None:
            if self._dyn_shape or not self._shape_been_send:
                shape = tensor_send.local_shape
                send_object(shape, global_rank)
            return None

        if self._dyn_shape or not self._shape_been_recv:
            shape = recv_object(global_rank)
            if not self._dyn_shape:
                self._shape_cache.append(shape)
            return shape
        shape = self._shape_cache[idx]
        return shape

    def _communicate_layout(self, global_rank, idx, tensor_send=None):
        """communicate layout and type obj."""
        if tensor_send is not None:
            if not self._layout_been_send:
                layout = tensor_send.layout
                dtype = repr(tensor_send.dtype).split('.')[-1]
                send_object([layout, dtype], global_rank)
            return None

        if self._shape_been_recv:
            return self._layout_cache[idx], self._dtype_cache[idx]
        layout, dtype_str = recv_object(global_rank)
        self._update_layout(layout)
        dtype = getattr(ms, dtype_str)
        self._layout_cache.append(layout)
        self._dtype_cache.append(dtype)
        return layout, dtype

    def _update_layout(self, layout):
        """update the received layout."""
        device_num = get_group_size()
        real_stage_num = self.stage_num // self._virtual_chunk_num
        device_num_per_stage = device_num // real_stage_num
        index = self.stage_index % real_stage_num
        rank_list = tuple(range(index * device_num_per_stage, (index + 1) * device_num_per_stage))
        layout.rank_list = rank_list
        layout.update_mesh()
        layout.update_compact_str()

    def _clear_recv_buffer(self, micro_index):
        """clear fwd and bwd recv buffer."""
        if micro_index not in self.args_recv_info:
            return
        for info in self.args_recv_info[micro_index]:
            info.buffer = None
        if micro_index not in self.grad_recv_info:
            return
        for info in self.grad_recv_info[micro_index]:
            info.buffer = None

    @property
    def is_first_stage(self):
        """return if is first stage."""
        return self.stage_index == 0

    @property
    def is_last_stage(self):
        """return if is last stage."""
        return self.stage_index == self.stage_num - 1

    def forward_one_chunk(self, micro_index, args=None, kwargs=None):
        """Execution a forward function."""
        if self.is_first_stage:
            composite_args = args
        else:
            if micro_index in self.args_recv_info:
                composite_args = [recv_info.buffer for recv_info in self.args_recv_info[micro_index]]
            else:
                raise RuntimeError(f"The exec order is wrong. The corresponding forward calculation \
                                    is executed before the Receive operation. micro is {micro_index}.")
        composite_kwargs = kwargs or {}
        out = self.submodule(*composite_args, **composite_kwargs)
        out_tuple = out if isinstance(out, tuple) else (out,)
        self.fwd_inputs_cache[micro_index] = (composite_args, composite_kwargs)
        self.fwd_outputs_cache[micro_index] = out_tuple
        if self.is_last_stage:
            self.last_stage_outputs = out
        return out

    def get_last_stage_sens(self, last_stage_outputs):
        """Get last stage sens"""
        p_sens = None
        if isinstance(last_stage_outputs, (list, tuple)):
            p_sens = []
            for _, out_i in enumerate(last_stage_outputs):
                if out_i.layout is not None:
                    repeat_num = out_i.layout.repeat_num()
                    sens_i = ops.fill(ops.DType()(out_i), out_i.local_shape, 1.0 / repeat_num)
                else:
                    sens_i = ops.fill(ops.DType()(out_i), out_i.shape, 1.0)
                p_sens.append(sens_i)
        else:
            if last_stage_outputs.layout is not None:
                repeat_num = last_stage_outputs.layout.repeat_num()
                p_sens = ops.fill(ops.DType()(last_stage_outputs), last_stage_outputs.local_shape, 1.0 / repeat_num)
            else:
                p_sens = ops.fill(ops.DType()(last_stage_outputs), last_stage_outputs.shape, 1.0)

        return p_sens

    def backward_one_chunk(self, micro_index, last_backward=False):
        """Execution a backward function."""
        if not self._has_backward:
            return None
        fwd_args, fwd_kwargs = self.fwd_inputs_cache.pop(micro_index)
        recv_args = []
        if last_backward:
            self.submodule.network.set_requires_grad_sync(True)
        if micro_index in self.grad_recv_info:
            recv_args = [recv_info.buffer for recv_info in self.grad_recv_info[micro_index]]
        if self.is_first_stage:
            grad_out = self._backward_func(*fwd_args, **fwd_kwargs, sens=recv_args)
        elif self.is_last_stage:
            sens = self.get_last_stage_sens(self.last_stage_outputs)
            grad_out = self._backward_func(*fwd_args, **fwd_kwargs, sens=sens)
        else:
            grad_out = self._backward_func(*fwd_args, **fwd_kwargs, sens=recv_args)
        self._clear_recv_buffer(micro_index)
        if not self.is_first_stage:
            self.bwd_cache[micro_index] = grad_out[0][:self._recv_num]
        # return grads for parameters
        if self.is_first_stage:
            return grad_out
        return grad_out[1]

    def _construct_forward_recv_info(self, micro_index, idx, global_rank):
        """construct forward recv info."""
        shape = self._communicate_shape(global_rank, idx)
        layout, dtype = self._communicate_layout(global_rank, idx)
        dtensor = DTensor.from_local(mint.empty(shape, dtype=dtype), layout)
        if micro_index in self.args_recv_info:
            recv_info = self.args_recv_info[micro_index][idx]
            recv_info.buffer = dtensor
            return recv_info
        return _RecvInfo(global_rank, dtensor)

    def exec_fwd_recv_ops(self, micro_index):
        """Execute the forward recv operation."""
        recv_infos = []
        comm_handles = []
        for idx in range(self._recv_num):
            global_rank = self._global_rank(self._src_stage[idx])
            recv_info = self._construct_forward_recv_info(micro_index, idx, global_rank)
            if micro_index not in self.args_recv_info:
                recv_infos.append(recv_info)
            handle = irecv(recv_info.buffer, global_rank)
            comm_handles.append(handle)
        self._layout_been_recv = True
        self._shape_been_recv = True
        if recv_infos:
            self.args_recv_info[micro_index] = recv_infos
        return comm_handles

    def _construct_backward_recv_info(self, micro_index, idx, global_rank, tensor_send):
        """construct backward recv info."""
        if micro_index not in self.grad_recv_info:
            buffer = mint.empty(tensor_send.local_shape, dtype=tensor_send.dtype)
            return _RecvInfo(global_rank, buffer)
        recv_info = self.grad_recv_info[micro_index][idx]
        recv_info.buffer = mint.empty(tensor_send.local_shape, dtype=tensor_send.dtype)
        return None

    def exec_fwd_send_ops(self, micro_index):
        """Execute the forward send operation."""
        comm_handle = []
        if self.is_last_stage:
            return comm_handle
        out = self.fwd_outputs_cache.pop(micro_index)
        bwd_recv_infos = []
        for idx, cur_out in enumerate(out):
            if len(self._dst_stage) > 1:
                dst_stage = self._dst_stage[idx]
            else:
                dst_stage = self._dst_stage[0]
            global_rank = self._global_rank(dst_stage)
            self._communicate_shape(global_rank, idx, tensor_send=cur_out)
            self._communicate_layout(global_rank, idx, tensor_send=cur_out)
            if self._has_backward:
                recv_info = self._construct_backward_recv_info(micro_index, idx, global_rank, cur_out)
                if recv_info is not None:
                    bwd_recv_infos.append(recv_info)
            handle = isend(out[idx], global_rank)
            comm_handle.append(handle)
        self._layout_been_send = True
        self._shape_been_send = True
        if bwd_recv_infos:
            self.grad_recv_info[micro_index] = bwd_recv_infos
        return comm_handle

    def exec_bwd_recv_ops(self, micro_index):
        """Execute the backward recv operation."""
        comm_handle = []
        if micro_index not in self.grad_recv_info:
            return comm_handle
        for recv_info in self.grad_recv_info[micro_index]:
            global_rank = recv_info.global_rank
            handle = irecv(recv_info.buffer, global_rank)
            handle.wait()
            comm_handle.append(handle)
        return comm_handle

    def exec_bwd_send_ops(self, micro_index):
        """Execute the backward send operation."""
        comm_handle = []
        if micro_index not in self.args_recv_info:
            return comm_handle
        out = self.bwd_cache.pop(micro_index)
        for idx, cur_out in enumerate(out):
            info = self.args_recv_info[micro_index][idx]
            global_rank = info.global_rank
            handle = isend(cur_out, global_rank)
            comm_handle.append(handle)
        return comm_handle

    def _construct_backward_func(self):
        """construct backward func."""
        self._backward_func = None
        if self.is_first_stage:
            self._backward_func = ops.GradOperation(get_by_list=True, sens_param=True)(
                self.submodule, self.submodule.trainable_params())
        elif self.is_last_stage:
            self._backward_func = ops.GradOperation(get_by_list=True, get_all=True, sens_param=True)(
                self.submodule, self.submodule.trainable_params())
        else:
            self._backward_func = ops.GradOperation(get_by_list=True, get_all=True, sens_param=True)(
                self.submodule, self.submodule.trainable_params())