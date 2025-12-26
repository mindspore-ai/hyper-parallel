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
import torch
import torch.distributed as dist
from torch.nn import Parameter
from hyper_parallel import DTensor, HSDPCell
from ._utils import _RecvInfo  # pylint: disable = E0402


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


class PipelineStage(ABC):
    """
    PipelineStage represents a pipeline stage in pipeline parallelism.

    PipelineStage requires the input of a segmented model.

    PipelineStage encapsulates the forward and backward functions used in PipelineSchedule,
    as well as P2P communication.

    Args:
        submodule: Segmented model.
        stage_index (int): Stage index of current stage.
        stage_num (int): Total stage number.
        group (str): Group of p2p communication.
        has_backward (bool, optional): Specify whether this stage has backward. Default ``True``.
        recv_info(P2PInfo, optional): Specify Receive information. Default ``None``.
        send_info(P2PInfo, optional): Specify Send information. Default ``None``.
    """
    def __init__(self, submodule, stage_index: int, stage_num: int, device, group=None,
                 src_stage=None, dst_stage=None, dyn_shape=False, has_backward=True, shared_parameters=None):
        super().__init__()
        self.submodule = submodule
        self.stage_index = stage_index
        self.stage_num = stage_num
        self.pp_group = self._check_pp_group(group)
        self.device = device
        self._has_backward = has_backward
        self._recv_info = []
        self._send_info = []
        self.src_stage = self._check_src_stage(src_stage)
        self.dst_stage = self._check_dst_stage(dst_stage)
        self._has_init = False
        self.fwd_outputs_cache = {}
        self.fwd_cache = {}
        self.last_stage_outputs = None
        self.args_recv_info = {}
        self.grad_recv_info = {}
        self.bwd_cache = {}
        self._meta_been_send = False
        self._meta_been_recv = False
        self._meta_cache = []
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
            rank_id = dist.get_rank()
            device_num = dist.get_world_size()
            real_stage_num = self.stage_num // self._virtual_chunk_num
            device_num_per_stage = device_num // real_stage_num
            index = self.stage_index % real_stage_num
            rank_ids = [rank_id + device_num_per_stage * (i - index) for i in range(real_stage_num)]
            # if the names are the same, an error will be reported
            self.pp_group = dist.new_group(rank_ids, use_local_synchronization=True)

    def clear_cache(self):
        """clear cache."""
        self.fwd_outputs_cache.clear()
        self.bwd_cache.clear()
        self._meta_cache.clear()

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
            dist.broadcast(param, group_ranks[0], group)

    def _global_rank(self, stage_index):
        real_stage_num = self.stage_num // self._virtual_chunk_num
        real_stage_index = stage_index % real_stage_num
        return dist.get_global_rank(self.pp_group, real_stage_index)

    def _init_shared_parameter_group(self, shared_stage):
        """init group of shared parameter."""
        group_ranks = []
        for stage in shared_stage:
            global_rank = self._global_rank(stage)
            group_ranks.append(global_rank)
        group = dist.new_group(group_ranks, use_local_synchronization=True)
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
            dist.all_reduce(grad, group=group)

    def _check_src_stage(self, src_stage):
        """check type for src_stage."""
        if src_stage is None:
            return self.stage_index - 1

        if isinstance(src_stage, int):
            return src_stage

        raise TypeError(f"Argument src_stage must be of type None, int, but got {type(each_stage)}.")

    def _check_dst_stage(self, dst_stage):
        """check type for dst_stage."""
        if dst_stage is None:
            return self.stage_index + 1

        if isinstance(dst_stage, int):
            return dst_stage

        raise TypeError(f"Argument dst_stage must be of type None, int, but got {type(dst_stage)}.")

    def _update_layout(self, layout):
        """update the received layout."""
        device_num = dist.get_world_size()
        real_stage_num = self.stage_num // self._virtual_chunk_num
        device_num_per_stage = device_num // real_stage_num
        index = self.stage_index % real_stage_num
        rank_list = tuple(range(index * device_num_per_stage, (index + 1) * device_num_per_stage))
        layout.rank_list = rank_list
        layout.update_mesh()
        layout.update_compact_str()

    def _clear_recv_buffer(self, recv_info, micro_index):
        """clear fwd and bwd recv buffer."""
        if micro_index not in recv_info:
            return
        for info in recv_info[micro_index]:
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
        self.fwd_cache[micro_index] = out_tuple
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
                if isinstance(out_i, DTensor):
                    repeat_num = out_i.layout.repeat_num()
                    sens_i = torch.full(out_i.local_shape, 1.0 / repeat_num, dtype=out_i.dtype, device=self.device)
                else:
                    sens_i = torch.full(out_i.shape, 1.0, dtype=out_i.dtype, device=self.device)
                p_sens.append(sens_i)
        else:
            if isinstance(last_stage_outputs, DTensor):
                repeat_num = last_stage_outputs.layout.repeat_num()
                p_sens = torch.full(last_stage_outputs.local_shape, 1.0 / repeat_num,
                                    dtype=last_stage_outputs.dtype, device=self.device)
            else:
                p_sens = torch.full(last_stage_outputs.shape, 1.0, dtype=last_stage_outputs.dtype,
                                    device=self.device)

        return p_sens

    def backward_one_chunk(self, micro_index, last_backward=False):
        """Execution a backward function."""
        if not self._has_backward:
            return
        recv_args = []
        if last_backward and isinstance(self.submodule, HSDPCell):
            self.submodule.set_requires_grad_sync(True)
        if micro_index in self.grad_recv_info:
            recv_args = [recv_info.buffer for recv_info in self.grad_recv_info[micro_index]]

        fwd_output = self.fwd_cache.pop(micro_index)
        if self.is_first_stage:
            torch.autograd.backward(fwd_output, grad_tensors=recv_args)
        elif self.is_last_stage:
            sens = self.get_last_stage_sens(self.last_stage_outputs)
            torch.autograd.backward(fwd_output, grad_tensors=sens)
        else:
            torch.autograd.backward(fwd_output, grad_tensors=recv_args)
        if not self.is_first_stage:
            input_grads = [recv_info.buffer.grad for recv_info in self.args_recv_info[micro_index]]
            self.bwd_cache[micro_index] = input_grads
        self._clear_recv_buffer(self.grad_recv_info, micro_index)
        self._clear_recv_buffer(self.args_recv_info, micro_index)

    def _construct_forward_recv_info(self, micro_index, idx, global_rank, meta):
        """construct forward recv info."""
        if isinstance(meta, DTensor):
            buffer = DTensor.from_local(torch.empty(meta.local_shape, dtype=meta.dtype,
                                        device=self.device), meta.layout)
        else:
            buffer = torch.empty(meta.shape, dtype=meta.dtype, device=self.device)
        buffer.requires_grad = True
        if micro_index in self.args_recv_info:
            recv_info = self.args_recv_info[micro_index][idx]
            recv_info.buffer = buffer
            return recv_info
        return _RecvInfo(global_rank, buffer)

    def exec_fwd_recv_ops(self, micro_index):
        """Execute the forward recv operation."""
        recv_infos = []
        comm_handles = []
        global_rank = self._global_rank(self.src_stage)
        meta_list = self._communicate_meta(global_rank)[0]
        for idx, meta in enumerate(meta_list):
            recv_info = self._construct_forward_recv_info(micro_index, idx, global_rank, meta)
            if micro_index not in self.args_recv_info:
                recv_infos.append(recv_info)
            handle = dist.irecv(recv_info.buffer, global_rank)
            comm_handles.append(handle)
        if recv_infos:
            self.args_recv_info[micro_index] = recv_infos
        return comm_handles

    def _construct_backward_recv_info(self, micro_index, idx, global_rank, tensor_send):
        """construct backward recv info."""
        if micro_index not in self.grad_recv_info:
            shape = tensor_send.shape if not isinstance(tensor_send, DTensor) else tensor_send.local_shape
            buffer = torch.empty(shape, dtype=tensor_send.dtype, device=self.device)
            return _RecvInfo(global_rank, buffer)
        recv_info = self.grad_recv_info[micro_index][idx]
        recv_info.buffer = torch.empty(shape, dtype=tensor_send.dtype, device=self.device)
        return None

    def _communicate_meta(self, global_rank, meta_send=None):
        """communicate meta."""
        if meta_send is not None:
            if self._dyn_shape or not self._meta_been_send:
                dist.send_object_list([meta_send], global_rank)
                self._meta_been_send = True
            return None

        if self._dyn_shape or not self._meta_been_recv:
            obj_list = [None]
            dist.recv_object_list(obj_list, global_rank)
            self._meta_been_recv = True
            if not self._dyn_shape:
                self._meta_cache = obj_list
            return obj_list
        obj_list = self._meta_cache
        return obj_list

    def exec_fwd_send_ops(self, micro_index):
        """Execute the forward send operation."""
        comm_handle = []
        if self.is_last_stage:
            return comm_handle
        out = self.fwd_outputs_cache.pop(micro_index)
        bwd_recv_infos = []
        output_meta = [each_out.to('meta') for each_out in out]
        global_rank = self._global_rank(self.dst_stage)
        self._communicate_meta(global_rank, output_meta)
        for idx, cur_out in enumerate(out):
            if self._has_backward:
                recv_info = self._construct_backward_recv_info(micro_index, idx, global_rank, cur_out)
                if recv_info is not None:
                    bwd_recv_infos.append(recv_info)
            handle = dist.isend(out[idx], global_rank)
            comm_handle.append(handle)
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
            handle = dist.irecv(recv_info.buffer, global_rank)
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
            handle = dist.isend(cur_out, global_rank)
            comm_handle.append(handle)
        return comm_handle
