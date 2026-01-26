# Copyright 2026 Huawei Technologies Co., Ltd
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
from hyper_parallel import DTensor
from hyper_parallel.platform import get_platform
from .utils import _RecvInfo  # pylint: disable = E0402
platform = get_platform()
PipelineStageBase = platform.PipelineStageBase


class SharedParameterInfo:
    """
    Used to specify information about shared parameter in pipeline parallel, including the parameter obj
    and the stages between which they are shared.

    Args:
        parameter (Parameter): The shared parameter object.
        shared_stage (list): The shared stage list.
    """
    def __init__(self, parameter, shared_stage):
        if not isinstance(parameter, platform.Parameter):
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


class PipelineStage(PipelineStageBase):
    """
    PipelineStage represents a pipeline stage in pipeline parallelism.

    PipelineStage requires the input of a segmented model.

    PipelineStage encapsulates the forward and backward functions used in PipelineSchedule,
    as well as P2P communication.

    Args:
        submodule: Segmented model.
        stage_index (int): Stage index of current stage.
        stage_num (int): Total stage number.
        group (ProcessGroup): Group of p2p communication.
        src_stage (int, optional): Src stage index for recv. Default ``None``
        dst_stage (int, optional): Dst stage index for send. Default ``None``
        dyn_shape (bool, optional): Specify whether this stage has dynamic shape. Default ``False``
        has_backward (bool, optional): Specify whether this stage has backward. Default ``True``.
        shared_parameters (SharedParameterInfo, optional): Specify shared parameter information. Default ``None``.
    """
    def __init__(self, submodule, stage_index: int, stage_num: int, device=None, group=None,
                 src_stage=None, dst_stage=None, dyn_shape=False, has_backward=True, shared_parameters=None):
        super().__init__(submodule, stage_index, stage_num, group, has_backward)
        self.submodule = submodule
        self.pp_group = self._check_pp_group(group)
        self.device = device
        self._has_backward = has_backward
        self._recv_info = []
        self._send_info = []
        self.src_stage = self._check_src_stage(src_stage)
        self.dst_stage = self._check_dst_stage(dst_stage)
        self._has_init = False
        self.last_stage_outputs = None
        self.args_recv_info = {}
        self.grad_recv_info = {}
        self._meta_been_send = False
        self._meta_been_recv = False
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
            rank_id = platform.get_rank()
            device_num = platform.get_world_size()
            real_stage_num = self.stage_num // self._virtual_chunk_num
            device_num_per_stage = device_num // real_stage_num
            index = self.stage_index % real_stage_num
            rank_ids = [rank_id + device_num_per_stage * (i - index) for i in range(real_stage_num)]
            self.pp_group = platform.create_group(rank_ids)

    def clear_states(self):
        """clear fwd and bwd recv_info list."""
        self.args_recv_info.clear()
        self.grad_recv_info.clear()

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
        """Sync shared parameters with Broadcast."""
        if self._shared_parameters is None:
            return
        for shared_param_info in self._shared_parameters:
            param = shared_param_info.parameter
            shared_stage = shared_param_info.shared_stage
            group, group_ranks = self._init_shared_parameter_group(shared_stage)
            shared_param_info.group = group
            platform.broadcast(param, group_ranks[0], group)

    def _global_rank(self, stage_index):
        real_stage_num = self.stage_num // self._virtual_chunk_num
        real_stage_index = stage_index % real_stage_num
        return platform.get_global_rank(self.pp_group, real_stage_index)

    def _init_shared_parameter_group(self, shared_stage):
        """init group of shared parameter."""
        group_ranks = []
        for stage in shared_stage:
            global_rank = self._global_rank(stage)
            group_ranks.append(global_rank)
        group = platform.create_group(group_ranks)
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
            platform.all_reduce(grad, group)

    def _check_src_stage(self, src_stage):
        """check type for src_stage."""
        if src_stage is None:
            return self.stage_index - 1

        if isinstance(src_stage, int):
            return src_stage

        raise TypeError(f"Argument src_stage must be of type None, int, but got {type(src_stage)}.")

    def _check_dst_stage(self, dst_stage):
        """check type for dst_stage."""
        if dst_stage is None:
            return self.stage_index + 1

        if isinstance(dst_stage, int):
            return dst_stage

        raise TypeError(f"Argument dst_stage must be of type None, int, but got {type(dst_stage)}.")

    def _update_layout(self, layout):
        """update the received layout."""
        device_num = platform.get_world_size()
        real_stage_num = self.stage_num // self._virtual_chunk_num
        device_num_per_stage = device_num // real_stage_num
        index = self.stage_index % real_stage_num
        rank_list = tuple(range(index * device_num_per_stage, (index + 1) * device_num_per_stage))
        layout.rank_list = rank_list
        layout.update_mesh()
        layout.update_compact_str()

    def get_last_stage_sens(self, last_stage_outputs):
        """Get last stage sens"""
        p_sens = None
        if isinstance(last_stage_outputs, (list, tuple)):
            p_sens = []
            for _, out_i in enumerate(last_stage_outputs):
                if isinstance(out_i, DTensor):
                    repeat_num = out_i.layout.repeat_num()
                    sens_i = platform.full_like(out_i.to_local(), 1.0 / repeat_num)
                else:
                    sens_i = platform.full_like(out_i, 1.0)
                p_sens.append(sens_i)
        else:
            if isinstance(last_stage_outputs, DTensor):
                repeat_num = last_stage_outputs.layout.repeat_num()
                p_sens = platform.full_like(last_stage_outputs.to_local(), 1.0 / repeat_num)
            else:
                p_sens = platform.full_like(last_stage_outputs, 1.0)

        return p_sens

    def _construct_forward_recv_info(self, micro_index, idx, global_rank, meta):
        """construct forward recv info."""
        # shape, type, layout
        if len(meta) == 3:
            self._update_layout(meta[2])
            buffer = DTensor.from_local(platform.new_tensor(meta[0], meta[1],
                                                            device=self.device), meta[2].mesh, meta[2].placements)
        else:
            buffer = platform.new_tensor(meta[0], meta[1], device=self.device)
        buffer.requires_grad = True
        if micro_index in self.args_recv_info:
            recv_info = self.args_recv_info[micro_index][idx]
            recv_info.buffer = buffer
            return recv_info
        return _RecvInfo(global_rank, buffer)

    def _communicate_meta(self, global_rank, meta_send=None):
        """communicate meta."""
        if meta_send is not None:
            if self._dyn_shape or not self._meta_been_send:
                platform.send_object_list([meta_send], global_rank)
                self._meta_been_send = True
            return None

        if self._dyn_shape or not self._meta_been_recv:
            obj_list = [None]
            platform.recv_object_list(obj_list, global_rank)
            self._meta_been_recv = True
            if not self._dyn_shape:
                self._meta_cache = obj_list
            return obj_list
        obj_list = self._meta_cache
        return obj_list

    def exec_fwd_recv_ops(self, micro_index):
        """Execute the forward recv operation."""
        recv_infos = []
        comm_handles = []
        global_rank = self._global_rank(self.src_stage)
        meta_list = self._communicate_meta(global_rank)[0]
        self._recv_num = len(meta_list)
        for idx, meta in enumerate(meta_list):
            recv_info = self._construct_forward_recv_info(micro_index, idx, global_rank, meta)
            if micro_index not in self.args_recv_info:
                recv_infos.append(recv_info)
            handle = platform.irecv(recv_info.buffer, global_rank)
            comm_handles.append(handle)
        if recv_infos:
            self.args_recv_info[micro_index] = recv_infos
        return comm_handles

    def _construct_backward_recv_info(self, micro_index, idx, global_rank, tensor_send):
        """construct backward recv info."""
        if micro_index not in self.grad_recv_info:
            shape = tensor_send.shape if not isinstance(tensor_send, DTensor) else tensor_send.local_shape
            buffer = platform.new_tensor(shape, tensor_send.dtype, device=self.device)
            return _RecvInfo(global_rank, buffer)
        recv_info = self.grad_recv_info[micro_index][idx]
        recv_info.buffer = platform.new_tensor(shape, tensor_send.dtype, device=self.device)
        return None

    def _extract_meta_from_tensor(self, tensor):
        """
        Extract meta info from tensor for communication.

        Args:
            tensor: Input tensor, can be DTensor or regular tensor

        Returns:
            list: Metadata containing shape, dtype and layout
        """
        if isinstance(tensor, DTensor):
            return [tensor.local_shape, tensor.dtype, tensor.layout]
        return [tensor.shape, tensor.dtype]

    def exec_fwd_send_ops(self, micro_index):
        """Execute the forward send operation."""
        comm_handle = []
        if self.is_last_stage:
            return comm_handle
        out = self.fwd_outputs_cache.pop(micro_index)
        bwd_recv_infos = []
        output_meta = [self._extract_meta_from_tensor(each_out) for each_out in out]
        global_rank = self._global_rank(self.dst_stage)
        self._communicate_meta(global_rank, output_meta)
        for idx, cur_out in enumerate(out):
            if self._has_backward:
                recv_info = self._construct_backward_recv_info(micro_index, idx, global_rank, cur_out)
                if recv_info is not None:
                    bwd_recv_infos.append(recv_info)
            handle = platform.isend(out[idx], global_rank)
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
            handle = platform.irecv(recv_info.buffer, global_rank)
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
            handle = platform.isend(cur_out, global_rank)
            comm_handle.append(handle)
        return comm_handle
