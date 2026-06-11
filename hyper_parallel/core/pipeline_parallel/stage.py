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
from typing import Optional
from types import SimpleNamespace

from hyper_parallel import DTensor
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.platform import get_platform
from .utils import _RecvInfo  # pylint: disable=E0402

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
        """Return the shared parameter object."""
        return self._parameter

    @property
    def shared_stage(self):
        """Return the list of stages sharing this parameter."""
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
        device (Union[str, Device], optional): Device on which the P2P communication buffers are
            allocated. Default ``None``, resolved to the current accelerator device. Under PyTorch an
            explicit ``None`` would place the buffers on CPU, which HCCL/NCCL rejects (``No backend type
            associated with device type cpu``); the fallback avoids that. MindSpore binds the device at
            process init and ignores this argument.
        group (ProcessGroup): Group of p2p communication.
        src_stage (int, optional): Src stage index for recv. Default ``None``
        dst_stage (int, optional): Dst stage index for send. Default ``None``
        dyn_shape (bool, optional): Specify whether this stage has dynamic shape. Default ``False``
        has_backward (bool, optional): Specify whether this stage has backward. Default ``True``.
        shared_parameters (SharedParameterInfo, optional): Specify shared parameter information. Default ``None``.
        mesh (DeviceMesh, optional): A **1-D PP sub-mesh**, e.g. ``full_mesh["pp"]``.
            Its ``rank_list`` contains the global ranks along the pipeline
            dimension for the current process.  When provided it is used to
            derive the PP communication group (``_init_pp_group``) and
            within-stage rank list (``_update_layout``), replacing the
            global-rank arithmetic that assumes PP is the only parallelism.
            Default ``None``.
    """
    def __init__(self, submodule, stage_index: int, stage_num: int, device=None, group=None,
                 src_stage=None, dst_stage=None, dyn_shape=False, has_backward=True,
                 shared_parameters=None, mesh: Optional[DeviceMesh] = None):
        super().__init__(submodule, stage_index, stage_num, group, has_backward)
        self.submodule = submodule
        self.pp_group = self._check_pp_group(group)
        self.device = device if device is not None else platform.device()
        self.mesh = mesh
        self._has_backward = has_backward
        self._recv_info = []
        self._send_info = []
        self.src_stage = self._check_src_stage(src_stage)
        self.dst_stage = self._check_dst_stage(dst_stage)
        self._has_init = False
        self.last_stage_outputs = None
        self.args_recv_info = {}
        self.grad_recv_info = {}
        # micro_index -> list of metas (matching ``_extract_meta_from_tensor``).  Captured at fwd-send
        # time so backward can read each output's ``requires_grad`` flag after ``fwd_outputs_cache``
        # is popped — needed to zero-pad sens on the MS GradOperation path.
        self._fwd_output_meta = {}
        self._meta_been_send = False
        self._meta_been_recv = False
        self._dyn_shape = dyn_shape
        self._shared_parameters = self._check_shared_parameters(shared_parameters)
        self._virtual_chunk_num = 1

    def init(self, virtual_chunk_num):
        """Initialize the pipeline stage: set virtual chunk count, PP group, and sync shared parameters."""
        self._virtual_chunk_num = virtual_chunk_num
        self._init_pp_group()
        self._sync_shared_parameters()

    def _init_pp_group(self):
        """init pipeline parallel communication group.

        When ``self.mesh`` (a 1-D PP sub-mesh) is provided, its sole dimension
        already represents the PP axis, so the group is obtained directly via
        ``mesh.get_group()``.  Otherwise falls back to global-rank arithmetic.
        """
        if self.pp_group is not None:
            return
        if self.mesh is not None:
            self.pp_group = self.mesh.get_group()
        else:
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
        self._fwd_output_meta.clear()

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
        if self.mesh is not None:
            # mesh is a 1-D PP sub-mesh; rank_list[i] is the global rank of stage i.
            return self.mesh.rank_list[real_stage_index]
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
            # platform.all_reduce expects group_info (with .group for Torch, or str for MindSpore)
            group_info = group if isinstance(group, str) else SimpleNamespace(group=group)
            platform.all_reduce(grad, group_info)

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
        """update the received layout.

        When ``self.mesh`` is set, resolve ``rank_list`` from the **layout's
        own** ``alias_name`` against the root mesh, so a layout that only
        spans part of the within-stage tile (e.g. ``("ep",)`` within a root
        whose non-PP dims are ``("dp", "ep")``) gets just its submesh ranks
        rather than the whole stage's ranks.  Otherwise falls back to
        global-rank arithmetic.
        """
        if self.mesh is not None:
            rank_list = self._get_layout_rank_list(layout)
        else:
            device_num = platform.get_world_size()
            real_stage_num = self.stage_num // self._virtual_chunk_num
            device_num_per_stage = device_num // real_stage_num
            index = self.stage_index % real_stage_num
            rank_list = tuple(range(index * device_num_per_stage, (index + 1) * device_num_per_stage))
        layout.rank_list = rank_list
        layout.update_mesh()
        layout.update_compact_str()

    def _get_layout_rank_list(self, layout) -> tuple:
        """Return the global ranks the given layout spans for this process.

        The serialised ``layout.alias_name`` records exactly the mesh dims
        the sender used.  Resolving ranks for **that** submesh (rather than
        the whole within-stage tile) keeps the receiver's group identical
        to the sender's — necessary when a tensor lives on, say, only the
        EP sub-axis of a (dp, ep) within-stage root.
        """
        root = self.mesh.root_mesh
        if root is None or root.ndim <= 1:
            # PP-only topology: one rank per stage
            return (platform.get_rank(),)
        pp_dim_names = set(self.mesh.mesh_dim_names or ())
        layout_dim_names = tuple(
            name for name in (layout.alias_name or ()) if name not in pp_dim_names
        )
        if not layout_dim_names:
            return (platform.get_rank(),)
        if len(layout_dim_names) == 1:
            submesh = root[layout_dim_names[0]]
        else:
            submesh = root[layout_dim_names]
        return tuple(submesh.rank_list)

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
        """construct forward recv info.

        ``meta`` layout — trailing element is always the sender tensor's
        ``requires_grad`` flag, so the recv buffer mirrors it and the
        backward send path can skip non-grad slots:
          * DTensor:  ``[local_shape, dtype, layout, requires_grad]``  (len 4)
          * regular:  ``[shape, dtype, requires_grad]``                (len 3)
        """
        requires_grad = bool(meta[-1])
        if len(meta) == 4:
            self._update_layout(meta[2])
            buffer = DTensor.from_local(platform.empty(meta[0], dtype=meta[1],
                                                       device=self.device), meta[2].mesh, meta[2].alias_placements)
        else:
            buffer = platform.empty(meta[0], dtype=meta[1], device=self.device)
        buffer.requires_grad = requires_grad
        if micro_index in self.args_recv_info:
            recv_info = self.args_recv_info[micro_index][idx]
            recv_info.buffer = buffer
            recv_info.requires_grad = requires_grad
            return recv_info
        return _RecvInfo(global_rank, buffer, requires_grad=requires_grad)

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

    def fwd_recv_specs(self, micro_index):
        """Prepare forward-recv buffers (+ bookkeeping) without launching.

        Returns a list of ``(op_type, tensor, peer_global_rank)`` tuples (all
        ``"irecv"``) ready to feed ``platform.irecv`` one-by-one or to pack
        into ``platform.batch_isend_irecv``.  Side effects (meta exchange,
        ``args_recv_info`` population) match :meth:`exec_fwd_recv_ops` so the
        two paths stay interchangeable.
        """
        recv_infos = []
        specs = []
        global_rank = self._global_rank(self.src_stage)
        meta_list = self._communicate_meta(global_rank)[0]
        self._recv_num = len(meta_list)
        for idx, meta in enumerate(meta_list):
            recv_info = self._construct_forward_recv_info(micro_index, idx, global_rank, meta)
            if micro_index not in self.args_recv_info:
                recv_infos.append(recv_info)
            specs.append(("irecv", recv_info.buffer, global_rank))
        if recv_infos:
            self.args_recv_info[micro_index] = recv_infos
        return specs

    def exec_fwd_recv_ops(self, micro_index):
        """Execute the forward recv operation."""
        return [platform.irecv(tensor, rank) for _, tensor, rank in self.fwd_recv_specs(micro_index)]

    def _construct_backward_recv_info(self, micro_index, idx, global_rank, tensor_send):
        """construct backward recv info."""
        if micro_index not in self.grad_recv_info:
            shape = tensor_send.shape if not isinstance(tensor_send, DTensor) else tensor_send.local_shape
            buffer = platform.empty(shape, dtype=tensor_send.dtype, device=self.device)
            return _RecvInfo(global_rank, buffer)
        recv_info = self.grad_recv_info[micro_index][idx]
        shape = tensor_send.shape if not isinstance(tensor_send, DTensor) else tensor_send.local_shape
        recv_info.buffer = platform.empty(shape, dtype=tensor_send.dtype, device=self.device)
        return None

    def _extract_meta_from_tensor(self, tensor):
        """
        Extract meta info from tensor for communication.

        Args:
            tensor: Input tensor, can be DTensor or regular tensor

        Returns:
            list: Metadata for the receiver.  The trailing element is
                always the tensor's ``requires_grad`` flag so the peer can
                mirror it on the recv buffer and skip backward send/recv
                for non-differentiable forward tensors.
                  * DTensor:  ``[local_shape, dtype, layout, requires_grad]``
                  * regular:  ``[shape, dtype, requires_grad]``
        """
        requires_grad = bool(tensor.requires_grad)
        if isinstance(tensor, DTensor):
            return [tensor.local_shape, tensor.dtype, tensor.layout, requires_grad]
        return [tensor.shape, tensor.dtype, requires_grad]

    def exec_fwd_send_ops(self, micro_index):
        """Execute the forward send operation.

        Only outputs with ``requires_grad=True`` reserve a slot in
        ``grad_recv_info`` — otherwise the peer would send back N grads
        while this side waits for fewer, and the irecv count would
        diverge.  ``bwd_idx`` tracks the position **within
        ``grad_recv_info[micro_index]``** (which skips non-grad outputs),
        so the buffer-reuse path in ``_construct_backward_recv_info``
        keeps aligned across micro-batches.

        The full output meta list is also stashed in ``_fwd_output_meta``
        so ``backward_one_chunk`` (esp. on MindSpore) can rebuild a
        zero-padded sens matching the wrapped forward's output structure.
        """
        return [platform.isend(tensor, rank) for _, tensor, rank in self.fwd_send_specs(micro_index)]

    def fwd_send_specs(self, micro_index):
        """Prepare forward-send tensors (+ bookkeeping) without launching.

        Returns ``(op_type, tensor, peer_global_rank)`` tuples (all
        ``"isend"``).  Performs the same meta exchange and ``grad_recv_info``
        reservation as the launching path, so the output tensors it returns
        must stay alive until the caller waits the resulting handle(s).
        """
        if self.is_last_stage:
            return []
        out = self.fwd_outputs_cache.pop(micro_index)
        bwd_recv_infos = []
        specs = []
        output_meta = [self._extract_meta_from_tensor(each_out) for each_out in out]
        # Keep meta alive for backward — fwd_outputs_cache has just been popped.
        self._fwd_output_meta[micro_index] = output_meta
        global_rank = self._global_rank(self.dst_stage)
        self._communicate_meta(global_rank, output_meta)
        bwd_idx = 0
        for idx, cur_out in enumerate(out):
            if self._has_backward and bool(getattr(cur_out, "requires_grad", False)):
                recv_info = self._construct_backward_recv_info(micro_index, bwd_idx, global_rank, cur_out)
                if recv_info is not None:
                    bwd_recv_infos.append(recv_info)
                bwd_idx += 1
            specs.append(("isend", out[idx], global_rank))
        if bwd_recv_infos:
            self.grad_recv_info[micro_index] = bwd_recv_infos
        return specs

    def bwd_recv_specs(self, micro_index):
        """Prepare backward-recv (grad) buffers without launching.

        Returns ``(op_type, tensor, peer_global_rank)`` tuples (all
        ``"irecv"``).  Empty when no grad is expected for ``micro_index``
        (e.g. its forward output had ``requires_grad=False``).
        """
        if micro_index not in self.grad_recv_info:
            return []
        return [("irecv", info.buffer, info.global_rank) for info in self.grad_recv_info[micro_index]]

    def exec_bwd_recv_ops(self, micro_index):
        """Execute the backward recv operation."""
        return [platform.irecv(tensor, rank) for _, tensor, rank in self.bwd_recv_specs(micro_index)]

    def exec_bwd_send_ops(self, micro_index):
        """Execute the backward send operation.

        ``backward_one_chunk`` filters ``bwd_cache[mi]`` to only contain
        grads for rg=True inputs, so it aligns 1:1 with the rg=True
        slots of ``args_recv_info[mi]`` (and 1:1 with the peer's
        ``grad_recv_info``).  Pairing via ``zip`` keeps send count and
        peer irecv count consistent.
        """
        return [platform.isend(tensor, rank) for _, tensor, rank in self.bwd_send_specs(micro_index)]

    def bwd_send_specs(self, micro_index):
        """Prepare backward-send (input-grad) tensors without launching.

        Returns ``(op_type, tensor, peer_global_rank)`` tuples (all
        ``"isend"``), pairing each grad with the rg=True slot of
        ``args_recv_info`` exactly as the launching path does.
        """
        if micro_index not in self.args_recv_info:
            return []
        out = self.bwd_cache.pop(micro_index)
        rg_infos = [info for info in self.args_recv_info[micro_index] if info.requires_grad]
        return [("isend", cur_out, info.global_rank) for cur_out, info in zip(out, rg_infos)]

    def execute_reduce_grad(self):
        """Trigger FSDP post-backward gradient reduction and root hook for the stage submodule."""
        if not isinstance(self.submodule, HSDPModule):
            return
        fsdp_module = self.submodule
        fsdp_module.set_is_last_backward(True)
        fsdp_module.set_reshard_after_backward(True)
        fsdp_module.set_requires_gradient_sync(True)

        for _, submod in platform.get_cells_and_names(fsdp_module):
            if not isinstance(submod, HSDPModule):
                continue
            sub_mod_state = submod.hsdp_scheduler.hsdp_state
            sub_mod_state.post_backward()
            sub_mod_state.reduce_params()

        # No public API exposes the root backward finalization; call the platform hook directly.
        fsdp_module.hsdp_scheduler._root_backward_hook()  # pylint: disable=protected-access

    def _build_padded_sens(self, micro_index):
        """Build an N-length sens list aligned with the forward output structure.

        MindSpore's GradOperation requires sens to match the wrapped forward's
        output signature.  ``grad_recv_info[mi]`` only holds K = rg-true slots,
        so the remaining N - K slots are filled with zero placeholders sized
        from the recorded meta.

        Returns:
            list: sens tensors, length equal to the forward output count.
                Empty if no meta is recorded (e.g. last stage or unknown mi).
        """
        metas = self._fwd_output_meta.get(micro_index)
        if not metas:
            return []
        grad_recv = self.grad_recv_info.get(micro_index, [])
        sens = []
        grad_idx = 0
        for meta in metas:
            if bool(meta[-1]):
                sens.append(grad_recv[grad_idx].buffer)
                grad_idx += 1
            else:
                # meta is [shape, dtype, rg] or [local_shape, dtype, layout, rg]
                sens.append(platform.zeros(meta[0], dtype=meta[1], device=self.device))
        return sens

    def _output_requires_grad_mask(self, micro_index):
        """Return the requires_grad mask for forward outputs at ``micro_index``.

        For non-last stages the mask is read from ``_fwd_output_meta`` (recorded
        during ``exec_fwd_send_ops``).  For the last stage no fwd-send runs, so
        the caller should derive the mask from ``last_stage_outputs`` instead.
        """
        metas = self._fwd_output_meta.get(micro_index)
        if not metas:
            return None
        return [bool(meta[-1]) for meta in metas]
