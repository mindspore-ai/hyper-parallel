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

import torch
import torch.distributed as dist
from torch._dynamo.eval_frame import OptimizedModule

from hyper_parallel import DTensor
from hyper_parallel.core.activation_memory.wrapper import SwapWrapper
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy
from hyper_parallel.core.pipeline_parallel._stage import PipelineStageBase
from hyper_parallel.core.pipeline_parallel.utils import _RecvInfo


class SharedParameterInfo:
    """
    Used to specify information about shared parameter in pipeline parallel, including the parameter obj
    and the stages between which they are shared.

    Args:
        parameter (Parameter): The shared parameter object.
        shared_stage (list): The shared stage list.
        owner_module (Module, optional): Module owning the parameter. When given
            with ``param_name``, the current parameter is re-fetched from it each
            access, so a meta-init pipeline that materializes the stage *after*
            this info is created (``to_empty`` replaces the Parameter object)
            still sees the live parameter (and its gradient) rather than a stale
            orphan. Default ``None`` keeps the parameter object as-is.
        param_name (str, optional): Attribute name of the parameter on
            ``owner_module`` (e.g. ``"weight"``).
    """
    def __init__(self, parameter, shared_stage, owner_module=None, param_name=None):
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"Argument 'parameter' must be type of Parameter, \
                             but got type {type(parameter)}.")
        if not isinstance(shared_stage, (list, tuple)):
            raise TypeError(f"Argument 'shared_stage' must be list or tuple, \
                             but got type {type(shared_stage)}.")
        self._shared_stage = shared_stage
        self._parameter = parameter
        self._owner_module = owner_module
        self._param_name = param_name
        self.group = None

    @property
    def parameter(self):
        """Return the shared parameter object."""
        if self._owner_module is not None and self._param_name is not None:
            return getattr(self._owner_module, self._param_name)
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
            associated with device type cpu``); the fallback avoids that.
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
        super().__init__(submodule, stage_index, stage_num, group, dyn_shape, has_backward)
        self.submodule = submodule
        self.pp_group = self._check_pp_group(group)
        self.device = device if device is not None else (torch.accelerator.current_accelerator() or torch.device("cpu"))
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
        # is popped.
        self._fwd_output_meta = {}
        self._meta_been_send = False
        self._meta_been_recv = False
        self._dyn_shape = dyn_shape
        self._shared_parameters = self._check_shared_parameters(shared_parameters)
        self._virtual_chunk_num = 1
        if self._has_sharded_params:
            if any(isinstance(module, OptimizedModule) for module in submodule.modules()):
                raise ValueError("HSDP pipeline stages do not support torch.compile")
            if any(isinstance(module, SwapWrapper) or "_swap_group_name" in module.__dict__
                   for module in submodule.modules()):
                raise ValueError("HSDP pipeline stages do not support activation swap")
            submodule.hsdp_scheduler.lazy_init()
            if submodule.hsdp_scheduler.scheduler_ctx.root_module is not submodule:
                raise ValueError("An HSDP parameter-management unit must belong to exactly one pipeline stage")
            stage_modules = set(submodule.modules())
            for scheduler in submodule.hsdp_scheduler.scheduler_ctx.all_hsdp_schedulers:
                if not set(scheduler.modules).issubset(stage_modules):
                    raise ValueError("An HSDP module group cannot span pipeline stages")
                if scheduler.comm_fusion_policy.enable_comm_fusion:
                    raise ValueError("HSDP pipeline stages do not support comm_fusion=True")
                if isinstance(scheduler.offload_policy, CPUOffloadPolicy):
                    raise ValueError("HSDP pipeline stages do not support parameter offload")

    @property
    def has_backward(self) -> bool:
        """Return whether this stage executes backward passes."""
        return self._has_backward

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
            rank_id = dist.get_rank()
            device_num = dist.get_world_size()
            real_stage_num = self.stage_num // self._virtual_chunk_num
            device_num_per_stage = device_num // real_stage_num
            index = self.stage_index % real_stage_num
            rank_ids = [rank_id + device_num_per_stage * (i - index) for i in range(real_stage_num)]
            self.pp_group = (
                dist.group.WORLD if rank_ids == list(range(device_num))
                else dist.new_group(rank_ids, use_local_synchronization=True)
            )

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
            # On the meta-init -> FSDP-wrap -> load path the storage is still on
            # meta here; the trainer re-runs this after materialization, so skip
            # until it is real (broadcasting a meta tensor is invalid).
            if getattr(param, "is_meta", False):
                continue
            shared_stage = shared_param_info.shared_stage
            group, group_ranks = self._init_shared_parameter_group(shared_stage)
            shared_param_info.group = group
            # Raw c10d collectives operate on the matching local TP/FSDP shard.
            local_param = param.to_local() if isinstance(param, DTensor) else param
            with torch.no_grad():
                dist.broadcast(local_param, group_ranks[0], group)

    def _global_rank(self, stage_index):
        real_stage_num = self.stage_num // self._virtual_chunk_num
        real_stage_index = stage_index % real_stage_num
        if self.mesh is not None:
            # mesh is a 1-D PP sub-mesh; rank_list[i] is the global rank of stage i.
            return self.mesh.rank_list[real_stage_index]
        return dist.get_global_rank(self.pp_group, real_stage_index)

    def _init_shared_parameter_group(self, shared_stage):
        """init group of shared parameter."""
        group_ranks = [self._global_rank(stage) for stage in shared_stage]
        # When the shared stages span the whole PP sub-mesh (the tied-embedding
        # pp=2 case), reuse the PP group: it is already created per dp line, so
        # this avoids a world-collective ``new_group`` that would deadlock under
        # FSDP where each dp line needs its own {stage0, stage1} group.
        if self.mesh is not None and len(group_ranks) == self.mesh.size():
            return self.mesh.get_group(), group_ranks
        group = dist.new_group(group_ranks, use_local_synchronization=True)
        return group, group_ranks

    def sync_shared_parameters_grad(self) -> None:
        """sync shared parameters' grad with AllReduce."""
        if self._shared_parameters is None or not self._has_backward:
            return
        for shared_param_info in self._shared_parameters:
            param = shared_param_info.parameter
            # Shared endpoints inherit the same tied/frozen configuration, so
            # ``requires_grad`` is group-wide. Frozen peers can all skip.
            if not param.requires_grad:
                continue
            group = shared_param_info.group
            # ``group`` is assigned by ``_sync_shared_parameters`` once the
            # parameter is materialized off meta; if that handshake has not run
            # there is no peer to reduce with yet, so skip.
            if group is None:
                continue

            grad = param.grad
            has_local_grad = grad is not None
            if has_local_grad:
                # Per-stage FSDP stores the reduced boundary-parameter grad as
                # a sharded DTensor. Reduce its local shard so matching DP
                # shards at the tied PP endpoints sum to one gradient.
                local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
            else:
                # Every member of a shared-parameter group must enter the same
                # collective even when this stage did not use the parameter in
                # backward. Derive the zero contribution from the live local
                # parameter shard so shape, dtype, and device match TP/FSDP.
                local_param = param.to_local() if isinstance(param, DTensor) else param
                local_grad = torch.full_like(local_param, 0)

            reduced_grad = local_grad.contiguous()
            handle = dist.all_reduce(reduced_grad, group=group, async_op=True)
            if handle is not None:
                handle.wait()

            if has_local_grad:
                # Torch may return a contiguous communication tensor for a
                # non-contiguous input. Preserve the original grad object (and
                # DTensor layout) while copying the completed reduction back.
                if reduced_grad is not local_grad:
                    local_grad.copy_(reduced_grad)
            elif param.requires_grad:
                if isinstance(param, DTensor):
                    param.grad = DTensor.from_local(reduced_grad, param.device_mesh, param.placements)
                else:
                    param.grad = reduced_grad

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

    def _update_layout(self, layout, sender_rank=None):
        """update the received layout.

        When ``self.mesh`` is set, resolve ``rank_list`` from the **layout's
        own** ``alias_name`` against the root mesh, so a layout that only
        spans part of the within-stage tile (e.g. ``("ep",)`` within a root
        whose non-PP dims are ``("dp", "ep")``) gets just its submesh ranks
        rather than the whole stage's ranks.  Otherwise falls back to
        global-rank arithmetic.
        """
        if self.mesh is not None:
            rank_list = self._get_layout_rank_list(layout, sender_rank)
        else:
            device_num = dist.get_world_size()
            real_stage_num = self.stage_num // self._virtual_chunk_num
            device_num_per_stage = device_num // real_stage_num
            index = self.stage_index % real_stage_num
            rank_list = tuple(range(index * device_num_per_stage, (index + 1) * device_num_per_stage))
        layout.rank_list = rank_list
        layout.update_mesh()
        layout.update_compact_str()

    def _get_layout_rank_list(self, layout, sender_rank=None) -> tuple:
        """Return the global ranks the given layout spans for this process.

        The serialised ``layout.alias_name`` records exactly the mesh dims
        the sender used.  Resolving ranks for **that** submesh (rather than
        the whole within-stage tile) keeps the receiver's group identical
        to the sender's — necessary when a tensor lives on, say, only the
        EP sub-axis of a (dp, ep) within-stage root.
        """
        root = self.mesh.root_mesh
        if root is None or root.ndim <= 1:
            # Flat (1-D world) root: pp_mesh.root_mesh is the world mesh, so the
            # layout's submesh can't be resolved by name. A PP edge shifts whole
            # stages by a fixed rank block and the within-stage tile is identical
            # across stages, so this receiver's submesh is the sender's submesh
            # (layout.rank_list) shifted by the P2P offset (me - sender_rank).
            me = dist.get_rank()
            cur_ranks = tuple(layout.rank_list or ())
            # The cached layout is reused across microbatches/steps and
            # _update_layout mutates it in place, so re-resolution must be
            # idempotent: once it holds this receiver's submesh (contains me),
            # return as-is rather than shifting again. Sender/receiver live in
            # disjoint PP stages, so me is only ever in an already-resolved list.
            if me in cur_ranks:
                return cur_ranks
            if sender_rank is not None and cur_ranks:
                offset = me - sender_rank
                return tuple(r + offset for r in cur_ranks)
            return (me,)
        pp_dim_names = set(self.mesh.mesh_dim_names or ())
        layout_dim_names = tuple(
            name for name in (layout.alias_name or ()) if name not in pp_dim_names
        )
        if not layout_dim_names:
            return (dist.get_rank(),)
        if len(layout_dim_names) == 1:
            submesh = root[layout_dim_names[0]]
        else:
            submesh = root[layout_dim_names]
        return tuple(submesh.rank_list)


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
            self._update_layout(meta[2], global_rank)
            buffer = DTensor.from_local(torch.empty(meta[0], dtype=meta[1], device=self.device),
                                        meta[2].mesh, meta[2].alias_placements)
        else:
            buffer = torch.empty(meta[0], dtype=meta[1], device=self.device)
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

    def fwd_recv_specs(self, micro_index):
        """Prepare forward-recv buffers (+ bookkeeping) without launching.

        Returns a list of ``(op_type, tensor, peer_global_rank)`` tuples (all
        ``"irecv"``) ready to feed ``dist.irecv`` one-by-one or to pack
        into ``dist.batch_isend_irecv``.  Side effects (meta exchange,
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
            buffer = recv_info.buffer
            local_buffer = buffer.to_local() if isinstance(buffer, DTensor) else buffer
            specs.append(("irecv", local_buffer, global_rank))
        if recv_infos:
            self.args_recv_info[micro_index] = recv_infos
        return specs

    def exec_fwd_recv_ops(self, micro_index):
        """Execute the forward recv operation."""
        return [dist.irecv(tensor, rank) for _, tensor, rank in self.fwd_recv_specs(micro_index)]

    def _construct_backward_recv_info(self, micro_index, idx, global_rank):
        """Reserve backward recv bookkeeping without allocating its device buffer."""
        if micro_index not in self.grad_recv_info:
            return _RecvInfo(global_rank)
        recv_info = self.grad_recv_info[micro_index][idx]
        recv_info.buffer = None
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
        so the recv-info reuse path in ``_construct_backward_recv_info``
        keeps aligned across micro-batches.

        The full output meta list is also stashed in ``_fwd_output_meta``
        so backward receive buffers match the grad-requiring output slots.
        """
        return [dist.isend(tensor, rank) for _, tensor, rank in self.fwd_send_specs(micro_index)]

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
        for cur_out in out:
            if self._has_backward and bool(getattr(cur_out, "requires_grad", False)):
                recv_info = self._construct_backward_recv_info(micro_index, bwd_idx, global_rank)
                if recv_info is not None:
                    bwd_recv_infos.append(recv_info)
                bwd_idx += 1
            local_output = cur_out.to_local() if isinstance(cur_out, DTensor) else cur_out
            specs.append(("isend", local_output, global_rank))
        if bwd_recv_infos:
            self.grad_recv_info[micro_index] = bwd_recv_infos
        return specs

    def bwd_recv_specs(self, micro_index: int) -> list[tuple[str, object, int]]:
        """Prepare backward-recv (grad) buffers without launching.

        Device buffers are allocated here, immediately before the scheduler
        posts the receive, instead of remaining resident from forward-send
        time. Repeated specs construction reuses an existing buffer.

        Returns ``(op_type, tensor, peer_global_rank)`` tuples (all
        ``"irecv"``). Empty when no grad is expected for ``micro_index``
        (e.g. its forward output had ``requires_grad=False``).
        """
        recv_infos = self.grad_recv_info.get(micro_index)
        if not recv_infos:
            return []
        grad_metas = [meta for meta in self._fwd_output_meta[micro_index] if bool(meta[-1])]
        specs = []
        for recv_info, meta in zip(recv_infos, grad_metas):
            if recv_info.buffer is None:
                recv_info.buffer = torch.empty(meta[0], dtype=meta[1], device=self.device)
            specs.append(("irecv", recv_info.buffer, recv_info.global_rank))
        return specs

    def exec_bwd_recv_ops(self, micro_index):
        """Execute the backward recv operation."""
        return [dist.irecv(tensor, rank) for _, tensor, rank in self.bwd_recv_specs(micro_index)]

    def exec_bwd_send_ops(self, micro_index):
        """Execute the backward send operation.

        ``backward_one_chunk`` filters ``bwd_cache[mi]`` to only contain
        grads for rg=True inputs, so it aligns 1:1 with the rg=True
        slots of ``args_recv_info[mi]`` (and 1:1 with the peer's
        ``grad_recv_info``).  Pairing via ``zip`` keeps send count and
        peer irecv count consistent.
        """
        return [dist.isend(tensor, rank) for _, tensor, rank in self.bwd_send_specs(micro_index)]

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
        return [
            ("isend", grad.to_local() if isinstance(grad, DTensor) else grad, info.global_rank)
            for grad, info in zip(out, rg_infos)
        ]
