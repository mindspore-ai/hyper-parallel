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
"""HSDP parameter"""
import functools
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.hsdp_refactor.hsdp_utils import OptimizerLevel, GroupInfo, HSDPConfigV2
from hyper_parallel import DeviceMesh


class HSDPParamV2:
    """
    HSDP parameter.
    """
    def __init__(
        self,
        cell, 
        param_name,
        param,
        config: HSDPConfigV2,
        platform
    ):
        self.cell = cell
        self.param_name = param_name
        self.param = param
        self.config = config
        self.mesh = self.config.mesh
        self.reshard_after_forward = self.config.reshard_after_forward
        self.shard_placement_fn = self.config.shard_placement_fn
        self.mp_policy = self.config.mp_policy
        self.offload_policy = self.config.offload_policy
        self.reduce_dtype = self.offload_policy.reduce_dtype
        self.platform = platform
        self.shard_size = 1
        self.unsharded_param = None
        self.sharded_param = None
        # self.acc_grad = None
        # self.grad = None
        self.sharded = False
        self.fully_sharded = False
        self.prefetch_handle = None
        self.prefetch_data = None
        self.param_buffer_start_index = 0
        self.param_buffer_end_index = 0
        self.grad_buffer_start_index = 0
        self.grad_buffer_end_index = 0

        if self.mesh.ndim == 1:
            self.fully_sharded = True
        elif self.mesh.ndim == 2:
            self.sharded = True
        self._init_param()
        group_name, group = self._create_sharded_dp_group()
        self.sharded_group_info = GroupInfo(group_name, group, self.shard_size)
        group_name, group = self._create_unsharded_dp_group()
        self.unsharded_group_info = GroupInfo(group_name, group, self.dp_size)


    def _init_sharded_param(self):
        """add and init sharded param"""
        raise NotImplementedError("HSDP param subclasses must implement _init_sharded_param")

    def _init_sharded_param_by_mesh(self):
        raise NotImplementedError("HSDP param subclasses must implement _init_sharded_param_by_mesh")

    def _init_unsharded_param(self):
        """add and init unshared param"""
        raise NotImplementedError("HSDP param subclasses must implement _init_unsharded_param")

    def _get_unsharded_param_data(self, async_op):
        """get unsharded param data with async comm"""
        local_data = self.platform.get_param_local_data(self.param)
        return self.platform.all_gather_into_tensor(local_data, self.sharded_group_info, async_op=async_op)



    def _create_sharded_dp_group(self, mesh: DeviceMesh | None = None):
        """create communication group for sharded parameter"""
        if self.shard_size <= 1:
            return "hsdp_sharded_dp_group_invalid", None
        if self.mesh:
            hsdp_shard_dim_alias = mesh.alias_name[-1]
            rank_list = mesh.get_rank_list_along_axis(hsdp_shard_dim_alias)
        else:
            rank_list = self._get_op_rank_list()
        rank_list_str = "_".join([str(i) for i in rank_list])
        group_name = "hsdp_sharded_dp_group_" + rank_list_str
        group = self.platform.create_group(rank_list, group_name)
        return group_name, group

    def _create_unsharded_dp_group(self, mesh: DeviceMesh | None = None):
        """create communication group for unsharded parameter"""
        if self.dp_size <= 1:
            return "hsdp_unsharded_dp_group_invalid", None
        if mesh.ndim == 2:
            replicate_dim_alias = self.mesh.alias_name[0]
            rank_list = self.mesh.get_rank_list_along_axis(replicate_dim_alias)
        else:
            rank_list = self._get_dp_rank_list()
        rank_list_str = "_".join([str(i) for i in rank_list])
        group_name = "hsdp_unshared_dp_group_" + rank_list_str
        group = self.platform.create_group(rank_list, group_name)
        return group_name, group

    def _init_param(self):
        """init hsdp parameter"""
        # TODO: 这里仔细调整下，去掉requres_acc_grad的判断，因为这个参数不再保留了
        self.param_shape = self.platform.get_param_local_shape(self.param)

        if self.shard_size == 1:
            self.sharded = False
            self.fully_sharded = False
            if self.config.requires_acc_grad and self.param.requires_grad:
                acc_grad_type = self.param.dtype
                if self.config.reduce_dtype is not None:
                    acc_grad_type = self.config.reduce_dtype
                self.acc_grad = self.platform.new_zero_parameter(self.param_shape, acc_grad_type, False,
                                                                 self.param.device)
            self.param.acc_grad = self.acc_grad
            return

        origin_param_shape = list(self.param_shape)
        self._init_unsharded_param()
        self._init_sharded_param()
        if self.config.requires_acc_grad and self.param.requires_grad:
            acc_grad_shape = origin_param_shape
            if self.config.shard_level != OptimizerLevel.SHARD_OPT:
                acc_grad_shape = self.sharded_param.shape
            acc_grad_type = self.param.dtype
            if self.config.reduce_dtype is not None:
                acc_grad_type = self.config.reduce_dtype
            self.acc_grad = self.platform.new_zero_parameter(acc_grad_shape, acc_grad_type, False, self.param.device)
        self.param.acc_grad = self.acc_grad
        self.sharded = True
        if self.shard_size == self.rank_size:
            self.fully_sharded = True
        else:
            self.fully_sharded = False

    def to_sharded(self):
        """change parameter to sharded state"""
        self.platform.update_param_data(self.param, self.sharded_param)

    def prefetch_unsharded(self):
        """prefetch unsharded param with async all gather"""
        if self.prefetch_handle is not None:
            return
        unshared_param_data, handle = self._get_unsharded_param_data(async_op=True)
        self.prefetch_data = unshared_param_data
        self.prefetch_handle = handle

    #pylint: disable=W0212
    def to_unsharded(self, async_op=False):
        """change parameter to unsharded state"""
        if self.prefetch_handle is not None:
            self.prefetch_handle.wait()
            self.platform.update_param_data(self.sharded_param, self.platform.get_param_local_data(self.param))
            self.platform.update_param_data(self.param, self.prefetch_data)
            self.prefetch_handle = None
            self.prefetch_data = None
            return

        unshared_param_data, handle = self._get_unsharded_param_data(async_op=async_op)
        if handle or async_op:
            # TODO: support async unshard
            raise NotImplementedError(f"support async allgather.")
        self.platform.update_param_data(self.sharded_param, self.platform.get_param_local_data(self.param))
        self.platform.update_param_data(self.param, unshared_param_data)

    def zero_acc_grad(self):
        """zero accumunication grad"""
        if self.param.acc_grad is not None:
            self.param.acc_grad.zero_()
