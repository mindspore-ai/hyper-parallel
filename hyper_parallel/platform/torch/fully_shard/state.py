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
"""Torch HSDP cell state"""
from typing import List
import torch
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_utils import _get_param_module_infos
from hyper_parallel.platform.torch.fully_shard.param import TorchHSDPParamV2
from hyper_parallel.platform.torch.fully_shard.utils import HSDPMeshInfo


class TorchHSDPStateV2(HSDPState):
    """Torch HSDP cell state"""
    def __init__(self, cell, mesh_info, config, platform):
        super().__init__(cell, mesh_info, config, platform)
        # self._init_mp_dtypes()
        # Do ReduceScatter/AllReduce for grad
        self.mp_policy = config.mp_policy
        self.offload_policy = config.offload_policy
        self.reduce_grads = True
        # Reshard parameter after backward
        self.reshard_after_backward = True
        # Requires AllReduce for grad When HSDP
        self.requires_all_reduce = True
        self._use_post_forward_mesh = False
        self._init_mp_dtypes()

    def _init_hsdp_params(self):
        """init hsdp parameters for cell"""
        # Cell 树内的全部parameters
        filtered_params = []
        for param_name, param in self.cell.named_parameters():
            if hasattr(param, "_hsdp_param_initialized") and param._hsdp_param_initialized:
                # 在HSDPParam._init_sharded_param中添加该属性，避免重复初始化
                # 通过_setattr_重新给cell绑定了param后，named_parameters会重复遍历到该param
                continue
            filtered_params.append(param)

        module_infos = _get_param_module_infos(filtered_params, [self.cell,])
        for param, module_info in zip(filtered_params, module_infos):
            hsdp_param = TorchHSDPParamV2(param,
                                          module_info,
                                          self.mesh_info,
                                          mp_policy=self.mp_policy,
                                          offload_policy=self.offload_policy
                                          )
            self.hsdp_params.append(hsdp_param)
            if hsdp_param.is_sharded:
                # TODO: 这个可能不需要了，后续根据mesh处理是否切分。
                self.sharded_hsdp_params.append(hsdp_param)

    def _init_mp_dtypes(self,):
        """init mp dtypes for hsdp parameters"""
        for hsdp_param in self.hsdp_params:
            hsdp_param.init_dtype_attrs(self.mp_policy)
        trainable_params: list[TorchHSDPParamV2] = [
            p for p in self.hsdp_params if p.sharded_param.requires_grad
        ]
        orig_dtypes = {p.orig_dtype for p in trainable_params}
        reduce_dtypes = {p.reduce_dtype for p in trainable_params}
        if len(trainable_params) > 0 and len(orig_dtypes) != 1:
            raise AssertionError(
                f"hsdp expects uniform original parameter dtype but got {orig_dtypes}"
            )
        self._orig_dtype = next(iter(orig_dtypes)) if trainable_params else None
        if len(trainable_params) > 0 and len(reduce_dtypes) != 1:
            raise AssertionError(
                f"hsdp expects uniform reduce dtype but got {reduce_dtypes}"
            )
        self._reduce_dtype = next(iter(reduce_dtypes)) if trainable_params else None


    def lazy_init(self):
        raise NotImplementedError("lazy_init not implemented in TorchHSDPStateV2")

    def reshard(self,):
        # TODO：补齐reshard接口，当前我们不考虑reshard_after_forward配置是int的情况，只考虑True/False
        # if self.scheduler_state == FSDPSchedulerState.FORWARD:
        #     if not self.reshard_after_forward:
        #         return
        #     if self._use_post_forward_mesh:
        #         # TODO: support reshard_after_forward=(int)
        #         raise NotImplementedError(f"For reshard, need support reshard_after_forward=(int).")
        #         self._to_sharded_post_forward()
        #         self._reshard_after_forward_event = self.device_handle.Event()
        #         if self._reshard_after_forward_event is not None:
        #             self._reshard_after_forward_event.record()
        #         return
        self.shard()

    def post_backward(self, *unused):
            # TODO: reshard + 梯度处理
            for hsdp_param in self.hsdp_params:
                hsdp_param.accumulate_unsharded_grad_if_needed()
            if not self.reduce_grads:
                if self.reshard_after_backward:
                    self.reshard()
                for hsdp_param in self.hsdp_params:
                    hsdp_param.to_accumulated_grad_if_needed()
                return
            hsdp_params_with_grad: List[TorchHSDPParamV2] = []
            unsharded_grads: List[torch.Tensor] = []
            for hsdp_param in self.hsdp_params:
                if not hasattr(hsdp_param, "_unsharded_param") or hsdp_param.unsharded_param is None:
                    continue
                # Frozen parameters (requires_grad=False) produce no
                # gradient — skip all reduce-scatter / all-reduce work.
                if not hsdp_param.sharded_param.requires_grad:
                    continue
                if hsdp_param.shard_world_size > 1:
                    if hsdp_param.unsharded_param.grad is None:
                        # Parameter requires grad but was not used in
                        # forward — all ranks skip consistently.
                        continue
                    reduced_grad, handle = hsdp_param.reduce_scatter_grad(dtype=self._reduce_dtype)
                if self.requires_all_reduce and hsdp_param.replicate_world_size > 1:
                    assert isinstance(hsdp_param.mesh_info, HSDPMeshInfo)
                    reduced_grad, handle = hsdp_param.all_reduce_grad(grad=reduced_grad)

                sharded_grad = hsdp_param.sharded_param.grad
                # grad is flattened, viewed into sharded_param.shape
                sharded_param_local_shape = hsdp_param.sharded_param.local_shape if isinstance(hsdp_param.sharded_param, DTensor) else hsdp_param.sharded_param.shape
                reduced_grad = reduced_grad.view(sharded_param_local_shape)
                to_accumulate_grad = sharded_grad is not None
                if hsdp_param.offload_to_cpu:
                    non_blocking = hsdp_param.pin_memory and not to_accumulate_grad
                    reduced_grad = reduced_grad.to(
                        torch.device("cpu"), non_blocking=non_blocking
                    )
                if sharded_grad is None:
                    hsdp_param.sharded_param.grad = reduced_grad
                else:
                    hsdp_param.sharded_param.grad += reduced_grad
                if hsdp_param.unsharded_accumulated_grad_data is not None:
                    hsdp_param.unsharded_accumulated_grad_data = None
                elif hsdp_param.unsharded_param.grad is not None:
                    hsdp_param.unsharded_param.grad = None

            if self.reshard_after_backward:
                self.reshard()
            if not hsdp_params_with_grad:
                # 没有要处理的梯度
                return

    def grad_comm_reduce(self, hsdp_params_with_grad: List[TorchHSDPParamV2], unsharded_grads: List[torch.Tensor]):
        # TODO: 能够支持处理FSDP和HSDP的场景, 可以先不接入分组ReduceScatter能力，后续再做优化
        for hsdp_param, unsharded_grad in zip(hsdp_params_with_grad, unsharded_grads):
            assert not isinstance(unsharded_grad, DTensor), "unsharded_grad shouldn't be DTensor in hyper-parallel."
            if hsdp_param.shard_world_size > 1:
                reduced_grad = self.platform.reduce_scatter_tensor(
                    unsharded_grad, hsdp_param.mesh_info.shard_process_group, dtype=hsdp_param.reduce_dtype)
            if self.requires_all_reduce and hsdp_param.replicate_world_size > 1 and not hsdp_param.fully_shard:
                assert isinstance(hsdp_param.mesh_info, HSDPMeshInfo)
                reduced_grad = self.platform.all_reduce_tensor(
                    reduced_grad, hsdp_param.mesh_info.replicate_process_group, dtype=hsdp_param.reduce_dtype)

            sharded_grad = hsdp_param.sharded_param.grad
            if sharded_grad is None:
                sharded_grad = reduced_grad
            else:
                hsdp_param.sharded_param.grad += reduced_grad

    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        self.reduce_grads = requires_grad_sync
