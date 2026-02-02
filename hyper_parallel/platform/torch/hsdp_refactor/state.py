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
from torch import Tensor
from hyper_parallel.core.hsdp_refactor.hsdp_state import HSDPState
from hyper_parallel.platform.torch.hsdp_refactor.param import TorchHSDPParamV2
from hyper_parallel.platform.torch.platform import TorchPlatform
from hyper_parallel.core.hsdp_refactor.hsdp_utils import FSDPSchedulerState

class TorchHSDPStateV2(HSDPState):
    """Torch HSDP cell state"""
    def __init__(self, cell, config, platform):
        super().__init__(cell, config, platform)
        self._init_mp_dtypes()
        self.reduce_grads = True
        self.reshard_after_backward = True
        self.all_reduce_grads = True

    def _init_hsdp_params(self):
        """init hsdp parameters for cell"""
        params = self.cell.named_parameters()
        for param_name, param in params:
            if hasattr(param, "has_hsdp_param"):
                continue
            hsdp_param = TorchHSDPParamV2(self.cell, param_name, param, self.config, self.platform)
            param.has_hsdp_param = True
            self.hsdp_params.append(hsdp_param)
            if hsdp_param.sharded:
                # TODO: 这个可能不需要了，后续根据mesh处理是否切分。
                self.sharded_hsdp_params.append(hsdp_param)

    def _init_mp_dtypes(self,):
        """init mp dtypes for hsdp parameters"""
        # TODO(fuchao): Get correct self._orig_dtypes, self._reduce_dtype
        for hsdp_param in self.hsdp_params:
            hsdp_param.init_mp_dtype()
        orig_dtypes = None
        reduce_dtypes = None

        self._orig_dtypes = None
        self._reduce_dtype = None


    def lazy_init(self):
        raise NotImplementedError("lazy_init not implemented in TorchHSDPStateV2")
        pass

    def reshard(self,):
        # TODO：补齐reshard接口，当前我们不考虑reshard_after_forward配置是int的情况，只考虑True/False
        if self.scheduler_state == FSDPSchedulerState.FORWARD:
            if not self.reshard_after_forward:
                return
            if self._use_post_forward_mesh:
                # TODO: support reshard_after_forward=(int)
                raise NotImplementedError(f"For reshard, need support reshard_after_forward=(int).")
                self._to_sharded_post_forward()
                self._reshard_after_forward_event = self.device_handle.Event()
                if self._reshard_after_forward_event is not None:
                    self._reshard_after_forward_event.record()
                return
        self.hsdp_state.shard()

    def post_backward(self, *unused):
        # TODO: reshard + 梯度处理
        self._training_state = FSDPSchedulerState.POST_BACKWARD
        for hsdp_param in self.hsdp_params:
            hsdp_param.acc_grad_if_needed()
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.reshard()
            for hsdp_param in self.hsdp_params:
                hsdp_param.to_accumulated_grad_if_needed()
            return
        hsdp_params_with_grad: list[TorchHSDPParamV2] = []
        unsharded_grads: list[torch.Tensor] = []
        for hsdp_param in self.hsdp_params:
            if not hasattr(hsdp_param, '_unsharded_param'):
                continue
            if hsdp_param.unsharded_accumulated_grad_data is not None:
                hsdp_params_with_grad.append(hsdp_param)
                unsharded_grads.append(hsdp_param.unsharded_accumulated_grad_data)
            elif hsdp_param._unsharded_param.grad is not None:
                hsdp_params_with_grad.append(hsdp_param)
                unsharded_grads.append(hsdp_param._unsharded_param.grad)
        if self.reshard_after_backward:
            self.reshard()
        if not hsdp_params_with_grad:
            # 没有要处理的梯度
            return
        self.grad_comm_reduce(hsdp_params_with_grad, unsharded_grads)

    def grad_comm_reduce(self, hsdp_params_with_grad, unsharded_grads):
        # TODO: 能够支持处理FSDP和HSDP的场景, 可以先不接入分组ReduceScatter能力，后续再做优化
        raise NotImplementedError(f"grad_comm_reduce not implemented in TorchHSDPStateV2")

