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
"""Torch HSDP cell state"""
from typing import List, Optional
import torch
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_utils import _get_param_module_infos
from hyper_parallel.platform.torch.fully_shard.param import TorchHSDPParamV2
from hyper_parallel.platform.torch.fully_shard.utils import HSDPMeshInfo, CPUOffloadPolicy


def _to_dtype_if_needed(
        tensor: torch.Tensor, dtype: Optional[torch.dtype]
) -> torch.Tensor:
    """Cast tensor to the given dtype if it differs from current dtype.

    Args:
        tensor: The input tensor to potentially cast.
        dtype: Target dtype. If None or same as tensor dtype, no-op.
    """
    if dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor

class TorchHSDPStateV2(HSDPState):
    """Torch HSDP cell state"""
    pre_reduce_scatter_params = []
    pre_all_reduce_params = []

    def __init__(self, cell, mesh_info, config, platform, device):
        super().__init__(cell, mesh_info, config, platform, device)
        # Do ReduceScatter/AllReduce for grad
        self.device = device
        self.mp_policy = config.mp_policy
        self.offload_policy = config.offload_policy
        self.reduce_grads = True
        # Reshard parameter after backward
        self.reshard_after_backward = True
        # Requires AllReduce for grad When HSDP
        self.requires_all_reduce = True
        self._use_post_forward_mesh = False
        # Reduce Op type for gradient reduction, default to AVG.
        self.reduce_op_type = torch.distributed.ReduceOp.AVG
        self._validate_cpu_offload_params()
        self._init_mp_dtypes()

    def _move_states_to_device(self):
        """move states to device"""
        # TODO: @celia DTensor support
        for param in self.cell.parameters():
            if hasattr(param, "_hsdp_param_initialized") and param._hsdp_param_initialized:
                continue
            if param.device == self.device or param.device.type == "meta":
                continue
            param.data = param.to(self.device)
        for buffer in self.cell.buffers():
            if buffer.device == self.device or buffer.device.type == "meta":
                continue
            buffer.data = buffer.to(self.device)

    def _init_hsdp_params(self):
        """init hsdp parameters for cell"""
        # Cell 树内的全部parameters
        filtered_params = []
        for _, param in self.cell.named_parameters():
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
                                          offload_policy=self.offload_policy,
                                          device=self.device,
                                          )
            self.hsdp_params.append(hsdp_param)
            if hsdp_param.is_sharded:
                # TODO: 这个可能不需要了，后续根据mesh处理是否切分。
                self.sharded_hsdp_params.append(hsdp_param)

    def _init_mp_dtypes(self):
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

    def _validate_cpu_offload_params(self):
        if not isinstance(self.offload_policy, CPUOffloadPolicy):
            return
        hsdp_params_not_on_cpu = [
            hsdp_param
            for hsdp_param in self.hsdp_params
            if hsdp_param.sharded_param.device.type != "cpu"
        ]
        if hsdp_params_not_on_cpu:
            raise RuntimeError(
                "HSDP parameters should be materialized on CPU when enabling CPU offloading. "
                'For example, load a CPU state dict or call module.to_empty(device="cpu"). '
                "Found following parameters on non-CPU device: "
                f"{[(hsdp_param._param_fqn, hsdp_param.sharded_param.device) for hsdp_param in hsdp_params_not_on_cpu]}\n"
            )

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
        for hsdp_param in self.hsdp_params:
            hsdp_param.accumulate_unsharded_grad_if_needed()
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.reshard()
            for hsdp_param in self.hsdp_params:
                hsdp_param.to_accumulated_grad_if_needed()
            return
        # reduce pre post_backward parameters, overlap with computing
        self.reduce_params()
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
                hsdp_param.reduce_scatter_grad(
                    dtype=self._reduce_dtype,
                    reduce_op=self.reduce_op_type
                )
                TorchHSDPStateV2.pre_reduce_scatter_params.append(hsdp_param)

            if self.requires_all_reduce and hsdp_param.replicate_world_size > 1:
                assert isinstance(hsdp_param.mesh_info, HSDPMeshInfo)
                reduced_grad = hsdp_param.reduce_scatter_output()
                hsdp_param.all_reduce_grad(grad=reduced_grad, dtype=self._reduce_dtype, reduce_op=self.reduce_op_type)
                if TorchHSDPStateV2.pre_reduce_scatter_params and \
                        TorchHSDPStateV2.pre_reduce_scatter_params[-1] == hsdp_param:
                    TorchHSDPStateV2.pre_reduce_scatter_params.pop()
                TorchHSDPStateV2.pre_all_reduce_params.append(hsdp_param)

        if self.reshard_after_backward:
            self.reshard()

    def reduce_params(self):
        """Apply reduced gradients from pre-staged HSDP parameters to sharded parameters.

        This function processes two lists of pre-queued HSDP parameters (`pre_reduce_scatter_params`
        and `pre_all_reduce_params`), retrieves the reduced gradients from asynchronous
        reduce-scatter/all-reduce operations, clears cached communication outputs, and applies
        the reduced gradients to the corresponding sharded parameters (including reshaping,
        dtype conversion, optional CPU offloading, and gradient accumulation/assignment).

        Note:
            - Parameters are processed in **FIFO (First-In-First-Out)** order (via `pop(0)`), ensuring
              gradient application order matches the order of gradient reduction operations.
            - After retrieving the reduced gradient, the cached communication output (reduce_scatter_output
              or all_reduce_output) is cleared to free memory and avoid stale data.
            - Gradient application logic (in `apply_reduced_grad`) includes:
              1. Reshaping the flat reduced gradient to match the local shard shape
              2. Optional dtype conversion to `param_type`
              3. Optional CPU offloading (per the HSDP parameter's offload policy)
              4. Assigning or accumulating the gradient to `sharded_param.grad`
        """
        need_synchronize = False
        while TorchHSDPStateV2.pre_reduce_scatter_params:
            pre_hsdp_param = TorchHSDPStateV2.pre_reduce_scatter_params.pop(0)
            reduced_grad = pre_hsdp_param.reduce_scatter_output()
            pre_hsdp_param.clear_reduce_scatter_output()
            need_synchronize = pre_hsdp_param.apply_reduced_grad(reduced_grad, self._orig_dtype) or need_synchronize

        while TorchHSDPStateV2.pre_all_reduce_params:
            pre_hsdp_param = TorchHSDPStateV2.pre_all_reduce_params.pop(0)
            reduced_grad = pre_hsdp_param.all_reduce_output()
            pre_hsdp_param.clear_all_reduce_output()
            need_synchronize = pre_hsdp_param.apply_reduced_grad(reduced_grad, self._orig_dtype) or need_synchronize
        if need_synchronize:
            if self.device.type == "npu":
                torch.npu.current_stream().synchronize()
            elif self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            else:
                raise NotImplementedError(f"Unsupported device type {self.device.type} for synchronization after CPU offload.")

    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        self.reduce_grads = requires_grad_sync

    def set_reduce_op_type(self, reduce_op_type: str):
        """set reduce op type for gradient reduction."""
        fsdp_support_reduce_op = {
            "sum": torch.distributed.ReduceOp.SUM,
            "avg": torch.distributed.ReduceOp.AVG,
        }
        if reduce_op_type not in fsdp_support_reduce_op:
            raise ValueError(f"Unsupported reduce op type {reduce_op_type}, supported types are {list(fsdp_support_reduce_op.keys())}")
        reduce_op: str = reduce_op_type.lower().strip()
        self.reduce_op_type = fsdp_support_reduce_op[reduce_op]
