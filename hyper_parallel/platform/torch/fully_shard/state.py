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
# pylint: disable=protected-access
from typing import Optional
import torch
import torch.distributed as dist
from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_utils import _get_param_module_infos
from hyper_parallel.core.fully_shard.utils import HSDPMeshInfo, DDPMeshInfo, CPUOffloadPolicy
from hyper_parallel.platform.torch.fully_shard.param import TorchHSDPParamV2
from hyper_parallel.platform.torch.fully_shard.param_group import get_comm_ctx, HSDPParamGroup


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
    # Record grad reduce-scatter handle.
    pre_reduce_scatter_params = []
    # Record grad allreduce handle (only for HSDP).
    pre_all_reduce_params = []

    def __init__(self, cell, mesh_info, config, platform, device):
        """
        Initialize TorchHSDPStateV2.

        Args:
            cell (nn.Module): The module whose parameters are managed by this state.
            mesh_info: Mesh topology for shard/replicate dimensions.
            config (HSDPConfigV2): HSDP configuration.
            platform (TorchPlatform): Torch platform abstraction.
            device (torch.device): Target device.
        """
        super().__init__(cell, mesh_info, config, platform, device)
        self.comm_fusion = config.comm_fusion
        # Do ReduceScatter/AllReduce for grad
        self.device = device
        self.mp_policy = config.mp_policy
        self.offload_policy = config.offload_policy
        self.reduce_grads = True
        # Reshard parameter after backward
        self.reshard_after_backward = True
        # Requires AllReduce for grad When HSDP
        self.requires_all_reduce = True
        # Reduce Op type for gradient reduction, default to AVG.
        self.reduce_op_type = torch.distributed.ReduceOp.AVG
        self._ignored_allreduce_works = []
        self._validate_cpu_offload_params()
        self._reset_sharded_params = False
        self._init_param_group()

    def _init_param_group(self):
        """Initialize fused parameter group for communication fusion.

        When ``comm_fusion`` is enabled, creates an ``HSDPParamGroup`` that packs all
        parameters into a single buffer for fused all-gather and reduce-scatter,
        replacing the per-parameter communication pattern.
        """
        if self.config.comm_fusion:
            # pylint: disable=E1128
            self.param_group = HSDPParamGroup(self.hsdp_params, self.mesh_info, self.device, self.mp_policy)

    def _move_states_to_device(self):
        """move states to device"""
        for mod in self.modules:
            for param in mod.parameters():
                if hasattr(param, "_hsdp_param_initialized") and param._hsdp_param_initialized:
                    continue
                if param.device == self.device or param.device.type == "meta":
                    continue
                param.data = param.to(self.device)
            for buffer in mod.buffers():
                if buffer.device == self.device or buffer.device.type == "meta":
                    continue
                buffer.data = buffer.to(self.device)

    def _init_hsdp_params(self):
        """init hsdp parameters and replicate parameters for cell."""
        replicate_params = self.config.replicate_params
        # all parameters in the module tree(s), deduplicated
        visited_params = set()
        filtered_params = []
        for mod in self.modules:
            for _, param in mod.named_parameters():
                if hasattr(param, "_hsdp_param_initialized") and param._hsdp_param_initialized:
                    continue
                if param in visited_params:
                    continue
                visited_params.add(param)
                filtered_params.append(param)

        module_infos = _get_param_module_infos(filtered_params, tuple(self.modules))
        for param, module_info in zip(filtered_params, module_infos):
            ddp_mesh_info = DDPMeshInfo(mesh=self.mesh_info.mesh, replicate_mesh_dim=0)
            mesh_info = ddp_mesh_info if param in replicate_params else self.mesh_info
            hsdp_param = TorchHSDPParamV2(param,
                                          module_info,
                                          mesh_info,
                                          mp_policy=self.mp_policy,
                                          offload_policy=self.offload_policy,
                                          device=self.device,
                                          )
            if param in replicate_params:
                self.replicate_params.append(hsdp_param)
            else:
                self.hsdp_params.append(hsdp_param)
                if hsdp_param.is_sharded:
                    self.sharded_hsdp_params.append(hsdp_param)

    def _init_mp_dtypes(self):
        """init mp dtypes for hsdp parameters and replicate parameters"""
        for hsdp_param in self.hsdp_params:
            hsdp_param.init_dtype_attrs(self.mp_policy)
        for replicate_param in self.replicate_params:
            replicate_param.init_dtype_attrs(self.mp_policy)
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
        """Validate that all parameters are on CPU when CPU offload policy is enabled."""
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
                f"{[(p._param_fqn, p.sharded_param.device) for p in hsdp_params_not_on_cpu]}\n"
            )

    def lazy_init(self):
        if not self._reset_sharded_params:
            for hsdp_param in self.hsdp_params:
                if hsdp_param.is_sharded:
                    hsdp_param.reset_sharded_param()
            self._reset_sharded_params = True
        self._validate_no_meta_params()
        self._validate_cpu_offload_params()
        self._init_mp_dtypes()

    def _validate_no_meta_params(self):
        param_names_on_meta = [
            hsdp_param._param_fqn
            for hsdp_param in self.hsdp_params
            if hsdp_param.sharded_param.device.type == "meta"
        ]
        if param_names_on_meta:
            raise RuntimeError(
                "HSDP parameters should be materialized from meta device before training, "
                f"but the following were still on meta device: {param_names_on_meta}\n"
                "For example, call module.to_empty(device) to materialize to device and "
                "call module.reset_parameters() on each module to initialize values."
            )

    def _allreduce_replicate_params(self, async_op=True) -> None:
        """
        DDP-style all-reduce for parameters in config.replicate_params.

        Do one all-reduce over the flattened 2D mesh so the final
        gradient is reduced over the full mesh.
        """
        for param in self.replicate_params:
            if not hasattr(param, "_unsharded_param") or param.unsharded_param is None:
                continue

            reduced_grad = _to_dtype_if_needed(param.unsharded_param.grad, self._reduce_dtype)
            flat_name = "reduce_all"
            flat_mesh = self.mesh_info.mesh.flatten(mesh_dim_name=flat_name)
            reduce_group = flat_mesh.get_group(flat_name)
            if reduce_group is not None and reduce_group.size() > 1:
                param.all_reduce_handle = torch.distributed.all_reduce(
                    reduced_grad, group=reduce_group, op=self.reduce_op_type, async_op=async_op
                )
            self._ignored_allreduce_works.append((param, reduced_grad))

    def _finish_ignored_allreduce(self) -> None:
        """
        Wait for async all-reduce of replicate_params and materialize param.grad.

        For each pending work, this:
          Waits on all associated handles to complete;
          Casts reduced_grad back to _orig_dtype if needed;
          Assigns the final tensor to param.grad.
        """
        if not self._ignored_allreduce_works:
            return

        need_synchronize = False
        for param, reduced_grad in self._ignored_allreduce_works:
            if param.all_reduce_handle:
                param.all_reduce_handle.wait()
            if self._orig_dtype is not None and reduced_grad.dtype != self._orig_dtype:
                reduced_grad = reduced_grad.to(self._orig_dtype)
            sharded_grad = param.sharded_param.grad
            to_accumulate_grad = sharded_grad is not None
            if param.offload_to_cpu:
                non_blocking = param.pin_memory and not to_accumulate_grad
                reduced_grad = reduced_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
                need_synchronize = True
            if sharded_grad is None:
                param.sharded_param.grad = param.to_sharded_dtensor(reduced_grad)
            else:
                param.sharded_param.grad._local_tensor += reduced_grad

            if param.unsharded_accumulated_grad_data is not None:
                param.unsharded_accumulated_grad = None
            elif param.unsharded_param.grad is not None:
                param.unsharded_param.grad = None

        if need_synchronize:
            if self.device.type == "npu":
                torch.npu.current_stream().synchronize()
            elif self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            else:
                raise NotImplementedError(
                    f"Unsupported device type {self.device.type} for synchronization after CPU offload.")

        self._ignored_allreduce_works.clear()

    def post_backward_for_comm_fusion(self):
        """post_backward_for_comm_fusion."""
        # Fused gradient reduction path: first apply any pending async reduction
        # from the previous module's backward (pipelined overlap), then issue
        # this module's fused reduce-scatter (+ all-reduce for HSDP).
        comm_ctx = get_comm_ctx()
        # Phase 2: apply grads for the param group whose all_reduce is done
        if comm_ctx.all_reduce_param_group is not None:
            comm_ctx.all_reduce_param_group.wait_all_reduce_and_apply_grad()
            comm_ctx.all_reduce_param_group = None
        # Phase 1: wait reduce_scatter, issue async all_reduce for previous layer
        if comm_ctx.pre_param_group is not None:
            comm_ctx.pre_param_group.wait_reduce_scatter_and_issue_all_reduce()
            comm_ctx.pre_param_group = None
        self.param_group.foreach_reduce(
            reduce_scatter_reduce_op=self.reduce_op_type
        )

    def post_backward(self, *unused):  # pylint: disable=unused-argument
        """Reduce gradients and reshard parameters after backward."""
        for hsdp_param in self.hsdp_params:
            hsdp_param.accumulate_unsharded_grad_if_needed()
        for replicate_param in self.replicate_params:
            replicate_param.accumulate_unsharded_grad_if_needed()
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.shard()
            for hsdp_param in self.hsdp_params:
                hsdp_param.to_accumulated_grad_if_needed()
            for replicate_param in self.replicate_params:
                replicate_param.to_accumulated_grad_if_needed()
            return
        self._allreduce_replicate_params()
        if not self.comm_fusion:
            self.reduce_params()
            for hsdp_param in self.hsdp_params:
                if not hasattr(hsdp_param, "_unsharded_param") or hsdp_param.unsharded_param is None:
                    continue
                # Frozen parameters (requires_grad=False) produce no
                # gradient — skip all reduce-scatter / all-reduce work.
                if not hsdp_param.sharded_param.requires_grad:
                    continue
                if hsdp_param.shard_world_size > 1:
                    hsdp_param.reduce_scatter_grad(
                        dtype=self._reduce_dtype,
                        reduce_op=self.reduce_op_type
                    )
                    TorchHSDPStateV2.pre_reduce_scatter_params.append([hsdp_param, self._orig_dtype])

                if self.requires_all_reduce and hsdp_param.replicate_world_size > 1:
                    assert isinstance(hsdp_param.mesh_info, HSDPMeshInfo)
                    reduced_grad = hsdp_param.reduce_scatter_output()
                    hsdp_param.all_reduce_grad(grad=reduced_grad, dtype=self._reduce_dtype, reduce_op=self.reduce_op_type)
                    if TorchHSDPStateV2.pre_reduce_scatter_params and \
                            TorchHSDPStateV2.pre_reduce_scatter_params[-1][0] == hsdp_param:
                        TorchHSDPStateV2.pre_reduce_scatter_params.pop()
                    TorchHSDPStateV2.pre_all_reduce_params.append([hsdp_param, self._orig_dtype])
        else:
            self.post_backward_for_comm_fusion()
        self._finish_ignored_allreduce()
        if self.reshard_after_backward:
            self.shard()

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
            pre_hsdp_param, pre_orig_dtype = TorchHSDPStateV2.pre_reduce_scatter_params.pop(0)
            reduced_grad = pre_hsdp_param.reduce_scatter_output()
            pre_hsdp_param.clear_reduce_scatter_output()
            need_synchronize = pre_hsdp_param.apply_reduced_grad(reduced_grad, pre_orig_dtype) or need_synchronize

        while TorchHSDPStateV2.pre_all_reduce_params:
            pre_hsdp_param, pre_orig_dtype = TorchHSDPStateV2.pre_all_reduce_params.pop(0)
            reduced_grad = pre_hsdp_param.all_reduce_output()
            pre_hsdp_param.clear_all_reduce_output()
            need_synchronize = pre_hsdp_param.apply_reduced_grad(reduced_grad, pre_orig_dtype) or need_synchronize
        if need_synchronize:
            if self.device.type == "npu":
                torch.npu.current_stream().synchronize()
            elif self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            else:
                raise NotImplementedError(
                    f"Unsupported device type {self.device.type} for synchronization after CPU offload.")

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
            raise ValueError(
                f"Unsupported reduce op type {reduce_op_type}, "
                f"supported types are {list(fsdp_support_reduce_op.keys())}")
        reduce_op: str = reduce_op_type.lower().strip()
        self.reduce_op_type = fsdp_support_reduce_op[reduce_op]
