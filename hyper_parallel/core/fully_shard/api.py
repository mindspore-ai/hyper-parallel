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
"""hybrid shard data parallel interface"""
from torch import nn
from typing import cast
from hyper_parallel.platform.platform import PlatformType
from hyper_parallel import DeviceMesh
from hyper_parallel.platform import get_platform

platform = get_platform()

origin_class_to_extend_class = {}


class _UnshardHandle:
    def __init__(self, hsdp_state=None):
        self._hsdp_state = hsdp_state

    def wait(self):
        if self._hsdp_state is not None:
            self._hsdp_state.wait_for_unshard()
            self._hsdp_state = None


class HSDPModule:
    """
    The hsdp block of neural networks with hsdp interface.

    Supported Platforms:
        ``MindSpore`` ``torch``
    """

    # pylint: disable=C0415
    def hsdp_init(self, platform_type, module, mesh, reshard_after_forward,
                  shard_placement_fn, mp_policy, offload_policy, ignored_params, device):
        """init hsdp2 scheduler."""
        scheduler_class = None
        if platform_type == PlatformType.MINDSPORE:
            from hyper_parallel.platform.mindspore.hsdp.scheduler import MindSporeHSDPScheduler
            scheduler_class = MindSporeHSDPScheduler
        else:
            from hyper_parallel.platform.torch.fully_shard.scheduler import TorchHSDPSchedulerV2
            scheduler_class = TorchHSDPSchedulerV2

        self.hsdp_scheduler = scheduler_class(module,
                                              mesh,
                                              reshard_after_forward,
                                              shard_placement_fn,
                                              mp_policy,
                                              offload_policy,
                                              ignored_params,
                                              device,
                                              )

    def set_requires_gradient_sync(self, requires_grad_sync):
        r"""
            set requires grad sync flag.
            Args:
                requires_grad_sync(bool): requires_grad_sync is used to control gradient sync process.
            Raises:
                ValueError: If `requires_grad_sync` is not bool.
        """
        if not isinstance(requires_grad_sync, bool):
            raise ValueError(f"requires_grad_sync must be bool but got {requires_grad_sync}.")
        if not hasattr(self, "hsdp_scheduler"):
            raise ValueError("call hsdp interface first.")

        for _, module in platform.get_cells_and_names(self):
            if isinstance(module, HSDPModule):
                module.hsdp_scheduler.set_requires_grad_sync(requires_grad_sync)

    def zero_grads(self):
        """zero accumunication grads"""
        if not hasattr(self, "hsdp_scheduler"):
            raise ValueError("call hsdp interface first.")
        if platform == PlatformType.PYTORCH:
            raise RuntimeError("zero_grads shouldn't be called in torch platform, use optimizer.zero_grad() instead.")
        for _, module in platform.get_cells_and_names(self):
            if isinstance(module, HSDPModule):
                module.hsdp_scheduler.zero_grads()

    def set_modules_to_forward_prefetch(self, modules):
        """set forward prefetch module list to prefetch all gather for unsharded parameters"""
        if not isinstance(modules, (tuple, list)):
            raise ValueError("modules must be HSDPModule list")
        for module in modules:
            if not isinstance(module, HSDPModule):
                raise ValueError(f"modules must be HSDPModule list but got {type(module)} in list.")
        if not hasattr(self, "hsdp_scheduler"):
            raise ValueError("call hsdp interface first.")
        self.hsdp_scheduler.set_forward_prefetch_cells(modules)

    def set_modules_to_backward_prefetch(self, modules):
        """set backward prefetch module list to prefetch all gather for unsharded parameters"""
        if not isinstance(modules, (tuple, list)):
            raise ValueError("modules must be HSDPModule list")
        for module in modules:
            if not isinstance(module, HSDPModule):
                raise ValueError(f"modules must be HSDPModule list but got {type(module)} in list.")
        if not hasattr(self, "hsdp_scheduler"):
            raise ValueError("call fully_shard interface first.")
        self.hsdp_scheduler.set_backward_prefetch_cells(modules)

    def reshard(self) -> None:
        """reshard all sharded parameters"""
        if not self.hsdp_scheduler:
            raise ValueError("hsdp_scheduler is None")
        scheduler_state = self.hsdp_scheduler.scheduler_state
        if scheduler_state:
            scheduler_state.shard()

    def unshard(self, async_op: bool = False):
        """unshard all sharded parameters"""
        if not isinstance(async_op, bool):
            raise ValueError(f"async_op should be a bool, got {type(async_op)}")
        if not self.hsdp_scheduler:
            raise ValueError("hsdp_scheduler is None")
        scheduler_state = self.hsdp_scheduler.scheduler_state
        if scheduler_state:
            scheduler_state.unshard(async_op=async_op)
            if async_op:
                return _UnshardHandle(hsdp_state=scheduler_state)
        return None

    def set_is_last_backward(self, is_last_backward: bool):
        """set is_last_backward flag"""
        self.hsdp_scheduler.scheduler_ctx.is_last_backward = is_last_backward

    def set_requires_all_reduce(self, requires_all_reduce: bool, *, recurse: bool = True) -> None:
        """set requires_all_reduce flag"""
        if not isinstance(requires_all_reduce, bool):
            raise ValueError(
                f"requires_all_reduce should be a bool, got {type(requires_all_reduce)}"
            )
        if not recurse:
            raise NotImplementedError(f"Currently impl is equal to recurse=True,\
                                      need support module_param mapping.")
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, HSDPModule):
                module.hsdp_scheduler.set_requires_all_reduce(requires_all_reduce)

    def set_reshard_after_forward(self, reshard_after_forward: bool, recurse: bool = True) -> None:
        """set reshard_after_forward flag"""
        if not isinstance(reshard_after_forward, bool):
            raise ValueError(
                f"reshard_after_forward should be a bool, got {type(reshard_after_forward)}"
            )
        if not recurse:
            raise NotImplementedError(f"Currently impl is equal to recurse=True,\
                                      need support module_param mapping.")
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, HSDPModule):
                module.hsdp_scheduler.set_reshard_after_forward(reshard_after_forward)

    def set_reshard_after_backward(self, reshard_after_backward: bool, recurse: bool = True) -> None:
        """set reshard_after_backward flag"""
        if not isinstance(reshard_after_backward, bool):
            raise ValueError(
                f"reshard_after_backward should be a bool, got {type(reshard_after_backward)}"
            )
        if not recurse:
            raise NotImplementedError(f"Currently impl is equal to recurse=True,\
                                      need support module_param mapping.")
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, HSDPModule):
                module.hsdp_scheduler.set_reshard_after_backward(reshard_after_backward)


def _extend_module_with_hsdp_interface(module):
    """extend Module with HSDPModule interface"""
    origin_class = module.__class__
    extend_class = origin_class_to_extend_class.get(origin_class, None)
    if extend_class is None:
        extend_class = type(f"HSDP{origin_class.__name__}", (HSDPModule, origin_class), {})
        origin_class_to_extend_class[origin_class] = extend_class
    module.__class__ = extend_class


# pylint: disable=C0415
def _check_module_valid(platform_type, module):
    """check module valid"""
    if platform_type == PlatformType.MINDSPORE:
        from mindspore.nn.cell import Cell
        if not isinstance(module, Cell):
            raise ValueError(f"module's type must be nn.cell but got {type(module)}.")
    else:
        from torch.nn import Module
        if not isinstance(module, Module):
            raise ValueError(f"module's type must be nn.Module but got {type(module)}.")


# pylint: disable=C0415
def _check_hsdp_input_valid(platform_type, module, shard_size, threshold, optimizer_level, enable_grad_accumulation,
                            grad_scale, reduce_dtype, comm_async, comm_fusion, bucket_size):
    """check hsdp input valid"""
    _check_module_valid(platform_type, module)
    if not isinstance(shard_size, int) or (shard_size <= 0 and shard_size != -1):
        raise ValueError(f"shard_size must be a positive integer, but got {shard_size}.")
    if not isinstance(threshold, int) or threshold < 0:
        raise ValueError(f"threshold must be a positive integer or 0, but got {threshold}.")
    if optimizer_level not in ["level1", "level2", "level3"]:
        raise ValueError(f"Optimizer level should in ['level1', 'level2', 'level3'], but got {optimizer_level}.")
    if not isinstance(enable_grad_accumulation, bool):
        raise ValueError(f"enable_grad_accumulation must be bool but got {enable_grad_accumulation}.")
    if not isinstance(grad_scale, float):
        raise ValueError(f"grad_scale must be float but got {grad_scale}.")
    if platform_type == PlatformType.MINDSPORE:
        from mindspore._c_expression.typing import Type
        if reduce_dtype is not None and not isinstance(reduce_dtype, Type):
            raise ValueError(f"reduce_dtype must be mindspore.dtype but got {reduce_dtype}.")
    else:
        import torch
        if reduce_dtype is not None and not isinstance(reduce_dtype, torch.dtype):
            raise ValueError(f"reduce_dtype must be torch.dtype but got {reduce_dtype}.")
    if not isinstance(comm_async, bool):
        raise ValueError(f"comm_async must be bool but got {comm_async}.")
    if not isinstance(comm_fusion, bool):
        raise ValueError(f"comm_fusion must be bool but got {comm_fusion}.")
    if not isinstance(bucket_size, int) or (bucket_size < 0 and bucket_size != -1):
        raise ValueError(f"bucket_size must be a positive integer or 0, but got {bucket_size}.")


def fully_shard(
        module: nn.Module,
        *,
        mesh: DeviceMesh | None = None,
        reshard_after_forward: bool | int | None = None,
        shard_placement_fn: None = None,
        mp_policy: None = None,
        offload_policy: None = None,
        ignored_params: set[nn.Parameter] | None = None,
        device = None,
):
    platform_type = platform.platform_type
    _extend_module_with_hsdp_interface(module)
    # TODO: mindspore does not support get_device_handle
    if device is None:
        import torch
        device_handle = platform.get_device_handle()  # return torch.npu or torch.cuda
        if device_handle.is_available():
            device = torch.device(device_handle.current_device())
        else:
            device = torch.device("cpu")
    module.hsdp_init(
        platform_type,
        module,
        mesh,
        reshard_after_forward,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        ignored_params,
        device,
    )
    return module


def hsdp_sync_stream():
    """wait for hsdp gradient handle to be completed"""
    if platform is None:
        return
    platform.wait_grad_handle()
