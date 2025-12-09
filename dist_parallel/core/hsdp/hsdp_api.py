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
from typing import Optional, Any
from dist_parallel.core.hsdp.hsdp_utils import PlatformType, OptimizerLevel

origin_class_to_extend_class = {}
optimizer_level_map = {
    "level1": OptimizerLevel.SHARD_OPT,
    "level2": OptimizerLevel.SHARD_OPT_GRAD,
    "level3": OptimizerLevel.SHARD_OPT_GRAD_PARAM,
}
current_platform = None

class HSDPCell:
    """
    The hsdp block of neural networks with hsdp interface.

    Supported Platforms:
        ``MindSpore`` ``torch``
    """
    def hsdp_init(self, platform_type, cell, shard_size, threshold, optimizer_level, enable_grad_accumulation,
                  grad_scale, reduce_dtype, comm_async, comm_fusion, bucket_size):
        """init hsdp scheduler."""
        scheduler_class = None
        self.platform_type = platform_type
        global current_platform
        if self.platform_type == PlatformType.MINDSPORE:
            from dist_parallel.platform.mindspore.platform import MindSporePlatform
            from dist_parallel.platform.mindspore.hsdp.scheduler import MindSporeHSDPScheduler
            current_platform = MindSporePlatform()
            scheduler_class = MindSporeHSDPScheduler
        else:
            from dist_parallel.platform.torch.platform import TorchPlatform
            from dist_parallel.platform.torch.hsdp.scheduler import TorchHSDPScheduler
            current_platform = TorchPlatform()
            scheduler_class = TorchHSDPScheduler
            

        self.hsdp_scheduler = scheduler_class(cell,
                                              shard_size,
                                              threshold,
                                              optimizer_level,
                                              enable_grad_accumulation,
                                              grad_scale,
                                              reduce_dtype,
                                              comm_async,
                                              comm_fusion,
                                              bucket_size)

    def set_requires_grad_sync(self, requires_grad_sync):
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
        self.hsdp_scheduler.set_requires_grad_sync(requires_grad_sync)
        for sub_cell in self.cells():
            if isinstance(sub_cell, HSDPCell):
                sub_cell.set_requires_grad_sync(requires_grad_sync)

    def zero_grads(self):
        """zero accumunication grads"""
        if not hasattr(self, "hsdp_scheduler"):
            raise ValueError("call hsdp interface first.")
        self.hsdp_scheduler.zero_grads()
        for sub_cell in self.cells():
            if isinstance(sub_cell, HSDPCell):
                sub_cell.zero_grads()

    def set_forward_prefetch_cells(self, hsdp_cell_list):
        """set forward prefetch cell list to prefetch all gather for unsharded parameters"""
        if not isinstance(hsdp_cell_list, (tuple, list)):
            raise ValueError("hsdp_cell_list must be HSDPCell list")
        for cell in hsdp_cell_list:
            if not isinstance(cell, HSDPCell):
                raise ValueError(f"hsdp_cell_list must be HSDPCell list but got {type(cell)} in list.")
        if not hasattr(self, "hsdp_scheduler"):
            raise ValueError("call hsdp interface first.")
        self.hsdp_scheduler.set_forward_prefetch_cells(hsdp_cell_list)

    def set_backward_prefetch_cells(self, hsdp_cell_list):
        """set backward prefetch cell list to prefetch all gather for unsharded parameters"""
        if not isinstance(hsdp_cell_list, (tuple, list)):
            raise ValueError("hsdp_cell_list must be HSDPCell list")
        for cell in hsdp_cell_list:
            if not isinstance(cell, HSDPCell):
                raise ValueError(f"hsdp_cell_list must be HSDPCell list but got {type(cell)} in list.")
        if not hasattr(self, "hsdp_scheduler"):
            raise ValueError("call hsdp interface first.")
        self.hsdp_scheduler.set_backward_prefetch_cells(hsdp_cell_list)

def _extend_cell_with_hsdp_interface(cell):
    """extend Cell with HSDPCell interface"""
    origin_class = cell.__class__
    extend_class = origin_class_to_extend_class.get(origin_class, None)
    if extend_class is None:
        extend_class = type(f"HSDP{origin_class.__name__}", (HSDPCell, origin_class), {})
        origin_class_to_extend_class[origin_class] = extend_class
    cell.__class__ = extend_class

def _check_hsdp_input_valid(platform_type, shard_size, threshold, optimizer_level, enable_grad_accumulation, grad_scale,
                            reduce_dtype, comm_async, comm_fusion, bucket_size):
    """check hsdp input valid"""
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

def hsdp(
        cell,
        shard_size: Optional[int] = -1,
        threshold: Optional[int] = 64,
        optimizer_level: Optional[str] = "level1",
        enable_grad_accumulation: Optional[bool] = False,
        grad_scale: Optional[float] = 1.0,
        reduce_dtype: Optional[Any] = None,
        comm_async: Optional[bool] = False,
        comm_fusion: Optional[bool] = False,
        bucket_size: Optional[int] = -1
):
    r"""
        apply hybrid sharded data parallel.

        Args:
            cell(Cell|Module): The cell to apply hsdp.
            shard_size (int, optional): Set the optimizer weight shard group size if you want to specific the
                maximum group size across devices. The numerical range can be (0, device_num] or -1. Default value
                is -1, which means the optimizer weight shard group size will be
                the data parallel group of each parameter.
            threshold (int, optional): Set the threshold of parallel optimizer. When parallel optimizer is
                enabled, parameters with size smaller than this threshold will not be
                sharded across the devices. Parameter size = shape[0] \* ... \*
                shape[n] \* size(dtype). Non-negative. Unit: KB. Default: 64.
            optimizer_level (str, optional): optimizer_level configuration is used to specify
                the splitting level for optimizer sharding. It is important to note that the implementation
                of optimizer sharding in static graph is inconsistent with dynamic graph like megatron,
                but the memory optimization effect is the same.
                It must be one of [ ``level1``, ``level2``, ``level3`` ]. Default: ``level1``.

                - level1:
                  Splitting is performed on weights and optimizer state.
                - level2:
                  Splitting is performed on weights, optimizer state, and gradients.
                - level3:
                  Splitting is performed on weights, optimizer state,
                  gradients, additionally, before the backward pass, the weights are further applied with
                  allgather communication to release the memory used by the forward pass allgather.
            enable_grad_accumulation (bool, optional): enable gradient accumulation. When gradient accumulation is
                enable, gradient synchronization should be explicitly called by `set_requires_grad_sync` interface.
            grad_scale (float, optional): gradient will scale with grad_scale.
            reduce_dtype (float, optional): gradient reduce dtype. Default value is None, which means gradient
                will be reduced with its origin dtype.
            comm_async (bool, optional): reduce gradient with async communication op for communication overlap.
                When comm_async is enable, ``hsdp_wait_grad_handle`` should be called before using generated
                gradient. Default value is False, which means gradient will be reduced with sync communication op.
            comm_fusion (bool, optional): fuse forward parameter allgathers and backward gradient reducescatters or
                allreduces into buffers communication to reduce the number of communication op. ``bucket_size` will
                further control the size of backward gradient reduce buffer size.
                Default value is False, which means communication op is not fused and will run one by one.
            bucket_size (int, optional): bucket_size is used to control the size of comm fusion buffer. Unit: KB.
                Default value is -1, which means gradient will be fused into a buffer. When value is 0, which means
                gradients will not be fused, each gradient acts as a buffer.

        Raises:
            ValueError: If the `cell` is not a cell.
            ValueError: If the `shard_size` is not a positive integer or -1.
            ValueError: If `threshold` is not a positive integer or 0.
            ValueError: If `optimizer_level` is not one of the [ ``level1``, ``level2``, ``level3`` ].
            ValueError: If `enable_grad_accumulation` is not bool.
            ValueError: If `grad_scale` is not float.
            ValueError: If `reduce_dtype` is not mindspore.dtype.
            ValueError: If `comm_async` is not bool.
            ValueError: If `comm_fusion` is not bool.
            ValueError: If the `bucket_size` is not a positive integer or -1.
        """
    platform_type = PlatformType.MINDSPORE
    try:
        from mindspore.nn.cell import Cell
        if not isinstance(cell, Cell):
            raise ValueError(f"cell's type must be nn.Module but got {type(cell)}.")
    except ImportError:
        from torch.nn import Module as Cell
        platform_type = PlatformType.PYTORCH
        if not isinstance(cell, Cell):
            raise ValueError(f"cell's type must be nn.Module but got {type(cell)}.")

    _check_hsdp_input_valid(
        platform_type,
        shard_size,
        threshold,
        optimizer_level,
        enable_grad_accumulation,
        grad_scale,
        reduce_dtype,
        comm_async,
        comm_fusion,
        bucket_size
    )
    optimizer_level = optimizer_level_map.get(optimizer_level)
    _extend_cell_with_hsdp_interface(cell)
    cell.hsdp_init(
        platform_type,
        cell,
        shard_size,
        threshold * 1024,
        optimizer_level,
        enable_grad_accumulation,
        grad_scale,
        reduce_dtype,
        comm_async,
        comm_fusion,
        bucket_size * 1024
    )
    return cell

def hsdp_wait_grad_handle():
    """wait for hsdp gradient handle to be completed"""
    global current_platform
    if current_platform is None:
        return
    current_platform.wait_grad_handle()
