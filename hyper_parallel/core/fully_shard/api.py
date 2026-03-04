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
"""hybrid shard data parallel interface"""
from typing import Any, Mapping, cast, Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions

from hyper_parallel.platform.platform import PlatformType
from hyper_parallel import DeviceMesh, init_device_mesh
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor import DTensor, distribute_tensor

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
        hsdp_state = self.hsdp_scheduler.hsdp_state
        if hsdp_state:
            hsdp_state.shard()

    def unshard(self, async_op: bool = False):
        """unshard all sharded parameters"""
        if not isinstance(async_op, bool):
            raise ValueError(f"async_op should be a bool, got {type(async_op)}")
        if not self.hsdp_scheduler:
            raise ValueError("hsdp_scheduler is None")
        hsdp_state = self.hsdp_scheduler.hsdp_state
        if hsdp_state:
            hsdp_state.unshard(async_op)  # pylint: disable=too-many-function-args
            if async_op:
                return _UnshardHandle(hsdp_state=hsdp_state)
        return None

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        """
        Load state dict by copying directly into local shards.

        Bypasses ``super().load_state_dict()`` because the standard PyTorch
        implementation triggers ``copy_`` through the DTensor dispatcher, which
        is not registered in the hyper-parallel layout system.

        Each value in ``state_dict`` is dispatched by type:
          - hyper DTensor: extract local shard and copy directly.
          - plain Tensor whose shape == local shard shape: copy as-is.
          - plain Tensor whose shape == global shape: distribute via
            ``distribute_tensor``, then copy the local shard.

        Args:
            state_dict (Mapping[str, Any]): Fully-qualified parameter/buffer
                names mapped to tensors (DTensor or plain Tensor).
            strict (bool): If ``True`` (default), missing or unexpected keys
                raise ``RuntimeError``, matching ``nn.Module.load_state_dict``
                semantics.
            assign (bool): Reserved for API compatibility with
                ``nn.Module.load_state_dict(assign=True)``. Currently unused.

        Raises:
            RuntimeError: When ``strict`` is ``True`` and keys do not match.
            ValueError: When a plain tensor shape matches neither the local
                shard shape nor the global shape of the target DTensor.
        """
        self_module = cast(nn.Module, self)

        target_map: dict[str, torch.Tensor] = {}
        for name, p in self_module.named_parameters():
            target_map[name] = p
        for name, b in self_module.named_buffers():
            target_map[name] = b

        if strict:
            expected_keys = set(self_module.state_dict().keys())
            missing = expected_keys - set(state_dict.keys())
            unexpected = set(state_dict.keys()) - expected_keys
            error_msgs: list[str] = []
            if missing:
                error_msgs.append(
                    "Missing key(s): " + ", ".join(repr(k) for k in sorted(missing))
                )
            if unexpected:
                error_msgs.append(
                    "Unexpected key(s): " + ", ".join(repr(k) for k in sorted(unexpected))
                )
            if error_msgs:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for "
                    f"{self_module.__class__.__name__}:\n\t"
                    + "\n\t".join(error_msgs)
                )

        with torch.no_grad():
            for key, val in state_dict.items():
                target = target_map.get(key)
                if target is None:
                    continue

                if isinstance(target, DTensor):
                    if isinstance(val, DTensor):
                        local_val = val.to_local()
                    else:
                        local_shape = tuple(target.local_shape)
                        global_shape = tuple(target.shape)
                        val_shape = tuple(val.shape)
                        if val_shape == local_shape:
                            local_val = val
                        elif val_shape == global_shape:
                            wrapped = distribute_tensor(
                                val.detach(), target.device_mesh, target.placements,
                            )
                            local_val = wrapped.to_local()
                        else:
                            raise ValueError(
                                f"load '{key}': plain tensor shape {val_shape} "
                                f"matches neither local shard {local_shape} "
                                f"nor global {global_shape}."
                            )
                    if target.to_local().is_meta:
                        # Meta tensor materialisation: replace the placeholder
                        target._local_tensor = local_val  # pylint: disable=protected-access
                    else:
                        target.to_local().copy_(local_val)
                else:
                    target.copy_(val)

        # Trigger load_state_dict post-hooks so that HSDP internal
        # bookkeeping (e.g. _sharded_param_data) stays in sync.
        for _, module in self_module.named_modules():
            hooks = module._load_state_dict_post_hooks  # pylint: disable=protected-access
            for hook in hooks.values():
                hook(module, None)

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

    def set_reduce_op_type(self, reduce_op_type) -> None:
        """
        set reduce_op_type for all reduce operations in HSDP
        support reduce_op_type "avg" and "sum", default is "avg"
        """
        if hsdp_state := self.hsdp_scheduler.hsdp_state:
            hsdp_state.set_reduce_op_type(reduce_op_type)


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
        mesh: Optional[DeviceMesh] = None,
        reshard_after_forward: Optional[Union[bool, int]] = None,
        shard_placement_fn: None = None,
        mp_policy: None = None,
        offload_policy: None = None,
        ignored_params: Optional[set[nn.Parameter]] = None,
        device = None,
):
    platform_type = platform.platform_type
    _extend_module_with_hsdp_interface(module)
    # TODO: mindspore does not support get_device_handle
    if device is None:
        device_handle = platform.get_device_handle()  # return torch.npu or torch.cuda
        if device_handle.is_available():
            device = torch.device(device_handle.current_device())
        else:
            device = torch.device("cpu")

    mesh = mesh or init_device_mesh(device_type=device, mesh_shape=(platform.get_world_size(),))

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


def _gather_full_state_dict(
    state_dict: dict[str, Any], cpu_offload: bool
) -> dict[str, Any]:
    """All-gather every DTensor shard into a full tensor.

    Args:
        state_dict: Model state dict with DTensor or plain tensor values.
        cpu_offload: If True, only rank-0 keeps the result on CPU;
            other ranks return an empty dict to save memory.
    """
    is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)

    gathered: dict[str, Any] = {}
    for key, val in state_dict.items():
        if isinstance(val, DTensor):
            val = val.full_tensor()
        if cpu_offload:
            if not is_rank0:
                del val
                continue
            if isinstance(val, torch.Tensor):
                val = val.cpu()
        gathered[key] = val

    if cpu_offload and not is_rank0:
        return {}
    return gathered


def _offload_sharded_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Move each shard to CPU without all-gathering.

    Args:
        state_dict: Model state dict with DTensor or plain tensor values.
    """
    offloaded: dict[str, Any] = {}
    for key, val in state_dict.items():
        if isinstance(val, DTensor):
            val = DTensor.from_local(
                val.to_local().cpu(), val.device_mesh, val.placements,
            )
        elif isinstance(val, torch.Tensor):
            val = val.cpu()
        offloaded[key] = val
    return offloaded


def get_model_state_dict(
    model: nn.Module,
    *,
    options: StateDictOptions | None = None,
) -> dict[str, Any]:
    """Return the model state dict with configurable gathering and offloading.

    Behaviour matrix:

    +-----------------+-------------+--------------------------------------+
    | full_state_dict | cpu_offload | result                               |
    +=================+=============+======================================+
    | False           | False       | DTensor (sharded, as-is)             |
    +-----------------+-------------+--------------------------------------+
    | False           | True        | DTensor local shard offloaded to CPU |
    +-----------------+-------------+--------------------------------------+
    | True            | False       | full Tensor on **every** rank        |
    +-----------------+-------------+--------------------------------------+
    | True            | True        | full Tensor on CPU, **rank 0 only**  |
    +-----------------+-------------+--------------------------------------+

    Args:
        model: The model whose state dict to retrieve.
        options: Controls full_state_dict, cpu_offload,
            ignore_frozen_params, and broadcast_from_rank0 flags.
    """
    options = options or StateDictOptions()

    if options.broadcast_from_rank0 and not options.full_state_dict:
        raise ValueError(
            "full_state_dict must be True when broadcast_from_rank0 is True."
        )

    state_dict: dict[str, Any] = model.state_dict()

    if options.ignore_frozen_params:
        frozen_keys = {
            name for name, p in model.named_parameters()
            if not p.requires_grad
        }
        for key in frozen_keys:
            state_dict.pop(key, None)

    if options.full_state_dict:
        return _gather_full_state_dict(state_dict, options.cpu_offload)

    if options.cpu_offload:
        return _offload_sharded_state_dict(state_dict)

    return state_dict


def hsdp_sync_stream():
    """wait for hsdp gradient handle to be completed"""
    if platform is None:
        return
    platform.wait_grad_handle()
