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
from collections import namedtuple
from typing import Any, List, Mapping, cast, Optional, Union

from hyper_parallel.platform.platform import PlatformType
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, OffloadPolicy
from hyper_parallel import DeviceMesh, init_device_mesh
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.fully_shard.hsdp_utils import (
    get_managed_modules_parameters,
    is_dtensor_managed_param,
    get_dtensor_managed_mesh,
)

platform = get_platform()

origin_class_to_extend_class = {}


def _resolve_comm_fusion_zero_copy_default(
    platform_type: PlatformType,
    comm_fusion: bool,
    comm_fusion_zero_copy: Optional[bool],
) -> bool:
    """Resolve backend-specific default for the comm_fusion zero-copy path."""
    if comm_fusion_zero_copy is not None:
        return comm_fusion_zero_copy
    if not comm_fusion:
        return False
    if platform_type == PlatformType.PYTORCH:
        return True
    if platform_type == PlatformType.MINDSPORE:
        return False
    return False


def _check_strict_keys(
    module: platform.Module, state_dict: Mapping[str, Any],
) -> None:
    """Raise ``RuntimeError`` if *state_dict* keys do not match *module*."""
    expected_keys = set(module.state_dict().keys())
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
            f"{module.__class__.__name__}:\n\t"
            + "\n\t".join(error_msgs)
        )


def _resolve_local_tensor(
    key: str, val: platform.Tensor, target: DTensor,
) -> platform.Tensor:
    """Return the local shard tensor to be loaded into *target*."""
    if isinstance(val, DTensor):
        return val.to_local()
    local_shape = tuple(target.local_shape)
    global_shape = tuple(target.shape)
    val_shape = tuple(val.shape)
    if val_shape == local_shape:
        return val
    if val_shape == global_shape:
        wrapped = distribute_tensor(
            val, target.device_mesh,
            target.layout.alias_placements if target.layout else target.placements,
        )
        return wrapped.to_local()

    raise ValueError(
        f"load '{key}': plain tensor shape {val_shape} "
        f"matches neither local shard {local_shape} "
        f"nor global {global_shape}."
    )


class _UnshardHandle:
    """Unshard handle for user call HSDPModule.unshard(async_op=True)"""
    def __init__(self, hsdp_state=None):
        """
        Initialize an async unshard handle.

        Args:
            hsdp_state (HSDPState, optional): The state to wait on. None means a no-op handle.
        """
        self._hsdp_state = hsdp_state

    def wait(self):
        """Block until the async unshard operation completes."""
        if self._hsdp_state is not None:
            self._hsdp_state.wait_for_unshard()
            self._hsdp_state = None


class HSDPModule:
    """
    The hsdp block of neural networks with hsdp interface.

    Supported Platforms:
        ``MindSpore`` ``torch``
    """

    def __init__(self):
        """Initialize HSDPModule."""
        self.hsdp_scheduler = None  # Initialized in hsdp_init()

    # pylint: disable=C0415
    def hsdp_init(self, platform_type, module, mesh, reshard_after_forward,
                  shard_placement_fn, mp_policy, offload_policy, ignored_params, replicate_params, device,
                  comm_fusion, comm_fusion_zero_copy: Optional[bool] = None):
        """init hsdp2 scheduler."""
        scheduler_class = None
        if platform_type == PlatformType.MINDSPORE:
            from hyper_parallel.platform.mindspore.fully_shard.scheduler import MindSporeHSDPSchedulerV2
            scheduler_class = MindSporeHSDPSchedulerV2
        else:
            from hyper_parallel.platform.torch.fully_shard.scheduler import TorchHSDPSchedulerV2
            scheduler_class = TorchHSDPSchedulerV2

        resolved_comm_fusion_zero_copy = _resolve_comm_fusion_zero_copy_default(
            platform_type,
            comm_fusion,
            comm_fusion_zero_copy,
        )

        self.hsdp_scheduler = scheduler_class(module,
                                              mesh,
                                              reshard_after_forward,
                                              shard_placement_fn,
                                              mp_policy,
                                              offload_policy,
                                              ignored_params,
                                              replicate_params,
                                              device,
                                              comm_fusion,
                                              resolved_comm_fusion_zero_copy,
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

    def zero_grad(self):
        """zero accumunication grads"""
        if not hasattr(self, "hsdp_scheduler"):
            raise ValueError("call hsdp interface first.")
        if platform.platform_type == PlatformType.PYTORCH:
            return super().zero_grad()
        for _, module in platform.get_cells_and_names(self):
            if isinstance(module, HSDPModule):
                module.hsdp_scheduler.zero_grad()

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
            assign (bool): When ``True`` *and* every value in ``state_dict`` is
                already a hyper DTensor, defer to the standard
                ``nn.Module.load_state_dict(assign=True)``, which replaces the
                module's parameters/buffers with the given DTensors instead of
                copying into existing storage. This is required when loading
                sharded DTensors onto a meta-device model (e.g.
                ``cpu_ram_efficient_loading``). If ``state_dict`` contains any
                plain tensor (local-shard or global shape), ``assign`` is
                ignored and the copy/distribute path below is used so the
                target stays a properly sharded DTensor.

        Raises:
            RuntimeError: When ``strict`` is ``True`` and keys do not match.
            ValueError: When a plain tensor shape matches neither the local
                shard shape nor the global shape of the target DTensor.
        """
        if assign and state_dict and all(
            isinstance(val, DTensor) for val in state_dict.values()
        ):
            return super().load_state_dict(state_dict, strict=strict, assign=True)
        self_module = cast(platform.Module, self)

        target_map: dict[str, platform.Tensor] = {}
        for name, p in platform.parameters_dict(self_module):
            target_map[name] = p
        for name, b in self_module.named_buffers():
            target_map[name] = b

        if strict:
            _check_strict_keys(self_module, state_dict)

        with platform.no_grad():
            for key, val in state_dict.items():
                target = target_map.get(key)
                if target is None:
                    continue

                if isinstance(target, DTensor):
                    val = _resolve_local_tensor(key, val, target)
                platform.load_into_param(target, val)

        # Trigger load_state_dict post-hooks so that HSDP internal
        # bookkeeping (e.g. _sharded_param_data) stays in sync.
        # Pass an IncompatibleKeys with the same attribute names as PyTorch
        # so external hooks can safely read .missing_keys/.unexpected_keys.
        _IK = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
        incompatible_keys = _IK([], [])
        for _, module in platform.get_cells_and_names(self_module):
            hooks = module._load_state_dict_post_hooks  # pylint: disable=protected-access
            for hook in hooks.values():
                hook(module, incompatible_keys)

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
            raise NotImplementedError(
                "Currently impl is equal to recurse=True, "
                "need support module_param mapping."
            )
        self_module = cast(platform.Module, self)
        for _, module in platform.get_cells_and_names(self_module):
            if isinstance(module, HSDPModule):
                module.hsdp_scheduler.set_requires_all_reduce(requires_all_reduce)

    def set_reshard_after_forward(self, reshard_after_forward: bool, recurse: bool = True) -> None:
        """set reshard_after_forward flag"""
        if not isinstance(reshard_after_forward, bool):
            raise ValueError(
                f"reshard_after_forward should be a bool, got {type(reshard_after_forward)}"
            )
        if not recurse:
            raise NotImplementedError(
                "Currently impl is equal to recurse=True, "
                "need support module_param mapping."
            )
        self_module = cast(platform.Module, self)
        for _, module in platform.get_cells_and_names(self_module):
            if isinstance(module, HSDPModule):
                module.hsdp_scheduler.set_reshard_after_forward(reshard_after_forward)

    def set_reshard_after_backward(self, reshard_after_backward: bool, recurse: bool = True) -> None:
        """set reshard_after_backward flag"""
        if not isinstance(reshard_after_backward, bool):
            raise ValueError(
                f"reshard_after_backward should be a bool, got {type(reshard_after_backward)}"
            )
        if not recurse:
            raise NotImplementedError(
                "Currently impl is equal to recurse=True, "
                "need support module_param mapping."
            )
        self_module = cast(platform.Module, self)
        for _, module in platform.get_cells_and_names(self_module):
            if isinstance(module, HSDPModule):
                module.hsdp_scheduler.set_reshard_after_backward(reshard_after_backward)

    def set_reduce_op_type(self, reduce_op_type) -> None:
        """
        Set reduce_op_type for all gradient reductions in fully_shard.

        Supports ``"avg"`` and ``"sum"``. Local-parameter FSDP/HSDP keeps the
        historical ``"avg"`` default, while DTensor-based paths default to ``"sum"``.
        """
        hsdp_state = self.hsdp_scheduler.hsdp_state
        if hsdp_state:
            hsdp_state.set_reduce_op_type(reduce_op_type)

    def set_gradient_scaling_factor(self, factor=None):
        """
        Set a multiplicative scaling factor applied to gradients after
        reduce-scatter / all-reduce and before they are written into
        ``sharded_param.grad``.

        ``factor`` may be ``None`` (disable scaling), a Python ``float``/``int``,
        or a 0-dim/1-element tensor. Setting ``factor`` to ``None`` (the default
        on construction) skips the scaling op entirely so no extra device-side
        ``mul_`` is launched on the hot path.

        Args:
            factor (None | float | int | platform.Tensor): Scaling coefficient.
                Use ``None`` to disable scaling.

        Raises:
            ValueError: If ``factor`` is not one of the supported types or is a
                tensor with more than one element.
        """
        if factor is not None:
            if isinstance(factor, bool):
                raise ValueError(
                    f"gradient_scaling_factor must be None, float, int or a 1-element Tensor, "
                    f"but got bool {factor}."
                )
            if isinstance(factor, platform.Tensor):
                if factor.numel() != 1:
                    raise ValueError(
                        f"gradient_scaling_factor tensor must have exactly 1 element, "
                        f"but got shape {tuple(factor.shape)}."
                    )
            elif not isinstance(factor, (float, int)):
                raise ValueError(
                    f"gradient_scaling_factor must be None, float, int or a 1-element Tensor, "
                    f"but got {type(factor).__name__}."
                )
        hsdp_state = self.hsdp_scheduler.hsdp_state
        if hsdp_state:
            hsdp_state.set_gradient_scaling_factor(factor)


def _extend_module_with_hsdp_interface(module):
    """Dynamically extend module's class to inherit from HSDPModule, adding HSDP capabilities."""
    origin_class = module.__class__
    extend_class = origin_class_to_extend_class.get(origin_class, None)
    if extend_class is None:
        extend_class = type(f"HSDP{origin_class.__name__}", (HSDPModule, origin_class), {})
        origin_class_to_extend_class[origin_class] = extend_class
    module.__class__ = extend_class


def _get_root_modules(modules: List[platform.Module]) -> List[platform.Module]:
    """
    Returns the modules in ``modules`` that are root modules (i.e. parent-less)
    with respect to the set ``modules``. In other words, these are the modules
    in ``modules`` that are not the child of any other module in ``modules``.

    Aligned with PyTorch torch.distributed.utils._get_root_modules.
    """
    root_modules: List[platform.Module] = []

    def _get_submodules(mod):
        if platform.platform_type == PlatformType.MINDSPORE:
            return set(c for _, c in mod.cells_and_names())
        return set(mod.modules())

    module_to_modules: dict[platform.Module, set] = {
        m: _get_submodules(m) for m in modules
    }
    for candidate in modules:
        is_root = True
        for mod, submodules in module_to_modules.items():
            if candidate is not mod and candidate in submodules:
                is_root = False
                break
        if is_root:
            root_modules.append(candidate)
    return root_modules


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


def _validate_module_for_fully_shard(
    module: Union[platform.Module, List[platform.Module]], platform_type
) -> None:
    """Validate module(s) for fully_shard. Platform-aware for single module."""
    if isinstance(module, list):
        if len(module) == 0:
            raise ValueError("fully_shard does not support empty list of modules.")
        for i, m in enumerate(module):
            try:
                _check_module_valid(platform_type, m)
            except ValueError:
                raise ValueError(
                    f"fully_shard expects nn.Module or list[nn.Module], "
                    f"but got list with {type(m).__name__} at index {i}."
                ) from None
    else:
        _check_module_valid(platform_type, module)


HsdpValidationOptions = namedtuple(
    "HsdpValidationOptions",
    [
        "shard_size",
        "threshold",
        "optimizer_level",
        "enable_grad_accumulation",
        "grad_scale",
        "reduce_dtype",
        "comm_async",
        "comm_fusion",
        "bucket_size",
    ],
)


def _validate_hsdp_shard_size(shard_size: int) -> None:
    if not isinstance(shard_size, int) or (shard_size <= 0 and shard_size != -1):
        raise ValueError(f"shard_size must be a positive integer, but got {shard_size}.")


def _validate_hsdp_threshold(threshold: int) -> None:
    if not isinstance(threshold, int) or threshold < 0:
        raise ValueError(f"threshold must be a positive integer or 0, but got {threshold}.")


def _validate_hsdp_optimizer_level(optimizer_level: str) -> None:
    if optimizer_level not in ["level1", "level2", "level3"]:
        raise ValueError(
            f"Optimizer level should in ['level1', 'level2', 'level3'], but got {optimizer_level}."
        )


def _validate_hsdp_reduce_dtype(platform_type: PlatformType, reduce_dtype) -> None:
    if platform_type == PlatformType.MINDSPORE:
        from mindspore._c_expression.typing import Type
        if reduce_dtype is not None and not isinstance(reduce_dtype, Type):
            raise ValueError(f"reduce_dtype must be mindspore.dtype but got {reduce_dtype}.")
        return
    import torch
    if reduce_dtype is not None and not isinstance(reduce_dtype, torch.dtype):
        raise ValueError(f"reduce_dtype must be torch.dtype but got {reduce_dtype}.")


def _check_hsdp_input_valid(platform_type, module, options: HsdpValidationOptions):
    """check hsdp input valid"""
    _check_module_valid(platform_type, module)
    _validate_hsdp_shard_size(options.shard_size)
    _validate_hsdp_threshold(options.threshold)
    _validate_hsdp_optimizer_level(options.optimizer_level)
    if not isinstance(options.enable_grad_accumulation, bool):
        raise ValueError(
            f"enable_grad_accumulation must be bool but got {options.enable_grad_accumulation}."
        )
    if not isinstance(options.grad_scale, float):
        raise ValueError(f"grad_scale must be float but got {options.grad_scale}.")
    _validate_hsdp_reduce_dtype(platform_type, options.reduce_dtype)
    if not isinstance(options.comm_async, bool):
        raise ValueError(f"comm_async must be bool but got {options.comm_async}.")
    if not isinstance(options.comm_fusion, bool):
        raise ValueError(f"comm_fusion must be bool but got {options.comm_fusion}.")
    if not isinstance(options.bucket_size, int) or (
        options.bucket_size < 0 and options.bucket_size != -1
    ):
        raise ValueError(
            f"bucket_size must be a positive integer or 0, but got {options.bucket_size}."
        )


def _get_device_from_mesh(mesh: DeviceMesh):
    """Extract and validate the torch device from the device mesh."""
    device = None
    device_type = mesh.device_type
    if device_type not in ("npu", "cuda"):
        raise AssertionError(
            f"hyper_parallel.fully_shard support device in [torch.npu, torch.cuda], "
            f"but got '{device_type}'"
        )
    if platform.platform_type == PlatformType.PYTORCH:
        device_handle = platform.get_device_handle(device_type)
        if device_handle is None:
            raise ValueError(
                f"hyper_parallel.fully_shard can't find device_handle of "
                f"'torch.{device_type}', check the environment."
            )
        if device_handle.is_available():
            import torch
            device = torch.device(device_handle.current_device())
    else:
        device = device_type
    return device


def _normalize_replicate_params(
    replicate_params: Optional[set[platform.Parameter]],
) -> set[platform.Parameter]:
    """
    Normalize replicate_params for fully_shard
    Args:
        replicate_params (Optional[set[nn.Parameter]]): Set of parameters to exclude from sharding.
    Returns:
        set[nn.Parameter]: Set of parameters to exclude from sharding.
    """
    if replicate_params is None:
        return set()
    out = set(replicate_params)
    for p in out:
        if not isinstance(p, (platform.Parameter, DTensor)):
            raise TypeError(
                "replicate_params must contain only nn.Parameter or DTensor, "
                f"got {type(p).__name__}."
            )
    return out


def _get_modules_parameters(modules, ignored_params=None):
    """Collect deduplicated parameters from module roots."""
    return get_managed_modules_parameters(modules, ignored_params)


def fully_shard(
        module: Union[platform.Module, List[platform.Module]],
        *,
        mesh: Optional[DeviceMesh] = None,
        reshard_after_forward: bool = True,
        shard_placement_fn: None = None,
        mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
        offload_policy: OffloadPolicy = OffloadPolicy(),
        ignored_params: Optional[set[platform.Parameter]] = None,
        replicate_params: Optional[set[platform.Parameter]] = None,
        comm_fusion: bool = False,
        comm_fusion_zero_copy: Optional[bool] = None,
) -> Union[platform.Module, List[platform.Module]]:

    """
    Apply fully_shard to a module (or list of modules) for distributed training with parameter sharding.

    This interface provides PyTorch-compatible HSDP (Hybrid Sharded Data Parallelism)
    functionality, enabling efficient training of large models by sharding parameters
    across multiple devices. The module is automatically enhanced with distributed
    capabilities including parameter sharding, gradient synchronization, and memory
    management.

    When a list of modules is passed, they are treated as one FSDP unit (parameters
    grouped together). Both PyTorch and MindSpore platforms support list input.

    Parameters:
        module (nn.Module or List[nn.Module]):
            The module(s) to apply fully_shard to. Modified in-place. When a list
            is passed, parameters from all modules are grouped as one FSDP unit.

        mesh (Optional[DeviceMesh], default=None):
            The device mesh defining the process topology for distributed training.
            If None, fully_shard keeps pure-DTensor modules on their original
            distributed layout and only creates a default 1D mesh when local
            parameters need explicit data-parallel/FSDP management.

        reshard_after_forward (bool, default=True):
            Whether to automatically reshard parameters after forward. When True,
            parameters are resharded immediately after they are no longer needed,
            freeing memory for subsequent operations. Set to False if you want to
            keep parameters unsharded for backward pass or manual control.

        shard_placement_fn (Callable, default=None):
            A callable that determines how to shard each parameter. The function
            should accept a parameter and return a Shard object specifying the
            sharding dimension, or None to use default sharding (dimension 0)

        mp_policy (MixedPrecisionPolicy, default=MixedPrecisionPolicy()):
            Mixed precision training policy controlling data type conversions.
        offload_policy (OffloadPolicy, default=OffloadPolicy()):
            Memory offload policy for reducing device memory usage.

        ignored_params (Optional[set[nn.Parameter]], default=None):
            Set of parameters to exclude from fully_shard management entirely.
            These parameters are left on the original module as regular parameters,
            are not sharded, and do not participate in fully_shard gradient
            synchronization. Use this for parameters that should remain outside
            the fully_shard lifecycle.

        comm_fusion  (bool, default=False):
            Whether enable all_gather fusion and reduce_scatter fusion.

        replicate_params (Optional[set[nn.Parameter]], default=None):
            Set of parameters to keep replicated while still managing them under
            fully_shard. These parameters are not sharded, but their gradients
            are still synchronized with DDP-style all-reduce over the current
            fully_shard communication domain. This differs from ``ignored_params``,
            which skips fully_shard management and gradient synchronization
            entirely for the selected parameters.

        comm_fusion_zero_copy (Optional[bool], default=None):
            Whether allow the experimental zero-copy path for
            ``comm_fusion``. When set to ``None``, fully_shard uses a backend-specific
            default:
            - PyTorch: enabled automatically when ``comm_fusion=True``
            - MindSpore: disabled automatically even when ``comm_fusion=True``
            When enabled, fully_shard may rebase sharded local parameter storage
            into one shared flat buffer so fused all-gather can read directly from
            contiguous memory. This path depends on optimizer compatibility with
            view-backed parameters.

    Returns:
        nn.Module or List[nn.Module]: The input module(s) with HSDP capabilities added.
    """
    platform_type = platform.platform_type
    _validate_module_for_fully_shard(module, platform_type)
    if platform_type == PlatformType.MINDSPORE:
        from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

        enable_mindspore_backward_compat()

    arg_module = module
    if isinstance(module, list):
        modules = tuple(_get_root_modules(module))
    else:
        modules = (module,)

    for mod in modules:
        _extend_module_with_hsdp_interface(mod)

    params = _get_modules_parameters(modules, ignored_params)
    has_dtensor_param = any(is_dtensor_managed_param(param) for param in params)
    replicate_params = _normalize_replicate_params(replicate_params)

    if mesh is None and not has_dtensor_param:
        mesh = init_device_mesh(device_type="npu", mesh_shape=(platform.get_world_size(),))
    if mesh is not None:
        device = _get_device_from_mesh(mesh)
    else:
        compat_mesh = None
        for param in params:
            dtensor_mesh = get_dtensor_managed_mesh(param)
            if dtensor_mesh is not None:
                compat_mesh = dtensor_mesh
                break
        if compat_mesh is None:
            raise ValueError("fully_shard could not resolve a DTensor mesh for compatibility mode.")
        device = _get_device_from_mesh(compat_mesh)

    init_modules = modules
    modules[0].hsdp_init(
        platform_type,
        init_modules,
        mesh,
        reshard_after_forward,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        ignored_params,
        replicate_params,
        device,
        comm_fusion,
        comm_fusion_zero_copy,
    )
    # Share the same scheduler handle with other roots so mods[i].unshard()/prefetch work
    if len(modules) > 1:
        for mod in modules[1:]:
            mod.hsdp_scheduler = modules[0].hsdp_scheduler
    return arg_module


def get_model_state_dict(model, *, options=None):
    """Get model state dict with platform-specific implementation.

    Delegates to the platform-specific implementation at runtime.
    Users import from here instead of platform internals.
    """
    return platform.get_model_state_dict(model, options=options)


def hsdp_sync_stream():
    """Wait for hsdp gradient handle to be completed."""
    platform.wait_grad_handle()
