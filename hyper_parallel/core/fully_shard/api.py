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
import warnings
from collections import OrderedDict, namedtuple
from typing import Any, Mapping, cast, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions

from hyper_parallel.platform.platform import PlatformType
from hyper_parallel.platform.torch.fully_shard.utils import MixedPrecisionPolicy, OffloadPolicy
from hyper_parallel import DeviceMesh, init_device_mesh
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor import DTensor, distribute_tensor

# Matches torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX.
# Defined here to avoid importing a private symbol at module scope.
_EXTRA_STATE_KEY_SUFFIX = "_extra_state"

platform = get_platform()

origin_class_to_extend_class = {}

_IncompatibleKeys = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])


def _raise_if_incompatible(
    module_name: str,
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    """Raise ``RuntimeError`` if there are missing or unexpected keys."""
    error_msgs: list[str] = []
    if missing_keys:
        error_msgs.append(
            "Missing key(s): " + ", ".join(repr(k) for k in sorted(missing_keys))
        )
    if unexpected_keys:
        error_msgs.append(
            "Unexpected key(s): " + ", ".join(repr(k) for k in sorted(unexpected_keys))
        )
    if error_msgs:
        raise RuntimeError(
            f"Error(s) in loading state_dict for "
            f"{module_name}:\n\t"
            + "\n\t".join(error_msgs)
        )


def _resolve_local_tensor(
    key: str, val: torch.Tensor, target: DTensor,
) -> torch.Tensor:
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
            val.detach(), target.device_mesh, target.placements,
        )
        return wrapped.to_local()
    raise ValueError(
        f"load '{key}': plain tensor shape {val_shape} "
        f"matches neither local shard {local_shape} "
        f"nor global {global_shape}."
    )


def _build_prefix_to_module(
    self_module: nn.Module,
    expected_state: dict[str, Any],
) -> dict[str, nn.Module]:
    """Map hook-modified prefixes to module objects.

    Primary path: ``expected_state`` is produced by ``state_dict(keep_vars=True)``,
    so its tensor values are *the same objects* as the live parameters/buffers.
    By matching ``id(val)`` we discover the hook-modified prefix for every
    module that owns at least one parameter or buffer.

    Fallback path: for modules with ``_extra_state`` but **no** parameters or
    buffers (rare, but possible), the raw module-tree prefix is used directly.
    This covers the common non-wrapper case.  If a wrapper also renames such a
    module's prefix, the fallback cannot discover the renamed prefix (no tensor
    identity to match), so the key may be reported as unexpected under
    ``strict=True`` — no data is lost because the module has no tensors.
    """
    # id(tensor) → module
    id_to_module: dict[int, nn.Module] = {}
    for module in self_module.modules():
        for p in module._parameters.values():  # pylint: disable=protected-access
            if p is not None:
                id_to_module[id(p)] = module
        for b in module._buffers.values():  # pylint: disable=protected-access
            if b is not None:
                id_to_module[id(b)] = module

    # Primary: expected_state tensor identity → hook-modified prefix → module
    prefix_to_module: dict[str, nn.Module] = {}
    for key, val in expected_state.items():
        if not isinstance(val, torch.Tensor):
            continue
        module = id_to_module.get(id(val))
        if module is None:
            continue
        expected_prefix = key[: key.rfind(".") + 1]  # "a.b.c.weight" → "a.b.c."
        if expected_prefix not in prefix_to_module:
            prefix_to_module[expected_prefix] = module

    # Fallback: raw tree walk for modules not yet mapped (extra-state-only).
    for name, module in self_module.named_modules():
        raw_prefix = f"{name}." if name else ""
        if raw_prefix not in prefix_to_module:
            prefix_to_module[raw_prefix] = module

    return prefix_to_module


def _prepare_mutable_state_dict(
    state_dict: Mapping[str, Any],
) -> OrderedDict:
    """Convert state_dict to mutable OrderedDict, preserving _metadata."""
    metadata = getattr(state_dict, '_metadata', None)
    mutable_sd: OrderedDict = OrderedDict(state_dict)
    if metadata is not None:
        mutable_sd._metadata = metadata  # type: ignore[attr-defined]  # pylint: disable=W0212
    return mutable_sd


def _run_load_pre_hooks(
    self_module: nn.Module,
    state_dict: OrderedDict,
    module_to_prefix: dict[int, str],
    strict: bool,
) -> None:
    """Run _load_state_dict_pre_hooks on all modules (parent-first).

    Replicates the pre-hook phase of PyTorch's recursive
    ``_load_from_state_dict`` dispatch.  Hooks can modify *state_dict*
    in-place (e.g. inject ``_extra_state`` keys).

    Temporary lists are used for missing/unexpected/error_msgs so that
    hook side-effects on these lists do not pollute the caller's
    strict-check logic.
    """
    metadata = getattr(state_dict, '_metadata', None)
    tmp_missing: list[str] = []
    tmp_unexpected: list[str] = []
    tmp_errors: list[str] = []

    for module in self_module.modules():
        hooks = module._load_state_dict_pre_hooks  # pylint: disable=protected-access
        if not hooks:
            continue
        prefix = module_to_prefix.get(id(module), "")
        local_metadata = (
            {} if metadata is None
            else metadata.get(prefix[:-1] if prefix else "", {})
        )
        for hook in hooks.values():
            # _WrappedHook auto-injects module; do NOT pass module manually.
            hook(state_dict, prefix, local_metadata, strict,
                 tmp_missing, tmp_unexpected, tmp_errors)

    if tmp_errors:
        raise RuntimeError(
            "Error(s) in load_state_dict pre-hooks:\n\t"
            + "\n\t".join(tmp_errors)
        )


def _overrides_set_extra_state(module: nn.Module) -> bool:
    """Return True if *module*'s class overrides ``set_extra_state``."""
    return (
        getattr(module.__class__, "set_extra_state", nn.Module.set_extra_state)
        is not nn.Module.set_extra_state
    )


def _load_extra_states(
    self_module: nn.Module,
    state_dict: Mapping[str, Any],
    expected_state: dict[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    prefix_to_module: dict[str, nn.Module] | None = None,
) -> None:
    """Load ``_extra_state`` keys using ``expected_state`` for key validation.

    Tensor keys already use ``expected_state`` to decide expected/unexpected
    (see ``load_state_dict`` Phase 1).  This function applies the **same**
    logic to ``_extra_state`` keys so that wrappers (e.g. Float16Module) and
    state-dict hooks that rename key prefixes are handled consistently.

    Before calling ``set_extra_state``, the target module's override is
    checked (matching ``nn.Module._load_from_state_dict`` semantics).
    If only ``get_extra_state`` is overridden but ``set_extra_state`` is not,
    the key is treated as unexpected under ``strict=True``.
    """
    extra_sd_keys = {
        k for k in state_dict if k.endswith(_EXTRA_STATE_KEY_SUFFIX)
    }
    extra_expected_keys = {
        k for k in expected_state if k.endswith(_EXTRA_STATE_KEY_SUFFIX)
    }
    if not extra_sd_keys and not extra_expected_keys:
        return

    # Reuse caller-provided map, or build if not supplied.
    if prefix_to_module is None:
        prefix_to_module = _build_prefix_to_module(self_module, expected_state)

    # Load _extra_state values present in both state_dict and expected_state.
    for key in sorted(extra_sd_keys):
        if key not in extra_expected_keys:
            if strict:
                unexpected_keys.append(key)
            continue
        prefix = key[: -len(_EXTRA_STATE_KEY_SUFFIX)]  # "a.b._extra_state" → "a.b."
        module = prefix_to_module.get(prefix)
        if module is not None and _overrides_set_extra_state(module):
            module.set_extra_state(state_dict[key])
        elif strict:
            # Module doesn't override set_extra_state (asymmetric get/set),
            # or module not found — treat as unexpected per PyTorch semantics.
            unexpected_keys.append(key)

    # Missing: expected _extra_state keys absent from state_dict.
    # Only report missing when the target module overrides set_extra_state,
    # matching PyTorch _load_from_state_dict semantics: if only get_extra_state
    # is overridden (asymmetric), the module cannot consume the value, so a
    # missing checkpoint key is not an error.
    if strict:
        for key in sorted(extra_expected_keys - extra_sd_keys):
            prefix = key[: -len(_EXTRA_STATE_KEY_SUFFIX)]
            module = prefix_to_module.get(prefix)
            if module is not None and _overrides_set_extra_state(module):
                missing_keys.append(key)


def _load_tensor_value(
    key: str, val: Any, target: torch.Tensor,
) -> None:
    """Copy *val* into *target*, handling DTensor and meta-init cases."""
    if isinstance(target, DTensor):
        local_val = _resolve_local_tensor(key, val, target)
        local_target = target._local_tensor  # pylint: disable=protected-access
        if local_target.is_meta:
            orig_requires_grad = target.requires_grad
            target._local_tensor = local_val  # pylint: disable=protected-access
            if local_val.requires_grad != orig_requires_grad:
                target.requires_grad_(orig_requires_grad)
        else:
            local_target.copy_(local_val)
    else:
        # Unwrap DTensor to avoid triggering __torch_function__
        # dispatch on copy_ (DTensor's copy_ is not supported).
        src = val.to_local() if isinstance(val, DTensor) else val
        target.copy_(src)


def _collect_missing_keys(
    expected_state: dict[str, Any],
    sd_keys: set[str],
) -> list[str]:
    """Return expected tensor keys absent from *sd_keys*.

    ``_extra_state`` keys are excluded — their missing/unexpected checks
    are handled by :func:`_load_extra_states`.
    """
    missing: list[str] = []
    for key in expected_state:
        if key in sd_keys:
            continue
        if key.endswith(_EXTRA_STATE_KEY_SUFFIX):
            continue  # Handled by _load_extra_states
        missing.append(key)
    return missing


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
        self.hsdp_scheduler = None  # initialized in hsdp_init()

    # pylint: disable=C0415
    def hsdp_init(self, platform_type, module, mesh, reshard_after_forward,
                  shard_placement_fn, mp_policy, offload_policy, ignored_params, device, comm_fusion):
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
                                              comm_fusion
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
            raise RuntimeError("zero_grads shouldn't be called in torch platform, use optimizer.zero_grad() instead.")
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
    ) -> _IncompatibleKeys:
        """
        Load state dict by copying directly into local shards.

        Bypasses ``super().load_state_dict()`` because the standard PyTorch
        implementation triggers ``copy_`` through the DTensor dispatcher, which
        is not registered in the hyper-parallel layout system.

        Each value in ``state_dict`` is dispatched by type:
          - ``_extra_state`` keys: dispatch to the owning module's
            ``set_extra_state()`` (if overridden).
          - hyper DTensor: extract local shard via ``to_local()`` and copy
            into the target's local tensor directly (never calls DTensor's
            ``copy_``).
          - plain Tensor whose shape == local shard shape: copy as-is.
          - plain Tensor whose shape == global shape: distribute via
            ``distribute_tensor``, then copy the local shard.

        The strict-key check follows the same semantics as
        ``nn.Module.load_state_dict``: all values are loaded first, then
        missing / unexpected keys are reported.  For ``_extra_state`` keys the
        check uses ``set_extra_state`` override detection (matching PyTorch's
        ``_load_from_state_dict``), not ``get_extra_state``.

        Args:
            state_dict (Mapping[str, Any]): Fully-qualified parameter/buffer/
                extra-state names mapped to values (DTensor, plain Tensor,
                or arbitrary objects for ``_extra_state``).
            strict (bool): If ``True`` (default), missing or unexpected keys
                raise ``RuntimeError``, matching ``nn.Module.load_state_dict``
                semantics.
            assign (bool): Accepted for API compatibility with
                ``nn.Module.load_state_dict(assign=True)`` but currently
                ignored; HSDP always copies into existing DTensor storage.

        Raises:
            RuntimeError: When ``strict`` is ``True`` and keys do not match.
            ValueError: When a plain tensor shape matches neither the local
                shard shape nor the global shape of the target DTensor.
        """
        if assign:
            warnings.warn(
                "HSDPModule.load_state_dict: assign=True is ignored; "
                "HSDP always copies into existing DTensor parameters.",
                stacklevel=2,
            )
        self_module = cast(nn.Module, self)

        # Use keep_vars=True so values are live parameter/buffer references
        # (not detached copies), which is required for meta-init materialisation
        # and ensures keys match checkpoint semantics (including state_dict hooks).
        expected_state = self_module.state_dict(keep_vars=True)

        # ---- Phase 0: run _load_state_dict_pre_hooks ----
        # Convert to mutable OrderedDict preserving _metadata, then
        # replay pre-hooks so they can modify state_dict in-place
        # (e.g. inject _extra_state keys), matching PyTorch semantics.
        state_dict = _prepare_mutable_state_dict(state_dict)
        prefix_to_module = _build_prefix_to_module(self_module, expected_state)
        module_to_prefix: dict[int, str] = {}
        for prefix, module in prefix_to_module.items():
            mid = id(module)
            if mid not in module_to_prefix:
                module_to_prefix[mid] = prefix
        _run_load_pre_hooks(self_module, state_dict, module_to_prefix, strict)

        target_map: dict[str, torch.Tensor] = {
            k: v for k, v in expected_state.items()
            if isinstance(v, torch.Tensor)
        }

        unexpected_keys: list[str] = []
        missing_keys: list[str] = []

        # ---- Phase 1: load tensor values from state_dict ----
        with torch.no_grad():
            for key, val in state_dict.items():
                if key.endswith(_EXTRA_STATE_KEY_SUFFIX):
                    continue  # Handled in Phase 2

                target = target_map.get(key)
                if target is None:
                    if strict and key not in expected_state:
                        unexpected_keys.append(key)
                    continue

                _load_tensor_value(key, val, target)

        # ---- Phase 2: load _extra_state keys ----
        # Uses expected_state for key validation (same as Phase 1 for tensors)
        # so that wrappers and state-dict hooks that rename key prefixes are
        # handled consistently.  Reuse prefix_to_module from Phase 0.
        _load_extra_states(
            self_module, state_dict, expected_state, strict,
            missing_keys, unexpected_keys, prefix_to_module,
        )

        # ---- Phase 3: check for missing tensor keys ----
        if strict:
            missing_keys.extend(
                _collect_missing_keys(expected_state, set(state_dict.keys()))
            )

        # Trigger load_state_dict post-hooks so that HSDP internal
        # bookkeeping (e.g. _sharded_param_data) stays in sync.
        # Execute before strict check so hooks fire even when strict
        # will raise, matching PyTorch's per-module hook semantics.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        for _, module in self_module.named_modules():
            hooks = module._load_state_dict_post_hooks  # pylint: disable=protected-access
            for hook in hooks.values():
                hook(module, incompatible_keys)

        if strict and (missing_keys or unexpected_keys):
            _raise_if_incompatible(
                self_module.__class__.__name__,
                missing_keys,
                unexpected_keys,
            )

        return incompatible_keys

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
            raise NotImplementedError("Currently impl is equal to recurse=True,"
                                      " need support module_param mapping.")
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
            raise NotImplementedError("Currently impl is equal to recurse=True,"
                                      " need support module_param mapping.")
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
            raise NotImplementedError("Currently impl is equal to recurse=True,"
                                      " need support module_param mapping.")
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
    """Dynamically extend module's class to inherit from HSDPModule, adding HSDP capabilities."""
    origin_class = module.__class__
    extend_class = origin_class_to_extend_class.get(origin_class, None)
    if extend_class is None:
        extend_class = type(f"HSDP{origin_class.__name__}", (HSDPModule, origin_class), {})
        origin_class_to_extend_class[origin_class] = extend_class
    module.__class__ = extend_class


def _get_device_from_mesh(mesh: DeviceMesh):
    """Extract and validate the torch device from the device mesh."""
    device = None
    device_type = mesh.device_type
    if device_type not in ("npu", "gpu"):
        raise AssertionError(
            f"hyper_parallel.fully_shard support device in [torch.npu, torch.gpu], but got '{device_type}'"
        )
    device_handle = platform.get_device_handle(device_type)
    if device_handle is None:
        raise ValueError(
            f"hyper_parallel.fully_shard can't find device_handle of 'torch.{device_type}', "
            "check the environment."
        )
    if device_handle.is_available():
        device = torch.device(device_handle.current_device())
    return device


def fully_shard(
        module: nn.Module,
        *,
        mesh: Optional[DeviceMesh] = None,
        reshard_after_forward: bool = True,
        shard_placement_fn: None = None,
        mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
        offload_policy: OffloadPolicy = OffloadPolicy(),
        ignored_params: Optional[set[nn.Parameter]] = None,
        comm_fusion: bool = False
):
    """
    Apply fully_shard to a module for distributed training with parameter sharding.

    This interface provides PyTorch-compatible HSDP (Hybrid Sharded Data Parallelism)
    functionality, enabling efficient training of large models by sharding parameters
    across multiple devices. The module is automatically enhanced with distributed
    capabilities including parameter sharding, gradient synchronization, and memory
    management.

    The function dynamically extends the module's class to inherit from HSDPModule,
    adding methods for manual control over sharding/unsharding, prefetching, and
    state management. This allows fine-grained control over when parameters are
    gathered for computation and resharded after use.

    Parameters:
        module (nn.Module):
            The module to apply fully_shard to. The module is modified in-place and
            enhanced with HSDP capabilities.

        mesh (Optional[DeviceMesh], default=None):
            The device mesh defining the process topology for distributed training.
            If None, a default 1D mesh with all processes in the sharding dimension
            is created. For HSDP mode, use a 2D mesh with dimensions configured
            for sharding (dim 1) and replication (dim 0).

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
            Set of parameters to exclude from sharding. These parameters remain
            fully replicated across all devices. Useful for small parameters where
            sharding overhead outweighs memory benefits, or parameters that must
            remain unsharded for correctness.
        comm_fusion  (bool, default=False):
            Whether enable all_gather fusion and reduce_scatter fusion.

    Returns:
        nn.Module: The input module with HSDP capabilities added. The module's
            class is dynamically extended to inherit from HSDPModule, providing
            additional methods for distributed training control.
    """
    platform_type = platform.platform_type
    _extend_module_with_hsdp_interface(module)
    # if mesh is None, Using Default npu mesh
    mesh = mesh or init_device_mesh(device_type="npu", mesh_shape=(platform.get_world_size(),))
    device = _get_device_from_mesh(mesh)
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
        comm_fusion
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
