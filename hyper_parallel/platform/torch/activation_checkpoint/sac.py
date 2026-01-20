# Copyright 2026 Huawei Technologies Co., Ltd
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

# Adapted from https://github.com/pytorch/pytorch/blob/release/2.6/torch/utils/checkpoint.py
# enhanced with selective checkpoint support swap
# ============================================================================
"""enhanced with selective checkpoint support swap"""
# pylint: disable=W0212, W0613, C0115, C0116, C0103, R1705
import enum
from typing import Any, Optional, Union

import torch
import torch.fx.traceback as fx_traceback
from torch._functorch._aot_autograd.functional_utils import is_fun
from torch.utils._pytree import tree_map
from torch.utils._python_dispatch import TorchDispatchMode
from .swap import SwapManager, SwapTensor, Storage  # patch code

def _is_compiling(func, args, kwargs):
    # Check if we are under AOTAutograd tracing
    # There should probably be a better way to do this...
    # TODO: unify _is_compiling across all compile stacks
    for arg in args:
        if isinstance(arg, torch.Tensor) and is_fun(arg):
            return True
    return False


class _VersionWrapper:
    # Check that cached tensors are not mutated.
    def __init__(self, val):
        self.val: Union[torch.Tensor, Any] = val
        self.version: Optional[int] = val._version if isinstance(val, torch.Tensor) else None

    def get_val(self, allow_cache_entry_mutation):
        if self.version is not None and not allow_cache_entry_mutation:
            if self.val._version != self.version:
                # Can we give user a stack trace of where the mutation happened?
                raise RuntimeError(
                    "Tensor cached during selective activation checkpoint has been mutated"
                )
        return self.val


def _maybe_detach(x, any_ret_has_alias_info):
    # We detach for two separate reasons:
    # - For view ops, we need to ensure that when the tensor is returned from
    #   CachedDispatchMode, as_view sees that the AutogradMeta is nullptr
    # - Avoid reference cycles
    # For case 1, it is not enough to check whether x has differentiable dtype
    # because non-differentiable dtype can have non-nullptr AutogradMeta, e.g.
    # when the tensor is a view.
    if isinstance(x, torch.Tensor) and (x.is_floating_point() or x.is_complex() or any_ret_has_alias_info):
        with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.ADInplaceOrView, False):
            # Ensure that view performed beneath autograd properly propagates
            # version counter. TODO: Use reentrant_dispatch instead of
            # manually manipulating dispatch keys. Using reentrant_dispatch
            # would respect inference_mode, though that is not relevant for
            # this case.
            x = x.detach()
    return x


class SelectiveCheckpointContext:
    """
    Context passed to policy function during selective checkpointing.

    This class is used to pass relevant metadata to the policy function during
    selective checkpointing. The metadata includes whether the current invocation
    of the policy function is during recomputation or not.

    Example:
        >>> # xdoctest: +SKIP(stub)
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    print(ctx.is_recompute)
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    def __init__(self, *, is_recompute):
        self.is_recompute = is_recompute


class CheckpointPolicy(enum.Enum):
    """
    Enum for specifying the policy for checkpointing during backpropagation.

    The following policies are supported:

    - ``{MUST,PREFER}_SAVE``: The operation's output will be saved during the forward
      pass and will not be recomputed during the backward pass
    - ``{MUST,PREFER}_RECOMPUTE``: The operation's output will not be saved during the
      forward pass and will be recomputed during the backward pass

    Use ``MUST_*`` over ``PREFER_*`` to indicate that the policy should not be overridden
    by other subsystems like `torch.compile`.

    .. note::
        A policy function that always returns ``PREFER_RECOMPUTE`` is
        equivalent to vanilla checkpointing.

        A policy function that returns ``PREFER_SAVE`` every op is
        NOT equivalent to not using checkpointing. Using such a policy would
        save additional tensors not limited to ones that are actually needed for
        gradient computation.
    """
    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3
    MUST_SWAP = 4  # patch code


def _policy_from_bool(b):
    # For backward compatibility
    return CheckpointPolicy.MUST_SAVE if b else CheckpointPolicy.PREFER_RECOMPUTE


SAC_IGNORED_OPS = {
    # AC inserts different number of detach during forward and recompute.
    torch.ops.aten.detach.default,
    # AC's determinism check invokes additional metadata ops during forward.
    # With subclasses involved, these metadata ops become dispatchable, this
    # can result in incorrectness if these ops are selected cached.
    torch.ops.prim.device.default,
} | set(torch._subclasses.functional_tensor.FunctionalTensor.metadata_fns)


class _CachingTorchDispatchMode(TorchDispatchMode):
    # Used together with _CachedTorchDispatchMode to implement SAC.
    def __init__(self, policy_fn, storage):
        self.policy_fn = policy_fn
        self.storage = storage
        self.add_to_storage = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=False),
                                func, *args, **kwargs)
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        is_compiling = _is_compiling(func, args, kwargs)

        if is_compiling:
            # Overwrite each node's "recompute" tag to add in the user annotation.
            fx_traceback.current_meta["recompute"] = policy

        out = func(*args, **kwargs)

        any_ret_has_alias_info = any(ret.alias_info is not None for ret in func._schema.returns)

        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE):
            storage = self.storage.save_storage[func]  # patch code
            storage.append(tree_map(lambda x: _VersionWrapper(_maybe_detach(x, any_ret_has_alias_info)), out))
        elif policy == CheckpointPolicy.MUST_SWAP:  # patch code
            if not self.add_to_storage:
                group_name = SwapManager().get_current_group_name()
                SwapManager().add_storage(group_name, self.storage)
                self.add_to_storage = True
            storage = self.storage.swap_storage[func]
            storage.append(tree_map(lambda x: SwapTensor(_maybe_detach(x, any_ret_has_alias_info)), out))
        return out


class _CachedTorchDispatchMode(TorchDispatchMode):
    # Used together with _CachedTorchDispatchMode to implement SAC.
    def __init__(self, policy_fn, storage, allow_cache_entry_mutation):
        self.policy_fn = policy_fn
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=True),
                                func, *args, **kwargs)
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        is_compiling = _is_compiling(func, args, kwargs)

        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE) or is_compiling:
            storage = self.storage.save_storage.get(func)  # patch code
            if storage is None:
                raise RuntimeError(f"{func} encountered during backward, but not found in storage")
            if len(storage) == 0:
                raise RuntimeError(
                    "Trying to backward an extra time. You are only allowed to backward once "
                    "on any region computed under selective activation checkpoint."
                )
            out = tree_map(lambda x: x.get_val(self.allow_cache_entry_mutation), storage.pop(0))
        elif policy == CheckpointPolicy.MUST_SWAP:  # patch code
            storage = self.storage.swap_storage.get(func)
            if storage is None:
                raise RuntimeError(f"{func} encountered during backward, but not found in storage")
            if len(storage) == 0:
                raise RuntimeError(
                    "Trying to backward an extra time. You are only allowed to backward once "
                    "on any region computed under selective activation checkpoint."
                )
            out = tree_map(lambda x: x.get_val(), storage.pop(0))
        else:
            out = func(*args, **kwargs)
        return out


def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
    """
    Helper to avoid recomputing certain ops during activation checkpointing.

    Use this with `torch.utils.checkpoint.checkpoint` to control which
    operations are recomputed during the backward pass.

    Args:
        policy_fn_or_list (Callable or List):
          - If a policy function is provided, it should accept a
            :class:`SelectiveCheckpointContext`, the :class:`OpOverload`, args and
            kwargs to the op, and return a :class:`CheckpointPolicy` enum value
            indicating whether the execution of the op should be recomputed or not.
          - If a list of operations is provided, it is equivalent to a policy
            returning `CheckpointPolicy.MUST_SAVE` for the specified
            operations and `CheckpointPolicy.PREFER_RECOMPUTE` for all other
            operations.
        allow_cache_entry_mutation (bool, optional): By default, an error is
            raised if any tensors cached by selective activation checkpoint are
            mutated in order to ensure correctness. If set to `True`, this check
            is disabled.
    Returns:
        A tuple of two context managers.

    Example:
        >>> # xdoctest: +REQUIRES(LINUX)
        >>> import functools
        >>>
        >>> x = torch.rand(10, 10, requires_grad=True)
        >>> y = torch.rand(10, 10, requires_grad=True)
        >>>
        >>> ops_to_save = [
        >>>    torch.ops.aten.mm.default,
        >>> ]
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    if op in ops_to_save:
        >>>        return CheckpointPolicy.MUST_SAVE
        >>>    else:
        >>>        return CheckpointPolicy.PREFER_RECOMPUTE
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> # or equivalently
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, ops_to_save)
        >>>
        >>> def fn(x, y):
        >>>     return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    # NB: If grad_mode is disabled, checkpoint would not run forward under
    #     context_fn anyway, so proceed as usual.
    if isinstance(policy_fn_or_list, list):
        for op in policy_fn_or_list:
            if not isinstance(op, torch._ops.OpOverload):
                _extra_msg = (
                    "Please update the OpOverloadPacket to a specific OpOverload."
                    "For example, if you have `torch.ops.aten.mm`, change it to `torch.ops.aten.mm.default`."
                ) if isinstance(op, torch._ops.OpOverloadPacket) else ""
                raise ValueError(
                    f"Expected op in `op_list` to be an OpOverload but got: {op} "
                    f"of type {type(op)}. {_extra_msg}"
                )

        def policy_fn(ctx, op, *args, **kwargs):
            if op in policy_fn_or_list:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE
    elif callable(policy_fn_or_list):
        policy_fn = policy_fn_or_list
    else:
        raise TypeError("policy_fn_or_list must be either a function or a list of ops.")

    storage = Storage()  # patch code
    return (
        _CachingTorchDispatchMode(policy_fn, storage),
        _CachedTorchDispatchMode(policy_fn, storage, allow_cache_entry_mutation),
    )
