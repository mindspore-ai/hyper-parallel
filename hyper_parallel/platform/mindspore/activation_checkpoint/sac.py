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
# ============================================================================
"""enhanced with selective checkpoint support swap"""
# pylint: disable=W0212, W0613, C0115, C0116, C0103, R1705
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import mindspore as ms
from mindspore import MsDispatchMode
from hyper_parallel.core.activation_checkpoint.swap import (
    SwapManager,
    Storage,
    SwapTensor,
)
from hyper_parallel.core.activation_checkpoint.activation_checkpoint import CheckpointPolicy
from hyper_parallel.platform import get_platform

platform = get_platform()


class _VersionWrapper:
    # Check that cached tensors are not mutated.
    def __init__(self, val):
        self.val: Union[ms.Tensor, Any] = val
        self.version: Optional[int] = (
            val._version if isinstance(val, ms.Tensor) else None
        )

    def get_val(self, allow_cache_entry_mutation):
        if self.version is not None and not allow_cache_entry_mutation:
            if self.val._version != self.version:
                # Can we give user a stack trace of where the mutation happened?
                raise RuntimeError(
                    "Tensor cached during selective activation checkpoint has been mutated"
                )
        return self.val


class _SwapCacheEntry:
    """Pair the recompute cache and swap record around the same tensor object."""
    def __init__(self, val, funcname, group_swap=False):
        self.save = _VersionWrapper(val)
        self.swap = SwapTensor(val, funcname, group_swap=group_swap)


def _maybe_detach(x):
    if isinstance(x, ms.Tensor) and (x.is_floating_point() or x.is_complex()):
        x = x.detach()
    return x


class SelectiveCheckpointContext:
    def __init__(self, *, is_recompute):
        self.is_recompute = is_recompute

SAC_IGNORED_OPS = {"StopGradient"}


class _CachingMindSporeDispatchMode(MsDispatchMode):
    def __init__(self, policy_fn, swap_storage, storage, group_swap=False):
        self.policy_fn = policy_fn
        self.swap_storage = swap_storage
        self.storage = storage
        self.add_to_storage = False
        self.group_swap = group_swap
        # Cache context and singleton to avoid per-dispatch allocation / lookup.
        self._swap_manager = SwapManager()
        self._group_prefix = ""

    def __ms_dispatch__(self, func, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func.name in SAC_IGNORED_OPS:
            return func(*args, **kwargs)
        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=False),
                                func, *args, **kwargs)

        out = func(*args, **kwargs)

        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE):
            self.storage[func.name].append(
                platform.tree_map(
                    lambda x: _VersionWrapper(_maybe_detach(x)), out
                )
            )
        elif policy == CheckpointPolicy.MUST_SWAP:
            if not self.add_to_storage:
                group_name = self._swap_manager.get_current_group_name()
                self._group_prefix = f"{group_name}::"
                self._swap_manager.add_storage(group_name, self.swap_storage)
                self.add_to_storage = True
            funcname = f"{self._group_prefix}{func.name}"
            group_swap = self.group_swap
            entries = platform.tree_map(
                lambda x: _SwapCacheEntry(_maybe_detach(x), funcname, group_swap=group_swap), out
            )
            self.storage[func.name].append(
                platform.tree_map(lambda x: x.save, entries)
            )
            self.swap_storage[func.name].append(
                platform.tree_map(lambda x: x.swap, entries)
            )
        elif policy != CheckpointPolicy.MUST_RECOMPUTE:
            raise RuntimeError(f"Checkpoint Activation: {func.name} encountered an invalid policy {policy}")
        return out


class _CachedMindSporeDispatchMode(MsDispatchMode):
    def __init__(self, policy_fn, swap_storage, storage, allow_cache_entry_mutation):
        self.policy_fn = policy_fn
        self.swap_storage = swap_storage
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation
        self._swap_cleared = False

    def __ms_dispatch__(self, func, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func.name in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=True),
                                func, *args, **kwargs)

        if not self._swap_cleared:
            self.swap_storage.clear()
            self._swap_cleared = True

        # MUST_SAVE and MUST_SWAP both restore from storage identically.
        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE, CheckpointPolicy.MUST_SWAP):
            storage = self.storage.get(func.name)
            if storage is None:
                raise RuntimeError(f"{func} encountered during backward, but not found in storage")
            if len(storage) == 0:
                raise RuntimeError(
                    "Trying to backward an extra time. You are only allowed to backward once "
                    "on any region computed under selective activation checkpoint."
                )
            out = platform.tree_map(lambda x: x.get_val(self.allow_cache_entry_mutation), storage.pop(0))
        else:
            out = func(*args, **kwargs)
        return out


def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False, group_swap=False):
    if policy_fn_or_list is None:
        def policy_fn(_ctx, _op, *_args, **_kwargs):
            return CheckpointPolicy.PREFER_RECOMPUTE
    elif callable(policy_fn_or_list):
        policy_fn = policy_fn_or_list
    else:
        raise TypeError("policy_fn_or_list must be either a function or a list of ops.")

    swap_storage = Storage()
    storage: Dict[Any, List[Any]] = defaultdict(list)
    return (
        _CachingMindSporeDispatchMode(policy_fn, swap_storage, storage, group_swap=group_swap),
        _CachedMindSporeDispatchMode(policy_fn, swap_storage, storage, allow_cache_entry_mutation)
    )
