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
from typing import Any, Optional, Union

import mindspore as ms
from mindspore import MsDispatchMode
from hyper_parallel.core.activation_checkpoint.swap import SwapManager, SwapTensor, Storage
from hyper_parallel.core.activation_checkpoint import CheckpointPolicy
from hyper_parallel.platform import get_platform

platform = get_platform()

class _VersionWrapper:
    # Check that cached tensors are not mutated.
    def __init__(self, val):
        self.val: Union[ms.Tensor, Any] = val
        self.version: Optional[int] = val._version if isinstance(val, ms.Tensor) else None

    def get_val(self, allow_cache_entry_mutation):
        if self.version is not None and not allow_cache_entry_mutation:
            if self.val._version != self.version:
                # Can we give user a stack trace of where the mutation happened?
                raise RuntimeError(
                    "Tensor cached during selective activation checkpoint has been mutated"
                )
        return self.val


def _maybe_detach(x):
    if isinstance(x, ms.Tensor) and (x.is_floating_point() or x.is_complex()):
        x = ms.ops.stop_gradient(x)
    return x


class SelectiveCheckpointContext:
    def __init__(self, *, is_recompute):
        self.is_recompute = is_recompute

SAC_IGNORED_OPS = {"StopGradient", "Reshape", "SelectExtView", "TransposeExtView", "Transpose", "LayerNormExt"}


class _CachingMindSporeDispatchMode(MsDispatchMode):
    def __init__(self, policy_fn, storage):
        self.policy_fn = policy_fn
        self.storage = storage
        self.add_to_storage = False

    def __ms_dispatch__(self, func, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func.name in SAC_IGNORED_OPS:
            return func(*args, **kwargs)
        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=False),
                                func, *args, **kwargs)

        out = func(*args, **kwargs)

        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE):
            storage = self.storage.save_storage[func.name]
            storage.append(platform.tree_map(lambda x: _VersionWrapper(_maybe_detach(x)), out))
        elif policy == CheckpointPolicy.MUST_SWAP:
            if not self.add_to_storage:
                group_name = SwapManager().get_current_group_name()
                SwapManager().add_storage(group_name, self.storage)
                self.add_to_storage = True
            storage = self.storage.swap_storage[func.name]
            storage.append(platform.tree_map(lambda x: SwapTensor(_maybe_detach(x)), out))
        return out


class _CachedMindSporeDispatchMode(MsDispatchMode):
    def __init__(self, policy_fn, storage, allow_cache_entry_mutation):
        self.policy_fn = policy_fn
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation

    def __ms_dispatch__(self, func, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func.name in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=True),
                                func, *args, **kwargs)

        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE):
            storage = self.storage.save_storage.get(func.name)  # patch code
            if storage is None:
                raise RuntimeError(f"{func} encountered during backward, but not found in storage")
            if len(storage) == 0:
                raise RuntimeError(
                    "Trying to backward an extra time. You are only allowed to backward once "
                    "on any region computed under selective activation checkpoint."
                )
            out = platform.tree_map(lambda x: x.get_val(self.allow_cache_entry_mutation), storage.pop(0))
        elif policy == CheckpointPolicy.MUST_SWAP:  # patch code
            storage = self.storage.swap_storage.get(func.name)
            if storage is None:
                raise RuntimeError(f"{func} encountered during backward, but not found in storage")
            if len(storage) == 0:
                raise RuntimeError(
                    "Trying to backward an extra time. You are only allowed to backward once "
                    "on any region computed under selective activation checkpoint."
                )
            out = platform.tree_map(lambda x: x.get_val(), storage.pop(0))
        else:
            out = func(*args, **kwargs)
        return out


def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
    if callable(policy_fn_or_list):
        policy_fn = policy_fn_or_list
    else:
        raise TypeError("policy_fn_or_list must be either a function or a list of ops.")

    storage = Storage()
    return (
        _CachingMindSporeDispatchMode(policy_fn, storage),
        _CachedMindSporeDispatchMode(policy_fn, storage, allow_cache_entry_mutation)
    )
