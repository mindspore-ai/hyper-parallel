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
"""Recognize framework-owned storage without changing the framework runtime."""

import sys
from functools import lru_cache
from typing import Any

import torch
from torch.utils._pytree import tree_iter, tree_map

from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor


def has_managed_inputs(args: tuple, kwargs: dict) -> bool:
    """Return whether framework code is consuming a tracked activation."""
    # This is a predicate: rebuilding every argument container dominates the
    # framework's otherwise metadata-only gradient view operations.
    return any(isinstance(value, ShadowTensor) for value in tree_iter((args, kwargs)))


@lru_cache(maxsize=256)
def _is_fsdp_module(module_name: str) -> bool:
    """Cache only immutable classification, never the live calling context."""
    return module_name.startswith((
        "hyper_parallel.core.fully_shard.", "torch.distributed.fsdp._fully_shard.",
    ))


def inside_fsdp_runtime() -> bool:
    """Identify parameter and collective hooks, excluding ordinary model execution."""
    frame = sys._getframe(1)  # pylint: disable=protected-access
    while frame is not None:
        module_name = frame.f_globals.get("__name__", "")
        if _is_fsdp_module(module_name):
            return True
        frame = frame.f_back
    return False


def dispatch_framework_op(func: Any, args: tuple, kwargs: dict) -> Any:
    """Export activation views to framework-owned storage at the dispatch boundary.

    FSDP may resize collective inputs after communication. Alias outputs must
    therefore own independent storage; writes still update the original inputs.
    Cloning entire byte storages preserves offsets and aliases among outputs.
    """
    activation_storages = set()

    def unwrap(value: Any) -> Any:
        """Resolve managed inputs while recording their source storage."""
        if not isinstance(value, ShadowTensor):
            return value
        resolved = value.resolve()
        activation_storages.add(resolved.untyped_storage())
        return resolved

    # Only local materialization bypasses modes. The framework op itself still
    # passes through any unrelated dispatch mode installed below this one.
    with torch._C._DisableTorchDispatch():  # pylint: disable=protected-access
        real_args, real_kwargs = tree_map(unwrap, (args, kwargs))
    result = func(*real_args, **real_kwargs)
    owned = {}

    def export(value: Any) -> Any:
        """Give aliased outputs storage owned by the framework caller."""
        if not isinstance(value, torch.Tensor):
            return value
        storage = value.untyped_storage()
        if storage not in activation_storages:
            return value
        if storage not in owned:
            raw = torch.empty(0, dtype=torch.uint8, device=value.device)
            raw.set_(storage, 0, (storage.nbytes(),), (1,))
            owned[storage] = raw.clone()
        output = torch.empty(0, dtype=value.dtype, device=value.device)
        output.set_(owned[storage].untyped_storage(), value.storage_offset(), value.size(), value.stride())
        return output

    with torch._C._DisableTorchDispatch():  # pylint: disable=protected-access
        return tree_map(export, result)
