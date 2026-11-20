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
"""Export framework reads without adopting framework-owned tensor storage."""

from typing import Any

import torch
from torch.utils._pytree import tree_iter, tree_map

from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor


def dispatch_framework_read(executor: Any, func: Any, args: tuple, kwargs: dict, *, native_inputs: bool = False) -> Any:
    """Record a framework read while keeping its outputs outside activation ownership.

    Args:
        executor: Active warmup or replay executor.
        func: Framework-owned tensor operation.
        args: Positional operation inputs.
        kwargs: Keyword operation inputs.
        native_inputs: Whether all tensor inputs are outside managed ownership.
    """
    output = None

    def framework_read(*inputs: Any, **options: Any) -> tuple:
        """Expose input lifetimes without adopting framework-owned outputs.

        Args:
            inputs: Positional operation inputs.
            options: Keyword operation inputs.
        """
        nonlocal output
        output = func(*inputs, **options) if native_inputs else dispatch_framework_op(func, inputs, options)
        recorder = getattr(executor, "record_framework_allocation", None)
        if recorder is not None:
            input_storages = {
                value.untyped_storage()._cdata for value in tree_iter((inputs, options))
                if isinstance(value, torch.Tensor)
            }
            allocated = {
                value.untyped_storage()._cdata: value.untyped_storage().nbytes()
                for value in tree_iter(output) if isinstance(value, torch.Tensor)
                and value.untyped_storage()._cdata not in input_storages
            }
            recorder(sum(allocated.values()))
        return ()

    framework_read.__name__ = "framework_read:" + func.__name__
    if hasattr(func, "_schema"):
        framework_read._schema = func._schema
    executor.dispatch(framework_read, args, kwargs)
    return output


def dispatch_framework_op(func: Any, args: tuple, kwargs: dict) -> Any:
    """Export activation views to framework-owned storage at the dispatch boundary.

    FSDP may resize collective inputs after communication. Alias outputs must
    therefore own independent storage; writes still update the original inputs.
    Cloning entire byte storages preserves offsets and aliases among outputs.

    Args:
        func: Framework-owned tensor operation.
        args: Positional operation inputs.
        kwargs: Keyword operation inputs.
    """
    activation_storages = set()

    def unwrap(value: Any) -> Any:
        """Resolve managed inputs while recording their source storage.

        Args:
            value: Input tree leaf.
        """
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
        """Give aliased outputs storage owned by the framework caller.

        Args:
            value: Output tree leaf.
        """
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
