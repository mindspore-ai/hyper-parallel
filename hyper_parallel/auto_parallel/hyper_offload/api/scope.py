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

from torch.utils._pytree import tree_iter

from hyper_parallel.auto_parallel.hyper_offload.execution.framework import (
    dispatch_framework_op,
    dispatch_framework_read,
)
from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor

__all__ = ["dispatch_framework_op", "dispatch_framework_read", "has_managed_inputs", "inside_fsdp_runtime"]


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
