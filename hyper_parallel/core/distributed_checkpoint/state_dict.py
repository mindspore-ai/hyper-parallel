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
"""Optimizer state dict core API — platform-agnostic forwarding layer.

This module provides ``get_optim_state_dict`` and ``set_optim_state_dict``
as core-layer entry points that delegate to the active platform backend
via ``get_platform()``.  No torch / mindspore imports are allowed here.
"""
from __future__ import annotations

from typing import Any

from hyper_parallel.platform import get_platform

platform = get_platform()


def get_optim_state_dict(model: Any, optimizer: Any, *, options: Any = None) -> Any:
    """Get optimizer state dict with platform-specific implementation.

    Delegates to the platform-specific implementation at runtime.

    Args:
        model: The model whose parameters are optimized.
        optimizer: The optimizer instance.
        options: Optional configuration (full_state_dict, cpu_offload,
            flatten_optimizer_state_dict, etc.).

    Returns:
        dict: Optimizer state dict with FQN-based keys.
    """
    return platform.get_optim_state_dict(model, optimizer, options=options)


def set_optim_state_dict(
    model: Any,
    optimizer: Any,
    optim_state_dict: Any,
    *,
    options: Any = None,
) -> None:
    """Set optimizer state dict with platform-specific implementation.

    Delegates to the platform-specific implementation at runtime.

    Args:
        model: The model whose parameters are optimized.
        optimizer: The target optimizer instance.
        optim_state_dict: The optimizer state dict to load.
        options: Optional configuration (full_state_dict, cpu_offload,
            strict, broadcast_from_rank0, etc.).
    """
    platform.set_optim_state_dict(model, optimizer, optim_state_dict, options=options)
