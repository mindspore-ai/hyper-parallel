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
"""loss_parallel context manager.

Context manager and utilities for distributed cross-entropy computation:
    - loss_parallel(): Context manager to enable distributed CE semantics
    - is_loss_parallel_active(): Debug/assert function
    - Internal state management via ContextVar
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from typing import Optional, ContextManager, TYPE_CHECKING

if TYPE_CHECKING:
    from hyper_parallel.core.dtensor.device_mesh import DeviceMesh

__all__ = [
    "loss_parallel",
    "is_loss_parallel_active",
]


# ========================================================================
# ContextVar state management
# ========================================================================

_loss_parallel_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_loss_parallel_active", default=False
)

_loss_parallel_count: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_loss_parallel_count", default=0
)

_loss_parallel_context_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "_loss_parallel_context_id", default=None
)

_global_context_id_counter: int = 0
_global_context_id_lock: threading.Lock = threading.Lock()

_loss_parallel_mesh: contextvars.ContextVar[Optional["DeviceMesh"]] = contextvars.ContextVar(
    "_loss_parallel_mesh", default=None
)

_loss_parallel_strict: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_loss_parallel_strict", default=True
)


# ========================================================================
# loss_parallel context manager
# ========================================================================

@contextlib.contextmanager
def loss_parallel(
    *,
    mesh: Optional["DeviceMesh"] = None,
    strict: bool = True,
) -> ContextManager[None]:
    """Context manager to enable distributed CE semantics.

    Args:
        mesh: Explicitly specify TP sub-mesh; None infers from participating DTensors;
            inference failure with strict=True raises error.
        strict: When True, layout contract violation raises ValueError or dedicated exception;
            False only warns or takes optional fallback path (if fallback_gather implemented).

    Nesting semantics:
        Reentrant counting; count must return to zero on exit. Supports nested usage.

    Example:
        >>> with loss_parallel():
        ...     logits = model(input)  # logits is DTensor, Shard(-1)
        ...     loss = F.cross_entropy(logits, target)
        ...     loss.backward()

    Note:
        PyTorch upstream loss_parallel() is often parameterless.
        HyperParallel adds parameters; see docs/compatibility/pytorch_diff.md.
    """
    global _global_context_id_counter

    old_count = _loss_parallel_count.get()
    old_active = _loss_parallel_active.get()
    old_mesh = _loss_parallel_mesh.get()
    old_strict = _loss_parallel_strict.get()
    old_context_id = _loss_parallel_context_id.get()

    _loss_parallel_count.set(old_count + 1)
    _loss_parallel_active.set(True)
    _loss_parallel_mesh.set(mesh)
    _loss_parallel_strict.set(strict)

    if old_count == 0:
        with _global_context_id_lock:
            _global_context_id_counter += 1
            current_context_id = _global_context_id_counter
        _loss_parallel_context_id.set(current_context_id)
    else:
        current_context_id = old_context_id

    try:
        yield
    finally:
        _loss_parallel_count.set(old_count)
        _loss_parallel_active.set(old_active)
        _loss_parallel_mesh.set(old_mesh)
        _loss_parallel_strict.set(old_strict)
        _loss_parallel_context_id.set(old_context_id)


# ========================================================================
# is_loss_parallel_active (optional debug API)
# ========================================================================

def is_loss_parallel_active() -> bool:
    """Debug or assert: whether current thread is in loss_parallel context.

    Returns:
        True: currently in loss_parallel context
        False: not in context
    """
    return _loss_parallel_active.get()


def get_loss_parallel_count() -> int:
    """Get current nesting count.

    Returns:
        int: Nesting depth (0 means not in context).
    """
    return _loss_parallel_count.get()


# ========================================================================
# Internal functions (for use by other modules)
# ========================================================================

def _get_loss_parallel_token() -> Optional[int]:
    """Get current context token for cache key differentiation.

    Returns:
        Optional[int]: Current context unique ID for differentiating cache keys.
            Returns None if not in context, returns positive integer for context ID.
    """
    return _loss_parallel_context_id.get()


def _get_loss_parallel_mesh() -> Optional["DeviceMesh"]:
    """Get the mesh set in current context.

    Returns:
        Optional[DeviceMesh]: Explicitly specified mesh, or None (infer from DTensor).
    """
    return _loss_parallel_mesh.get()


def _get_loss_parallel_strict() -> bool:
    """Get the strict setting in current context.

    Returns:
        bool: Whether in strict mode.
    """
    return _loss_parallel_strict.get()
