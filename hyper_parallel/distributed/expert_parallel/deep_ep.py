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

"""Lazy DeepEP-Ascend primitives and explicit communication-buffer ownership.

These are low-level transport operations. Autograd callers must invoke them
inside their own differentiable dispatch/combine Function, rather than losing
the training graph through raw forward-only native bindings.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch



def _require_explicit_vjp(*values: Any) -> None:
    if not torch.is_grad_enabled():
        return
    for value in values:
        leaves = value if isinstance(value, tuple) else (value,)
        if any(isinstance(t, torch.Tensor) and t.requires_grad for t in leaves):
            raise RuntimeError("raw DeepEP transport requires an explicit autograd Function; use its no-grad forward")


def create_ep_buffer(*args: Any, **kwargs: Any) -> Any:
    """Construct the optional native EPBuffer without importing it at module load."""
    backend = import_module("deep_ep")
    return backend.EPBuffer(*args, **kwargs)


def deep_ep_dispatch(buffer: Any, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
                     topk_indices: torch.Tensor | None = None, topk_weights: torch.Tensor | None = None,
                     **options: Any) -> Any:
    """Dispatch expanded expert slots; preserve the returned native handle and event."""
    _require_explicit_vjp(x, topk_weights)
    if options.get("do_handle_copy", False):
        raise ValueError("DeepEP-Ascend requires do_handle_copy=False")
    options.update(do_expand=True, do_handle_copy=False)
    return buffer.dispatch(x, topk_indices, topk_weights, **options)


def deep_ep_combine(buffer: Any, x: torch.Tensor, handle: Any, **options: Any) -> Any:
    """Combine slots using their dispatch handle, retaining optional weights and event."""
    _require_explicit_vjp(x, options.get("topk_weights"), options.get("bias"))
    return buffer.combine(x, handle, **options)


def deep_ep_wait(event: Any) -> Any:
    """Order the current compute stream after communication and its deferred epilogue."""
    return event.current_stream_wait()


def deep_ep_destroy(buffer: Any) -> None:
    """Release a native buffer; managed DeepEPBuffer instances also drain pending events."""
    buffer.destroy()


class DeepEPBuffer:
    """Own an EPBuffer and retain asynchronous events until they have been consumed.

    Handles remain owned by their call sites; this object never copies or
    discards a live dispatch handle. Closing orders outstanding work before
    native destruction, and prevents further dispatch/combine on this owner.
    """

    def __init__(self, *args: Any, **options: Any) -> None:
        """Create a managed native buffer with explicit destruction enabled."""
        options.setdefault("explicitly_destroy", True)
        self.native = create_ep_buffer(*args, **options)
        self._events: list[Any] = []
        self._closed = False

    def _retain(self, result: Any) -> Any:
        event = result[-1] if isinstance(result, tuple) else result
        if callable(getattr(event, "current_stream_wait", None)):
            self._events.append(event)
        return result

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("DeepEP buffer is closed")

    def dispatch(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
                 topk_indices: torch.Tensor | None = None, topk_weights: torch.Tensor | None = None,
                 **options: Any) -> Any:
        """Dispatch with the Ascend layout contract and retain the completion event."""
        self._require_open()
        return self._retain(deep_ep_dispatch(self.native, x, topk_indices, topk_weights, **options))

    def combine(self, x: torch.Tensor, handle: Any, **options: Any) -> Any:
        """Combine without changing the native dispatch handle or return structure."""
        self._require_open()
        return self._retain(deep_ep_combine(self.native, x, handle, **options))

    def wait(self) -> None:
        """Wait each retained event; keep the failing event alive if a wait raises."""
        while self._events:
            event = self._events[0]
            # Native synchronous or explicitly released EventOverlap objects have no live event.
            if getattr(event, "event", event) is not None:
                deep_ep_wait(event)
            self._events.pop(0)

    def destroy(self) -> None:
        """Drain asynchronous work before releasing the native buffer, exactly once."""
        if self._closed:
            return
        self.wait()
        deep_ep_destroy(self.native)
        self._closed = True

    close = destroy

    def __enter__(self) -> DeepEPBuffer:
        """Retain the buffer through a transport scope."""
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Wait and destroy on both normal and exceptional exit."""
        self.destroy()
