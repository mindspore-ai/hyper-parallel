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
"""Validation-only tracing for DeepSeek-V4.1 shared attention state."""

from __future__ import annotations

import contextvars
import functools
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedAttentionState,
)


_MAX_SHARED_STATE_TRACE_EVENTS = 10000
_TRACE_METHODS = {
    "publish_compressed_kv": ("publish", "compressed_kv", None),
    "require_compressed_kv": ("consume", "compressed_kv", "consumer_layer"),
    "publish_index_key": ("publish", "index_key", None),
    "require_index_key": ("consume", "index_key", "consumer_layer"),
    "publish_topk_indices": ("publish", "topk_indices", None),
    "require_topk_indices": ("consume", "topk_indices", "consumer_layer"),
    "publish_candidate_blocks": ("publish", "candidate_blocks", None),
    "require_candidate_blocks": ("consume", "candidate_blocks", "consumer_layer"),
}


@dataclass
class _SharedStateTrace:
    """Bounded events and weak logical identities for one optimizer step."""

    events: list[dict[str, Any]] = field(default_factory=list)
    state_identities: dict[int, tuple[weakref.ReferenceType[Any], int]] = field(
        default_factory=dict
    )
    next_state_id: int = 0

    def logical_state_id(self, state: Any) -> int:
        """Return a stable ID without extending the state's tensor lifetime."""
        object_id = id(state)
        current = self.state_identities.get(object_id)
        if current is not None and current[0]() is state:
            return current[1]
        state_id = self.next_state_id
        self.next_state_id += 1
        self.state_identities[object_id] = (weakref.ref(state), state_id)
        return state_id


@dataclass(frozen=True)
class _SharedStateTraceToken:
    """Context token and original methods restored after one traced step."""

    context_token: contextvars.Token
    original_methods: dict[str, Callable[..., Any]]


_ACTIVE_TRACE: contextvars.ContextVar[_SharedStateTrace | None] = contextvars.ContextVar(
    "deepseek_v41_shared_state_trace",
    default=None,
)


def _argument(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        index: int,
        name: str,
) -> Any:
    """Read one original method argument without changing its call contract."""
    return args[index] if len(args) > index else kwargs.get(name)


def _record_event(
        trace: _SharedStateTrace,
        state: SharedCompressedAttentionState,
        action: str,
        key: str,
        source_layer: int | None,
        consumer_layer: int | None,
) -> None:
    """Record one bounded semantic event without retaining tensor state."""
    if len(trace.events) >= _MAX_SHARED_STATE_TRACE_EVENTS:
        raise RuntimeError(
            "[HP-STATE-003] shared-state validation trace exceeded "
            f"{_MAX_SHARED_STATE_TRACE_EVENTS} events"
        )
    trace.events.append(
        {
            "state_id": trace.logical_state_id(state),
            "action": action,
            "key": key,
            "source_layer": source_layer,
            "consumer_layer": consumer_layer,
        }
    )


def _traced_method(
        original: Callable[..., Any],
        action: str,
        key: str,
        consumer_name: str | None,
) -> Callable[..., Any]:
    """Wrap one state method only for the active validation step."""
    @functools.wraps(original)
    def wrapper(state: SharedCompressedAttentionState, *args: Any, **kwargs: Any) -> Any:
        trace = _ACTIVE_TRACE.get()
        if trace is not None:
            source_layer = _argument(args, kwargs, 0, "source_layer")
            consumer_layer = (
                None
                if consumer_name is None
                else _argument(args, kwargs, 1, consumer_name)
            )
            _record_event(
                trace,
                state,
                action,
                key,
                source_layer,
                consumer_layer,
            )
        return original(state, *args, **kwargs)

    return wrapper


def begin_shared_state_trace() -> _SharedStateTraceToken:
    """Install validation-only state-method wrappers for one optimizer step."""
    if _ACTIVE_TRACE.get() is not None:
        raise RuntimeError("DeepSeek-V4.1 shared-state tracing is already active")
    trace = _SharedStateTrace()
    context_token = _ACTIVE_TRACE.set(trace)
    original_methods = {}
    try:
        for method_name, (action, key, consumer_name) in _TRACE_METHODS.items():
            original = getattr(SharedCompressedAttentionState, method_name)
            original_methods[method_name] = original
            setattr(
                SharedCompressedAttentionState,
                method_name,
                _traced_method(original, action, key, consumer_name),
            )
    except Exception:
        for method_name, original in original_methods.items():
            setattr(SharedCompressedAttentionState, method_name, original)
        _ACTIVE_TRACE.reset(context_token)
        raise
    return _SharedStateTraceToken(context_token, original_methods)


def finish_shared_state_trace(
        token: _SharedStateTraceToken,
) -> tuple[dict[str, Any], ...]:
    """Restore production methods and return validation events."""
    trace = _ACTIVE_TRACE.get()
    try:
        return () if trace is None else tuple(trace.events)
    finally:
        for method_name, original in token.original_methods.items():
            setattr(SharedCompressedAttentionState, method_name, original)
        _ACTIVE_TRACE.reset(token.context_token)


__all__ = ["begin_shared_state_trace", "finish_shared_state_trace"]
